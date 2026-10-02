"""Forward Prime Inference requests while capping the upload rate.

An agent episode sends its whole conversation, with images, on every turn. One
DeepSeek film uploads about 270 MB. Many episodes at once fill the uplink queue
of a home network. Delay rises to seconds, packets drop, and the sandbox tunnel
breaks. This adapter sends request bodies upstream at a fixed rate. It changes
no byte of any request or response.

Usage:
    <env-venv>/python uplink_limiter_adapter.py --port 18082 --rate-file limiter-rate-mb-s.txt
    uv run eval ... --client.base-url http://127.0.0.1:18082/api/v1

The rate file holds one number, in megabytes per second. The adapter reads it
again when the file changes, so the rate can change while runs are active.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Final

import httpx
from aiohttp import web

UPSTREAM: Final = "https://api.pinference.ai"
CHUNK_BYTES: Final = 16 * 1024
RATE_CHECK_SECONDS: Final = 5
REPORT_SECONDS: Final = 60
MIN_RATE_MB_S: Final = 0.05
MAX_RATE_MB_S: Final = 50.0
MAX_REQUEST_BYTES: Final = 256 * 1024 * 1024
BLOCKED_REQUEST_HEADERS: Final = frozenset(
    {"host", "content-length", "connection", "accept-encoding", "authorization", "x-api-key", "transfer-encoding"}
)
BLOCKED_RESPONSE_HEADERS: Final = frozenset({"content-length", "content-encoding", "transfer-encoding", "connection"})


def prime_credentials() -> tuple[str, str | None]:
    config = json.loads((Path.home() / ".prime" / "config.json").read_text())
    api_key = os.environ.get("PRIME_API_KEY") or config["api_key"]
    team_id = os.environ.get("PRIME_TEAM_ID") or config.get("team_id")
    return api_key, team_id


def read_rate(rate_file: Path) -> float:
    """Bytes per second from the rate file. Raises ValueError for a value outside the allowed range."""
    megabytes = float(rate_file.read_text().strip())
    if not MIN_RATE_MB_S <= megabytes <= MAX_RATE_MB_S:
        raise ValueError(f"{megabytes} MB/s is outside {MIN_RATE_MB_S} to {MAX_RATE_MB_S}")
    return megabytes * 1_000_000


class UploadLimiter:
    """One rate for all uploads. Each chunk reserves the next free time slot, so
    concurrent requests take turns and share the rate evenly. An idle limiter
    keeps no credit, so a burst never exceeds one chunk."""

    def __init__(self, bytes_per_second: float) -> None:
        self.bytes_per_second = bytes_per_second
        self.next_free = time.monotonic()
        self.reserved_bytes = 0
        self.in_flight = 0

    def delay_for(self, byte_count: int) -> float:
        now = time.monotonic()
        start = max(now, self.next_free)
        self.next_free = start + byte_count / self.bytes_per_second
        self.reserved_bytes += byte_count
        return start - now


async def limited(body: bytes, limiter: UploadLimiter) -> AsyncIterator[bytes]:
    for offset in range(0, len(body), CHUNK_BYTES):
        chunk = body[offset : offset + CHUNK_BYTES]
        delay = limiter.delay_for(len(chunk))
        if delay > 0:
            await asyncio.sleep(delay)
        yield chunk


def upstream_headers(request: web.Request, api_key: str, team_id: str | None, body_bytes: int) -> dict[str, str]:
    headers = {name: value for name, value in request.headers.items() if name.lower() not in BLOCKED_REQUEST_HEADERS}
    headers["accept-encoding"] = "identity"
    headers["authorization"] = f"Bearer {api_key}"
    headers["x-api-key"] = api_key
    # An explicit length makes httpx send the streamed body without chunked encoding.
    headers["content-length"] = str(body_bytes)
    if team_id is not None:
        headers["x-prime-team-id"] = team_id
    return headers


async def forward(request: web.Request) -> web.StreamResponse:
    body = await request.read()
    limiter: UploadLimiter = request.app["limiter"]
    client: httpx.AsyncClient = request.app["client"]
    api_key, team_id = prime_credentials()
    limiter.in_flight += 1
    try:
        try:
            upstream = await client.send(
                client.build_request(
                    "POST",
                    UPSTREAM + request.path_qs,
                    headers=upstream_headers(request, api_key, team_id, len(body)),
                    content=limited(body, limiter),
                ),
                stream=True,
            )
        except httpx.HTTPError as error:
            print(f"upstream request failed after {len(body) / 1e6:.1f} MB: {error!r}", flush=True)
            return web.json_response({"error": f"Prime Inference request failed: {error!r}"}, status=502)
        try:
            response = web.StreamResponse(
                status=upstream.status_code,
                headers={n: v for n, v in upstream.headers.items() if n.lower() not in BLOCKED_RESPONSE_HEADERS},
            )
            await response.prepare(request)
            async for chunk in upstream.aiter_bytes():
                await response.write(chunk)
            await response.write_eof()
            if upstream.status_code >= 400:
                print(f"upstream {upstream.status_code} for {request.path} ({len(body) / 1e6:.1f} MB)", flush=True)
            return response
        finally:
            await upstream.aclose()
    finally:
        limiter.in_flight -= 1


async def watch_rate(app: web.Application, rate_file: Path) -> None:
    limiter: UploadLimiter = app["limiter"]
    while True:
        await asyncio.sleep(RATE_CHECK_SECONDS)
        try:
            rate = read_rate(rate_file)
        except (OSError, ValueError) as error:
            print(f"rate file not used, keeping {limiter.bytes_per_second / 1e6:.2f} MB/s: {error}", flush=True)
            continue
        if rate != limiter.bytes_per_second:
            print(f"rate changed: {limiter.bytes_per_second / 1e6:.2f} -> {rate / 1e6:.2f} MB/s", flush=True)
            limiter.bytes_per_second = rate


async def report(app: web.Application) -> None:
    limiter: UploadLimiter = app["limiter"]
    previous = limiter.reserved_bytes
    while True:
        await asyncio.sleep(REPORT_SECONDS)
        sent = limiter.reserved_bytes - previous
        previous = limiter.reserved_bytes
        backlog = max(limiter.next_free - time.monotonic(), 0.0)
        print(
            f"{time.strftime('%H:%M:%S')} sent {sent / 1e6:.0f} MB in {REPORT_SECONDS}s "
            f"({sent / REPORT_SECONDS / 1e6:.2f} MB/s of {limiter.bytes_per_second / 1e6:.2f}), "
            f"{limiter.in_flight} requests in flight, queue {backlog:.0f}s",
            flush=True,
        )


async def start(app: web.Application) -> None:
    app["client"] = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=None, pool=None))
    app["tasks"] = [asyncio.create_task(watch_rate(app, app["rate_file"])), asyncio.create_task(report(app))]


async def stop(app: web.Application) -> None:
    for task in app["tasks"]:
        task.cancel()
    await app["client"].aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--rate-file", type=Path, required=True)
    args = parser.parse_args()
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app["rate_file"] = args.rate_file
    app["limiter"] = UploadLimiter(read_rate(args.rate_file))
    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    app.router.add_post("/{tail:.*}", forward)
    print(f"limiting uploads to {app['limiter'].bytes_per_second / 1e6:.2f} MB/s", flush=True)
    web.run_app(app, host="127.0.0.1", port=args.port, print=None)


if __name__ == "__main__":
    main()
