"""Move Claude Code's token reminder into the user message for Gemini requests.

Prime's Gemini gateway returns "Requests ending with a model turn are not
supported" when a user message that holds only an image tool result is followed
by a system-role message. Claude Code sends its <total_tokens> reminder as that
system message after every tool result. This adapter moves the reminder text
into the user message that holds the image result. Nothing else in a request
changes, and every response passes through unchanged.

Usage:
    <env-venv>/python gemini_gateway_adapter.py --port 18081
    uv run eval ... --client.base-url http://127.0.0.1:18081/api/v1
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Final

import httpx
from aiohttp import web

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Message = dict[str, Json]

UPSTREAM: Final = "https://api.pinference.ai"
BLOCKED_REQUEST_HEADERS: Final = frozenset(
    {"host", "content-length", "connection", "accept-encoding", "authorization", "x-api-key", "transfer-encoding"}
)
BLOCKED_RESPONSE_HEADERS: Final = frozenset({"content-length", "content-encoding", "transfer-encoding", "connection"})
MAX_REQUEST_BYTES: Final = 64 * 1024 * 1024


def prime_credentials() -> tuple[str, str | None]:
    """Credentials from the environment, as in a sandbox, else from the Prime CLI config file."""
    api_key = os.environ.get("PRIME_API_KEY")
    if api_key is not None:
        return api_key, os.environ.get("PRIME_TEAM_ID")
    config = json.loads((Path.home() / ".prime" / "config.json").read_text())
    return config["api_key"], os.environ.get("PRIME_TEAM_ID") or config.get("team_id")


def holds_image_result(message: Message) -> bool:
    content = message.get("content")
    if message.get("role") != "user" or not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "tool_result"
        and isinstance(block.get("content"), list)
        and any(isinstance(part, dict) and part.get("type") == "image" for part in block["content"])
        for block in content
    )


def as_blocks(content: Json) -> list[Json]:
    return content if isinstance(content, list) else [{"type": "text", "text": content}]


def merge_reminders(messages: list[Message]) -> list[Message]:
    """Append each system message that follows an image tool result to that user message."""
    merged: list[Message] = []
    for message in messages:
        previous = merged[-1] if merged else None
        if message.get("role") == "system" and previous is not None and holds_image_result(previous):
            merged[-1] = {**previous, "content": [*as_blocks(previous["content"]), *as_blocks(message["content"])]}
        else:
            merged.append(message)
    return merged


def upstream_headers(request: web.Request, api_key: str, team_id: str | None) -> dict[str, str]:
    headers = {name: value for name, value in request.headers.items() if name.lower() not in BLOCKED_REQUEST_HEADERS}
    headers["accept-encoding"] = "identity"
    headers["authorization"] = f"Bearer {api_key}"
    headers["x-api-key"] = api_key
    if team_id is not None:
        headers["x-prime-team-id"] = team_id
    return headers


async def forward(request: web.Request) -> web.StreamResponse:
    try:
        body = json.loads(await request.read())
    except json.JSONDecodeError:
        return web.json_response({"error": "Request body is not JSON"}, status=400)
    if isinstance(body, dict) and isinstance(body.get("messages"), list):
        body["messages"] = merge_reminders(body["messages"])
    api_key, team_id = prime_credentials()
    client: httpx.AsyncClient = request.app["client"]
    try:
        upstream = await client.send(
            client.build_request(
                "POST",
                UPSTREAM + request.path,
                headers=upstream_headers(request, api_key, team_id),
                content=json.dumps(body),
            ),
            stream=True,
        )
    except httpx.HTTPError as error:
        return web.json_response({"error": f"Prime Inference request failed: {error!r}"}, status=502)
    try:
        response = web.StreamResponse(
            status=upstream.status_code,
            headers={name: value for name, value in upstream.headers.items() if name.lower() not in BLOCKED_RESPONSE_HEADERS},
        )
        await response.prepare(request)
        async for chunk in upstream.aiter_bytes():
            await response.write(chunk)
        await response.write_eof()
        if upstream.status_code >= 400:
            print(f"upstream {upstream.status_code} for {request.path}", flush=True)
        return response
    finally:
        await upstream.aclose()


async def open_client(app: web.Application) -> None:
    app["client"] = httpx.AsyncClient(timeout=httpx.Timeout(connect=10.0, read=None, write=60.0, pool=None))


async def close_client(app: web.Application) -> None:
    await app["client"].aclose()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    app = web.Application(client_max_size=MAX_REQUEST_BYTES)
    app.on_startup.append(open_client)
    app.on_cleanup.append(close_client)
    app.router.add_post("/{tail:.*}", forward)
    web.run_app(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
