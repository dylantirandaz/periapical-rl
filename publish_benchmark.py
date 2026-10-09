"""Build four public benchmarks from saved data, without model calls.

Requires Python 3.12, Pillow, and the production dental environment dependencies.
Use the original six-run public data backup, not the reduced five-example site:

    python publish_benchmark.py --periapical-source /path/to/site-backup \\
        --runs-root /path/to/saved-runs --production-root /path/to/dental-environments \\
        --diagnosis-correction /path/to/diagnosis-rescore-0.3.1.json \\
        --output /path/to/new-output

The output contains 12 full-run summaries and 20 selected traces. Scores use
the complete source runs. Each sample uses five fixed ranks from the best run.
Private provider data is removed; returned public reasoning is retained.
Sources are read-only. An existing output directory is refused. After checking
the output and the rendered site, replace only the site's generated data tree.
This command does not deploy the site.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import io
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, TypedDict
from urllib.parse import urlsplit

from PIL import Image

type Json = None | bool | int | float | str | list[Json] | dict[str, Json]
type Object = dict[str, Json]

SITE = Path(__file__).resolve().parent
REDACTED = "[private data removed]"
PRIVATE_KEY = re.compile(
    r"api[_-]?key|authorization|authentication|password|passwd|secret|"
    r"access[_-]?token|refresh[_-]?token|auth[_-]?token|bearer|cookie|"
    r"encrypted|signature|credential|private[_-]?key|signed[_-]?url|^token$|^key$", re.I
)
URL = re.compile(r"https?://[^\s<>\"']+", re.I)
# Tunnel hostnames are per-run infrastructure. Cloudflare timeout errors print them as bare values, not URLs.
TUNNEL_HOST = re.compile(r"\b[a-z0-9-]+\.tunnel\.pinfra\.io\b", re.I)
DATA_IMAGE = re.compile(r"data:image/[a-z0-9.+-]+;base64,[a-z0-9+/=\r\n]+", re.I)
# A tool result can print a harness log. Encrypted reasoning in it is text, not a message field.
ENCRYPTED_REASONING = re.compile(r"((?:redacted_thinking\W{1,8}data|encrypted_content)\W{1,8})[A-Za-z0-9+/=_-]{24,}")


class FilmRow(TypedDict):
    film: str
    task_key: str
    boxes: Json
    labels: Json
    predictions: Json
    reward: float | None
    parsed: bool
    reply: str | None
    answer_source: Literal["final_reply", "report.json"]
    score_note: str | None
    score_provenance: Object
    has_thinking: bool
    model_calls: int
    images_viewed: int
    stop_condition: str | None
    status: Literal["success", "error"]
    errors: list[str]
    warnings: list[str]
    trace_path: str
    image_path: str | None
    attempts: list[Json]


@dataclass(frozen=True)
class RunConfig:
    model: str
    harness: str
    harness_version: str
    expected: int
    max_turns: int
    dataset_revision: str
    route: str
    retries: Object



def value(raw: object) -> Json:
    """Validate the JSON boundary, including finite numbers and string keys."""
    if raw is None or isinstance(raw, (str, bool, int)):
        return raw
    if isinstance(raw, float) and math.isfinite(raw):
        return raw
    if isinstance(raw, list):
        return [value(item) for item in raw]
    if isinstance(raw, dict):
        result: Object = {}
        for key, item in raw.items():
            if not isinstance(key, str):
                raise ValueError("JSON object key is not text")
            result[key] = value(item)
        return result
    raise ValueError("Invalid or non-finite JSON value")


def obj(raw: Json, label: str) -> Object:
    if not isinstance(raw, dict):
        raise ValueError(f"{label} must be an object")
    return raw


def items(raw: Json, label: str) -> list[Json]:
    if not isinstance(raw, list):
        raise ValueError(f"{label} must be an array")
    return raw


def text(raw: Json, label: str) -> str:
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{label} must be non-empty text")
    return raw


def integer(raw: Json, label: str) -> int:
    if not isinstance(raw, int) or isinstance(raw, bool):
        raise ValueError(f"{label} must be an integer")
    return raw


def load(path: Path) -> Json:
    return value(json.loads(path.read_text()))


def dump(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, allow_nan=False, separators=(",", ":")))


class PublicTrace:
    """Keep public message content; remove private fields at every nesting level."""

    def __init__(self, directory: Path, config: Object, output_root: Path) -> None:
        self.directory = directory
        self.output_root = output_root
        self.secrets: set[str] = set()
        self.errors: list[str] = []
        self.images: dict[str, Object] = {}
        self.collect_secrets(config)

    def collect_secrets(self, source: Json, private: bool = False) -> None:
        if isinstance(source, dict):
            for key, item in source.items():
                hidden = private or bool(PRIVATE_KEY.search(key)) or key in {"headers", "extra_headers"}
                self.collect_secrets(item, hidden)
                if key == "env" and isinstance(item, dict):
                    for env_value in item.values():
                        if isinstance(env_value, str) and len(env_value) >= 8:
                            self.secrets.add(env_value)
        elif isinstance(source, list):
            for item in source:
                self.collect_secrets(item, private)
        elif private and isinstance(source, str) and len(source) >= 8:
            self.secrets.add(source)

    def image(self, encoded: str) -> Json:
        try:
            payload = base64.b64decode(re.sub(r"\s", "", encoded), validate=True)
            source_hash = hashlib.sha256(payload).hexdigest()
            if source_hash in self.images:
                return self.images[source_hash]
            with Image.open(io.BytesIO(payload)) as image:
                original_size = list(image.size)
                image.load()
                if image.mode in ("RGBA", "LA") or "transparency" in image.info:
                    rgba = image.convert("RGBA")
                    display = Image.new("RGB", image.size, "white")
                    display.paste(rgba, mask=rgba.getchannel("A"))
                else:
                    display = image.convert("RGB")
                display.thumbnail((640, 640), Image.Resampling.LANCZOS)
                display_size = list(display.size)
                display.info.clear()
                output = io.BytesIO()
                display.save(output, format="WEBP", quality=70, method=6)
                saved = output.getvalue()
        except (ValueError, binascii.Error, OSError, Image.DecompressionBombError):
            self.errors.append("An embedded image could not be exported")
            return "[image unavailable]"
        target = self.output_root / "data" / "images" / (hashlib.sha256(saved).hexdigest() + ".webp")
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            target.write_bytes(saved)
        result: Object = {"type": "url", "url": target.relative_to(self.output_root).as_posix(),
                          "original_size": list(original_size), "display_size": list(display_size),
                          "original_bytes": len(payload)}
        self.images[source_hash] = result
        return result

    def string(self, source: str) -> str:
        # Preserve metadata when an image occurs inside a textual tool result.
        # Tool arguments and results can contain JSON encoded inside a string.
        if source.lstrip().startswith(("{", "[")):
            try:
                decoded = value(json.loads(source))
            except (ValueError, TypeError):
                pass
            else:
                cleaned = self.clean(decoded)
                if cleaned != decoded:
                    source = json.dumps(cleaned, ensure_ascii=False)
        source = DATA_IMAGE.sub(lambda match: json.dumps(self.image(match.group().split(",", 1)[1]), separators=(",", ":")), source)
        source = ENCRYPTED_REASONING.sub(lambda match: match.group(1) + "[encrypted reasoning removed]", source)
        for secret in sorted(self.secrets, key=len, reverse=True):
            source = source.replace(secret, REDACTED)
        source = re.sub(r"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]+", REDACTED, source)
        source = re.sub(r"(?i)((?:[a-z0-9_]*(?:api_key|access_token|auth_token|password|secret)[a-z0-9_]*|authorization)[\"']?\s*[=:]\s*)(?:\"[^\"]*\"|'[^']*'|[^\s,;}]+)", lambda match: match.group(1) + REDACTED, source)
        source = re.sub(r"\b(?:sk-[A-Za-z0-9_-]{8,}|gAAAA[A-Za-z0-9_=-]{30,}|eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+)", REDACTED, source)
        source = re.sub(r"/(?:Users|home)/[^\s<>\"']+", "[private path removed]", source)

        def safe_url(match: re.Match[str]) -> str:
            url = match.group()
            try:
                parsed = urlsplit(url)
            except ValueError:
                return REDACTED
            if parsed.username or parsed.password or parsed.query or parsed.fragment or re.search(r"sandbox|tunnel|pinference|primeintellect|localhost|127\.0\.0\.1", parsed.netloc, re.I):
                return "[private URL removed]"
            return url

        return TUNNEL_HOST.sub("[private host removed]", URL.sub(safe_url, source))

    def clean(self, source: Json) -> Json:
        if isinstance(source, str):
            if source.startswith("data:image/") and ";base64," in source:
                return self.image(source.split(",", 1)[1])
            return self.string(source)
        if isinstance(source, list):
            return [self.clean(item) for item in source]
        if isinstance(source, dict):
            if source.get("type") in ("redacted_thinking", "redacted_reasoning", "encrypted_reasoning"):
                return None
            if source.get("type") == "image_url":
                image_url = source.get("image_url")
                url = image_url.get("url") if isinstance(image_url, dict) else image_url
                if isinstance(url, str) and url.startswith("data:image/") and ";base64," in url:
                    return self.image(url.split(",", 1)[1])
            if source.get("type") == "base64" and isinstance(source.get("data"), str):
                encoded = text(source.get("data"), "image data")
                return self.image(encoded)
            result: Object = {}
            for key, item in source.items():
                if PRIVATE_KEY.search(key) or key.lower() in {"headers", "extra_headers", "env", "environment", "env_vars", "provider_state"}:
                    continue
                result[key] = self.clean(item)
            return result
        return source

    def tools(self, source: Json) -> str:
        schemas = self.clean(items(source, "trace.tools"))
        canonical = json.dumps(schemas, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
        reference = hashlib.sha256(canonical.encode()).hexdigest()[:16]
        target = self.directory / "tools" / (reference + ".json")
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(canonical)
        elif target.read_text() != canonical:
            raise ValueError("Tool schema hash collision")
        return reference


def image_paths(source: Json) -> set[str]:
    if isinstance(source, str):
        return set(re.findall(r"data/images/[a-f0-9]{64}\.webp", source))
    children = source.values() if isinstance(source, dict) else source if isinstance(source, list) else []
    result: set[str] = set()
    for child in children:
        result.update(image_paths(child))
    return result




def error_messages(source: Object, cleaner: PublicTrace) -> list[str]:
    result: list[str] = []
    for raw in items(source.get("errors", []), "errors"):
        if isinstance(raw, dict):
            kind = raw.get("type", "Error")
            message = raw.get("message", "No error detail was recorded")
            result.append(cleaner.string(f"{kind}: {message}"))
        else:
            result.append(cleaner.string(str(raw)))
    return result


def score(trace: Object) -> float:
    rewards = obj(trace.get("rewards"), "trace.rewards")
    if set(rewards) != {"box_f1", "label_f1"}:
        raise ValueError("Both worklist reward components are required")
    total = 0.0
    for name in ("box_f1", "label_f1"):
        reward = obj(rewards[name], name)
        number = reward.get("score")
        if isinstance(number, bool) or not isinstance(number, (int, float)) or not 0 <= number <= 1:
            raise ValueError(f"{name} has no valid score")
        if reward.get("weight") != 0.5:
            raise ValueError(f"{name} has an unexpected weight")
        total += number * 0.5
    return total


def task_identity(record: Object, line: int) -> str:
    task = obj(record.get("task"), "record.task")
    for field in ("key", "hash"):
        identity = task.get(field)
        if isinstance(identity, str) and identity:
            return field + ":" + identity
    data = obj(task.get("data"), "task.data")
    idx = data.get("idx")
    if isinstance(idx, int) and not isinstance(idx, bool):
        return "idx:" + str(idx)
    raise ValueError(f"Record {line} has no stable task identity")


def film_row(key: str, records: list[tuple[int, Object]], config: RunConfig,
             directory: Path, resolved: Object, output_root: Path,
             labels: Json, predictions: list[Json], parsed: bool,
             score_note: str | None, score_provenance: Object) -> FilmRow:
    cleaner = PublicTrace(directory, resolved, output_root)
    for _, record in records:
        for raw_trace in items(record.get("traces", []), "record.traces"):
            cleaner.collect_secrets(obj(raw_trace, "trace").get("agent"))
    attempts: list[Json] = []
    public_attempts: list[Json] = []
    has_thinking = False
    model_calls = 0
    images_viewed = 0
    selected_reply: str | None = None
    selected_reward: float | None = None
    selected_errors: list[str] = []
    selected_warnings: list[str] = []
    selected_stop: str | None = None
    data = obj(obj(records[-1][1].get("task"), "task").get("data"), "task.data")
    prompt = items(data.get("prompt", []), "task prompt")
    clean_prompt = cleaner.clean(prompt)
    prompt_images = image_paths(clean_prompt)
    for attempt, (line, record) in enumerate(records, 1):
        framework_errors = error_messages(record, cleaner)
        raw_traces = items(record.get("traces", []), "record.traces")
        public_traces: list[Json] = []
        trace: Object | None = None
        attempt_images: set[str] = set()
        attempt_prompt_images = set(prompt_images)
        attempt_has_thinking = False
        attempt_calls = 0
        for trace_number, raw_trace in enumerate(raw_traces):
            trace = obj(raw_trace, "trace")
            framework_errors.extend(error_messages(trace, cleaner))
            nodes: list[Json] = []
            sampled = False
            for node_number, raw_node in enumerate(items(trace.get("nodes", []), "trace.nodes")):
                node = obj(raw_node, "node")
                message = cleaner.clean(obj(node.get("message"), "message"))
                nodes.append({"index": node_number, "parent": node.get("parent"),
                              "sampled": node.get("sampled"), "timestamp": node.get("timestamp"), "message": message})
                message_images = image_paths(message)
                attempt_images.update(message_images)
                sampled |= node.get("sampled") is True
                if not sampled:
                    # Harnesses may resize the prompt before the first model call.
                    attempt_prompt_images.update(message_images)
                if isinstance(message, dict):
                    returned = message.get("reasoning_content")
                    attempt_has_thinking |= isinstance(returned, str) and bool(returned)
            calls: list[Json] = []
            for raw_call in items(trace.get("calls", []), "trace.calls"):
                call = obj(raw_call, "model call")
                calls.append(cleaner.clean({
                    field: call[field]
                    for field in ("node", "model", "sampling", "finish_reason", "usage", "time", "error", "policy")
                    if field in call
                }))
            attempt_calls += len(calls)
            public_traces.append({"id": cleaner.clean(trace.get("id")), "index": trace_number,
                                  "ok": trace.get("ok"), "is_completed": trace.get("is_completed"),
                                  "stop_condition": cleaner.clean(trace.get("stop_condition")),
                                  "nodes": nodes, "tools_ref": cleaner.tools(trace.get("tools", [])),
                                  "calls": calls,
                                  "rewards": cleaner.clean(trace.get("rewards", {})),
                                  "metrics": cleaner.clean(trace.get("metrics", {})),
                                  "score_provenance": cleaner.clean(trace.get("score_provenance")),
                                  "errors": error_messages(trace, cleaner)})
        has_thinking = attempt_has_thinking
        model_calls = attempt_calls
        images_viewed = len(attempt_images - attempt_prompt_images)
        stop = trace.get("stop_condition") if trace is not None else None
        attempt_stop = stop if isinstance(stop, str) else None
        reply: str | None = None
        reward: float | None = None
        # The framework's verdict decides whether errors it recorded block scoring.
        # After an ok verdict they are warnings, for example a cleanup failure after the answer
        # or Claude Code's service error message when the turn limit stopped the run.
        verdict_ok = (record.get("ok") is True and len(raw_traces) == 1 and trace is not None
                      and trace.get("ok") is True and trace.get("is_completed") is True)
        errors: list[str] = [] if verdict_ok else list(framework_errors)
        warnings: list[str] = list(dict.fromkeys(framework_errors)) if verdict_ok else []
        if record.get("ok") is not True:
            errors.append("The episode did not finish successfully")
        if len(raw_traces) != 1 or trace is None:
            errors.append("Expected one agent trace for this episode")
        else:
            try:
                agent = obj(obj(trace.get("agent"), "trace.agent").get("config"), "trace agent config")
                harness = obj(agent.get("harness"), "trace harness")
                if (agent.get("model"), harness.get("id"), harness.get("version"), agent.get("max_turns")) != (config.model, config.harness, config.harness_version, config.max_turns):
                    raise ValueError("Trace agent differs from the resolved run config")
                reply = obj(trace.get("info", {}), "trace.info").get("submitted")
                if reply is not None and not isinstance(reply, str):
                    raise ValueError("The saved report must be text or null")
                if trace.get("ok") is not True or trace.get("is_completed") is not True:
                    raise ValueError("The agent trace did not finish successfully")
                if not reply:
                    warnings.append("No report.json content was saved. The production scorer treats this as an empty report.")
                reward = score(trace)
            except ValueError as error:
                errors.append(cleaner.string(str(error)))
        errors = list(dict.fromkeys(errors))
        if errors:
            reward = None
        provenance: Object = {"attempt": attempt, "source_line": line,
                              "episode_id": cleaner.clean(record.get("id")),
                              "selected": attempt == len(records),
                              "status": "error" if errors else "success", "reward": reward,
                              "errors": list(errors), "warnings": warnings}
        attempts.append(provenance)
        public_attempts.append({**provenance, "traces": public_traces})
        selected_reply, selected_reward, selected_errors, selected_warnings = reply, reward, errors, warnings
        selected_stop = attempt_stop
    selected_errors = list(dict.fromkeys(selected_errors + cleaner.errors))
    if selected_errors:
        selected_reward = None
    if cleaner.errors:
        raise ValueError("; ".join(cleaner.errors))
    filename = hashlib.sha256(key.encode()).hexdigest() + ".json"
    trace_path = directory / "traces" / filename
    dump(trace_path, {"schema": 2, "task_key": key, "film": cleaner.clean(data.get("name")),
                      "prompt": clean_prompt, "boxes": data.get("boxes", []), "labels": labels,
                      "answer_source": "report.json", "score_note": score_note,
                      "score_provenance": score_provenance, "attempts": public_attempts,
                      "reasoning_notice": "Only reasoning returned by the API is shown. Private encrypted reasoning is not included."})


    safe_reply = cleaner.string(selected_reply) if selected_reply is not None else None
    return {"film": cleaner.string(text(data.get("name"), "film name")), "task_key": key,
            "boxes": data.get("boxes", []), "labels": labels, "predictions": cleaner.clean(predictions),
            "reward": selected_reward, "parsed": parsed, "reply": safe_reply if selected_reply else None,
            "answer_source": "report.json", "score_note": score_note, "score_provenance": score_provenance,
            "has_thinking": has_thinking, "model_calls": model_calls, "images_viewed": images_viewed,
            "status": "error" if selected_errors else "success", "errors": selected_errors,
            "warnings": selected_warnings,
            "stop_condition": selected_stop,
            "trace_path": trace_path.relative_to(output_root).as_posix(),
            "image_path": next(iter(sorted(prompt_images)), None), "attempts": attempts}


def mean_interval(values: list[float], z: float = 1.96) -> list[float]:
    """95% interval for the mean reward over films (normal approximation), kept inside 0 to 1."""
    count = len(values)
    mean = sum(values) / count
    spread = math.sqrt(sum((v - mean) ** 2 for v in values) / (count - 1) / count) if count > 1 else 0.0
    return [max(0.0, mean - z * spread), min(1.0, mean + z * spread)]


def wilson_interval(successes: int, total: int, z: float = 1.96) -> list[float]:
    """95% Wilson interval for a share of films. It stays inside 0 to 1 and is sensible near 0."""
    share = successes / total
    denominator = 1 + z * z / total
    centre = (share + z * z / (2 * total)) / denominator
    half = z * math.sqrt(share * (1 - share) / total + z * z / (4 * total * total)) / denominator
    return [max(0.0, centre - half), min(1.0, centre + half)]


if __name__ == "__main__":
    from publish_catalog import main

    main()
