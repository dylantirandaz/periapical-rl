"""Publish a verifiers eval run (traces.jsonl) to the site as benchmark traces.

Usage (needs the periapical_lesions package for the benchmark's own parser):
    <env-venv>/python publish_benchmark.py <run-dir> [--label "GPT-6 Astra · one call"]
"""

import argparse
import json
import time
from pathlib import Path

from periapical_lesions.taskset import parse_lesions

SITE = Path(__file__).resolve().parent


def slug(model: str) -> str:
    return "benchmark-" + model.split("/")[-1].lower()


def film_row(record: dict) -> dict:
    trace = record["traces"][0]
    data = record["task"]["data"]
    message = trace["nodes"][-1]["message"]
    reply = message["content"]
    lesions = parse_lesions(reply)
    weights = trace["rewards"]
    reward = sum(r["score"] * r["weight"] for r in weights.values()) / sum(r["weight"] for r in weights.values())
    thinking = message.get("reasoning_content") or ""
    return {
        "film": data["name"],
        "boxes": data["boxes"],
        "pai": data["pai"],
        "lesions": lesions,
        "reward": reward,
        "parsed": float(bool(lesions)),
        "reply": reply,
        "thinking": thinking,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--model")
    parser.add_argument("--harness", default="one call")
    args = parser.parse_args()

    records = [json.loads(line) for line in (args.run_dir / "traces.jsonl").open()]
    films = [film_row(r) for r in records if r.get("ok")]
    config = json.loads((args.run_dir / "configs" / "resolved" / "eval.json").read_text())
    model = args.model or config["model"]
    run_id = slug(model)
    mean = sum(f["reward"] for f in films) / len(films)

    run_dir = SITE / "data" / run_id
    (run_dir / "evals").mkdir(parents=True, exist_ok=True)
    (run_dir / "evals" / "test_0000.json").write_text(json.dumps(films, separators=(",", ":")))
    (run_dir / "summary.json").write_text(json.dumps({
        "id": run_id, "name": args.label, "stage": "benchmark", "model": model,
        "harness": args.harness, "films": len(films), "eval_views": ["test_0000"],
        "mean_reward": mean, "updated": int(time.time()),
    }, separators=(",", ":")))

    index_path = SITE / "data" / "benchmarks.json"
    index = [b for b in (json.loads(index_path.read_text()) if index_path.exists() else []) if b["id"] != run_id]
    index.append({"id": run_id, "name": args.label, "model": model, "harness": args.harness,
                  "films": len(films), "mean_reward": mean, "updated": int(time.time())})
    index_path.write_text(json.dumps(sorted(index, key=lambda b: b["updated"], reverse=True), separators=(",", ":")))
    print(f"{run_id}: {len(films)} films, mean reward {mean:.3f}")


if __name__ == "__main__":
    main()
