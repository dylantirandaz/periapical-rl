"""Publish a Tinker RL run's progress to the live site (GitHub Pages), every INTERVAL_S seconds.

    python sync.py --run-dir ~/tinker-rl/runs/inkling --run-id inkling --name "Training run #1" --max-steps 100
"""
import argparse
import io
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path

from datasets import Image, load_dataset
from PIL import Image as PILImage

DATASET = "tirandazdylan/periapical-lesions-pai"
SITE = Path(__file__).resolve().parent
THUMB_WIDTH = 720
INTERVAL_S = 90
LIVE_WINDOW_S = 600
"""A run counts as live while its logs changed within this many seconds."""


def load_films():
    rows = load_dataset(DATASET, data_files={"train": "data/train-*.parquet"}, split="train")
    rows = rows.cast_column("image", Image(decode=False))
    index = {name: i for i, name in enumerate(rows["image_id"])}
    return rows, index


def film_record(rows, index, name: str) -> dict:
    row = rows[index[name]]
    w, h = row["width"], row["height"]
    thumb = SITE / "films" / f"{name}.jpg"
    if not thumb.exists():
        image = PILImage.open(io.BytesIO(row["image"]["bytes"])).convert("L")
        image = image.resize((THUMB_WIDTH, round(image.height * THUMB_WIDTH / image.width)), PILImage.LANCZOS)
        thumb.parent.mkdir(exist_ok=True)
        image.save(thumb, "JPEG", quality=78, optimize=True)
    boxes = [[x / w, y / h, (x + bw) / w, (y + bh) / h] for x, y, bw, bh in row["objects"]["bbox"]]
    return {"film": name, "boxes": boxes, "pai": [p + 3 for p in row["objects"]["pai"]]}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open()] if path.exists() else []


def summarize(args, run_dir: Path) -> tuple[dict, dict[int, list[dict]]]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    steps = [{"step": m["step"], "reward": m["env/all/reward"], "lesion_f1": m.get("env/all/lesion_f1", 0),
              "pai_accuracy": m.get("env/all/pai_accuracy", 0), "parsed": m.get("env/all/parsed", 0)}
             for m in metrics if "env/all/reward" in m]
    tests = [{"step": m["step"], **{k.removeprefix("test/"): v for k, v in m.items() if k.startswith("test/")}}
             for m in metrics if "test/reward" in m]
    rollouts = defaultdict(list)
    for event in read_jsonl(run_dir / "live.jsonl"):
        rollouts[event["step"]].append(event)
    newest = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()), default=0)
    summary = {"id": args.run_id, "name": args.name, "model": args.model, "max_steps": args.max_steps,
               "live": time.time() - newest < LIVE_WINDOW_S, "updated": int(newest),
               "current_step": max([*rollouts, *(s["step"] for s in steps), 0]),
               "steps": steps, "tests": tests, "rollout_steps": sorted(rollouts)}
    return summary, rollouts


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def publish(message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=SITE, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=SITE).returncode:
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=SITE, check=True)
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=SITE, check=True)


def sync_once(args, rows, index) -> dict:
    run_dir = Path(args.run_dir).expanduser()
    summary, rollouts = summarize(args, run_dir)
    data = SITE / "data" / args.run_id
    for step, events in rollouts.items():
        by_film = defaultdict(list)
        for event in events:
            by_film[event["film"]].append({k: event[k] for k in ("reward", "lesions", "parsed", "lesion_f1", "pai_accuracy")})
        write_json(data / "steps" / f"{step:04d}.json",
                   [{**film_record(rows, index, film), "rollouts": attempts} for film, attempts in by_film.items()])
    write_json(data / "summary.json", summary)
    runs_path = SITE / "data" / "runs.json"
    runs = [r for r in (json.loads(runs_path.read_text()) if runs_path.exists() else []) if r["id"] != args.run_id]
    runs.append({k: summary[k] for k in ("id", "name", "model", "max_steps", "live", "updated", "current_step")})
    write_json(runs_path, sorted(runs, key=lambda r: r["updated"], reverse=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--model", default="Inkling")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    rows, index = load_films()
    while True:
        summary = sync_once(args, rows, index)
        publish(f"Update {args.run_id} to step {summary['current_step']}")
        if args.once or not summary["live"]:
            break
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
