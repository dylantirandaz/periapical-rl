"""Publish a Tinker training run's progress to the live site (GitHub Pages), every INTERVAL_S seconds.

    python sync.py --run-dir ~/tinker-rl/runs/sft --run-id sft --name "Fine-tuning" --stage fine-tuning --max-steps 483
"""
import argparse
import io
import json
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import mean

from datasets import Image, load_dataset
from huggingface_hub import snapshot_download
from PIL import Image as PILImage

from periapical_lesions.taskset import parse_lesions

DATASET = "tirandazdylan/periapical-lesions-pai"
SPLITS = ("train", "validation", "test")
EVAL_SPLITS = ("validation", "test")
SITE = Path(__file__).resolve().parent
DATA_DIR = Path.home() / ".cache" / "periapical-lesions-pai"
THUMB_WIDTH = 720
INTERVAL_S = 90
LIVE_WINDOW_S = 900
"""A run counts as live while its logs changed within this many seconds."""


def load_films() -> dict:
    """Film name -> (split rows, row index), across every split."""
    films = {}
    for split in SPLITS:
        if not any(DATA_DIR.glob(f"data/{split}-*.parquet")):
            snapshot_download(DATASET, repo_type="dataset", local_dir=DATA_DIR, allow_patterns=["data/*.parquet"])
        rows = load_dataset("parquet", data_files={split: str(DATA_DIR / f"data/{split}-*.parquet")}, split=split)
        rows = rows.cast_column("image", Image(decode=False))
        films.update({name: (rows, i) for i, name in enumerate(rows["image_id"])})
    return films


def film_record(films: dict, name: str) -> dict:
    rows, index = films[name]
    row = rows[index]
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


def summarize(args, run_dir: Path) -> tuple[dict, dict[int, list[dict]], list[dict]]:
    metrics = read_jsonl(run_dir / "metrics.jsonl")
    rollouts = defaultdict(list)
    for event in read_jsonl(run_dir / "live.jsonl"):
        rollouts[event["step"]].append(event)
    # From every rollout: the trainer's own env metrics skip groups whose rollouts all scored the same.
    steps = [{"step": step, "reward": mean(e.get("train_reward", e["reward"]) for e in events),
              "benchmark": mean(e["reward"] for e in events), "parsed": mean(e.get("parsed", 0) for e in events)}
             for step, events in sorted(rollouts.items())]
    evals = []
    for split in EVAL_SPLITS:
        rows = [m for m in metrics if f"{split}/reward" in m]
        evals += [{"step": m["step"], "split": split, "file": f"{split}_{k:03d}.jsonl",
                   **{key.removeprefix(f"{split}/"): v for key, v in m.items() if key.startswith(f"{split}/")}}
                  for k, m in enumerate(rows)]
    loss = [{"step": m["step"], "nll": m["train_mean_nll"]} for m in metrics if "train_mean_nll" in m]
    newest = max((p.stat().st_mtime for p in run_dir.rglob("*") if p.is_file()), default=0)
    summary = {"id": args.run_id, "name": args.name, "stage": args.stage, "model": args.model, "max_steps": args.max_steps,
               "live": time.time() - newest < LIVE_WINDOW_S, "updated": int(newest),
               "current_step": max([*rollouts, *(s["step"] for s in steps), *(e["step"] for e in evals), *(l["step"] for l in loss), 0]),
               "steps": steps, "evals": [{k: v for k, v in e.items() if k != "file"} for e in evals],
               "loss": loss[-1:] if loss else [], "rollout_steps": sorted(rollouts)}
    return summary, rollouts, evals


def write_json(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, separators=(",", ":")))


def publish(message: str) -> None:
    subprocess.run(["git", "add", "-A"], cwd=SITE, check=True)
    if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=SITE).returncode:
        subprocess.run(["git", "commit", "-q", "-m", message], cwd=SITE, check=True)
        subprocess.run(["git", "push", "-q", "origin", "main"], cwd=SITE, check=True)


def sync_once(args, films: dict) -> dict:
    run_dir = Path(args.run_dir).expanduser()
    summary, rollouts, evals = summarize(args, run_dir)
    data = SITE / "data" / args.run_id
    for step, events in rollouts.items():
        by_film = defaultdict(list)
        for event in events:
            by_film[event["film"]].append({k: event[k] for k in ("reward", "lesions", "parsed", "lesion_f1", "pai_accuracy")}
                                          | {"train_reward": event.get("train_reward", event["reward"])})
        write_json(data / "steps" / f"{step:04d}.json",
                   [{**film_record(films, film), "rollouts": attempts} for film, attempts in by_film.items()])
    for ev in evals:
        target = data / "evals" / f"{ev['split']}_{ev['step']:04d}.json"
        rows = [row for row in read_jsonl(run_dir / ev["file"]) if row.get("sample", 0) == 0]
        if rows and not target.exists():
            write_json(target, [{**film_record(films, row["film"]), "lesions": parse_lesions(row["reply"]),
                                 "reward": row["reward"], "parsed": row["parsed"]} for row in rows])
    summary["eval_views"] = sorted(p.stem for p in (data / "evals").glob("*.json")) if (data / "evals").exists() else []
    write_json(data / "summary.json", summary)
    runs_path = SITE / "data" / "runs.json"
    runs = [r for r in (json.loads(runs_path.read_text()) if runs_path.exists() else []) if r["id"] != args.run_id]
    runs.append({k: summary[k] for k in ("id", "name", "stage", "model", "max_steps", "live", "updated", "current_step")})
    write_json(runs_path, sorted(runs, key=lambda r: r["updated"], reverse=True))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--stage", choices=("fine-tuning", "rl"), default="rl")
    parser.add_argument("--model", default="Inkling")
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    films = load_films()
    while True:
        summary = sync_once(args, films)
        publish(f"Update {args.run_id} to step {summary['current_step']}")
        if args.once or not summary["live"]:
            break
        time.sleep(INTERVAL_S)


if __name__ == "__main__":
    main()
