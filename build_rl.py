"""Build assets/rl.json: validation score by RL step, rescored with the benchmark's current scorer.

Usage, with the environment venv (it has periapical_lesions):
  python build_rl.py <rl-run-dir> <untrained-eval.json> [<output>]

<rl-run-dir>/evals/validation_NNNN.json holds one row per film: film, boxes, pai, lesions (the model's parsed
answer). Every row is scored again with `film_scores` from the taskset, so the numbers use the leaderboard's
formula (1/2 lesion F1 + 1/2 graded F1) whatever formula was current when the run was logged. <untrained-eval.json>
is the same evaluation of the model before any training, on the same films.

Each checkpoint gets a bootstrap 95% interval over films and a shuffle control: its answers scored against every
other film's labels. A model that reads the film scores far lower there than on the real labels.
"""

from __future__ import annotations

import json
import random
import statistics
import sys
from pathlib import Path

from periapical_lesions.taskset import film_scores

BOOTSTRAP_RESAMPLES = 2000
SEED = 0


def score(row: dict, boxes: list, pai: list) -> float:
    lesion_f1, graded_f1 = film_scores(row["lesions"], boxes, pai)
    return 0.5 * lesion_f1 + 0.5 * graded_f1


def interval(values: list[float]) -> list[float]:
    rng = random.Random(SEED)
    means = sorted(statistics.mean(rng.choices(values, k=len(values))) for _ in range(BOOTSTRAP_RESAMPLES))
    return [round(means[int(0.025 * BOOTSTRAP_RESAMPLES)], 4), round(means[int(0.975 * BOOTSTRAP_RESAMPLES)], 4)]


def evaluate(rows: list[dict]) -> dict:
    real = [score(row, row["boxes"], row["pai"]) for row in rows]
    shuffled = [
        statistics.mean(score(row, other["boxes"], other["pai"]) for other in rows if other is not row)
        for row in rows
    ]
    return {"score": round(statistics.mean(real), 4), "ci95": interval(real), "shuffled": round(statistics.mean(shuffled), 4)}


def main(run_dir: Path, untrained_path: Path, output: Path) -> None:
    files = sorted((run_dir / "evals").glob("validation_*.json"), key=lambda path: int(path.stem.split("_")[1]))
    curve = [{"step": int(path.stem.split("_")[1]), **evaluate(json.loads(path.read_text()))} for path in files]
    first = json.loads(files[0].read_text())
    untrained = json.loads(untrained_path.read_text())
    if [row["film"] for row in untrained] != [row["film"] for row in first]:
        raise SystemExit("the untrained evaluation covers different films than the RL evaluations")
    ladder = [
        {"label": "untrained", "score": evaluate(untrained)["score"]},
        {"label": "fine-tuned", "score": curve[0]["score"]},
        {"label": "+ RL", "score": curve[-1]["score"]},
    ]
    paired = [score(late, late["boxes"], late["pai"]) - score(early, early["boxes"], early["pai"])
              for early, late in zip(first, json.loads(files[-1].read_text()))]
    output.write_text(json.dumps({
        "split": "validation", "films": len(first), "ladder": ladder, "curve": curve,
        "rl_gain": {"mean": round(statistics.mean(paired), 4), "ci95": interval(paired)},
    }, indent=1) + "\n")
    print(f"{output}: {len(curve)} checkpoints, ladder {[rung['score'] for rung in ladder]}, paired RL gain {statistics.mean(paired):+.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3] if len(sys.argv) > 3 else "assets/rl.json"))
