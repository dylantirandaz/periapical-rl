"""Build assets/training.json: the validation score through fine-tuning and then RL, in one curve.

Usage, with the environment venv (it has periapical_lesions):
    python build_training.py <sft-run-dir> <rl-run-dir> [<output>]

Both runs' evals/validation_NNNN.json hold one row per film: film, boxes, pai, lesions (the model's parsed
answer). Every row is scored again with `film_scores` from the taskset, so the numbers use the leaderboard's
formula (1/2 lesion F1 + 1/2 graded F1) whatever formula was current when the runs were logged.

The RL run started from the fine-tuning run's last checkpoint, so its steps continue the same axis: RL step s
is plotted at sft_max_step + s. Each checkpoint gets a bootstrap 95% interval over films and a shuffle control
(its answers scored against every other film's labels); a model that reads the film scores far lower there.
Both runs sampled their validation answers, so the two measurements of the shared checkpoint can differ by
about one standard error; the intervals overlap.
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


def phase(run_dir: Path, step_offset: int) -> tuple[list[dict], list[list[dict]]]:
    files = sorted((run_dir / "evals").glob("validation_*.json"), key=lambda path: int(path.stem.split("_")[1]))
    points, all_rows = [], []
    for path in files:
        rows = json.loads(path.read_text())
        all_rows.append(rows)
        points.append({"step": int(path.stem.split("_")[1]) + step_offset, **evaluate(rows)})
    films = [[row["film"] for row in rows] for rows in all_rows]
    if any(film_list != films[0] for film_list in films):
        raise SystemExit(f"{run_dir}: its evaluations cover different films")
    return points, all_rows


def main(sft_dir: Path, rl_dir: Path, output: Path) -> None:
    sft_points, sft_rows = phase(sft_dir, 0)
    rl_points, rl_rows = phase(rl_dir, sft_points[-1]["step"])
    if [row["film"] for row in sft_rows[0]] != [row["film"] for row in rl_rows[0]]:
        raise SystemExit("the two runs validated on different films")
    paired = [score(late, late["boxes"], late["pai"]) - score(early, early["boxes"], early["pai"])
              for early, late in zip(rl_rows[0], rl_rows[-1])]
    ladder = [
        {"label": "untrained", "score": sft_points[0]["score"]},
        {"label": "fine-tuned", "score": rl_points[0]["score"]},
        {"label": "+ RL", "score": rl_points[-1]["score"]},
    ]
    output.write_text(json.dumps({
        "split": "validation", "films": len(sft_rows[0]),
        "phases": [
            {"name": "fine-tuned", "points": sft_points},
            {"name": "+ RL", "points": rl_points},
        ],
        "ladder": ladder,
        "rl_gain": {"mean": round(statistics.mean(paired), 4), "ci95": interval(paired)},
    }, indent=1) + "\n")
    print(f"{output}: fine-tune {[p['score'] for p in sft_points]}, RL {[p['score'] for p in rl_points]}, "
          f"paired RL gain {statistics.mean(paired):+.3f}")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3] if len(sys.argv) > 3 else "assets/training.json"))
