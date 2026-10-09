"""Build a public catalog from saved runs. Do not change a source tree."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import shutil
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING, Literal, TypedDict, assert_never

from publish_benchmark import (
    SITE, Json, Object, PublicTrace, RunConfig, dump, film_row, image_paths,
    integer, items, load, mean_interval, obj, score, task_identity, text, value,
    wilson_interval,
)

if TYPE_CHECKING:
    from dentex_teeth.taskset import Finding as DentexFinding
    from pediatric_dental_disease.taskset import Finding as PediatricFinding

type ParsedFinding = DentexFinding | PediatricFinding


class ScoredFilm(TypedDict):
    film: str
    task_key: str
    reward: float
    offset: int
    line: int
    terms: tuple[float, float]
    original_reward: float
    model_calls: int
    stop_condition: Json

DIAGNOSIS_NOTE = (
    "Scores were computed again with the corrected 0.3.1 production scorer from saved report.json files. "
    "The scorer now reads the disease key. Reports and model messages did not change. No new model run was made."
)
WORKLIST_NOTE = (
    "These saved worklist runs used an earlier report command that omitted .py. "
    "The messages are unchanged. These are not new runs of the corrected report command."
)


@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    short_name: str
    count: int
    package: str
    project: str
    mode: Literal["diagnosis", "enumeration"] | None
    run_names: tuple[str, str]
    description: str
    task: str
    score_text: str


SPECS = (
    Spec("dentex-diagnosis", "DENTEX diagnosis", "Diagnosis", 74, "dentex_teeth", "dentex-teeth", "diagnosis",
         ("dentex-diagnosis-opus55-v3", "dentex-diagnosis-haiku45-v3"),
         "Find teeth with disease on adult panoramic films and report the disease class.",
         "Box each tooth with disease. Use one of four disease classes.",
         "Reward is 0.5 × box F1 + 0.5 × label F1. Boxes match one-to-one at IoU ≥ 0.5."),
    Spec("dentex-enumeration", "DENTEX enumeration", "Enumeration", 66, "dentex_teeth", "dentex-teeth", "enumeration",
         ("dentex-enumeration-opus55-v3", "dentex-enumeration-haiku45-v3"),
         "Find and number the teeth on adult panoramic films.",
         "Box each tooth and give its FDI tooth number.",
         "Reward is 0.5 × box F1 + 0.5 × label F1. Boxes match one-to-one at IoU ≥ 0.7."),
    Spec("pediatric-dental-disease", "Pediatric dental disease", "Pediatric disease", 30,
         "pediatric_dental_disease", "pediatric-dental-disease", None,
         ("pediatric-v2-opus55", "pediatric-v2-haiku45"),
         "Find teeth with disease on children's panoramic films.",
         "Box each tooth with disease. Use one of six disease classes.",
         "Reward is 0.5 × box F1 + 0.5 × label F1. Boxes match one-to-one at IoU ≥ 0.5."),
)


def digest(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def same(actual: object, expected: object, label: str) -> None:
    if isinstance(actual, (int, float)) and not isinstance(actual, bool) and isinstance(expected, (int, float)):
        equal = math.isclose(actual, expected, rel_tol=0, abs_tol=1e-12)
    else:
        equal = actual == expected
    if not equal:
        raise ValueError(f"{label} differs from the saved evidence")


def rank_sample[Row: Object | ScoredFilm](rows: list[Row]) -> list[Row]:
    if len(rows) < 5:
        raise ValueError("At least five films are required")
    ordered = sorted(rows, key=lambda row: (row["reward"], row["film"]))
    return [ordered[round((len(ordered) - 1) * q)] for q in (0, .25, .5, .75, 1)]


def selection(name: str) -> str:
    return (f"Five score-stratified examples from {name}, the highest-scoring complete run. "
            "Sort by (reward, film ID), then select indices round((n−1)×q) for q = 0, 0.25, 0.5, 0.75, 1. "
            "These are not a random sample. Full-run scores do not come from these five examples.")


def safe_source(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError("A public asset is missing or outside its source tree")
    return path


def copy_images(payload: Json, source: Path, output: Path) -> None:
    for relative in sorted(image_paths(payload)):
        path = safe_source(source, relative)
        if path.stem != digest(path):
            raise ValueError("A public image differs from its content hash")
        target = output / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copyfile(path, target)


def periapical(source: Path, output: Path) -> tuple[list[Object], Object]:
    index_path = source / "data/benchmarks.json"
    entries = [obj(row, "periapical run") for row in items(load(index_path), "periapical index")]
    if len(entries) != 6 or len({entry.get("id") for entry in entries}) != 6:
        raise ValueError("The original six-run periapical index is required")
    for entry in entries:
        if entry.get("benchmark", "periapical-lesions") != "periapical-lesions":
            raise ValueError("The periapical source must be the original public data")
        if entry.get("status") != "complete" or entry.get("expected") != 200 or entry.get("succeeded") != 200:
            raise ValueError("The original complete 200-film periapical runs are required")
    winner = max(entries, key=lambda entry: entry["mean_reward"])
    same(winner["id"], "benchmark-gpt-6-astra-codex", "Periapical winner")
    run_id = text(winner["id"], "run ID")
    row_path = safe_source(source, f"data/{run_id}/evals/test_0000.json")
    rows = [obj(row, "periapical row") for row in items(load(row_path), "periapical rows")]
    if len(rows) != winner["expected"] or len({row["film"] for row in rows}) != len(rows):
        raise ValueError("The periapical winner must have all 200 original films")
    if any(row.get("status") != "success" or not isinstance(row.get("reward"), (int, float)) for row in rows):
        raise ValueError("The periapical winner has an unscored film")
    same(sum(row["reward"] for row in rows) / len(rows), winner["mean_reward"], "Periapical mean")
    selected = rank_sample(rows)
    directory = output / "data" / run_id
    cleaner = PublicTrace(directory, {}, output)
    provenance = {"method": "Original published scores; no new scoring or model calls.",
                  "source_index_sha256": digest(index_path), "source_rows_sha256": digest(row_path)}
    result = []
    for original in selected:
        row = obj(cleaner.clean(original), "public row")
        row["labels"] = row.pop("pai")
        row["predictions"] = [{"box": obj(item, "lesion")["box"], "label": obj(item, "lesion")["pai"]}
                              for item in items(row.pop("lesions"), "lesions")]
        row.update(answer_source="final_reply", score_note=None)
        trace_path = safe_source(source, text(row["trace_path"], "trace path"))
        payload = obj(cleaner.clean(load(trace_path)), "public trace")
        same(payload.get("schema"), 2, "Periapical trace schema")
        payload["labels"] = payload.pop("pai")
        payload.update(answer_source="final_reply", score_note=None)
        row["score_provenance"] = {**provenance, "source_trace_sha256": digest(trace_path)}
        payload["score_provenance"] = row["score_provenance"]
        for attempt in items(payload["attempts"], "attempts"):
            for raw_trace in items(obj(attempt, "attempt")["traces"], "traces"):
                trace = obj(raw_trace, "trace")
                ref = text(trace["tools_ref"], "tools reference")
                tools = safe_source(source, f"data/{run_id}/tools/{ref}.json")
                trace["tools_ref"] = cleaner.tools(load(tools))
        copy_images(payload, source, output)
        copy_images(row, source, output)
        dump(output / text(row["trace_path"], "trace path"), payload)
        result.append(row)
    if cleaner.errors:
        raise ValueError("; ".join(cleaner.errors))
    dump(directory / "evals/test_0000.json", result)
    # Preserve every existing aggregate value. Never compute a mean from the five rows.
    for entry in entries:
        entry.update(benchmark="periapical-lesions", public_traces=5 if entry["id"] == run_id else 0)
        entry["score_provenance"] = provenance
        dump(output / "data" / text(entry["id"], "run ID") / "summary.json",
             {**entry, "stage": "benchmark", "eval_views": ["test_0000"] if entry["public_traces"] else []})
    catalog = {
        "id": "periapical-lesions", "name": "Periapical lesions", "short_name": "Periapical lesions",
        "description": "Find periapical lesions and assign a periapical index score on panoramic films.",
        "task": "Box each periapical lesion and give its PAI grade, from 3 to 5.",
        "score": "Reward is 0.5 × lesion F1 + 0.5 × PAI F1. Boxes match one-to-one at IoU ≥ 0.3.",
        "protocol": "The scored answer is the final chat reply. One attempt per film, up to 50 agent turns. The six published model scores are unchanged.",
        "coverage": "Each model was evaluated on the same 200 films from a 387-film test split. This is not full test-split coverage.",
        "test_films": 387, "evaluated_films": 200,
        "source": {"name": "Panoramic radiographs with periapical lesions",
                   "url": "https://doi.org/10.17632/kx52tk2ddj.3",
                   "attribution": "Do HV, Mendeley Data V3, 2024. Duplicate and augmented films were removed; boxes were converted and splits were added. Display images were reduced to 640 pixels.",
                   "license": "CC BY 4.0"},
        "selection": selection(text(winner["name"], "model name")), "sample_run": run_id,
        "sample_films": [row["film"] for row in result], "score_note": None,
    }
    return entries, catalog


def run_config(resolved: Object, spec: Spec, revision: str) -> RunConfig:
    env = obj(resolved.get("env"), "env")
    taskset = obj(env.get("taskset"), "taskset")
    agent = obj(env.get("agent"), "agent")
    harness = obj(agent.get("harness"), "harness")
    same(taskset.get("id"), spec.project, "Taskset")
    same(taskset.get("split"), "test", "Split")
    if spec.mode:
        same(taskset.get("mode"), spec.mode, "DENTEX mode")
    same(taskset.get("revision", revision), revision, "Dataset revision")
    same(resolved.get("num_rollouts"), 1, "Rollout count")
    count = resolved.get("num_tasks")
    same(spec.count if count is None else integer(count, "num_tasks"), spec.count, "Film count")
    same(agent.get("max_turns"), 50, "Turn limit")
    same(harness.get("id"), "claude_code", "Harness")
    client = obj(agent.get("client") or resolved.get("client"), "client")
    same(client.get("base_url"), "https://api.pinference.ai/api/v1", "Inference route")
    retries = obj(env.get("retries", {"max_retries": 0, "include": []}), "retries")
    retry_count = integer(retries.get("max_retries", 0), "retry count")
    retry_types = items(retries.get("include", []), "retry types")
    if retry_count < 0 or not all(isinstance(item, str) for item in retry_types):
        raise ValueError("Invalid retry policy")
    return RunConfig(text(agent.get("model") or resolved.get("model"), "model"), "claude_code",
                     text(harness.get("version"), "harness version"), spec.count, 50, revision,
                     "Prime Inference", {"max_retries": retry_count, "include": retry_types})


def report_scores(record: Object, config: RunConfig, spec: Spec, scorer: ModuleType) -> tuple[Object, Object, list[ParsedFinding], tuple[float, float]]:
    traces = items(record.get("traces"), "traces")
    if len(traces) != 1:
        raise ValueError("Each saved episode must have one trace")
    trace = obj(traces[0], "trace")
    if record.get("ok") is not True or trace.get("ok") is not True or trace.get("is_completed") is not True:
        raise ValueError("A saved episode is not complete")
    agent = obj(obj(trace.get("agent"), "trace.agent").get("config"), "agent config")
    harness = obj(agent.get("harness"), "agent harness")
    same((agent.get("model"), harness.get("id"), harness.get("version"), agent.get("max_turns")),
         (config.model, config.harness, config.harness_version, config.max_turns), "Trace agent")
    data = obj(obj(record.get("task"), "task").get("data"), "task data")
    trace_data = obj(obj(trace.get("task"), "trace.task").get("data"), "trace task data")
    for field in ("name", "boxes", "labels" if spec.mode else "diseases"):
        same(data.get(field), trace_data.get(field), "Trace reference")
    if spec.mode:
        same(data.get("mode"), spec.mode, "Film mode")
    boxes = items(data.get("boxes"), "reference boxes")
    labels = items(data.get("labels" if spec.mode else "diseases"), "reference labels")
    if len(boxes) != len(labels) or any(not isinstance(label, str) for label in labels):
        raise ValueError("Reference box and label counts or types differ")
    for box in boxes:
        if not isinstance(box, list) or len(box) != 4 or any(isinstance(v, bool) or not isinstance(v, (int, float)) for v in box):
            raise ValueError("A reference box must have four finite numbers")
    submitted = obj(trace.get("info", {}), "trace.info").get("submitted")
    if submitted is not None and not isinstance(submitted, str):
        raise ValueError("The saved report must be text or null")
    answer = scorer.parse_answer(submitted, spec.mode) if spec.mode else scorer.parse_findings(submitted)
    reference = [scorer.Finding(tuple(box), label) for box, label in zip(boxes, labels, strict=True)]
    terms = scorer.film_scores(answer, reference, spec.mode) if spec.mode else scorer.film_scores(answer, reference)
    score(trace)  # Validate the saved reward terms before comparison.
    return data, trace, answer, terms


def read_run(path: Path, spec: Spec, scorer: ModuleType, correction: Object | None) -> tuple[Object, list[ScoredFilm], Object, RunConfig]:
    resolved = obj(load(path / "configs/resolved/eval.json"), "resolved config")
    config = run_config(resolved, spec, scorer.REVISION)
    expected_model = "anthropic/claude-opus-5.5" if "opus55" in path.name else "anthropic/claude-haiku-4.5"
    same(config.model, expected_model, "Saved model")
    sidecar_films: dict[str, Object] = {}
    if correction:
        for raw in items(correction.get("films"), "correction films"):
            film = obj(raw, "correction film")
            name = text(film.get("image_id"), "correction film ID")
            if name in sidecar_films:
                raise ValueError("Duplicate film in score correction")
            sidecar_films[name] = film
    rows: list[ScoredFilm] = []
    seen: set[str] = set()
    keys: set[str] = set()
    source_hash = hashlib.sha256()
    with (path / "traces.jsonl").open("rb") as stream:
        line = 0
        while True:
            offset = stream.tell()
            raw = stream.readline()
            if not raw:
                break
            line += 1
            source_hash.update(raw)
            try:
                record = obj(value(json.loads(raw)), "episode")
                data, trace, answer, terms = report_scores(record, config, spec, scorer)
                name = text(data.get("name"), "film ID")
                key = task_identity(record, line)
                if name in seen or key in keys:
                    raise ValueError("Duplicate film or task in a one-attempt run")
                seen.add(name)
                keys.add(key)
                reward = sum(terms) / 2
                original_reward = score(trace)
                submitted = obj(trace.get("info", {}), "trace.info").get("submitted")
                if correction:
                    evidence = sidecar_films.get(name)
                    if evidence is None:
                        raise ValueError("Film is absent from the correction sidecar")
                    for field, actual in (("original_reward", original_reward), ("corrected_reward", reward),
                                          ("box_f1", terms[0]), ("label_f1", terms[1]),
                                          ("parsed_findings", len(answer)), ("report_present", bool(submitted and submitted.strip()))):
                        same(actual, evidence.get(field), f"{name} {field}")
                else:
                    for field, number in zip(("box_f1", "label_f1"), terms, strict=True):
                        same(number, obj(obj(trace["rewards"], "rewards")[field], field).get("score"), f"{name} {field}")
                rows.append({"film": name, "task_key": key, "reward": reward, "offset": offset, "line": line,
                             "terms": terms, "original_reward": original_reward,
                             "model_calls": len(items(trace.get("calls", []), "calls")),
                             "stop_condition": trace.get("stop_condition")})
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Cannot publish {path.name}, line {line}: {error}") from error
    if len(rows) != spec.count:
        raise ValueError(f"{path.name} does not contain the full {spec.count}-film split")
    if correction:
        same(seen, set(sidecar_films), "Correction film set")
        same(source_hash.hexdigest(), correction.get("source_sha256"), "Correction source hash")
    rewards = [row["reward"] for row in rows]
    mean = sum(rewards) / len(rewards)
    if correction:
        same(mean, correction.get("mean_corrected_reward"), "Corrected full-run mean")
    passes = sum(reward == 1 for reward in rewards)
    entry = {"id": path.name, "name": "Claude Opus 5.5" if "opus55" in path.name else "Claude Haiku 4.5",
             "benchmark": spec.id, "public_traces": 0, "model": config.model, "harness": config.harness,
             "harness_version": config.harness_version, "split": "test", "expected": spec.count,
             "attempted": len(rows), "succeeded": len(rows), "errors": 0, "films": len(rows), "status": "complete",
             "mean_reward": mean, "pass_rate": passes / len(rows), "pass_count": passes,
             "updated": int((path / "traces.jsonl").stat().st_mtime), "reason": None,
             "max_turns": config.max_turns, "dataset_revision": config.dataset_revision,
             "attempt_count": len(rows), "extra_attempts": 0,
             "turn_limit_stops": sum(row["stop_condition"] == "max_turns" for row in rows),
             "route": config.route, "retries": config.retries, "selection_policy": "One saved attempt per film",
             "full_test_split": True,
             "mean_reward_ci95": correction["reward_ci95"] if correction else mean_interval(rewards),
             "mean_reward_ci_method": "95% percentile bootstrap; 10000 resamples; seed 20261008" if correction else "95% normal interval over films",
             "pass_rate_ci95": wilson_interval(passes, len(rows)),
             "outcomes": {"pass": passes, "partial": sum(0 < reward < 1 for reward in rewards), "zero": sum(reward == 0 for reward in rewards)},
             "avg_model_calls": sum(row["model_calls"] for row in rows) / len(rows),
             "score_note": DIAGNOSIS_NOTE if correction else WORKLIST_NOTE,
             "score_provenance": {"source_run": path.name, "source_sha256": source_hash.hexdigest(),
                                  "scorer_sha256": digest(Path(scorer.__file__)),
                                  "method": "Production scorer applied to unchanged saved report.json content."}}
    return entry, rows, resolved, config


def report_parsed(submitted: str | None, spec: Spec, predictions: Sequence[ParsedFinding]) -> bool:
    if not submitted or not submitted.strip():
        return False
    if predictions:
        return True
    # A valid empty report is distinct from the parser's empty fallback.
    try:
        report = json.loads(submitted)
    except ValueError:
        return False
    key = "teeth" if spec.mode == "enumeration" else "findings"
    return isinstance(report, dict) and report.get(key) == []


def worklist(spec: Spec, runs: Path, production: Path, sidecar: Object, sidecar_hash: str, output: Path) -> tuple[list[Object], Object]:
    from dentex_teeth.taskset import Finding as DentexFinding
    from pediatric_dental_disease.taskset import Finding as PediatricFinding

    scorer = importlib.import_module(spec.package + ".taskset")
    expected_source = production / "environments" / spec.project / spec.package / "taskset.py"
    same(Path(scorer.__file__).resolve(), expected_source.resolve(), "Production scorer source")
    corrections = {obj(row, "correction run")["run"]: obj(row, "correction run") for row in items(sidecar["runs"], "correction runs")}
    if spec.mode == "diagnosis":
        same(set(corrections), set(spec.run_names), "Correction run set")
        same(sidecar.get("revision"), scorer.REVISION, "Correction dataset revision")
        same(sidecar.get("scorer_environment_version"), "0.3.1", "Correction scorer version")
    scanned = []
    for name in spec.run_names:
        correction = corrections[name] if spec.mode == "diagnosis" else None
        scanned.append(read_run(runs / name, spec, scorer, correction))
    entries = [run[0] for run in scanned]
    winner, rows, resolved, config = max(scanned, key=lambda run: run[0]["mean_reward"])
    if winner["model"] != "anthropic/claude-opus-5.5":
        raise ValueError("The highest-scoring worklist run is not the expected saved Opus run")
    winner["public_traces"] = 5
    for entry in entries:
        if spec.mode == "diagnosis":
            entry["score_provenance"].update(correction_sha256=sidecar_hash,
                                             rollout_environment_version="0.3.0", scorer_environment_version="0.3.1")
    selected = rank_sample(rows)
    directory = output / "data" / winner["id"]
    published = []
    with (runs / winner["id"] / "traces.jsonl").open("rb") as stream:
        for selected_row in selected:
            stream.seek(selected_row["offset"])
            record = obj(value(json.loads(stream.readline())), "selected episode")
            data, trace, answer, terms = report_scores(record, config, spec, scorer)
            same(data.get("name"), selected_row["film"], "Selected film")
            same(sum(terms) / 2, selected_row["reward"], "Selected reward")
            provenance = dict(winner["score_provenance"])
            if spec.mode == "diagnosis":
                provenance["original_rewards"] = trace["rewards"]
                trace["rewards"] = {field: {"score": number, "weight": 0.5}
                                    for field, number in zip(("box_f1", "label_f1"), terms, strict=True)}
            trace["score_provenance"] = provenance
            predictions: list[Json] = []
            for finding in answer:
                if isinstance(finding, DentexFinding):
                    label = finding.label
                elif isinstance(finding, PediatricFinding):
                    label = finding.disease
                else:
                    assert_never(finding)
                predictions.append({"box": list(finding.box), "label": label})
            submitted = obj(trace.get("info", {}), "trace.info").get("submitted")
            parsed = report_parsed(submitted, spec, answer)
            row = film_row(selected_row["task_key"], [(selected_row["line"], record)], config, directory, resolved,
                           output, data["labels" if spec.mode else "diseases"], predictions, parsed,
                           winner["score_note"], provenance)
            if row["status"] != "success":
                raise ValueError("A selected film could not be published: " + "; ".join(row["errors"]))
            same(row["reward"], selected_row["reward"], "Public reward")
            if submitted and not parsed:
                row["warnings"].append("The production parser found no readable report. The saved text and its score are unchanged.")
            published.append(row)
    dump(directory / "evals/test_0000.json", published)
    for entry in entries:
        dump(output / "data" / entry["id"] / "summary.json",
             {**entry, "stage": "benchmark", "eval_views": ["test_0000"] if entry["public_traces"] else []})
    source = ({"name": "DENTEX Challenge 2023", "url": "https://doi.org/10.5281/zenodo.7812323",
               "attribution": "Hamamci et al., DENTEX, 2023; Hamamci et al., MICCAI, 2023. Films were converted to grayscale JPEG; repeated films were removed and film-level splits were added. Display images were reduced to 640 pixels.",
               "license": "CC BY 4.0"} if spec.mode else
              {"name": "Children's Dental Panoramic Radiographs Dataset", "url": "https://doi.org/10.6084/m9.figshare.c.6317013.v1",
               "attribution": "Zhang et al., Scientific Data 10:380, 2023. Films were converted to grayscale JPEG; boxes were converted and disease labels were translated to English. Display images were reduced to 640 pixels.",
               "license": "CC0 1.0"})
    catalog = {"id": spec.id, "name": spec.name, "short_name": spec.short_name, "description": spec.description,
               "task": spec.task, "score": spec.score_text,
               "protocol": "Offline reporting worklist. The scored answer is the saved report.json file, not the final chat reply. One attempt per film, up to 50 agent turns. " + WORKLIST_NOTE,
               "coverage": f"Both saved models cover all {spec.count} test films. These are research results, not evidence of clinical reliability.",
               "test_films": spec.count, "evaluated_films": spec.count, "source": source,
               "selection": selection(winner["name"]), "sample_run": winner["id"],
               "sample_films": [row["film"] for row in published], "score_note": winner["score_note"]}
    return entries, catalog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--periapical-source", type=Path, required=True, help="Original public site or backup root; must contain data/benchmarks.json")
    parser.add_argument("--runs-root", type=Path, required=True, help="Saved worklist benchmark directory")
    parser.add_argument("--production-root", type=Path, required=True, help="Production dental-environments repository")
    parser.add_argument("--diagnosis-correction", type=Path, required=True, help="Saved diagnosis-rescore-0.3.1.json sidecar")
    parser.add_argument("--output", type=Path, required=True, help="New site-output directory; public assets go into its data subdirectory")
    args = parser.parse_args()
    source, runs, production, correction_path, output = (
        path.resolve() for path in (args.periapical_source, args.runs_root, args.production_root, args.diagnosis_correction, args.output))
    for protected in (SITE, source, runs, production, correction_path.parent):
        if output.is_relative_to(protected) or protected.is_relative_to(output):
            raise ValueError("The output must be outside every source tree and the current site")
    if output.exists():
        raise ValueError("The output directory already exists. Use a new directory; no data is deleted.")
    for spec in SPECS:
        package_root = production / "environments" / spec.project
        if str(package_root) not in sys.path:
            sys.path.insert(0, str(package_root))
    sidecar = obj(load(correction_path), "diagnosis correction")
    output.mkdir(parents=True)
    entries, first = periapical(source, output)
    catalog = [first]
    for spec in SPECS:
        additions, item = worklist(spec, runs, production, sidecar, digest(correction_path), output)
        entries.extend(additions)
        catalog.append(item)
    dump(output / "data/benchmarks.json", entries)
    # Write the catalog last. It marks a finished output tree.
    dump(output / "data/catalog.json", catalog)
    print(f"Built {len(entries)} full-run rows and 20 trace examples in {output}")
