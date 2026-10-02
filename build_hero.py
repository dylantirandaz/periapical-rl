"""Build the benchmark page hero: a few annotated radiographs from the published results.

Usage (needs Pillow; the raw traces are not in this repository):
    <env-venv>/python build_hero.py /path/to/astra-codex-200/traces.jsonl

For each film in HERO_FILMS it cuts the full-resolution radiograph out of the raw trace prompt,
saves a 1600 px wide WebP in assets/hero/, and records the dentist boxes and the model's boxes
and reward from the published rows in assets/hero.json. Box coordinates are fractions of the
image size, so they match any scaled copy of the film.
"""

from __future__ import annotations

import base64
import io
import json
import sys
from pathlib import Path
from typing import Final

from PIL import Image

SITE: Final = Path(__file__).resolve().parent
RUN_ID: Final = "benchmark-gpt-6-astra-codex"
MODEL_NAME: Final = "GPT-6 Astra"
HERO_WIDTH: Final = 1600
# Order matters: the page starts with the first film. Two hits, two partial matches, one miss.
HERO_FILMS: Final = ("14711", "16521", "06691", "05491", "07681")


def film_image(trace_line: str) -> tuple[str, Image.Image]:
    record = json.loads(trace_line)
    data = record["task"]["data"]
    url = data["prompt"][0]["content"][1]["image_url"]["url"]
    prefix = "data:image/jpeg;base64,"
    if not url.startswith(prefix):
        raise ValueError(f"Film {data['name']} does not start with a JPEG data URL")
    return data["name"], Image.open(io.BytesIO(base64.b64decode(url.removeprefix(prefix)))).convert("RGB")


def main() -> None:
    traces_path = Path(sys.argv[1])
    rows = {row["film"]: row for row in json.loads((SITE / "data" / RUN_ID / "evals" / "test_0000.json").read_text())}
    missing = [film for film in HERO_FILMS if film not in rows]
    if missing:
        raise ValueError(f"Films not in the published rows: {missing}")
    wanted = set(HERO_FILMS)
    images: dict[str, Image.Image] = {}
    with traces_path.open() as stream:
        for line in stream:
            if not any(f'"name": "{film}"' in line[:400_000] or f'"name":"{film}"' in line[:400_000] for film in wanted - images.keys()):
                continue
            name, image = film_image(line)
            if name in wanted:
                images[name] = image
            if images.keys() == wanted:
                break
    if images.keys() != wanted:
        raise ValueError(f"Radiographs not found in the traces: {sorted(wanted - images.keys())}")

    output = SITE / "assets" / "hero"
    output.mkdir(parents=True, exist_ok=True)
    items = []
    for film in HERO_FILMS:
        image = images[film]
        height = round(image.height * HERO_WIDTH / image.width)
        resized = image.resize((HERO_WIDTH, height), Image.Resampling.LANCZOS)
        target = output / f"{film}.webp"
        resized.save(target, format="WEBP", quality=80, method=6)
        row = rows[film]
        items.append({
            "film": film,
            "image": target.relative_to(SITE).as_posix(),
            "size": [HERO_WIDTH, height],
            "dentists": [{"box": box, "pai": pai} for box, pai in zip(row["boxes"], row["pai"], strict=True)],
            "model": {"name": MODEL_NAME, "reward": row["reward"], "lesions": row["lesions"]},
        })
        print(f"{film}: {image.size} -> {HERO_WIDTH}x{height}, {target.stat().st_size / 1000:.0f} KB, "
              f"{len(row['boxes'])} dentist box(es), {len(row['lesions'])} model box(es), reward {row['reward']:.2f}")
    (SITE / "assets" / "hero.json").write_text(json.dumps(items, separators=(",", ":")))


if __name__ == "__main__":
    main()
