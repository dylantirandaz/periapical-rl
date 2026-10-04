"""Render the page background, assets/smoke.webp: slow smoke in purple, blue, pink and orange.

Usage (needs numpy, scipy and Pillow):
    python build_smoke.py

The picture is made once, here, so the site ships one small image and no script. Two smooth noise fields,
bent by two more (domain warping, which gives the swirls), pick a colour from PALETTE and how much of it to
show. The result stays dark so light text on top stays readable; site.css drifts the image slowly. Output is
deterministic: the same SEED gives the same file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import numpy as np
from PIL import Image
from scipy import ndimage

SITE: Final = Path(__file__).resolve().parent
WIDTH: Final = 1600
HEIGHT: Final = 900
SEED: Final = 11

# Colour along the smoke, 0 to 1: sky blue, blue, indigo, violet, purple, pink, coral, orange (sRGB).
PALETTE: Final = np.array([
    (0.00, 84, 156, 228),
    (0.12, 54, 96, 208),
    (0.28, 86, 62, 214),
    (0.44, 134, 66, 222),
    (0.58, 180, 74, 208),
    (0.72, 226, 86, 172),
    (0.86, 242, 112, 124),
    (1.00, 248, 156, 100),
])
BACKGROUND: Final = np.array((9.0, 8.0, 14.0))
SMOKE_GAIN: Final = 1.12


def smooth_noise(rng: np.random.Generator, cell: float) -> np.ndarray:
    """Cubic-interpolated random grid with about one random value per `cell` pixels, zero mean, unit spread."""
    grid_height, grid_width = int(HEIGHT / cell) + 4, int(WIDTH / cell) + 4
    grid = rng.standard_normal((grid_height, grid_width))
    field = ndimage.zoom(grid, (HEIGHT / (grid_height - 3), WIDTH / (grid_width - 3)), order=3, mode="reflect")
    field = field[:HEIGHT, :WIDTH]
    return (field - field.mean()) / field.std()


def fractal(rng: np.random.Generator, largest_cell: float, octaves: int) -> np.ndarray:
    """Smooth noise at several scales, big shapes strongest, scaled to 0 to 1."""
    total = np.zeros((HEIGHT, WIDTH))
    for octave in range(octaves):
        total += smooth_noise(rng, largest_cell / 2**octave) * 0.55**octave
    total = (total - total.min()) / (total.max() - total.min())
    return total


def warp(field: np.ndarray, rng: np.random.Generator, strength: float) -> np.ndarray:
    """Sample `field` at positions pushed around by two smooth displacement fields."""
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    push_rows = (fractal(rng, 420, 3) - 0.5) * 2 * strength
    push_columns = (fractal(rng, 420, 3) - 0.5) * 2 * strength
    return ndimage.map_coordinates(field, [rows + push_rows, columns + push_columns], order=1, mode="reflect")


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def palette_colour(position: np.ndarray) -> np.ndarray:
    return np.stack([np.interp(position, PALETTE[:, 0], PALETTE[:, channel]) for channel in (1, 2, 3)], axis=-1)


def render() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    base = warp(warp(fractal(rng, 520, 3), rng, 260), rng, 140)    # which region is bluer or more purple
    body = warp(warp(fractal(rng, 300, 5), rng, 220), rng, 110)    # how thick the smoke is
    veins = warp(fractal(rng, 140, 4), rng, 90)                    # thin wisps inside the smoke

    wisps = 1.0 - np.abs(2.0 * veins - 1.0)
    thickness = smoothstep(0.30, 0.90, body) ** 1.15 * (0.5 + 0.5 * wisps**2)
    cool = smoothstep(0.10, 0.90, base)
    # Only the thickest smoke turns pink and orange, so warm colours glow instead of going brown.
    core = smoothstep(0.50, 0.95, thickness)
    position = 0.05 + 0.55 * cool + 0.40 * core

    smoke = palette_colour(position) * (thickness * SMOKE_GAIN)[..., None]
    glow = palette_colour(0.05 + 0.55 * cool) * (0.05 + 0.16 * fractal(rng, 700, 2))[..., None]

    # Weaker toward the middle, where the page's text sits.
    rows, columns = np.mgrid[0:HEIGHT, 0:WIDTH].astype(np.float64)
    from_middle = np.hypot((columns / WIDTH - 0.5) * 1.15, (rows / HEIGHT - 0.5) * 1.0)
    edge_weight = 0.55 + 0.45 * smoothstep(0.1, 0.62, from_middle)

    picture = BACKGROUND + (smoke + glow) * edge_weight[..., None]
    picture += rng.normal(0.0, 1.1, picture.shape)  # grain, so dark gradients do not band
    return np.clip(picture, 0, 255).astype(np.uint8)


def main() -> None:
    output = SITE / "assets" / "smoke.webp"
    Image.fromarray(render(), "RGB").save(output, "WEBP", quality=74, method=6)
    print(f"{output}: {output.stat().st_size / 1024:.0f} KB, {WIDTH}x{HEIGHT}")


if __name__ == "__main__":
    main()
