"""Render the page backgrounds, assets/tiedye.webp and assets/tiedye-portrait.webp: a faint tie-dye on black.

Usage (needs numpy, scipy and Pillow):
    python build_tiedye.py

The pictures are made once, here, so the site ships two small images and no script. A few spiral-folded rosettes
(the classic tie-dye shape) sit at the sides, where the page's content does not cover them. What makes it read as
tie-dye rather than as coloured rings:
  - each wedge is a bold dye colour, and each colour bleeds into the next one around the colour wheel;
  - the folds are pale, undyed streaks, and the dye thins to a pastel next to them;
  - the dye is mottled where it pooled, and the lines and edges wander and feather like dye wicking through cloth.
Everything is dimmed by BRIGHTNESS, so on the black page it stays a hint of colour. site.css picks the landscape or
the portrait picture by screen shape. Output is deterministic: the same SEED gives the same files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
from PIL import Image
from scipy import ndimage

SITE: Final = Path(__file__).resolve().parent
SEED: Final = 5
BRIGHTNESS: Final = 0.20  # 1.0 is full-strength dye


class Rosette(NamedTuple):
    x: float  # centre, as a fraction of the picture width
    y: float  # centre, as a fraction of the picture height
    radius: float  # pixels; the dye fades out towards this distance
    wedges: int  # folds around the centre
    twist: float  # how far the wedges curl into a spiral, in wedges per doubling of distance from the centre
    hue_shift: float  # where on the colour wheel the first wedge starts, 0 to 1


class Layout(NamedTuple):
    filename: str
    width: int
    height: int
    rosettes: tuple[Rosette, ...]


LAYOUTS: Final = (
    Layout("tiedye.webp", 1600, 900, (
        Rosette(0.00, 0.30, 720, 8, 1.3, 0.00),
        Rosette(1.00, 0.70, 760, 8, -1.2, 0.38),
        Rosette(0.92, -0.04, 480, 6, 1.0, 0.66),
        Rosette(0.10, 1.04, 500, 6, -1.0, 0.20),
    )),
    Layout("tiedye-portrait.webp", 900, 1600, (
        Rosette(0.00, 0.25, 440, 8, 1.3, 0.00),
        Rosette(1.00, 0.60, 470, 8, -1.2, 0.38),
        Rosette(0.95, 0.00, 300, 6, 1.0, 0.66),
        Rosette(0.05, 1.00, 340, 6, -1.0, 0.20),
    )),
)

# Dye colours evenly around the colour wheel (sRGB): hot pink, red, orange, yellow, lime, turquoise, blue, violet, magenta.
DYES: Final = np.array([
    (255, 40, 150),
    (255, 64, 70),
    (255, 122, 16),
    (255, 206, 24),
    (124, 236, 64),
    (24, 222, 200),
    (44, 124, 255),
    (150, 72, 255),
    (226, 60, 232),
])
UNDYED: Final = np.array((255, 244, 250))  # the cloth where the dye did not reach
RAGGED: Final = 0.09  # how far folds wander, in wedges
CONTRAST: Final = 1.7  # above 1 darkens the thin dye and keeps the thick dye bold
MERGE: Final = 6.0  # where rosettes overlap, the stronger one dominates; higher means a narrower blend


class Fields(NamedTuple):
    """Smooth noise shared by every rosette, zero mean and unit spread."""

    wobble: np.ndarray  # slow drift of the scalloped rings
    ragged: np.ndarray  # wander of the fold lines
    extent: np.ndarray  # how far the dye reaches, so its edge is uneven
    mottle: np.ndarray  # dye pooling


def smooth_noise(rng: np.random.Generator, cell: float, width: int, height: int) -> np.ndarray:
    """Cubic-interpolated random grid with about one random value per `cell` pixels, zero mean, unit spread."""
    grid_height, grid_width = int(height / cell) + 4, int(width / cell) + 4
    grid = rng.standard_normal((grid_height, grid_width))
    field = ndimage.zoom(grid, (height / (grid_height - 3), width / (grid_width - 3)), order=3, mode="reflect")
    field = field[:height, :width]
    return (field - field.mean()) / field.std()


def smoothstep(low: float, high: float, values: np.ndarray) -> np.ndarray:
    """0 at `low`, 1 at `high`, smooth between; works when high < low (falling edge)."""
    scaled = np.clip((values - low) / (high - low), 0.0, 1.0)
    return scaled * scaled * (3.0 - 2.0 * scaled)


def dye_colour(position: np.ndarray) -> np.ndarray:
    """Colour at `position` around the colour wheel; 0 and 1 are the same colour."""
    stops = np.linspace(0.0, 1.0, len(DYES) + 1)
    wrapped = np.concatenate([DYES, DYES[:1]])
    around = position % 1.0
    return np.stack([np.interp(around, stops, wrapped[:, channel]) for channel in range(3)], axis=-1)


def rosette_layer(layout: Layout, columns: np.ndarray, rows: np.ndarray, rosette: Rosette, fields: Fields) -> tuple[np.ndarray, np.ndarray]:
    """The colour and how strongly it shows, at every pixel, for one rosette."""
    from_x = columns - rosette.x * layout.width
    from_y = rows - rosette.y * layout.height
    distance = np.hypot(from_x, from_y)
    turn = np.arctan2(from_y, from_x) / (2.0 * np.pi)
    phase = rosette.wedges * turn + rosette.twist * np.log1p(distance / (0.2 * rosette.radius)) + RAGGED * fields.ragged
    wedge = np.floor(phase)
    across = phase - wedge  # 0 to 1 across one wedge
    from_fold = np.minimum(across, 1.0 - across)  # 0 on a fold, 0.5 in the middle of a wedge

    # A wedge is one dye colour that bleeds into the next one near its far edge. Along a fold the cloth is
    # undyed, and the dye thins to a pastel on either side of it.
    dye = dye_colour((wedge + smoothstep(0.55, 1.0, across)) / rosette.wedges + rosette.hue_shift)
    pale = 0.8 * (1.0 - smoothstep(0.015, 0.2, from_fold))
    colour = dye * (1.0 - pale)[..., None] + UNDYED * pale[..., None]

    rings = 0.84 + 0.16 * np.cos(2.0 * np.pi * distance / (0.14 * rosette.radius) + 3.0 * fields.wobble)
    reach = smoothstep(1.0, 0.5, distance * (1.0 + 0.16 * fields.extent) / rosette.radius)  # full dye near the middle
    tied = smoothstep(0.0, 0.05 * rosette.radius, distance)  # the undyed spot where the cloth was tied
    pooling = np.clip(0.9 + 0.1 * fields.mottle, 0.5, 1.15)
    return colour, (reach * rings * pooling * tied) ** CONTRAST


def render(layout: Layout) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    rows, columns = np.mgrid[0 : layout.height, 0 : layout.width].astype(np.float64)

    def noise(cell: float) -> np.ndarray:
        return smooth_noise(rng, cell, layout.width, layout.height)

    bent_columns = columns + 22.0 * noise(240) + 8.0 * noise(70)
    bent_rows = rows + 22.0 * noise(240) + 8.0 * noise(70)
    fields = Fields(wobble=noise(160), ragged=noise(38) + 0.25 * noise(14), extent=noise(300), mottle=noise(30))

    layers = [rosette_layer(layout, bent_columns, bent_rows, rosette, fields) for rosette in layout.rosettes]
    colours = np.stack([colour for colour, _ in layers])
    strengths = np.stack([strength for _, strength in layers])
    weights = strengths**MERGE  # a soft maximum: the strongest rosette sets the colour, with no hard seam where two meet
    colour = (colours * weights[..., None]).sum(axis=0) / (weights.sum(axis=0)[..., None] + 1e-9)
    dye = colour * strengths.max(axis=0)[..., None]
    bled = 0.72 * dye + 0.28 * ndimage.gaussian_filter(dye, sigma=(5, 5, 0))
    bled = ndimage.gaussian_filter(bled, sigma=(0.9, 0.9, 0))

    picture = bled * BRIGHTNESS
    picture += rng.normal(0.0, 0.9, picture.shape[:2])[..., None]  # grain, the same in every channel, so dark gradients do not band
    return np.clip(picture, 0, 255).astype(np.uint8)


def main() -> None:
    for layout in LAYOUTS:
        output = SITE / "assets" / layout.filename
        Image.fromarray(render(layout), "RGB").save(output, "WEBP", quality=86, method=6)
        print(f"{output}: {output.stat().st_size / 1024:.0f} KB, {layout.width}x{layout.height}")


if __name__ == "__main__":
    main()
