"""Render the page backgrounds, assets/tiedye.webp and assets/tiedye-portrait.webp: a very faint tie-dye on black.

Usage (needs numpy, scipy and Pillow):
    python build_tiedye.py

The pictures are made once, here, so the site ships two small images and no script. A few spiral-folded rosettes
(the classic tie-dye shape) sit near the edges, where the page's content does not cover them: each wedge takes the
next dye colour around the colour wheel, wedges are separated by dark resist lines, and the edges bleed.
Coordinates are bent by smooth noise so the lines wander like hand-dyed cloth. Everything is then dimmed by
BRIGHTNESS, so on the black page it is a hint of colour, not a colour. site.css picks the landscape or the portrait
picture by screen shape. Output is deterministic: the same SEED gives the same files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, NamedTuple

import numpy as np
from PIL import Image
from scipy import ndimage

SITE: Final = Path(__file__).resolve().parent
SEED: Final = 5
BRIGHTNESS: Final = 0.19  # 1.0 is full-strength dye


class Rosette(NamedTuple):
    x: float  # centre, as a fraction of the picture width
    y: float  # centre, as a fraction of the picture height
    radius: float  # pixels; the dye fades out beyond this
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
        Rosette(0.02, 0.38, 480, 8, 1.2, 0.00),
        Rosette(0.98, 0.64, 520, 8, -1.1, 0.40),
        Rosette(0.94, 0.00, 340, 6, 1.0, 0.70),
        Rosette(0.07, 1.00, 380, 6, -1.0, 0.22),
    )),
    Layout("tiedye-portrait.webp", 900, 1600, (
        Rosette(0.00, 0.26, 340, 8, 1.2, 0.00),
        Rosette(1.00, 0.58, 380, 8, -1.1, 0.40),
        Rosette(0.92, 0.01, 260, 6, 1.0, 0.70),
        Rosette(0.06, 1.00, 300, 6, -1.0, 0.22),
    )),
)

# Dye colours evenly around the colour wheel (sRGB): violet, blue, turquoise, green, yellow, orange, red, magenta.
DYES: Final = np.array([
    (122, 62, 230),
    (64, 108, 240),
    (48, 200, 214),
    (84, 214, 120),
    (238, 220, 84),
    (246, 146, 60),
    (238, 78, 108),
    (200, 66, 190),
])


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


def rosette_layer(layout: Layout, columns: np.ndarray, rows: np.ndarray, rosette: Rosette, wobble: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The dye colour and how strongly it is dyed, at every pixel, for one rosette."""
    from_x = columns - rosette.x * layout.width
    from_y = rows - rosette.y * layout.height
    distance = np.hypot(from_x, from_y)
    turn = np.arctan2(from_y, from_x) / (2.0 * np.pi)
    phase = rosette.wedges * turn + rosette.twist * np.log1p(distance / (0.2 * rosette.radius))
    wedge = np.floor(phase)
    across = phase - wedge  # 0 to 1 across one wedge

    # A wedge is one colour that bleeds into the next one near its far edge; a dark line marks the fold.
    colour = dye_colour((wedge + smoothstep(0.62, 1.0, across)) / rosette.wedges + rosette.hue_shift)
    fold = smoothstep(0.0, 0.09, across) * smoothstep(1.0, 0.91, across)
    rings = 0.8 + 0.2 * np.cos(2.0 * np.pi * distance / (0.14 * rosette.radius) + 3.0 * wobble)
    fade = smoothstep(1.0, 0.3, distance / rosette.radius)  # full dye near the middle, none past the radius
    centre = smoothstep(0.0, 0.06 * rosette.radius, distance)  # the undyed spot where the cloth was tied
    return colour, fade * rings * fold * centre


def render(layout: Layout) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    rows, columns = np.mgrid[0 : layout.height, 0 : layout.width].astype(np.float64)
    noise = lambda cell: smooth_noise(rng, cell, layout.width, layout.height)  # noqa: E731
    bent_columns = columns + 26.0 * noise(240) + 9.0 * noise(80)
    bent_rows = rows + 26.0 * noise(240) + 9.0 * noise(80)
    wobble = noise(160)

    layers = [rosette_layer(layout, bent_columns, bent_rows, rosette, wobble) for rosette in layout.rosettes]
    colours = np.stack([colour for colour, _ in layers])
    strengths = np.stack([strength for _, strength in layers])
    winner = strengths.argmax(axis=0)  # where two rosettes meet, the stronger dye shows; hues never muddy together
    dye = np.take_along_axis(colours, winner[None, ..., None], axis=0)[0] * strengths.max(axis=0)[..., None]
    bled = 0.62 * dye + 0.38 * ndimage.gaussian_filter(dye, sigma=(7, 7, 0))
    bled = ndimage.gaussian_filter(bled, sigma=(1.2, 1.2, 0))

    picture = bled * BRIGHTNESS
    picture += rng.normal(0.0, 0.9, picture.shape)  # grain, so dark gradients do not band
    return np.clip(picture, 0, 255).astype(np.uint8)


def main() -> None:
    for layout in LAYOUTS:
        output = SITE / "assets" / layout.filename
        Image.fromarray(render(layout), "RGB").save(output, "WEBP", quality=86, method=6)
        print(f"{output}: {output.stat().st_size / 1024:.0f} KB, {layout.width}x{layout.height}")


if __name__ == "__main__":
    main()
