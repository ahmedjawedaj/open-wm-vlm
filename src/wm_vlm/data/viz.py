"""Image views of a sample, shared by preview scripts and training code."""

from __future__ import annotations

from PIL import Image

from wm_vlm.data.rendering import contact_sheet, render_shape
from wm_vlm.data.tetris import to_voxels


def question_images(sample: dict, canvas: int = 532) -> list[Image.Image]:
    """Inference inputs only: reference pair(s), query, then labelled options.

    Never pass ground-truth states or actions to ordinary B0/B1 inference.
    `sample_images` and `sample_sheet` include supervision and are for inspection.
    """
    cells = [r[k] for r in sample["refs"] for k in ("before", "after")]
    cells += [sample["query"], *sample["options"]]
    return [render_shape(to_voxels(c), canvas) for c in cells]


def sample_images(sample: dict, canvas: int = 532) -> dict[str, list[Image.Image]]:
    """All images of one sample, grouped by role. Rendering is deterministic."""
    return {
        "refs": [render_shape(to_voxels(r[k]), canvas) for r in sample["refs"] for k in ("before", "after")],
        "query": [render_shape(to_voxels(sample["query"]), canvas)],
        "states": [render_shape(to_voxels(s), canvas) for s in sample["states"][1:]],
        "options": [render_shape(to_voxels(o), canvas) for o in sample["options"]],
    }


def sample_sheet(sample: dict, canvas: int = 532, thumb: int = 180) -> Image.Image:
    """One row per role: references, query, ground-truth states, options."""
    groups = sample_images(sample, canvas)
    width = max(len(v) for v in groups.values())
    rows = []
    for key in ("refs", "query", "states", "options"):
        row = contact_sheet(groups[key], columns=width, thumb=thumb)
        # contact_sheet sizes by row count, so pad to a fixed width for stacking
        padded = Image.new("RGB", (width * thumb, thumb), (245, 245, 245))
        padded.paste(row, (0, 0))
        rows.append(padded)
    sheet = Image.new("RGB", (width * thumb, thumb * len(rows) + 6 * (len(rows) - 1)), (120, 120, 120))
    for i, r in enumerate(rows):
        sheet.paste(r, (0, i * (thumb + 6)))
    return sheet
