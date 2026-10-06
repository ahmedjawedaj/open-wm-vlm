"""Deterministic rendering of shapes to RGB images.

No text or fonts are drawn, so output bytes do not depend on installed fonts.
The default canvas is 532 pixels, which is 19 x 28. Qwen2.5-VL uses 28 pixel
effective patches, so a 532 pixel image becomes exactly 19 x 19 = 361 visual
tokens, matching the token grid reported in the paper. (A 512 pixel image is
rounded by the Qwen processor to 504 pixels, which gives only 18 x 18 = 324.)
"""

from __future__ import annotations

import hashlib

from PIL import Image, ImageDraw

from wm_vlm.data.geometry import Voxels, bounding_box

BG = (255, 255, 255)
GRID = (214, 214, 214)
EDGE = (30, 30, 30)

# 3D face colours: top, +x face, +y face
TOP = (236, 236, 236)
FACE_X = (170, 190, 220)
FACE_Y = (120, 140, 180)
CELL_2D = (70, 120, 200)

# Fixed integer isometric projection (no floating point).
#
# The unit size is the largest that keeps every shape with up to 7 cubes, in all
# 24 orientations, inside the 532 pixel canvas (worst case 416 x 480 pixels).
# It is fixed rather than fitted per image so that pixel scale is constant
# across states, which matters for learning the dynamics.
ISO_DX = 52  # horizontal screen offset per unit along x or y
ISO_DY = 30  # vertical screen offset per unit along x or y
ISO_DZ = 60  # vertical screen offset per unit along z

BOARD_2D = 7  # the 2D board is BOARD_2D x BOARD_2D cells


def render_shape(cells: Voxels, canvas: int = 532) -> Image.Image:
    dim = len(cells[0])
    return render_2d(cells, canvas) if dim == 2 else render_3d(cells, canvas)


def render_2d(cells: Voxels, canvas: int = 532) -> Image.Image:
    """Square board, y axis points up so that a +90 degree turn is counterclockwise."""
    w, h = bounding_box(cells)
    if w > BOARD_2D or h > BOARD_2D:
        raise ValueError(f"shape {w}x{h} does not fit a {BOARD_2D}x{BOARD_2D} board")
    img = Image.new("RGB", (canvas, canvas), BG)
    draw = ImageDraw.Draw(img)
    cell = canvas // BOARD_2D
    offset = (canvas - cell * BOARD_2D) // 2
    ox = (BOARD_2D - w) // 2
    oy = (BOARD_2D - h) // 2
    for i in range(BOARD_2D + 1):
        p = offset + i * cell
        draw.line([(p, offset), (p, offset + BOARD_2D * cell)], fill=GRID, width=1)
        draw.line([(offset, p), (offset + BOARD_2D * cell, p)], fill=GRID, width=1)
    for x, y in cells:
        gx = ox + x
        gy = BOARD_2D - 1 - (oy + y)  # flip so y points up
        x0 = offset + gx * cell
        y0 = offset + gy * cell
        draw.rectangle([x0, y0, x0 + cell, y0 + cell], fill=CELL_2D, outline=EDGE, width=2)
    return img


def _project(x: int, y: int, z: int) -> tuple[int, int]:
    """Right-handed view from the (+,+,+) octant: x goes lower-left, y lower-right, z up."""
    return ((y - x) * ISO_DX, (x + y) * ISO_DY - z * ISO_DZ)


def render_3d(cells: Voxels, canvas: int = 532) -> Image.Image:
    img = Image.new("RGB", (canvas, canvas), BG)
    draw = ImageDraw.Draw(img)
    # Screen bounds of all cube corners, used to centre the drawing.
    corners = [
        _project(x + dx, y + dy, z + dz)
        for (x, y, z) in cells
        for dx in (0, 1)
        for dy in (0, 1)
        for dz in (0, 1)
    ]
    min_x = min(c[0] for c in corners)
    max_x = max(c[0] for c in corners)
    min_y = min(c[1] for c in corners)
    max_y = max(c[1] for c in corners)
    off_x = canvas // 2 - (min_x + max_x) // 2
    off_y = canvas // 2 - (min_y + max_y) // 2

    def pt(x: int, y: int, z: int) -> tuple[int, int]:
        px, py = _project(x, y, z)
        return (px + off_x, py + off_y)

    # Painter's algorithm: far cubes (small x+y+z) first.
    for x, y, z in sorted(cells, key=lambda c: (c[0] + c[1] + c[2], c[2], c[1], c[0])):
        top = [pt(x, y, z + 1), pt(x + 1, y, z + 1), pt(x + 1, y + 1, z + 1), pt(x, y + 1, z + 1)]
        fx = [pt(x + 1, y, z), pt(x + 1, y + 1, z), pt(x + 1, y + 1, z + 1), pt(x + 1, y, z + 1)]
        fy = [pt(x, y + 1, z), pt(x + 1, y + 1, z), pt(x + 1, y + 1, z + 1), pt(x, y + 1, z + 1)]
        draw.polygon(fy, fill=FACE_Y, outline=EDGE)
        draw.polygon(fx, fill=FACE_X, outline=EDGE)
        draw.polygon(top, fill=TOP, outline=EDGE)
    return img


def image_digest(img: Image.Image) -> str:
    """Content hash of the pixel data, used to reject visually identical options."""
    return hashlib.sha256(img.tobytes()).hexdigest()


def contact_sheet(images: list[Image.Image], columns: int, thumb: int = 200) -> Image.Image:
    """Grid of thumbnails for quick visual inspection."""
    rows = (len(images) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), (245, 245, 245))
    for i, im in enumerate(images):
        t = im.resize((thumb, thumb), Image.LANCZOS)
        sheet.paste(t, ((i % columns) * thumb, (i // columns) * thumb))
    return sheet
