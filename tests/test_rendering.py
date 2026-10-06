from wm_vlm.data.geometry import apply_rotation, rotation_group
from wm_vlm.data.rendering import (
    _project,
    image_digest,
    render_2d,
    render_3d,
    render_shape,
)
from wm_vlm.data.shapes import enumerate_classes

CANVAS = 532
MARGIN = 8


def test_canvas_matches_qwen_patch_grid():
    # Qwen2.5-VL uses 28 pixel effective patches: 532 / 28 = 19, so 19 x 19 = 361 tokens.
    assert CANVAS % 28 == 0
    assert (CANVAS // 28) ** 2 == 361


def test_render_size_and_mode():
    for _dim, shape in ((2, enumerate_classes(2, 4)[0]), (3, enumerate_classes(3, 4)[0])):
        img = render_shape(shape, CANVAS)
        assert img.size == (CANVAS, CANVAS)
        assert img.mode == "RGB"


def test_rendering_is_deterministic():
    for dim in (2, 3):
        for shape in enumerate_classes(dim, 5)[:10]:
            assert image_digest(render_shape(shape, CANVAS)) == image_digest(render_shape(shape, CANVAS))


def test_every_3d_shape_fits_the_canvas_in_all_orientations():
    worst_w = worst_h = 0
    for n in range(1, 8):
        for shape in enumerate_classes(3, n):
            for m in rotation_group(3):
                cells = apply_rotation(shape, m)
                pts = [
                    _project(x + a, y + b, z + c)
                    for (x, y, z) in cells
                    for a in (0, 1)
                    for b in (0, 1)
                    for c in (0, 1)
                ]
                w = max(p[0] for p in pts) - min(p[0] for p in pts)
                h = max(p[1] for p in pts) - min(p[1] for p in pts)
                worst_w, worst_h = max(worst_w, w), max(worst_h, h)
    assert worst_w + 2 * MARGIN <= CANVAS
    assert worst_h + 2 * MARGIN <= CANVAS


def test_every_2d_shape_fits_the_board_in_all_orientations():
    for n in range(1, 8):
        for shape in enumerate_classes(2, n):
            for m in rotation_group(2):
                render_2d(apply_rotation(shape, m), CANVAS)  # raises if it does not fit


def test_distinct_poses_render_distinctly_for_2d():
    shape = enumerate_classes(2, 5)[3]
    from wm_vlm.data.geometry import distinct_poses

    poses = distinct_poses(shape)
    assert len({image_digest(render_2d(p, CANVAS)) for p in poses}) == len(poses)


def test_3d_projection_axes_are_right_handed():
    # Seen from the first octant, x, y, z run counterclockwise. In screen coordinates
    # (y down) the cross product of the projected x and y axes should point the same
    # way as the projected z axis is "up", which is the right-handed arrangement.
    ox, oy = _project(0, 0, 0)
    ax = (_project(1, 0, 0)[0] - ox, _project(1, 0, 0)[1] - oy)
    ay = (_project(0, 1, 0)[0] - ox, _project(0, 1, 0)[1] - oy)
    az = (_project(0, 0, 1)[0] - ox, _project(0, 0, 1)[1] - oy)
    assert az[1] < 0  # z points up the screen
    assert ax[0] < 0 < ay[0]  # x goes left, y goes right
    # painter order: the corner nearest the viewer, (1,1,1), projects lower than the origin
    assert _project(1, 1, 0)[1] > oy


def test_drawing_order_does_not_depend_on_input_order():
    cells = ((0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1))
    assert image_digest(render_3d(cells, CANVAS)) == image_digest(render_3d(cells[::-1], CANVAS))


def test_cube_straight_behind_another_is_fully_hidden():
    """In this view the cubes (0,0,0) and (1,1,1) project onto the same hexagon, so the
    nearer one hides the farther one. This is a real ambiguity of single-view rendering.
    The generator guards against it by rejecting samples whose options render identically."""
    assert image_digest(render_3d(((0, 0, 0), (1, 1, 1)), CANVAS)) == image_digest(
        render_3d(((1, 1, 1),), CANVAS)
    )


def test_nearer_cube_hides_the_face_of_the_cube_behind_it():
    both = render_3d(((0, 0, 0), (1, 0, 0)), CANVAS)
    back_only = render_3d(((0, 0, 0),), CANVAS)
    front_only = render_3d(((1, 0, 0),), CANVAS)
    assert image_digest(both) not in {image_digest(back_only), image_digest(front_only)}
