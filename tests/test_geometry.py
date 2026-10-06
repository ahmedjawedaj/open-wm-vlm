import pytest

from wm_vlm.data.geometry import (
    ANGLES,
    AXES_3D,
    apply_rotation,
    atomic_rotation,
    canonical_form,
    compose,
    determinant,
    distinct_poses,
    full_group,
    identity,
    inverse,
    is_connected,
    mirror_canonical_form,
    normalize,
    rotation_group,
    stabilizer_size,
)

L3 = normalize([(0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 1, 0), (2, 1, 1)])
L2 = normalize([(0, 0), (0, 1), (0, 2), (1, 0)])


def test_group_sizes():
    assert len(rotation_group(2)) == 4
    assert len(rotation_group(3)) == 24
    assert len(full_group(2)) == 8
    assert len(full_group(3)) == 48


def test_rotation_group_is_proper_and_closed():
    for dim in (2, 3):
        group = set(rotation_group(dim))
        assert all(determinant(m) == 1 for m in group)
        for a in group:
            for b in group:
                prod = tuple(
                    tuple(sum(a[i][k] * b[k][j] for k in range(dim)) for j in range(dim)) for i in range(dim)
                )
                assert prod in group


def test_quarter_turns_follow_right_hand_rule():
    # +90 degrees about Z sends the x axis to the y axis, about X sends y to z, about Y sends z to x.
    def act(m, v):
        return tuple(sum(m[i][j] * v[j] for j in range(3)) for i in range(3))

    assert act(atomic_rotation(3, "Z", 90), (1, 0, 0)) == (0, 1, 0)
    assert act(atomic_rotation(3, "X", 90), (0, 1, 0)) == (0, 0, 1)
    assert act(atomic_rotation(3, "Y", 90), (0, 0, 1)) == (1, 0, 0)


def test_2d_quarter_turn_is_counterclockwise():
    m = atomic_rotation(2, "Z", 90)
    assert tuple(sum(m[i][j] * v for j, v in enumerate((1, 0))) for i in range(2)) == (0, 1)


@pytest.mark.parametrize("axis", AXES_3D)
def test_rotate_twice_equals_180(axis):
    r90 = atomic_rotation(3, axis, 90)
    r180 = atomic_rotation(3, axis, 180)
    assert apply_rotation(apply_rotation(L3, r90), r90) == apply_rotation(L3, r180)


@pytest.mark.parametrize("axis", AXES_3D)
def test_four_quarter_turns_is_identity(axis):
    out = L3
    for _ in range(4):
        out = apply_rotation(out, atomic_rotation(3, axis, 90))
    assert out == L3


@pytest.mark.parametrize("axis", AXES_3D)
@pytest.mark.parametrize("angle", ANGLES)
def test_inverse_rotation_restores_shape(axis, angle):
    m = atomic_rotation(3, axis, angle)
    assert apply_rotation(apply_rotation(L3, m), inverse(m)) == L3


def test_2d_inverse_and_composition():
    for angle in ANGLES:
        m = atomic_rotation(2, "Z", angle)
        assert apply_rotation(apply_rotation(L2, m), inverse(m)) == L2
    assert compose([("Z", 90), ("Z", 90)], 2) == atomic_rotation(2, "Z", 180)
    assert compose([("Z", 270), ("Z", 90)], 2) == identity(2)


def test_sequential_application_matches_composition():
    steps = [("X", 90), ("Z", 270), ("Y", 180)]
    seq = L3
    for axis, angle in steps:
        seq = apply_rotation(seq, atomic_rotation(3, axis, angle))
    assert seq == apply_rotation(L3, compose(steps, 3))


def test_canonical_form_is_rotation_invariant():
    canon = canonical_form(L3)
    for m in rotation_group(3):
        assert canonical_form(apply_rotation(L3, m)) == canon


def test_mirror_canonical_form_merges_mirror_images():
    mirrored = apply_rotation(L3, ((-1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert mirror_canonical_form(mirrored) == mirror_canonical_form(L3)


def test_chirality_is_preserved_by_rotation_only_canonical_form():
    # The screw tetracube is chiral, its mirror image is a different rotational class.
    screw = normalize([(0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 1)])
    mirror = apply_rotation(screw, ((-1, 0, 0), (0, 1, 0), (0, 0, 1)))
    assert canonical_form(screw) != canonical_form(mirror)
    assert mirror_canonical_form(screw) == mirror_canonical_form(mirror)


def test_stabilizer_and_poses():
    cube = normalize([(0, 0, 0)])
    assert stabilizer_size(cube) == 24
    assert len(distinct_poses(cube)) == 1
    assert stabilizer_size(L3) == 1
    assert len(distinct_poses(L3)) == 24


def test_connectivity():
    assert is_connected(L3)
    assert not is_connected(normalize([(0, 0), (2, 0)]))
