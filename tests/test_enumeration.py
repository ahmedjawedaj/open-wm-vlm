"""Shape enumeration is checked against published counts.

2D (one-sided polyominoes, OEIS A000988): 1, 1, 2, 7, 18, 60, 196 for n = 1..7
3D (polycubes up to rotation, OEIS A000162): 1, 1, 2, 8, 29, 166, 1023 for n = 1..7
"""

import pytest

from wm_vlm.data.geometry import is_connected, mirror_canonical_form
from wm_vlm.data.shapes import build_pools, class_bucket, enumerate_classes

COUNTS_2D = {1: 1, 2: 1, 3: 2, 4: 7, 5: 18, 6: 60, 7: 196}
COUNTS_3D = {1: 1, 2: 1, 3: 2, 4: 8, 5: 29, 6: 166, 7: 1023}


@pytest.mark.parametrize("n,expected", COUNTS_2D.items())
def test_polyomino_counts(n, expected):
    assert len(enumerate_classes(2, n)) == expected


@pytest.mark.parametrize("n,expected", COUNTS_3D.items())
def test_polycube_counts(n, expected):
    assert len(enumerate_classes(3, n)) == expected


def test_all_classes_connected_and_correct_size():
    for dim in (2, 3):
        for n in range(1, 7):
            for shape in enumerate_classes(dim, n):
                assert len(shape) == n
                assert len(set(shape)) == n
                assert is_connected(shape)


def test_mirror_pairs_share_a_bucket():
    for shape in enumerate_classes(3, 5):
        # every class has the same bucket as any shape with the same mirror form
        assert class_bucket(shape, 1) == class_bucket(mirror_canonical_form(shape), 1)


def test_pools_are_disjoint_and_cover_all_classes():
    pools = build_pools(3, 7, (4, 5, 6), (7,), 0.25)
    for n in (4, 5, 6):
        train, held = set(pools.train[n]), set(pools.heldout[n])
        assert not train & held
        assert train | held == set(enumerate_classes(3, n))
        # no mirror pair straddles the boundary
        train_m = {mirror_canonical_form(s) for s in train}
        held_m = {mirror_canonical_form(s) for s in held}
        assert not train_m & held_m
    assert set(pools.ood[7]) == set(enumerate_classes(3, 7))
    assert 0.15 < len(pools.heldout[6]) / len(enumerate_classes(3, 6)) < 0.35


def test_ood_counts_never_appear_in_train_pool():
    pools = build_pools(2, 7, (4, 5, 6), (7,), 0.0)
    assert 7 not in pools.train and 7 not in pools.heldout
    assert all(len(s) == 7 for s in pools.ood[7])
