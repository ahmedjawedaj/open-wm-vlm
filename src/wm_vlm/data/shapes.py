"""Enumeration of shape classes and deterministic assignment to splits."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cache

from wm_vlm.data.geometry import (
    Voxels,
    canonical_form,
    mirror_canonical_form,
    neighbors,
    normalize,
    stabilizer_size,
)


@cache
def enumerate_classes(dim: int, n: int) -> tuple[Voxels, ...]:
    """All connected shapes with n cells, up to rotation (chirality is kept).

    Known counts (OEIS A000988 for 2D, A000162 for 3D):
      2D: 1, 2, 7, 18, 60, 196 for n = 2..7  (n=1 gives 1)
      3D: 1, 2, 8, 29, 166, 1023 for n = 2..7
    """
    if n < 1:
        raise ValueError("n must be at least 1")
    if n == 1:
        return (canonical_form(((0,) * dim,)),)
    found: set[Voxels] = set()
    for shape in enumerate_classes(dim, n - 1):
        cells = set(shape)
        for cell in shape:
            for nb in neighbors(cell, dim):
                if nb not in cells:
                    found.add(canonical_form(normalize(cells | {nb})))
    return tuple(sorted(found))


def class_bucket(shape: Voxels, seed: int) -> float:
    """Stable value in [0, 1) shared by a shape and its mirror image.

    Using the mirror-invariant form keeps mirror pairs in the same split, which
    prevents leakage through symmetry (the paper makes the same choice).
    """
    key = f"{seed}|{mirror_canonical_form(shape)}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") / 2**64


@dataclass(frozen=True)
class ShapePools:
    """Shape classes available to each pool, grouped by cell count."""

    train: dict[int, tuple[Voxels, ...]]
    heldout: dict[int, tuple[Voxels, ...]]
    ood: dict[int, tuple[Voxels, ...]]

    def get(self, pool: str) -> dict[int, tuple[Voxels, ...]]:
        return {"train": self.train, "heldout": self.heldout, "ood": self.ood}[pool]


@cache
def build_pools(
    dim: int,
    seed: int,
    train_counts: tuple[int, ...],
    ood_counts: tuple[int, ...],
    heldout_fraction: float,
) -> ShapePools:
    train: dict[int, tuple[Voxels, ...]] = {}
    heldout: dict[int, tuple[Voxels, ...]] = {}
    ood: dict[int, tuple[Voxels, ...]] = {}
    cut = 1.0 - heldout_fraction
    for n in train_counts:
        classes = enumerate_classes(dim, n)
        train[n] = tuple(c for c in classes if class_bucket(c, seed) < cut)
        heldout[n] = tuple(c for c in classes if class_bucket(c, seed) >= cut)
    for n in ood_counts:
        ood[n] = enumerate_classes(dim, n)
    return ShapePools(train=train, heldout=heldout, ood=ood)


@cache
def asymmetric(shape: Voxels) -> bool:
    """True when no non-trivial rotation maps the shape onto itself."""
    return stabilizer_size(shape) == 1
