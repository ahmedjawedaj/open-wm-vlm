"""Integer geometry for polyominoes (2D) and polycubes (3D).

A shape is a sorted tuple of integer coordinate tuples, translated so that the
minimum coordinate along every axis is zero. All operations are exact integer
arithmetic, so every result is reproducible bit for bit.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import cache
from itertools import product

Voxels = tuple[tuple[int, ...], ...]
Matrix = tuple[tuple[int, ...], ...]

AXES_3D = ("X", "Y", "Z")
ANGLES = (90, 180, 270)


def normalize(cells: Iterable[Sequence[int]]) -> Voxels:
    """Translate to the origin and sort, giving a unique representation."""
    pts = [tuple(int(v) for v in c) for c in cells]
    dim = len(pts[0])
    mins = [min(p[i] for p in pts) for i in range(dim)]
    return tuple(sorted(tuple(p[i] - mins[i] for i in range(dim)) for p in pts))


def _matmul(a: Matrix, b: Matrix) -> Matrix:
    n = len(a)
    return tuple(tuple(sum(a[i][k] * b[k][j] for k in range(n)) for j in range(n)) for i in range(n))


def identity(dim: int) -> Matrix:
    return tuple(tuple(1 if i == j else 0 for j in range(dim)) for i in range(dim))


def determinant(m: Matrix) -> int:
    n = len(m)
    if n == 2:
        return m[0][0] * m[1][1] - m[0][1] * m[1][0]
    return (
        m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    )


def axis_rotation_90(axis: str) -> Matrix:
    """Counter-clockwise quarter turn about X, Y or Z (right-hand rule).

    For 2D, the only valid axis is "Z" (in-plane rotation).
    """
    if axis == "X":
        return ((1, 0, 0), (0, 0, -1), (0, 1, 0))
    if axis == "Y":
        return ((0, 0, 1), (0, 1, 0), (-1, 0, 0))
    if axis == "Z":
        return ((0, -1, 0), (1, 0, 0), (0, 0, 1))
    raise ValueError(f"unknown axis {axis!r}")


def rotation_2d_90() -> Matrix:
    return ((0, -1), (1, 0))


def atomic_rotation(dim: int, axis: str, angle: int) -> Matrix:
    """Matrix for one atomic mental action: axis in {X,Y,Z}, angle in {90,180,270}."""
    if angle not in ANGLES:
        raise ValueError(f"angle must be one of {ANGLES}, got {angle}")
    quarter = rotation_2d_90() if dim == 2 else axis_rotation_90(axis)
    if dim == 2 and axis != "Z":
        raise ValueError("2D rotations are in-plane, axis must be 'Z'")
    out = identity(dim)
    for _ in range(angle // 90):
        out = _matmul(quarter, out)
    return out


@cache
def rotation_group(dim: int) -> tuple[Matrix, ...]:
    """All proper rotations that map the integer lattice to itself.

    2D gives the cyclic group C4 (4 elements), 3D gives the chiral octahedral
    group (24 elements). Order is deterministic.
    """
    if dim == 2:
        gens = [rotation_2d_90()]
    else:
        gens = [axis_rotation_90("X"), axis_rotation_90("Y"), axis_rotation_90("Z")]
    seen = {identity(dim)}
    frontier = [identity(dim)]
    while frontier:
        nxt = []
        for m in frontier:
            for g in gens:
                p = _matmul(g, m)
                if p not in seen:
                    seen.add(p)
                    nxt.append(p)
        frontier = nxt
    return tuple(sorted(seen))


@cache
def full_group(dim: int) -> tuple[Matrix, ...]:
    """Rotations plus reflections (signed permutation matrices)."""
    mats = []
    for perm in _permutations(dim):
        for signs in product((1, -1), repeat=dim):
            m = tuple(tuple(signs[i] if perm[i] == j else 0 for j in range(dim)) for i in range(dim))
            mats.append(m)
    return tuple(sorted(mats))


def _permutations(n: int) -> list[tuple[int, ...]]:
    from itertools import permutations

    return list(permutations(range(n)))


def apply_rotation(cells: Voxels, m: Matrix) -> Voxels:
    """Apply a matrix to every cell, then renormalize the translation."""
    dim = len(m)
    rotated = [tuple(sum(m[i][j] * c[j] for j in range(dim)) for i in range(dim)) for c in cells]
    return normalize(rotated)


def canonical_form(cells: Voxels) -> Voxels:
    """Smallest representation over the rotation group (chirality is preserved)."""
    dim = len(cells[0])
    return min(apply_rotation(cells, m) for m in rotation_group(dim))


def mirror_canonical_form(cells: Voxels) -> Voxels:
    """Smallest representation over rotations and reflections."""
    dim = len(cells[0])
    return min(apply_rotation(cells, m) for m in full_group(dim))


def distinct_poses(cells: Voxels) -> tuple[Voxels, ...]:
    """Distinct rotated copies of the shape, ordered deterministically."""
    dim = len(cells[0])
    return tuple(sorted({apply_rotation(cells, m) for m in rotation_group(dim)}))


def stabilizer_size(cells: Voxels) -> int:
    """Number of rotations that map the shape onto itself."""
    dim = len(cells[0])
    base = normalize(cells)
    return sum(1 for m in rotation_group(dim) if apply_rotation(base, m) == base)


def is_asymmetric(cells: Voxels) -> bool:
    """True when every rotation yields a different pose."""
    return stabilizer_size(cells) == 1


def bounding_box(cells: Voxels) -> tuple[int, ...]:
    dim = len(cells[0])
    return tuple(max(c[i] for c in cells) + 1 for i in range(dim))


def is_connected(cells: Voxels) -> bool:
    """Face connectivity (4-neighborhood in 2D, 6-neighborhood in 3D)."""
    cellset = set(cells)
    dim = len(cells[0])
    start = next(iter(cellset))
    seen = {start}
    stack = [start]
    while stack:
        cur = stack.pop()
        for nb in neighbors(cur, dim):
            if nb in cellset and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return len(seen) == len(cellset)


def neighbors(cell: Sequence[int], dim: int) -> list[tuple[int, ...]]:
    out = []
    for axis in range(dim):
        for delta in (-1, 1):
            n = list(cell)
            n[axis] += delta
            out.append(tuple(n))
    return out


def compose(steps: Sequence[tuple[str, int]], dim: int) -> Matrix:
    """Net rotation of a sequence of atomic actions applied in order."""
    net = identity(dim)
    for axis, angle in steps:
        net = _matmul(atomic_rotation(dim, axis, angle), net)
    return net


def inverse(m: Matrix) -> Matrix:
    """Inverse of a rotation matrix is its transpose."""
    n = len(m)
    return tuple(tuple(m[j][i] for j in range(n)) for i in range(n))
