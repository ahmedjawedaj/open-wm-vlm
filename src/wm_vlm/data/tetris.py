"""Sample generation for Tetris-2D and Tetris-3D rotation tasks.

Task (our reconstruction of the paper's protocol, see docs/dataset_protocol.md):

  * One or more reference shapes are shown before and after the same rotation.
  * A new query shape is shown.
  * The model must pick, from four options, the query shape after the same rotation.
  * The rotation is a sequence of atomic mental actions (axis in {X,Y,Z}, angle in
    {90,180,270}, and 2D uses in-plane turns only). After each action there is an
    objectively verifiable ground-truth visual state.

Every sample is a pure function of (config, split, index, variant).
"""

from __future__ import annotations

from functools import cache

from wm_vlm.data.config import DatasetConfig, SplitSpec
from wm_vlm.data.geometry import (
    ANGLES,
    AXES_3D,
    Voxels,
    apply_rotation,
    atomic_rotation,
    compose,
    distinct_poses,
    identity,
    rotation_group,
)
from wm_vlm.data.rendering import image_digest, render_shape
from wm_vlm.data.rng import DetRng
from wm_vlm.data.shapes import ShapePools, asymmetric, build_pools

MAX_ATTEMPTS = 500
OPTION_LABELS = "ABCDEFGH"


def pools_for(cfg: DatasetConfig) -> ShapePools:
    return build_pools(cfg.dim, cfg.seed, cfg.train_counts, cfg.ood_counts, cfg.heldout_fraction)


@cache
def _pool_lists(cfg: DatasetConfig, pool: str):
    """Flat deterministic lists used for sampling inside one pool."""
    by_count = pools_for(cfg).get(pool)
    counts = tuple(sorted(n for n, shapes in by_count.items() if shapes))
    query_ok = {n: tuple(s for s in by_count[n] if _query_ok(s, cfg.dim, cfg.num_options)) for n in counts}
    ref_ok = {n: tuple(s for s in by_count[n] if asymmetric(s)) for n in counts}
    return counts, query_ok, ref_ok


def _query_ok(shape: Voxels, dim: int, num_options: int) -> bool:
    """Query shapes need enough distinct poses to build all options."""
    if dim == 2:
        return asymmetric(shape)
    return len(distinct_poses(shape)) >= num_options


def _random_pose(shape: Voxels, rng: DetRng, dim: int) -> Voxels:
    return apply_rotation(shape, rng.choice(rotation_group(dim)))


def _sample_steps(rng: DetRng, dim: int, lo: int, hi: int) -> list[tuple[str, int]]:
    length = rng.randint(lo, hi)
    steps: list[tuple[str, int]] = []
    for _ in range(length):
        if dim == 2:
            axis = "Z"
        else:
            choices = [a for a in AXES_3D if not steps or a != steps[-1][0]]
            axis = rng.choice(choices)
        steps.append((axis, rng.choice(ANGLES)))
    return steps


def action_text(dim: int, axis: str, angle: int) -> str:
    if dim == 2:
        return f"Rotate the shape {angle} degrees counterclockwise."
    return f"Rotate the shape {angle} degrees about the {axis} axis."


def question_text(cfg: DatasetConfig) -> str:
    n_refs = cfg.num_ref_pairs
    parts = []
    idx = 1
    if n_refs == 1:
        parts.append(
            f"Image {idx} shows a reference shape and image {idx + 1} shows the same shape after a rotation."
        )
        idx += 2
    else:
        parts.append(
            f"Images {idx} to {idx + 2 * n_refs - 1} show {n_refs} reference shapes, each followed by the same "
            "shape after one shared rotation."
        )
        idx += 2 * n_refs
    parts.append(f"Image {idx} shows a new shape.")
    letters = ", ".join(OPTION_LABELS[: cfg.num_options - 1]) + f" or {OPTION_LABELS[cfg.num_options - 1]}"
    parts.append(
        f"Apply the same rotation to the new shape. Which option ({letters}) shows the result? "
        "The options follow the new shape in alphabetical order."
    )
    return " ".join(parts)


def generate_sample(cfg: DatasetConfig, split: SplitSpec, index: int, variant: int = 0) -> dict:
    """Build one sample as a JSON-serialisable dict of plain Python types."""
    dim = cfg.dim
    counts, query_ok, ref_ok = _pool_lists(cfg, split.pool)
    ref_counts = [n for n in counts if ref_ok[n]]
    query_counts = [n for n in counts if query_ok[n]]
    if not query_counts or not ref_counts:
        raise ValueError(f"pool {split.pool!r} has no usable shapes for {cfg.name}")

    rng = DetRng("sample", cfg.name, cfg.seed, split.name, index, variant)
    for _ in range(MAX_ATTEMPTS):
        steps = _sample_steps(rng, dim, split.steps[0], split.steps[1])
        net = compose(steps, dim)
        if net == identity(dim):
            continue

        # Query shape and its pose after each atomic action.
        qn = rng.choice(query_counts)
        q_class = rng.choice(query_ok[qn])
        query = _random_pose(q_class, rng, dim)
        states = [query]
        for axis, angle in steps:
            states.append(apply_rotation(states[-1], atomic_rotation(dim, axis, angle)))
        answer_pose = states[-1]
        if answer_pose == query:
            continue

        # Reference shapes (asymmetric so the shared rotation is uniquely identifiable).
        refs = []
        used_counts: list[int] = []
        for _ in range(cfg.num_ref_pairs):
            preferred = [n for n in ref_counts if n != qn and n not in used_counts]
            preferred = preferred or [n for n in ref_counts if n != qn] or ref_counts
            rn = rng.choice(preferred)
            used_counts.append(rn)
            ref_class = rng.choice(ref_ok[rn])
            before = _random_pose(ref_class, rng, dim)
            after = apply_rotation(before, net)
            refs.append((before, after))

        # Options: the answer plus distinct other poses of the query shape.
        others = [p for p in distinct_poses(query) if p != answer_pose]
        if len(others) < cfg.num_options - 1:
            continue
        rng.shuffle(others)
        options = [answer_pose] + others[: cfg.num_options - 1]
        rng.shuffle(options)
        answer = options.index(answer_pose)

        # Reject samples whose options cannot be told apart in the rendered image.
        digests = {image_digest(render_shape(o, cfg.canvas)) for o in options}
        if len(digests) != len(options):
            continue

        return {
            "id": f"{cfg.name}-{split.name}-{index:06d}",
            "task": cfg.name,
            "split": split.name,
            "index": index,
            "variant": variant,
            "refs": [{"before": _to_list(b), "after": _to_list(a)} for b, a in refs],
            "query": _to_list(query),
            "steps": [{"axis": a, "angle": g} for a, g in steps],
            "states": [_to_list(s) for s in states],
            "options": [_to_list(o) for o in options],
            "answer": answer,
            "answer_label": OPTION_LABELS[answer],
            "question": question_text(cfg),
            "actions": [action_text(dim, a, g) for a, g in steps],
        }
    raise RuntimeError(f"no valid sample for {split.name}[{index}] after {MAX_ATTEMPTS} attempts")


def _to_list(cells: Voxels) -> list[list[int]]:
    return [list(c) for c in cells]


def to_voxels(cells: list[list[int]]) -> Voxels:
    return tuple(tuple(int(v) for v in c) for c in cells)


def content_key(sample: dict) -> str:
    """Underlying symbolic question, independent of labels and hidden traces.

    Two different action sequences can give the same before/after reference pair.
    They must not turn a repeated input into an apparently new test question.
    Options are omitted deliberately so relabelled distractors do not hide repeats.
    This is a symbolic check, not a proof of pixel-level uniqueness under occlusion.
    """
    import hashlib
    import json

    blob = json.dumps(
        {"refs": sample["refs"], "query": sample["query"]},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def state_description(cells: Voxels) -> str:
    """Plain text description of a state, used for the text-only baselines (B3, B4)."""
    coords = " ".join("(" + ",".join(str(v) for v in c) + ")" for c in cells)
    return f"cells: {coords}"
