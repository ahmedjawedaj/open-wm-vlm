"""Properties that every generated sample must satisfy.

These run on a modest number of samples from each split so the suite stays fast.
Full-size audits are a separate release check, not part of the default CI suite.
"""

from pathlib import Path

import pytest

from wm_vlm.data.builder import build_dataset
from wm_vlm.data.config import DatasetConfig, SplitSpec
from wm_vlm.data.geometry import (
    apply_rotation,
    atomic_rotation,
    compose,
    distinct_poses,
    is_connected,
    mirror_canonical_form,
    rotation_group,
    stabilizer_size,
)
from wm_vlm.data.rendering import image_digest, render_shape
from wm_vlm.data.shapes import build_pools
from wm_vlm.data.tetris import generate_sample, to_voxels

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "dataset"
N_PER_SPLIT = 40


def load(name: str) -> DatasetConfig:
    return DatasetConfig.from_json(CONFIGS / f"{name}.json")


def small(cfg: DatasetConfig, n: int = N_PER_SPLIT) -> DatasetConfig:
    d = cfg.to_dict()
    for s in d["splits"]:
        s["n"] = min(s["n"], n)
    return DatasetConfig.from_dict(d)


@pytest.fixture(scope="module", params=["tetris2d", "tetris3d"])
def built(request):
    cfg = small(load(request.param))
    return cfg, build_dataset(cfg)


def all_samples(built):
    cfg, data = built
    for split, samples in data.items():
        for s in samples:
            yield cfg, split, s


def test_split_sizes(built):
    cfg, data = built
    for spec in cfg.splits:
        assert len(data[spec.name]) == spec.n


def test_answer_is_net_rotation_of_query(built):
    for cfg, _, s in all_samples(built):
        steps = [(x["axis"], x["angle"]) for x in s["steps"]]
        net = compose(steps, cfg.dim)
        expected = apply_rotation(to_voxels(s["query"]), net)
        assert to_voxels(s["options"][s["answer"]]) == expected
        assert to_voxels(s["states"][-1]) == expected


def test_states_follow_each_atomic_action(built):
    for cfg, _, s in all_samples(built):
        assert len(s["states"]) == len(s["steps"]) + 1
        assert len(s["actions"]) == len(s["steps"])
        for k, step in enumerate(s["steps"]):
            m = atomic_rotation(cfg.dim, step["axis"], step["angle"])
            assert to_voxels(s["states"][k + 1]) == apply_rotation(to_voxels(s["states"][k]), m)


def test_reference_rotation_is_uniquely_identifiable(built):
    """Exactly one rotation maps each reference before-image onto its after-image,
    and it is the same one that was applied to the query."""
    for cfg, _, s in all_samples(built):
        steps = [(x["axis"], x["angle"]) for x in s["steps"]]
        net = compose(steps, cfg.dim)
        for ref in s["refs"]:
            before, after = to_voxels(ref["before"]), to_voxels(ref["after"])
            matches = [m for m in rotation_group(cfg.dim) if apply_rotation(before, m) == after]
            assert matches == [net]
            assert stabilizer_size(before) == 1


def test_options_are_distinct_poses_of_the_query_shape(built):
    for cfg, _, s in all_samples(built):
        opts = [to_voxels(o) for o in s["options"]]
        assert len(opts) == cfg.num_options
        assert len(set(opts)) == len(opts)
        poses = set(distinct_poses(to_voxels(s["query"])))
        assert set(opts) <= poses


def test_options_render_to_distinct_images(built):
    for cfg, _, s in list(all_samples(built))[:60]:
        digests = {image_digest(render_shape(to_voxels(o), cfg.canvas)) for o in s["options"]}
        assert len(digests) == cfg.num_options


def test_answer_position_is_balanced(built):
    cfg, data = built
    counts = [0] * cfg.num_options
    for s in data[cfg.splits[0].name]:
        counts[s["answer"]] += 1
    assert min(counts) > 0  # no letter is missing in even a small sample


def test_net_rotation_is_never_identity(built):
    for _cfg, _, s in all_samples(built):
        assert to_voxels(s["states"][0]) != to_voxels(s["states"][-1])


def test_consecutive_3d_actions_use_different_axes(built):
    for cfg, _, s in all_samples(built):
        if cfg.dim == 3:
            axes = [x["axis"] for x in s["steps"]]
            assert all(a != b for a, b in zip(axes, axes[1:], strict=False))


def test_all_shapes_connected(built):
    for _cfg, _, s in all_samples(built):
        for ref in s["refs"]:
            assert is_connected(to_voxels(ref["before"]))
        assert is_connected(to_voxels(s["query"]))


def test_step_counts_respect_split_spec(built):
    cfg, data = built
    for spec in cfg.splits:
        for s in data[spec.name]:
            assert spec.steps[0] <= len(s["steps"]) <= spec.steps[1]


def test_split_shape_membership(built):
    """Train and in-distribution tests draw query shapes from the train pool,
    held-out tests from the held-out pool, OOD tests from unseen cell counts."""
    cfg, data = built
    pools = build_pools(cfg.dim, cfg.seed, cfg.train_counts, cfg.ood_counts, cfg.heldout_fraction)
    from wm_vlm.data.geometry import canonical_form

    for spec in cfg.splits:
        allowed = {c for shapes in pools.get(spec.pool).values() for c in shapes}
        for s in data[spec.name]:
            assert canonical_form(to_voxels(s["query"])) in allowed
            for ref in s["refs"]:
                assert canonical_form(to_voxels(ref["before"])) in allowed


def test_no_mirror_leakage_between_train_and_heldout():
    cfg = small(load("tetris3d"), 60)
    data = build_dataset(cfg)
    from wm_vlm.data.geometry import canonical_form

    train = {mirror_canonical_form(canonical_form(to_voxels(s["query"]))) for s in data["train"]}
    held = {mirror_canonical_form(canonical_form(to_voxels(s["query"]))) for s in data["c_id_test"]}
    assert not train & held


def test_ood_uses_only_unseen_cell_counts(built):
    cfg, data = built
    for s in data["ood_test"]:
        assert len(s["query"]) in cfg.ood_counts
        assert len(s["query"]) not in cfg.train_counts


def test_no_duplicate_questions_across_splits(built):
    from wm_vlm.data.tetris import content_key

    cfg, data = built
    keys = [content_key(s) for samples in data.values() for s in samples]
    assert len(keys) == len(set(keys))


def test_3d_reference_count_differs_from_query_count():
    cfg = small(load("tetris3d"))
    data = build_dataset(cfg)
    different = sum(1 for s in data["train"] if all(len(r["before"]) != len(s["query"]) for r in s["refs"]))
    assert cfg.num_ref_pairs == 1
    assert different == len(data["train"])


def test_unusable_pool_raises():
    cfg = load("tetris2d")
    d = cfg.to_dict()
    d["ood_counts"] = [1]  # a single cell has no asymmetric pose set
    bad = DatasetConfig.from_dict(d)
    with pytest.raises(ValueError):
        generate_sample(bad, SplitSpec("ood_test", "ood", 1, (1, 1)), 0)
