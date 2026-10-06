"""Gate G1 acceptance: dataset generation is deterministic and independent of order."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from wm_vlm.data.builder import build_dataset, dumps_jsonl, sha256_text, write_dataset
from wm_vlm.data.config import DatasetConfig
from wm_vlm.data.rng import DetRng
from wm_vlm.data.tetris import generate_sample

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs" / "dataset"


def small(name: str, n: int = 30) -> DatasetConfig:
    d = json.loads((CONFIGS / f"{name}.json").read_text())
    for s in d["splits"]:
        s["n"] = min(s["n"], n)
    return DatasetConfig.from_dict(d)


def test_splitmix64_matches_reference_vector():
    # Published SplitMix64 outputs for a zero state (independent of this repo).
    rng = DetRng("anything")
    rng._state = 0
    assert rng.next_u64() == 0xE220A8397B1DCDAF
    assert rng.next_u64() == 0x6E789E6AA1B965F4
    assert rng.next_u64() == 0x06C45D188009454F


def test_rng_stream_is_pinned():
    # Regression pin. If these change, every generated dataset changes.
    rng = DetRng("golden", 1, 2, 3)
    assert [rng.next_u64() for _ in range(3)] == [
        12593990659594871057,
        4474015985268068389,
        8945998474391780638,
    ]
    rng = DetRng("golden", 1, 2, 3)
    assert [rng.randbelow(1000) for _ in range(5)] == [57, 389, 638, 108, 52]
    items = list(range(8))
    DetRng("shuf", 1).shuffle(items)
    assert items == [0, 2, 5, 4, 1, 6, 3, 7]


def test_rng_labels_matter():
    assert DetRng("x", 1).next_u64() != DetRng("x", 2).next_u64()


@pytest.mark.parametrize("n", [0, -1, 2**64 + 1, True])
def test_randbelow_rejects_out_of_range_bounds(n):
    with pytest.raises(ValueError):
        DetRng("invalid").randbelow(n)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("tetris2d", "d545862086656ec88fec17f4a29f88237547fb2be22975c9f6ee42a79f7ddba5"),
        ("tetris3d", "0830b6387828ba6a9e2587ea37f09f4a968416d31da4ed4c2f3f6e7ea6ea5d48"),
    ],
)
def test_version_2_small_dataset_golden_hash(name, expected):
    combined = hashlib.sha256()
    for samples in build_dataset(small(name, 3)).values():
        combined.update(sha256_text(dumps_jsonl(samples)).encode("ascii"))
    assert combined.hexdigest() == expected


@pytest.mark.parametrize("name", ["tetris2d", "tetris3d"])
def test_two_builds_are_byte_identical(name):
    cfg = small(name)
    a = {k: dumps_jsonl(v) for k, v in build_dataset(cfg).items()}
    b = {k: dumps_jsonl(v) for k, v in build_dataset(cfg).items()}
    assert a == b


@pytest.mark.parametrize("name", ["tetris2d", "tetris3d"])
def test_each_sample_regenerates_in_isolation(name):
    cfg = small(name)
    data = build_dataset(cfg)
    for spec in cfg.splits:
        for i in (0, spec.n // 2, spec.n - 1):
            rec = data[spec.name][i]
            again = generate_sample(cfg, spec, rec["index"], rec["variant"])
            assert again == rec


@pytest.mark.parametrize("name", ["tetris2d", "tetris3d"])
def test_building_in_reverse_order_gives_same_samples(name):
    cfg = small(name)
    spec = cfg.splits[0]
    forward = [generate_sample(cfg, spec, i) for i in range(10)]
    backward = [generate_sample(cfg, spec, i) for i in reversed(range(10))][::-1]
    assert forward == backward


def test_different_seed_changes_the_data():
    cfg = small("tetris3d")
    d = cfg.to_dict()
    d["seed"] += 1
    other = DatasetConfig.from_dict(d)
    spec = cfg.splits[0]
    assert generate_sample(cfg, spec, 0) != generate_sample(other, other.splits[0], 0)


def test_manifest_hashes_match_files(tmp_path):
    cfg = small("tetris2d", 20)
    manifest = write_dataset(cfg, tmp_path)
    for name, info in manifest["splits"].items():
        text = (tmp_path / f"{name}.jsonl").read_text()
        assert sha256_text(text) == info["sha256"]
        assert len(text.splitlines()) == info["n"]
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["dataset_sha256"] == manifest["dataset_sha256"]


def test_hashes_are_stable_across_processes(tmp_path):
    """A fresh interpreter must produce the same dataset hash (guards against hash
    randomisation and dict or set ordering leaking into the output)."""
    cfg_path = tmp_path / "cfg.json"
    d = json.loads((CONFIGS / "tetris3d.json").read_text())
    for s in d["splits"]:
        s["n"] = 12
    cfg_path.write_text(json.dumps(d))
    hashes = []
    for seed in ("1", "2", "random"):
        out = tmp_path / f"out_{seed}"
        env = {"PYTHONHASHSEED": seed, "PYTHONPATH": str(ROOT / "src")}
        res = subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "generate_data.py"),
                "--config",
                str(cfg_path),
                "--out",
                str(out),
            ],
            capture_output=True,
            text=True,
            env=env,
        )
        assert res.returncode == 0, res.stderr
        hashes.append(json.loads((out / "manifest.json").read_text())["dataset_sha256"])
    assert len(set(hashes)) == 1


def test_rendered_pixels_are_stable():
    from wm_vlm.data.rendering import image_digest, render_shape
    from wm_vlm.data.tetris import to_voxels

    cfg = small("tetris3d")
    s = generate_sample(cfg, cfg.splits[0], 3)
    first = [image_digest(render_shape(to_voxels(o), cfg.canvas)) for o in s["options"]]
    second = [image_digest(render_shape(to_voxels(o), cfg.canvas)) for o in s["options"]]
    assert first == second


@pytest.mark.parametrize("name", ["tetris2d", "tetris3d"])
def test_parallel_build_is_identical_to_sequential(name):
    cfg = small(name, 25)
    seq = {k: dumps_jsonl(v) for k, v in build_dataset(cfg).items()}
    par = {k: dumps_jsonl(v) for k, v in build_dataset(cfg, workers=2).items()}
    assert seq == par
