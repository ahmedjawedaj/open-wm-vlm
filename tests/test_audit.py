"""Audits must reject corrupted provenance and handle intentionally omitted extras."""

import copy
import json
from pathlib import Path

import pytest

from wm_vlm.data.audit import audit_dataset
from wm_vlm.data.builder import write_dataset
from wm_vlm.data.config import DatasetConfig
from wm_vlm.data.tetris import content_key, generate_sample
from wm_vlm.data.viz import question_images

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def dataset(tmp_path):
    data = DatasetConfig.from_json(ROOT / "configs/dataset/tetris2d.json").to_dict()
    for spec in data["splits"]:
        spec["n"] = 4
    cfg = DatasetConfig.from_dict(data)
    write_dataset(cfg, tmp_path)
    return cfg, tmp_path


def mutate_manifest(path, field, value):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest))


def test_audit_accepts_valid_dataset(dataset):
    cfg, path = dataset
    report = audit_dataset(cfg, path)
    assert report["problems"] == []
    assert report["total"] == 16
    assert report["regenerated"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("dataset_sha256", "0" * 64, "combined dataset hash"),
        ("config_sha256", "0" * 64, "configuration differs"),
        ("generator_version", "unknown", "generator version"),
        ("hash_scope", "pixels", "hash scope"),
        ("config", {}, "configuration differs"),
    ],
)
def test_audit_rejects_manifest_tampering(dataset, field, value, message):
    cfg, path = dataset
    mutate_manifest(path, field, value)
    assert any(message in problem for problem in audit_dataset(cfg, path)["problems"])


def test_audit_rejects_byte_corruption(dataset):
    cfg, path = dataset
    file = path / "train.jsonl"
    file.write_bytes(file.read_bytes() + b"\n")
    # Empty JSONL records are malformed, even if a text reader normalizes newlines.
    with pytest.raises(ValueError):
        audit_dataset(cfg, path)


def test_audit_rejects_reordered_records(dataset):
    cfg, path = dataset
    file = path / "train.jsonl"
    rows = file.read_text().splitlines()
    file.write_text("\n".join(reversed(rows)) + "\n")
    problems = audit_dataset(cfg, path)["problems"]
    assert any("order mismatch" in p for p in problems)


def test_audit_checks_manifest_row_count(dataset):
    cfg, path = dataset
    manifest = json.loads((path / "manifest.json").read_text())
    manifest["splits"]["train"]["n"] += 1
    mutate_manifest(path, "splits", manifest["splits"])
    assert any("row count" in p for p in audit_dataset(cfg, path)["problems"])


def test_audit_accepts_no_extra_build(dataset, tmp_path):
    cfg, _ = dataset
    path = tmp_path / "no-extra"
    write_dataset(cfg, path, include_extra=False)
    report = audit_dataset(cfg, path)
    assert report["problems"] == []
    assert report["omitted_extra"] == ["depth3_test"]


def test_audit_rejects_stale_extra_file(dataset):
    cfg, path = dataset
    (path / "old_split.jsonl").write_text("{}\n")
    assert any("split list" in p for p in audit_dataset(cfg, path)["problems"])


def test_writer_refuses_to_overwrite_existing_dataset(dataset):
    cfg, path = dataset
    original = (path / "manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        write_dataset(cfg, path)
    assert (path / "manifest.json").read_bytes() == original


def test_hidden_traces_and_option_order_do_not_hide_duplicate_questions(dataset):
    cfg, _ = dataset
    sample = generate_sample(cfg, cfg.splits[0], 0)
    other = copy.deepcopy(sample)
    other["steps"] = [{"axis": "Z", "angle": 90}] * 3
    other["states"] = []
    other["options"].reverse()
    other["answer"] = 3 - sample["answer"]
    assert content_key(sample) == content_key(other)
    other["query"] = [[0, 0]]
    assert content_key(sample) != content_key(other)


def test_inference_images_do_not_read_ground_truth(dataset):
    cfg, _ = dataset
    sample = generate_sample(cfg, cfg.splits[0], 0)
    for key in ("states", "steps", "actions", "answer", "answer_label"):
        sample.pop(key)
    assert len(question_images(sample, cfg.canvas)) == 2 * cfg.num_ref_pairs + 1 + cfg.num_options


def test_source_is_not_gitignored():
    import subprocess

    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "src/wm_vlm/data/tetris.py"], cwd=ROOT, capture_output=True
    )
    assert result.returncode == 1, result.stdout.decode()
