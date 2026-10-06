"""Reject invalid user configurations before expensive enumeration or file writes."""

from pathlib import Path

import pytest

from wm_vlm.data.config import DatasetConfig

CONFIG = Path(__file__).resolve().parents[1] / "configs/dataset/tetris3d.json"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "../escape"),
        ("dim", 4),
        ("dim", True),
        ("canvas", 128),
        ("num_options", 9),
        ("num_options", 1),
        ("num_ref_pairs", 0),
        ("heldout_fraction", 1),
        ("heldout_fraction", float("nan")),
        ("seed", 1.5),
        ("train_counts", [4, 4]),
        ("train_counts", [8]),
        ("ood_counts", [4]),
        ("train_counts", []),
    ],
)
def test_invalid_configuration_fails(field, value):
    data = DatasetConfig.from_json(CONFIG).to_dict()
    data[field] = value
    with pytest.raises(ValueError):
        DatasetConfig.from_dict(data)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("name", "../escape"),
        ("n", -1),
        ("n", 1.5),
        ("pool", "unknown"),
        ("steps", [0, 2]),
        ("steps", [3, 2]),
        ("steps", [1]),
        ("extra", "false"),
    ],
)
def test_invalid_split_fails(field, value):
    data = DatasetConfig.from_json(CONFIG).to_dict()
    data["splits"][0][field] = value
    with pytest.raises(ValueError):
        DatasetConfig.from_dict(data)


def test_duplicate_split_name_fails():
    data = DatasetConfig.from_json(CONFIG).to_dict()
    data["splits"][1]["name"] = "train"
    with pytest.raises(ValueError):
        DatasetConfig.from_dict(data)
