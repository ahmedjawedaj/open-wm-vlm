"""Dataset configuration objects, loaded from JSON files in configs/dataset/."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

GENERATOR_VERSION = "2"


def _name(value: str) -> None:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", value):
        raise ValueError("names must start with a lowercase letter and contain only a-z, 0-9, _ or -")


def _integer(value: int, label: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{label} must be an integer >= {minimum}")


@dataclass(frozen=True)
class SplitSpec:
    name: str
    pool: str  # "train", "heldout" or "ood"
    n: int
    steps: tuple[int, int]  # inclusive range of atomic rotations per sample
    extra: bool = False  # extra splits are ours, not part of the paper protocol

    def __post_init__(self) -> None:
        _name(self.name)
        if self.pool not in ("train", "heldout", "ood"):
            raise ValueError(f"unknown pool {self.pool!r}")
        _integer(self.n, "split size")
        if len(self.steps) != 2:
            raise ValueError("steps must be an inclusive [minimum, maximum] pair")
        for step in self.steps:
            _integer(step, "step count", 1)
        if self.steps[0] > self.steps[1]:
            raise ValueError("minimum steps cannot exceed maximum steps")
        if type(self.extra) is not bool:
            raise ValueError("extra must be a boolean")


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dim: int
    seed: int
    canvas: int
    train_counts: tuple[int, ...]
    ood_counts: tuple[int, ...]
    heldout_fraction: float
    num_ref_pairs: int
    num_options: int
    splits: tuple[SplitSpec, ...]

    def __post_init__(self) -> None:
        _name(self.name)
        if type(self.dim) is not int or self.dim not in (2, 3):
            raise ValueError("dim must be 2 or 3")
        _integer(self.seed, "seed")
        _integer(self.canvas, "canvas", 496 if self.dim == 3 else 14)
        _integer(self.num_ref_pairs, "num_ref_pairs", 1)
        _integer(self.num_options, "num_options", 2)
        if self.num_options > (4 if self.dim == 2 else 8):
            raise ValueError("num_options exceeds the supported distinct poses or labels")
        if type(self.heldout_fraction) not in (float, int) or not 0 <= self.heldout_fraction < 1:
            raise ValueError("heldout_fraction must be in [0, 1)")
        for label, counts in (("train_counts", self.train_counts), ("ood_counts", self.ood_counts)):
            if not counts or len(set(counts)) != len(counts):
                raise ValueError(f"{label} must be nonempty with no duplicates")
            for count in counts:
                _integer(count, label, 1)
                if count > 7:
                    raise ValueError("the built-in renderers support at most 7 cells")
        if set(self.train_counts) & set(self.ood_counts):
            raise ValueError("train_counts and ood_counts must be disjoint")
        names = [s.name for s in self.splits]
        if not names or len(set(names)) != len(names):
            raise ValueError("split names must be nonempty and unique")
        if any(s.pool == "heldout" for s in self.splits) and self.heldout_fraction == 0:
            raise ValueError("a heldout split requires a positive heldout_fraction")

    @classmethod
    def from_dict(cls, d: dict) -> DatasetConfig:
        splits = tuple(
            SplitSpec(
                name=s["name"],
                pool=s["pool"],
                n=s["n"],
                steps=tuple(s["steps"]),
                extra=s.get("extra", False),
            )
            for s in d["splits"]
        )
        return cls(
            name=d["name"],
            dim=d["dim"],
            seed=d["seed"],
            canvas=d["canvas"],
            train_counts=tuple(d["train_counts"]),
            ood_counts=tuple(d["ood_counts"]),
            heldout_fraction=d["heldout_fraction"],
            num_ref_pairs=d["num_ref_pairs"],
            num_options=d.get("num_options", 4),
            splits=splits,
        )

    @classmethod
    def from_json(cls, path: str | Path) -> DatasetConfig:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "dim": self.dim,
            "seed": self.seed,
            "canvas": self.canvas,
            "train_counts": list(self.train_counts),
            "ood_counts": list(self.ood_counts),
            "heldout_fraction": self.heldout_fraction,
            "num_ref_pairs": self.num_ref_pairs,
            "num_options": self.num_options,
            "splits": [
                {
                    "name": s.name,
                    "pool": s.pool,
                    "n": s.n,
                    "steps": list(s.steps),
                    "extra": s.extra,
                }
                for s in self.splits
            ],
        }

    def config_hash(self) -> str:
        blob = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode()).hexdigest()

    def split(self, name: str) -> SplitSpec:
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(name)
