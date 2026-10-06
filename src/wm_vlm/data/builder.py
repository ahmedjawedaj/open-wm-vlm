"""Build, hash and write complete datasets."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import PIL

from wm_vlm.data.config import GENERATOR_VERSION, DatasetConfig, SplitSpec
from wm_vlm.data.tetris import content_key, generate_sample

MAX_VARIANTS = 100


def _first_draw(args: tuple[DatasetConfig, SplitSpec, int]) -> dict:
    cfg, split, index = args
    return generate_sample(cfg, split, index, 0)


def build_split(cfg: DatasetConfig, split: SplitSpec, seen: set[str], workers: int = 1) -> list[dict]:
    """Generate one split. `seen` carries content keys across splits so that no
    question appears twice anywhere in the dataset.

    With workers > 1 the first draw of every sample is computed in parallel. The
    result is identical to the sequential build because the duplicate check below
    always runs in index order.
    """
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            first = list(pool.map(_first_draw, ((cfg, split, i) for i in range(split.n)), chunksize=64))
    else:
        first = None
    out: list[dict] = []
    for i in range(split.n):
        for variant in range(MAX_VARIANTS):
            if variant == 0 and first is not None:
                sample = first[i]
            else:
                sample = generate_sample(cfg, split, i, variant)
            key = content_key(sample)
            if key not in seen:
                seen.add(key)
                out.append(sample)
                break
        else:
            raise RuntimeError(f"could not find a unique sample for {split.name}[{i}]")
    return out


def build_dataset(cfg: DatasetConfig, include_extra: bool = True, workers: int = 1) -> dict[str, list[dict]]:
    if type(workers) is not int or workers < 1:
        raise ValueError("workers must be a positive integer")
    seen: set[str] = set()
    result: dict[str, list[dict]] = {}
    for split in cfg.splits:
        if split.extra and not include_extra:
            continue
        result[split.name] = build_split(cfg, split, seen, workers)
    return result


def dumps_jsonl(samples: list[dict]) -> str:
    return "".join(json.dumps(s, sort_keys=True, separators=(",", ":")) + "\n" for s in samples)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def write_dataset(
    cfg: DatasetConfig, out_dir: str | Path, include_extra: bool = True, workers: int = 1
) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if (out / "manifest.json").exists() or any(out.glob("*.jsonl")):
        raise FileExistsError(f"{out} already contains dataset files. Generate into a new directory.")
    data = build_dataset(cfg, include_extra=include_extra, workers=workers)
    manifest = {
        "generator_version": GENERATOR_VERSION,
        "config": cfg.to_dict(),
        "config_sha256": cfg.config_hash(),
        "hash_scope": "symbolic-jsonl-utf8",
        "pillow_version": PIL.__version__,
        "splits": {},
    }
    combined = hashlib.sha256()
    # Stage all files, then publish the manifest last as the completion marker.
    with TemporaryDirectory(prefix=".building-", dir=out) as staging:
        stage = Path(staging)
        for name, samples in data.items():
            payload = dumps_jsonl(samples).encode("utf-8")
            (stage / f"{name}.jsonl").write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            combined.update(digest.encode("ascii"))
            manifest["splits"][name] = {"n": len(samples), "sha256": digest}
        manifest["dataset_sha256"] = combined.hexdigest()
        (stage / "manifest.json").write_bytes(
            (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
        )
        for name in data:
            (stage / f"{name}.jsonl").replace(out / f"{name}.jsonl")
        (stage / "manifest.json").replace(out / "manifest.json")
    return manifest
