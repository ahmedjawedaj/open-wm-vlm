"""Validate dataset bytes, provenance, split membership and regeneration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from wm_vlm.data.config import GENERATOR_VERSION, DatasetConfig
from wm_vlm.data.geometry import canonical_form, mirror_canonical_form
from wm_vlm.data.tetris import content_key, generate_sample, pools_for, to_voxels


def audit_dataset(cfg: DatasetConfig, directory: str | Path, *, regenerate: bool = True) -> dict:
    """Audit bytes and samples. Optional extra splits may be omitted.

    Malformed inputs raise ValueError, JSON, or I/O errors. A fast audit does
    not claim regeneration. Hashes cover symbolic JSONL, not rendered pixels.
    """
    root = Path(directory)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    problems: list[str] = []
    if manifest.get("generator_version") != GENERATOR_VERSION:
        problems.append("generator version differs from installed generator")
    if manifest.get("config") != cfg.to_dict() or manifest.get("config_sha256") != cfg.config_hash():
        problems.append("manifest configuration differs from supplied configuration")
    if manifest.get("hash_scope") != "symbolic-jsonl-utf8":
        problems.append("unknown or missing hash scope")
    declared = manifest.get("splits", {})
    expected = {s.name for s in cfg.splits}
    required = {s.name for s in cfg.splits if not s.extra}
    if not required <= set(declared) or not set(declared) <= expected:
        problems.append("manifest has missing required or unknown splits")
    if {p.stem for p in root.glob("*.jsonl")} != set(declared):
        problems.append("JSONL files do not match manifest split list")
    combined = hashlib.sha256()
    seen: set[str] = set()
    duplicates = mismatches = total = 0
    summaries = []
    pools = pools_for(cfg)
    for spec in cfg.splits:
        if spec.name not in declared:
            continue
        path = root / f"{spec.name}.jsonl"
        if not path.is_file():
            problems.append(f"{spec.name}: missing JSONL file")
            continue
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        combined.update(digest.encode("ascii"))
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        info = declared[spec.name]
        hash_ok = digest == info.get("sha256")
        if not hash_ok:
            problems.append(f"{spec.name}: file hash differs from manifest")
        if len(rows) != spec.n or info.get("n") != spec.n:
            problems.append(f"{spec.name}: row count differs from config or manifest")
        allowed = {c for group in pools.get(spec.pool).values() for c in group}
        answers = [0] * cfg.num_options
        counts: dict[int, int] = {}
        lengths: dict[int, int] = {}
        for index, rec in enumerate(rows):
            total += 1
            if (
                rec.get("index") != index
                or rec.get("split") != spec.name
                or rec.get("task") != cfg.name
                or rec.get("id") != f"{cfg.name}-{spec.name}-{index:06d}"
            ):
                problems.append(f"{spec.name}[{index}]: identity or order mismatch")
            variant = rec.get("variant")
            if type(variant) is not int or variant < 0:
                problems.append(f"{spec.name}[{index}]: invalid variant")
                continue
            key = content_key(rec)
            duplicates += key in seen
            seen.add(key)
            shape_roles = [rec["query"]]
            shape_roles += [r[k] for r in rec["refs"] for k in ("before", "after")]
            if any(canonical_form(to_voxels(c)) not in allowed for c in shape_roles):
                problems.append(f"{spec.name}[{index}]: reference or query outside shape pool")
            answer = rec.get("answer")
            if type(answer) is not int or not 0 <= answer < cfg.num_options:
                problems.append(f"{spec.name}[{index}]: invalid answer")
            else:
                answers[answer] += 1
            count, length = len(rec["query"]), len(rec["steps"])
            counts[count] = counts.get(count, 0) + 1
            lengths[length] = lengths.get(length, 0) + 1
            if regenerate and generate_sample(cfg, spec, index, variant) != rec:
                mismatches += 1
        summaries.append(
            {
                "name": spec.name,
                "pool": spec.pool,
                "n": len(rows),
                "hash_ok": hash_ok,
                "counts": counts,
                "steps": lengths,
                "answers": answers,
            }
        )
    if combined.hexdigest() != manifest.get("dataset_sha256"):
        problems.append("combined dataset hash differs from actual file hashes")
    if duplicates:
        problems.append(f"{duplicates} duplicate symbolic questions")
    if mismatches:
        problems.append(f"{mismatches} samples did not regenerate")
    pool_classes = {
        name: {mirror_canonical_form(c) for group in pools.get(name).values() for c in group}
        for name in ("train", "heldout", "ood")
    }
    for left, right in (("train", "heldout"), ("train", "ood"), ("heldout", "ood")):
        if pool_classes[left] & pool_classes[right]:
            problems.append(f"shape pool leakage: {left}/{right}")
    return {
        "manifest": manifest,
        "splits": summaries,
        "problems": problems,
        "duplicates": duplicates,
        "total": total,
        "mismatches": mismatches,
        "regenerated": regenerate,
        "omitted_extra": [s.name for s in cfg.splits if s.extra and s.name not in declared],
    }
