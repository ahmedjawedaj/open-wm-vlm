"""Audit a generated dataset and write a Markdown report. Fails on invalid data."""

from __future__ import annotations

import argparse
from pathlib import Path

from wm_vlm.data.audit import audit_dataset
from wm_vlm.data.config import DatasetConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--skip-regen", action="store_true", help="integrity and membership audit only")
    args = parser.parse_args()
    try:
        result = audit_dataset(
            DatasetConfig.from_json(args.config), args.data, regenerate=not args.skip_regen
        )
    except (OSError, ValueError, KeyError, TypeError, IndexError) as error:
        parser.exit(1, f"Audit failed: {error}\n")
    manifest = result["manifest"]
    lines = [
        f"# Dataset audit: {manifest['config']['name']}",
        "",
        f"Generator version: `{manifest['generator_version']}`",
        f"Dataset hash: `{manifest['dataset_sha256']}`",
        f"Config hash: `{manifest['config_sha256']}`",
        f"Pillow version: `{manifest.get('pillow_version', 'unknown')}`",
        "",
        "Hashes cover UTF-8 symbolic JSONL bytes. Rendered pixels are not hashed.",
        "",
        "| split | pool | samples | cell counts | steps | answer counts | hash ok |",
        "|---|---|---|---|---|---|---|",
    ]
    for split in result["splits"]:
        counts = ", ".join(f"{k}:{v}" for k, v in sorted(split["counts"].items()))
        steps = ", ".join(f"{k}:{v}" for k, v in sorted(split["steps"].items()))
        answers = "/".join(map(str, split["answers"]))
        lines.append(
            f"| {split['name']} | {split['pool']} | {split['n']} | {counts} | {steps} | "
            f"{answers} | {'yes' if split['hash_ok'] else 'NO'} |"
        )
    lines += [
        "",
        f"Duplicate symbolic questions: {result['duplicates']}",
        "Reference and query membership and mirror-separated shape pools checked.",
    ]
    if result["omitted_extra"]:
        lines.append("Extra splits intentionally omitted: " + ", ".join(result["omitted_extra"]))
    if result["regenerated"]:
        lines.append(
            f"Regeneration audit: {result['total'] - result['mismatches']}/{result['total']} samples reproduced exactly"
        )
    else:
        lines.append("Regeneration audit: SKIPPED. This report checks integrity and membership only.")
    lines += ["", "## Result", "", "FAIL: " + ", ".join(result["problems"]) if result["problems"] else "PASS"]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    raise SystemExit(1 if result["problems"] else 0)


if __name__ == "__main__":
    main()
