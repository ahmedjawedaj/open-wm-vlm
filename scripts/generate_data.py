"""Generate a full Tetris dataset (symbolic JSONL files plus a hash manifest).

Images are not stored. They are rendered on demand by wm_vlm.data.rendering and
are a function of the symbolic sample and renderer. The manifest hashes the
UTF-8 JSONL bytes, not rendered pixels. It also records the Pillow version.

Example:
    python scripts/generate_data.py --config configs/dataset/tetris2d.json --out data/tetris2d
"""

from __future__ import annotations

import argparse
import json

from wm_vlm.data.builder import write_dataset
from wm_vlm.data.config import DatasetConfig


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--no-extra", action="store_true", help="skip our extra splits (depth generalization)")
    p.add_argument(
        "--workers", type=int, default=1, help="processes for parallel generation, output is identical"
    )
    args = p.parse_args()

    cfg = DatasetConfig.from_json(args.config)
    manifest = write_dataset(cfg, args.out, include_extra=not args.no_extra, workers=args.workers)
    print(json.dumps({k: v for k, v in manifest.items() if k != "config"}, indent=2))


if __name__ == "__main__":
    main()
