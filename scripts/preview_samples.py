"""Write contact sheets for visual verification of generated samples.

Each sheet has four rows: reference pairs (before, after, before, after ...),
the query shape, the ground-truth state after every mental action, and the
four answer options. The correct option is named in the file name.

Example:
    python scripts/preview_samples.py --config configs/dataset/tetris3d.json \
        --split sc_id_test --count 12 --out previews/tetris3d
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from wm_vlm.data.config import GENERATOR_VERSION, DatasetConfig
from wm_vlm.data.tetris import generate_sample
from wm_vlm.data.viz import sample_sheet


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--split", default=None, help="split name, default is the first split")
    p.add_argument("--count", type=int, default=8)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--data", help="use saved records and their deduplication variants instead of first draws")
    args = p.parse_args()

    cfg = DatasetConfig.from_json(args.config)
    split = cfg.split(args.split) if args.split else cfg.splits[0]
    if args.count < 1 or args.start < 0 or args.start + args.count > split.n:
        p.error("preview range must be within the configured split")
    records = None
    if args.data:
        data = Path(args.data)
        manifest = json.loads((data / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest["config_sha256"] != cfg.config_hash()
            or manifest["generator_version"] != GENERATOR_VERSION
        ):
            p.error("saved dataset version or configuration differs from the supplied configuration")
        records = [
            json.loads(line)
            for line in (data / f"{split.name}.jsonl").read_text(encoding="utf-8").splitlines()
        ]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for i in range(args.start, args.start + args.count):
        s = records[i] if records is not None else generate_sample(cfg, split, i)
        path = out / f"{s['id']}_answer-{s['answer_label']}.png"
        sample_sheet(s, cfg.canvas).save(path)
        print(path)


if __name__ == "__main__":
    main()
