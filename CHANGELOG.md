# Changelog

## 0.1.0 (unreleased)

Initial experimental dataset and research groundwork. Training, checkpoints and model evaluation are not implemented.

### Generator version 2

- Corrected Tetris-3D to one reference pair, with a cell count different from the query where possible.
- Deduplicate symbolic questions independently of hidden action traces and shuffled options.
- Validate configuration and refuse overwriting existing dataset files.
- Write explicit UTF-8 bytes and publish the manifest last.
- Record hash scope and Pillow version. Validate manifest configuration, version, per-split and combined hashes, sample order, and reference/query membership.
- Support auditing datasets generated without optional extra splits.

Version 1 uses the previous protocol. Version 2 hashes are intentionally different. Do not compare results across these versions as if they used the same benchmark.

### Repository standards

Public contribution, conduct and security policies, CI, and explicit Git/package publication checks. Local plans and implementation handoffs are excluded from the public repository and distributions.
