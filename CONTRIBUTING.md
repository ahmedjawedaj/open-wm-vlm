# Contributing

This is an experimental research project. Contributions should make the current dataset groundwork easier to trust and the proposed experiments easier to run. Browse [open issues](https://github.com/ahmedjawedaj/open-wm-vlm/issues) for scoped tasks.

## Development setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -c constraints-dev.txt -e '.[dev]'
make check
```

Use `make format` to format code. `make build` builds a source distribution and a wheel. The Makefile uses the local `.venv` when available, or override `PY`. CI tests Python 3.10, 3.12 and 3.13 and checks that the installed wheel imports without checkout paths.

## Pull requests

Fork the repository, create a focused branch and submit a PR explaining the problem, resulting behavior and validation. Open an issue first for a substantial architecture or protocol change. Small fixes can go directly to a PR. Keep generated datasets, environments, weights and credentials out of commits.

Add regression coverage for correctness changes. Run `make check` and include the command output summary. If generator behavior changes, bump `GENERATOR_VERSION`, regenerate and fully audit both default datasets, and update the public dataset protocol, dataset card, audit reports and changelog. Dataset hashes are protocol identifiers, not accuracy claims. A change to the pinned Pillow version requires checking rendered output and hashes.

Samples use `DetRng`, sorted iteration and exact integer geometry. Never use unseeded or version-dependent random sampling in generation. The split definitions and sampling assumptions are part of the public protocol.

## Research integrity

- Treat the current datasets as a reconstruction. Record deviations in the public dataset protocol and dataset card.
- Ordinary baseline inputs contain the question and reference/query/option images only. Supervision fields and preview sheets expose the answers or hidden traces.
- Tune on a validation partition of train. Freeze ID/OOD tests before experiments.
- Commit dated hypotheses before their corresponding runs. Distinguish planned experiments from registered or completed experiments.
- Save config, Git revision, generator and dataset hashes, dependency versions, seed and raw predictions with each run.
- Report failures and uncertainty. Do not infer causality merely from a failed intervention ordering or claim equivalence from a nonsignificant difference.
- Update derived tables with scripts once results exist. Planned scripts are not available tools yet.

Contributions are accepted under the repository's MIT license. Credit upstream work and preserve applicable notices. Follow the [code of conduct](CODE_OF_CONDUCT.md) and [security policy](SECURITY.md).

## Maintainer workflow and publication standards

Use focused branches and pull requests targeting `main`. Squash merges keep history readable. CI must pass and review discussions must be resolved before merge. A maintainer reviews outside contributions. Dataset behavior changes must include the generator version and full regeneration evidence.

Run `make public-check` after staging and before committing. It checks the Git index and local documentation links. CI also checks source distributions and wheels. Public docs are explicitly listed by the checker and packaging manifest. When adding a public documentation page, update both lists deliberately.

Keep local notes under `.local/`. Internal plans and implementation handoffs must remain untracked and must not appear in packages, release attachments, issues or pull-request descriptions. Stage specific paths and inspect `git diff --cached --stat` before committing.

Do not publish a package or checkpoint merely because CI passes. Release notes must describe actual implemented behavior, validation, compatibility and licensing. Tag releases only once their supported scope is ready.
