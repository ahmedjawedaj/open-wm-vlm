"""Local research notes must not enter commits or source/wheel distributions."""

import importlib.util
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location(
    "publication", Path(__file__).resolve().parents[1] / "scripts/check_publication.py"
)
assert SPEC is not None and SPEC.loader is not None
publication = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publication)


@pytest.mark.parametrize(
    "path",
    [
        "CLAUDE.md",
        "NOVELTY.md",
        "PAPER_CLAIMS.md",
        "docs/extended_plan.md",
        "docs/release_review.md",
        "docs/new-internal-note.md",
        ".local/claude_implementation_guide.md",
        "scripts/private/notes.py",
        "data/tetris2d/train.jsonl",
        "src/wm_vlm/__pycache__/config.pyc",
    ],
)
def test_private_paths_rejected(path):
    assert not publication.allowed(path)
    assert not publication.allowed(path, distribution=True)


def test_link_to_ignored_working_file_is_rejected():
    files = {"README.md": b"[notes](docs/extended_plan.md)", "docs/dataset_card.md": b"Public docs"}
    assert publication.check_files(files) == ["Unpublished local link: README.md -> docs/extended_plan.md"]


def test_public_document_links_allowed():
    files = {"README.md": b"[data](docs/dataset_card.md)", "docs/dataset_card.md": b"[home](../README.md)"}
    assert publication.check_files(files) == []


def test_public_docs_are_listed_in_the_package_manifest():
    manifest = (Path(__file__).resolve().parents[1] / "MANIFEST.in").read_text()
    missing = [doc for doc in sorted(publication.PUBLIC_DOCS) if doc not in manifest]
    assert missing == []
