"""Fail if a Git tree or distribution includes local/private working documents.

Run before committing and publishing. Public documentation uses an explicit
allowlist so a newly added internal note cannot be published accidentally.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT_FILES = {
    ".gitignore",
    "CHANGELOG.md",
    "CITATION.cff",
    "CODE_OF_CONDUCT.md",
    "CONTRIBUTING.md",
    "LICENSE",
    "MANIFEST.in",
    "Makefile",
    "README.md",
    "SECURITY.md",
    "constraints-dev.txt",
    "pyproject.toml",
}
PUBLIC_DOCS = {
    "docs/answer_parsing.md",
    "docs/dataset_card.md",
    "docs/dataset_protocol.md",
    "docs/dataset_stats_tetris2d.md",
    "docs/dataset_stats_tetris3d.md",
}
PRIVATE_PARTS = {".local", "internal", "private", "planning", "feedback", "claude.md"}


def allowed(path: str, *, distribution: bool = False) -> bool:
    parts = PurePosixPath(path).parts
    if not parts or ".." in parts or any(p.lower() in PRIVATE_PARTS for p in parts):
        return False
    if path in ROOT_FILES or path in PUBLIC_DOCS:
        return True
    suffix = PurePosixPath(path).suffix
    if path.startswith("docs/assets/") and suffix == ".png":
        return True
    if parts[0] in {"src", "tests", "scripts"} and suffix == ".py":
        return True
    if path.startswith("configs/dataset/") and suffix == ".json":
        return True
    if path.startswith(".github/") and suffix in {".yml", ".yaml", ".md"}:
        return True
    if path == ".github/CODEOWNERS":
        return True
    if distribution:
        if path in {"PKG-INFO", "setup.cfg"}:
            return True
        if path.startswith("src/open_wm_vlm.egg-info/") and (suffix == ".txt" or path.endswith("/PKG-INFO")):
            return True
        if parts[0] == "wm_vlm" and suffix == ".py":
            return True
        if parts[0].endswith(".dist-info"):
            return path.endswith(("/METADATA", "/WHEEL", "/RECORD", "/top_level.txt", "/licenses/LICENSE"))
    return False


def git_files(root: Path) -> dict[str, bytes]:
    names = subprocess.check_output(["git", "ls-files", "-z"], cwd=root).decode().split("\0")
    # Read the index, not the working files. The index is what the next commit publishes.
    return {name: subprocess.check_output(["git", "show", f":{name}"], cwd=root) for name in names if name}


def archive_files(path: Path) -> dict[str, bytes]:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            return {name: archive.read(name) for name in archive.namelist() if not name.endswith("/")}
    with tarfile.open(path, "r:gz") as archive:
        files = {}
        for member in archive.getmembers():
            if member.isfile():
                stream = archive.extractfile(member)
                if stream is not None:
                    files[
                        str(PurePosixPath(member.name).relative_to(PurePosixPath(member.name).parts[0]))
                    ] = stream.read()
        return files


def check_files(files: dict[str, bytes], *, distribution: bool = False) -> list[str]:
    errors = [f"Non-public path: {name}" for name in files if not allowed(name, distribution=distribution)]
    for name, payload in files.items():
        if not name.endswith(".md"):
            continue
        for target in re.findall(r"\]\(([^)]+)\)", payload.decode("utf-8")):
            if target.startswith(("https:", "http:", "mailto:", "#")):
                continue
            target = target.split("#", 1)[0]
            destination = (PurePosixPath(name).parent / target).as_posix()
            # Collapse relative directory links without consulting ignored working files.
            stack: list[str] = []
            for part in PurePosixPath(destination).parts:
                if part == "..":
                    if stack:
                        stack.pop()
                elif part != ".":
                    stack.append(part)
            destination = "/".join(stack)
            if destination not in files:
                errors.append(f"Unpublished local link: {name} -> {target}")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", nargs="+", type=Path, help="inspect source distribution and wheel")
    args = parser.parse_args()
    if args.archives:
        for archive in args.archives:
            errors = check_files(archive_files(archive), distribution=True)
            if errors:
                parser.exit(1, "\n".join(errors) + "\n")
            print(f"Public distribution check passed: {archive.name}")
    else:
        files = git_files(Path(__file__).resolve().parents[1])
        if not files:
            parser.exit(1, "No staged or tracked files to inspect.\n")
        errors = check_files(files)
        if errors:
            parser.exit(1, "\n".join(errors) + "\n")
        print(f"Public Git tree check passed: {len(files)} files")


if __name__ == "__main__":
    main()
