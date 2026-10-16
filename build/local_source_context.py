#!/usr/bin/env python3
"""Copy local sources into a disposable Docker context, optionally including working-tree edits."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess

from sources import ROOT, git, gitlinks


WORKING_TREE_REPOSITORIES = {"cpython", "numpy", "crossjit-generator", "npbench"}


def mirror(origin, commit, target, public_url=None):
    if target.exists():
        raise ValueError("choose an empty source-context directory: " + str(target))
    subprocess.run(["git", "init", "--quiet", str(target)], check=True)
    subprocess.run(["git", "-C", str(target), "fetch", "--depth=1", "file://"+str(Path(origin).absolute()), commit], check=True)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", commit], check=True, stdout=subprocess.DEVNULL)
    if public_url:
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin", public_url], check=True)


def snapshot(origin, target, public_url=None):
    """Copy tracked and nonignored files, then commit only inside the new context.

    Independent snapshot commits make existing source/policy verification work
    without modifying the original repository or claiming its HEAD is clean.
    Recursive submodules are copied from their current working trees too.
    """
    origin = Path(origin).resolve()
    if target.exists():
        raise ValueError("choose an empty source-context directory: " + str(target))
    original_commit = git(origin, "rev-parse", "HEAD")
    index = subprocess.check_output(["git", "-C", str(origin), "ls-files", "--stage", "-z"])
    submodules = set()
    for entry in index.split(b"\0"):
        if not entry:
            continue
        metadata, name = entry.split(b"\t", 1)
        mode, _, stage = metadata.split()
        if stage != b"0":
            raise ValueError("resolve index conflicts before snapshotting: " + str(origin))
        if mode == b"160000":
            submodules.add(os.fsdecode(name))
    files = subprocess.check_output([
        "git", "-C", str(origin), "ls-files", "--cached", "--others", "--exclude-standard", "-z"])
    target.mkdir(parents=True)
    subprocess.run(["git", "init", "--quiet", str(target)], check=True)
    for name in sorted({os.fsdecode(p) for p in files.split(b"\0") if p}):
        if name in submodules:
            continue
        source, destination = origin / name, target / name
        if not source.exists() and not source.is_symlink():
            continue  # Preserve working-tree deletions.
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.is_symlink():
            destination.symlink_to(os.readlink(source))
        elif source.is_file():
            shutil.copy2(source, destination)
        else:
            raise ValueError("unsupported source entry: " + str(source))
    for name in sorted(submodules):
        child = origin / name
        if not child.is_dir():
            raise ValueError("local submodule is not initialized: " + str(child))
        snapshot(child, target / name)
    subprocess.run(["git", "-C", str(target), "-c", "advice.addEmbeddedRepo=false",
                    "add", "--all", "--force"], check=True)
    # Stable metadata gives identical source content the same snapshot identity.
    env = dict(os.environ, GIT_AUTHOR_NAME="CrossJIT local snapshot",
               GIT_AUTHOR_EMAIL="snapshot@crossjit.invalid",
               GIT_COMMITTER_NAME="CrossJIT local snapshot",
               GIT_COMMITTER_EMAIL="snapshot@crossjit.invalid",
               GIT_AUTHOR_DATE="2000-01-01T00:00:00+0000",
               GIT_COMMITTER_DATE="2000-01-01T00:00:00+0000")
    subprocess.run(["git", "-C", str(target), "-c", "commit.gpgsign=false",
                    "commit", "--quiet", "--allow-empty", "-m", "Local working-tree snapshot of " + original_commit],
                   check=True, env=env)
    if public_url:
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin", public_url], check=True)
    return original_commit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkouts", type=Path, required=True, help="JSON object mapping all repository names to local paths")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--working-tree", action="store_true",
                   help="Include current candidate/Generator/NPBench edits and nonignored new files; baselines remain locked")
    args = p.parse_args()
    checkouts = json.loads(args.checkouts.read_text())
    lock = json.loads((ROOT / "config/sources.lock.json").read_text())
    if args.output.exists():
        p.error("choose a new source-context directory: " + str(args.output))
    missing = set(lock["repositories"]) - set(checkouts)
    if missing:
        p.error("missing local checkouts: " + ", ".join(sorted(missing)))
    output = args.output.resolve()
    for path in checkouts.values():
        if output.is_relative_to(Path(path).resolve()):
            p.error("source context must be outside the source repositories")
    for name, row in lock["repositories"].items():
        original = Path(checkouts[name])
        destination = output / name
        if args.working_tree and name in WORKING_TREE_REPOSITORIES:
            original_commit = snapshot(original, destination, row["url"])
            row["commit"] = git(destination, "rev-parse", "HEAD")
            row["requested_ref"] = "local-working-tree"
            row["submodules"] = gitlinks(destination, row["commit"])
            row["local_snapshot"] = dict(original_commit=original_commit,
                                         tree=git(destination, "rev-parse", "HEAD^{tree}"))
            continue
        if gitlinks(original, row["commit"]) != row["submodules"]:
            raise ValueError("local objects/submodules differ from source lock: " + name)
        mirror(original, row["commit"], destination, row["url"])
        for relative, commit in sorted(row["submodules"].items(), key=lambda pair: pair[0].count("/")):
            child = destination / relative
            if child.is_dir() and not any(child.iterdir()):
                child.rmdir()
            mirror(original / relative, commit, child)
    if args.working_tree:
        lock["status"] = "local_worktree_snapshot_not_final_paper_freeze"
    (output / "source-lock.json").write_text(json.dumps(lock, indent=2, sort_keys=True) + "\n")
    print("Local source context: " + str(output))


if __name__ == "__main__":
    main()
