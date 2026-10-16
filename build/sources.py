#!/usr/bin/env python3
"""Resolve refs once and clone exact locked commits, including recursive Git links."""
from __future__ import annotations
import argparse
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import re
import shlex
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


@contextmanager
def download_guard(path):
    directory = Path(git(path, "rev-parse", "--absolute-git-dir"))
    with (directory / "crossjit-download.guard").open("a") as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise ValueError("another source download is using: " + str(path)) from None
        try:
            yield directory
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)


@contextmanager
def checkout_guard(target):
    # Cover init, remote registration and checkout too, before .git exists.
    target.parent.mkdir(parents=True, exist_ok=True)
    with (target.parent / ("." + target.name + ".checkout.guard")).open("a") as guard:
        while True:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                print("Waiting for another source checkout: " + str(target), file=sys.stderr, flush=True)
                time.sleep(5)
        try:
            yield
        finally:
            fcntl.flock(guard, fcntl.LOCK_UN)


def prepare_submodules(root, recovery_root=None):
    """Preserve unusable clone remnants; reject edits in usable checkouts."""
    root = Path(root)
    if recovery_root is None:
        recovery_root = root.parent / ".crossjit-source-recovery"
    gitdir = Path(git(root, "rev-parse", "--absolute-git-dir"))
    modules = root / ".gitmodules"
    if not modules.is_file():
        return
    paths = subprocess.run(["git", "config", "--file", str(modules), "--get-regexp", r"^submodule\..*\.path$"],
                           text=True, stdout=subprocess.PIPE, check=False)
    if paths.returncode not in (0, 1):
        raise subprocess.CalledProcessError(paths.returncode, paths.args)
    for line in paths.stdout.splitlines():
        key, relative = line.split(None, 1)
        name = key[len("submodule."):-len(".path")]
        child = root / relative
        module_dir = gitdir / "modules" / name
        usable = False
        if (child / ".git").exists():
            head = subprocess.run(["git", "-C", str(child), "rev-parse", "--verify", "HEAD^{commit}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            usable = head.returncode == 0
        if usable:
            subprocess.run(["git", "-C", str(child), "diff", "--quiet", "--ignore-submodules=all", "HEAD"], check=True)
            prepare_submodules(child, recovery_root)
            continue
        incomplete = child.exists() and any(child.iterdir())
        if module_dir.exists():
            head = subprocess.run(["git", "--git-dir", str(module_dir), "rev-parse", "--verify", "HEAD^{commit}"],
                                  stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            incomplete = incomplete or head.returncode != 0
        if incomplete:
            recovery_root.mkdir(parents=True, exist_ok=True)
            backup = Path(tempfile.mkdtemp(prefix=root.name + "-", dir=recovery_root))
            if child.exists():
                child.rename(backup / "worktree")
            if module_dir.exists():
                module_dir.rename(backup / "gitdir")
            print("Preserved incomplete submodule clone at " + str(backup) + "; retrying "
                  + str(child), file=sys.stderr, flush=True)


def recover_git_locks(directory):
    """Recover interrupted downloads only while no Git process is running."""
    locks = [path for path in directory.rglob("*.lock") if path.is_file() and not path.is_symlink()]
    if not locks:
        return True
    # Git lockfiles do not store a PID. Be conservative about external Git
    # users as well as other downloads; ps works on both macOS and Linux/WSL.
    try:
        processes = subprocess.check_output(["ps", "-A", "-o", "stat=", "-o", "comm="], text=True)
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError("cannot check whether Git locks are stale in " + str(directory)) from error
    for line in processes.splitlines():
        fields = line.split(None, 1)
        if len(fields) != 2:
            continue
        status, command = fields
        executable = Path(command).name
        if not status.startswith("Z") and (executable == "git" or executable.startswith("git-")):
            print("Waiting for active Git processes before recovering download locks in "
                  + str(directory), file=sys.stderr, flush=True)
            return False
    for lock in locks:
        lock.unlink(missing_ok=True)
        print("Removed stale Git download lock: " + str(lock), file=sys.stderr, flush=True)
    return True


def run_network_command(command, env):
    # Network inactivity is bounded by Git/SSH settings below, rather than a
    # wall-clock deadline that kills an otherwise progressing large download.
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        code = process.wait()
    except BaseException:
        # Stop transport and submodule children too. SIGTERM gives Git a chance
        # to remove its lockfiles; SIGKILL is only a fallback during shutdown.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
        raise
    if code:
        raise subprocess.CalledProcessError(code, command)
    return subprocess.CompletedProcess(command, code)


def network_git(path, *args):
    """Retry network operations until success; Ctrl-C still stops the caller."""
    command = ["git", "-c", "http.lowSpeedLimit=1", "-c", "http.lowSpeedTime=60",
               "-C", str(path), *args]
    env = dict(os.environ, GIT_TERMINAL_PROMPT="0")
    env["GIT_SSH_COMMAND"] = (env.get("GIT_SSH_COMMAND", "ssh") +
                              " -o BatchMode=yes -o ConnectTimeout=30"
                              " -o ServerAliveInterval=15 -o ServerAliveCountMax=3")
    delay, attempt = 5, 1
    with download_guard(path) as directory:
        while True:
            if recover_git_locks(directory):
                if args and args[0] == "submodule":
                    prepare_submodules(path)
                try:
                    return run_network_command(command, env)
                except subprocess.CalledProcessError as error:
                    print(f"Git attempt {attempt} failed ({error}); retrying in {delay}s: "
                          + shlex.join(command), file=sys.stderr, flush=True)
            time.sleep(delay)
            delay = min(delay * 2, 60)
            attempt += 1


def gitlinks(root, commit, prefix=""):
    result = {}
    for line in git(root, "ls-tree", "-r", commit).splitlines():
        metadata, path = line.split("\t", 1)
        mode, kind, sha = metadata.split()
        if kind == "commit":
            result[prefix + path] = sha
            child = Path(root) / path
            if not child.exists():
                raise ValueError("recursive submodule is not initialized: " + str(child))
            result.update(gitlinks(child, sha, prefix + path + "/"))
    return result


def checkout_one(url, commit, target):
    target = Path(target)
    with checkout_guard(target):
        checkout_locked(url, commit, target)


def checkout_locked(url, commit, target):
    if target.exists():
        head = subprocess.run(["git", "-C", str(target), "rev-parse", "--verify", "HEAD"],
                              text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if head.returncode == 0 and head.stdout.strip() != commit:
            raise ValueError("existing source checkout differs from lock: " + str(target))
        if head.returncode == 0:
            subprocess.run(["git", "-C", str(target), "diff", "--quiet", "--ignore-submodules=all", "HEAD"], check=True)
            # A previous download may have stopped while initializing submodules.
            network_git(target, "submodule", "update", "--init", "--recursive", "--depth=1")
            return
        # Resume a checkout interrupted after init but before its first checkout.
        remote = subprocess.run(["git", "-C", str(target), "remote", "get-url", "origin"],
                                text=True, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        if remote.returncode != 0 and (target / ".git").is_dir():
            subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
        elif remote.returncode != 0 or remote.stdout.strip() != url:
            raise ValueError("existing source remote differs from lock: " + str(target))
    else:
        subprocess.run(["git", "init", str(target)], check=True, stdout=subprocess.DEVNULL)
        subprocess.run(["git", "-C", str(target), "remote", "add", "origin", url], check=True)
    network_git(target, "fetch", "--depth=1", "origin", commit)
    subprocess.run(["git", "-C", str(target), "checkout", "--detach", commit], check=True, stdout=subprocess.DEVNULL)
    network_git(target, "submodule", "update", "--init", "--recursive", "--depth=1")


def resolve(config, local):
    records = {}
    with tempfile.TemporaryDirectory(prefix="crossjit-resolve-") as directory:
        for name, row in config["repositories"].items():
            url, ref = row["url"], row["ref"]
            if name in local:
                source = Path(local[name])
                commit = git(source, "rev-parse", ref + "^{commit}")
            else:
                source = Path(directory) / name
                subprocess.run(["git", "init", str(source)], check=True, stdout=subprocess.DEVNULL)
                subprocess.run(["git", "-C", str(source), "remote", "add", "origin", url], check=True)
                network_git(source, "fetch", "--depth=1", "origin", ref)
                commit = git(source, "rev-parse", "FETCH_HEAD^{commit}")
                subprocess.run(["git", "-C", str(source), "checkout", "--detach", commit], check=True)
                network_git(source, "submodule", "update", "--init", "--recursive", "--depth=1")
            if row.get("fixed") and commit != ref:
                raise ValueError("fixed baseline/benchmark ref must be a full matching SHA")
            if not re.fullmatch(r"[0-9a-f]{40}", commit):
                raise ValueError("invalid source commit")
            records[name] = dict(url=url, requested_ref=ref, commit=commit, submodules=gitlinks(source, commit))
    return dict(schema_version=1, status="resolved_main_snapshot", repositories=records)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("action", choices=("resolve", "checkout", "verify"))
    p.add_argument("--config", type=Path, default=ROOT / "config/sources.json")
    p.add_argument("--lock", type=Path, default=ROOT / "config/sources.lock.json")
    p.add_argument("--root", type=Path, default=ROOT / "sources")
    p.add_argument("--local", action="append", default=[], help="name=existing checkout; lock committed objects only")
    args = p.parse_args()
    if args.action == "resolve":
        data = resolve(json.loads(args.config.read_text()), dict(item.split("=", 1) for item in args.local))
        args.lock.parent.mkdir(parents=True, exist_ok=True)
        args.lock.write_text(json.dumps(data, sort_keys=True, indent=2)+"\n")
        return
    lock = json.loads(args.lock.read_text())
    for name, row in lock["repositories"].items():
        source = args.root / name
        if args.action == "checkout":
            checkout_one(row["url"], row["commit"], source)
        if git(source, "rev-parse", "HEAD") != row["commit"] or gitlinks(source, row["commit"]) != row["submodules"]:
            raise ValueError("source/submodule lock mismatch: " + name)
        subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD"], check=True)
    print("All source commits and recursive submodules match the lock.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Source download interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
