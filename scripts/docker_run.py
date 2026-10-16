#!/usr/bin/env python3
"""Run a credential-free, network-disabled experiment container with writable results."""
import argparse
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image", default="crossjit-evaluation:production")
    p.add_argument("--results", type=Path, required=True)
    p.add_argument("--cpuset-cpus")
    p.add_argument("--memory", help="Optional Docker memory limit; recorded in experiment manifest")
    p.add_argument("--scripts-dir", type=Path, help="Mount updated evaluation scripts into an existing image (read-only)")
    p.add_argument("command", nargs=argparse.REMAINDER, help="e.g. scripts/run_rq1_performance.py --warmup 500 --output /results/rq1")
    args = p.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        p.error("provide the container Python script and arguments")
    if args.scripts_dir and not args.scripts_dir.is_dir():
        p.error("scripts directory does not exist: " + str(args.scripts_dir))
    args.results.mkdir(parents=True, exist_ok=True)
    inspect = subprocess.check_output(["docker", "image", "inspect", args.image, "--format", "{{.Architecture}}"], text=True).strip()
    native = subprocess.check_output(["docker", "info", "--format", "{{.Architecture}}"], text=True).strip()
    aliases = {"x86_64": "amd64", "aarch64": "arm64"}
    if aliases.get(inspect, inspect) != aliases.get(native, native):
        raise SystemExit("image architecture differs from Docker host; emulated performance runs are rejected")
    command = ["docker", "run", "--rm", "--network=none", "--user", f"{os.getuid()}:{os.getgid()}",
               "--mount", f"type=bind,source={args.results.absolute()},target=/results"]
    if args.cpuset_cpus:
        command += ["--cpuset-cpus", args.cpuset_cpus]
    if args.memory:
        command += ["--memory", args.memory]
    if args.scripts_dir:
        command += ["--mount", f"type=bind,source={args.scripts_dir.resolve()},target=/opt/evaluation/scripts,readonly"]
    subprocess.run([*command, args.image, *args.command], check=True)


if __name__ == "__main__":
    main()
