#!/usr/bin/env python3
"""Build locked Docker stages; resolve moving refs only with --develop."""
import argparse
from pathlib import Path
import subprocess
from sources import ROOT
import json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tag", default="crossjit-evaluation:calibration")
    p.add_argument("--target", choices=("toolchain", "calibration", "fixed-strategy", "production"), default="calibration")
    p.add_argument("--develop", action="store_true")
    p.add_argument("--policy", type=Path)
    p.add_argument("--optimization-preset", choices=("v32", "legacy-generated"), default="v32")
    p.add_argument("--platform", choices=("linux/amd64", "linux/arm64"))
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--source-context", type=Path, help="Local commit mirrors or working-tree snapshots from local_source_context.py; skips source cloning")
    p.add_argument("--log", type=Path, help="Save the complete build log without truncating compiler output")
    args = p.parse_args()
    if args.jobs < 1:
        p.error("jobs must be positive")
    if args.target == "production" and args.policy is None:
        p.error("production requires --policy pointing to the completed local calibration bundle")
    if args.develop:
        subprocess.run(["python3", str(ROOT / "build/sources.py"), "resolve"], check=True)
    lock = json.loads((ROOT / "config/toolchain.lock.json").read_text())
    command = ["docker", "buildx", "build", "--load", "--file", str(ROOT / "docker/Dockerfile"),
               "--target", args.target, "--tag", args.tag,
               "--build-arg", "BASE_IMAGE="+lock["base_image"],
               "--build-arg", "UBUNTU_SNAPSHOT="+lock["ubuntu_snapshot"],
               "--build-arg", "LLVM_PACKAGE_VERSION="+lock["llvm_package_version"],
               "--build-arg", "BUILD_JOBS="+str(args.jobs),
               "--build-arg", "OPTIMIZATION_PRESET="+args.optimization_preset]
    if args.source_context and args.target != "toolchain":
        index = command.index("--target") + 1
        command[index] = args.target + "-offline"
        command += ["--build-context", "source-cache="+str(args.source_context.absolute())]
    if args.platform:
        command += ["--platform", args.platform]
    if args.policy:
        command += ["--build-context", "policy="+str(args.policy.absolute())]
    if args.log:
        args.log.parent.mkdir(parents=True, exist_ok=True)
        with args.log.open("w") as log:
            subprocess.run([*command, "--progress=plain", str(ROOT)], check=True, stdout=log, stderr=subprocess.STDOUT)
    else:
        subprocess.run([*command, str(ROOT)], check=True)


if __name__ == "__main__":
    main()
