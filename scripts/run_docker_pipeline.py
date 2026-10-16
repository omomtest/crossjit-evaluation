#!/usr/bin/env python3
"""Snapshot local sources, build Docker images, optionally calibrate, and run selected RQs."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys

from common import ROOT, digest, hash_tree, read_json, save


def count(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return result


def warmup_count(value):
    result = int(value)
    if result < 0:
        raise argparse.ArgumentTypeError("must be nonnegative")
    return result


def random_seed(value):
    result = int(value)
    if not 0 <= result < 2**32:
        raise argparse.ArgumentTypeError("must be in [0, 2**32)")
    return result


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--results", type=Path, required=True, help="Host directory for source snapshot, logs, policy, and RQ results")
    p.add_argument("--warmup", type=warmup_count, default=300, help="Workload invocations before timing in each RQ worker")
    p.add_argument("--jit-stress", "--python-jit-stress", type=int, choices=(0, 1), default=1)
    p.add_argument("--calibrate", action=argparse.BooleanOptionalAction, default=False,
                   help="Calibrate generated admission locally; default reproduces the complete frozen v32 strategy")
    p.add_argument("--optimization-preset", choices=("v32", "legacy-generated"), default="v32",
                   help="Frozen strategy used without --calibrate/--policy")
    p.add_argument("--policy", type=Path, help="Reuse an existing policy instead of recalibrating")
    p.add_argument("--calibration-config", type=Path, default=ROOT / "config/calibration.json")
    p.add_argument("--checkouts", type=Path, help="Optional local checkout map; default downloads locked HTTPS sources")
    p.add_argument("--source-context", type=Path, help="Reuse a prepared local source context instead of snapshotting")
    p.add_argument("--image", help="Reuse an existing image, skip source/build stages, and mount current scripts")
    p.add_argument("--image-prefix", default="crossjit-evaluation")
    p.add_argument("--jobs", type=count, default=8)
    p.add_argument("--platform", choices=("linux/amd64", "linux/arm64"))
    p.add_argument("--rqs", nargs="+", choices=("rq1", "rq2", "rq3", "rq4"), default=["rq1", "rq2", "rq3", "rq4"])
    p.add_argument("--workloads", nargs="+", help="NPBench subset for RQ1/RQ2; unavailable with full RQ4 overhead")
    p.add_argument("--repeat", type=count, default=100, help="Timed workload invocations per measurement worker")
    p.add_argument("--trials", type=count, default=10, help="Independent process pairs per comparison")
    p.add_argument("--blas-threads", type=count, default=6, help="BLAS/OpenMP thread count for RQ workers")
    p.add_argument("--preset", choices=("S", "M", "L", "XL", "paper"), default="S",
                   help="NPBench input-size preset for RQ1, RQ2 and RQ4; RQ3 uses its own size grid")
    p.add_argument("--seed", type=random_seed, default=20261002, help="RQ input and experiment-order random seed")
    p.add_argument("--timeout", type=count, default=1800, help="Timeout in seconds per RQ worker")
    p.add_argument("--memory-budget-mb", type=count, default=1024,
                   help="RQ3 estimated workload memory budget in MiB; separate from Docker --memory")
    p.add_argument("--cpu-affinity")
    p.add_argument("--cpuset-cpus")
    p.add_argument("--memory")
    p.add_argument("--dry-run", action="store_true", help="Print commands without creating files or invoking Docker")
    return p


def plan(args):
    """Return ordered commands and protocol identity, without mutating anything."""
    if args.calibrate and args.policy:
        raise ValueError("choose --calibrate or --policy")
    if args.image and (args.calibrate or args.policy or args.source_context):
        raise ValueError("--image skips building; omit --calibrate, --policy, and --source-context")
    if args.workloads and "rq4" in args.rqs:
        raise ValueError("RQ4 overhead requires all 53 RQ2 workloads; omit --workloads or select other RQs")
    if args.policy and not (args.policy / "receipt.json").is_file():
        raise ValueError("policy must contain receipt.json")
    if args.source_context and not (args.source_context / "source-lock.json").is_file():
        raise ValueError("source context must contain source-lock.json")
    if args.checkouts and not args.checkouts.is_file():
        raise ValueError("checkout map does not exist")
    if args.calibrate and not args.calibration_config.is_file():
        raise ValueError("calibration config does not exist")
    output = args.results.resolve()
    context = args.source_context.resolve() if args.source_context else output / "sources"
    slug = re.sub(r"[^a-z0-9_.-]+", "-", output.name.lower()).strip(".-") or "run"
    image = args.image or f"{args.image_prefix}:{slug}-{'production' if args.calibrate or args.policy else 'fixed'}"
    calibration_image = f"{args.image_prefix}:{slug}-calibration"
    commands = []
    python = sys.executable

    def build(label, target, tag, policy=None):
        command = [python, str(ROOT / "build/docker_build.py"), "--target", target,
                   "--tag", tag, "--jobs", str(args.jobs)]
        if args.source_context or args.checkouts:
            command += ["--source-context", str(context)]
        if args.platform:
            command += ["--platform", args.platform]
        if policy:
            command += ["--policy", str(policy)]
        if target == "fixed-strategy":
            command += ["--optimization-preset", args.optimization_preset]
        commands.append((label, command))

    def run(label, selected_image, script, options):
        command = [python, str(ROOT / "scripts/docker_run.py"), "--image", selected_image, "--results", str(output)]
        if args.cpuset_cpus:
            command += ["--cpuset-cpus", args.cpuset_cpus]
        if args.memory:
            command += ["--memory", args.memory]
        if args.image:
            command += ["--scripts-dir", str(ROOT / "scripts")]
        command += ["scripts/" + script, *options]
        commands.append((label, command))

    if not args.image:
        if args.checkouts and not args.source_context and not (context / "source-lock.json").is_file():
            commands.append(("snapshot", [python, str(ROOT / "build/local_source_context.py"),
                             "--checkouts", str(args.checkouts.resolve()), "--working-tree", "--output", str(context)]))
        if args.calibrate:
            build("build-calibration", "calibration", calibration_image)
            calibration_options = ["--metadata", "/opt/crossjit/sources/numpy/build-evaluation/numpy/_core/generated_crossjit_kernels.json",
                                   "--profile", "/opt/crossjit/sources/numpy/evaluation-build-inputs/profile.json",
                                   "--calibration-config", "/results/inputs/calibration.json",
                                   "--jit-stress", str(args.jit_stress), "--output", "/results/calibration"]
            if args.cpu_affinity:
                calibration_options += ["--cpu-affinity", args.cpu_affinity]
            run("calibration", calibration_image, "calibrate.py", calibration_options)
            build("build-production", "production", image, output / "calibration/policy")
        elif args.policy:
            build("build-production", "production", image, args.policy.resolve())
        else:
            build("build-fixed", "fixed-strategy", image)

    selected = set(args.rqs)
    if "rq4" in selected:
        selected.update(("rq2", "rq3"))
    if "rq3" in selected:
        selected.add("rq2")  # Captured execution descriptions come from RQ2.
    options = ["--warmup", str(args.warmup), "--jit-stress", str(args.jit_stress),
               "--repeat", str(args.repeat), "--trials", str(args.trials),
               "--blas-threads", str(args.blas_threads), "--preset", args.preset,
               "--seed", str(args.seed), "--timeout", str(args.timeout),
               "--memory-budget-mb", str(args.memory_budget_mb)]
    if args.cpu_affinity:
        options += ["--cpu-affinity", args.cpu_affinity]
    phases = [("rq1", "run_rq1_performance.py", "rq1"),
              ("rq2", "run_rq2_coverage.py", "rq2"),
              ("rq2-adi", "run_microbench_case_study.py", "rq2"),
              ("rq3", "run_rq3_composition.py", "rq3"),
              ("rq3-storage", "run_rq3_storage.py", "rq3"),
              ("rq4-callback", "run_rq4_attribution.py", "rq4"),
              ("rq4-inactive", "run_rq4_overhead.py", "rq4"),
              ("rq4-guard", "run_rq4_guard.py", "rq4")]
    for label, script, rq in phases:
        if rq not in selected:
            continue
        script_options = [*options, "--output", "/results/" + label]
        if args.workloads and label in ("rq1", "rq2", "rq3"):
            script_options += ["--workloads", *args.workloads]
        if label == "rq3-storage":
            script_options += ["--reach", "/results/rq2/reach.json"]
        if label == "rq4-inactive":
            script_options += ["--coverage", "/results/rq2/coverage.json"]
        if label == "rq4-guard":
            script_options = ["--scope", "/results/rq3", "--output", "/results/rq4-guard"]
        run(label, image, script, script_options)
    configuration = dict(warmup=args.warmup, jit_stress=args.jit_stress, calibrate=args.calibrate,
                         optimization_preset="existing-image" if args.image else "calibrated-generated" if args.calibrate or args.policy else args.optimization_preset,
                         policy=str(args.policy.resolve()) if args.policy else None, image=image,
                         source_context=str(context) if args.source_context or args.checkouts else None, rqs=sorted(selected), workloads=args.workloads,
                         repeat=args.repeat, trials=args.trials, blas_threads=args.blas_threads, preset=args.preset,
                         seed=args.seed, timeout=args.timeout, memory_budget_mb=args.memory_budget_mb,
                         jobs=args.jobs, platform=args.platform, cpu_affinity=args.cpu_affinity,
                         cpuset_cpus=args.cpuset_cpus, memory=args.memory,
                         calibration_config_sha256=digest(args.calibration_config) if args.calibrate else None,
                         policy_sha256=hash_tree(args.policy) if args.policy else None,
                         source_lock_sha256=digest(context / "source-lock.json") if args.source_context else None,
                         checkouts_sha256=digest(args.checkouts) if args.checkouts and not args.image and not args.source_context else None,
                         scripts_sha256=hash_tree(ROOT / "scripts"), build_sha256=hash_tree(ROOT / "build"),
                         dockerfile_sha256=digest(ROOT / "docker/Dockerfile"), config_sha256=hash_tree(ROOT / "config"))
    return commands, configuration


def main():
    p = argument_parser()
    args = p.parse_args()
    try:
        commands, configuration = plan(args)
    except ValueError as exc:
        p.error(str(exc))
    if args.dry_run:
        for label, command in commands:
            print(label + ": " + shlex.join(command))
        return
    output = args.results.resolve()
    manifest_path = output / "pipeline.json"
    previous = None
    if manifest_path.is_file():
        previous = read_json(manifest_path)
        if previous["configuration"] != configuration:
            p.error("results belong to another pipeline configuration; choose a new --results directory")
    elif output.exists() and any(output.iterdir()):
        p.error("choose a new --results directory for this pipeline")
    output.mkdir(parents=True, exist_ok=True)
    if args.calibrate:
        (output / "inputs").mkdir(exist_ok=True)
        shutil.copyfile(args.calibration_config, output / "inputs/calibration.json")
    manifest = previous or dict(schema_version=1, configuration=configuration, phases=[], complete=False)
    successful = {phase["phase"] for phase in manifest["phases"] if phase["exitcode"] == 0}
    manifest["complete"] = False
    save(manifest_path, manifest)
    failed = False
    for label, command in commands:
        # Keep completed calibration evidence/policies byte-for-byte unchanged
        # on resume; regenerating their timestamps would invalidate RQ identity.
        if label in successful:
            print(label + ": already completed; preserving recorded artifacts", flush=True)
            continue
        print("\n" + label + ": " + shlex.join(command), flush=True)
        log_path = output / "logs" / (label + ".log")
        log_path.parent.mkdir(exist_ok=True)
        start = datetime.now(timezone.utc).isoformat()
        with log_path.open("w") as log:
            process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        manifest["phases"] = [phase for phase in manifest["phases"] if phase["phase"] != label]
        manifest["phases"].append(dict(phase=label, command=command, exitcode=code, started_at_utc=start, log=str(log_path)))
        failed |= code != 0
        save(manifest_path, manifest)
        if code and not label.startswith("rq"):
            raise SystemExit(code)
    manifest["complete"] = not failed
    save(manifest_path, manifest)
    if failed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
