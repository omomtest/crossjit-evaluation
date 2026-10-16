#!/usr/bin/env python3
"""Clone locked sources locally, copy into Docker, run fixed-protocol RQ1, and export evidence."""
import argparse
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import shlex
import shutil
import subprocess
import sys

from common import ROOT, canonical_hash, digest, read_json, save, system_info
from rq1_bundle import bundle
from run_docker_pipeline import count, random_seed


def label(value):
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,79}", value) or ".." in value:
        raise argparse.ArgumentTypeError("use 1-80 lowercase letters, digits, '.', '_' or '-'; no '..'")
    return value


def capture(command):
    return subprocess.check_output(command, text=True).strip()


def run(command):
    print(shlex.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True, cwd=ROOT)


def host_os():
    if platform.system() == "Darwin":
        return "macos"
    return "windows-wsl2" if "microsoft" in platform.release().lower() else "linux"


def native_platform():
    aliases = {"x86_64": "amd64", "aarch64": "arm64", "arm64": "arm64", "amd64": "amd64"}
    client = aliases.get(platform.machine().lower())
    server = capture(["docker", "info", "--format", "{{.OSType}}/{{.Architecture}}"])
    os_type, _, architecture = server.partition("/")
    if os_type != "linux" or aliases.get(architecture.lower()) != client or client is None:
        raise ValueError("Docker must run local Linux containers on the client's native amd64/arm64 architecture")
    return "linux/" + client


def argument_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--machine", type=label, required=True, help="Unique physical-machine label")
    p.add_argument("--run-id", type=label, default="rq1-v32-w300-stress1-s", help="Reuse the same ID to resume")
    p.add_argument("--workspace", type=Path, default=ROOT.parent / "rq1-runs")
    p.add_argument("--jobs", type=count, default=4, help="Build jobs, not benchmark concurrency")
    p.add_argument("--optimization-preset", choices=("v32", "legacy-generated"), default="v32",
                   help="v32 reproduces all frozen acceptance policies; legacy-generated reproduces the old 8-active build")
    p.add_argument("--blas-threads", type=count, default=6)
    p.add_argument("--trials", type=count, default=10)
    p.add_argument("--repeat", type=count, default=100)
    p.add_argument("--preset", choices=("S", "M", "L", "XL", "paper"), default="S")
    p.add_argument("--seed", type=random_seed, default=20261002)
    p.add_argument("--timeout", type=count, default=1800)
    p.add_argument("--cpuset-cpus")
    p.add_argument("--cpu-affinity")
    p.add_argument("--memory", help="Docker hard memory limit, e.g. 12g")
    p.add_argument("--hardware-label", default="", help="Host CPU model / Mac model; stored outside the VM")
    p.add_argument("--notes", default="", help="Host OS version, VM resources, power mode, etc.")
    p.add_argument("--workloads", nargs="+", help="Smoke subset; omit for full53")
    p.add_argument("--dry-run", action="store_true", help="Print the workflow without downloads, Docker, or writes")
    return p


def pipeline_command(args, run_root, target_platform):
    command = [sys.executable, str(ROOT / "scripts/run_docker_pipeline.py"),
               "--source-context", str(run_root / "source-context"),
               "--results", str(run_root / "experiment"), "--rqs", "rq1", "--no-calibrate",
               "--image-prefix", f"crossjit-rq1-{args.machine}-{args.run_id}",
               "--warmup", "300", "--jit-stress", "1", "--platform", target_platform,
               "--optimization-preset", args.optimization_preset]
    for key in ("jobs", "blas_threads", "trials", "repeat", "preset", "seed", "timeout",
                "cpuset_cpus", "cpu_affinity", "memory"):
        value = getattr(args, key)
        if value is not None:
            command += ["--" + key.replace("_", "-"), str(value)]
    if args.workloads:
        command += ["--workloads", *args.workloads]
    return command


def copy_evaluation(target):
    for folder in ("scripts", "build", "config", "docker", "docs"):
        for path in (ROOT / folder).rglob("*"):
            if (not path.is_file() or path.is_symlink() or "__pycache__" in path.parts
                    or ".cache" in path.parts or path.name.endswith(".local.json")):
                continue
            destination = target / path.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, destination)
    shutil.copyfile(ROOT / "README.md", target / "README.md")


def main():
    p = argument_parser()
    args = p.parse_args()
    if platform.system() not in ("Linux", "Darwin"):
        p.error("on Windows run this script inside WSL2 Ubuntu")
    if args.trials < 10 and not args.workloads:
        p.error("full RQ1 requires at least 10 process pairs; use --workloads for smoke")
    workspace = args.workspace.expanduser().resolve()
    run_root = workspace / "runs" / args.machine / args.run_id
    lock = read_json(ROOT / "config/sources.lock.json")
    # Separate checkout caches when the lock changes; never reset an existing tree.
    checkout_root = workspace / "checkouts" / canonical_hash(lock)[:16]
    archive = workspace / "exports" / f"{args.machine}--{args.run_id}.tar.gz"
    if args.dry_run:
        print("1. Check local Docker/Buildx and native architecture; record host metadata")
        print("2. Clone exact locked commits and recursive submodules into " + str(checkout_root))
        print("3. Copy local commit mirrors into " + str(run_root / "source-context"))
        print("4. " + shlex.join(pipeline_command(args, run_root, "linux/<native>")))
        print("5. Export raw data, summaries, manifests, logs and evaluation scripts: " + str(archive))
        return
    for executable in ("git", "docker"):
        if not shutil.which(executable):
            p.error("missing executable: " + executable)
    target_platform = native_platform()
    capture(["docker", "buildx", "version"])
    metadata = run_root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    command = pipeline_command(args, run_root, target_platform)
    configuration = dict(machine=args.machine, run_id=args.run_id, platform=target_platform,
                         optimization_preset=args.optimization_preset,
                         optimization_bundle_sha256=digest(ROOT / "config/optimizations/v32/bundle.json") if args.optimization_preset == "v32" else None,
                         source_lock_sha256=digest(ROOT / "config/sources.lock.json"),
                         toolchain_lock_sha256=digest(ROOT / "config/toolchain.lock.json"),
                         pipeline_command=command, hardware_label=args.hardware_label, notes=args.notes)
    state = metadata / "run.json"
    if state.is_file() and read_json(state)["configuration"] != configuration:
        p.error("run ID belongs to different settings; choose another --run-id")
    if not state.is_file():
        save(state, dict(schema_version=1, configuration=configuration,
                         created_at_utc=datetime.now(timezone.utc).isoformat(), pipeline_exitcode=None))
        save(metadata / "host.json", dict(host_os=host_os(), platform=platform.platform(),
             architecture=platform.machine(), hostname=platform.node(), hardware_label=args.hardware_label,
             notes=args.notes, system=system_info(), python_version=sys.version,
             docker_version=capture(["docker", "version", "--format", "{{json .}}"]),
             docker_info=capture(["docker", "info", "--format", "{{json .}}"])))
        shutil.copyfile(ROOT / "config/sources.lock.json", metadata / "sources.lock.json")
        shutil.copyfile(ROOT / "config/toolchain.lock.json", metadata / "toolchain.lock.json")
        copy_evaluation(run_root / "evaluation")
    # checkout/verify reject edits, a different HEAD, or mismatched recursive links.
    run([sys.executable, ROOT / "build/sources.py", "checkout", "--root", checkout_root])
    mapping = metadata / "checkouts.json"
    save(mapping, {name: str(checkout_root / name) for name in lock["repositories"]})
    context = run_root / "source-context"
    if not (context / "source-lock.json").is_file():
        if context.exists():
            p.error("partial source context; use a new --run-id (original evidence is preserved)")
        run([sys.executable, ROOT / "build/local_source_context.py", "--checkouts", mapping, "--output", context])
    if read_json(context / "source-lock.json") != lock:
        p.error("prepared source context does not match source lock")
    run([sys.executable, ROOT / "build/sources.py", "verify", "--root", context])
    print(shlex.join(command), flush=True)
    code = subprocess.run(command, cwd=ROOT).returncode
    status = read_json(state)
    status["pipeline_exitcode"] = code
    save(state, status)
    bundle(run_root, archive, args.machine, args.run_id)
    print("Evidence bundle: " + str(archive), flush=True)
    print("Transfer it together with " + archive.name + ".sha256", flush=True)
    raise SystemExit(code)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(str(exc)) from exc
    except KeyboardInterrupt:
        print("RQ1 interrupted; rerun the same command to resume.", file=sys.stderr)
        raise SystemExit(130) from None
