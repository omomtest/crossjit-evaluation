#!/usr/bin/env python3
"""Build two isolated JIT CPython/NumPy environments; no experiment runs here."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def run(argv, cwd=None, env=None):
    subprocess.run(list(map(str, argv)), cwd=cwd, env=env, check=True)


def build_python(source, prefix, venv, lock, jobs):
    env = dict(os.environ, CC="gcc-13", CXX="g++-13", CFLAGS=lock["cflags"], PYTHON_JIT="1")
    run([source / "configure", "--prefix="+str(prefix), *lock["configure_flags"]], cwd=source, env=env)
    run(["make", "regen-cases", "regen-jit"], cwd=source, env=env)
    run(["make", "-j"+str(jobs)], cwd=source, env=env)
    run(["make", "install"], cwd=source, env=env)
    run([prefix / "bin/python3", "-m", "venv", venv])
    run([venv / "bin/python3", "-m", "pip", "install", "-r", ROOT / "config/build-requirements.lock.txt"])


def build_numpy(source, venv, lock, policy=None, calibration=False, jobs=2, optimization_preset="legacy-generated"):
    python = venv / "bin/python3"
    env = dict(os.environ, PATH=str(venv / "bin")+os.pathsep+os.environ["PATH"],
               CC="gcc-13", CXX="g++-13", FC="gfortran-13", CFLAGS=lock["cflags"], PYTHON_JIT="1", PYTHON_JIT_CROSSJIT="0")
    flags = list(lock["numpy_flags"])
    if policy:
        from install_policy import install_policy
        install_policy(source, policy)
        flags += ["-Dcrossjit-affine-profile="+str(policy / "profile.json"),
                  "-Dcrossjit-affine-policy="+str(policy / "execution_decision_db.json"),
                  "-Dcrossjit-production-selection="+str(policy / "production_selection.json")]
    elif calibration:
        from install_policy import prepare_calibration
        profile = prepare_calibration(source)
        flags += ["-Dcrossjit-affine-profile="+str(profile)]
    elif optimization_preset == "v32":
        from frozen_optimizations import install_frozen, meson_flags
        flags += meson_flags(install_frozen(source))
    build = source / "build-evaluation"
    meson = source / "vendored-meson/meson/meson.py"
    command = [python, meson, "setup", build, "--prefix="+str(venv), *flags]
    if build.exists():
        command.append("--reconfigure")
    run(command, cwd=source, env=env)
    run([python, meson, "compile", "-C", build, "-j", jobs], cwd=source, env=env)
    run([python, meson, "install", "-C", build], cwd=source, env=env)


def manifest(prefix, sources, policy, calibration, optimization_preset="legacy-generated"):
    sys.path.insert(0, str(ROOT / "scripts"))
    from common import digest, save
    # Probe each interpreter in a fresh process; this script itself may use system Python.
    probes = {}
    for name in ("baseline", "candidate"):
        path = prefix / (name+"-probe.json")
        env = dict(os.environ, PYTHON_JIT="1", OPENBLAS_NUM_THREADS="1", PYTHON_JIT_CROSSJIT="0")
        run([prefix / (name+"-venv/bin/python3"), "-B", ROOT / "scripts/worker.py", "--probe", "--output", path], env=env)
        probes[name] = json.loads(path.read_text())
    hashes = {}
    for record in probes.values():
        hashes.update({p: digest(p) for p in record["binary_files"]})
    artifacts = {}
    for name in ("generated_crossjit_kernels.c", "generated_crossjit_kernels.h", "generated_crossjit_kernels.json",
                 "generated_crossjit_native_policy.h", "generated_crossjit_execution.h",
                 "generated_crossjit_loops.h", "generated_crossjit_backend.h"):
        for path in (sources / "numpy/build-evaluation").rglob(name):
            artifacts[str(path)] = digest(path)
    packages = subprocess.check_output(["dpkg-query", "-W", "-f=${Package}=${Version}\n"], text=True).splitlines()
    pip = {}
    for name in probes:
        pip[name] = subprocess.check_output([str(prefix / (name+"-venv/bin/python3")), "-m", "pip", "freeze", "--all"], text=True).splitlines()
    overrides = {}
    for relative in ("numpy/_core/code_generators/generate_crossjit_kernels.py", "numpy/_core/code_generators/crossjit_decision_environment.json",
                     "benchmarks/benchmarks/crossjit/results/g14/g14_decision_db.json",
                     "benchmarks/benchmarks/crossjit/results/profiled/pf4_inner_contiguous/execution_decision_db.json",
                     "benchmarks/benchmarks/crossjit/results/profiled/pf5_scalar_zero_stride/execution_decision_db.json"):
        path = sources / "numpy" / relative
        if path.is_file():
            overrides[relative] = digest(path)
    frozen = None
    if not policy and not calibration and optimization_preset == "v32":
        from frozen_optimizations import identity, validate_bundle
        frozen = identity(sources / "numpy")
        expected = validate_bundle()["expected_generated_hashes"]
        actual = {Path(path).name: value for path, value in artifacts.items()}
        for name, value in expected.items():
            if actual.get(name) != value:
                raise ValueError("v32 generated artifact differs from acceptance build: " + name)
    save(prefix / "build-manifest.json", dict(schema_version=1, source_lock=json.loads((ROOT / "config/sources.lock.json").read_text()),
                                             toolchain_lock=json.loads((ROOT / "config/toolchain.lock.json").read_text()),
                                             source_root=str(sources), probes=probes, binary_hashes=hashes,
                                             generated_artifacts=artifacts, effective_generation_input_hashes=overrides,
                                             apt_packages=packages, pip_packages=pip,
                                             policy_hashes={p.name: digest(p) for p in policy.glob("*.json")} if policy else frozen["installed_policy_hashes"] if frozen else {},
                                             optimization_preset="calibrated-generated" if policy else "calibration" if calibration else optimization_preset,
                                             frozen_optimization_policy=frozen,
                                             calibration_receipt=json.loads((policy / "receipt.json").read_text()) if policy else None,
                                             production_ready=bool(policy), calibration_capability_build=calibration,
                                             fixed_strategy_migration=not policy and not calibration))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sources", type=Path, default=Path("/opt/crossjit/sources"))
    p.add_argument("--prefix", type=Path, default=Path("/opt/crossjit"))
    p.add_argument("--jobs", type=int, default=min(8, os.cpu_count() or 2))
    p.add_argument("--policy", type=Path)
    p.add_argument("--calibration", action="store_true")
    p.add_argument("--optimization-preset", choices=("v32", "legacy-generated"), default="v32",
                   help="Frozen build configuration; v32 reproduces the complete acceptance policies")
    p.add_argument("--numpy-only", action="store_true")
    p.add_argument("--python-only", action="store_true")
    args = p.parse_args()
    if args.policy and args.calibration:
        p.error("choose either a calibrated production policy or a calibration capability build")
    if args.numpy_only and args.python_only:
        p.error("choose one build component")
    lock = json.loads((ROOT / "config/toolchain.lock.json").read_text())
    args.prefix.mkdir(parents=True, exist_ok=True)
    for name, repo in (("baseline", "baseline-cpython"), ("candidate", "cpython")):
        if not args.numpy_only:
            build_python(args.sources / repo, args.prefix / (name+"-python"), args.prefix / (name+"-venv"), lock, args.jobs)
    if args.python_only:
        return
    build_numpy(args.sources / "baseline-numpy", args.prefix / "baseline-venv", lock, jobs=args.jobs)
    build_numpy(args.sources / "numpy", args.prefix / "candidate-venv", lock, args.policy, args.calibration, args.jobs,
                args.optimization_preset)
    manifest(args.prefix, args.sources, args.policy, args.calibration, args.optimization_preset)


if __name__ == "__main__":
    main()
