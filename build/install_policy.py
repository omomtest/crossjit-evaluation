"""Explicit generation-input overlays, applied only to disposable build checkouts.

The offline generator first materializes the complete candidate universe, then
the independently measured production selection prunes it. No runtime proof or
guard is changed. Every overlay is hashed in build-manifest.json.
"""
import hashlib
import json
from pathlib import Path
import shutil


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True)+"\n")


def complete_generation_universe(source):
    path = source / "numpy/_core/code_generators/generate_crossjit_kernels.py"
    text = path.read_text()
    marker = "    resolution = resolve_generation_requests("
    insertion = "    # Evaluation build overlay: enumerate inventory; production selection below prunes it.\n    selected_hashes = frozenset(request.signature.stable_hash() for request in requests)\n"
    if insertion in text:
        return
    if text.count(marker) != 1:
        raise ValueError("unsupported generator frontend: universe overlay marker changed")
    path.write_text(text.replace(marker, insertion + marker))


def prepare_calibration(source):
    complete_generation_universe(source)
    from decision_environment import environment
    save(source / "numpy/_core/code_generators/crossjit_decision_environment.json",
         environment("gcc-13", ["-std=c11", "-O3", "-fno-strict-aliasing", "-fPIC"]))
    directory = source / "evaluation-build-inputs"
    directory.mkdir(exist_ok=True)
    existing = source / "benchmarks/benchmarks/crossjit/results/profiled/pf6_coverage/opportunity/coverage_report.json"
    shutil.copyfile(existing, directory / "profile.json")
    # The compile inventory must not admit using another machine's measured cost.
    save(source / "benchmarks/benchmarks/crossjit/results/g14/g14_decision_db.json", dict(schema_version=1, records=[]))
    for phase in ("pf4_inner_contiguous", "pf5_scalar_zero_stride"):
        target = source / "benchmarks/benchmarks/crossjit/results/profiled" / phase / "execution_decision_db.json"
        data = json.loads(target.read_text())
        data["records"] = []
        save(target, data)
    return directory / "profile.json"


def install_policy(source, bundle):
    receipt = json.loads((bundle / "receipt.json").read_text())
    for name, expected in receipt["file_hashes"].items():
        if hashlib.sha256((bundle / name).read_bytes()).hexdigest() != expected:
            raise ValueError("calibration policy file drift: " + name)
    import subprocess
    from decision_environment import environment
    source_root = source.parent
    for name, sha in receipt["source_identity"].get("source_heads", {}).items():
        actual = subprocess.check_output(["git", "-C", str(source_root / name), "rev-parse", "HEAD"], text=True).strip()
        if sha is None or actual != sha:
            raise ValueError("policy source identity differs from build: " + name)
    expected_env = json.loads((bundle / "decision_environment.json").read_text())
    if environment("gcc-13", expected_env["compiler_flags"]) != expected_env:
        raise ValueError("calibration toolchain/target differs from production build")
    complete_generation_universe(source)
    shutil.copyfile(bundle / "decision_environment.json", source / "numpy/_core/code_generators/crossjit_decision_environment.json")
    shutil.copyfile(bundle / "decision_db.json", source / "benchmarks/benchmarks/crossjit/results/g14/g14_decision_db.json")
    for phase in ("pf4_inner_contiguous", "pf5_scalar_zero_stride"):
        target = source / "benchmarks/benchmarks/crossjit/results/profiled" / phase / "execution_decision_db.json"
        data = json.loads(target.read_text())
        data["records"] = []
        save(target, data)
