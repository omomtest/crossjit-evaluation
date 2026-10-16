#!/usr/bin/env python3
"""Build an isolated evaluation runtime, preserving installed RQ1 environments."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, default=ROOT.parent)
    p.add_argument("--dependencies-venv", type=Path)
    p.add_argument("--numpy-build", type=Path)
    p.add_argument("--jobs", type=int, default=8)
    args = p.parse_args()
    source = args.source_root.resolve()
    dependencies = (args.dependencies_venv or source / "venv").resolve()
    numpy_build = (args.numpy_build or source / "numpy/build-crossjit-native-20261004").resolve()
    if args.jobs < 1:
        p.error("jobs must be positive")
    options = json.loads((numpy_build / "meson-info/intro-buildoptions.json").read_text())
    options = {o["name"]: o["value"] for o in options if o["name"].startswith("crossjit-")}
    bundle = json.loads((ROOT / "config/optimizations/v32/bundle.json").read_text())
    policy = {}
    for name, rel in (("crossjit-affine-profile", "profile.json"),
                      ("crossjit-execution-policy", "execution_decision_db.json"),
                      ("crossjit-native-policy", "native_policy.json"),
                      ("crossjit-production-selection", "production_selection.json")):
        path = Path(options[name])
        actual = digest(path)
        if actual != bundle["files"][rel]["sha256"]:
            p.error("selected build is not the shared frozen v32 policy: " + name)
        policy[rel] = dict(path=str(path), sha256=actual)
    subprocess.run(["make", f"-j{args.jobs}", "python"], cwd=source / "cpython", check=True)
    env = dict(os.environ, CCACHE_DIR=str(ROOT / ".cache/redesign-ccache"), PYTHONDONTWRITEBYTECODE="1")
    subprocess.run([str(dependencies / "bin/ninja"), "-C", str(numpy_build), f"-j{args.jobs}"], env=env, check=True)
    runtime = ROOT / ".runtime"
    subprocess.run([str(source / "cpython/python"), "-B", "-m", "venv", "--without-pip", str(runtime)], check=True)
    sites = list((dependencies / "lib").glob("python*/site-packages"))
    if len(sites) != 1:
        p.error("dependency venv must have one site-packages directory")
    target = next((runtime / "lib").glob("python*/site-packages"))
    (target / "evaluation-dependencies.pth").write_text(str(sites[0]) + "\n")
    shutil.copytree(sites[0] / "numpy", target / "numpy", dirs_exist_ok=True)
    extensions = list((numpy_build / "numpy/_core").glob("_multiarray_umath*.so"))
    if len(extensions) != 1:
        p.error("NumPy build must contain one _multiarray_umath extension")
    shutil.copyfile(extensions[0], target / "numpy/_core" / extensions[0].name)
    # The installed dependency package supplies unchanged Python modules;
    # build-generated version/backend metadata must describe the new binary.
    for name in ("version.py", "__config__.py"):
        shutil.copyfile(numpy_build / "numpy" / name, target / "numpy" / name)
    manifest = dict(schema_version=1, purpose="RQ2–4/ADI isolated local runtime",
        frozen_optimization_policy=bundle, fixed_strategy_migration=True,
        policy_inputs=policy, policy_input_checks_passed=True,
        binary_sha256={str(source / "cpython/python"): digest(source / "cpython/python"),
                       str(target / "numpy/_core" / extensions[0].name): digest(extensions[0])},
        numpy_build_options=options,
        source_sha256={str(path): digest(path) for path in (
            source / "cpython/Python/crossjit.c", source / "numpy/numpy/_core/src/umath/jit_provider.c",
            source / "numpy/numpy/_core/src/umath/jit_native_stages.h", source / "numpy/numpy/_core/src/umath/jit_execution_audit.h",
            source / "crossjit-generator/crossjit_generator/execution/plan.c.in")})
    manifest_path = ROOT / ".runtime/build-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    cfg = json.loads((ROOT / "config/experiment.json").read_text())
    cfg.update(candidate_python=str(runtime / "bin/python"), baseline_python=str(runtime / "bin/python"),
               npbench=str(source / "npbench"), source_root=str(source), build_manifest=str(manifest_path))
    path = ROOT / "config/experiment.redesign-local.json"
    path.write_text(json.dumps(cfg, indent=2) + "\n")
    print("Prepared runtime/config:", path)


if __name__ == "__main__":
    main()
