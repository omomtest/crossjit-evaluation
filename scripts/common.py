"""Portable, standard-library orchestration; measured workers use their own Python."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1


def read_json(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, prefix=".writing-", delete=False) as f:
        json.dump(value, f, indent=2, sort_keys=True, allow_nan=False)
        f.write("\n")
        temp = Path(f.name)
    temp.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def hash_tree(root, patterns=("*.py", "*.json")):
    root = Path(root)
    files = sorted({p for pattern in patterns for p in root.rglob(pattern)
                    if p.is_file() and "__pycache__" not in p.parts and ".git" not in p.parts})
    return canonical_hash({str(p.relative_to(root)): digest(p) for p in files})


def write_csv(path, rows):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with tempfile.NamedTemporaryFile(mode="w", newline="", dir=path.parent, prefix=".writing-", delete=False) as f:
        temporary = Path(f.name)
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    temporary.replace(path)


def clean_env(threads=6, crossjit=False, diagnostic=False, extra=None, affinity=None):
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("PYTHON_JIT", "NPY_CROSSJIT", "NUMPY_JIT", "CROSSJIT_"))
           and k not in ("PYTHONPATH", "PYTHONHOME", "LD_PRELOAD")}
    env.update(PYTHONDONTWRITEBYTECODE="1", PYTHONHASHSEED="0", PYTHON_JIT="1",
               PYTHON_JIT_CROSSJIT="1" if crossjit else "0",
               OPENBLAS_NUM_THREADS=str(threads), OMP_NUM_THREADS=str(threads), MKL_NUM_THREADS=str(threads))
    if diagnostic:
        env["NPY_CROSSJIT_DIAGNOSTICS"] = "1"
    if affinity:
        env["CROSSJIT_EVALUATION_AFFINITY"] = ",".join(map(str, affinity))
    env.update(extra or {})
    return env


def runtime_environment(cfg):
    """Explicit protocol settings survive cleanup; ambient JIT overrides do not."""
    return {"PYTHON_JIT_STRESS": str(cfg.get("jit_stress", 0))}


def apply_affinity():
    raw = os.environ.get("CROSSJIT_EVALUATION_AFFINITY")
    if raw:
        if not hasattr(os, "sched_setaffinity"):
            raise RuntimeError("CPU affinity is unavailable on this OS")
        requested = {int(x) for x in raw.split(",")}
        if not requested <= os.sched_getaffinity(0):
            raise ValueError("requested CPUs are outside the container/host affinity")
        os.sched_setaffinity(0, requested)


def parse_affinity(value):
    if value is None or isinstance(value, list):
        return value
    result = set()
    for segment in value.split(","):
        a, sep, b = segment.partition("-")
        result.update(range(int(a), int(b) + 1) if sep else [int(a)])
    if not result or min(result) < 0:
        raise ValueError("affinity must contain nonnegative CPU numbers")
    return sorted(result)


def parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", type=Path, default=ROOT / "config/experiment.json")
    p.add_argument("--jit-stress", "--python-jit-stress", type=int, choices=(0, 1),
                   help="Set PYTHON_JIT_STRESS for all measured workers (0=off, 1=on)")
    for key in ("warmup", "repeat", "trials", "blas-threads", "seed", "timeout", "memory-budget-mb"):
        p.add_argument("--" + key, type=int)
    for key in ("baseline-python", "candidate-python", "npbench", "source-root", "build-manifest", "warmup-protocol"):
        p.add_argument("--" + key)
    p.add_argument("--cpu-affinity")
    p.add_argument("--preset", choices=("S", "M", "L", "XL", "paper"))
    p.add_argument("--suite", type=Path, default=ROOT / "config/suite.json")
    p.add_argument("--workloads", nargs="+")
    p.add_argument("--output", type=Path, required=True)
    return p


def configuration(args, require_warmup=True):
    cfg = read_json(args.config)
    keys = ("warmup", "repeat", "trials", "blas_threads", "seed", "timeout", "memory_budget_mb", "baseline_python",
            "candidate_python", "npbench", "source_root", "build_manifest", "warmup_protocol", "cpu_affinity", "preset", "jit_stress")
    for key in keys:
        value = getattr(args, key, None)
        if value is not None:
            cfg[key] = value
    # An explicitly requested confirmed protocol selects its own W* unless the
    # caller also supplies --warmup; do not let the fixed default hide it.
    if getattr(args, "warmup_protocol", None) and getattr(args, "warmup", None) is None:
        cfg["warmup"] = None
    cfg["cpu_affinity"] = parse_affinity(cfg.get("cpu_affinity"))
    cfg.setdefault("jit_stress", 0)
    if type(cfg["jit_stress"]) is not int or cfg["jit_stress"] not in (0, 1):
        raise ValueError("jit_stress must be 0 or 1")
    for key in ("repeat", "trials", "blas_threads", "timeout", "memory_budget_mb"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if cfg.get("warmup") is not None and (not isinstance(cfg["warmup"], int) or cfg["warmup"] < 0):
        raise ValueError("warmup must be a nonnegative integer")
    if not isinstance(cfg["seed"],int) or not 0 <= cfg["seed"] < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    if cfg.get("warmup") is None and cfg.get("warmup_protocol"):
        protocol = read_json(cfg["warmup_protocol"])
        if protocol.get("state") != "confirmed" or protocol.get("W_star") is None:
            raise ValueError("warmup protocol has not been confirmed")
        cfg["warmup"] = protocol["W_star"]
        cfg["warmup_from_protocol"] = True
    if require_warmup and cfg.get("warmup") is None:
        raise ValueError("warmup is unset: supply --warmup or a confirmed --warmup-protocol")
    for key in ("baseline_python", "candidate_python", "npbench", "source_root", "build_manifest", "warmup_protocol"):
        if cfg.get(key):
            cfg[key] = str(Path(cfg[key]).expanduser().absolute())
    return cfg


def workload_specs(args, cfg):
    suite = read_json(args.suite)
    inventory = {row["name"]: row for row in suite["benchmarks"]}
    names = args.workloads or list(inventory)
    if len(set(names)) != len(names) or any(n not in inventory for n in names):
        raise ValueError("unknown or duplicate workload")
    return [dict(kind="npbench", name=n, preset=cfg["preset"], overrides={}) for n in names]


def system_info():
    def read(path):
        try:
            return Path(path).read_text().strip()
        except OSError:
            return None
    caches = []
    for path in sorted(Path("/sys/devices/system/cpu/cpu0/cache").glob("index*")):
        caches.append({k: read(path / k) for k in ("level", "type", "size", "shared_cpu_list")})
    return dict(platform=platform.platform(), machine=platform.machine(), kernel=platform.release(),
                cpu_info=read("/proc/cpuinfo"), memory_info=read("/proc/meminfo"), caches=caches,
                affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                os_release=read("/etc/os-release"),
                cgroup={name: read("/sys/fs/cgroup/" + name) for name in ("cpu.max", "cpuset.cpus.effective", "cpuset.mems.effective", "memory.max")},
                power_governor=read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
                temperatures={str(p): read(p) for p in Path("/sys/class/thermal").glob("thermal_zone*/temp")})


def probe_python(executable, cfg):
    with tempfile.TemporaryDirectory(prefix="crossjit-probe-") as directory:
        path = Path(directory) / "probe.json"
        subprocess.run([executable, "-B", str(ROOT / "scripts/worker.py"), "--probe", "--output", str(path)],
                       env=clean_env(cfg["blas_threads"], extra=runtime_environment(cfg), affinity=cfg["cpu_affinity"]), cwd=directory,
                       check=True, timeout=60, stdout=subprocess.DEVNULL)
        return read_json(path)


def identity(cfg, variants=()):
    probes = {"baseline": probe_python(cfg["baseline_python"], cfg), "candidate": probe_python(cfg["candidate_python"], cfg)}
    for variant in variants:
        if variant.get("python"):
            probes[variant["name"]] = probe_python(variant["python"], cfg)
    binaries = {}
    for probe in probes.values():
        for path in probe["binary_files"]:
            binaries[path] = digest(path)
    build_path = Path(cfg["build_manifest"]) if cfg.get("build_manifest") else None
    if build_path and build_path.is_file():
        binaries[str(build_path)] = digest(build_path)
    for variant in variants:
        if variant.get("build_manifest"):
            binaries[str(Path(variant["build_manifest"]).absolute())] = digest(variant["build_manifest"])
    heads = {}
    for repo in ("cpython", "numpy", "crossjit-generator", "npbench"):
        path = Path(cfg["source_root"]) / repo
        if path.is_dir():
            try:
                heads[repo] = subprocess.check_output(["git", "-c", "safe.directory="+str(path), "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            except subprocess.CalledProcessError:
                heads[repo] = None
    cpu = Path("/proc/cpuinfo")
    fingerprint = sorted({line.strip() for line in cpu.read_text().splitlines()
                          if line.split(":", 1)[0].strip() in ("model name", "vendor_id", "CPU implementer", "CPU architecture", "CPU part")}) if cpu.is_file() else [platform.processor()]
    controls = system_info()
    return dict(probes=probes, file_hashes=binaries, scripts_hash=hash_tree(ROOT / "scripts"),
                npbench_hash=hash_tree(cfg["npbench"]),
                npbench_root=cfg["npbench"], source_heads=heads,
                platform_identity=dict(machine=platform.machine(), cpu=fingerprint, kernel=platform.release(),
                                       effective_affinity=cfg["cpu_affinity"] or controls["affinity"],
                                       container_resources=controls["cgroup"]),
                suite_hash=digest(ROOT / "config/suite.json"), runtime_protocol=dict(jit_stress=cfg.get("jit_stress", 0)),
                build=read_json(build_path) if build_path and build_path.is_file() else None)


def verify_identity(frozen):
    for path, expected in frozen["file_hashes"].items():
        if digest(path) != expected:
            raise RuntimeError("artifact drift: " + path)
    if frozen["scripts_hash"] != hash_tree(ROOT / "scripts"):
        raise RuntimeError("evaluation scripts changed during experiment")
    if frozen["npbench_hash"] != hash_tree(frozen["npbench_root"]):
        raise RuntimeError("NPBench sources changed during experiment")


def freeze(output, cfg, experiment, specs, variants=(), extra=None):
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=True)
    frozen = identity(cfg, variants)
    if frozen["build"] and frozen["build"].get("calibration_capability_build") and experiment not in ("calibration", "semantic-validation", "smoke"):
        raise RuntimeError("a calibration capability image cannot run production performance; rebuild with a measured policy")
    if not all(probe["jit_enabled"] for probe in frozen["probes"].values()):
        raise RuntimeError("all measured interpreters must have the experimental JIT enabled")
    if not frozen["probes"]["candidate"]["provider_api"] or not frozen["probes"]["candidate"]["host_api"]:
        raise RuntimeError("candidate lacks the CrossJIT Host/Provider ABI")
    manifest = dict(schema_version=SCHEMA, experiment=experiment, configuration=cfg, specs=specs,
                    variants=list(variants), identity=frozen, system=system_info(), extra=extra or {},
                    timing_scope="complete Python workload call; copying and validation outside timer",
                    measurement_status="interim_explicit_warmup")
    build = frozen["build"] or {}
    receipt = build.get("calibration_receipt")
    if receipt:
        origin = receipt["source_identity"]["platform_identity"]
        current = frozen["platform_identity"]
        manifest["policy_mode"] = ("local_calibrated_policy" if all(origin.get(k) == current.get(k) for k in ("machine", "cpu", "kernel"))
                                   else "fixed_policy_migration")
        calibrated_stress = receipt["source_identity"].get("runtime_protocol", {}).get("jit_stress", 0)
        manifest["calibration_runtime_protocol_matches"] = calibrated_stress == cfg.get("jit_stress", 0)
        if not manifest["calibration_runtime_protocol_matches"]:
            manifest["policy_mode"] = "fixed_policy_migration"
    else:
        manifest["policy_mode"] = "fixed_policy_migration" if build.get("fixed_strategy_migration") else "unverified_native_or_calibration"
    protocol_path = cfg.get("warmup_protocol")
    if protocol_path:
        protocol = read_json(protocol_path)
        binding = canonical_hash(dict(identity=frozen, blas_threads=cfg["blas_threads"], affinity=cfg["cpu_affinity"],
                                      preset=cfg["preset"], repeat=cfg["repeat"]))
        if protocol.get("state") == "confirmed" and protocol.get("binding") == binding and cfg["warmup"] >= protocol["W_star"]:
            manifest["measurement_status"] = "confirmed_tested_grid_warmup"
        elif cfg.get("warmup_from_protocol"):
            raise ValueError("warmup protocol does not match this source/policy/platform/protocol")
    if manifest["policy_mode"] == "fixed_policy_migration":
        manifest["measurement_status"] = "fixed_policy_migration_"+manifest["measurement_status"]
    if cfg.get("jit_stress", 0):
        manifest["measurement_status"] = "jit_stress_" + manifest["measurement_status"]
    path = output / "manifest.json"
    if path.exists():
        previous = read_json(path)
        # Temperature and scheduler observations may vary on resume. Identities/protocol may not.
        comparison = {k: v for k, v in manifest.items() if k != "system"}
        if {k: v for k, v in previous.items() if k != "system"} != comparison:
            raise RuntimeError("output directory belongs to a different identity/configuration")
        manifest = previous
    else:
        save(path, manifest)
    return output, manifest


def cli(main):
    try:
        main()
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        raise SystemExit(str(exc)) from exc
