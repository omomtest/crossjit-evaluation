"""Isolated RQ2–4/CS1 engine; the RQ1 execution module is unchanged."""
from pathlib import Path
import json
import random
import shlex
import statistics
import subprocess
import tempfile
import time

from common import ROOT, canonical_hash, clean_env, digest, read_json, runtime_environment, save, verify_identity


def spec_id(spec):
    return spec["name"] + "-" + canonical_hash(spec)[:12]


class RunProgress:
    """Count planned worker processes, including resumed and skipped cells."""

    def __init__(self, experiment, total):
        self.experiment = experiment
        self.total = total
        self.current = 0

    def next(self, spec, mode):
        self.current += 1
        runtime = "baseline" if mode == "baseline" else "crossjit"
        return f"{self.experiment}-{runtime}-{spec['name']}-({self.current}/{self.total})"

    def skip(self, spec, mode, trial, diagnostic=False):
        prefix = self.next(spec, mode)
        phase = "diagnostic" if diagnostic else "timing"
        print(f"{prefix} skipped: baseline reference failed; mode={mode} phase={phase} trial={trial}", flush=True)


def print_cell_result(prefix, cell, reference=False, diagnostic=False, cached=False):
    status = "cached (validated)" if cached else "completed"
    if reference or diagnostic:
        phase = "reference" if reference else "diagnostic"
        print(f"{prefix} {status}: {phase}; correct={cell['correct']}", flush=True)
        return
    samples = cell["runtime_ns"]
    print(f"{prefix} {status}: correct={cell['correct']} samples={len(samples)} "
          f"median={statistics.median(samples)/1e6:.6f} ms "
          f"mean={statistics.mean(samples)/1e6:.6f} ms "
          f"min={min(samples)/1e6:.6f} ms max={max(samples)/1e6:.6f} ms "
          f"first_call={cell['first_call_ns']/1e6:.6f} ms "
          f"process_wall={cell['process_wall_ns']/1e9:.3f} s", flush=True)


def variant_modes(cfg, variants=()):
    protocol = runtime_environment(cfg)
    result = {"baseline": dict(python=cfg["baseline_python"], crossjit=False, environment=dict(protocol)),
              "off": dict(python=cfg["candidate_python"], crossjit=False, environment=dict(protocol)),
              "on": dict(python=cfg["candidate_python"], crossjit=True, environment=dict(protocol))}
    for v in variants:
        result[v["name"]] = dict(python=v.get("python") or cfg["candidate_python"], crossjit=True,
                                 environment={**v.get("environment", {}), **protocol})
    return result


def verify_cell(cell, task, reference):
    if cell.get("task_hash") != task["task_hash"] or not cell.get("correct"):
        raise RuntimeError("cell identity/correctness mismatch")
    if cell.get("jit_stress", 0) != task.get("jit_stress", 0):
        raise RuntimeError("worker JIT stress setting differs from task")
    if len(cell["runtime_ns"]) != task["repeat"] or len(cell["warmup_runtime_ns"]) != task["warmup"]:
        raise RuntimeError("incomplete invocation samples")
    if not task["is_reference"] and cell.get("validated_invocations") != task["warmup"] + task["repeat"]:
        raise RuntimeError("incomplete independent output validation")
    if digest(reference) != cell["reference_sha256"]:
        raise RuntimeError("baseline reference drift")
    for path, expected in cell["input_hashes"].items():
        if digest(path) != expected:
            raise RuntimeError("saved input drift: " + path)


def run_cell(output, manifest, spec, mode, trial, diagnostic=False, reference=False, checkpoints=(), progress=None):
    cfg = manifest["configuration"]
    modes = variant_modes(cfg, manifest["variants"])
    settings = modes[mode]
    name = spec_id(spec)
    raw_dir = output / "raw"
    label = f"{name}.{mode}.t{trial}.{'diag' if diagnostic else 'timing'}"
    result_path = raw_dir / (label + ".json")
    ref = output / "references" / (name + ".json")
    task = dict(spec=spec, npbench=cfg["npbench"], mode=mode, trial=trial, seed=cfg["seed"],
                warmup=0 if reference else cfg["warmup"], repeat=1 if reference else cfg["repeat"],
                diagnostics=diagnostic, is_reference=reference, checkpoints=list(checkpoints),
                reference=str(ref), inputs=str(output / "inputs" / name),
                identity_hash=canonical_hash(manifest["identity"]), jit_stress=cfg.get("jit_stress", 0))
    task["task_hash"] = canonical_hash(task)
    verify_identity(manifest["identity"])
    progress = progress if progress is not None else RunProgress(manifest["experiment"], 1)
    prefix = progress.next(spec, mode)
    phase = "reference" if reference else "diagnostic" if diagnostic else "timing"
    detail = f"mode={mode} phase={phase} trial={trial} warmup={task['warmup']} repeat={task['repeat']}"
    if "preset" in spec:
        detail += f" preset={spec['preset']}"
    if spec.get("overrides"):
        detail += " parameters=" + json.dumps(spec["overrides"], sort_keys=True)
    print(f"{prefix} {detail}", flush=True)
    if result_path.exists():
        cell = read_json(result_path)
        verify_cell(cell, task, ref)
        print_cell_result(prefix, cell, reference, diagnostic, cached=True)
        return cell
    task_path = output / "tasks" / (label + ".json")
    save(task_path, task)
    raw_dir.mkdir(parents=True, exist_ok=True)
    log_path = output / "logs" / (label + ".log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    command = [settings["python"], "-B", str(ROOT / "scripts/redesign_worker.py"),
               "--task", str(task_path), "--output", str(result_path)]
    env = clean_env(cfg["blas_threads"], settings["crossjit"], diagnostic,
                    settings["environment"], cfg["cpu_affinity"])
    # Show only evaluation-controlled settings, never the inherited environment.
    if diagnostic:
        env["PYTHON_JIT_CROSSJIT_DEBUG"] = "1"
        env["NPY_CROSSJIT_EVALUATION_AUDIT"] = "1"
    env_keys = {"PYTHONDONTWRITEBYTECODE", "PYTHONHASHSEED", "PYTHON_JIT", "PYTHON_JIT_CROSSJIT",
                "OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                "NPY_CROSSJIT_DIAGNOSTICS", "CROSSJIT_EVALUATION_AFFINITY", *settings["environment"]}
    assignments = [f"{key}={env[key]}" for key in sorted(env_keys) if key in env]
    try:
        with tempfile.TemporaryDirectory(prefix="crossjit-measure-") as cwd, log_path.open("w") as log:
            print(f"{prefix} command: {shlex.join(['env', *assignments, *command])}", flush=True)
            print(f"{prefix} cwd={cwd} log={log_path}", flush=True)
            start = time.perf_counter_ns()
            subprocess.run(command, cwd=cwd, env=env,
                           stdout=log, stderr=subprocess.STDOUT, timeout=cfg["timeout"], check=True)
        cell = read_json(result_path)
        cell["process_wall_ns"] = time.perf_counter_ns() - start
        save(result_path, cell)
        verify_cell(cell, task, ref)
        verify_identity(manifest["identity"])
        print_cell_result(prefix, cell, reference, diagnostic)
        return cell
    except (subprocess.SubprocessError, AssertionError, RuntimeError) as exc:
        save(output / "failures" / (label + ".json"), dict(task=task, error=str(exc), log=str(log_path)))
        print(f"{prefix} failed: {exc}; log={log_path}", flush=True)
        return None


def run_matrix(output, manifest, modes=("baseline", "on"), diagnostics=True):
    specs, cfg = manifest["specs"], manifest["configuration"]
    rng = random.Random(cfg["seed"])
    cells = []
    per_trial = len(modes) + (sum(mode != "baseline" for mode in modes) if diagnostics else 0)
    progress = RunProgress(manifest["experiment"], len(specs) * (1 + cfg["trials"] * per_trial))
    for spec in specs:
        reference = run_cell(output, manifest, spec, "baseline", "reference", reference=True, progress=progress)
        for trial in range(cfg["trials"]):
            order = list(modes)
            if reference is not None:
                rng.shuffle(order)
            for mode in order:
                if reference is None:
                    progress.skip(spec, mode, trial)
                    continue
                cell = run_cell(output, manifest, spec, mode, trial, progress=progress)
                if cell:
                    cells.append(cell)
            if diagnostics:
                for mode in modes:
                    if mode != "baseline":
                        if reference is None:
                            progress.skip(spec, mode, trial, diagnostic=True)
                            continue
                        cell = run_cell(output, manifest, spec, mode, trial, diagnostic=True, progress=progress)
                        if cell:
                            cells.append(cell)
    return cells
