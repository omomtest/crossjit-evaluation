#!/usr/bin/env python3
"""Independent machine calibration; writes a policy bundle and never rebuilds implicitly."""
from collections import defaultdict
from datetime import datetime, timezone
import itertools
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

from common import ROOT, canonical_hash, clean_env, cli, configuration, digest, freeze, parser, read_json, runtime_environment, save
from statistics_tools import cv


def normalize_metadata(metadata):
    kernels = []
    for original in metadata["kernels"]:
        kernel = dict(original)
        # Legacy flat artifacts serialize their descriptor inside the execution
        # specialization. Use that exact committed descriptor, never invent one.
        if "iteration_descriptor" not in kernel:
            kernel["iteration_descriptor"] = kernel["execution_specialization"]["iteration_descriptor"]
        kernels.append(kernel)
    return dict(metadata, kernels=kernels)


def measure(output, manifest, metadata, settings, harness, specializations):
    cfg = manifest["configuration"]
    kernel_by_id = {k["execution_specialization_hash"]: k for k in metadata["kernels"]}
    plans = []
    for key in specializations:
        case = kernel_by_id[key]
        scalars = any(op["access_kind"] == "SCALAR" for op in case["iteration_descriptor"]["operands"])
        strided = any(op["access_kind"] in ("FIXED_STRIDE", "GENERAL_STRIDED") for op in case["iteration_descriptor"]["operands"])
        for shape, scalar, stride in itertools.product(settings["shapes"], settings["scalar_kinds"] if scalars else ["numpy"],
                                                       settings["affine_strides"] if strided else [2]):
            outer, inner = map(int, shape.split("x"))
            if case["iteration_descriptor"]["iteration_rank_class"] == "FLAT" and outer != 1:
                continue
            if case["iteration_descriptor"]["iteration_rank_class"] == "OUTER_INNER" and outer == 1:
                continue
            plans.append(dict(specialization=key, shape=shape, scalar=scalar, stride=stride))
    save(output / "calibration-grid.json", dict(plans=plans, settings=settings, timing="complete Provider tuning Python loop; control reported separately"))
    start = time.monotonic()
    rows, failed = [], []
    rng = random.Random(settings["seed"])
    for number, plan in enumerate(plans):
        key = canonical_hash(plan)[:16]
        values = defaultdict(list)
        evidence = []
        complete = True
        schedule = [(trial, mode) for trial in range(settings["processes"]) for mode in ("baseline", "generated", "generic")]
        rng.shuffle(schedule)
        for trial, mode in schedule:
            path = output / "raw" / f"{key}.{mode}.t{trial}.json"
            if time.monotonic()-start > settings["budget_seconds"]:
                complete = False
                failed.append(dict(plan=plan, reason="predeclared budget exhausted"))
                break
            argv = [cfg["candidate_python"], "-B", str(ROOT / "scripts/calibration_worker.py"),
                    "--provider-harness", str(harness), "--worker", "--mode", mode,
                    "--generated-metadata", str(output / "metadata.json"), "--specializations", plan["specialization"],
                    "--shapes", plan["shape"], "--scalar-kind", plan["scalar"], "--affine-stride", str(plan["stride"])]
            for arg in ("samples", "target_elements", "min_iterations", "max_iterations"):
                argv += ["--"+arg.replace("_", "-"), str(settings[arg])]
            env = clean_env(1, mode != "baseline", extra=runtime_environment(cfg), affinity=cfg["cpu_affinity"])
            if mode != "baseline":
                env.update(NPY_CROSSJIT_MIN_WORK="0", NPY_CROSSJIT_PROFILED_FORCE_GENERATED="1")
                if mode == "generic":
                    env["NPY_CROSSJIT_DIAGNOSTIC_DISABLE_GENERATED"] = "1"
            try:
                if path.is_file():
                    data = read_json(path)
                else:
                    print(f"calibration {number+1}/{len(plans)} {mode} trial={trial}", flush=True)
                    remaining = settings["budget_seconds"] - (time.monotonic()-start)
                    completed = subprocess.run(argv, env=env, check=True, timeout=min(cfg["timeout"], max(.01, remaining)), capture_output=True, text=True)
                    import json
                    records = json.loads(completed.stdout)
                    if len(records) != 1:
                        raise ValueError("expected one independently checked descriptor cell")
                    data = dict(plan=plan, trial=trial, mode=mode, record=records[0])
                    save(path, data)
                if data["plan"] != plan or not data["record"]["correct"]:
                    raise ValueError("calibration resume identity/correctness mismatch")
                record = data["record"]
                # Provider harness already asserts full topology, path HIT and no replay/error.
                values[mode].append(record["process_median_ns"])
                evidence.append(record)
            except (subprocess.SubprocessError, ValueError) as exc:
                complete = False
                failure = dict(plan=plan, mode=mode, trial=trial, reason=str(exc),
                               stderr=getattr(exc, "stderr", None))
                save(output / "failures" / (key+f".{mode}.{trial}.json"), failure)
                failed.append(failure)
                break
        if complete:
            medians = {mode: statistics.median(v) for mode, v in values.items()}
            noise = max([*(cv(v) or 0 for v in values.values()), *(r["process_cv"] for r in evidence)])
            observed_classes = {r["observed_scalar_class_mask"] for r in evidence if r["mode"] != "baseline"}
            profitable = (noise <= settings["max_cv"] and medians["baseline"]/medians["generated"] >= settings["speedup_threshold"]
                          and medians["generated"] < medians["generic"] and len(observed_classes) == 1)
            rows.append(dict(plan=plan, elements=int(plan["shape"].split("x")[0])*int(plan["shape"].split("x")[1]),
                             baseline_ns=medians["baseline"], generated_ns=medians["generated"], generic_ns=medians["generic"],
                             noise=noise, profitable=profitable, scalar_class_mask=next(iter(observed_classes)) if len(observed_classes) == 1 else None,
                             independent_processes=values))
    return rows, failed, plans


def policy_bundle(output, manifest, metadata, rows, settings, environment, profile, plans):
    cfg = manifest["configuration"]
    sys.path.insert(0, str(Path(cfg["source_root"]) / "crossjit-generator"))
    import crossjit_generator
    from crossjit_generator.backend.c import CBackend
    from crossjit_generator.candidate import ExecutionSpecialization
    from crossjit_generator.decision import (ArtifactDecision, DecisionDB, DecisionEnvironment, DecisionRecord,
        ExecutionDecisionDB, ExecutionDecisionRecord, MeasurementPoint, MeasurementProvenance, ProfitabilityEnvelope, RuntimeDecision)
    from crossjit_generator.domains.elementwise import ElementwiseOptimizer
    from crossjit_generator.signature import OptimizationSignature
    provenance = MeasurementProvenance("generic_signature_microbenchmark", settings["processes"], settings["samples"], datetime.now(timezone.utc).isoformat())
    records, structural, selections = [], {}, []
    for kernel in metadata["kernels"]:
        signature = OptimizationSignature.from_data(kernel["optimization_signature"])
        specialization = ExecutionSpecialization.from_data(kernel["execution_specialization"])
        env = DecisionEnvironment(provider=signature.provider, provider_abi_version=signature.provider_abi_version,
            recipe_hashes=tuple(op.recipe_hash for op in signature.operations), generator_version=crossjit_generator.__version__,
            optimizer_version=ElementwiseOptimizer.VERSION,
            backend_version=CBackend.AFFINE_VERSION if kernel["entry_abi"] == "affine_v2" else CBackend.VERSION,
            **environment)
        case_rows = [r for r in rows if r["plan"]["specialization"] == kernel["execution_specialization_hash"]]
        groups = defaultdict(list)
        for row in case_rows:
            groups[row["elements"]].append(row)
        points, cells = [], []
        for elements, samples in sorted(groups.items()):
            # A class/stride with partial or negative evidence never inherits a favorable sibling cell.
            original = min(r["baseline_ns"] for r in samples)
            generated = max(r["generated_ns"] for r in samples)
            generic = min(r["generic_ns"] for r in samples)
            noise = max(r["noise"] for r in samples)
            expected = sum(p["specialization"] == kernel["execution_specialization_hash"] and
                           int(p["shape"].split("x")[0])*int(p["shape"].split("x")[1]) == elements for p in plans)
            profitable = len(samples) == expected and all(r["profitable"] for r in samples)
            points.append(MeasurementPoint(elements, elements, original, generated, generic, original/generated,
                original/generic, noise, max(0, 1-noise/settings["max_cv"]),
                RuntimeDecision.GENERATED if profitable else RuntimeDecision.ORIGINAL,
                "conservative complete-path measurements; no interpolation to unmeasured sizes"))
            if profitable:
                for row in samples:
                    outer, inner = map(int, row["plan"]["shape"].split("x"))
                    cells.append(dict(outer_count=outer, inner_extent=inner, scalar_class_mask=row["scalar_class_mask"]))
        cells = list({canonical_hash(c): c for c in cells}.values())
        selected = bool(cells) and len(cells) <= 64
        if points:
            record = ExecutionDecisionRecord(signature, specialization, env,
                ArtifactDecision.AUTO_SELECTED if selected else ArtifactDecision.AUTO_REJECTED,
                provenance, ProfitabilityEnvelope(tuple(points)), kernel["candidate_hash"],
                min(p.confidence for p in points), "local calibration; unmeasured descriptor cells remain rejected")
            records.append(record)
            if kernel["entry_abi"] != "affine_v2":
                if signature.stable_hash() not in structural or selected:
                    structural[signature.stable_hash()] = DecisionRecord(signature, env, record.artifact_decision,
                        provenance, record.profitability, 0, 0, 0, kernel["candidate_hash"], record.confidence, record.reason)
        selections.append(dict(candidate_hash=kernel["candidate_hash"], production_artifact_decision="SELECT" if selected else "REJECT",
                               measured_iteration_cells=cells if selected else [], value_class_policy_version=1))
    bundle = output / "policy"
    save(bundle / "profile.json", profile)
    save(bundle / "decision_environment.json", environment)
    save(bundle / "decision_db.json", DecisionDB(structural.values()).to_data())
    database = ExecutionDecisionDB(records, profile_identity_hash=canonical_hash(profile),
                                    candidate_universe_hash=canonical_hash(sorted(k["execution_specialization_hash"] for k in metadata["kernels"])))
    save(bundle / "execution_decision_db.json", database.to_data())
    import hashlib
    universe = hashlib.sha256("\n".join(sorted({k["candidate_hash"] for k in metadata["kernels"]})).encode()).hexdigest()
    save(bundle / "production_selection.json", dict(schema_version=2, policy_name="CROSSJIT-V3-AFFINE-V2",
        runtime_profitability_retuned=True, candidate_universe_hash=universe, records=selections,
        measurement_inputs=[dict(path="../measurements.json", sha256=digest(output / "measurements.json"))],
        summary=dict(production_selected_artifact_count=sum(r["production_artifact_decision"] == "SELECT" for r in selections),
                     production_rejected_artifact_count=sum(r["production_artifact_decision"] == "REJECT" for r in selections))))
    save(bundle / "receipt.json", dict(schema_version=1, source_identity=manifest["identity"],
         file_hashes={p.name: digest(p) for p in bundle.glob("*.json") if p.name != "receipt.json"},
         selected=sum(r["production_artifact_decision"] == "SELECT" for r in selections),
         calibration_forcing_is_not_production_admission=True))


def main():
    p = parser(__doc__)
    p.add_argument("--calibration-config", type=Path, default=ROOT / "config/calibration.json")
    p.add_argument("--metadata", type=Path, required=True, help="unpruned metadata from the calibration image")
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--specializations", nargs="+", help="subset sanity check only; remaining artifacts are REJECT")
    p.add_argument("--compiler", default="gcc-13")
    args = p.parse_args()
    cfg = configuration(args, require_warmup=False)
    cfg.update(warmup=0, warmup_protocol=None, blas_threads=1)
    metadata, settings, profile = normalize_metadata(read_json(args.metadata)), read_json(args.calibration_config), read_json(args.profile)
    if settings["processes"] < 3 or settings["samples"] < 1 or settings["max_cv"] <= 0 or settings["budget_seconds"] <= 0:
        raise ValueError("invalid calibration processes/samples/noise/budget")
    harness = Path(cfg["source_root"]) / "numpy/benchmarks/benchmarks/crossjit/run_profiled_execution_tuning.py"
    sys.path.insert(0, str(ROOT / "build"))
    from decision_environment import environment as compiler_environment
    environment = compiler_environment(args.compiler, settings["compiler_flags"])
    extra = dict(metadata_sha256=digest(args.metadata), profile_sha256=digest(args.profile),
                 harness_sha256=digest(harness), settings=settings, environment=environment,
                 selected_specializations=args.specializations, diagnostic_calibration=True)
    output, manifest = freeze(args.output, cfg, "calibration", [], extra=extra)
    save(output / "metadata.json", metadata)
    choices = args.specializations or [k["execution_specialization_hash"] for k in metadata["kernels"]]
    if not set(choices) <= {k["execution_specialization_hash"] for k in metadata["kernels"]}:
        raise ValueError("unknown calibration specialization")
    rows, failures, plans = measure(output, manifest, metadata, settings, harness, choices)
    save(output / "measurements.json", dict(rows=rows, failures=failures, expected_cells=len(plans),
                                            complete=len(rows)==len(plans), source_identity=manifest["identity"]))
    policy_bundle(output, manifest, metadata, rows, settings, environment, profile, plans)
    if len(rows) != len(plans):
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
