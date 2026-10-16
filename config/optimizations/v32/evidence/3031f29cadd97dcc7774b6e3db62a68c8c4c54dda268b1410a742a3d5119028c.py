#!/usr/bin/env python3
"""Prune new measured cells lacking stable NPBench benefit; preserve prior policy."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "crossjit-generator"))
from crossjit_generator.decision import ExecutionDecisionDB


def load(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("policy", "calibration-manifest", "attribution", "prior-selection", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    a = p.parse_args()
    evidence = load(a.attribution)
    manifest = evidence["manifest"]
    if manifest["run_mode"] != "development" or not manifest["python_jit_stress"] or manifest["trials"] < 10:
        raise ValueError("requires >=10 independent development stress pairs")
    for name, value in manifest["hashes"].items():
        if digest(name) != value:
            raise ValueError("workload evidence input drift: " + name)
    for filename in ("execution_decision_db.json", "production_selection.json", "execution_value_decisions.json"):
        if digest(a.policy / filename) not in manifest["hashes"].values():
            raise ValueError("candidate policy differs from frozen workload experiment: " + filename)
    rows = {r["workload"]: r for r in evidence["rows"]}
    if len(rows) != len(evidence["rows"]):
        raise ValueError("duplicate workload effect")
    accepted = {name for name, r in rows.items() if r["candidate_keep_supported"] and
                r["candidate_stability"]["stable"] and r["candidate_performance_direction"] == "positive" and
                all(r["candidate_on_generated_exec_hit"]) and not any(r["candidate_on_exec_miss"]) and
                not any(r["candidate_on_exec_error"])}
    plans = load(a.calibration_manifest)["plans"]
    cells = {(r["specialization"], r["outer_count"], r["inner_extent"]): set(r["workloads"]) for r in plans}
    database = load(a.policy / "execution_decision_db.json")
    selection = load(a.policy / "production_selection.json")
    decisions = load(a.policy / "execution_value_decisions.json")
    prior_selection = load(a.prior_selection)
    old_selected = {r["candidate_hash"] for r in prior_selection["records"] if r["production_artifact_decision"] == "SELECT"}
    old_records = {}
    for item in selection["preserved_prior_execution_dbs"]:
        if digest(item["path"]) != item["sha256"]:
            raise ValueError("prior DB drift")
        for r in load(item["path"])["records"]:
            if r["specialization_hash"] in old_records and old_records[r["specialization_hash"]] != r:
                raise ValueError("conflicting prior records")
            old_records[r["specialization_hash"]] = r
    records = {r["specialization_hash"]: r for r in database["records"]}
    policies = {r["candidate_hash"]: r for r in selection["records"]}
    withdrawn = []
    for d in decisions:
        if d["decision"] != "GENERATED":
            continue
        ident = d["identity"]
        key = ident["execution_specialization_hash"]
        names = cells[(key, ident["outer_count"], ident["inner_extent"])]
        if not names <= rows.keys():
            raise ValueError("missing workload effect for a new admitted cell")
        d["workload_effects"] = {n: rows[n] for n in sorted(names)}
        d["effect_evidence_scope"] = "development"
        if names <= accepted:
            continue
        record = records[key]
        policy = policies[record["artifact_hash"]]
        cell = {k: ident[k] for k in ("outer_count", "inner_extent", "scalar_class_mask")}
        if cell in policy["measured_iteration_cells"]:
            policy["measured_iteration_cells"].remove(cell)
        d.update(calibration_decision="GENERATED", decision="ORIGINAL", withheld_for_npbench_effect=True,
                 reason="No stable positive whole-workload stress effect; local calibration retained as historical evidence")
        withdrawn.append(dict(identity=ident, workloads=sorted(names)))
        size = cell["outer_count"] * cell["inner_extent"]
        remaining = policy["measured_iteration_cells"]
        if not any(c["outer_count"] * c["inner_extent"] == size for c in remaining):
            for point in record["profitability"]["points"]:
                if point["min_elements"] <= size <= point["max_elements"]:
                    if key in old_records and point in old_records[key]["profitability"]["points"]:
                        raise ValueError("cannot withdraw an old measured range")
                    point.update(decision="ORIGINAL", reason=d["reason"])
        if not remaining and policy["candidate_hash"] not in old_selected:
            policy["production_artifact_decision"] = "REJECT"
        record["artifact_decision"] = "AUTO_SELECTED" if any(
            pt["decision"] == "GENERATED" for pt in record["profitability"]["points"]) else "AUTO_REJECTED"
    for key, old in old_records.items():
        current = records[key]
        if any(current[field] != old[field] for field in ("signature", "specialization", "environment", "artifact_hash")) or any(
                point not in current["profitability"]["points"] for point in old["profitability"]["points"]):
            raise ValueError("prior identity or measured range was changed: " + key)
    for old in prior_selection["records"]:
        if old["candidate_hash"] in old_selected:
            current = policies[old["candidate_hash"]]
            if current["production_artifact_decision"] != "SELECT" or any(
                    c not in current.get("measured_iteration_cells", []) for c in old.get("measured_iteration_cells", [])):
                raise ValueError("prior selected cell was lost")
    ExecutionDecisionDB.from_data(database)
    selected = sum(r["production_artifact_decision"] == "SELECT" for r in selection["records"])
    selection["summary"] = dict(production_selected_artifact_count=selected,
                                production_rejected_artifact_count=len(selection["records"]) - selected)
    inputs = [a.attribution, a.calibration_manifest, a.prior_selection, Path(__file__)]
    selection["measurement_inputs"].extend(dict(path=str(f.resolve()), sha256=digest(f),
                                               kind="development-workload-selection") for f in inputs)
    a.output.mkdir(parents=True, exist_ok=True)
    result = dict(run_mode="development", python_jit_stress=True, formal_performance_gate_evaluated=False,
                  selected_artifacts=selected, retained_workloads=sorted(accepted), withdrawn_cells=withdrawn,
                  preserved_prior_records=len(old_records))
    for name, value in (("execution_decision_db.json", database), ("production_selection.json", selection),
                        ("execution_value_decisions.json", decisions), ("retention.json", result)):
        (a.output / name).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
