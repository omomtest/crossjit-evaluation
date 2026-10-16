"""Scope/policy preflight, leaving RQ1's common configuration untouched."""
from pathlib import Path
import shutil

from common import ROOT, canonical_hash, digest, probe_python, read_json, save


SINGLE = dict(name="single", mechanism="Host candidate node limit",
              environment={"PYTHON_JIT_CROSSJIT_MAX_REGION_NODES": "1"},
              admission="unchanged production legality, validation and profitability")


def same_binary(cfg, require_single=False):
    # OFF in the integrated candidate is also the independent reference process.
    cfg = dict(cfg, baseline_python=cfg["candidate_python"])
    if require_single:
        probe = probe_python(cfg["candidate_python"], cfg)
        symbol = b"PYTHON_JIT_CROSSJIT_MAX_REGION_NODES"
        if not any(symbol in Path(p).read_bytes() for p in probe["binary_files"]):
            raise RuntimeError("candidate binary lacks the Host SINGLE switch; rebuild CPython")
    return cfg


def archive_policy(output, policy=None, manifest=None):
    """Archive immutable evidence, including measured negatives and unmeasured cells."""
    policy = Path(policy or ROOT / "config/optimizations/v32")
    bundle = read_json(policy / "bundle.json") if (policy / "bundle.json").is_file() else None
    if bundle:
        for rel, entry in bundle["files"].items():
            if digest(policy / rel) != entry["sha256"]:
                raise RuntimeError("policy evidence drift: " + rel)
    target = output / "policy-evidence"
    files = {}
    for path in sorted({*policy.rglob("*.json"), *policy.rglob("*.py")}):
        rel = path.relative_to(policy)
        files[str(rel)] = digest(path)
        destination = target / rel
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and digest(destination) != files[str(rel)]:
            raise RuntimeError("archived policy drift: " + str(rel))
        shutil.copyfile(path, destination)
    generated = read_json(policy / "execution_decision_db.json") if (policy / "execution_decision_db.json").is_file() else {}
    native = read_json(policy / "native_policy.json") if (policy / "native_policy.json").is_file() else {}
    build = (manifest["identity"].get("build") or {}) if manifest else {}
    declared = build.get("frozen_optimization_policy", {}).get("files", {})
    verified = bool(declared) and all(files.get(rel) == item["sha256"] for rel, item in declared.items())
    if declared and not verified:
        raise RuntimeError("policy archive does not match the selected build manifest")
    save(output / "admission-evidence.json", dict(
        source=str(policy), files=files, identity=canonical_hash(files),
        generated_records=generated.get("records", []), native_records=native.get("records", []),
        scope="Frozen policy/evidence archive; policy-to-selected-binary binding requires a matching build manifest.",
        costs_scope="Complete original/candidate costs and CI retained from supplied measurements; unmeasured is not measured-negative.",
        policy_verified_against_binary=verified))


def admission_inventory(output, reach):
    evidence = read_json(output / "admission-evidence.json")
    decisions = {r["key"]: r for r in evidence["native_records"]}
    rows = []
    for row in reach["rows"]:
        if row["mode"] not in ("single", "on"):
            continue
        final = row["lifetime"]["provider"]
        for key, observation in final.get("native_candidates", {}).items():
            decision = decisions.get(key)
            status = ("accepted" if decision["decision"] == "ACCEPT" else "measured_negative") if decision else "unmeasured"
            rows.append(dict(workload=row["workload"], mode=row["mode"], trial=row["trial"],
                namespace="native_measured_key", identity=key, admission=status,
                observation=observation, frozen_evidence=decision,
                binding_verified=evidence["policy_verified_against_binary"]))
        for observation in final.get("observed_signatures", []):
            # Without an exported profitability cell key, do not relabel every
            # unsuccessful preparation as measured-negative.
            rows.append(dict(workload=row["workload"], mode=row["mode"], trial=row["trial"],
                namespace="generated_normalized_signature", observation=observation,
                admission="accepted" if observation.get("accepted_count", 0) else "unresolved_without_cost_cell_key",
                frozen_evidence=None))
    save(output / "admission-inventory.json", dict(rows=rows,
        scope="Candidate preparation observations; separate from timed-window HIT inventory.",
        limitations="Native decisions are joined by exact measured key. Generated cost-cell identity is not exported; its full frozen evidence is archived separately."))
