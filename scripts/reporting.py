"""Summaries retain every workload and distinguish production timing from diagnostics."""
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from common import ROOT, read_json, save, write_csv
from execution import spec_id
from statistics_tools import bootstrap, gm, paired_summary, aggregate_pairs


def numeric_delta(before, after):
    return {key: value - before.get(key, 0) for key, value in after.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)}


def signature_key(record):
    from common import canonical_hash
    keys = ("canonical_signature_hash", "node_count", "nodes", "dtype_class", "refined_layout_class",
            "iteration_rank_class", "operand_access_classes", "outer_shape_class", "sample_value_class_mask")
    return canonical_hash({key: record[key] for key in keys if key in record})


def diagnostic_window(before, after, repeat):
    host = numeric_delta(before.get("host", {}), after.get("host", {}))
    provider = numeric_delta(before.get("provider", {}), after.get("provider", {}))
    previous = {signature_key(r): r for r in before.get("provider", {}).get("observed_signatures", [])}
    inventory = []
    generated_histogram = Counter()
    for record in after.get("provider", {}).get("observed_signatures", []):
        key = signature_key(record)
        old = previous.get(key, {})
        hits = record.get("generated_hit_count", 0) - old.get("generated_hit_count", 0)
        if hits > 0:
            generated_histogram[record["node_count"]] += hits
            inventory.append(dict(identity=key, backend="generated", hits=hits, node_count=record["node_count"], record=record))
    native_histogram = Counter()
    old_native = before.get("provider", {}).get("native_candidates", {})
    for key, record in after.get("provider", {}).get("native_candidates", {}).items():
        hits = record.get("execute", 0) - old_native.get(key, {}).get("execute", 0)
        if hits > 0:
            nodes = sum(part.startswith("n:") for part in key.split("|"))
            native_histogram[nodes] += hits
            inventory.append(dict(identity=key, backend="native", hits=hits, node_count=nodes, record=record))
    histogram = generated_histogram + native_histogram
    backend_hits = {name: provider.get(name + "_exec_hit", 0) for name in ("generated", "native", "mixed")}
    dropped = {key: value for key, value in after.get("provider", {}).items() if key.endswith("records_dropped") and value}
    return dict(host=host, provider=provider, actual_signatures=inventory, node_histogram=dict(histogram),
                backend_hits=backend_hits,
                generated_node_histogram=dict(generated_histogram), native_node_histogram=dict(native_histogram),
                generated_hit_per_invocation=backend_hits["generated"] / repeat,
                executed_hit_per_invocation=sum(backend_hits.values()) / repeat,
                generated_nodes_per_invocation=sum(int(n) * hits for n, hits in generated_histogram.items()) / repeat,
                executed_nodes_per_invocation=sum(int(n) * hits for n, hits in histogram.items()) / repeat,
                active=any(backend_hits.values()),
                diagnostics_complete=not dropped, dropped_records=dropped)


def summarize_matrix(directory, denominator="baseline", numerator="on"):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    cfg = manifest["configuration"]
    raw = [read_json(p) for p in sorted((directory / "raw").glob("*.json"))]
    cells = {(spec_id(c["spec"]), c["mode"], c["trial"], c["diagnostics"]): c
             for c in raw if c.get("correct") and c["trial"] != "reference"}
    rows = []
    full = True
    suite_ratios = [[] for _ in range(cfg["trials"])]
    for spec in manifest["specs"]:
        key = spec_id(spec)
        base, cand, diagnostics = [], [], []
        for trial in range(cfg["trials"]):
            left, right = cells.get((key, denominator, trial, False)), cells.get((key, numerator, trial, False))
            if left and right:
                base.append(left["median_ns"])
                cand.append(right["median_ns"])
                suite_ratios[trial].append(left["median_ns"] / right["median_ns"])
            diag = cells.get((key, numerator, trial, True))
            if diag:
                diagnostics.append(diagnostic_window(diag["preparation"], diag["final"], diag["repeat"]))
        row = dict(workload=spec["name"], spec=spec, comparison=f"{denominator}/{numerator}",
                   diagnostics=diagnostics, historical_stability=read_json(ROOT / "config/suite.json")["historical_stability"].get(spec["name"]),
                   active=any(d["active"] for d in diagnostics) if diagnostics else None,
                   active_every_diagnostic=all(d["active"] for d in diagnostics) if len(diagnostics) == cfg["trials"] else None,
                   activity_evidence="matched diagnostic processes; post-warmup measured window",
                   complete=len(base) == cfg["trials"])
        full = full and row["complete"]
        if base:
            row.update(paired_summary(base, cand, cfg))
        else:
            row.update(speedup=None, process_pairs=0, stability="missing", speedup_ci95=[None, None])
        rows.append(row)
    def aggregate(predicate):
        values = [r["speedup"] for r in rows if r.get("speedup") is not None and predicate(r)]
        return gm(values) if full else None
    suite_names = {r["name"] for r in read_json(ROOT / "config/suite.json")["benchmarks"]}
    full53 = (full and len(rows) == 53 and {r["workload"] for r in rows} == suite_names and
              all(r["spec"]["kind"] == "npbench" and not r["spec"].get("overrides") for r in rows))
    summary = dict(schema_version=1, experiment=manifest["experiment"], comparison=f"{denominator}/{numerator}",
                   measurement_status=manifest["measurement_status"], complete=full,
                   full_53_npbench=full53, requested_workloads=len(rows), rows=rows,
                   GM_requested=aggregate(lambda r: True),
                   GM_all=aggregate(lambda r: True) if full53 else None,
                   GM_stable=aggregate(lambda r: r["stability"] == "stable"),
                   GM_active=aggregate(lambda r: r["active"] is True),
                   GM_active_stable=aggregate(lambda r: r["active"] is True and r["stability"] == "stable"),
                   GM_inactive=aggregate(lambda r: r["active"] is False),
                   active_total=sum(r["active"] is True for r in rows), inactive_total=sum(r["active"] is False for r in rows),
                   platform_execution_confirmed=any(r["active"] is True for r in rows),
                   equivalent_performance_comparison_eligible=full and any(r["active"] is True for r in rows),
                   active_stable=sum(r["active"] is True and r["stability"] == "stable" for r in rows),
                   active_unstable=sum(r["active"] is True and r["stability"] == "unstable" for r in rows),
                   active_stability_unconfirmed=sum(r["active"] is True and r["stability"].startswith("unconfirmed") for r in rows),
                   suite_GM=[gm(v) if len(v) == len(rows) else None for v in suite_ratios],
                   worst_regressions=sorted([dict(workload=r["workload"], speedup=r["speedup"], ci=r["speedup_ci95"])
                                             for r in rows if r["speedup"] is not None], key=lambda r: r["speedup"])[:10])
    valid_gms = [v for v in summary["suite_GM"] if v is not None]
    summary["suite_GM_ci95"] = aggregate_pairs(rows, cfg["seed"])["ci95"]
    for name, predicate in (("GM_requested", lambda r: True), ("GM_all", lambda r: full53),
                            ("GM_active", lambda r: r["active"] is True),
                            ("GM_inactive", lambda r: r["active"] is False),
                            ("GM_stable", lambda r: r["stability"] == "stable"),
                            ("GM_active_stable", lambda r: r["active"] is True and r["stability"] == "stable")):
        summary[name + "_ci95"] = aggregate_pairs([r for r in rows if predicate(r)], cfg["seed"])["ci95"]
    label = "summary" if (denominator, numerator) == ("baseline", "on") else f"summary-{denominator}-{numerator}"
    save(directory / (label + ".json"), summary)
    write_csv(directory / (label + ".csv"), [{k: v for k, v in r.items() if k != "diagnostics"} for r in rows])
    return summary


def coverage_report(directory, costs=None):
    directory = Path(directory)
    manifest = read_json(directory / "manifest.json")
    cells = [read_json(p) for p in sorted((directory / "raw").glob("*.diag.json"))]
    rows = []
    host_stages = ("recognized_nodes", "formed_regions", "physical_proof_attempted", "physical_proof_validated", "host_regions_proven", "rewritten_regions", "published_regions")
    provider_stages = ("regions_prepare_called", "regions_provider_accepted", "guard_plans_lowered", "execution_plans_lowered", "regions_executed", "guard_hit", "guard_miss", "exec_hit", "exec_miss", "exec_error", "generated_exec_hit", "native_exec_hit", "mixed_exec_hit", "generic_fallback_hit", "internal_allocations", "internal_buffer_bytes", "internal_object_allocations", "packing_allocations", "packing_bytes", "output_allocations")
    for cell in cells:
        window = diagnostic_window(cell["preparation"], cell["final"], cell["repeat"])
        h, p = window["host"], window["provider"]
        # Formation generally happened during warmup; retain the lifetime funnel separately.
        final = cell["final"]
        blockers = {key: value for key, value in {**final["host"], **final["provider"]}.items()
                    if isinstance(value, int) and value and ("reject" in key or "miss" in key or "unsupported" in key)}
        histogram = window["node_histogram"]
        hits = sum(histogram.values())
        categories = dict(
            host_physical_proof=dict(attempted=final["host"].get("physical_proof_attempted"),
                                     validated=final["host"].get("physical_proof_validated")),
            provider_operation_or_contract={k:final["provider"].get(k) for k in
                                           ("regions_rejected_unsupported_node", "regions_rejected_unsupported_contract")},
            registry_match=dict(misses=final["provider"].get("generated_registry_miss"),
                                note="Registry miss is evidence of no matching entry, not a kernel-cost measurement."),
            guard_rejection=dict(misses=window["provider"].get("guard_miss")),
            mixed_admission=dict(unprofitable=final["provider"].get("regions_rejected_unprofitable"),
                                 other=final["provider"].get("regions_rejected_other"),
                                 rejection_histogram=final["provider"].get("rejection_histogram")),
            generator_capability=None, unmeasured_cell_or_value_class=None, measured_negative_profitability=None,
            activation=dict(timed_generated_hits=window["provider"].get("generated_exec_hit"),
                            timed_native_hits=window["provider"].get("native_exec_hit"),
                            timed_mixed_hits=window["provider"].get("mixed_exec_hit"),
                            rewrites_during_window=window["host"].get("rewritten_regions"),
                            note="No HIT alone does not establish an activation blocker."))
        rows.append(dict(workload=cell["spec"]["name"], spec=cell["spec"], trial=cell["trial"], mode=cell["mode"],
                         active=window["active"], lifetime_funnel=dict(host={k: final["host"].get(k) for k in host_stages},
                                                                    provider={k: final["provider"].get(k) for k in provider_stages}),
                         timed_window=window, actual_max_nodes=max(map(int, histogram), default=None),
                         actual_weighted_mean_nodes=sum(int(n)*count for n, count in histogram.items())/hits if hits else None,
                         blocker_counters=blockers, blocker_categories=categories,
                         classification_scope="Evidence groups may overlap; unavailable primary causes remain null.",
                         candidates=final["provider"].get("candidates", []),
                         unsupported_nodes=final["provider"].get("unsupported_nodes", []),
                         blocker_classification="raw evidence; unprofitable is not automatically kernel cost",
                         eliminated_array_estimate=sum(max(0, r["node_count"]-1)*r["hits"] for r in window["actual_signatures"] if r["backend"] == "generated"),
                         estimate_assumption="generated single-output elementwise DAG only; native stage intermediates are not assumed eliminated; not measured traffic"))
    report = dict(schema_version=1, rows=rows, complete_cells=len(rows),
                  expected_cells=len(manifest["specs"])*manifest["configuration"]["trials"],
                  cost_coverage=None, cost_coverage_status="missing_verified_nonoverlapping_opportunity_costs")
    if costs:
        from common import canonical_hash
        data = read_json(costs)
        if data.get("identity_hash") != canonical_hash(manifest["identity"]) or data.get("overlap_policy") != "disjoint":
            raise ValueError("cost universe must be identity-bound and explicitly disjoint")
        entries = data["opportunities"]
        if len({x["id"] for x in entries}) != len(entries):
            raise ValueError("duplicate cost opportunities")
        available = {r["identity"] for row in rows for r in row["timed_window"]["actual_signatures"] if r["backend"] == "generated"}
        if any(x["original_cost_ns"] <= 0 or x["opportunity_count"] < x["generated_hits"] or
               x["generated_hits"] < 0 or (x["generated_hits"] and x["signature_identity"] not in available) for x in entries):
            raise ValueError("invalid or unobserved opportunity cost record")
        total = sum(x["opportunity_count"]*x["original_cost_ns"] for x in entries)
        report["cost_coverage"] = sum(x["generated_hits"]*x["original_cost_ns"] for x in entries)/total if total else None
        report["cost_coverage_status"] = "candidate_original_path_cost_fraction; not total-runtime coverage"
    save(directory / "coverage.json", report)
    write_csv(directory / "coverage.csv", rows)
    return report
