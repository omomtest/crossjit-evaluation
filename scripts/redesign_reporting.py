"""Reach/boundaries and three-mode paired-process reporting for the redesign."""
from collections import Counter
from pathlib import Path
import random
import statistics

from common import canonical_hash, read_json, save, write_csv
from reporting import diagnostic_window
from redesign_execution import spec_id
from statistics_tools import gm, paired_summary, percentile


def observed_identity(record):
    # Match the Provider's actual observed_signature_equal predicate. The
    # legacy report expects operand_access_classes, but this ABI exports
    # access_kinds/scalar_mask; omitting them merges distinct execution records.
    keys = ("input_count", "node_count", "nodes", "dtype_class", "layout_class",
            "refined_layout_class", "iteration_rank_class", "iteration_descriptor_version",
            "outer_shape_class", "scalar_mask", "access_kinds")
    return canonical_hash({k: record[k] for k in keys if k in record})


def region_window(before, after, repeat):
    window = diagnostic_window(before, after, repeat)
    previous = {observed_identity(r): r for r in before.get("provider", {}).get("observed_signatures", [])}
    generated = []
    histogram = Counter()
    for record in after.get("provider", {}).get("observed_signatures", []):
        key = observed_identity(record)
        hits = record.get("generated_hit_count", 0) - previous.get(key, {}).get("generated_hit_count", 0)
        if hits > 0:
            histogram[record["node_count"]] += hits
            generated.append(dict(identity=key, backend="generated", hits=hits, node_count=record["node_count"], record=record))
    window["actual_signatures"] = generated + [r for r in window["actual_signatures"] if r["backend"] != "generated"]
    window["generated_node_histogram"] = dict(histogram)
    combined = histogram + Counter({int(n): h for n, h in window["native_node_histogram"].items()})
    window["node_histogram"] = dict(combined)
    window["generated_nodes_per_invocation"] = sum(n*h for n, h in histogram.items()) / repeat
    window["executed_nodes_per_invocation"] = sum(n*h for n, h in combined.items()) / repeat
    return window


def classify(windows):
    if any(any(int(n) >= 2 and hits > 0 for n, hits in w["node_histogram"].items()) for w in windows):
        return "multi_node"
    if any(w["active"] for w in windows):
        return "single_node"
    return "inactive"


def reach_report(output, modes=("on",)):
    manifest = read_json(output / "manifest.json")
    cfg = manifest["configuration"]
    rows, violations = [], []
    for spec in manifest["specs"]:
        for mode in modes:
            for trial in range(cfg["trials"]):
                path = output / "raw" / f"{spec_id(spec)}.{mode}.t{trial}.diag.json"
                if not path.is_file():
                    violations.append(dict(workload=spec["name"], mode=mode, trial=trial, reason="missing diagnostic"))
                    continue
                cell = read_json(path)
                if not cell.get("correct"):
                    violations.append(dict(workload=spec["name"], mode=mode, trial=trial, reason="incorrect diagnostic"))
                    continue
                window = region_window(cell["preparation"], cell["final"], cell["repeat"])
                for signature in window["actual_signatures"]:
                    if signature["backend"] == "generated":
                        signature["lowering_path"] = "typed_generated_dag"
                    else:
                        entry = cell["final"]["provider"].get("native_candidates", {}).get(signature["identity"], {})
                        kinds = {s["kind"] for s in entry.get("execution_description", {}).get("stages", [])}
                        signature["lowering_path"] = ("mixed_stages" if kinds == {0, 1} else
                            "generated_stages" if kinds == {0} else "backend_stages" if kinds else "native_elementwise")
                hist = window["node_histogram"]
                single = sum(h for n, h in hist.items() if int(n) == 1)
                multi = sum(h for n, h in hist.items() if int(n) >= 2)
                tracked = sum(hist.values())
                total = window["provider"].get("exec_hit", 0)
                complete_inventory = window["diagnostics_complete"] and tracked == total
                if mode == "single" and (multi or not complete_inventory):
                    violations.append(dict(workload=spec["name"], mode=mode, trial=trial,
                                           reason="SINGLE has multi-node HIT or incomplete node inventory"))
                boundaries = {side: {k: v for k, v in cell["final"][side].items()
                                    if isinstance(v, int) and v and any(t in k for t in ("reject", "miss", "unsupported"))}
                              for side in ("host", "provider")}
                # A counter cannot identify the first failed candidate or a parent retry.
                # Keep layers separate until an identity-linked event ABI exists.
                rows.append(dict(workload=spec["name"], spec=spec, mode=mode, trial=trial,
                    group=classify([window]), timed_window=window,
                    single_node_hits=single, multi_node_hits=multi,
                    max_executed_nodes=max(map(int, hist), default=0),
                    timed_outcomes={k: window["provider"].get(k, 0) for k in
                                    ("exec_hit", "exec_miss", "exec_error", "guard_hit", "guard_miss", "guard_error")},
                    inventory_complete=complete_inventory, unassociated_hits=total-tracked,
                    construction=dict(initial=cell["initial"], post_warmup=cell["preparation"]),
                    lifetime=cell["final"], boundary_counters=boundaries,
                    decisive_failure=None, failure_status="unexplained_without_candidate_trace_link",
                    trace_log=str(output / "logs" / path.with_suffix(".log").name),
                    actual_signatures=window["actual_signatures"],
                    source_site_count=None,
                    eliminated_materializations_estimate=sum(max(0, r["node_count"]-1)*r["hits"]
                        for r in window["actual_signatures"] if r["backend"] == "generated"),
                    verified_eliminated_objects=None,
                    allocations={k: window["provider"].get(k) for k in
                                 ("internal_allocations", "internal_buffer_bytes", "internal_object_allocations",
                                  "packing_allocations", "packing_bytes", "output_allocations")}))
    full_groups = {}
    for spec in manifest["specs"]:
        windows = [r["timed_window"] for r in rows if r["workload"] == spec["name"] and r["mode"] == "on"]
        full_groups[spec["name"]] = classify(windows) if len(windows) == cfg["trials"] else "missing"
    report = dict(schema_version=2, rows=rows, full_groups=full_groups,
                  group_counts=dict(Counter(full_groups.values())), violations=violations,
                  complete_cells=len(rows), expected_cells=len(manifest["specs"])*len(modes)*cfg["trials"],
                  complete=not violations,
                  limitations=["Counters are stratified observations, not a unified opportunity funnel.",
                               "No candidate/trace/retry identity ABI: decisive blockers remain unexplained.",
                               "Generated (nodes-1)*HIT is a structural estimate, not saved time or native buffers.",
                               "Fresh final output is retained; allocation bytes are cumulative, not peak storage."])
    save(output / "reach.json", report)
    write_csv(output / "reach.csv", [{k: v for k, v in r.items() if k not in ("lifetime", "construction")} for r in rows])
    return report


from statistics_tools import aggregate_pairs


def composition_report(output, reach):
    manifest = read_json(output / "manifest.json")
    cfg = manifest["configuration"]
    comparisons = []
    for left, right in (("off", "single"), ("single", "on"), ("off", "on")):
        rows = []
        for spec in manifest["specs"]:
            a, b = [], []
            for trial in range(cfg["trials"]):
                paths = [output / "raw" / f"{spec_id(spec)}.{mode}.t{trial}.timing.json" for mode in (left, right)]
                if all(p.is_file() for p in paths):
                    cells = [read_json(p) for p in paths]
                    if all(c.get("correct") for c in cells):
                        a.append(cells[0]["median_ns"])
                        b.append(cells[1]["median_ns"])
            row = dict(workload=spec["name"], group=reach["full_groups"][spec["name"]],
                       complete=len(a) == cfg["trials"])
            row.update(paired_summary(a, b, cfg) if a else dict(process_ratios=[], speedup=None, speedup_ci95=[None, None]))
            rows.append(row)
        comparisons.append(dict(comparison=f"{left}/{right}", rows=rows,
            all_workloads=aggregate_pairs(rows, cfg["seed"]),
            groups={g: aggregate_pairs([r for r in rows if r["group"] == g], cfg["seed"])
                    for g in ("multi_node", "single_node", "inactive")}))
    suite = read_json(Path(__file__).resolve().parents[1] / "config/suite.json")
    report = dict(schema_version=2, comparisons=comparisons,
                  full_53_npbench={s["name"] for s in manifest["specs"]} == {s["name"] for s in suite["benchmarks"]}
                                  and len(manifest["specs"]) == 53,
                  complete=reach["complete"] and all(r["complete"] for c in comparisons for r in c["rows"]),
                  interpretation="SINGLE/FULL is the joint-region effect, including backend/inventory changes; not pure fusion time.",
                  grouping="Frozen rule: FULL diagnostics with multi-node HIT, only single-node HIT, or no HIT.",
                  single_node_reached=any(r["single_node_hits"] for r in reach["rows"] if r["mode"] == "single"))
    save(output / "composition.json", report)
    write_csv(output / "composition.csv", [dict(comparison=c["comparison"], **r) for c in comparisons for r in c["rows"]])
    return report
