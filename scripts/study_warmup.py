#!/usr/bin/env python3
"""Screen default-policy warmup; confirm only with full-suite execution and ten process pairs."""
import math
from pathlib import Path

from common import canonical_hash, cli, configuration, freeze, parser, read_json, save, workload_specs
from execution import run_cell, run_matrix
from reporting import diagnostic_window, summarize_matrix


def rate_close(a, b):
    return a == b == 0 or abs(a-b)/max(a, b) <= .05


def reached_close(a, b):
    ia = {r["identity"]: r["hits"] for r in a["actual_signatures"]}
    ib = {r["identity"]: r["hits"] for r in b["actual_signatures"]}
    if set(ia) != set(ib) or not all(rate_close(ia[k], ib[k]) for k in ia):
        return False
    ha, hb = a["node_histogram"], b["node_histogram"]
    ta, tb = sum(ha.values()), sum(hb.values())
    tv = sum(abs(ha.get(k, 0)/ta-hb.get(k, 0)/tb) for k in set(ha)|set(hb))/2 if ta and tb else 0 if ta == tb else 1
    return (tv <= .05 and rate_close(a["executed_hit_per_invocation"], b["executed_hit_per_invocation"])
            and rate_close(a["executed_nodes_per_invocation"], b["executed_nodes_per_invocation"]))


def terminal(cell):
    warmup, repeat = cell["warmup"], cell["repeat"]
    if warmup < 2*repeat:
        return None
    points = cell["checkpoints"]
    start = cell["initial"] if warmup == 2*repeat else points.get(str(warmup-2*repeat))
    middle = points.get(str(warmup-repeat))
    if start is None or middle is None:
        return None
    windows = [diagnostic_window(start, middle, repeat),
               diagnostic_window(middle, cell["preparation"], repeat),
               diagnostic_window(cell["preparation"], cell["final"], repeat)]
    return dict(windows=windows, saturated=all(w["diagnostics_complete"] and not w["host"].get("rewritten_regions", 0) for w in windows)
                and all(reached_close(windows[0], w) for w in windows[1:]))


def evaluate_point(args, cfg, specs, warmup, label, trials):
    cfg = dict(cfg, warmup=warmup, warmup_protocol=None, trials=trials)
    output, manifest = freeze(args.output / label, cfg, "warmup-study", specs)
    # Timing has no detailed instrumentation. Diagnostics use three terminal windows.
    run_matrix(output, manifest, modes=("baseline", "off", "on"), diagnostics=False)
    results = []
    checkpoints = [v for v in (warmup-2*cfg["repeat"], warmup-cfg["repeat"]) if v > 0]
    for spec in specs:
        for trial in range(min(trials, 3)):
            cell = run_cell(output, manifest, spec, "on", trial, diagnostic=True, checkpoints=checkpoints)
            if cell:
                results.append(dict(workload=spec["name"], trial=trial, terminal=terminal(cell),
                                    window=diagnostic_window(cell["preparation"], cell["final"], cell["repeat"])))
    summary = summarize_matrix(output)
    onoff = summarize_matrix(output, "off", "on")
    return dict(warmup=warmup, summary=summary, onoff=onoff, diagnostics=results,
                identity=manifest["identity"], complete=summary["complete"] and len(results)==len(specs)*min(trials, 3))


def select(points, full_suite):
    evaluated = []
    candidate = None
    for index, point in enumerate(points):
        reasons, saturated, residual = [], [], []
        later = points[index+1:]
        if not full_suite:
            reasons.append("not_all_53")
        if len(later) < 2:
            reasons.append("needs_two_higher_points")
        if not all(p["complete"] for p in [point, *later]):
            reasons.append("incomplete_matrix")
        active = {r["workload"] for r in point["diagnostics"] if r["window"]["active"]}
        for other in later:
            if active != {r["workload"] for r in other["diagnostics"] if r["window"]["active"]}:
                reasons.append("active_set_changed")
            a, b = point["summary"], other["summary"]
            if a["GM_all"] and b["GM_all"] and abs(math.log(a["GM_all"]/b["GM_all"])) > .02:
                reasons.append("GM_changes_above_2_percent")
            aci, bci = a["suite_GM_ci95"], b["suite_GM_ci95"]
            if None in aci+bci or aci[0] > bci[1] or bci[0] > aci[1]:
                reasons.append("GM_CI_missing_or_disjoint")
        weights = {}
        for name in active:
            samples = [r for p in [point, *later] for r in p["diagnostics"] if r["workload"] == name]
            valid = (samples and all(r["terminal"] and r["terminal"]["saturated"] and r["window"]["active"] for r in samples)
                     and all(reached_close(samples[0]["window"], r["window"]) for r in samples[1:]))
            (saturated if valid else residual).append(name)
            weights[name] = max((r["window"]["executed_nodes_per_invocation"] for r in samples), default=0)
        if not active or len(saturated)/len(active) < .95:
            reasons.append("less_than_95_percent_active_workloads_saturated")
        total = sum(weights.values())
        if not total or sum(weights[n] for n in saturated)/total < .95:
            reasons.append("less_than_95_percent_executed_nodes_saturated")
        evaluated.append(dict(warmup=point["warmup"], reasons=sorted(set(reasons)), residual_workloads=residual,
                              saturated_workloads=saturated))
        if not reasons and candidate is None:
            candidate = point["warmup"]
    return candidate, evaluated


def main():
    p = parser(__doc__)
    p.add_argument("--warmups", nargs="+", type=int, default=[100, 500, 1000, 2000, 4000, 8500])
    p.add_argument("--confirm", action="store_true", help="Run ten independent pairs at selected W; never select by speedup")
    args = p.parse_args()
    cfg = configuration(args, require_warmup=False)
    cfg["warmup_protocol"] = None
    values = sorted(set(args.warmups))
    if min(values) < 0 or len(values) != len(args.warmups):
        raise ValueError("warmup points must be unique and nonnegative")
    specs = workload_specs(args, cfg)
    points = [evaluate_point(args, cfg, specs, w, f"W{w}", cfg["trials"]) for w in values]
    candidate, evaluation = select(points, len(specs)==53)
    protocol = dict(schema_version=1, state="pending", W_star=None, candidate_warmup=candidate,
                    evaluations=evaluation, limitation="finite tested-grid evidence; unresolved saturation is retained")
    if candidate is not None and args.confirm:
        confirmation = evaluate_point(args, cfg, specs, candidate, f"confirmation-W{candidate}", 10)
        confirmation_points = [confirmation, *[point for point in points if point["warmup"] > candidate]]
        confirmed, checks = select(confirmation_points, len(specs)==53)
        rows = confirmation["summary"]["rows"]
        stability_ok = all(r["stability"] == "stable" for r in rows if r["active"])
        if confirmed == candidate and stability_ok:
            protocol.update(state="confirmed", W_star=candidate,
                            binding=canonical_hash(dict(identity=confirmation["identity"], blas_threads=cfg["blas_threads"],
                                                       affinity=cfg["cpu_affinity"], preset=cfg["preset"], repeat=cfg["repeat"])))
        protocol["confirmation"] = dict(checks=checks, active_stability_pass=stability_ok)
    save(args.output / "warmup-protocol.json", protocol)
    save(args.output / "warmup-sweep.json", dict(points=points, protocol=protocol))
    if protocol["state"] != "confirmed":
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
