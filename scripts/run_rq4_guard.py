#!/usr/bin/env python3
"""RQ4: net amortized guard-MISS cost from complete-call times and matched counts."""
import argparse
from collections import Counter, defaultdict
from pathlib import Path
import statistics

from common import cli, read_json, save
from statistics_tools import aggregate_pairs


def guard_report(scope, output):
    scope = Path(scope)
    cfg = read_json(scope / "manifest.json")["configuration"]
    reach = read_json(scope / "reach.json")
    composition = read_json(scope / "composition.json")
    if not reach["complete"] or not composition["complete"]:
        raise ValueError("complete same-binary scope timing and diagnostic records are required")
    full = next(c for c in composition["comparisons"] if c["comparison"] == "off/on")
    counts, per_call = defaultdict(Counter), defaultdict(list)
    for row in reach["rows"]:
        if row["mode"] == "on":
            counts[row["workload"]].update(row["timed_outcomes"])
            per_call[row["workload"]].append(row["timed_outcomes"]["guard_miss"] / cfg["repeat"])
    groups = {}
    for name in ("miss_present", "active_no_miss", "inactive"):
        rows = [r for r in full["rows"] if
                (counts[r["workload"]]["guard_miss"] > 0 if name == "miss_present" else
                 counts[r["workload"]]["exec_hit"] > 0 and counts[r["workload"]]["guard_miss"] == 0
                 if name == "active_no_miss" else counts[r["workload"]]["exec_hit"] == 0)]
        groups[name] = aggregate_pairs(rows, cfg["seed"])
        groups[name]["slower"] = sum(r["direction"] == "slower" for r in rows)
    proxies = []
    for row in full["rows"]:
        name = row["workload"]
        if not counts[name]["guard_miss"]:
            continue
        if len(per_call[name]) != cfg["trials"] or len(set(per_call[name])) != 1:
            raise ValueError("guard-MISS frequency varies; fixed-count normalization is not justified")
        misses = statistics.median(per_call[name])
        proxies.append(dict(workload=name, misses_per_call=misses,
            miss_fraction=counts[name]["guard_miss"]/(counts[name]["guard_miss"]+counts[name]["guard_hit"]),
            delta_ns=row["delta_ns"], delta_ci95_ns=row["delta_ci95_ns"],
            net_ns_per_miss=row["delta_ns"]/misses,
            net_ns_per_miss_ci95=[x/misses for x in row["delta_ci95_ns"]],
            speedup=row["speedup"], speedup_ci95=row["speedup_ci95"]))
    report = dict(complete=True, groups=groups, rows=proxies,
        interpretation="Net complete-call FULL minus OFF time per separately recorded MISS; includes HIT benefits and other path differences. This is not isolated guard latency.")
    save(Path(output) / "guard.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scope", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    guard_report(args.scope, args.output)


if __name__ == "__main__":
    cli(main)
