"""Confidence intervals resample process pairs, never individual timed invocations."""
import math
import random
import statistics
from functools import lru_cache


def gm(values):
    values = list(values)
    return math.exp(statistics.mean(map(math.log, values))) if values else None


def cv(values):
    return statistics.stdev(values) / statistics.mean(values) if len(values) > 1 and statistics.mean(values) else None


def percentile(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    i = int(pos)
    return values[i] + (values[min(i + 1, len(values) - 1)] - values[i]) * (pos - i)


def bootstrap(values, seed=20261002, statistic=statistics.median, samples=10000):
    if len(values) < 3:
        return [None, None]
    rng = random.Random(seed)
    results = [statistic(rng.choices(values, k=len(values))) for _ in range(samples)]
    return [percentile(results, .025), percentile(results, .975)]


def stability(base, cand, ratios, limits, ci):
    if len(ratios) < limits["min_trials"]:
        return "unconfirmed_insufficient_processes"
    def spread(values):
        med = statistics.median(values)
        return (cv(values) <= limits["max_time_cv"] and
                statistics.median(abs(v - med) for v in values) / med <= limits["max_relative_mad"] and
                (percentile(values, .75) - percentile(values, .25)) / med <= limits["max_relative_iqr"] and
                percentile(values, .9) / percentile(values, .1) <= limits["max_ratio"])
    ratio_med = statistics.median(base) / statistics.median(cand)
    stable = (spread(base) and spread(cand) and spread(ratios) and
              cv(ratios) <= limits["max_speedup_cv"] and
              (ci[1] - ci[0]) / ratio_med <= limits["max_relative_ci_width"])
    return "stable" if stable else "unstable"


@lru_cache(maxsize=1024)
def paired_bootstrap(base, cand, seed=20261002, samples=10000):
    """Recompute the ratio and difference of medians with paired trial indices."""
    if len(base) != len(cand) or not base or any(x <= 0 for x in (*base, *cand)):
        raise ValueError("paired positive process times are required")
    rng = random.Random(seed)
    ratios, differences = [], []
    for _ in range(samples):
        indices = rng.choices(range(len(base)), k=len(base))
        b = statistics.median(base[i] for i in indices)
        c = statistics.median(cand[i] for i in indices)
        ratios.append(b / c)
        differences.append(c - b)
    return tuple(ratios), tuple(differences)


def aggregate_pairs(rows, seed=20261002, samples=10000):
    """Share bootstrap trial indices across workloads and recompute S_A and GM."""
    if not rows or any(not r["complete"] for r in rows):
        return dict(workloads=len(rows), speedup=None, ci95=[None, None])
    counts = {len(r["process_baseline_ns"]) for r in rows}
    if len(counts) != 1 or any(len(r["process_candidate_ns"]) not in counts for r in rows):
        raise ValueError("aggregate requires the same process-pair count")
    estimate = gm(statistics.median(r["process_baseline_ns"]) /
                  statistics.median(r["process_candidate_ns"]) for r in rows)
    if next(iter(counts)) < 3:
        return dict(workloads=len(rows), speedup=estimate, ci95=[None, None])
    vectors = [paired_bootstrap(tuple(r["process_baseline_ns"]),
                               tuple(r["process_candidate_ns"]), seed, samples)[0] for r in rows]
    draws = [gm(values) for values in zip(*vectors, strict=True)]
    return dict(workloads=len(rows), speedup=estimate,
                ci95=[percentile(draws, .025), percentile(draws, .975)],
                resampling_unit="process-pair index shared across workloads; recompute S_A and GM")


def paired_summary(base, cand, cfg):
    ratios = [b / c for b, c in zip(base, cand, strict=True)]
    draws, differences = paired_bootstrap(tuple(base), tuple(cand), cfg["seed"])
    ci = [percentile(draws, .025), percentile(draws, .975)] if len(base) >= 3 else [None, None]
    return dict(process_pairs=len(ratios), baseline_median_ns=statistics.median(base),
                candidate_median_ns=statistics.median(cand),
                speedup=statistics.median(base) / statistics.median(cand),
                paired_median_speedup=statistics.median(ratios), speedup_ci95=ci,
                ratio_estimator="ratio of process medians (S_A)",
                delta_ns=statistics.median(cand)-statistics.median(base),
                delta_ci95_ns=[percentile(differences,.025),percentile(differences,.975)] if len(base)>=3 else [None,None],
                baseline_cv=cv(base), candidate_cv=cv(cand), speedup_cv=cv(ratios),
                stability=stability(base, cand, ratios, cfg["stability"], ci),
                direction=("unresolved" if ci[0] is None or ci[0] <= 1 <= ci[1]
                           else "faster" if ci[0] > 1 else "slower"),
                process_baseline_ns=base, process_candidate_ns=cand, process_ratios=ratios)
