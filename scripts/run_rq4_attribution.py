#!/usr/bin/env python3
"""RQ4: callback organization and checked capability ablations."""
from pathlib import Path
from redesign_common import same_binary

from common import ROOT, cli, configuration, freeze, parser, read_json, save, workload_specs
from execution import run_matrix
from reporting import diagnostic_window, summarize_matrix


def available_variants(cfg, path):
    variants, unavailable = [], []
    for variant in read_json(path)["variants"]:
        if variant.get("environment"):
            source = Path(cfg["source_root"]) / "numpy/numpy/_core/src/umath/jit_provider.c"
            if not source.is_file() or variant["source_symbol"] not in source.read_text():
                unavailable.append(dict(variant, status="unavailable", reason="runtime switch is absent from selected source"))
                continue
            # Check the selected native extension too: source can be ahead of the installed binary.
            from common import probe_python
            files = probe_python(cfg["candidate_python"], cfg)["binary_files"]
            if not any(variant["source_symbol"].encode() in Path(p).read_bytes() for p in files if p.endswith(".so")):
                unavailable.append(dict(variant, status="unavailable", reason="switch is absent from installed binary"))
                continue
        elif not variant.get("python"):
            unavailable.append(dict(variant, status="unavailable"))
            continue
        elif not variant.get("build_manifest") or not Path(variant["build_manifest"]).is_file():
            raise ValueError("a separate binary ablation requires its build_manifest: " + variant["name"])
        variants.append(variant)
    return variants, unavailable


def main():
    p = parser(__doc__)
    p.add_argument("--variants", type=Path, default=ROOT / "config/variants.json")
    p.set_defaults(workloads=["adi", "deriche", "channel_flow", "hdiff"])
    args = p.parse_args()
    cfg = same_binary(configuration(args))
    variants, unavailable = available_variants(cfg, args.variants)
    output, manifest = freeze(args.output, cfg, "rq4-attribution", workload_specs(args, cfg), variants,
                              extra=dict(unavailable=unavailable))
    run_matrix(output, manifest, modes=("baseline", "off", "on", *(v["name"] for v in variants)))
    summaries = [summarize_matrix(output), summarize_matrix(output, "off", "on")]
    violations = []
    for variant in variants:
        summaries.append(summarize_matrix(output, variant["name"], "on"))
        if variant.get("invariant") == "same_reached_inventory":
            for spec in manifest["specs"]:
                from execution import spec_id
                for trial in range(cfg["trials"]):
                    paths = [output / "raw" / f"{spec_id(spec)}.{mode}.t{trial}.diag.json" for mode in ("on", variant["name"])]
                    if not all(path.is_file() for path in paths):
                        violations.append(dict(workload=spec["name"], trial=trial, variant=variant["name"], reason="missing diagnostic pair"))
                        continue
                    windows = []
                    for path in paths:
                        cell = read_json(path)
                        windows.append(diagnostic_window(cell["preparation"], cell["final"], cell["repeat"]))
                    def inventory(window):
                        return (window["node_histogram"], [(r["identity"], r["hits"]) for r in window["actual_signatures"]],
                                {k: window["provider"].get(k) for k in ("generated_exec_hit", "native_exec_hit", "mixed_exec_hit", "internal_allocations", "output_allocations")})
                    if inventory(windows[0]) != inventory(windows[1]):
                        violations.append(dict(workload=spec["name"], trial=trial, variant=variant["name"], reason="reached inventory changed; isolated callback attribution invalid"))
    save(output / "ablation-status.json", dict(unavailable=unavailable, invariant_violations=violations,
                                               diagnostic_ablation=True, production_acceptance=False))
    if violations or any(not s["complete"] for s in summaries):
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
