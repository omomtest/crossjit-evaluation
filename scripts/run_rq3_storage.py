#!/usr/bin/env python3
"""RQ3: captured Provider descriptions and storage-permission ablation."""
from pathlib import Path
from common import canonical_hash, cli, configuration, freeze, parser, read_json, save
from generator_evidence import captured_descriptions, synthesize_evidence
from redesign_common import same_binary


def main():
    p = parser(__doc__)
    p.add_argument("--reach", type=Path, required=True, help="Matching RQ2 reach.json")
    args = p.parse_args()
    cfg = same_binary(configuration(args))
    origin = read_json(args.reach.parent / "manifest.json")
    reach = read_json(args.reach)
    if not reach["complete"] or origin["experiment"] != "rq2":
        raise ValueError("RQ3 storage replay requires complete RQ2 diagnostics")
    captured = captured_descriptions(reach)
    output, manifest = freeze(args.output, cfg, "rq3-storage", [], extra=dict(
        reach_sha256=canonical_hash(reach), origin_identity=canonical_hash(origin["identity"])))
    if manifest["identity"] != origin["identity"] or cfg != origin["configuration"]:
        raise ValueError("RQ3 storage descriptions must match source/binary/protocol identity")
    save(output / "typed-dags.json", dict(rows=[dict(workload=row["workload"], trial=row["trial"],
        mode=row["mode"], signature=signature, runtime_allocations=row["allocations"], trace_log=row["trace_log"])
        for row in reach["rows"] for signature in row["actual_signatures"] if signature["backend"] == "generated"],
        scope="Actual generated typed-DAG inventory; distinct from runtime backend stage descriptions.",
        traversal_events=None, traversal_status="No traversal-event ABI; source-node count is not a measured traversal count."))
    report = synthesize_evidence(output, cfg["source_root"], captured)
    save(output / "responsibilities.json", dict(
        provider=["value IDs/type/access/domain facts", "binding and dependencies", "storage classes/reuse permissions/overlap mask"],
        generator=["dependency stage order", "producer and last use", "storage slots and reuse sources", "output storage chain", "release masks"],
        bridge=["primitive/backend execution", "concrete allocations and logical ownership", "FPE/errors, publication and cleanup"],
        source_audit=dict(planner="crossjit-generator/crossjit_generator/execution/plan.c.in",
                          bridge="numpy/numpy/_core/src/umath/jit_native_stages.h",
                          scope="The actual planner is compiled/replayed; opaque capabilities retain their backend algorithm.")))
    save(output / "status.json", dict(complete=report["complete"],
        workload_capture_available=bool(captured),
        gaps=[] if captured else ["No executed native stage description captured; planner fixtures alone do not establish workload synthesis."]))
    if not captured:
        raise SystemExit(2)


if __name__ == "__main__":
    cli(main)
