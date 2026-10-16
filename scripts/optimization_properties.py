#!/usr/bin/env python3
"""Reorganize immutable RQ2/RQ3 evidence; audit raw deltas and allocation routes.

No measurement, source modification, policy change, or archive rewrite. Fusion
frequencies are path-structure/HIT inference; array removal is a structural
estimate. Production allocation counters corroborate whole-window totals.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys

METRICS = ("fused_traversals", "avoided_arrays_estimate", "generated_fused_traversals",
           "generated_avoided_arrays_estimate", "reuse_links", "allocations_saved_structural")


def internal_array_count(inputs, nodes, scalar_inputs, output):
    """Count graph-internal values whose original operands imply array results."""
    scalar = {i: i in scalar_inputs for i in range(inputs)}
    consumers = Counter(v for node in nodes for v in (node["left"], node["right"]))
    for node in nodes:
        scalar[node["output"]] = scalar[node["left"]] and scalar[node["right"]]
    internal = [node["output"] for node in nodes if node["output"] != output]
    if any(not consumers[v] for v in internal):
        raise ValueError("unused graph value cannot be counted as an eliminated intermediate")
    return sum(not scalar[v] for v in internal)


def native_topology(key):
    parts = key.split("|")
    inputs, nodes_count, output = map(int, parts[3:6])
    operands = [p for p in parts if p.startswith(("a:", "s:", "c:"))]
    if len(operands) != inputs:
        raise ValueError("native-key operand inventory is incomplete")
    nodes = []
    for part in parts:
        if part.startswith("n:"):
            op, left, right, out = map(int, part[2:].split(":"))
            nodes.append(dict(op=op, left=left, right=right, output=out))
    if len(nodes) != nodes_count:
        raise ValueError("native-key topology is incomplete")
    return inputs, nodes, {i for i, p in enumerate(operands) if p.startswith("s:")}, output


def attributes(signature, replay):
    n = signature["node_count"]
    weights = dict.fromkeys(METRICS, 0)
    if signature["backend"] == "generated":
        record = signature["record"]
        inputs, nodes = record["input_count"], record["nodes"]
        scalar = {i for i in range(inputs) if record["scalar_mask"] & (1 << i)}
        output = nodes[-1]["output"]
        path = "typed_generated_dag"
    else:
        key = signature["identity"]
        inputs, nodes, scalar, output = native_topology(key)
        path = {"E1": "generated_generic_traversal", "E2": "single_native_loop",
                "S1": "native_stage_sequence"}.get(key.split("|")[0])
        if path is None:
            raise ValueError("unclassified native execution path: " + key)
    if len(nodes) != n:
        raise ValueError("node count/topology mismatch")
    if path in ("typed_generated_dag", "generated_generic_traversal"):
        weights["fused_traversals"] = int(n >= 2)
        weights["avoided_arrays_estimate"] = internal_array_count(inputs, nodes, scalar, output)
        if path == "typed_generated_dag":
            weights["generated_fused_traversals"] = weights["fused_traversals"]
            weights["generated_avoided_arrays_estimate"] = weights["avoided_arrays_estimate"]
        model = dict(output_allocations=1, internal_allocations=0)
    elif path == "single_native_loop":
        if n != 1:
            raise ValueError("E2 is not a single operation")
        model = dict(output_allocations=1, internal_allocations=0)
    else:
        description = signature["record"]["execution_description"]
        plan = signature["record"]["synthesized_plan"]
        if description["stage_count"] != n or any(s["kind"] != 1 for s in description["stages"]):
            raise ValueError("new stage structure requires explicit traversal classification")
        weights["reuse_links"] = sum(v != 255 for v in plan["reuse_source"])
        variants = replay[signature["identity"]]
        if variants[1]["plan"] != plan:
            raise ValueError("permissions-on replay differs from actual runtime route")
        weights["allocations_saved_structural"] = variants[0]["allocations"] - variants[1]["allocations"]
        model = dict(output_allocations=0, internal_allocations=0)
        scalar_stages = {int(p.split(":")[1]) for p in signature["identity"].split("|")
                         if p.startswith("v:") and int(p.split(":")[3])}
        for i, stage in enumerate(description["stages"]):
            v = stage["output"]
            if plan["reuse_source"][v] != 255:
                continue
            if plan["output_storage"][v]:
                model["output_allocations"] += 1
            elif i not in scalar_stages:
                model["internal_allocations"] += 1
        if model["output_allocations"] + model["internal_allocations"] != variants[1]["allocations"]:
            raise ValueError("scalar/private storage not represented by synthetic allocation replay")
    return dict(path=path, input_count=inputs, nodes=nodes, output_value=output,
                weights=weights, allocation_model_per_hit=model,
                evidence_kind="executed_path_structure_and_successful_hit_delta")


def range_summary(values):
    return dict(median=statistics.median(values), min=min(values), max=max(values))

