"""Replay captured descriptions through the selected Generator C template."""
import json
from pathlib import Path
import shutil
import subprocess

from common import ROOT, digest, read_json, save


def fixtures():
    def value(kind):
        return dict(kind=kind, storage_class=1)
    def stage(inputs, output, kind=1, after=0, reuse=0):
        return dict(kind=kind, capability=0, inputs=inputs, output=output, after=after, reuse_candidates=reuse)
    chain = dict(input_count=1, value_count=4, stage_count=3, output_value=3,
                 values=[value(0), value(1), value(1), value(2)],
                 stages=[stage([0], 1, 0), stage([1], 2, after=1, reuse=2),
                         stage([2], 3, after=2, reuse=4)],
                 capabilities=[dict(binding=0, input_count=1, overlap_inputs=1)])
    fork = dict(input_count=1, value_count=5, stage_count=4, output_value=4,
                values=[value(0), value(1), value(1), value(1), value(2)],
                stages=[stage([0], 1), stage([1], 2, reuse=2), stage([1], 3, reuse=2),
                        dict(kind=1, capability=1, inputs=[2, 3], output=4, after=4, reuse_candidates=12)],
                capabilities=[dict(binding=0, input_count=1, overlap_inputs=1),
                              dict(binding=1, input_count=2, overlap_inputs=3)])
    negative = dict(input_count=1, value_count=2, stage_count=1, output_value=1,
                    values=[value(0), value(2)], stages=[stage([0], 1, 0)],
                    capabilities=[dict(binding=0, input_count=1, overlap_inputs=0)])
    return [dict(identity="strict-generated-chain", description=chain, scope="planner_fixture"),
            dict(identity="fork-last-use", description=fork, scope="planner_fixture"),
            dict(identity="no-private-reuse", description=negative, scope="planner_fixture")]


def captured_descriptions(coverage):
    records = {}
    for row in coverage["rows"]:
        window = row["timed_window"]
        before = row["construction"]["post_warmup"]["provider"].get("native_candidates", {})
        for key, entry in row["lifetime"]["provider"].get("native_candidates", {}).items():
            if "execution_description" not in entry:
                continue
            hits = entry.get("execute", 0) - before.get(key, {}).get("execute", 0)
            if hits <= 0:
                continue  # A prepared description is not an executed workload region.
            records.setdefault(key, dict(identity=key, description=entry["execution_description"],
                original_plan=entry["synthesized_plan"], scope="captured_workload_description",
                workload=row["workload"], trial=row["trial"], mode=row["mode"], timed_hits=hits,
                allocations=row["allocations"], trace_log=row["trace_log"]))
    return list(records.values())


def initializer(d):
    def seq(values):
        return "{" + ",".join(str(int(v)) for v in values) + "}"
    if not (0 < d["input_count"] <= 16 and 0 < d["stage_count"] <= 12 and d["value_count"] <= 28):
        raise ValueError("description exceeds replay harness bounds")
    values = ",".join("{%d,%d}" % (v["kind"], v["storage_class"]) for v in d["values"])
    caps = ",".join("{%d,%d,%d}" % (c["binding"], c["input_count"], c["overlap_inputs"]) for c in d["capabilities"])
    stages = ",".join("{.kind=%d,.capability=%d,.input_count=%d,.inputs=%s,.output=%d,.after=%d,.reuse_candidates=%d}" %
                      (s["kind"], s["capability"], len(s["inputs"]), seq(s["inputs"]), s["output"], s["after"], s["reuse_candidates"])
                      for s in d["stages"])
    return ("{.input_count=%d,.value_count=%d,.stage_count=%d,.output_value=%d,.values={%s},.stages={%s},.capabilities={%s},.capability_count=%d}" %
            (d["input_count"], d["value_count"], d["stage_count"], d["output_value"], values, stages, caps, len(d["capabilities"])))


def synthesize_evidence(output, source_root, captured=()):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    template = Path(source_root) / "crossjit-generator/crossjit_generator/execution/plan.c.in"
    compiler = shutil.which("cc") or shutil.which("gcc-13") or shutil.which("gcc")
    if not compiler:
        raise RuntimeError("RQ3 storage replay needs a C compiler for actual Generator template replay")
    text = template.read_text().replace("@NODES@", "12").replace("@INPUTS@", "16").replace("@RANK@", "8").replace("@OPCODES@", "CJ_AUDIT_OP=0")
    shutil.copyfile(template, output / "plan.c.in")
    records = [*fixtures(), *captured]
    (output / "execution.h").write_text(text)
    (output / "descriptions.h").write_text("#define DESCRIPTION_COUNT %d\nstatic CjExecutionDescription descriptions[]={\n%s\n};\n" %
                                          (len(records), ",\n".join(initializer(r["description"]) for r in records)))
    shutil.copyfile(ROOT / "scripts/planner_bridge.c", output / "planner_bridge.c")
    command = [compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(output),
               str(output / "planner_bridge.c"), "-o", str(output / "planner_bridge")]
    with (output / "compile.log").open("w") as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    result = subprocess.run([str(output / "planner_bridge")], capture_output=True, text=True, check=True)
    (output / "execution.jsonl").write_text(result.stdout)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    for row in rows:
        record = records[row["description"]]
        row["identity"] = record["identity"]
        row["scope"] = record["scope"]
        if row["permissions"] and record.get("original_plan") != row["plan"] and "original_plan" in record:
            raise RuntimeError("replayed Generator plan differs from captured runtime plan")
    fork_plan = next(r["plan"] for r in rows if r["identity"] == "fork-last-use" and r["permissions"])
    if fork_plan["reuse_source"][2] != 255 or fork_plan["reuse_source"][3] != 1:
        raise RuntimeError("fork last use permits early overwrite or loses legal reuse")
    report = dict(schema_version=1, complete=True, records=records, rows=rows,
        captured_workload_descriptions=len(captured), planner_template_sha256=digest(template),
        bridge_source_sha256=digest(ROOT / "scripts/planner_bridge.c"), command=command,
        limitations=["Replay executes a synthetic allocation/ownership bridge, not original numerical bindings.",
                     "Replay bytes/peak storage belong to eight-double fixture buffers; runtime allocations are separate.",
                     "Permissions-off plans are revalidated planning ablations, never forced into production admission."])
    save(output / "synthesis.json", report)
    return report
