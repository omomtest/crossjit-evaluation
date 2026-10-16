#!/usr/bin/env python3
"""Collect the current paper RQ1–RQ4 using a single frozen configuration."""
from pathlib import Path
import subprocess
import sys
from common import ROOT, cli, configuration, parser, save

def main():
    p = parser(__doc__)
    args = p.parse_args()
    cfg = configuration(args)
    output = args.output.absolute()
    save(output / "effective-config.json", cfg)
    phases = [("rq1", "run_rq1_performance.py", []),
              ("rq2", "run_rq2_coverage.py", []),
              ("rq2-adi", "run_microbench_case_study.py", []),
              ("rq3", "run_rq3_composition.py", []),
              ("rq3-storage", "run_rq3_storage.py", ["--reach", str(output / "rq2/reach.json")]),
              ("rq4-callback", "run_rq4_attribution.py", []),
              ("rq4-inactive", "run_rq4_overhead.py", ["--coverage", str(output / "rq2/coverage.json")])]
    status = []
    for label, script, extra in phases:
        command = [sys.executable, "-B", str(ROOT / "scripts" / script), "--config", str(output / "effective-config.json"),
                   "--output", str(output / label), *extra]
        if args.workloads and label in ("rq1", "rq2", "rq3"):
            command += ["--workloads", *args.workloads]
        code = subprocess.run(command).returncode
        status.append(dict(phase=label, exitcode=code))
        save(output / "status.json", dict(phases=status, complete=len(status)==len(phases) and not any(r["exitcode"] for r in status)))
        if code:
            raise SystemExit(code)
    subprocess.run([sys.executable, str(ROOT / "scripts/run_rq4_guard.py"), "--scope", str(output / "rq3"),
                    "--output", str(output / "rq4-guard")], check=True)

if __name__ == "__main__":
    cli(main)
