#!/usr/bin/env python3
"""Recompute identity-checked summaries; optionally plot with an independent analysis Python."""
import argparse
from pathlib import Path
import subprocess
import sys

from common import ROOT, read_json, verify_identity
from reporting import summarize_matrix


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory", type=Path)
    p.add_argument("--denominator", default="baseline")
    p.add_argument("--numerator", default="on")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--plot-python", default=sys.executable, help="Python with the optional Matplotlib analysis dependencies")
    args = p.parse_args()
    verify_identity(read_json(args.directory / "manifest.json")["identity"])
    summarize_matrix(args.directory, args.denominator, args.numerator)
    if args.plot:
        label = "summary" if (args.denominator, args.numerator) == ("baseline", "on") else f"summary-{args.denominator}-{args.numerator}"
        subprocess.run([args.plot_python, "-B", str(ROOT / "scripts/plot_results.py"),
                        str(args.directory / (label+".json")), "--output",
                        str(args.directory / f"speedups-{args.denominator}-{args.numerator}.svg")], check=True)


if __name__ == "__main__":
    main()
