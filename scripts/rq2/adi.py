#!/usr/bin/env python3
"""Current paper RQ2 entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_microbench_case_study import main
from common import cli
if __name__ == "__main__":
    cli(main)
