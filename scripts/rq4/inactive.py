#!/usr/bin/env python3
"""Current paper RQ4 entry point."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_rq4_overhead import main
from common import cli
if __name__ == "__main__":
    cli(main)
