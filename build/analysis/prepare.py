#!/usr/bin/env python3
"""Build an optional Python 3.12 plotting environment, separate from measurement binaries."""
import argparse
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--python", default="python3.12")
    p.add_argument("--venv", type=Path, required=True)
    args = p.parse_args()
    subprocess.run([args.python, "-m", "venv", str(args.venv)], check=True)
    subprocess.run([str(args.venv / "bin/python"), "-m", "pip", "install", "-r",
                    str(Path(__file__).with_name("requirements.lock.txt"))], check=True)


if __name__ == "__main__":
    main()
