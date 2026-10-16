#!/usr/bin/env python3
"""Describe the compiler actually used, without assuming x86 or copying an external machine record."""
import argparse
import json
from pathlib import Path
import subprocess


def environment(compiler, flags):
    def output(argv, source=None):
        return subprocess.check_output([compiler, *argv], input=source, text=True).strip()
    macros = output([*flags, "-dM", "-E", "-x", "c", "-"], "")
    features = []
    for line in macros.splitlines():
        name = line.split()[1]
        if name in ("__SSE__", "__SSE2__", "__SSE3__", "__SSSE3__", "__SSE4_1__", "__SSE4_2__", "__AVX__", "__AVX2__", "__ARM_NEON", "__ARM_NEON__", "__ARM_FEATURE_SVE"):
            features.append(name.strip("_").lower())
    return dict(artifact_format_version=1, compiler_id="gcc", compiler_version=output(["-dumpfullversion"]),
                target_triple=output(["-dumpmachine"]), target_features=sorted(set(features)), compiler_flags=flags)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--compiler", default="gcc-13")
    p.add_argument("--flags", nargs="+", default=["-std=c11", "-O3", "-fno-strict-aliasing", "-fPIC"])
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(environment(args.compiler, args.flags), indent=2, sort_keys=True)+"\n")


if __name__ == "__main__":
    main()
