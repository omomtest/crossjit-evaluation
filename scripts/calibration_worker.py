#!/usr/bin/env python3
"""Use Provider's complete-path tuning harness with all compiled descriptors.

FORCE is confined to this calibration process; Host proof and Provider domain
checks remain intact. Unsupported/unreachable cells are recorded by controller
as rejected and never copied from old machine measurements.
"""
import argparse
import importlib.util
from pathlib import Path
import sys

from common import apply_affinity


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--provider-harness", type=Path, required=True)
    args, remaining = pre.parse_known_args()
    spec = importlib.util.spec_from_file_location("provider_tuning", args.provider_harness)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module._cases = lambda metadata, layouts: sorted([dict(k, profiled_source=k.get("profiled_source", "manifest_inventory"))
                                                      for k in metadata["kernels"]], key=lambda k: k["execution_specialization_hash"])
    sys.argv = [str(args.provider_harness), *remaining]
    apply_affinity()
    module.main()


if __name__ == "__main__":
    main()
