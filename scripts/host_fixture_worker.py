"""Independent synthetic Host evidence; explicitly not a numerical Provider."""
import argparse
from pathlib import Path
import unittest
import _testinternalcapi as host
from test.test_crossjit import CrossJitSyntheticProviderTests
from common import save


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    rows = []
    class Result(unittest.TextTestResult):
        def addSuccess(self, test):
            rows.append(dict(test=test.id(), host=host.crossjit_stats(),
                             synthetic_provider=host.crossjit_synthetic_stats()))
            super().addSuccess(test)
    names = ("test_semantic_reuse_never_reuses_physical_mapping",
             "test_fallible_operation_with_live_local_stays_materialized",
             "test_effect_boundaries_prefix_authority_and_fault_lifetime",
             "test_transient_local_forwarding_guard_replay_and_ownership")
    result = unittest.TextTestRunner(verbosity=2, resultclass=Result).run(
        unittest.TestSuite(CrossJitSyntheticProviderTests(name) for name in names))
    complete = result.wasSuccessful() and len(rows) == len(names) and not result.skipped
    save(args.output, dict(complete=complete, rows=rows, scope="synthetic Host ABI/physical proof/recovery/lifetime fixtures",
                          generator_domain_reuse_claim=False))
    if not complete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
