"""Paper-estimator regressions where paired-ratio statistics give another answer."""
from pathlib import Path
import statistics
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import read_json
from statistics_tools import aggregate_pairs, paired_summary


class PaperStatisticsTests(unittest.TestCase):
    def test_ratio_of_medians_is_used_for_point_and_paired_resampling(self):
        cfg = read_json(ROOT / 'config/experiment.json')
        base, candidate = [1., 10., 100.], [1., 2., 100.]
        result = paired_summary(base, candidate, cfg)
        self.assertEqual(result['speedup'], 5.)
        self.assertEqual(result['paired_median_speedup'], 1.)
        self.assertEqual(result['delta_ns'], -8.)
        self.assertEqual(result['speedup_ci95'], [1., 5.])

    def test_workload_gm_uses_ratio_of_medians(self):
        rows = [dict(complete=True, process_baseline_ns=[1., 10., 100.],
                     process_candidate_ns=[1., 2., 100.])]
        result = aggregate_pairs(rows, samples=1000)
        self.assertAlmostEqual(result['speedup'], 5.)


if __name__ == '__main__':
    unittest.main()
