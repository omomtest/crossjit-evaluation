"""Evidence invariants: actual HIT, SINGLE exclusion, storage and no-RQ1 dispatch."""
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_json, save
from common import digest
from redesign_common import archive_policy
from generator_evidence import synthesize_evidence
from redesign_execution import spec_id
from redesign_reporting import aggregate_pairs, classify, reach_report, region_window
from run_microbench_case_study import case_checks
from run_redesigned_evaluation import stages
from run_redesigned_docker_pipeline import redesigned_plan
from run_docker_pipeline import argument_parser
from types import SimpleNamespace


class RedesignTests(unittest.TestCase):
    def test_distinct_scalar_access_signatures_keep_independent_hit_deltas(self):
        first = dict(node_count=1, input_count=2, scalar_mask=0, access_kinds=[1, 1])
        second = dict(node_count=1, input_count=2, scalar_mask=1, access_kinds=[0, 1])
        before = dict(host={}, provider=dict(generated_exec_hit=30, observed_signatures=[
            dict(first, generated_hit_count=10), dict(second, generated_hit_count=20)]))
        after = dict(host={}, provider=dict(generated_exec_hit=33, observed_signatures=[
            dict(first, generated_hit_count=11), dict(second, generated_hit_count=22)]))
        window = region_window(before, after, 1)
        self.assertEqual(window["node_histogram"], {1: 3})
        self.assertEqual(len(window["actual_signatures"]), 2)

    def test_policy_archive_preserves_replay_source_and_checks_build_binding(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            policy = root / "policy"
            policy.mkdir()
            (policy / "measurement.py").write_text("# original measurement harness\n")
            save(policy / "native_policy.json", dict(records=[]))
            save(policy / "execution_decision_db.json", dict(records=[]))
            files = {p.name: dict(sha256=digest(p)) for p in policy.iterdir()}
            bundle = dict(files=files)
            save(policy / "bundle.json", bundle)
            manifest = dict(identity=dict(build=dict(frozen_optimization_policy=bundle)))
            archive_policy(root / "output", policy, manifest)
            report = read_json(root / "output/admission-evidence.json")
            self.assertTrue(report["policy_verified_against_binary"])
            self.assertTrue((root / "output/policy-evidence/measurement.py").is_file())
            (policy / "measurement.py").write_text("# drift\n")
            with self.assertRaises(RuntimeError):
                archive_policy(root / "another", policy, manifest)

    def test_process_pair_resampling_reaggregates_workload_gm(self):
        # Every paired index gives reciprocal effects. Shared resampling keeps
        # the full-suite GM exactly one; independent workload draws would not.
        rows = [dict(complete=True, process_baseline_ns=v, process_candidate_ns=[1.,1.,1.]) for v in ([2., 4., 8.], [.5, .25, .125])]
        report = aggregate_pairs(rows, 3, samples=1000)
        self.assertEqual(report["ci95"], [1., 1.])
        self.assertIsNone(aggregate_pairs([dict(complete=False)], 3)["speedup"])

    def test_source_chain_and_formation_do_not_prove_actual_multi_node(self):
        no_hit = dict(active=False, node_histogram={})
        single = dict(active=True, node_histogram={1: 10})
        multi = dict(active=True, node_histogram={2: 1})
        self.assertEqual(classify([no_hit]), "inactive")
        self.assertEqual(classify([single]), "single_node")
        self.assertEqual(classify([single, multi]), "multi_node")

    def test_single_mode_rejects_actual_multi_node_inventory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            spec = dict(name="test", kind="npbench")
            save(output / "manifest.json", dict(specs=[spec], configuration=dict(trials=1)))
            empty = dict(host={}, provider={})
            final = dict(host={"formed_regions": 1}, provider=dict(exec_hit=2, generated_exec_hit=2,
                observed_signatures=[dict(canonical_signature_hash=1, node_count=2, generated_hit_count=2)]))
            save(output / "raw" / f"{spec_id(spec)}.single.t0.diag.json", dict(spec=spec, mode="single", trial=0,
                correct=True, repeat=2, initial=empty, preparation=empty, final=final))
            report = reach_report(output, ("single",))
            self.assertFalse(report["complete"])
            self.assertEqual(report["rows"][0]["multi_node_hits"], 2)
            self.assertIsNone(report["rows"][0]["verified_eliminated_objects"])

    def test_real_generator_storage_liveness_and_all_failure_stages(self):
        with tempfile.TemporaryDirectory() as directory:
            report = synthesize_evidence(directory, ROOT.parent)
            by_key = {(r["identity"], r["permissions"]): r for r in report["rows"]}
            on, off = by_key["strict-generated-chain", 1], by_key["strict-generated-chain", 0]
            self.assertLess(on["allocations"], off["allocations"])
            self.assertEqual(on["owner_transfers"], 2)
            self.assertEqual(on["failure_checks"], 6)
            self.assertEqual(on["physical_frees"], on["allocations"])
            self.assertEqual(by_key["fork-last-use", 1]["plan"]["reuse_source"][2], 255)
            self.assertEqual(by_key["no-private-reuse", 1]["allocations"], by_key["no-private-reuse", 0]["allocations"])

    def test_one_click_has_only_new_rqs_and_cases(self):
        args = SimpleNamespace(output=Path("/tmp/redesign-plan"), workloads=None, skip_secondary=False, policy_evidence=None)
        phases = stages(args, {})
        self.assertEqual([name for name, _ in phases], ["rq2-adi", "rq2", "rq3", "rq3-storage", "rq4-callback", "rq4-inactive", "rq4-guard"])
        self.assertFalse(any("run_rq1" in part or "run_rq3_microbench" in part for _, command in phases for part in command))
        args = argument_parser().parse_args(["--results", "/tmp/docker-plan", "--image", "production"])
        args.skip_secondary = True
        phases, cfg = redesigned_plan(args)
        self.assertEqual(len(phases), 1)
        self.assertNotIn("rq1", cfg["rqs"])
        self.assertIn("scripts/run_redesigned_evaluation.py", phases[0][1])


if __name__ == "__main__":
    unittest.main()
