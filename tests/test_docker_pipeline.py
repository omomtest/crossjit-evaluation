import argparse
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import clean_env, configuration, freeze, parser, read_json, runtime_environment, save
from execution import variant_modes, verify_cell
import run_docker_pipeline as pipeline


class StressProtocolTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.config = self.root / "config.json"
        self.cfg = dict(read_json(ROOT / "config/experiment.json"), warmup=2, jit_stress=0)
        save(self.config, self.cfg)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cli_overrides_config_and_reaches_fresh_process(self):
        for stress in (0, 1):
            args = parser("test").parse_args(["--config", str(self.config), "--output", str(self.root),
                                               "--warmup", "7", "--jit-stress", str(stress)])
            cfg = configuration(args)
            self.assertEqual(cfg["warmup"], 7)
            for name, mode in variant_modes(cfg, [{"name": "ablation", "environment": {"PYTHON_JIT_STRESS": "wrong"}}]).items():
                with self.subTest(stress=stress, mode=name), patch.dict("os.environ", {"PYTHON_JIT_STRESS": "ambient", "NPY_CROSSJIT_MIN_WORK": "0"}):
                    env = clean_env(extra=mode["environment"])
                    self.assertNotIn("NPY_CROSSJIT_MIN_WORK", env)
                    value = subprocess.check_output([sys.executable, "-c", "import os; print(os.environ['PYTHON_JIT_STRESS'])"], env=env, text=True)
                    self.assertEqual(value.strip(), str(stress))

    def test_ambient_stress_is_not_a_protocol_setting(self):
        with patch.dict("os.environ", {"PYTHON_JIT_STRESS": "1"}):
            self.assertNotIn("PYTHON_JIT_STRESS", clean_env())
            self.assertEqual(clean_env(extra=runtime_environment(self.cfg))["PYTHON_JIT_STRESS"], "0")

    def test_invalid_config_stress_rejected(self):
        for value in (2, -1, "1", True):
            save(self.config, dict(self.cfg, jit_stress=value))
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "jit_stress"):
                configuration(argparse.Namespace(config=self.config))

    def test_explicit_confirmed_protocol_overrides_fixed_default(self):
        protocol = self.root / "protocol.json"
        save(protocol, dict(state="confirmed", W_star=500))
        save(self.config, dict(self.cfg, warmup=300))
        args = parser("test").parse_args(["--config", str(self.config), "--output", str(self.root),
                                         "--warmup-protocol", str(protocol)])
        cfg = configuration(args)
        self.assertEqual(cfg["warmup"], 500)
        self.assertTrue(cfg["warmup_from_protocol"])
        args.warmup = 700
        self.assertEqual(configuration(args)["warmup"], 700)
        args.warmup = None
        save(protocol, dict(state="pending", W_star=500))
        with self.assertRaisesRegex(ValueError, "has not been confirmed"):
            configuration(args)

    def test_cell_verifies_actual_worker_stress(self):
        with self.assertRaisesRegex(RuntimeError, "worker JIT stress"):
            verify_cell(dict(task_hash="x", correct=True, jit_stress=0), dict(task_hash="x", jit_stress=1), None)

    def test_stress_change_cannot_resume_and_has_explicit_label(self):
        fake = dict(build=None, probes={name: dict(jit_enabled=True, provider_api=True, host_api=True) for name in ("baseline", "candidate")})
        with patch("common.identity", return_value=fake):
            _, manifest = freeze(self.root / "results", dict(self.cfg, jit_stress=1), "rq1", [])
            self.assertTrue(manifest["measurement_status"].startswith("jit_stress_"))
            with self.assertRaisesRegex(RuntimeError, "different identity/configuration"):
                freeze(self.root / "results", dict(self.cfg, jit_stress=0), "rq1", [])

    def test_calibration_trigger_mismatch_is_recorded(self):
        platform = dict(machine="x", cpu="y", kernel="z")
        fake = dict(probes={name: dict(jit_enabled=True, provider_api=True, host_api=True) for name in ("baseline", "candidate")},
                    platform_identity=platform, build=dict(calibration_receipt=dict(source_identity=dict(platform_identity=platform, runtime_protocol=dict(jit_stress=0)))))
        with patch("common.identity", return_value=fake):
            _, manifest = freeze(self.root / "results", dict(self.cfg, jit_stress=1), "rq1", [])
        self.assertFalse(manifest["calibration_runtime_protocol_matches"])
        self.assertEqual(manifest["policy_mode"], "fixed_policy_migration")


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.mapping = self.root / "checkouts.json"
        self.mapping.write_text("{}")

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *options):
        return pipeline.argument_parser().parse_args(["--results", str(self.root / "results"), "--checkouts", str(self.mapping),
                                                       "--warmup", "17", *options])

    def test_calibration_builds_before_rqs_and_uses_one_snapshot(self):
        commands, cfg = pipeline.plan(self.args("--calibrate", "--jit-stress", "1"))
        self.assertEqual([label for label, _ in commands], ["snapshot", "build-calibration", "calibration", "build-production",
                                                           "rq1", "rq2", "rq2-adi", "rq3", "rq3-storage", "rq4-callback", "rq4-inactive", "rq4-guard"])
        builds = [cmd for label, cmd in commands if label.startswith("build")]
        self.assertEqual(builds[0][builds[0].index("--source-context") + 1], builds[1][builds[1].index("--source-context") + 1])
        for label, cmd in commands:
            if label == "calibration" or label.startswith("rq") and label != "rq4-guard":
                self.assertEqual(cmd[cmd.index("--jit-stress") + 1], "1")
            if label.startswith("rq") and label != "rq4-guard":
                self.assertEqual(cmd[cmd.index("--warmup") + 1], "17")
        self.assertEqual(cfg["jit_stress"], 1)

    def test_no_calibration_builds_fixed_strategy(self):
        commands, _ = pipeline.plan(self.args("--no-calibrate"))
        labels = [label for label, _ in commands]
        self.assertEqual(labels[:2], ["snapshot", "build-fixed"])
        self.assertNotIn("calibration", labels)
        self.assertIn("fixed-strategy", commands[1][1])
        self.assertEqual(commands[1][1][commands[1][1].index("--optimization-preset") + 1], "v32")

    def test_legacy_generated_is_an_explicit_comparison(self):
        commands, cfg = pipeline.plan(self.args("--no-calibrate", "--optimization-preset", "legacy-generated"))
        self.assertEqual(commands[1][1][commands[1][1].index("--optimization-preset") + 1], "legacy-generated")
        self.assertEqual(cfg["optimization_preset"], "legacy-generated")

    def test_policy_reuse_skips_calibration(self):
        policy = self.root / "policy"
        policy.mkdir()
        (policy / "receipt.json").write_text("{}")
        commands, _ = pipeline.plan(self.args("--policy", str(policy)))
        self.assertEqual([label for label, _ in commands][:2], ["snapshot", "build-production"])
        self.assertIn(str(policy), commands[1][1])

    def test_existing_image_skips_build_and_mounts_updated_scripts(self):
        commands, _ = pipeline.plan(self.args("--image", "existing:image", "--rqs", "rq1", "--workloads", "go_fast"))
        self.assertEqual([label for label, _ in commands], ["rq1"])
        self.assertIn("--scripts-dir", commands[0][1])
        self.assertIn("go_fast", commands[0][1])

    def test_rq4_automatically_collects_full_rq2_first(self):
        commands, _ = pipeline.plan(self.args("--image", "existing:image", "--rqs", "rq4"))
        self.assertEqual([label for label, _ in commands], ["rq2", "rq2-adi", "rq3", "rq3-storage", "rq4-callback", "rq4-inactive", "rq4-guard"])
        self.assertEqual(commands[-2][1][-2:], ["--coverage", "/results/rq2/coverage.json"])

    def test_invalid_combinations_are_rejected(self):
        for options in [("--image", "existing:image", "--calibrate"), ("--workloads", "go_fast")]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                pipeline.plan(self.args(*options))

    def test_dry_run_has_no_side_effects(self):
        args = self.args("--dry-run")
        with patch.object(pipeline, "argument_parser") as parser_mock, patch.object(pipeline.subprocess, "Popen") as process_mock, patch("builtins.print"):
            parser_mock.return_value.parse_args.return_value = args
            pipeline.main()
        process_mock.assert_not_called()
        self.assertFalse(args.results.exists())

    def test_failed_build_stops_before_experiments_and_is_saved(self):
        args = self.args()
        commands, cfg = pipeline.plan(args)
        process = unittest.mock.Mock(stdout=iter(["failure\n"]))
        process.wait.return_value = 7
        with patch.object(pipeline, "argument_parser") as parser_mock, patch.object(pipeline, "plan", return_value=([commands[1], commands[2]], cfg)), patch.object(pipeline.subprocess, "Popen", return_value=process) as process_mock, patch("builtins.print"):
            parser_mock.return_value.parse_args.return_value = args
            with self.assertRaises(SystemExit) as exit_context:
                pipeline.main()
        self.assertEqual(exit_context.exception.code, 7)
        self.assertEqual(process_mock.call_count, 1)
        result = read_json(args.results / "pipeline.json")
        self.assertFalse(result["complete"])
        self.assertEqual(result["phases"][0]["exitcode"], 7)

    def test_resume_preserves_completed_calibration_and_policy(self):
        args = self.args("--calibrate")
        commands, cfg = pipeline.plan(args)
        args.results.mkdir()
        phases = [dict(phase=label, exitcode=0) for label, _ in commands if not label.startswith("rq")]
        save(args.results / "pipeline.json", dict(configuration=cfg, phases=phases, complete=False))
        evidence = args.results / "evidence.json"
        evidence.write_text('"frozen measurement"')
        process = unittest.mock.Mock(stdout=iter([]))
        process.wait.return_value = 0
        with patch.object(pipeline, "argument_parser") as parser_mock, patch.object(pipeline, "plan", return_value=(commands, cfg)), patch.object(pipeline.subprocess, "Popen", return_value=process) as process_mock, patch("builtins.print"):
            parser_mock.return_value.parse_args.return_value = args
            pipeline.main()
        self.assertEqual(process_mock.call_count, 8)
        self.assertTrue(all("scripts/docker_run.py" in call.args[0][1] for call in process_mock.call_args_list))
        self.assertEqual(evidence.read_text(), '"frozen measurement"')
        self.assertTrue(read_json(args.results / "pipeline.json")["complete"])


if __name__ == "__main__":
    unittest.main()
