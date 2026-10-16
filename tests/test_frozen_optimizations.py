import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
sys.path.insert(0, str(ROOT / "scripts"))
from frozen_optimizations import BUNDLE, digest, install_frozen, validate_bundle
from optimization_coverage import check_coverage


class FrozenOptimizationTests(unittest.TestCase):
    def test_bundle_retains_all_measured_inputs_and_relocates_native_evidence(self):
        manifest = validate_bundle()
        original = json.loads((BUNDLE / "native_policy.json").read_text())
        with tempfile.TemporaryDirectory() as temporary:
            directory = install_frozen(Path(temporary) / "arbitrary-host/numpy")
            relocated = json.loads((directory / "native_policy.json").read_text())
            for a, b in zip(original["records"], relocated["records"], strict=True):
                self.assertEqual(a["key"], b["key"])
                self.assertEqual(a["decision"], b["decision"])
                self.assertEqual(a["evidence"]["sha256"], b["evidence"]["sha256"])
                self.assertEqual(digest(Path(b["evidence"]["path"])), a["evidence"]["sha256"])
                self.assertTrue(Path(b["evidence"]["path"]).is_relative_to(directory))
            # The real NumPy validator checks complete-path paired measurements.
            generator = ROOT.parent / "numpy/numpy/_core/code_generators/generate_crossjit_native_policy.py"
            if generator.is_file():
                output = Path(temporary) / "native.h"
                subprocess.run([sys.executable, str(generator), "--policy", str(directory / "native_policy.json"),
                                "--output", str(output)], check=True)
                self.assertEqual(digest(output), manifest["expected_generated_hashes"]["generated_crossjit_native_policy.h"])
                self.assertIn("POLICY_COUNT 156", output.read_text())
        self.assertEqual(digest(BUNDLE / "native_policy.json"), manifest["files"]["native_policy.json"]["sha256"])

    def test_corrupt_measurement_is_rejected_before_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary) / "bundle"
            shutil.copytree(BUNDLE, bundle)
            evidence = next((bundle / "evidence").iterdir())
            evidence.write_text("changed measurement")
            with self.assertRaisesRegex(ValueError, "evidence drift"):
                install_frozen(Path(temporary) / "numpy", bundle)

    def manifest(self):
        return dict(configuration=dict(preset="S", jit_stress=1, warmup=300),
                    identity=dict(build=dict(frozen_optimization_policy=dict(
                        expected_S_stress1_activity=validate_bundle()["expected_S_stress1_activity"]))))

    def test_real_execution_is_required_even_if_workload_is_active(self):
        row = dict(workload="hdiff", active=True, spec=dict(overrides={}),
                   diagnostics=[dict(backend_hits=dict(generated=300, native=0), provider={})])
        result = check_coverage(self.manifest(), dict(rows=[row]))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["failures"][0]["missing_backend"], "native")
        row["diagnostics"][0]["backend_hits"]["native"] = 300
        self.assertEqual(check_coverage(self.manifest(), dict(rows=[row]))["status"], "passed")

    def test_normal_protocol_is_not_compared_with_stress_coverage(self):
        manifest = self.manifest()
        manifest["configuration"]["jit_stress"] = 0
        self.assertEqual(check_coverage(manifest, dict(rows=[]))["status"], "not_applicable")


if __name__ == "__main__":
    unittest.main()
