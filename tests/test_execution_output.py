"""Exercise logging with real workers, validated resume, and failure scheduling."""
from contextlib import redirect_stdout
import io
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import read_json, save
import execution
import redesign_execution
import run_rq2_coverage as coverage


class ExecutionOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="evaluation output ")
        self.output = Path(self.tmp.name)
        self.spec = dict(kind="overhead", name="no_provider")
        self.cfg = dict(baseline_python=sys.executable, candidate_python=sys.executable,
                        npbench=str(ROOT), seed=7, warmup=2, repeat=3, trials=2,
                        blas_threads=1, cpu_affinity=None, timeout=30, jit_stress=0)
        self.manifest = dict(experiment="rq4-attribution", configuration=self.cfg,
                             specs=[self.spec], identity={},
                             variants=[dict(name="ablation", environment={"NPY_CROSSJIT_TEST": "1"})])
        # Identity probing needs the production installation. Output checking,
        # independent references, saved inputs, and workers themselves stay real.
        self.identity_patch = patch.object(execution, "verify_identity")
        self.identity_patch.start()

    def tearDown(self):
        self.identity_patch.stop()
        self.tmp.cleanup()

    def matrix(self, **options):
        output = io.StringIO()
        with redirect_stdout(output):
            cells = execution.run_matrix(self.output, self.manifest, **options)
        return cells, output.getvalue()

    def progress_headers(self, output):
        return re.findall(r"-\((\d+)/(\d+)\) mode=", output)

    def test_actual_commands_results_and_validated_resume(self):
        modes = ("baseline", "off", "on", "ablation")
        with patch.object(execution.subprocess, "run", wraps=subprocess.run) as launched:
            cells, output = self.matrix(modes=modes)
        self.assertEqual(len(cells), 14)
        self.assertEqual(self.progress_headers(output), [(str(n), "15") for n in range(1, 16)])
        self.assertIn("rq4-attribution-baseline-no_provider-(1/15)", output)
        self.assertIn("rq4-attribution-crossjit-no_provider-", output)
        self.assertIn("mode=off phase=timing", output)
        self.assertIn("mode=ablation phase=timing", output)
        self.assertEqual(output.count("median="), 8)
        commands = [shlex.split(line.split(" command: ", 1)[1]) for line in output.splitlines() if " command: " in line]
        for displayed, call in zip(commands, launched.call_args_list, strict=True):
            argv = call.args[0]
            self.assertEqual(displayed[-len(argv):], argv)
            for assignment in displayed[1:-len(argv)]:
                key, value = assignment.split("=", 1)
                self.assertEqual(call.kwargs["env"][key], value)
        for cell in cells:
            if not cell["diagnostics"]:
                self.assertIn(f"median={cell['median_ns']/1e6:.6f} ms", output)
        saved = {p: p.read_bytes() for p in (self.output / "raw").glob("*.json")}
        with patch.object(execution.subprocess, "run") as launched:
            resumed_cells, resumed = self.matrix(modes=modes)
        launched.assert_not_called()
        self.assertEqual(resumed_cells, cells)
        self.assertEqual(saved, {p: p.read_bytes() for p in saved})
        self.assertEqual(resumed.count("cached (validated)"), 15)
        self.assertEqual(self.progress_headers(resumed), self.progress_headers(output))
        self.assertEqual(resumed.count("median="), 8)

    def test_reference_failure_accounts_for_skipped_tasks(self):
        failure = subprocess.TimeoutExpired([sys.executable], 30)
        with patch.object(execution.subprocess, "run", side_effect=failure) as launched:
            cells, output = self.matrix()
        self.assertEqual(cells, [])
        self.assertEqual(launched.call_count, 1)
        self.assertIn("rq4-attribution-baseline-no_provider-(1/7) failed:", output)
        self.assertEqual(output.count("skipped:"), 6)
        self.assertIn("-(7/7) skipped:", output)
        failure_path, = (self.output / "failures").glob("*.json")
        self.assertIn("timed out", read_json(failure_path)["error"])
        self.assertTrue(Path(read_json(failure_path)["log"]).is_file())

    def test_corrupt_resume_is_rejected_before_success_output(self):
        self.matrix(diagnostics=False)
        name = execution.spec_id(self.spec)
        path = self.output / "raw" / f"{name}.baseline.treference.timing.json"
        cell = read_json(path)
        save(path, dict(cell, task_hash="corrupted"))
        output = io.StringIO()
        with redirect_stdout(output), self.assertRaisesRegex(RuntimeError, "identity/correctness"):
            execution.run_matrix(self.output, self.manifest, diagnostics=False)
        self.assertNotIn("cached (validated)", output.getvalue())

    def test_rq2_counts_references_and_diagnostics_only(self):
        self.manifest["experiment"] = "rq2"
        output = io.StringIO()
        with patch.object(coverage, "configuration", return_value=self.cfg), \
             patch.object(coverage, "workload_specs", return_value=[self.spec]), \
             patch.object(coverage, "freeze", return_value=(self.output, self.manifest)), \
             patch.object(coverage, "coverage_report", return_value=dict(complete_cells=2, expected_cells=2)), \
             patch.object(coverage, "reach_report", return_value=dict(complete=True)), \
             patch.object(redesign_execution, "verify_identity"), \
             patch.object(sys, "argv", ["run_rq2_coverage.py", "--output", str(self.output)]), \
             redirect_stdout(output):
            coverage.main()
        text = output.getvalue()
        self.assertEqual(self.progress_headers(text), [("1", "3"), ("2", "3"), ("3", "3")])
        self.assertIn("rq2-crossjit-no_provider-(3/3)", text)
        self.assertNotIn("median=", text)


if __name__ == "__main__":
    unittest.main()
