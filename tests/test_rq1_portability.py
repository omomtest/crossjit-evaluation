import io
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from common import digest, read_json, save
from collect_rq1_results import import_bundle, tables
from rq1_bundle import bundle, unpack
import run_rq1_machine as machine


class PortableRQ1Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.run = self.root / "run"
        save(self.run / "metadata/host.json", dict(host_os="macos", hardware_label="test CPU"))
        (self.run / "experiment/rq1/raw").mkdir(parents=True)
        (self.run / "experiment/rq1/raw/a.json").write_text('{"correct": true}\n')
        self.archive = self.root / "machine--run.tar.gz"

    def tearDown(self):
        self.tmp.cleanup()

    def test_roundtrip_deduplicates_identical_imports_and_keeps_raw_bytes(self):
        bundle(self.run, self.archive, "mac-test", "rq1-test")
        destination = import_bundle(self.archive, self.root / "collection")
        self.assertEqual((destination / "experiment/rq1/raw/a.json").read_bytes(),
                         (self.run / "experiment/rq1/raw/a.json").read_bytes())
        self.assertEqual(import_bundle(self.archive, self.root / "collection"), destination)
        (destination / "experiment/rq1/raw/a.json").write_text("changed")
        with self.assertRaisesRegex(ValueError, "evidence changed"):
            import_bundle(self.archive, self.root / "collection")

    def test_different_run_cannot_overwrite_existing_evidence(self):
        bundle(self.run, self.archive, "mac-test", "rq1-test")
        import_bundle(self.archive, self.root / "collection")
        (self.run / "experiment/rq1/raw/a.json").write_text("changed")
        bundle(self.run, self.archive, "mac-test", "rq1-test")
        with self.assertRaisesRegex(ValueError, "different evidence"):
            import_bundle(self.archive, self.root / "collection")

    def test_archive_transfer_corruption_is_rejected(self):
        bundle(self.run, self.archive, "mac-test", "rq1-test")
        with self.archive.open("ab") as f:
            f.write(b"transfer corruption")
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            unpack(self.archive, self.root / "destination")
        self.assertFalse((self.root / "destination").exists())

    def malicious_archive(self, name, kind=tarfile.REGTYPE, file_hash=False):
        with tarfile.open(self.archive, "w:gz") as tar:
            member = tarfile.TarInfo(name)
            member.type = kind
            member.linkname = "/tmp/elsewhere"
            member.size = 1 if kind == tarfile.REGTYPE else 0
            tar.addfile(member, io.BytesIO(b"x") if kind == tarfile.REGTYPE else None)
            if file_hash:
                import json
                content = json.dumps(dict(schema_version=1, files={name: dict(size=1, sha256="wrong")})).encode()
                index = tarfile.TarInfo("bundle.json")
                index.size = len(content)
                tar.addfile(index, io.BytesIO(content))
        self.archive.with_name(self.archive.name + ".sha256").write_text(digest(self.archive))

    def test_traversal_symlinks_and_noncanonical_paths_are_rejected(self):
        for name, kind in [("../escape", tarfile.REGTYPE), ("/absolute", tarfile.REGTYPE),
                           ("a/./b", tarfile.REGTYPE), ("a\\b", tarfile.REGTYPE),
                           ("link", tarfile.SYMTYPE), ("hardlink", tarfile.LNKTYPE)]:
            self.malicious_archive(name, kind)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "unsafe"):
                unpack(self.archive, self.root / "destination")
        self.assertFalse((self.root / "escape").exists())

    def test_per_file_hash_is_checked_even_with_valid_archive_checksum(self):
        self.malicious_archive("data.json", file_hash=True)
        with self.assertRaisesRegex(ValueError, "file hash mismatch"):
            unpack(self.archive, self.root / "destination")

    def test_bundle_excludes_checkouts_and_keeps_incomplete_runs(self):
        (self.run / "source-context").mkdir()
        (self.run / "source-context/large-source.c").write_text("source")
        index = bundle(self.run, self.archive, "mac-test", "rq1-test")
        self.assertNotIn("source-context/large-source.c", index["files"])
        output = self.root / "collection"
        import_bundle(self.archive, output)
        rows = tables(output)
        self.assertEqual(rows[0]["status"], "incomplete")
        self.assertTrue((output / "machines.csv").is_file())

    def test_machine_comparisons_retain_separate_rows_and_subset_status(self):
        cfg = dict(warmup=300, jit_stress=1, trials=10, repeat=100, blas_threads=6,
                   preset="S", seed=20261002, timeout=1800)
        save(self.run / "experiment/rq1/manifest.json", dict(configuration=cfg,
             identity=dict(build={}, scripts_hash="s", suite_hash="suite"),
             policy_mode="fixed_policy_migration", system=dict(machine="arm64")))
        save(self.run / "experiment/pipeline.json", dict(complete=True))
        save(self.run / "experiment/rq1/summary.json", dict(complete=True, full_53_npbench=False,
             measurement_status="jit_stress_fixed_policy_migration_interim_explicit_warmup",
             GM_all=None, GM_requested=1.2, suite_GM_ci95=[1.1, 1.3], active_total=1,
             platform_execution_confirmed=True, equivalent_performance_comparison_eligible=True,
             rows=[dict(workload="test", speedup=1.2, diagnostics=[])]))
        output = self.root / "collection"
        for name in ("mac-test", "windows-test"):
            bundle(self.run, self.archive, name, "rq1-test")
            import_bundle(self.archive, output)
        rows = tables(output)
        self.assertEqual({row["machine"] for row in rows}, {"mac-test", "windows-test"})
        self.assertTrue(all(not row["fixed_protocol_comparison_eligible"] for row in rows))
        self.assertEqual(len(read_json(output / "index.json")["runs"]), 2)

    def test_native_architecture_mismatch_rejected_before_build(self):
        with patch.object(machine.platform, "machine", return_value="arm64"), \
                patch.object(machine, "capture", return_value="linux/x86_64"):
            with self.assertRaisesRegex(ValueError, "native"):
                machine.native_platform()

    def test_dry_run_does_not_call_docker_or_create_workspace(self):
        workspace = self.root / "new-workspace"
        with patch.object(sys, "argv", ["run_rq1_machine.py", "--machine", "test",
             "--workspace", str(workspace), "--dry-run"]), \
             patch.object(machine, "capture") as capture, patch.object(machine, "run") as run:
            machine.main()
        capture.assert_not_called()
        run.assert_not_called()
        self.assertFalse(workspace.exists())

    def test_real_source_mirrors_are_portable_and_verify_without_network(self):
        sys.path.insert(0, str(ROOT / "build"))
        from local_source_context import mirror
        from sources import git
        source = self.root / "local-git"
        source.mkdir()
        subprocess.run(["git", "init", "--quiet", str(source)], check=True)
        (source / "source.c").write_text("frozen source\n")
        subprocess.run(["git", "-C", str(source), "add", "."], check=True)
        subprocess.run(["git", "-C", str(source), "-c", "user.name=Test", "-c", "user.email=test@example.invalid",
                        "-c", "commit.gpgsign=false", "commit", "--quiet", "-m", "fixture"], check=True)
        commit = git(source, "rev-parse", "HEAD")
        mirror(source, commit, self.root / "source-context/repo", "https://example.invalid/repo.git")
        self.assertEqual(git(self.root / "source-context/repo", "rev-parse", "HEAD"), commit)
        self.assertEqual((self.root / "source-context/repo/source.c").read_text(), "frozen source\n")


if __name__ == "__main__":
    unittest.main()
