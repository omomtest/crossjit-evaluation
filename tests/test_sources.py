from contextlib import nullcontext
import os
from pathlib import Path
import subprocess
import signal
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "build"))
import sources


class NetworkRetryTests(unittest.TestCase):
    def test_failures_retry_until_success_with_capped_backoff(self):
        command = ["git", "fetch"]
        failures = [subprocess.CalledProcessError(128, command)] * 8
        success = subprocess.CompletedProcess(command, 0)
        with patch.object(sources, "run_network_command", side_effect=[*failures, success]) as run, \
                patch.object(sources, "download_guard", return_value=nullcontext(Path("/source/.git"))), \
                patch.object(sources, "recover_git_locks", return_value=True), \
                patch.object(sources.time, "sleep") as sleep, patch("builtins.print"), \
                patch.dict(os.environ, {"GIT_SSH_COMMAND": "ssh -o StrictHostKeyChecking=yes"}):
            self.assertIs(sources.network_git("/source", "fetch", "origin", "commit"), success)
        self.assertEqual(run.call_count, 9)
        self.assertEqual([call.args[0] for call in sleep.call_args_list], [5, 10, 20, 40, 60, 60, 60, 60])
        env = run.call_args.args[1]
        self.assertEqual(env["GIT_TERMINAL_PROMPT"], "0")
        self.assertIn("StrictHostKeyChecking=yes", env["GIT_SSH_COMMAND"])
        self.assertIn("ConnectTimeout=30", env["GIT_SSH_COMMAND"])

    def test_progressing_download_has_no_wall_clock_timeout(self):
        with patch.object(sources.subprocess, "Popen") as popen:
            popen.return_value.wait.return_value = 0
            sources.run_network_command(["git", "fetch"], {})
        popen.return_value.wait.assert_called_once_with()
        self.assertTrue(popen.call_args.kwargs["start_new_session"])

    def test_user_can_interrupt_retry(self):
        with patch.object(sources, "run_network_command", side_effect=KeyboardInterrupt) as run, \
                patch.object(sources, "download_guard", return_value=nullcontext(Path("/source/.git"))), \
                patch.object(sources, "recover_git_locks", return_value=True), \
                patch.object(sources.time, "sleep") as sleep:
            with self.assertRaises(KeyboardInterrupt):
                sources.network_git("/source", "fetch", "origin", "commit")
        self.assertEqual(run.call_count, 1)
        sleep.assert_not_called()

    def test_live_git_locks_are_preserved_then_recovered_when_git_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            gitdir = Path(directory)
            lock = gitdir / "shallow.lock"
            lock.write_text("pending transaction\n")
            child = gitdir / "modules" / "child" / "index.lock"
            child.parent.mkdir(parents=True)
            child.write_text("pending submodule checkout\n")
            with patch.object(sources.subprocess, "check_output", return_value="S /usr/bin/git\n"):
                self.assertFalse(sources.recover_git_locks(gitdir))
            self.assertTrue(lock.exists())
            self.assertTrue(child.exists())
            with patch.object(sources.subprocess, "check_output", return_value="Z git\nS python3\n"):
                self.assertTrue(sources.recover_git_locks(gitdir))
            self.assertFalse(lock.exists())
            self.assertFalse(child.exists())

    def test_lock_recovery_fails_closed_if_process_check_is_unavailable(self):
        with tempfile.TemporaryDirectory() as directory:
            gitdir = Path(directory)
            lock = gitdir / "shallow.lock"
            lock.touch()
            with patch.object(sources.subprocess, "check_output", side_effect=FileNotFoundError):
                with self.assertRaisesRegex(RuntimeError, "cannot check"):
                    sources.recover_git_locks(gitdir)
            self.assertTrue(lock.exists())

    def test_interrupt_stops_real_transport_children_and_allows_git_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "worker.py"
            worker.write_text('''import os, signal, subprocess, sys, time
from pathlib import Path
root = Path(sys.argv[1])
lock = root / "shallow.lock"
lock.touch()
def stop(signum, frame):
    lock.unlink(missing_ok=True)
    raise SystemExit(0)
signal.signal(signal.SIGTERM, stop)
child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
(root / "ready").write_text(str(child.pid))
time.sleep(60)
''')
            driver = root / "driver.py"
            driver.write_text('''import os, sys
sys.path.insert(0, sys.argv[1])
import sources
try:
    sources.run_network_command([sys.executable, sys.argv[2], sys.argv[3]], os.environ.copy())
except KeyboardInterrupt:
    raise SystemExit(130)
''')
            process = subprocess.Popen([sys.executable, str(driver), str(ROOT / "build"), str(worker), str(root)],
                                       start_new_session=True)
            try:
                deadline = time.monotonic() + 5
                while not (root / "ready").exists() and time.monotonic() < deadline:
                    time.sleep(0.02)
                self.assertTrue((root / "ready").exists(), "fake transport did not start")
                child_pid = int((root / "ready").read_text())
                process.send_signal(signal.SIGINT)
                self.assertEqual(process.wait(timeout=5), 130)
                self.assertFalse((root / "shallow.lock").exists())
                child = subprocess.run(["ps", "-p", str(child_pid), "-o", "stat="],
                                       text=True, stdout=subprocess.PIPE)
                self.assertTrue(child.returncode != 0 or child.stdout.strip().startswith("Z"),
                                "transport child survived cancellation")
            finally:
                if process.poll() is None:
                    process.kill()
                process.wait()


class SourceCheckoutTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.origin = self.repository("origin")
        self.commit = sources.git(self.origin, "rev-parse", "HEAD")

    def run_git(self, path, *args):
        subprocess.run(["git", "-C", str(path), *args], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def repository(self, name):
        path = self.root / name
        subprocess.run(["git", "init", "-q", str(path)], check=True)
        self.run_git(path, "config", "user.name", "Source test")
        self.run_git(path, "config", "user.email", "source-test@example.invalid")
        (path / "input.txt").write_text(name + "\n")
        self.run_git(path, "add", "input.txt")
        self.run_git(path, "commit", "-qm", "initial")
        return path

    def test_checkout_recovers_fetch_and_recursive_submodule_failures(self):
        leaf = self.repository("leaf")
        middle = self.repository("middle")
        for parent, child in ((middle, leaf), (self.origin, middle)):
            self.run_git(parent, "-c", "protocol.file.allow=always", "submodule", "add", str(child), "child")
            self.run_git(parent, "commit", "-qam", "add submodule")
        commit = sources.git(self.origin, "rev-parse", "HEAD")
        target = self.root / "checkout"
        real_run = sources.run_network_command
        failures = {"fetch": 1, "submodule": 1}

        def flaky_run(command, env):
            for operation in failures:
                if operation in command and failures[operation]:
                    failures[operation] -= 1
                    (target / ".git" / ("shallow.lock" if operation == "fetch" else "index.lock")).touch()
                    raise subprocess.CalledProcessError(128, command)
            return real_run(command, env)

        with patch.object(sources, "run_network_command", side_effect=flaky_run), \
                patch.object(sources.time, "sleep") as sleep, \
                patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            sources.checkout_one(str(self.origin), commit, target)
        self.assertEqual(sleep.call_count, 2)
        self.assertEqual(sources.git(target, "rev-parse", "HEAD"), commit)
        self.assertEqual(sources.gitlinks(target, commit), {
            "child": sources.git(middle, "rev-parse", "HEAD"),
            "child/child": sources.git(leaf, "rev-parse", "HEAD"),
        })
        with patch.object(sources, "network_git", wraps=sources.network_git) as network, \
                patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            sources.checkout_one(str(self.origin), commit, target)
        self.assertEqual(network.call_args.args[1], "submodule")
        self.assertEqual(network.call_count, 1)

    def test_resume_after_interruption_before_first_checkout(self):
        target = self.root / "checkout"
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        self.run_git(target, "remote", "add", "origin", str(self.origin))
        sources.checkout_one(str(self.origin), self.commit, target)
        self.assertEqual(sources.git(target, "rev-parse", "HEAD"), self.commit)

    def test_resume_after_init_before_remote_registration(self):
        target = self.root / "checkout"
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        sources.checkout_one(str(self.origin), self.commit, target)
        self.assertEqual(sources.git(target, "remote", "get-url", "origin"), str(self.origin))
        self.assertEqual(sources.git(target, "rev-parse", "HEAD"), self.commit)

    def test_submodule_retry_preserves_nonempty_clone_remnants_and_recovers(self):
        leaf = self.repository("leaf")
        self.run_git(self.origin, "-c", "protocol.file.allow=always", "submodule", "add", str(leaf), "child")
        self.run_git(self.origin, "commit", "-qam", "add submodule")
        commit = sources.git(self.origin, "rev-parse", "HEAD")
        target = self.root / "checkout"
        real_run = sources.run_network_command
        failed = False

        def partial_clone(command, env):
            nonlocal failed
            if "submodule" in command and not failed:
                failed = True
                child = target / "child"
                child.mkdir(exist_ok=True)
                (child / "partial-pack").write_text("interrupted clone\n")
                module = target / ".git" / "modules" / "child"
                subprocess.run(["git", "init", "-q", "--bare", str(module)], check=True)
                (module / "shallow.lock").touch()
                raise subprocess.CalledProcessError(128, command)
            return real_run(command, env)

        with patch.object(sources, "run_network_command", side_effect=partial_clone), \
                patch.object(sources.time, "sleep") as sleep, \
                patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            sources.checkout_one(str(self.origin), commit, target)
        sleep.assert_called_once_with(5)
        self.assertEqual(sources.git(target / "child", "rev-parse", "HEAD"), sources.git(leaf, "rev-parse", "HEAD"))
        backups = list((self.root / ".crossjit-source-recovery").glob("*/worktree/partial-pack"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), "interrupted clone\n")

    def test_submodule_wrong_head_is_resumed_but_local_edits_are_rejected(self):
        leaf = self.repository("leaf")
        pinned = sources.git(leaf, "rev-parse", "HEAD")
        self.run_git(self.origin, "-c", "protocol.file.allow=always", "submodule", "add", str(leaf), "child")
        self.run_git(self.origin, "commit", "-qam", "add submodule")
        commit = sources.git(self.origin, "rev-parse", "HEAD")
        target = self.root / "checkout"
        with patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            sources.checkout_one(str(self.origin), commit, target)
        child = target / "child"
        self.run_git(child, "config", "user.name", "Source test")
        self.run_git(child, "config", "user.email", "source-test@example.invalid")
        (child / "input.txt").write_text("another checked-out commit\n")
        self.run_git(child, "commit", "-qam", "different head")
        with patch.dict(os.environ, {"GIT_ALLOW_PROTOCOL": "file"}):
            sources.checkout_one(str(self.origin), commit, target)
        self.assertEqual(sources.git(child, "rev-parse", "HEAD"), pinned)
        (child / "input.txt").write_text("local modification\n")
        with patch.object(sources.time, "sleep") as sleep:
            with self.assertRaises(subprocess.CalledProcessError):
                sources.checkout_one(str(self.origin), commit, target)
        sleep.assert_not_called()
        self.assertEqual((child / "input.txt").read_text(), "local modification\n")

    def test_concurrent_checkout_waits_before_initializing_target(self):
        target = self.root / "checkout"
        driver = '''import sys, time
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import sources
real_sleep = time.sleep
sources.time.sleep = lambda delay: real_sleep(0.02)
sources.checkout_one(sys.argv[2], sys.argv[3], Path(sys.argv[4]))
'''
        with sources.checkout_guard(target):
            process = subprocess.Popen([sys.executable, "-c", driver, str(ROOT / "build"),
                                        str(self.origin), self.commit, str(target)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
            try:
                self.assertIn("Waiting for another source checkout", process.stderr.readline())
                self.assertFalse(target.exists())
            except BaseException:
                process.kill()
                process.wait()
                raise
        try:
            _, stderr = process.communicate(timeout=5)
            self.assertEqual(process.returncode, 0, stderr)
        finally:
            if process.poll() is None:
                process.kill()
            process.wait()
        self.assertEqual(sources.git(target, "rev-parse", "HEAD"), self.commit)

    def test_resume_recovers_legacy_shallow_lock_without_redownloading_other_sources(self):
        target = self.root / "checkout"
        subprocess.run(["git", "init", "-q", str(target)], check=True)
        self.run_git(target, "remote", "add", "origin", str(self.origin))
        lock = target / ".git" / "shallow.lock"
        lock.touch()
        real_output = subprocess.check_output

        def output(command, **kwargs):
            return "S python3\n" if command[0] == "ps" else real_output(command, **kwargs)

        with patch.object(sources.subprocess, "check_output", side_effect=output):
            sources.checkout_one(str(self.origin), self.commit, target)
        self.assertFalse(lock.exists())
        self.assertEqual(sources.git(target, "rev-parse", "HEAD"), self.commit)

    def test_concurrent_download_cannot_enter_same_repository(self):
        with sources.download_guard(self.origin):
            with self.assertRaisesRegex(ValueError, "another source download"):
                with sources.download_guard(self.origin):
                    self.fail("entered guard twice")

    def test_existing_mismatched_or_dirty_checkout_fails_without_retry(self):
        with patch.object(sources, "network_git") as network:
            with self.assertRaisesRegex(ValueError, "differs from lock"):
                sources.checkout_one(str(self.origin), "0" * 40, self.origin)
            (self.origin / "input.txt").write_text("uncommitted\n")
            with self.assertRaises(subprocess.CalledProcessError):
                sources.checkout_one(str(self.origin), self.commit, self.origin)
        network.assert_not_called()

    def test_remote_resolve_uses_requested_ref_and_records_recursive_lock(self):
        self.run_git(self.origin, "branch", "locked-ref")
        (self.origin / "input.txt").write_text("new head\n")
        self.run_git(self.origin, "commit", "-qam", "new head")
        config = {"repositories": {
            "moving": {"url": str(self.origin), "ref": "locked-ref"},
            "fixed": {"url": str(self.origin), "ref": self.commit, "fixed": True},
        }}
        lock = sources.resolve(config, {})
        for name, row in config["repositories"].items():
            self.assertEqual(lock["repositories"][name], dict(
                url=row["url"], requested_ref=row["ref"], commit=self.commit, submodules={}))


if __name__ == "__main__":
    unittest.main()
