"""Whistle console regressions; no model, board, or pyserial required."""
import io
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rich.console import Console

from tools import needle_tui as tui


class WhistleSessionTests(unittest.TestCase):
    def test_firmware_errors_leave_ready_or_transcribing_state(self):
        for events, message in (
            ([], "whistle_open rc=-2 (flash whistle_d4.cact to 0x210000)"),
            (["EVT recording", "EVT released held_ms=1000"], "encode"),
            (["EVT recording", "EVT released held_ms=1000"], "decode"),
        ):
            with self.subTest(message=message):
                sess = tui.WhistleSession()
                for event in events:
                    sess.feed(event)
                sess.feed("ERR " + message)
                self.assertEqual(sess.status, "error")
                self.assertEqual(sess.error, message)

    def test_error_clears_previous_led_action_and_renders_literal_message(self):
        sess = tui.WhistleSession()
        sess.feed("ACT led color=red mode=solid duration=3")
        message = "cannot open model[1].cact [bold]"
        sess.feed("ERR " + message)
        output = io.StringIO()
        Console(file=output, width=120, height=30, color_system=None).print(
            tui.render_whistle(sess, "test board")
        )
        self.assertIn(message, output.getvalue())
        self.assertIsNone(sess.action)

    def test_new_recording_or_ready_event_clears_error(self):
        for event, status in (
            ("EVT recording", "recording"),
            ("EVT ready model=whistle depth=4", "ready"),
            ("EVT  READY - hold the button on GPIO7 and say a command", "ready"),
        ):
            with self.subTest(event=event):
                sess = tui.WhistleSession()
                sess.feed("ERR encode")
                sess.feed(event)
                self.assertIsNone(sess.error)
                self.assertEqual(sess.status, status)

    def test_successful_turn_after_error(self):
        sess = tui.WhistleSession()
        for event in (
            "ERR encode", "EVT recording", "EVT released held_ms=1000",
            "TXT Make it blue.", "ACT led color=blue mode=solid duration=3",
        ):
            sess.feed(event)
        self.assertEqual(sess.status, "done")
        self.assertIsNone(sess.error)
        self.assertEqual(sess.text, "Make it blue.")
        self.assertEqual(sess.action["color"], "blue")


class WhistleConsoleTests(unittest.TestCase):
    def setUp(self):
        self.output = io.StringIO()
        console = Console(file=self.output, width=120, height=30, color_system=None)
        console_patch = patch.object(tui, "CONSOLE", console)
        console_patch.start()
        self.addCleanup(console_patch.stop)
        # Exercise Rich's real rendering without a background refresh thread.
        live = tui.Live
        live_patch = patch.object(
            tui, "Live", side_effect=lambda *a, **kw: live(*a, auto_refresh=False, **kw)
        )
        live_patch.start()
        self.addCleanup(live_patch.stop)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.folder = Path(folder.name)
        self.args = SimpleNamespace(
            serial=None, whistle_binary="unused", whistle_model="unused",
            local_wav="clip.wav",
        )

    def run_host(self, script):
        shell = shutil.which("sh")
        if not shell:
            self.skipTest("host-process tests require a POSIX shell")
        path = self.folder / "host.sh"
        path.write_text(script)
        self.args.whistle_binary = shell
        self.args.whistle_model = str(path)
        return tui.whistle_main(self.args)

    def test_host_error_is_visible_and_exit_status_is_preserved(self):
        status = self.run_host("printf 'cannot open missing.cact\\n' >&2\nexit 7\n")
        self.assertEqual(status, 7)
        self.assertIn("cannot open missing.cact", self.output.getvalue())

    def test_host_failure_without_diagnostics_still_reports_status(self):
        status = self.run_host("exit 2\n")
        self.assertEqual(status, 2)
        self.assertIn("host engine exited with status 2", self.output.getvalue())

    def test_normal_stderr_diagnostics_do_not_mark_success_as_error(self):
        status = self.run_host(
            "printf 'keywords: 100 token sequences\\n' >&2\n"
            "printf 'EVT ready model=whistle depth=4\\nEVT recording\\n"
            "EVT released held_ms=1000\\nTXT Make it blue.\\n"
            "ACT led color=blue mode=solid duration=3\\n'\n"
        )
        self.assertEqual(status, 0)
        self.assertIn("Make it blue.", self.output.getvalue())
        self.assertNotIn("keywords:", self.output.getvalue())

    def test_protocol_error_is_failure_even_if_host_exits_zero(self):
        status = self.run_host("printf 'EVT released held_ms=1000\\nERR encode\\n'\n")
        self.assertEqual(status, 1)
        self.assertIn("encode", self.output.getvalue())

    def test_no_speech_is_not_a_failure(self):
        status = self.run_host("printf 'EVT no_speech\\nTXT \\nACT none\\n'\n")
        self.assertEqual(status, 0)
        self.assertIn("nothing heard", self.output.getvalue())

    def test_missing_executable_reports_error_without_traceback(self):
        self.args.whistle_binary = str(self.folder / "missing-wh-run")
        self.assertEqual(tui.whistle_main(self.args), 1)
        self.assertIn("missing-wh-run", self.output.getvalue())

    def test_serial_open_error_is_visible(self):
        self.args.serial = "/dev/missing-board"
        with patch.object(tui, "SerialBackend", side_effect=OSError("cannot open board")):
            self.assertEqual(tui.whistle_main(self.args), 1)
        self.assertIn("cannot open board", self.output.getvalue())

    def test_serial_firmware_error_is_displayed_until_user_exits(self):
        self.args.serial = "/dev/test-board"
        backend = Mock(name="serial backend")
        backend.name = "test board"
        backend.q.get.side_effect = ["ERR mmap_failed size=14614528", KeyboardInterrupt]
        with patch.object(tui, "SerialBackend", return_value=backend):
            self.assertEqual(tui.whistle_main(self.args), 130)
        self.assertIn("mmap_failed size=14614528", self.output.getvalue())
        backend.close.assert_called_once()

    def test_interrupt_reaps_child_and_closes_streams(self):
        proc = Mock()
        proc.stdout = io.StringIO()
        proc.poll.return_value = None
        for needs_kill in (False, True):
            with self.subTest(needs_kill=needs_kill):
                proc.stdout = io.StringIO()
                proc.reset_mock()
                proc.wait.side_effect = (
                    [subprocess.TimeoutExpired("host", 2), 0] if needs_kill else None
                )
                with patch.object(tui.subprocess, "Popen", return_value=proc), \
                     patch.object(tui, "Live", side_effect=KeyboardInterrupt):
                    status = tui.whistle_main(self.args)
                self.assertEqual(status, 130)
                proc.terminate.assert_called_once()
                self.assertTrue(proc.wait.called)
                self.assertEqual(proc.kill.called, needs_kill)
                self.assertTrue(proc.stdout.closed)


if __name__ == "__main__":
    unittest.main()
