import io
import json
import os
import sys
import unittest
from unittest import mock

from helpers import SIMPLE_SPEC, TempHome

from autoexp import handoff
from autoexp.brief import bootstrap_ci, iqm, render_brief
from autoexp.engine import Engine
from autoexp.events import EventLog, State
from autoexp.hooks import run_hook
from autoexp.slurm import FakeBackend
from autoexp.templates import BEGIN, install_project_files


class HandoffTests(unittest.TestCase):
    def setUp(self):
        self.t = TempHome()
        self.cfg = self.t.cfg()
        self.log = EventLog(self.cfg)

    def tearDown(self):
        self.t.close()

    def test_session_baton_and_handoff(self):
        sid = handoff.session_start(self.log, "codex", "gpt-x", purpose="debug the canary")
        handoff.baton_write(self.cfg, self.log, sid, {"goal": "debug", "next": ["rerun T1"],
                                                       "open_questions": ["threshold per project?"]})
        state = State.load(self.log)
        self.assertTrue(state.sessions[sid]["baton"])
        text = handoff.render_handoff(self.cfg, state)
        self.assertIn("NOT RUNNING", text)
        self.assertIn("rerun T1", text)
        self.assertIn("threshold per project?", text)
        self.assertTrue((self.cfg.sessions_dir / f"{sid}.baton.md").exists())
        with self.assertRaises(ValueError):
            handoff.baton_write(self.cfg, self.log, sid, {"mood": "great"})

    def test_task_leases(self):
        tid = handoff.task_add(self.log, "check the smoke", tier="any", accept="smoke succeeded")
        handoff.task_claim(self.log, State.load(self.log), tid, "s-a")
        with self.assertRaisesRegex(ValueError, "held by s-a"):
            handoff.task_claim(self.log, State.load(self.log), tid, "s-b")
        strong = handoff.task_add(self.log, "plan the next campaign", tier="strong")
        text = handoff.render_handoff(self.cfg, State.load(self.log), tier="any")
        self.assertIn("check the smoke", text)
        self.assertNotIn("plan the next campaign", text)
        self.assertIn("need a stronger tier", text)
        self.assertTrue(strong)

    def test_unclean_session_gets_reconstructed_baton(self):
        sid = handoff.session_start(self.log, "claude", "small-model")
        self.cfg.data["watch"]["session_unclean_hours"] = -1
        closed = handoff.close_unclean_sessions(self.cfg, self.log, State.load(self.log))
        self.assertEqual(closed, [sid])
        b = State.load(self.log).batons[-1]
        self.assertTrue(b["reconstructed"])

    def test_hooks_session_start_and_end(self):
        env_file = self.t.path / "claude_env"
        payload = json.dumps({"session_id": "abcdef1234567890", "model": "some-model"})
        out = io.StringIO()
        with mock.patch.dict(os.environ, {"CLAUDE_ENV_FILE": str(env_file)}), \
                mock.patch.object(sys, "stdin", io.StringIO(payload)), mock.patch.object(sys, "stdout", out):
            run_hook(self.cfg, "session-start", "claude")
        msg = json.loads(out.getvalue())
        ctx = msg["hookSpecificOutput"]["additionalContext"]
        self.assertIn("claude-abcdef123456", ctx)
        self.assertIn("HANDOFF", ctx)
        self.assertIn("AUTOEXP_SESSION=claude-abcdef123456", env_file.read_text())
        with mock.patch.object(sys, "stdin", io.StringIO(payload)):
            run_hook(self.cfg, "session-end", "claude")
        state = State.load(self.log)
        s = state.sessions["claude-abcdef123456"]
        self.assertTrue(s["baton"] and s["ended"])

    def test_hook_never_raises(self):
        with mock.patch.object(sys, "stdin", io.StringIO("not json")):
            self.assertEqual(run_hook(self.cfg, "bogus", "codex"), 0)

    def test_project_files(self):
        proj = self.t.path / "proj"
        proj.mkdir()
        (proj / "AGENTS.md").write_text("# my project\n\nown notes\n")
        install_project_files(proj)
        install_project_files(proj)  # idempotent
        text = (proj / "AGENTS.md").read_text()
        self.assertEqual(text.count(BEGIN), 1)
        self.assertIn("own notes", text)
        self.assertEqual((proj / "CLAUDE.md").read_text(), "@AGENTS.md\n")


class StatsTests(unittest.TestCase):
    def test_iqm_and_ci(self):
        self.assertAlmostEqual(iqm([1, 2, 3, 100]), 2.5)
        lo, hi = bootstrap_ci([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertLess(lo, 3.0)
        self.assertGreater(hi, 3.0)

    def test_brief_renders(self):
        t = TempHome()
        try:
            cfg = t.cfg()
            spec = t.write("demo.yaml", SIMPLE_SPEC)
            eng = Engine(cfg, FakeBackend(), echo=lambda m: None)
            eng.create_campaign(spec)
            eng.tick()
            text = render_brief(cfg, eng.state, {"demo": eng.spec("demo").data})
            self.assertIn("## demo [active]", text)
        finally:
            t.close()


if __name__ == "__main__":
    unittest.main()
