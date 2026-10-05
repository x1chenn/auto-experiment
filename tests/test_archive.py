import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock

from helpers import TempHome

from autoexp import archive
from autoexp.engine import Engine, job_dir
from autoexp.events import EventLog, State
from autoexp.hooks import run_hook
from autoexp.slurm import FakeBackend

SPEC = """\
name: arch
part: "part-1 baselines"
hypothesis: "lr=0.1 beats lr=0.2"
test: true
run_root: ./runs
command: "echo {lr} {seed}"
params: {lr: [0.1, 0.2]}
seeds: [0, 1, 2]
stages:
  - {name: smoke, auto: true}
  - {name: full, auto: false}
analysis: {primary: ret, result_file: final.json}
"""


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.t = TempHome()
        self.cfg = self.t.cfg()
        self.fake = FakeBackend()
        self.log = EventLog(self.cfg)
        eng = Engine(self.cfg, self.fake, echo=lambda m: None)
        eng.create_campaign(self.t.write("arch.yaml", SPEC))
        eng.tick()
        for a in list(eng.state.active_attempts()):
            run = eng.state.runs[a["run_id"]]
            jd = job_dir(run, a["job_id"])
            jd.mkdir(parents=True)
            (jd / "result.json").write_text(json.dumps({"status": "ok"}))
            value = 10.0 + run["seed"] if run["params"]["lr"] == 0.1 else 5.0 + run["seed"]
            Path(run["run_dir"], "final.json").write_text(json.dumps({"ret": value}))
            self.fake.set(a["job_id"], "COMPLETED")
        eng.tick()
        self.state = State.load(self.log)

    def tearDown(self):
        self.t.close()

    def test_stage_results_are_frozen(self):
        snap = self.state.campaigns["arch"]["results"]["smoke"]
        rows = snap["table"]["rows"]
        self.assertEqual(rows[0]["label"], "lr=0.1")
        self.assertAlmostEqual(rows[0]["median"], 11.0)
        self.assertEqual(len(rows[0]["runs"]), 3)
        self.assertEqual(self.state.stage("arch", "full")["status"], "awaiting_approval")

    def test_findings_need_evidence_and_numbers_are_checked(self):
        with self.assertRaisesRegex(ValueError, "needs evidence"):
            archive.add_finding(self.log, self.state, "something")
        with self.assertRaisesRegex(ValueError, "not a run id"):
            archive.add_finding(self.log, self.state, "x", evidence=["made-up"])
        fid, warn = archive.add_finding(self.log, self.state, "lr=0.1 has median 11.00 vs 6.00",
                                        campaign="arch", stage="smoke")
        self.assertEqual(warn, [])
        fid2, warn2 = archive.add_finding(self.log, State.load(self.log), "lr=0.1 reaches 42.5",
                                          campaign="arch", stage="smoke")
        self.assertTrue(warn2 and "42.5" in warn2[0])
        st = State.load(self.log)
        self.assertEqual(st.findings[fid]["status"], "tentative")
        archive.update_finding(self.log, st, fid, "supported", "3 seeds, CIs disjoint")
        fid3, _ = archive.add_finding(self.log, State.load(self.log), "refined claim", campaign="arch",
                                      stage="smoke", status="supported", supersedes=fid2)
        st = State.load(self.log)
        self.assertEqual(st.findings[fid]["status"], "supported")
        self.assertEqual(st.findings[fid2]["status"], "superseded")
        self.assertEqual(st.findings[fid2]["superseded_by"], fid3)
        self.assertEqual(st.findings[fid]["part"], "part-1 baselines")

    def test_archive_documents(self):
        fid, _ = archive.add_finding(self.log, self.state, "lr=0.1 beats lr=0.2", campaign="arch",
                                     stage="smoke", status="supported")
        self.log.append("note.added", {"text": "remember to try lr=0.05", "campaign": "arch"})
        self.log.append("decision.recorded", {"text": "keep 3 seeds", "why": "cheap"})
        res = archive.update(self.cfg, self.log, rebuild=True)
        root = Path(res["root"])
        memory = (root / "MEMORY.md").read_text()
        self.assertIn(fid, memory)
        self.assertIn("part-1 baselines", memory)
        self.assertIn("approval needed", memory)
        self.assertLessEqual(len(memory.splitlines()), archive.MEMORY_MAX_LINES)
        nb = (root / "notebooks" / "arch.md").read_text()
        self.assertIn("frozen", nb)
        self.assertIn("lr=0.1", nb)
        self.assertIn("remember to try lr=0.05", nb)
        day = sorted((root / "journal").glob("*.md"))[0]
        text = day.read_text()
        self.assertIn("results frozen for `smoke`", text)
        self.assertIn("keep 3 seeds", text)
        self.assertIn(fid, (root / "FINDINGS.md").read_text())
        weekly = list((root / "journal" / "weekly").glob("*.md"))
        self.assertEqual(len(weekly), 1)
        # deterministic: regenerating changes nothing but MEMORY.md (which carries a timestamp)
        again = archive.update(self.cfg, self.log, rebuild=True)
        self.assertEqual(again["written"], ["MEMORY.md"])

    def test_compacted_session_is_reground(self):
        archive.update(self.cfg, self.log)
        out = io.StringIO()
        payload = json.dumps({"session_id": "0123456789abcdef", "source": "compact"})
        with mock.patch.object(sys, "stdin", io.StringIO(payload)), mock.patch.object(sys, "stdout", out):
            run_hook(self.cfg, "session-start", "claude")
        ctx = json.loads(out.getvalue())["hookSpecificOutput"]["additionalContext"]
        self.assertIn("context was just compacted", ctx)
        self.assertIn("# MEMORY", ctx)
        self.assertIn("# HANDOFF", ctx)


    def test_state_git(self):
        import subprocess
        self.cfg.data["state_git"] = {"enabled": True, "every_minutes": 60, "push": False}
        archive.git_init(self.cfg)
        subprocess.run(["git", "-C", str(self.cfg.home), "config", "user.name", "t"], check=True)
        subprocess.run(["git", "-C", str(self.cfg.home), "config", "user.email", "t@example.org"], check=True)
        (self.cfg.home / "secrets").mkdir()
        (self.cfg.home / "secrets" / "token").write_text("x")
        self.assertTrue(archive.git_due(self.cfg))
        self.assertTrue(archive.git_commit(self.cfg)["committed"])
        files = subprocess.run(["git", "-C", str(self.cfg.home), "ls-files"], capture_output=True,
                               text=True).stdout.split()
        self.assertTrue(any(f.startswith("events/") for f in files))
        self.assertFalse(any(f.startswith("secrets/") or f.endswith(".lock") for f in files))
        self.assertFalse(archive.git_due(self.cfg))
        self.assertEqual(archive.git_commit(self.cfg)["reason"], "no changes")


if __name__ == "__main__":
    unittest.main()
