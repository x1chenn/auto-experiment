import json
import unittest
from pathlib import Path

from helpers import SIMPLE_SPEC, TempHome

from autoexp import nodes
from autoexp.engine import Engine, job_dir
from autoexp.slurm import FakeBackend


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.t = TempHome()
        self.cfg = self.t.cfg()
        self.fake = FakeBackend(known_nodes={"n001", "n002", "n003"})
        self.spec = self.t.write("demo.yaml", SIMPLE_SPEC)
        self.msgs = []

    def tearDown(self):
        self.t.close()

    def eng(self):
        return Engine(self.cfg, self.fake, echo=self.msgs.append)

    def active(self, eng, stage=None):
        out = []
        for a in eng.state.active_attempts():
            run = eng.state.runs[a["run_id"]]
            if stage is None or run["stage"] == stage:
                out.append((run, a))
        return out

    def finish(self, eng, run, a, status="ok", state="COMPLETED", node="n001", echo=None, elapsed="00:05:00"):
        jd = job_dir(run, a["job_id"])
        jd.mkdir(parents=True, exist_ok=True)
        if status is not None:
            (jd / "result.json").write_text(json.dumps({"status": status, "node": node}))
        if status == "preflight_failed":
            (jd / "preflight.json").write_text(json.dumps({"ok": False}))
        echo = echo if echo is not None else {k: run["params"][k] for k in ("lr", "beta")}
        Path(run["run_dir"], "echo.json").write_text(json.dumps(echo))
        self.fake.set(a["job_id"], state, node=node, elapsed=elapsed)

    def test_happy_path_and_approval_gate(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        s = eng.tick()
        self.assertEqual(s["submitted"], 2)  # smoke: 2 lr x seed 0
        for run, a in self.active(eng):
            self.finish(eng, run, a)
        eng.tick()
        self.assertEqual(eng.state.stage("demo", "smoke")["status"], "succeeded")
        self.assertEqual(eng.state.stage("demo", "pilot")["status"], "running")
        self.assertEqual(len(self.active(eng, "pilot")), 4)
        for run, a in self.active(eng):
            self.finish(eng, run, a)
        eng.tick()
        self.assertEqual(eng.state.stage("demo", "full")["status"], "awaiting_approval")
        self.assertEqual(self.active(eng), [])
        eng.approve("demo", "full")
        eng.tick()
        self.assertEqual(eng.state.stage("demo", "full")["status"], "running")
        # sbatch carries the tag in --extra and never touches --comment
        script = self.fake.scripts[self.active(eng)[0][1]["job_id"]].read_text()
        self.assertIn("#SBATCH --extra=ae:demo/full/", script)
        self.assertNotIn("--comment", script)

    def test_autonomy_policy_overrides_spec(self):
        self.cfg.data["auto_stages"] = ["smoke"]
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        for run, a in self.active(eng):
            self.finish(eng, run, a)
        eng.tick()
        st = eng.state.stage("demo", "pilot")
        self.assertEqual(st["status"], "awaiting_approval")
        self.assertIn("policy", st["reason"])

    def test_preflight_failure_flags_node_and_retries_elsewhere(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        run, a = self.active(eng)[0]
        self.finish(eng, run, a, status="preflight_failed", state="FAILED", node="n002")
        other_run, other_a = self.active(eng)[1]
        eng.tick()
        self.assertEqual(eng.state.attempts[a["attempt_id"]]["classification"], "infra/node")
        new = eng.state.active_attempt_of(eng.state.runs[run["run_id"]])
        self.assertIsNotNone(new)
        self.assertEqual(new["excludes"], ["n002"])
        self.assertIn("--exclude=n002", self.fake.scripts[new["job_id"]].read_text())
        entry = nodes.entries(self.cfg.nodes_path)["n002@demo"]
        self.assertEqual(entry["scope"], "demo")  # test campaign: scoped flag

    def test_oom_raises_memory_once(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        run, a = self.active(eng)[0]
        self.finish(eng, run, a, status="code_failed", state="OUT_OF_MEMORY")
        eng.tick()
        new = eng.state.active_attempt_of(eng.state.runs[run["run_id"]])
        self.assertEqual(new["resources"]["mem"], "1536M")
        self.finish(eng, eng.state.runs[run["run_id"]], new, status="code_failed", state="OUT_OF_MEMORY")
        eng.tick()
        self.assertEqual(eng.state.runs[run["run_id"]]["status"], "failed")

    def test_code_failure_is_not_retried_and_blocks(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        runs = self.active(eng)
        self.finish(eng, *runs[0], status="code_failed", state="FAILED")
        self.finish(eng, *runs[1])
        eng.tick()
        self.assertEqual(eng.state.runs[runs[0][0]["run_id"]]["status"], "failed")
        self.assertEqual(eng.state.campaigns["demo"]["status"], "blocked")
        self.assertEqual(eng.state.stage("demo", "pilot")["status"], "pending")

    def test_exit_zero_without_contract_is_failure(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        run, a = self.active(eng)[0]
        self.finish(eng, run, a, status="contract", state="COMPLETED")
        eng.tick()
        self.assertEqual(eng.state.runs[run["run_id"]]["final_class"], "contract")

    def test_noop_parameter_blocks_stage(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        for run, a in self.active(eng):
            self.finish(eng, run, a, echo={"lr": 0.1, "beta": 1})  # both arms echo the same lr
        eng.tick()
        st = eng.state.stage("demo", "smoke")
        self.assertEqual(st["status"], "failed")
        self.assertIn("no-op", st["reason"])
        self.assertEqual(eng.state.noops[0]["param"], "lr")

    def test_instant_death_without_runner_is_node_problem(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        run, a = self.active(eng)[0]
        self.finish(eng, run, a, status=None, state="FAILED", node="n003", elapsed="00:00:01")
        eng.tick()
        self.assertEqual(eng.state.attempts[a["attempt_id"]]["classification"], "infra/node")

    def test_stalled_heartbeat_is_cancelled_and_retried(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        run, a = self.active(eng)[0]
        self.fake.set(a["job_id"], "RUNNING", start="2000-01-01T00:00:00")
        eng.tick()  # observe RUNNING with a start long ago and no heartbeat
        self.assertIn(a["job_id"], self.fake.cancelled)
        self.finish(eng, run, a, status="terminated", state="CANCELLED")
        eng.tick()
        self.assertEqual(eng.state.attempts[a["attempt_id"]]["classification"], "infra/hung")
        self.assertIsNotNone(eng.state.active_attempt_of(eng.state.runs[run["run_id"]]))

    def test_state_survives_reload(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        again = self.eng()
        self.assertEqual(len(again.state.active_attempts()), 2)
        self.assertEqual(again.tick()["submitted"], 0)  # nothing is submitted twice

    def test_cancel_campaign(self):
        eng = self.eng()
        eng.create_campaign(self.spec)
        eng.tick()
        n = eng.cancel_campaign("demo")
        self.assertEqual(n, 2)
        eng.tick()
        self.assertEqual(eng.state.campaigns["demo"]["status"], "cancelled")
        self.assertTrue(all(r["status"] == "cancelled" for r in eng.state.runs.values()))


if __name__ == "__main__":
    unittest.main()
