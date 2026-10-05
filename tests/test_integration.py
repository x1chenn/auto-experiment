"""End to end on the local backend: real runner, real canary, real contracts."""

import sys
import time
import unittest

from helpers import TempHome

from autoexp.engine import Engine
from autoexp.slurm import LocalBackend

SPEC = """\
name: local-faults
test: true
run_root: ./runs
command: >-
  PYTHONPATH={autoexp_src} {python} -m autoexp.canary --out {run_dir} --steps {steps}
  --lr 0.0003 --beta {beta} --seed {seed} --seconds-per-step 0.005 --log-every 5 --fault {fault}
params:
  fault: [none, crash, preflight, exit0, nan, ignore_beta]
fixed: {steps: 40, beta: 0.9}
seeds: [0]
resources: {cpus: 1, mem: 1G, time: "00:05:00"}
stages:
  - {name: faults, auto: true, max_failures: 4}
preflight:
  - name: fake-cuda-check
    cmd: "PYTHONPATH={autoexp_src} {python} -m autoexp.canary --preflight --fault {fault}"
contract:
  files: [final_eval.json]
  metrics_file: metrics.jsonl
  metric_keys: [eval/return]
  final_step_at_least: "{steps}"
  echo_file: config_echo.json
  echo_keys: [beta, steps, seed]
  finite: {file: final_eval.json, keys: [eval/return]}
analysis: {primary: eval/return, result_file: final_eval.json}
"""

EXPECTED = {
    "none": ("succeeded", "ok"),
    "crash": ("failed", "code"),
    "preflight": ("succeeded", "ok"),
    "exit0": ("failed", "contract"),
    "nan": ("failed", "diverged"),
    "ignore_beta": ("failed", "config_mismatch"),
}


class LocalEndToEnd(unittest.TestCase):
    def setUp(self):
        self.t = TempHome(config=f"python: {sys.executable}\nbackend: local\n")
        self.cfg = self.t.cfg()
        self.spec = self.t.write("faults.yaml", SPEC)

    def tearDown(self):
        self.t.close()

    def test_fault_matrix(self):
        backend = LocalBackend(self.cfg.home / "local_backend")
        eng = Engine(self.cfg, backend, echo=lambda m: None)
        eng.create_campaign(self.spec)
        deadline = time.time() + 120
        while time.time() < deadline:
            eng = Engine(self.cfg, backend, echo=lambda m: None)  # fresh replay each tick, like the brain
            eng.tick()
            if eng.state.campaigns["local-faults"]["status"] != "active":
                break
            time.sleep(0.5)
        state = eng.state
        got = {r["params"]["fault"]: (r["status"], r["final_class"]) for r in state.runs.values()}
        self.assertEqual(got, EXPECTED, msg=str(got))
        pre = next(r for r in state.runs.values() if r["params"]["fault"] == "preflight")
        classes = [state.attempts[a]["classification"] for a in pre["attempts"]]
        self.assertEqual(classes, ["infra/node", "ok"])
        self.assertEqual(state.stage("local-faults", "faults")["status"], "succeeded")
        self.assertEqual(state.campaigns["local-faults"]["status"], "done")


if __name__ == "__main__":
    unittest.main()
