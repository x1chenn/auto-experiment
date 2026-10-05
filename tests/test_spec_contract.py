import json
import math
import unittest
from pathlib import Path

from helpers import ROOT, SIMPLE_SPEC, TempHome

from autoexp import contract
from autoexp.spec import Spec, SpecError
from autoexp.util import format_time_s, parse_mem_mb, parse_time_s, subst


class SpecTests(unittest.TestCase):
    def setUp(self):
        self.t = TempHome()

    def tearDown(self):
        self.t.close()

    def test_plan_is_deterministic(self):
        p = self.t.write("s.yaml", SIMPLE_SPEC)
        a, b = Spec.load(p).plan("pilot"), Spec.load(p).plan("pilot")
        self.assertEqual([r["run_id"] for r in a], [r["run_id"] for r in b])
        self.assertEqual(len(a), 4)  # 2 lr x 1 beta x 2 seeds
        self.assertEqual(len(Spec.load(p).plan("smoke")), 2)
        self.assertEqual(a[0]["varying"], ["lr"])

    def test_unknown_key_is_an_error(self):
        p = self.t.write("s.yaml", SIMPLE_SPEC + "seed: [3]\n")
        with self.assertRaisesRegex(SpecError, "unknown key"):
            Spec.load(p)

    def test_unused_grid_param_is_an_error(self):
        p = self.t.write("s.yaml", SIMPLE_SPEC.replace("echo {lr} {beta}", "echo {lr}"))
        with self.assertRaisesRegex(SpecError, "never appear"):
            Spec.load(p)

    def test_unknown_placeholder_is_an_error(self):
        p = self.t.write("s.yaml", SIMPLE_SPEC.replace("{seed} >", "{sed} >"))
        with self.assertRaisesRegex(SpecError, "unknown placeholder"):
            Spec.load(p)

    def test_shell_variables_are_not_placeholders(self):
        p = self.t.write("s.yaml", SIMPLE_SPEC.replace("echo {lr}", "echo ${HOME} {lr}"))
        Spec.load(p)
        self.assertEqual(subst("a ${X} {y}", {"y": 1, "X": 2}), "a ${X} 1")

    def test_examples_validate(self):
        for name in ("campaign.yaml", "faults.yaml"):
            spec = Spec.load(ROOT / "examples" / "canary" / name)
            self.assertTrue(spec.plan(spec.stage_names[0]))


class UnitTests(unittest.TestCase):
    def test_units(self):
        self.assertEqual(parse_mem_mb("1G"), 1024)
        self.assertEqual(parse_mem_mb("1500M"), 1500)
        self.assertEqual(parse_time_s("1-00:00:00"), 86400)
        self.assertEqual(parse_time_s("00:15:00"), 900)
        self.assertEqual(parse_time_s("30"), 1800)
        self.assertEqual(format_time_s(5400), "01:30:00")


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.t = TempHome()
        self.d = self.t.path / "run"
        self.d.mkdir()
        self.c = {"files": ["final.json"], "metrics_file": "m.jsonl", "metric_keys": ["ret"],
                  "final_step_at_least": "{steps}", "echo_file": "echo.json", "echo_keys": ["lr"],
                  "finite": {"file": "final.json", "keys": ["ret"]}}
        self.m = {"steps": 10, "lr": 0.001}

    def tearDown(self):
        self.t.close()

    def _write(self, ret=1.0, last_step=10, echo_lr=0.001):
        (self.d / "m.jsonl").write_text("".join(json.dumps({"step": s, "ret": 1.0}) + "\n" for s in range(1, last_step + 1)))
        (self.d / "final.json").write_text(json.dumps({"ret": ret}))
        (self.d / "echo.json").write_text(json.dumps({"lr": echo_lr}))

    def test_ok(self):
        self._write()
        self.assertTrue(contract.check(self.d, self.c, self.m)["ok"])

    def test_short_run(self):
        self._write(last_step=7)
        r = contract.check(self.d, self.c, self.m)
        self.assertEqual(r["kind"], "contract")

    def test_echo_mismatch(self):
        self._write(echo_lr=0.0003)
        self.assertEqual(contract.check(self.d, self.c, self.m)["kind"], "config_mismatch")

    def test_nan(self):
        self._write(ret=math.nan)
        self.assertEqual(contract.check(self.d, self.c, self.m)["kind"], "diverged")

    def test_noop_detection(self):
        runs = [{"run_id": "a", "params": {"beta": 0.1}, "varying": ["beta"]},
                {"run_id": "b", "params": {"beta": 0.9}, "varying": ["beta"]}]
        same = {"a": {"beta": 0.5}, "b": {"beta": 0.5}}
        diff = {"a": {"beta": 0.1}, "b": {"beta": 0.9}}
        self.assertEqual(contract.detect_noops(runs, same)[0]["param"], "beta")
        self.assertEqual(contract.detect_noops(runs, diff), [])


if __name__ == "__main__":
    unittest.main()
