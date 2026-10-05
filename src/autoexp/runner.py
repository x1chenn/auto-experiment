"""In-job wrapper: ``python -m autoexp.runner <bundle.json>``.

Every experiment job runs through this wrapper, which

1. notices silent requeues (``SLURM_RESTART_COUNT``) and records each start;
2. runs preflight checks (GPU visible, user checks) and exits 75 if the node is
   unusable, instead of letting the program fall back to CPU or crash later;
3. launches the user command in its own process group and writes a heartbeat
   with the latest training step every 30 s;
4. on SIGUSR1 (time limit approaching) forwards the signal so the program can
   checkpoint, waits for the whole process group to exit, then requeues the job;
5. checks the artifact contract and writes ``result.json``. Exit code 0 with a
   broken contract is reported as a failure.

The brain never trusts this process's exit code alone: it reads result.json.
"""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import contract as contract_mod
from .slurm import clean_env
from .util import last_json_line, now, read_json, tail_lines, write_json

EXIT_OK, EXIT_CONTRACT, EXIT_CODE, EXIT_RUNNER, EXIT_PREFLIGHT, EXIT_TERM = 0, 3, 4, 70, 75, 143
HEARTBEAT_SECONDS = 30


def log(msg: str) -> None:
    print(f"[autoexp {now()}] {msg}", flush=True)


class Runner:
    def __init__(self, bundle: Dict[str, Any], bundle_dir: Path):
        self.b = bundle
        self.bundle_dir = bundle_dir
        self.run_dir = Path(bundle["run_dir"])
        self.job = os.environ.get("SLURM_JOB_ID", "local")
        self.restart = int(os.environ.get("SLURM_RESTART_COUNT") or 0)
        self.host = socket.gethostname().split(".")[0]
        self.jdir = self.run_dir / ".autoexp" / f"job-{self.job}"
        self.jdir.mkdir(parents=True, exist_ok=True)
        self.proc: Optional[subprocess.Popen] = None
        self.usr1 = False
        self.term = False
        self.stop_heartbeat = threading.Event()
        self.started = now()
        self.t0 = time.time()
        self.last_step: Any = None
        self.last_progress = now()

    # ------------------------------------------------------------ environment
    def env(self) -> Dict[str, str]:
        env = dict(os.environ)
        env.update({
            "AUTOEXP_RUN_DIR": str(self.run_dir), "AUTOEXP_RUN_ID": self.b["run_id"],
            "AUTOEXP_CAMPAIGN": self.b["campaign"], "AUTOEXP_STAGE": self.b["stage"],
            "AUTOEXP_ATTEMPT": str(self.b["attempt"]), "AUTOEXP_RESTART_COUNT": str(self.restart),
            "AUTOEXP_JOB_ID": self.job, "PYTHONUNBUFFERED": "1",
        })
        return env

    def shell(self, cmd: str) -> List[str]:
        recipe = self.bundle_dir / "recipe.sh"
        return ["bash", "-c", f"source {recipe} && {cmd}"]

    # ------------------------------------------------------------ preflight
    def preflight(self) -> Dict[str, Any]:
        checks = []
        gpus = int(self.b.get("gpus") or 0)
        if gpus:
            checks.append(self._check("gpu-visible", ["nvidia-smi", "-L"], 30,
                                      lambda out: sum(1 for ln in out.splitlines() if ln.startswith("GPU ")) >= gpus))
        for item in self.b.get("preflight") or []:
            checks.append(self._check(item.get("name", "check"), self.shell(item["cmd"]),
                                      int(item.get("timeout", 120))))
        return {"ok": all(c["ok"] for c in checks), "checks": checks, "host": self.host,
                "job": self.job, "restart": self.restart, "ts": now()}

    def _check(self, name: str, argv: List[str], timeout: int, accept=None) -> Dict[str, Any]:
        t = time.time()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, timeout=timeout,
                                  env=self.env(), cwd=self.b["workdir"])
            out = (proc.stdout or "") + (proc.stderr or "")
            ok = proc.returncode == 0 and (accept is None or accept(proc.stdout or ""))
            rc = proc.returncode
        except subprocess.TimeoutExpired:
            out, ok, rc = f"timed out after {timeout}s", False, None
        except OSError as exc:
            out, ok, rc = str(exc), False, None
        return {"name": name, "ok": ok, "rc": rc, "seconds": round(time.time() - t, 1),
                "output": out.strip().splitlines()[-5:]}

    # ------------------------------------------------------------ heartbeat
    def _progress(self) -> None:
        pfile = (self.b.get("progress") or {}).get("file")
        if not pfile:
            return
        rec = last_json_line(self.run_dir / pfile)
        key = (self.b.get("progress") or {}).get("step_key", "step")
        if rec and key in rec and rec[key] != self.last_step:
            self.last_step = rec[key]
            self.last_progress = now()

    def _heartbeat_loop(self) -> None:
        while True:
            self._progress()
            alive = self.proc is not None and self.proc.poll() is None
            write_json(self.jdir / "heartbeat.json", {
                "ts": now(), "host": self.host, "job": self.job, "restart": self.restart,
                "pid": self.proc.pid if self.proc else None, "child_alive": alive,
                "step": self.last_step, "last_progress": self.last_progress,
                "elapsed_s": int(time.time() - self.t0)})
            if self.stop_heartbeat.wait(HEARTBEAT_SECONDS):
                return

    # ------------------------------------------------------------ signals
    def _on_usr1(self, signum, frame) -> None:
        self.usr1 = True
        log("SIGUSR1: time limit approaching, asking the program to checkpoint")
        self._signal_group(signal.SIGUSR1)

    def _on_term(self, signum, frame) -> None:
        self.term = True
        log("SIGTERM received, stopping the program")
        self._signal_group(signal.SIGTERM)

    def _signal_group(self, sig: int) -> None:
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            pass

    def _group_alive(self) -> bool:
        if self.proc is None:
            return False
        try:
            os.killpg(self.proc.pid, 0)
            return True
        except (ProcessLookupError, PermissionError):
            return False

    def _wait_group(self, seconds: float) -> bool:
        deadline = time.time() + seconds
        while time.time() < deadline:
            if not self._group_alive():
                return True
            time.sleep(0.5)
        return not self._group_alive()

    # ------------------------------------------------------------ main flow
    def result(self, status: str, **extra: Any) -> Dict[str, Any]:
        res = {"status": status, "job": self.job, "restart": self.restart, "node": self.host,
               "attempt": self.b["attempt"], "started": self.started, "finished": now(),
               "duration_s": int(time.time() - self.t0)}
        res.update(extra)
        write_json(self.jdir / "result.json", res)
        write_json(self.run_dir / ".autoexp" / "result.json", res)
        return res

    def run(self) -> int:
        write_json(self.jdir / f"start-{self.restart}.json",
                   {"ts": now(), "host": self.host, "job": self.job, "restart": self.restart})
        log(f"run {self.b['run_id']} attempt {self.b['attempt']} job {self.job} "
            f"restart {self.restart} on {self.host}")
        if self.restart:
            log(f"this is restart #{self.restart} of the job (requeued); the program should resume")

        pf = self.preflight()
        write_json(self.jdir / "preflight.json", pf)
        if not pf["ok"]:
            bad = [c["name"] for c in pf["checks"] if not c["ok"]]
            log(f"preflight failed on {self.host}: {bad}; refusing to run here")
            self.result("preflight_failed", failed_checks=bad)
            return EXIT_PREFLIGHT

        signal.signal(signal.SIGUSR1, self._on_usr1)
        signal.signal(signal.SIGTERM, self._on_term)
        self.proc = subprocess.Popen(["bash", str(self.bundle_dir / "command.sh")], cwd=self.b["workdir"],
                                     env=self.env(), start_new_session=True)
        hb = threading.Thread(target=self._heartbeat_loop, daemon=True)
        hb.start()
        rc = self.proc.wait()
        if self.usr1 or self.term:
            grace = 240 if self.usr1 else 25
            if not self._wait_group(grace):
                log("program did not exit in time; terminating its process group")
                self._signal_group(signal.SIGTERM)
                if not self._wait_group(20):
                    self._signal_group(signal.SIGKILL)
        self.stop_heartbeat.set()
        hb.join(timeout=5)
        self._progress()

        mapping = self.b["mapping"]
        report = contract_mod.check(self.run_dir, self.b.get("contract") or {}, mapping)
        tail = tail_lines(Path(self.b["output"].replace("%j", self.job)), n=40) if self.b.get("output") else []

        if self.term:
            self.result("terminated", exit_code=rc, contract=report)
            return EXIT_TERM
        if self.usr1 and not report["ok"]:
            return self._requeue(rc, report)
        if rc != 0:
            self.result("code_failed", exit_code=rc, contract=report, log_tail=tail[-20:])
            return EXIT_CODE
        if not report["ok"]:
            log(f"exit code 0 but the contract failed: {report['failures']}")
            self.result(report["kind"], exit_code=rc, contract=report)
            return EXIT_CONTRACT
        log("contract satisfied")
        self.result("ok", exit_code=rc, contract=report)
        return EXIT_OK

    def _requeue(self, rc: int, report: Dict[str, Any]) -> int:
        resume = self.b.get("resume") or {}
        if not resume.get("enabled"):
            self.result("timeout", exit_code=rc, contract=report)
            return EXIT_TERM
        if self.restart >= int(resume.get("max_restarts", 10)):
            log(f"restart limit {resume.get('max_restarts')} reached; not requeueing")
            self.result("timeout_exhausted", exit_code=rc, contract=report)
            return EXIT_TERM
        if self.b.get("backend") != "slurm":
            self.result("timeout", exit_code=rc, contract=report)
            return EXIT_TERM
        self.result("requeued", exit_code=rc, contract=report)
        log(f"requeueing job {self.job} (restart {self.restart + 1} will resume from checkpoint)")
        proc = subprocess.run(["scontrol", "requeue", self.job], capture_output=True, text=True,
                              env=clean_env(drop_ld=True), timeout=60)
        if proc.returncode != 0:
            log(f"scontrol requeue failed: {proc.stderr.strip()}")
            self.result("requeue_failed", exit_code=rc, contract=report, error=proc.stderr.strip())
            return EXIT_TERM
        time.sleep(120)  # Slurm stops this incarnation; never reached normally
        return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1:
        print("usage: python -m autoexp.runner <bundle.json>", file=sys.stderr)
        return 2
    bundle_path = Path(argv[0])
    bundle = read_json(bundle_path)
    if not isinstance(bundle, dict):
        print(f"cannot read bundle {bundle_path}", file=sys.stderr)
        return EXIT_RUNNER
    runner = Runner(bundle, bundle_path.parent)
    try:
        return runner.run()
    except Exception as exc:  # the wrapper must always leave a verdict behind
        import traceback
        traceback.print_exc()
        runner.result("runner_error", error=f"{type(exc).__name__}: {exc}")
        return EXIT_RUNNER


if __name__ == "__main__":
    sys.exit(main())
