"""The brain: one long-lived job per user that keeps ticking.

Many clusters forbid cron and scrontab and frown on daemons on login nodes, so
the brain is itself a batch job on a CPU partition. Shortly before its time
limit it receives SIGUSR1 and requeues itself (same job id), which works even
under a QOS that allows a single submitted job per user. A heartbeat lease
prevents two brains from running at once without ever holding a lock open,
and ``autoexp brain ensure`` (run from any CLI call or agent hook) resubmits
the brain if it died.
"""

from __future__ import annotations

import datetime as _dt
import os
import shlex
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

from .archive import git_commit, git_due
from .archive import update as archive_update
from .brief import render_brief, write_brief
from .config import Config
from .engine import AUTOEXP_SRC, Engine
from .events import EventLog
from .handoff import brain_status, close_unclean_sessions, write_handoff
from .slurm import Backend, clean_env, make_backend, render_sbatch
from .util import age_seconds, atomic_write, now, read_json, short_lock, write_json

BRAIN_JOB_NAME = "ae:brain"
LEASE_STALE_S = 300


def _me() -> str:
    return f"{socket.gethostname().split('.')[0]}:{os.getpid()}"


def _holder_job_dead(lease: Dict[str, Any], backend: Optional[Backend]) -> bool:
    """True if the lease holder's batch job is already over according to the scheduler."""
    job = lease.get("job")
    if not job or backend is None or job == os.environ.get("SLURM_JOB_ID"):
        return False
    try:
        info = backend.query([str(job)]).get(str(job))
    except Exception:
        return False
    return info is not None and (info.terminal or info.state in ("COMPLETING", "STOPPED"))


def acquire_lease(cfg: Config, backend: Optional[Backend] = None) -> bool:
    """Take the lease if it is free, stale, or held by a job the scheduler has ended.
    A holder that is still alive notices the takeover at its next renewal and stops."""
    with short_lock(cfg.home / ".lease.lock"):
        lease = read_json(cfg.lease_path) or {}
        age = age_seconds(lease.get("heartbeat"))
        if (lease and lease.get("holder") != _me() and age is not None and age < LEASE_STALE_S
                and not _holder_job_dead(lease, backend)):
            return False
        write_json(cfg.lease_path, {"holder": _me(), "host": socket.gethostname().split(".")[0],
                                    "pid": os.getpid(), "job": os.environ.get("SLURM_JOB_ID"),
                                    "started": now(), "heartbeat": now()})
        return True


def renew_lease(cfg: Config) -> bool:
    with short_lock(cfg.home / ".lease.lock"):
        lease = read_json(cfg.lease_path) or {}
        if lease.get("holder") != _me():
            return False
        lease["heartbeat"] = now()
        write_json(cfg.lease_path, lease)
        return True


def release_lease(cfg: Config) -> None:
    with short_lock(cfg.home / ".lease.lock"):
        lease = read_json(cfg.lease_path) or {}
        if lease.get("holder") == _me():
            lease["heartbeat"] = "1970-01-01T00:00:00+00:00"
            lease["released"] = now()
            write_json(cfg.lease_path, lease)


class Brain:
    def __init__(self, cfg: Config, backend: Optional[Backend] = None):
        self.cfg = cfg
        self.backend = backend or make_backend(cfg)
        self.log = EventLog(cfg)
        self.stop = False
        self.requeue = False

    def _on_usr1(self, signum, frame) -> None:
        self.requeue = True
        self.stop = True

    def _on_term(self, signum, frame) -> None:
        self.stop = True

    def due_for_brief(self) -> bool:
        at = str(self.cfg["brain"].get("brief_at") or "")
        if not at:
            return False
        hh, mm = (int(x) for x in at.split(":"))
        t = _dt.datetime.now()
        return (t.hour, t.minute) >= (hh, mm) and not (self.cfg.briefs_dir / f"{t.date().isoformat()}.md").exists()

    def one_tick(self) -> Dict[str, Any]:
        eng = Engine(self.cfg, self.backend, self.log, echo=lambda m: print(f"[brain {now()}] {m}", flush=True))
        summary = eng.tick(wait=60)
        if summary.get("skipped"):
            return summary
        if summary["finished"] or summary["submitted"] or summary["stalled"]:
            self.log.append("brain.tick", summary)
        closed = close_unclean_sessions(self.cfg, self.log, eng.state)
        if closed:
            eng.state = type(eng.state).load(self.log)
        write_handoff(self.cfg, eng.state)
        specs = {n: eng.spec(n).data for n in eng.state.campaigns}
        archive_update(self.cfg, self.log, specs=specs)
        if git_due(self.cfg):
            res = git_commit(self.cfg)
            if res.get("push_error"):
                print(f"[brain {now()}] state push failed: {res['push_error']}", flush=True)
        if self.due_for_brief():
            path = write_brief(self.cfg, render_brief(self.cfg, eng.state, specs))
            self.log.append("brief.written", {"path": str(path)})
        return summary

    def run(self, once: bool = False) -> int:
        self.cfg.ensure_dirs()
        deadline = time.time() + LEASE_STALE_S + 60
        while not acquire_lease(self.cfg, self.backend):
            lease = read_json(self.cfg.lease_path) or {}
            if time.time() > deadline:
                print(f"another brain keeps the lease ({lease.get('holder')}, job {lease.get('job')}); exiting",
                      flush=True)
                return 1
            print(f"[brain {now()}] waiting for the lease held by {lease.get('holder')} (job {lease.get('job')})",
                  flush=True)
            time.sleep(15)
        signal.signal(signal.SIGUSR1, self._on_usr1)
        signal.signal(signal.SIGTERM, self._on_term)
        self.log.append("brain.started", {"host": socket.gethostname().split(".")[0], "pid": os.getpid(),
                                          "job": os.environ.get("SLURM_JOB_ID"),
                                          "restart": os.environ.get("SLURM_RESTART_COUNT", "0")})
        tick_s = int(self.cfg["brain"]["tick_seconds"])
        try:
            while not self.stop:
                t0 = time.time()
                try:
                    self.one_tick()
                except Exception as exc:
                    traceback.print_exc()
                    self.log.append("brain.error", {"error": f"{type(exc).__name__}: {exc}"[:800]})
                if not renew_lease(self.cfg):
                    print("lease taken over by another brain; stopping", flush=True)
                    return 1
                if once:
                    break
                while not self.stop and time.time() - t0 < tick_s:
                    time.sleep(1)
        finally:
            release_lease(self.cfg)
        if self.requeue and os.environ.get("SLURM_JOB_ID") and self.backend.name == "slurm":
            self.log.append("brain.requeue", {"job": os.environ["SLURM_JOB_ID"]})
            subprocess.run(["scontrol", "requeue", os.environ["SLURM_JOB_ID"]], env=clean_env(drop_ld=True),
                           capture_output=True, text=True, timeout=60)
            time.sleep(120)
        return 0


def brain_script(cfg: Config) -> str:
    b = cfg["brain"]
    res = {"partition": b.get("partition"), "cpus": b.get("cpus", 1), "mem": b.get("mem", "2G"),
           "time": b.get("time", "1-00:00:00"), "gpus": 0, "sbatch": b.get("lines") or []}
    cmd = (f"env AUTOEXP_HOME={shlex.quote(str(cfg.home))} PYTHONPATH={shlex.quote(AUTOEXP_SRC)} "
           f"{shlex.quote(cfg.python)} -m autoexp brain run")
    return render_sbatch(job_name=BRAIN_JOB_NAME, resources=res, output=str(cfg.logs_dir / "brain-%j.out"),
                         tag="ae:brain", excludes=[], defaults=cfg["sbatch"], command=cmd,
                         signal_seconds=int(b.get("signal_seconds", 600)))


def submit_brain(cfg: Config, backend: Optional[Backend] = None) -> str:
    backend = backend or make_backend(cfg)
    existing = backend.find_jobs(BRAIN_JOB_NAME)
    if existing:
        return existing[0]
    cfg.ensure_dirs()
    path = cfg.home / "brain.sbatch"
    atomic_write(path, brain_script(cfg))
    jid = backend.submit(path)
    EventLog(cfg).append("brain.submitted", {"job": jid})
    return jid


def ensure_brain(cfg: Config, backend: Optional[Backend] = None) -> Dict[str, Any]:
    status = brain_status(cfg)
    if status.get("alive"):
        return {"action": "none", "reason": "brain alive", "job": status.get("job")}
    backend = backend or make_backend(cfg)
    existing = backend.find_jobs(BRAIN_JOB_NAME)
    if existing:
        return {"action": "none", "reason": "brain job queued", "job": existing[0]}
    jid = submit_brain(cfg, backend)
    return {"action": "submitted", "job": jid}
