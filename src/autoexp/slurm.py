"""Scheduler backends and sbatch rendering.

``SlurmBackend`` talks to a real cluster. ``sacct`` is the source of truth for
terminal states because ``squeue`` forgets finished jobs after MinJobAge (often
30 s); ``squeue`` is only a fallback for jobs that accounting has not caught up
with yet. ``LocalBackend`` runs the same sbatch scripts as background processes,
which is enough for demos and integration tests on a laptop. ``FakeBackend`` is
an in-memory double for unit tests.
"""

from __future__ import annotations

import os
import re
import shlex
import signal
import socket
import subprocess
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from .util import parse_time_s

ACTIVE_STATES = {"PENDING", "RUNNING", "REQUEUED", "SUSPENDED", "CONFIGURING", "COMPLETING",
                 "RESIZING", "REQUEUE_HOLD", "REQUEUE_FED", "SIGNALING", "STAGE_OUT", "STOPPED",
                 "SUBMITTED"}
TERMINAL_STATES = {"COMPLETED", "FAILED", "TIMEOUT", "NODE_FAIL", "OUT_OF_MEMORY", "CANCELLED",
                   "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED", "SPECIAL_EXIT"}


@dataclass
class JobInfo:
    job_id: str
    state: str
    exit_code: str = ""
    node: str = ""
    elapsed_s: int = 0
    start: str = ""
    end: str = ""
    restarts: int = 0
    reason: str = ""
    partition: str = ""
    extra: str = ""

    @property
    def terminal(self) -> bool:
        return self.state in TERMINAL_STATES


def clean_env(drop_ld: bool = False) -> Dict[str, str]:
    """Environment for scheduler commands.

    SLURM_*/SBATCH_*/SRUN_* variables of the job we may be running inside would
    leak into child jobs; conda's LD_LIBRARY_PATH is known to break scontrol.
    """
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("SLURM_", "SBATCH_", "SRUN_", "SALLOC_"))}
    if drop_ld:
        env.pop("LD_LIBRARY_PATH", None)
    return env


def normalize_state(raw: str) -> str:
    # "CANCELLED by 12345" -> "CANCELLED"; "RUNNING+" -> "RUNNING"
    return (raw or "").split()[0].rstrip("+") if raw else ""


def render_sbatch(*, job_name: str, resources: Dict, output: str, tag: str, excludes: List[str],
                  defaults: Dict, command: str, signal_seconds: Optional[int] = None) -> str:
    """Render a batch script. ``command`` is exec'ed so it receives Slurm's signals."""
    part = resources.get("partition")
    if isinstance(part, (list, tuple)):
        part = ",".join(part)
    lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}"]
    if part:
        lines.append(f"#SBATCH --partition={part}")
    if defaults.get("account"):
        lines.append(f"#SBATCH --account={defaults['account']}")
    if defaults.get("comment"):
        lines.append(f"#SBATCH --comment={defaults['comment']}")
    lines += [
        f"#SBATCH --cpus-per-task={int(resources.get('cpus') or 1)}",
        f"#SBATCH --mem={resources.get('mem') or '4G'}",
        f"#SBATCH --time={resources.get('time') or '01:00:00'}",
        f"#SBATCH --output={output}",
        "#SBATCH --open-mode=append",
        "#SBATCH --requeue",
        f"#SBATCH --extra={tag}",
    ]
    gpus = int(resources.get("gpus") or 0)
    if gpus:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if signal_seconds:
        lines.append(f"#SBATCH --signal=B:USR1@{int(signal_seconds)}")
    if excludes:
        lines.append(f"#SBATCH --exclude={','.join(sorted(excludes))}")
    for raw in list(defaults.get("lines") or []) + list(resources.get("sbatch") or []):
        raw = str(raw).strip()
        lines.append(raw if raw.startswith("#SBATCH") else f"#SBATCH {raw}")
    lines += ["", "set -u", f"exec {command}", ""]
    return "\n".join(lines)


def output_path_of(script: Path) -> Optional[str]:
    for line in Path(script).read_text().splitlines():
        m = re.match(r"#SBATCH\s+--output=(\S+)", line)
        if m:
            return m.group(1)
    return None


class Backend:
    name = "base"

    def submit(self, script: Path) -> str:
        raise NotImplementedError

    def query(self, job_ids: Iterable[str]) -> Dict[str, JobInfo]:
        raise NotImplementedError

    def cancel(self, job_id: str) -> None:
        raise NotImplementedError

    def nodes(self) -> Optional[set]:
        """All node names known to the scheduler (None if unknown)."""
        return None

    def find_jobs(self, name: str) -> List[str]:
        """Active job ids with exactly this job name (for singleton jobs like the brain)."""
        return []


class SlurmBackend(Backend):
    name = "slurm"
    SACCT_FIELDS = "JobIDRaw,State,ExitCode,NodeList,ElapsedRaw,Start,End,Restarts,Reason,Partition,Extra"

    def _run(self, args: List[str], timeout: int = 60) -> subprocess.CompletedProcess:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout,
                              env=clean_env(drop_ld=True))

    def submit(self, script: Path) -> str:
        proc = self._run(["sbatch", "--parsable", str(script)], timeout=120)
        ids = [ln.strip() for ln in proc.stdout.splitlines() if re.match(r"^\d+(;\S+)?$", ln.strip())]
        if proc.returncode != 0 or not ids:
            msg = (proc.stderr.strip() or proc.stdout.strip()).splitlines()
            msg = [m for m in msg if "[BILLING]" not in m]
            raise RuntimeError("sbatch failed: " + (" | ".join(msg[-3:]) or f"rc={proc.returncode}"))
        return ids[-1].split(";")[0]

    def query(self, job_ids: Iterable[str]) -> Dict[str, JobInfo]:
        ids = sorted({str(j) for j in job_ids if j})
        out: Dict[str, JobInfo] = {}
        for i in range(0, len(ids), 200):
            chunk = ids[i:i + 200]
            proc = self._run(["sacct", "-X", "-P", "-n", "-j", ",".join(chunk),
                              f"--format={self.SACCT_FIELDS}"])
            for line in proc.stdout.splitlines():
                parts = line.split("|")
                if len(parts) < 11:
                    continue
                jid = parts[0]
                out[jid] = JobInfo(
                    job_id=jid, state=normalize_state(parts[1]), exit_code=parts[2],
                    node="" if parts[3] in ("None assigned", "(null)") else parts[3],
                    elapsed_s=int(parts[4] or 0), start=parts[5], end=parts[6],
                    restarts=int(parts[7] or 0), reason=parts[8], partition=parts[9], extra=parts[10])
        missing = [j for j in ids if j not in out]
        if missing:
            proc = self._run(["squeue", "-h", "-j", ",".join(missing), "-o", "%i|%T|%N|%r|%P"])
            for line in proc.stdout.splitlines():
                parts = line.split("|")
                if len(parts) >= 5:
                    out[parts[0]] = JobInfo(job_id=parts[0], state=normalize_state(parts[1]),
                                            node=parts[2], reason=parts[3], partition=parts[4])
        return out

    def cancel(self, job_id: str) -> None:
        self._run(["scancel", str(job_id)])

    def nodes(self) -> Optional[set]:
        proc = self._run(["sinfo", "-h", "-N", "-o", "%N"])
        if proc.returncode != 0:
            return None
        return {ln.strip() for ln in proc.stdout.splitlines() if ln.strip()}

    def find_jobs(self, name: str) -> List[str]:
        user = os.environ.get("USER") or ""
        proc = self._run(["squeue", "-h", "-u", user, "-n", name, "-o", "%i|%T"])
        return [ln.split("|")[0] for ln in proc.stdout.splitlines()
                if ln.strip() and normalize_state(ln.split("|")[1]) in ACTIVE_STATES]


class LocalBackend(Backend):
    """Runs batch scripts as detached local processes. State lives in a directory
    so that separate CLI invocations agree on what is running."""

    name = "local"

    def __init__(self, state_dir: Path):
        self.dir = Path(state_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _next_id(self) -> str:
        counter = self.dir / "counter"
        n = int(counter.read_text()) + 1 if counter.exists() else 1
        counter.write_text(str(n))
        return str(900000 + n)

    def submit(self, script: Path) -> str:
        jid = self._next_id()
        out = output_path_of(script) or str(self.dir / f"{jid}.out")
        out = out.replace("%j", jid)
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        env = clean_env()
        env.update(SLURM_JOB_ID=jid, SLURM_RESTART_COUNT="0", SLURMD_NODENAME=socket.gethostname())
        status = self.dir / f"{jid}.rc"
        wrapper = f"bash {shlex.quote(str(script))} >> {shlex.quote(out)} 2>&1; echo $? > {shlex.quote(str(status))}"
        proc = subprocess.Popen(["bash", "-c", wrapper], env=env, start_new_session=True,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        (self.dir / f"{jid}.pid").write_text(str(proc.pid))
        # Reap the child in the background so a long-lived brain never accumulates zombies.
        threading.Thread(target=proc.wait, daemon=True).start()
        return jid

    def query(self, job_ids: Iterable[str]) -> Dict[str, JobInfo]:
        out = {}
        host = socket.gethostname().split(".")[0]
        for jid in job_ids:
            pid_file, rc_file = self.dir / f"{jid}.pid", self.dir / f"{jid}.rc"
            if rc_file.exists():
                rc = int(rc_file.read_text().strip() or 1)
                state = "COMPLETED" if rc == 0 else ("CANCELLED" if (self.dir / f"{jid}.cancelled").exists() else "FAILED")
                out[jid] = JobInfo(job_id=jid, state=state, exit_code=f"{rc}:0", node=host)
            elif pid_file.exists():
                out[jid] = JobInfo(job_id=jid, state="RUNNING", node=host)
        return out

    def cancel(self, job_id: str) -> None:
        pid_file = self.dir / f"{job_id}.pid"
        if pid_file.exists():
            (self.dir / f"{job_id}.cancelled").write_text("1")
            try:
                os.killpg(int(pid_file.read_text()), signal.SIGTERM)
            except (ProcessLookupError, PermissionError, ValueError):
                pass


@dataclass
class FakeBackend(Backend):
    """In-memory scheduler double for unit tests."""

    name: str = "fake"
    jobs: Dict[str, JobInfo] = field(default_factory=dict)
    scripts: Dict[str, Path] = field(default_factory=dict)
    known_nodes: Optional[set] = None
    fail_submit: bool = False
    cancelled: List[str] = field(default_factory=list)
    _n: int = 1000

    def submit(self, script: Path) -> str:
        if self.fail_submit:
            raise RuntimeError("sbatch failed: fake")
        self._n += 1
        jid = str(self._n)
        self.jobs[jid] = JobInfo(job_id=jid, state="PENDING")
        self.scripts[jid] = Path(script)
        return jid

    def query(self, job_ids: Iterable[str]) -> Dict[str, JobInfo]:
        return {j: self.jobs[j] for j in job_ids if j in self.jobs}

    def cancel(self, job_id: str) -> None:
        self.cancelled.append(job_id)
        if job_id in self.jobs:
            self.jobs[job_id].state = "CANCELLED"

    def nodes(self) -> Optional[set]:
        return self.known_nodes

    def set(self, job_id: str, state: str, node: str = "n001", elapsed: str = "00:05:00", **kw) -> None:
        info = self.jobs[job_id]
        info.state, info.node, info.elapsed_s = state, node, parse_time_s(elapsed)
        for k, v in kw.items():
            setattr(info, k, v)


def make_backend(cfg) -> Backend:
    name = cfg.backend_name
    if name == "slurm":
        return SlurmBackend()
    if name == "local":
        return LocalBackend(cfg.home / "local_backend")
    raise ValueError(f"unknown backend {name!r}")
