"""The deterministic core: submit, reconcile, classify, retry, advance stages.

One call of ``Engine.tick()`` brings the world in line with the event log:

1. poll the scheduler for every unfinished attempt (sacct is the truth);
2. cancel attempts whose heartbeat or training progress stalled;
3. classify each finished attempt (ok / infra/* / code / contract /
   config_mismatch / diverged / cancelled) from the scheduler state, the
   runner's result.json, its preflight record and known log signatures;
4. flag bad nodes, schedule retries for infrastructure failures with adjusted
   resources, and finish runs that cannot or should not be retried;
5. advance each campaign through its stages, honoring the autonomy policy;
6. submit planned runs within the concurrency limits.

No language model is involved anywhere in this module.
"""

from __future__ import annotations

import math
import os
import re
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

from . import contract as contract_mod
from . import nodes as nodes_mod
from .config import Config
from .events import EventLog, State, class_family
from .slurm import Backend, JobInfo, render_sbatch
from .spec import Spec
from .util import (age_seconds, atomic_write, file_hash, format_time_s, now, parse_mem_mb,
                   parse_time_s, read_json, release_lock, stable_hash, subst, tail_lines, try_lock,
                   write_json)

AUTOEXP_SRC = str(Path(__file__).resolve().parents[1])

# Built-in log signatures; groups can extend them in <shared>/signatures.yaml.
DEFAULT_SIGNATURES = [
    {"pattern": r"uncorrectable ECC|ECC error", "class": "infra/node"},
    {"pattern": r"CUDA driver (initialization failed|version is insufficient)", "class": "infra/node"},
    {"pattern": r"no CUDA-capable device|cudaErrorNoDevice|CUDA unavailable", "class": "infra/node"},
    {"pattern": r"NVIDIA-SMI has failed|Unable to determine the device handle", "class": "infra/node"},
    {"pattern": r"oom[-_ ]kill|Out Of Memory|out-of-memory handler|MemoryError", "class": "infra/oom"},
    {"pattern": r"CUDA out of memory", "class": "code/gpu_oom"},
]

TICK_LOCK_STALE_S = 300  # refreshed while held; only a dead holder's lock gets this old

RETRYABLE = {"infra/node", "infra/preempt", "infra/unknown", "infra/runner", "infra/lost",
             "infra/oom", "infra/timeout", "infra/hung", "infra/requeue_failed"}


class Engine:
    def __init__(self, cfg: Config, backend: Backend, log: Optional[EventLog] = None, echo=print):
        self.cfg = cfg
        self.backend = backend
        self.log = log or EventLog(cfg)
        self.state = State.load(self.log)
        self.echo = echo
        self._specs: Dict[str, Spec] = {}
        self._known_nodes: Optional[set] = None
        self._signatures: Optional[List[Dict[str, Any]]] = None

    # ------------------------------------------------------------ plumbing
    def emit(self, type_: str, data: Dict[str, Any]) -> Dict[str, Any]:
        ev = self.log.append(type_, data)
        self.state.apply(ev)
        return ev

    def spec(self, campaign: str) -> Spec:
        if campaign not in self._specs:
            path = Path(self.state.campaigns[campaign]["spec_path"])
            self._specs[campaign] = Spec.load(path)
        return self._specs[campaign]

    def known_nodes(self) -> Optional[set]:
        if self._known_nodes is None:
            try:
                self._known_nodes = self.backend.nodes()
            except Exception:
                self._known_nodes = None
        return self._known_nodes

    def signatures(self) -> List[Dict[str, Any]]:
        if self._signatures is None:
            sigs = list(DEFAULT_SIGNATURES)
            path = self.cfg.signatures_path
            if path.exists():
                with open(path) as fh:
                    extra = yaml.safe_load(fh) or []
                sigs = list(extra) + sigs  # group-specific rules win
            self._signatures = sigs
        return self._signatures

    # ------------------------------------------------------------ campaigns
    def create_campaign(self, spec_path: Path, name: Optional[str] = None, dry_run: bool = False) -> Dict[str, Any]:
        spec = Spec.load(spec_path)
        if name:
            spec.data["name"] = name
            spec.validate()
        if spec.name in self.state.campaigns:
            raise ValueError(f"campaign '{spec.name}' already exists; pass --name to start another")
        plans = {st: spec.plan(st) for st in spec.stage_names}
        if dry_run:
            return {"name": spec.name, "stages": {k: len(v) for k, v in plans.items()}, "plans": plans}
        cdir = self.cfg.campaigns_dir / spec.name
        cdir.mkdir(parents=True, exist_ok=True)
        frozen = cdir / "spec.yaml"
        atomic_write(frozen, f"# frozen copy of {spec.source} at {now()}\n" + spec.to_yaml())
        atomic_write(cdir / "original.yaml", Path(spec.source).read_text())
        self.emit("campaign.created", {
            "name": spec.name, "spec_path": str(frozen), "source": str(spec.source),
            "spec_hash": file_hash(frozen), "stages": spec.stage_names,
            "hypothesis": spec.data.get("hypothesis"), "test": bool(spec.data.get("test")),
            "part": spec.data.get("part"), "success": spec.data.get("success"),
            "provenance": git_provenance(Path(spec.data["workdir"])),
        })
        # Submitting a campaign is the human approval of its first stage.
        self.emit("stage.status", {"campaign": spec.name, "stage": spec.stage_names[0], "status": "approved",
                                   "reason": "approved by submission"})
        return {"name": spec.name, "stages": {k: len(v) for k, v in plans.items()}}

    def approve(self, campaign: str, stage: str) -> None:
        self.state = State.load(self.log)
        st = self.state.stage(campaign, stage)
        if st is None:
            raise ValueError(f"no stage {campaign}/{stage}")
        if st["status"] not in ("pending", "awaiting_approval"):
            raise ValueError(f"stage {campaign}/{stage} is {st['status']}, not awaiting approval")
        self.emit("stage.status", {"campaign": campaign, "stage": stage, "status": "approved",
                                   "reason": "approved by a human"})

    def cancel_campaign(self, campaign: str, reason: str = "cancelled by user") -> int:
        self.state = State.load(self.log)
        n = 0
        for run in self.state.runs_of(campaign):
            a = self.state.active_attempt_of(run)
            if a:
                self.emit("attempt.cancel_requested", {"attempt_id": a["attempt_id"], "reason": "campaign_cancelled"})
                self.backend.cancel(a["job_id"])
                n += 1
            if run["status"] not in ("succeeded", "failed", "cancelled"):
                self.emit("run.finished", {"run_id": run["run_id"], "status": "cancelled",
                                           "classification": "cancelled"})
        self.emit("campaign.status", {"campaign": campaign, "status": "cancelled", "reason": reason})
        return n

    # ------------------------------------------------------------ tick
    def tick(self, wait: float = 0.0) -> Dict[str, Any]:
        """One reconciliation pass. Only one tick runs at a time across all processes
        (brain, CLI, agents); a caller that cannot get the lock within ``wait`` seconds
        skips its tick, because the holder is doing the same work."""
        lock = self.cfg.home / ".tick.lock"
        if not try_lock(lock, timeout=wait, stale=TICK_LOCK_STALE_S):
            return {"skipped": "another tick is in progress"}
        self._lock = lock
        try:
            # Decide on the latest events, not on whatever was loaded at construction.
            self.state = State.load(self.log)
            return self._tick()
        finally:
            release_lock(lock)

    def _keep_lock(self) -> None:
        """Refresh the tick lock during long ticks, so only a dead holder's lock goes stale."""
        lock = getattr(self, "_lock", None)
        if lock is not None:
            try:
                os.utime(lock, None)
            except OSError:
                pass

    def _tick(self) -> Dict[str, Any]:
        summary = {"polled": 0, "finished": 0, "retried": 0, "submitted": 0, "stalled": 0}
        self._known_nodes = None
        active = self.state.active_attempts()
        infos = self.backend.query([a["job_id"] for a in active]) if active else {}
        summary["polled"] = len(active)
        for a in active:
            self._keep_lock()
            info = infos.get(a["job_id"])
            if info is None:
                self._handle_missing(a, summary)
                continue
            self._observe(a, info)
            if info.terminal:
                self._finalize(a, info)
                summary["finished"] += 1
            elif info.state == "RUNNING":
                if self._check_stall(a, info):
                    summary["stalled"] += 1
        summary["retried"] = sum(1 for r in self.state.runs.values() if r["status"] == "retrying")
        for name, c in list(self.state.campaigns.items()):
            if c["status"] == "active":
                self._advance(name)
        summary["submitted"] = self._submit_pending()
        return summary

    def _observe(self, a: Dict[str, Any], info: JobInfo) -> None:
        changed = {}
        if info.state != a.get("state"):
            changed["state"] = info.state
        if info.node and info.node != a.get("node"):
            changed["node"] = info.node
        if info.restarts != a.get("restarts", 0):
            changed["restarts"] = info.restarts
        if info.start and info.start not in ("Unknown", "None") and info.start != a.get("start"):
            changed["start"] = info.start
        if changed:
            changed.update(attempt_id=a["attempt_id"], job_id=a["job_id"], reason=info.reason)
            self.emit("attempt.state", changed)

    def _handle_missing(self, a: Dict[str, Any], summary: Dict[str, int]) -> None:
        age = age_seconds(a.get("submitted")) or 0
        if age > self.cfg["watch"]["lost_after_minutes"] * 60:
            info = JobInfo(job_id=a["job_id"], state="LOST")
            self._finalize(a, info, forced="infra/lost", detail="scheduler has no record of this job")
            summary["finished"] += 1

    def _check_stall(self, a: Dict[str, Any], info: JobInfo) -> bool:
        if a.get("cancel_reason"):
            return False
        run = self.state.runs[a["run_id"]]
        spec = self.spec(run["campaign"])
        hb = read_json(job_dir(run, a["job_id"]) / "heartbeat.json")
        running_for = age_seconds(a.get("start")) or 0
        reason = None
        stale_min = self.cfg["watch"]["heartbeat_stale_minutes"]
        if running_for > stale_min * 60:
            hb_age = age_seconds((hb or {}).get("ts"))
            if hb_age is None or hb_age > stale_min * 60:
                reason = f"no runner heartbeat for {stale_min}+ min"
        stall_min = (spec.data.get("progress") or {}).get("stall_minutes")
        if reason is None and stall_min and hb and hb.get("child_alive"):
            prog_age = age_seconds(hb.get("last_progress"))
            if prog_age is not None and prog_age > float(stall_min) * 60 and running_for > float(stall_min) * 60:
                reason = f"no training progress for {stall_min}+ min (last step {hb.get('step')})"
        if reason:
            self.emit("attempt.cancel_requested", {"attempt_id": a["attempt_id"], "reason": "hung", "detail": reason})
            self.backend.cancel(a["job_id"])
            self.echo(f"cancelled hung job {a['job_id']} ({run['run_id']}): {reason}")
            return True
        return False

    # ------------------------------------------------------------ classification
    def classify(self, a: Dict[str, Any], info: JobInfo) -> Tuple[str, str, Dict[str, Any]]:
        run = self.state.runs[a["run_id"]]
        jdir = job_dir(run, a["job_id"])
        res = read_json(jdir / "result.json") or {}
        pf = read_json(jdir / "preflight.json") or {}
        out = Path(a.get("output", "").replace("%j", a["job_id"]))
        tail = tail_lines(out, n=200) if a.get("output") else []
        sig = self._match_signature(tail)
        status = res.get("status")
        cancel = a.get("cancel_reason")
        state = info.state

        if cancel == "hung":
            return "infra/hung", "cancelled after a stall", res
        if cancel == "campaign_cancelled":
            return "cancelled", "campaign cancelled", res
        if status == "preflight_failed" or (pf and not pf.get("ok", True)):
            return "infra/node", f"preflight failed: {res.get('failed_checks') or 'see preflight.json'}", res
        if status == "ok":
            return "ok", "contract satisfied", res
        if status in ("contract", "config_mismatch", "diverged"):
            fails = (res.get("contract") or {}).get("failures") or []
            return status, "; ".join(fails[:3]), res
        if status == "runner_error":
            return "infra/runner", res.get("error", "runner error"), res
        if status in ("requeued", "timeout"):
            return "infra/timeout", "time limit reached", res
        if status == "requeue_failed":
            return "infra/requeue_failed", res.get("error", ""), res
        if status == "timeout_exhausted":
            return "infra/timeout_exhausted", "restart limit reached", res
        if state == "OUT_OF_MEMORY" or sig == "infra/oom":
            return "infra/oom", "out of memory", res
        if status == "terminated":
            mapping = {"PREEMPTED": "infra/preempt", "TIMEOUT": "infra/timeout", "NODE_FAIL": "infra/node",
                       "CANCELLED": "cancelled", "DEADLINE": "infra/timeout"}
            return mapping.get(state, "infra/unknown"), f"terminated ({state})", res
        if status == "code_failed":
            if sig and sig.startswith("infra/"):
                return sig, "log matches a known infrastructure signature", res
            return (sig or "code"), f"exit code {res.get('exit_code')}", res
        # No verdict from the runner: fall back to the scheduler's view.
        if state in ("NODE_FAIL", "BOOT_FAIL"):
            return "infra/node", state, res
        if state == "PREEMPTED":
            return "infra/preempt", state, res
        if state in ("TIMEOUT", "DEADLINE"):
            return "infra/timeout", state, res
        if state == "CANCELLED":
            return "cancelled", f"cancelled outside auto-experiment ({info.exit_code})", res
        if state == "LOST":
            return "infra/lost", "lost", res
        if sig:
            return sig, "log signature", res
        if info.elapsed_s <= 10 and len(tail) <= 2:
            return "infra/node", f"died after {info.elapsed_s}s with no output ({info.exit_code})", res
        return "infra/unknown", f"{state} {info.exit_code} without a runner verdict", res

    def _match_signature(self, lines: List[str]) -> Optional[str]:
        text = "\n".join(lines)
        for sig in self.signatures():
            if re.search(sig["pattern"], text, flags=re.IGNORECASE):
                return sig["class"]
        return None

    # ------------------------------------------------------------ finishing attempts
    def _finalize(self, a: Dict[str, Any], info: JobInfo, forced: Optional[str] = None, detail: str = "") -> None:
        run = self.state.runs[a["run_id"]]
        if forced:
            klass, res = forced, {}
        else:
            klass, detail, res = self.classify(a, info)
        node = info.node or res.get("node") or a.get("node") or ""
        self.emit("attempt.finished", {
            "attempt_id": a["attempt_id"], "run_id": run["run_id"], "job_id": a["job_id"],
            "slurm_state": info.state, "exit_code": info.exit_code, "node": node,
            "elapsed_s": info.elapsed_s, "restarts": info.restarts, "classification": klass,
            "detail": detail, "result_status": res.get("status"),
        })
        self.echo(f"{run['run_id']}: job {a['job_id']} -> {klass} ({detail})")
        if klass == "infra/node" and node:
            spec = self.spec(run["campaign"])
            scope = run["campaign"] if spec.data.get("test") else None
            entry = nodes_mod.flag(self.cfg.nodes_path, node, detail,
                                   evidence=[f"job {a['job_id']} {now()}"], scope=scope)
            self.emit("node.flagged", {"node": node, "status": entry["status"], "reason": detail,
                                       "job_id": a["job_id"], "scope": scope, "expires": entry["expires"]})
        if klass == "ok":
            self.emit("run.finished", {"run_id": run["run_id"], "status": "succeeded", "classification": "ok"})
            return
        overrides = self._retry_overrides(run, a, klass)
        if overrides is not None:
            self.emit("run.retry", {"run_id": run["run_id"], "reason": klass, "overrides": overrides})
        else:
            self.emit("run.finished", {"run_id": run["run_id"],
                                       "status": "cancelled" if klass == "cancelled" else "failed",
                                       "classification": klass})

    def _retry_overrides(self, run: Dict[str, Any], a: Dict[str, Any], klass: str) -> Optional[Dict[str, Any]]:
        """Resource changes for the next attempt, or None if this run must not be retried."""
        if klass not in RETRYABLE:
            return None
        spec = self.spec(run["campaign"])
        policy = spec.data["retry"]
        history = [x["classification"] for x in self.state.attempts_of(run) if x["finished"]]
        same = history.count(klass)
        infra_total = sum(1 for k in history if class_family(k) == "infra")
        res = spec.resources_for(run["stage"], run.get("overrides"))
        if klass == "infra/oom":
            if same > int(policy["oom"]):
                return None
            mb = parse_mem_mb(res["mem"])
            return {"mem": f"{int(math.ceil(mb * float(policy['oom_factor'])))}M"}
        if klass == "infra/timeout":
            if same > int(policy["timeout"]) + (5 if spec.data["resume"].get("enabled") else 0):
                return None
            if spec.data["resume"].get("enabled"):
                return {}
            return {"time": format_time_s(parse_time_s(res["time"]) * float(policy["timeout_factor"]))}
        if klass == "infra/hung":
            return {} if same <= int(policy["hung"]) else None
        if klass == "infra/preempt":
            return {} if same <= int(policy["preempt"]) else None
        return {} if infra_total <= int(policy["infra"]) else None

    # ------------------------------------------------------------ stages
    def _advance(self, campaign: str) -> None:
        c = self.state.campaigns[campaign]
        spec = self.spec(campaign)
        stages = c["stages"]
        for i, st in enumerate(stages):
            status = st["status"]
            if status == "succeeded":
                continue
            if status == "failed":
                self.emit("campaign.status", {"campaign": campaign, "status": "blocked",
                                              "reason": f"stage {st['name']} failed: {st['reason']}"})
                return
            if status == "pending":
                if i > 0 and stages[i - 1]["status"] != "succeeded":
                    return
                sdef = spec.stage(st["name"])
                if sdef.get("auto") and st["name"] in (self.cfg["auto_stages"] or []):
                    self._start_stage(campaign, st["name"], "auto (spec + autonomy policy)")
                else:
                    why = ("not in auto_stages policy" if sdef.get("auto") else "spec requires approval")
                    self.emit("stage.status", {"campaign": campaign, "stage": st["name"],
                                               "status": "awaiting_approval", "reason": why})
                return
            if status == "awaiting_approval":
                return
            if status == "approved":
                self._start_stage(campaign, st["name"], st.get("reason") or "approved")
                return
            if status == "running":
                if not self._finish_stage_if_done(campaign, st["name"]):
                    return
                continue
        if all(s["status"] == "succeeded" for s in stages):
            self.emit("campaign.status", {"campaign": campaign, "status": "done", "reason": "all stages succeeded"})

    def _start_stage(self, campaign: str, stage: str, why: str) -> None:
        spec = self.spec(campaign)
        for plan in spec.plan(stage):
            if plan["run_id"] not in self.state.runs:
                self.emit("run.planned", plan)
        self.emit("stage.status", {"campaign": campaign, "stage": stage, "status": "running", "reason": why})

    def _finish_stage_if_done(self, campaign: str, stage: str) -> bool:
        runs = self.state.runs_of(campaign, stage)
        if any(r["status"] not in ("succeeded", "failed", "cancelled") for r in runs):
            return False
        failed = [r for r in runs if r["status"] != "succeeded"]
        spec = self.spec(campaign)
        allowed = int(spec.stage(stage).get("max_failures", 0))
        echoes = {r["run_id"]: self._echo_of(r) for r in runs if r["status"] == "succeeded"}
        noops = contract_mod.detect_noops([r for r in runs if r["status"] == "succeeded"], echoes)
        for finding in noops:
            self.emit("noop.detected", dict(finding, campaign=campaign, stage=stage))
        self._freeze_results(campaign, stage, runs)
        if len(failed) > allowed:
            classes = sorted({r.get("final_class") or "?" for r in failed})
            reason = f"{len(failed)}/{len(runs)} runs failed ({', '.join(classes)})"
            self.emit("stage.status", {"campaign": campaign, "stage": stage, "status": "failed", "reason": reason})
            self.emit("campaign.status", {"campaign": campaign, "status": "blocked", "reason": f"{stage}: {reason}"})
            return False
        if noops:
            params = ", ".join(f["param"] for f in noops)
            reason = f"no-op parameter(s): {params} (arms that asked for different values ran the same config)"
            self.emit("stage.status", {"campaign": campaign, "stage": stage, "status": "failed", "reason": reason})
            self.emit("campaign.status", {"campaign": campaign, "status": "blocked", "reason": f"{stage}: {reason}"})
            return False
        self.emit("stage.status", {"campaign": campaign, "stage": stage, "status": "succeeded",
                                   "reason": f"{len(runs) - len(failed)}/{len(runs)} runs satisfied the contract"})
        return True

    def _freeze_results(self, campaign: str, stage: str, runs: List[Dict[str, Any]]) -> None:
        """Snapshot the stage's results into the event log, so they survive even if
        run directories are deleted or result files are overwritten later."""
        from .brief import results_table
        classes: Dict[str, int] = {}
        for r in runs:
            k = r.get("final_class") or r["status"]
            classes[k] = classes.get(k, 0) + 1
        table = results_table(self.state, self.spec(campaign).data, campaign, stage)
        self.emit("stage.results", {"campaign": campaign, "stage": stage, "n_runs": len(runs),
                                    "classes": classes, "table": table})

    def _echo_of(self, run: Dict[str, Any]) -> Optional[dict]:
        spec = self.spec(run["campaign"])
        efile = (spec.data.get("contract") or {}).get("echo_file")
        if not efile:
            return None
        return read_json(Path(run["run_dir"]) / efile)

    # ------------------------------------------------------------ submission
    def _submit_pending(self) -> int:
        active_total = len(self.state.active_attempts())
        limit_total = int(self.cfg["max_active_jobs"])
        n = 0
        for run in sorted(self.state.runs.values(), key=lambda r: r["planned_ts"]):
            if not run["needs_submit"]:
                continue
            c = self.state.campaigns.get(run["campaign"])
            if not c or c["status"] != "active":
                continue
            spec = self.spec(run["campaign"])
            limit_c = int(spec.data["limits"].get("max_active_jobs", limit_total))
            active_c = sum(1 for a in self.state.active_attempts()
                           if self.state.runs[a["run_id"]]["campaign"] == run["campaign"])
            if active_total >= limit_total or active_c >= limit_c:
                continue
            self._keep_lock()
            if self._submit_attempt(run):
                active_total += 1
                n += 1
        return n

    def _submit_attempt(self, run: Dict[str, Any]) -> bool:
        spec = self.spec(run["campaign"])
        n = len(run["attempts"]) + 1
        attempt_id = f"{run['run_id']}#a{n}"
        res = spec.resources_for(run["stage"], run.get("overrides"))
        if not res.get("partition"):
            res["partition"] = self.cfg.get("default_partition")
        run_dir = Path(run["run_dir"])
        bdir = run_dir / ".autoexp" / f"attempt-{n}"
        bdir.mkdir(parents=True, exist_ok=True)
        mapping = dict(run["params"])
        mapping.update(seed=run["seed"], run_dir=str(run_dir), python=self.cfg.python, autoexp_src=AUTOEXP_SRC,
                       campaign=run["campaign"], stage=run["stage"], run_id=run["run_id"],
                       workdir=spec.data["workdir"], attempt=n)
        command = subst(spec.data["command"], mapping)
        recipe = subst(spec.data.get("recipe") or "", mapping)
        if spec.data.get("recipe_file"):
            recipe = Path(spec.data["recipe_file"]).read_text() + "\n" + recipe
        atomic_write(bdir / "recipe.sh", "# frozen environment recipe\n" + recipe + "\n")
        atomic_write(bdir / "command.sh", "#!/bin/bash\n# frozen command\n" + command + "\n")
        output = str(run_dir / ".autoexp" / "slurm-%j.out")
        resume = spec.data["resume"]
        bundle = {
            "run_id": run["run_id"], "campaign": run["campaign"], "stage": run["stage"], "attempt": n,
            "run_dir": str(run_dir), "workdir": spec.data["workdir"], "gpus": int(res.get("gpus") or 0),
            "preflight": [dict(p, cmd=subst(p["cmd"], mapping)) for p in spec.data.get("preflight") or []],
            "contract": spec.data.get("contract") or {}, "mapping": mapping,
            "progress": spec.data.get("progress") or {}, "resume": resume, "output": output,
            "backend": self.backend.name, "resources": res,
        }
        write_json(bdir / "bundle.json", bundle)
        excludes = nodes_mod.excludes(self.cfg.nodes_path, run["campaign"], self.known_nodes())
        tag = f"ae:{run['run_id']};plan={self.state.campaigns[run['campaign']]['spec_hash']}"
        runner_cmd = (f"env PYTHONPATH={shlex.quote(AUTOEXP_SRC)} {shlex.quote(self.cfg.python)} "
                      f"-m autoexp.runner {shlex.quote(str(bdir / 'bundle.json'))}")
        script = render_sbatch(
            job_name=f"ae:{run['campaign']}:{run['stage']}"[:120], resources=res, output=output, tag=tag,
            excludes=excludes, defaults=self.cfg["sbatch"], command=runner_cmd,
            signal_seconds=resume.get("signal_seconds") if resume.get("enabled") else None)
        sbatch_path = bdir / "job.sbatch"
        atomic_write(sbatch_path, script)
        protocol = stable_hash({"sbatch": script, "recipe": recipe, "command": command}, 12)
        try:
            job_id = self.backend.submit(sbatch_path)
        except Exception as exc:
            self.emit("attempt.submit_failed", {"run_id": run["run_id"], "attempt": n, "error": str(exc)[:500]})
            self.echo(f"submit failed for {run['run_id']}: {exc}")
            if run["submit_failures"] >= 3:
                self.emit("run.finished", {"run_id": run["run_id"], "status": "failed", "classification": "submit"})
            return False
        self.emit("attempt.submitted", {
            "attempt_id": attempt_id, "run_id": run["run_id"], "n": n, "job_id": job_id,
            "resources": res, "excludes": excludes, "bundle": str(bdir), "output": output,
            "protocol_hash": protocol,
        })
        self.echo(f"submitted {run['run_id']} attempt {n} as job {job_id}")
        return True


# ---------------------------------------------------------------- helpers

def job_dir(run: Dict[str, Any], job_id: str) -> Path:
    return Path(run["run_dir"]) / ".autoexp" / f"job-{job_id}"


def git_provenance(path: Path) -> Dict[str, Any]:
    def git(*args: str) -> str:
        try:
            return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True,
                                  timeout=20).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            return ""
    commit = git("rev-parse", "HEAD")
    if not commit:
        return {}
    dirty = git("status", "--porcelain")
    return {"commit": commit, "dirty": bool(dirty), "dirty_hash": stable_hash(git("diff"), 12) if dirty else None}
