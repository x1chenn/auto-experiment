"""Append-only event log and the state derived from it.

Every change goes through ``EventLog.append``. Writers are serialized by a
short lock and every event gets a monotonically increasing ``seq``. ``State``
is never stored as the truth: it is rebuilt by replaying the log, so a lost
cache or a crashed process can never leave the system in an unknown state.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

from .config import Config
from .util import default_actor, now, short_lock, today


class EventLog:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.dir = cfg.events_dir
        self.lock = cfg.home / ".events.lock"
        self.seq_file = self.dir / ".seq"

    def append(self, type_: str, data: Dict[str, Any], actor: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        self.dir.mkdir(parents=True, exist_ok=True)
        with short_lock(self.lock):
            try:
                seq = int(self.seq_file.read_text().strip() or 0) + 1
            except (OSError, ValueError):
                seq = self._max_seq_on_disk() + 1
            ev = {"seq": seq, "ts": now(), "type": type_, "actor": actor or default_actor(), "data": data}
            with open(self.dir / f"{today()}.jsonl", "a") as fh:
                fh.write(json.dumps(ev, sort_keys=True, default=str) + "\n")
                fh.flush()
                os.fsync(fh.fileno())
            self.seq_file.write_text(f"{seq}\n")
        return ev

    def _max_seq_on_disk(self) -> int:
        best = 0
        for ev in self.iter():
            best = max(best, int(ev.get("seq", 0)))
        return best

    def iter(self) -> Iterator[Dict[str, Any]]:
        if not self.dir.exists():
            return
        for path in sorted(self.dir.glob("*.jsonl")):
            with open(path) as fh:
                for lineno, line in enumerate(fh, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        yield json.loads(line)
                    except ValueError:
                        print(f"[autoexp] skipping corrupt event {path.name}:{lineno}", file=sys.stderr)


FAMILY_INFRA = "infra"


def class_family(klass: Optional[str]) -> str:
    return (klass or "").split("/", 1)[0]


class State:
    """In-memory view rebuilt from events. Handlers are tolerant of unknown types."""

    def __init__(self) -> None:
        self.seq = 0
        self.campaigns: Dict[str, Dict[str, Any]] = {}
        self.runs: Dict[str, Dict[str, Any]] = {}
        self.attempts: Dict[str, Dict[str, Any]] = {}
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.batons: List[Dict[str, Any]] = []
        self.tasks: Dict[str, Dict[str, Any]] = {}
        self.decisions: List[Dict[str, Any]] = []
        self.noops: List[Dict[str, Any]] = []
        self.node_flags: List[Dict[str, Any]] = []
        self.brain: Dict[str, Any] = {}
        self.by_session: Counter = Counter()

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, log: EventLog) -> "State":
        st = cls()
        for ev in log.iter():
            st.apply(ev)
        return st

    def apply(self, ev: Dict[str, Any]) -> None:
        self.seq = max(self.seq, int(ev.get("seq", 0)))
        sess = (ev.get("actor") or {}).get("session")
        if sess:
            self.by_session[sess] += 1
            if sess in self.sessions:
                self.sessions[sess]["last_ts"] = ev["ts"]
                self.sessions[sess]["events"] += 1
        handler = getattr(self, "_on_" + ev["type"].replace(".", "_"), None)
        if handler:
            handler(ev, ev.get("data") or {})

    # ------------------------------------------------------------ campaigns
    def _on_campaign_created(self, ev, d):
        self.campaigns[d["name"]] = {
            "name": d["name"],
            "spec_path": d["spec_path"],
            "spec_hash": d.get("spec_hash"),
            "hypothesis": d.get("hypothesis"),
            "test": bool(d.get("test")),
            "created": ev["ts"],
            "actor": ev.get("actor"),
            "status": "active",
            "status_reason": None,
            "stages": [{"name": s, "status": "pending", "reason": None, "ts": ev["ts"]} for s in d["stages"]],
        }

    def _on_campaign_status(self, ev, d):
        c = self.campaigns.get(d["campaign"])
        if c:
            c["status"] = d["status"]
            c["status_reason"] = d.get("reason")
            c["status_ts"] = ev["ts"]

    def _on_stage_status(self, ev, d):
        st = self.stage(d["campaign"], d["stage"])
        if st is not None:
            st["status"] = d["status"]
            st["reason"] = d.get("reason")
            st["ts"] = ev["ts"]

    def stage(self, campaign: str, stage: str) -> Optional[Dict[str, Any]]:
        c = self.campaigns.get(campaign)
        if not c:
            return None
        for st in c["stages"]:
            if st["name"] == stage:
                return st
        return None

    # ------------------------------------------------------------ runs
    def _on_run_planned(self, ev, d):
        run = dict(d)
        run.update({"status": "planned", "attempts": [], "overrides": {}, "needs_submit": True,
                    "submit_failures": 0, "planned_ts": ev["ts"], "final_class": None})
        self.runs[d["run_id"]] = run

    def _on_run_retry(self, ev, d):
        run = self.runs.get(d["run_id"])
        if run:
            run["needs_submit"] = True
            run["status"] = "retrying"
            run["overrides"].update(d.get("overrides") or {})
            run["retry_reason"] = d.get("reason")

    def _on_run_finished(self, ev, d):
        run = self.runs.get(d["run_id"])
        if run:
            run["status"] = d["status"]
            run["final_class"] = d.get("classification")
            run["needs_submit"] = False
            run["finished_ts"] = ev["ts"]

    def _on_attempt_submitted(self, ev, d):
        a = dict(d)
        a.update({"state": "SUBMITTED", "finished": False, "submitted": ev["ts"], "node": "",
                  "restarts": 0, "classification": None, "cancel_reason": None})
        self.attempts[d["attempt_id"]] = a
        run = self.runs.get(d["run_id"])
        if run:
            run["attempts"].append(d["attempt_id"])
            run["needs_submit"] = False
            run["status"] = "queued"

    def _on_attempt_submit_failed(self, ev, d):
        run = self.runs.get(d["run_id"])
        if run:
            run["submit_failures"] += 1
            run["last_error"] = d.get("error")

    def _on_attempt_state(self, ev, d):
        a = self.attempts.get(d["attempt_id"])
        if not a:
            return
        for key in ("state", "node", "restarts", "start", "reason"):
            if key in d:
                a[key] = d[key]
        run = self.runs.get(a["run_id"])
        if run and d.get("state") == "RUNNING":
            run["status"] = "running"
        elif run and d.get("state") in ("PENDING", "REQUEUED"):
            run["status"] = "queued"

    def _on_attempt_cancel_requested(self, ev, d):
        a = self.attempts.get(d["attempt_id"])
        if a:
            a["cancel_reason"] = d.get("reason")

    def _on_attempt_finished(self, ev, d):
        a = self.attempts.get(d["attempt_id"])
        if not a:
            return
        a.update({k: v for k, v in d.items() if k != "attempt_id"})
        a["finished"] = True
        a["finished_ts"] = ev["ts"]

    # ------------------------------------------------------------ other records
    def _on_noop_detected(self, ev, d):
        self.noops.append(dict(d, ts=ev["ts"]))

    def _on_node_flagged(self, ev, d):
        self.node_flags.append(dict(d, ts=ev["ts"]))

    def _on_session_started(self, ev, d):
        self.sessions[d["session"]] = dict(d, started=ev["ts"], last_ts=ev["ts"], ended=None,
                                           baton=False, events=0)

    def _on_session_ended(self, ev, d):
        s = self.sessions.get(d["session"])
        if s:
            s["ended"] = ev["ts"]

    def _on_baton_written(self, ev, d):
        self.batons.append(dict(d, ts=ev["ts"], actor=ev.get("actor")))
        s = self.sessions.get(d.get("session"))
        if s:
            s["baton"] = True

    def _on_task_added(self, ev, d):
        self.tasks[d["task"]] = dict(d, status="open", holder=None, lease_until=None, created=ev["ts"])

    def _on_task_claimed(self, ev, d):
        t = self.tasks.get(d["task"])
        if t:
            t.update(status="claimed", holder=d.get("holder"), lease_until=d.get("lease_until"))

    def _on_task_released(self, ev, d):
        t = self.tasks.get(d["task"])
        if t:
            t.update(status="open", holder=None, lease_until=None)

    def _on_task_done(self, ev, d):
        t = self.tasks.get(d["task"])
        if t:
            t.update(status="done", done_ts=ev["ts"], evidence=d.get("evidence"), holder=d.get("holder"))

    def _on_decision_recorded(self, ev, d):
        self.decisions.append(dict(d, ts=ev["ts"], actor=ev.get("actor")))

    def _on_brain_started(self, ev, d):
        self.brain.update(d, started=ev["ts"])

    def _on_brain_tick(self, ev, d):
        self.brain["last_tick"] = ev["ts"]

    # ------------------------------------------------------------ queries
    def runs_of(self, campaign: str, stage: Optional[str] = None) -> List[Dict[str, Any]]:
        return [r for r in self.runs.values()
                if r["campaign"] == campaign and (stage is None or r["stage"] == stage)]

    def attempts_of(self, run: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [self.attempts[a] for a in run["attempts"] if a in self.attempts]

    def active_attempts(self) -> List[Dict[str, Any]]:
        return [a for a in self.attempts.values() if not a["finished"]]

    def active_attempt_of(self, run: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        for a in self.attempts_of(run):
            if not a["finished"]:
                return a
        return None
