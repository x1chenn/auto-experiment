"""The baton protocol: sessions, batons, tasks and the generated HANDOFF.md.

Sessions are disposable; state lives outside them. Any agent (any vendor, any
model size) or person registers a session, reads HANDOFF.md, claims a task,
and leaves a baton when it stops. If it stops without one (crash, context
limit, closed window), a baton is reconstructed from the events it caused.
HANDOFF.md is generated from the event log and never edited by hand, so it
cannot go stale without saying so.
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from . import nodes as nodes_mod
from .config import Config
from .events import EventLog, State
from .util import age_seconds, atomic_write, new_id, now, read_json

TIERS = ("any", "standard", "strong")
BATON_FIELDS = ("goal", "done", "in_flight", "next", "open_questions", "unverified", "touched")
HANDOFF_MAX_LINES = 200


# ---------------------------------------------------------------- sessions

def session_start(log: EventLog, agent: str, model: str = "", purpose: str = "",
                  session: Optional[str] = None, tier: Optional[str] = None) -> str:
    sid = session or new_id(f"s-{(agent or 'human')[:10]}-")
    log.append("session.started", {"session": sid, "agent": agent or "human", "model": model,
                                   "purpose": purpose, "tier": tier or guess_tier(model),
                                   "cwd": os.getcwd()},
               actor=_actor(sid, agent, model))
    return sid


def session_end(log: EventLog, sid: str) -> None:
    log.append("session.ended", {"session": sid}, actor=_actor(sid))


def guess_tier(model: str) -> str:
    m = (model or "").lower()
    if any(k in m for k in ("opus", "fable", "xhigh", "high", "gpt-6", "o3")):
        return "strong"
    if any(k in m for k in ("haiku", "mini", "nano", "low")):
        return "any"
    return "standard" if m else "any"


def _actor(sid: Optional[str], agent: Optional[str] = None, model: Optional[str] = None) -> Dict[str, str]:
    from .util import default_actor
    a = default_actor()
    if sid:
        a["session"] = sid
    if agent:
        a["agent"] = agent
    if model:
        a["model"] = model
    return a


# ---------------------------------------------------------------- batons

def baton_write(cfg: Config, log: EventLog, sid: str, fields: Dict[str, Any],
                reconstructed: bool = False) -> Dict[str, Any]:
    unknown = set(fields) - set(BATON_FIELDS)
    if unknown:
        raise ValueError(f"unknown baton field(s) {sorted(unknown)}; allowed: {BATON_FIELDS}")
    data = {k: fields.get(k) for k in BATON_FIELDS if fields.get(k)}
    data.update(session=sid, reconstructed=reconstructed)
    log.append("baton.written", data, actor=_actor(sid))
    path = cfg.sessions_dir / f"{sid}.baton.md"
    atomic_write(path, render_baton(data, now()))
    return data


def reconstruct_baton(state: State, sid: str) -> Dict[str, Any]:
    """Best-effort baton from what a session did, for sessions that left none."""
    done, in_flight = [], []
    # Attribute jobs submitted while the session was open to it.
    sess = state.sessions.get(sid) or {}
    started = sess.get("started")
    for run in state.runs.values():
        for att in state.attempts_of(run):
            if started and att.get("submitted", "") >= started and (not sess.get("ended") or att["submitted"] <= sess["ended"]):
                (done if att["finished"] else in_flight).append(
                    f"job {att['job_id']} {run['run_id']} -> {att.get('classification') or att.get('state')}")
    return {"goal": sess.get("purpose") or "(not recorded)",
            "done": done[-20:] or [f"{state.by_session.get(sid, 0)} recorded actions; see `autoexp log --session {sid}`"],
            "in_flight": in_flight[-20:],
            "unverified": ["reconstructed automatically: the session ended without writing a baton"]}


def render_baton(b: Dict[str, Any], ts: str) -> str:
    lines = [f"# Baton {b.get('session')} ({ts})" + ("  [reconstructed]" if b.get("reconstructed") else "")]
    for key in BATON_FIELDS:
        val = b.get(key)
        if not val:
            continue
        lines.append(f"\n**{key.replace('_', ' ')}**")
        if isinstance(val, list):
            lines += [f"- {_fmt(v)}" for v in val]
        else:
            lines.append(_fmt(val))
    return "\n".join(lines) + "\n"


def _fmt(v: Any) -> str:
    if isinstance(v, dict):
        return "; ".join(f"{k}: {x}" for k, x in v.items())
    return str(v)


def close_unclean_sessions(cfg: Config, log: EventLog, state: State) -> List[str]:
    """Reconstruct batons for sessions that went quiet without leaving one."""
    limit = cfg["watch"]["session_unclean_hours"] * 3600
    closed = []
    for sid, s in state.sessions.items():
        if s.get("baton") or s.get("ended"):
            continue
        quiet = age_seconds(s.get("last_ts") or s.get("started")) or 0
        if quiet > limit:
            baton_write(cfg, log, sid, reconstruct_baton(state, sid), reconstructed=True)
            log.append("session.ended", {"session": sid, "unclean": True})
            closed.append(sid)
    return closed


# ---------------------------------------------------------------- tasks

def task_add(log: EventLog, text: str, tier: str = "any", accept: str = "", kind: str = "ops",
             campaign: Optional[str] = None) -> str:
    if tier not in TIERS:
        raise ValueError(f"tier must be one of {TIERS}")
    tid = new_id("T")
    log.append("task.added", {"task": tid, "text": text, "tier": tier, "accept": accept, "kind": kind,
                              "campaign": campaign})
    return tid


def task_claim(log: EventLog, state: State, tid: str, holder: str, hours: float = 4.0) -> None:
    t = state.tasks.get(tid)
    if t is None:
        raise ValueError(f"no task {tid}")
    if t["status"] == "done":
        raise ValueError(f"task {tid} is already done")
    if t["status"] == "claimed" and t.get("holder") != holder and not _lease_expired(t):
        raise ValueError(f"task {tid} is held by {t['holder']} until {t['lease_until']}")
    until = (_dt.datetime.now().astimezone() + _dt.timedelta(hours=hours)).isoformat(timespec="seconds")
    log.append("task.claimed", {"task": tid, "holder": holder, "lease_until": until})


def _lease_expired(t: Dict[str, Any]) -> bool:
    age = age_seconds(t.get("lease_until"))
    return age is not None and age > 0


# ---------------------------------------------------------------- HANDOFF.md

def brain_status(cfg: Config) -> Dict[str, Any]:
    lease = read_json(cfg.lease_path) or {}
    age = age_seconds(lease.get("heartbeat"))
    return {"alive": age is not None and age < 300, "age_s": age, **lease}


def render_handoff(cfg: Config, state: State, tier: Optional[str] = None) -> str:
    L: List[str] = []
    add = L.append
    add("# HANDOFF (generated - do not edit)")
    add(f"generated_at: {now()} | as_of_event: {state.seq} | regenerate: `autoexp handoff --write`")
    add("")
    b = brain_status(cfg)
    if b.get("alive"):
        add(f"**Brain:** alive (job {b.get('job')}, host {b.get('host')}, heartbeat {int(b['age_s'])}s ago)")
    else:
        add("**Brain:** NOT RUNNING - nothing advances by itself. Run `autoexp brain ensure`"
            " (or `autoexp tick` by hand).")
    add("")

    add("## Campaigns")
    active = [c for c in state.campaigns.values() if c["status"] in ("active", "blocked")]
    if not active:
        add("(none active)")
    for c in sorted(active, key=lambda c: c["created"]):
        counts: Dict[str, int] = {}
        for r in state.runs_of(c["name"]):
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        stages = " -> ".join(f"{s['name']}:{s['status']}" for s in c["stages"])
        add(f"- **{c['name']}** [{c['status']}] {stages}")
        add(f"  runs: {', '.join(f'{k}={v}' for k, v in sorted(counts.items())) or 'none yet'}")
        if c.get("hypothesis"):
            add(f"  hypothesis: {c['hypothesis']}")
        if c["status"] == "blocked":
            add(f"  BLOCKED: {c.get('status_reason')}")
        for s in c["stages"]:
            if s["status"] == "awaiting_approval":
                add(f"  AWAITING APPROVAL: stage `{s['name']}` ({s.get('reason')}) -> "
                    f"`autoexp approve {c['name']} {s['name']}` (human decision)")
    add("")

    add("## In flight")
    act = state.active_attempts()
    if not act:
        add("(no jobs)")
    for a in sorted(act, key=lambda a: a["submitted"])[:25]:
        add(f"- job {a['job_id']} {a['state']:<9} {a.get('node') or '-':<8} {a['run_id']}")
    if len(act) > 25:
        add(f"- ... and {len(act) - 25} more (`autoexp status`)")
    add("")

    add("## Recent failures (24 h)")
    recent = [a for a in state.attempts.values() if a["finished"] and a.get("classification") not in (None, "ok")
              and (age_seconds(a.get("finished_ts")) or 1e9) < 86400]
    if not recent:
        add("(none)")
    for a in sorted(recent, key=lambda a: a["finished_ts"])[-12:]:
        add(f"- {a['classification']}: {a['run_id']} job {a['job_id']} on {a.get('node') or '?'} - {a.get('detail')}")
    for n in state.noops[-5:]:
        add(f"- NO-OP PARAMETER: {n['campaign']}/{n['stage']} `{n['param']}` asked {n['asked']} echoed {n['echoed']}")
    add("")

    add("## Nodes excluded right now")
    ex = [(k, e) for k, e in nodes_mod.entries(cfg.nodes_path).items() if nodes_mod.active(e)]
    add(", ".join(f"{k} ({e.get('status')}, until {e.get('expires', '')[:16]})" for k, e in ex) or "(none)")
    add("")

    add("## Open tasks")
    open_t = [t for t in state.tasks.values() if t["status"] != "done"]
    rank = {t: i for i, t in enumerate(TIERS)}
    visible = [t for t in open_t if tier is None or rank[t["tier"]] <= rank.get(tier, 2)]
    if not visible:
        add("(none)")
    for t in sorted(visible, key=lambda t: t["created"]):
        holder = f" - held by {t['holder']} until {t['lease_until'][:16]}" if t["status"] == "claimed" else ""
        add(f"- [{t['task']}] ({t['tier']}/{t['kind']}) {t['text']}" + (f" | accept: {t['accept']}" if t.get("accept") else "") + holder)
    hidden = len(open_t) - len(visible)
    if hidden:
        add(f"- ({hidden} more task(s) need a stronger tier)")
    add("")

    add("## Last batons")
    if not state.batons:
        add("(none)")
    for bt in state.batons[-3:]:
        who = bt.get("actor") or {}
        tag = " [reconstructed]" if bt.get("reconstructed") else ""
        add(f"- {bt['ts'][:16]} {bt.get('session')} ({who.get('agent', '?')}/{who.get('model', '?')}){tag}: "
            f"{_fmt(bt.get('goal') or '')}")
        for key in ("next", "open_questions", "in_flight"):
            for item in (bt.get(key) or [])[:4]:
                add(f"  - {key}: {_fmt(item)}")
    add("")

    add("## Recent decisions")
    if not state.decisions:
        add("(none)")
    for d in state.decisions[-5:]:
        add(f"- {d['ts'][:16]} {d.get('text')} (why: {d.get('why')})")

    if len(L) > HANDOFF_MAX_LINES:
        L = L[:HANDOFF_MAX_LINES - 1] + ["... (truncated; use `autoexp status`)"]
    return "\n".join(L) + "\n"


def write_handoff(cfg: Config, state: State) -> Path:
    atomic_write(cfg.handoff_path, render_handoff(cfg, state))
    return cfg.handoff_path


def load_baton_file(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError("baton file must be a YAML mapping")
    return data
