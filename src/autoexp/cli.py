"""Command line interface: ``autoexp <command>`` (or ``python -m autoexp``).

The CLI is the lowest common denominator for every human and every agent:
anything that can run a shell command can drive auto-experiment.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List

import yaml

from . import __version__
from . import nodes as nodes_mod
from .config import CONFIG_TEMPLATE, load_config
from .events import EventLog, State
from .util import atomic_write, default_actor


def _cfg():
    cfg = load_config()
    cfg.ensure_dirs()
    return cfg


def _engine(cfg):
    from .engine import Engine
    from .slurm import make_backend
    return Engine(cfg, make_backend(cfg))


def _print_json(obj: Any) -> None:
    print(json.dumps(obj, indent=2, sort_keys=True, default=str))


# ---------------------------------------------------------------- commands

def cmd_init(args) -> int:
    cfg = load_config()
    cfg.ensure_dirs()
    path = cfg.home / "config.yaml"
    if not path.exists():
        atomic_write(path, CONFIG_TEMPLATE.format(python=sys.executable))
        print(f"wrote {path} (edit account/partitions there)")
    else:
        print(f"{path} exists; leaving it alone")
    if args.project:
        from .templates import install_project_files
        for line in install_project_files(Path(args.project)):
            print(line)
    print(f"state directory: {cfg.home}\nshared directory: {cfg.shared}")
    return 0


def cmd_check(args) -> int:
    cfg = load_config()
    ok = True
    print(f"autoexp {__version__}, python {sys.version.split()[0]} ({sys.executable})")
    print(f"home   {cfg.home}  {'ok' if cfg.home.exists() else 'missing (run autoexp init)'}")
    print(f"shared {cfg.shared}")
    print(f"backend {cfg.backend_name}")
    if cfg.backend_name == "slurm":
        for tool in ("sbatch", "sacct", "squeue", "scancel", "scontrol", "sinfo"):
            found = shutil.which(tool)
            ok &= bool(found)
            print(f"  {tool:<9} {found or 'NOT FOUND'}")
    py = cfg.python
    print(f"runner python {py} {'ok' if os.path.exists(py) else 'NOT FOUND'}")
    return 0 if ok else 1


def cmd_submit(args) -> int:
    cfg = _cfg()
    eng = _engine(cfg)
    info = eng.create_campaign(Path(args.spec), name=args.name, dry_run=args.dry_run)
    if args.dry_run:
        for stage, plans in info["plans"].items():
            print(f"stage {stage}: {len(plans)} run(s)")
            for p in plans[: args.show]:
                print(f"  {p['run_id']}  {p['label']}")
        return 0
    print(f"campaign {info['name']} created: " + ", ".join(f"{k}={v} runs" for k, v in info["stages"].items()))
    if not args.no_tick:
        summary = eng.tick()
        print(f"tick: {summary}")
    return 0


def cmd_tick(args) -> int:
    cfg = _cfg()
    eng = _engine(cfg)
    summary = eng.tick()
    from .handoff import write_handoff
    write_handoff(cfg, eng.state)
    print(json.dumps(summary))
    return 0


def _campaign_rows(state: State, name: str) -> Dict[str, Any]:
    c = state.campaigns[name]
    counts: Dict[str, int] = {}
    for r in state.runs_of(name):
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    return {"name": name, "status": c["status"], "reason": c.get("status_reason"),
            "stages": {s["name"]: s["status"] for s in c["stages"]}, "runs": counts}


def cmd_status(args) -> int:
    cfg = _cfg()
    state = State.load(EventLog(cfg))
    names = [args.campaign] if args.campaign else [n for n, c in state.campaigns.items()
                                                    if args.all or c["status"] in ("active", "blocked")]
    from .handoff import brain_status
    data = {"brain": brain_status(cfg), "campaigns": [_campaign_rows(state, n) for n in names if n in state.campaigns],
            "active_jobs": [{"job": a["job_id"], "state": a["state"], "node": a.get("node"), "run": a["run_id"]}
                            for a in state.active_attempts()]}
    if args.json:
        _print_json(data)
        return 0
    b = data["brain"]
    print(f"brain: {'alive job ' + str(b.get('job')) if b.get('alive') else 'not running'}")
    for c in data["campaigns"]:
        stages = " -> ".join(f"{k}:{v}" for k, v in c["stages"].items())
        print(f"{c['name']} [{c['status']}] {stages}")
        print(f"   runs: {c['runs']}" + (f"\n   reason: {c['reason']}" if c["reason"] else ""))
    if data["active_jobs"]:
        print("jobs:")
        for j in data["active_jobs"]:
            print(f"   {j['job']:<10} {j['state']:<10} {j['node'] or '-':<10} {j['run']}")
    return 0


def cmd_runs(args) -> int:
    cfg = _cfg()
    state = State.load(EventLog(cfg))
    rows = []
    for r in sorted(state.runs_of(args.campaign, args.stage), key=lambda r: r["run_id"]):
        atts = state.attempts_of(r)
        last = atts[-1] if atts else {}
        rows.append({"run": r["run_id"], "label": r["label"], "status": r["status"],
                     "attempts": len(atts), "last_job": last.get("job_id"), "last_class": last.get("classification"),
                     "node": last.get("node"), "run_dir": r["run_dir"]})
    if args.json:
        _print_json(rows)
        return 0
    for x in rows:
        print(f"{x['status']:<10} a{x['attempts']} job {x['last_job'] or '-':<9} {x['last_class'] or '':<20} "
              f"{x['run']}  [{x['label']}]")
    return 0


def cmd_approve(args) -> int:
    cfg = _cfg()
    eng = _engine(cfg)
    eng.approve(args.campaign, args.stage)
    print(f"approved {args.campaign}/{args.stage}")
    if not args.no_tick:
        print(f"tick: {eng.tick()}")
    return 0


def cmd_cancel(args) -> int:
    cfg = _cfg()
    eng = _engine(cfg)
    n = eng.cancel_campaign(args.campaign, reason=args.reason)
    print(f"cancelled {args.campaign} ({n} job(s) cancelled)")
    return 0


def cmd_brief(args) -> int:
    cfg = _cfg()
    eng = _engine(cfg)
    from .brief import render_brief, write_brief
    specs = {n: eng.spec(n).data for n in eng.state.campaigns}
    text = render_brief(cfg, eng.state, specs, hours=args.hours)
    if args.write:
        print(f"wrote {write_brief(cfg, text)}")
    else:
        print(text)
    return 0


def cmd_handoff(args) -> int:
    cfg = _cfg()
    from .handoff import render_handoff, write_handoff
    state = State.load(EventLog(cfg))
    if args.write:
        print(f"wrote {write_handoff(cfg, state)}")
    else:
        print(render_handoff(cfg, state, tier=args.tier), end="")
    return 0


def cmd_log(args) -> int:
    cfg = _cfg()
    out: List[Dict[str, Any]] = []
    for ev in EventLog(cfg).iter():
        if args.session and (ev.get("actor") or {}).get("session") != args.session:
            continue
        if args.type and not ev["type"].startswith(args.type):
            continue
        out.append(ev)
    for ev in out[-args.tail:]:
        print(json.dumps(ev, sort_keys=True, default=str))
    return 0


def cmd_nodes(args) -> int:
    cfg = _cfg()
    if args.action == "list":
        for key, e in sorted(nodes_mod.entries(cfg.nodes_path).items()):
            live = "active" if nodes_mod.active(e) else "expired"
            print(f"{key:<24} {e.get('status'):<8} {live:<8} until {e.get('expires', '')[:16]}  {e.get('reason')}")
    elif args.action == "flag":
        if not args.node or not args.reason:
            print("usage: autoexp nodes flag NODE --reason TEXT [--days D]")
            return 2
        e = nodes_mod.flag(cfg.nodes_path, args.node, args.reason, evidence=[f"manual by {default_actor()['user']}"],
                           hours=args.days * 24 if args.days else None)
        EventLog(cfg).append("node.flagged", {"node": args.node, "status": e["status"], "reason": args.reason,
                                              "expires": e["expires"], "manual": True})
        print(f"{args.node}: {e['status']} until {e['expires']}")
    elif args.action == "clear":
        print(f"removed {nodes_mod.clear(cfg.nodes_path, args.node)} entr(y/ies)")
    return 0


def cmd_brain(args) -> int:
    cfg = _cfg()
    from .brain import Brain, ensure_brain, submit_brain
    from .handoff import brain_status
    if args.action == "run":
        return Brain(cfg).run(once=args.once)
    if args.action == "submit":
        print(f"brain job {submit_brain(cfg)}")
    elif args.action == "ensure":
        print(json.dumps(ensure_brain(cfg)))
    elif args.action == "status":
        _print_json(brain_status(cfg))
    return 0


def cmd_session(args) -> int:
    cfg = _cfg()
    from .handoff import render_handoff, session_end, session_start
    log = EventLog(cfg)
    if args.action == "start":
        sid = session_start(log, args.agent, args.model or "", args.purpose or "", tier=args.tier)
        print(f"session: {sid}")
        print(f"export AUTOEXP_SESSION={sid}   # pass --session {sid} to later commands if env does not persist")
        if not args.quiet:
            print()
            print(render_handoff(cfg, State.load(log), tier=args.tier), end="")
    else:
        sid = args.session or os.environ.get("AUTOEXP_SESSION")
        if not sid:
            print("need --session")
            return 2
        session_end(log, sid)
        print(f"ended {sid}")
    return 0


def cmd_baton(args) -> int:
    cfg = _cfg()
    from .handoff import baton_write, load_baton_file, reconstruct_baton, render_baton
    log = EventLog(cfg)
    state = State.load(log)
    sid = args.session or os.environ.get("AUTOEXP_SESSION")
    if args.action == "show":
        for b in state.batons[-args.last:]:
            print(render_baton(b, b["ts"]))
        return 0
    if not sid:
        print("need --session (or AUTOEXP_SESSION)")
        return 2
    if args.auto:
        fields = reconstruct_baton(state, sid)
    elif args.file:
        fields = load_baton_file(Path(args.file))
    else:
        fields = {k: v for k, v in {"goal": args.goal, "done": args.done, "in_flight": args.in_flight,
                                     "next": args.next, "open_questions": args.question,
                                     "unverified": args.unverified, "touched": args.touched}.items() if v}
        if not fields:
            print("nothing to write: give --goal/--done/--next/... or --file or --auto")
            return 2
    b = baton_write(cfg, log, sid, fields, reconstructed=bool(args.auto))
    print(render_baton(b, "now"))
    return 0


def cmd_task(args) -> int:
    cfg = _cfg()
    from .handoff import task_add, task_claim
    log = EventLog(cfg)
    state = State.load(log)
    holder = args.session or os.environ.get("AUTOEXP_SESSION") or default_actor()["user"]
    if args.action == "add":
        tid = task_add(log, args.text, tier=args.tier, accept=args.accept or "", kind=args.kind,
                       campaign=args.campaign)
        print(tid)
    elif args.action == "claim":
        task_claim(log, state, args.text, holder, hours=args.hours)
        print(f"{args.text} claimed by {holder}")
    elif args.action == "release":
        log.append("task.released", {"task": args.text, "holder": holder})
    elif args.action == "done":
        log.append("task.done", {"task": args.text, "holder": holder, "evidence": args.evidence or []})
        print(f"{args.text} done")
    else:
        for t in state.tasks.values():
            if t["status"] != "done" or args.all:
                print(f"{t['task']} [{t['status']}] ({t['tier']}/{t['kind']}) {t['text']}"
                      + (f" | holder {t['holder']}" if t.get("holder") else ""))
    return 0


def cmd_decide(args) -> int:
    cfg = _cfg()
    EventLog(cfg).append("decision.recorded", {"text": args.text, "why": args.why, "campaign": args.campaign})
    print("recorded")
    return 0


def cmd_note(args) -> int:
    cfg = _cfg()
    log = EventLog(cfg)
    if args.campaign and args.campaign not in State.load(log).campaigns:
        print(f"no campaign {args.campaign}")
        return 2
    log.append("note.added", {"text": args.text, "campaign": args.campaign})
    print("noted")
    return 0


def cmd_finding(args) -> int:
    cfg = _cfg()
    from .archive import add_finding, render_findings, update_finding
    log = EventLog(cfg)
    state = State.load(log)
    if args.action == "add":
        if not args.text:
            print("usage: autoexp finding add \"<claim>\" --campaign C [--stage S] [--evidence RUN_ID ...]")
            return 2
        fid, warnings = add_finding(log, state, args.text, campaign=args.campaign, stage=args.stage,
                                    evidence=args.evidence, status=args.status or "tentative",
                                    part=args.part, supersedes=args.supersedes)
        print(fid)
        for w in warnings:
            print(f"warning: {w}")
    elif args.action == "update":
        update_finding(log, state, args.text, args.status, args.why)
        print(f"{args.text} -> {args.status}")
    else:
        print(render_findings(state), end="")
    return 0


def cmd_archive(args) -> int:
    cfg = _cfg()
    from .archive import git_commit, git_init, update
    if args.git_init:
        print(f"state repository: {git_init(cfg)} (set state_git.enabled: true in config.yaml)")
    eng = _engine(cfg)
    specs = {n: eng.spec(n).data for n in eng.state.campaigns}
    res = update(cfg, EventLog(cfg), specs=specs, days=args.day, rebuild=args.rebuild)
    print(f"archive: {res['root']}")
    for w in res["written"]:
        print(f"  wrote {w}")
    if args.commit or args.git_init:
        print(json.dumps(git_commit(cfg, push=args.push or None)))
    return 0


def cmd_memory(args) -> int:
    cfg = _cfg()
    from .archive import archive_dir, update
    path = archive_dir(cfg) / "MEMORY.md"
    if args.refresh or not path.exists():
        eng = _engine(cfg)
        update(cfg, EventLog(cfg), specs={n: eng.spec(n).data for n in eng.state.campaigns})
    print(path.read_text(), end="")
    return 0


def cmd_hook(args) -> int:
    from .hooks import run_hook
    return run_hook(load_config(), args.event, args.vendor)


# ---------------------------------------------------------------- parser

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="autoexp", description="experiment orchestration on Slurm")
    p.add_argument("--version", action="version", version=f"autoexp {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("init", help="create the state directory and a config template")
    s.add_argument("--project", help="also install AGENTS.md / CLAUDE.md handover files into this project")
    s.set_defaults(fn=cmd_init)

    s = sub.add_parser("check", help="check the environment")
    s.set_defaults(fn=cmd_check)

    s = sub.add_parser("submit", help="create a campaign from a spec and start its first stage")
    s.add_argument("spec")
    s.add_argument("--name")
    s.add_argument("--dry-run", action="store_true")
    s.add_argument("--show", type=int, default=20, help="runs to list per stage in --dry-run")
    s.add_argument("--no-tick", action="store_true")
    s.set_defaults(fn=cmd_submit)

    s = sub.add_parser("tick", help="reconcile once: poll, classify, retry, advance, submit")
    s.set_defaults(fn=cmd_tick)

    s = sub.add_parser("status", help="campaigns and jobs")
    s.add_argument("campaign", nargs="?")
    s.add_argument("--all", action="store_true")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_status)

    s = sub.add_parser("runs", help="runs of a campaign")
    s.add_argument("campaign")
    s.add_argument("--stage")
    s.add_argument("--json", action="store_true")
    s.set_defaults(fn=cmd_runs)

    s = sub.add_parser("approve", help="approve a stage that awaits a human")
    s.add_argument("campaign")
    s.add_argument("stage")
    s.add_argument("--no-tick", action="store_true")
    s.set_defaults(fn=cmd_approve)

    s = sub.add_parser("cancel", help="cancel a campaign and its jobs")
    s.add_argument("campaign")
    s.add_argument("--reason", default="cancelled by user")
    s.set_defaults(fn=cmd_cancel)

    s = sub.add_parser("brief", help="deterministic report of the last hours")
    s.add_argument("--hours", type=float, default=24.0)
    s.add_argument("--write", action="store_true")
    s.set_defaults(fn=cmd_brief)

    s = sub.add_parser("handoff", help="print or write the generated HANDOFF.md")
    s.add_argument("--write", action="store_true")
    s.add_argument("--tier", choices=["any", "standard", "strong"])
    s.set_defaults(fn=cmd_handoff)

    s = sub.add_parser("log", help="print events")
    s.add_argument("--session")
    s.add_argument("--type")
    s.add_argument("--tail", type=int, default=50)
    s.set_defaults(fn=cmd_log)

    s = sub.add_parser("nodes", help="node health registry")
    s.add_argument("action", choices=["list", "flag", "clear"])
    s.add_argument("node", nargs="?")
    s.add_argument("--reason")
    s.add_argument("--days", type=float)
    s.set_defaults(fn=cmd_nodes)

    s = sub.add_parser("brain", help="the long-lived supervisor job")
    s.add_argument("action", choices=["run", "submit", "ensure", "status"])
    s.add_argument("--once", action="store_true")
    s.set_defaults(fn=cmd_brain)

    s = sub.add_parser("session", help="register or end an agent/human session")
    s.add_argument("action", choices=["start", "end"])
    s.add_argument("--agent", default="human")
    s.add_argument("--model")
    s.add_argument("--purpose")
    s.add_argument("--tier", choices=["any", "standard", "strong"])
    s.add_argument("--session")
    s.add_argument("--quiet", action="store_true")
    s.set_defaults(fn=cmd_session)

    s = sub.add_parser("baton", help="write or show handover batons")
    s.add_argument("action", choices=["write", "show"])
    s.add_argument("--session")
    s.add_argument("--file")
    s.add_argument("--auto", action="store_true", help="reconstruct from the session's recorded actions")
    s.add_argument("--goal")
    s.add_argument("--done", action="append")
    s.add_argument("--in-flight", action="append")
    s.add_argument("--next", action="append")
    s.add_argument("--question", action="append")
    s.add_argument("--unverified", action="append")
    s.add_argument("--touched", action="append")
    s.add_argument("--last", type=int, default=3)
    s.set_defaults(fn=cmd_baton)

    s = sub.add_parser("task", help="shared task board with leases")
    s.add_argument("action", choices=["add", "claim", "release", "done", "list"])
    s.add_argument("text", nargs="?", help="task text (add) or task id")
    s.add_argument("--tier", default="any", choices=["any", "standard", "strong"])
    s.add_argument("--kind", default="ops")
    s.add_argument("--accept")
    s.add_argument("--campaign")
    s.add_argument("--session")
    s.add_argument("--hours", type=float, default=4.0)
    s.add_argument("--evidence", action="append")
    s.add_argument("--all", action="store_true")
    s.set_defaults(fn=cmd_task)

    s = sub.add_parser("decide", help="record a decision and its reason")
    s.add_argument("text")
    s.add_argument("--why", required=True)
    s.add_argument("--campaign")
    s.set_defaults(fn=cmd_decide)

    s = sub.add_parser("note", help="add a free-text note to the record (optionally for a campaign)")
    s.add_argument("text")
    s.add_argument("--campaign")
    s.set_defaults(fn=cmd_note)

    s = sub.add_parser("finding", help="the findings ledger: claims with evidence and a status")
    s.add_argument("action", choices=["add", "update", "list"])
    s.add_argument("text", nargs="?", help="claim (add) or finding id (update)")
    s.add_argument("--campaign")
    s.add_argument("--stage", help="cite the frozen results of this stage")
    s.add_argument("--evidence", action="append", help="run id, finding id, job:<id>, commit:<sha>, file:<path>")
    s.add_argument("--status", choices=["tentative", "supported", "refuted", "superseded"])
    s.add_argument("--why")
    s.add_argument("--part")
    s.add_argument("--supersedes")
    s.set_defaults(fn=cmd_finding)

    s = sub.add_parser("archive", help="regenerate journals, notebooks, FINDINGS.md and MEMORY.md")
    s.add_argument("--day", action="append", help="YYYY-MM-DD (repeatable); default today and yesterday")
    s.add_argument("--rebuild", action="store_true", help="regenerate every day")
    s.add_argument("--git-init", action="store_true", help="make the state directory a git repository")
    s.add_argument("--commit", action="store_true", help="commit the state directory now")
    s.add_argument("--push", action="store_true", help="with --commit: also push to the first remote")
    s.set_defaults(fn=cmd_archive)

    s = sub.add_parser("memory", help="print MEMORY.md, the long-term index")
    s.add_argument("--refresh", action="store_true")
    s.set_defaults(fn=cmd_memory)

    s = sub.add_parser("hook", help="agent lifecycle hook (reads JSON on stdin)")
    s.add_argument("event", choices=["session-start", "pre-compact", "session-end"])
    s.add_argument("--vendor", default="agent")
    s.set_defaults(fn=cmd_hook)
    return p


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    try:
        code = args.fn(args)
    except (ValueError, KeyError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        code = 2
    sys.exit(code or 0)
