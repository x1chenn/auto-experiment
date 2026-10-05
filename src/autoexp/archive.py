"""Memory consolidation: structured archives derived from the event log.

Long sessions forget; people forget; notes drift. The event log forgets
nothing, but it is too raw to read. This module turns it into layered,
bounded documents that every session can rely on:

    archive/
      MEMORY.md                  long-term index, <= 150 lines: what we know
      FINDINGS.md                numbered findings with evidence and status history
      notebooks/<campaign>.md    one living notebook per campaign: intent, timeline,
                                 frozen results per stage, failures, findings, notes
      journal/YYYY-MM-DD.md      what happened that day (progress, results, decisions,
                                 sessions, infrastructure, notes, what is still open)
      journal/weekly/YYYY-Www.md weekly rollup of the daily journals

Everything is regenerated from events, deterministically: a day that did not
change produces a byte-identical journal, so the archive can live in a
(private) git repository and its history stays meaningful. Knowledge enters
only through events: ``autoexp finding`` (claims with evidence), ``autoexp
note`` (free text), decisions, batons and frozen stage results.
"""

from __future__ import annotations

import datetime as _dt
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .brief import results_table, table_md
from .config import Config
from .events import EventLog, State, class_family
from .util import atomic_write, now

MEMORY_MAX_LINES = 150
FINDING_STATUSES = ("tentative", "supported", "refuted", "superseded")


# ---------------------------------------------------------------- helpers

def archive_dir(cfg: Config) -> Path:
    custom = cfg.get("archive_dir")
    return Path(custom).expanduser() if custom else cfg.home / "archive"


def campaign_of(ev: Dict[str, Any]) -> Optional[str]:
    d = ev.get("data") or {}
    if d.get("campaign"):
        return d["campaign"]
    if ev["type"] == "campaign.created":
        return d.get("name")
    for key in ("run_id", "attempt_id"):
        if d.get(key):
            return str(d[key]).split("/", 1)[0]
    if d.get("scope"):
        return d["scope"]
    return None


def day_of(ev: Dict[str, Any]) -> str:
    return ev["ts"][:10]


def week_of(day: str) -> str:
    y, w, _ = _dt.date.fromisoformat(day).isocalendar()
    return f"{y}-W{w:02d}"


def _write_if_changed(path: Path, text: str) -> bool:
    try:
        if path.read_text() == text:
            return False
    except OSError:
        pass
    atomic_write(path, text)
    return True


def _who(actor: Optional[Dict[str, Any]]) -> str:
    a = actor or {}
    if a.get("agent"):
        return f"{a['agent']}/{a.get('model') or '?'}"
    return a.get("user") or "?"


def _items(v: Any) -> List[str]:
    if not v:
        return []
    if isinstance(v, list):
        return [x if isinstance(x, str) else json.dumps(x, sort_keys=True) for x in v]
    return [str(v)]


def headline(table: Optional[Dict[str, Any]]) -> str:
    if not table or not table.get("rows"):
        return "no results"
    best = table["rows"][0]
    return f"best {best['label']}: IQM {best['iqm']:.2f} (n={best['n']}) on {table['metric']}"


# ---------------------------------------------------------------- findings

def finding_check_numbers(claim: str, tables: Iterable[Dict[str, Any]]) -> List[str]:
    """Numbers in a claim that appear in none of the cited tables (2-decimal match).

    Integers below 10 (counts, stage numbers) are ignored.
    """
    values = set()
    for t in tables:
        for r in (t or {}).get("rows", []):
            for k in ("median", "iqm", "mean", "ci_lo", "ci_hi", "min", "max"):
                v = r.get(k)
                if isinstance(v, (int, float)) and v == v:
                    values.add(f"{v:.2f}")
                    values.add(f"{v:.1f}")
            values.add(str(r.get("n")))
            for num in re.findall(r"-?\d+(?:\.\d+)?(?:e-?\d+)?", str(r.get("label", ""))):
                values.add(num)  # parameter values in arm labels (lr=0.1) are legitimate
                try:
                    values.add(f"{float(num):.2f}")
                except ValueError:
                    pass
    missing = []
    for num in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?", claim):
        if "." not in num and abs(int(num)) < 10:
            continue
        x = float(num)
        if f"{x:.2f}" not in values and f"{x:.1f}" not in values and num not in values:
            missing.append(num)
    return missing


def add_finding(log: EventLog, state: State, claim: str, campaign: Optional[str] = None,
                stage: Optional[str] = None, evidence: Optional[List[str]] = None,
                status: str = "tentative", part: Optional[str] = None,
                supersedes: Optional[str] = None) -> Tuple[str, List[str]]:
    """Record a finding. Returns (id, warnings). Evidence must resolve."""
    if status not in FINDING_STATUSES[:2]:
        raise ValueError("a new finding is 'tentative' or 'supported'")
    evidence = list(evidence or [])
    warnings: List[str] = []
    tables = []
    if campaign:
        c = state.campaigns.get(campaign)
        if c is None:
            raise ValueError(f"no campaign {campaign}")
        part = part or c.get("part")
        if stage:
            snap = c["results"].get(stage)
            if snap is None:
                raise ValueError(f"stage {campaign}/{stage} has no frozen results yet; cite run ids instead")
            tables.append(snap.get("table"))
            evidence.append(f"results:{campaign}/{stage}@{snap.get('seq')}")
    for ev in evidence:
        if ev.startswith("results:"):
            continue
        if ev in state.runs:
            continue
        if ev.startswith("F") and ev in state.findings:
            continue
        if ev.startswith(("job:", "commit:", "file:", "url:")):
            continue
        raise ValueError(f"evidence {ev!r} is not a run id, finding id, results reference, job:, commit:, file: or url:")
    if not evidence:
        raise ValueError("a finding needs evidence (--campaign/--stage, run ids, job:, commit:, file:)")
    if supersedes and supersedes not in state.findings:
        raise ValueError(f"no finding {supersedes}")
    if tables:
        stray = finding_check_numbers(claim, tables)
        if stray:
            warnings.append(f"numbers {stray} do not appear in the cited results table; check them")
    ev = log.append("finding.added", {"claim": claim, "campaign": campaign, "stage": stage,
                                      "evidence": evidence, "status": status, "part": part,
                                      "supersedes": supersedes, "warnings": warnings})
    fid = f"F{ev['seq']}"
    if supersedes:
        log.append("finding.updated", {"id": supersedes, "status": "superseded", "superseded_by": fid,
                                       "why": f"superseded by {fid}"})
    return fid, warnings


def update_finding(log: EventLog, state: State, fid: str, status: str, why: str) -> None:
    if fid not in state.findings:
        raise ValueError(f"no finding {fid}")
    if status not in FINDING_STATUSES:
        raise ValueError(f"status must be one of {FINDING_STATUSES}")
    if not why:
        raise ValueError("say why (--why)")
    log.append("finding.updated", {"id": fid, "status": status, "why": why})


# ---------------------------------------------------------------- renderers

def render_daily(day: str, evs: List[Dict[str, Any]], state: State) -> str:
    L: List[str] = []
    add = L.append
    last_seq = max((e.get("seq", 0) for e in evs), default=0)
    add(f"# Journal {day}")
    add(f"_generated from events up to #{last_seq}; edit nothing here: add `autoexp note`/`autoexp finding`_")
    add("")
    by_type = Counter(e["type"] for e in evs)
    fin = [e["data"] for e in evs if e["type"] == "attempt.finished"]
    classes = Counter(d["classification"] for d in fin)
    retries = sum(1 for e in evs if e["type"] == "attempt.submitted" and e["data"].get("n", 1) > 1)
    add("## Summary")
    add(f"- jobs submitted: {by_type['attempt.submitted']} (of which retries: {retries}); "
        f"attempts finished: {len(fin)}" + (f" ({', '.join(f'{k}={v}' for k, v in sorted(classes.items()))})" if fin else ""))
    created = [e["data"]["name"] for e in evs if e["type"] == "campaign.created"]
    if created:
        add(f"- campaigns created: {', '.join(created)}")
    done_st = [(e["data"]["campaign"], e["data"]["stage"], e["data"]["status"]) for e in evs
               if e["type"] == "stage.status" and e["data"]["status"] in ("succeeded", "failed")]
    if done_st:
        add(f"- stages finished: {', '.join(f'{c}/{s}={st}' for c, s, st in done_st)}")
    nf = [e for e in evs if e["type"] == "finding.added"]
    if nf:
        add(f"- findings recorded: {len(nf)}")
    add("")

    per: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in evs:
        c = campaign_of(e)
        if c and c in state.campaigns:
            per[c].append(e)
    if per:
        add("## Campaigns")
    for name in sorted(per):
        cevs = per[name]
        c = state.campaigns[name]
        add(f"### {name}" + (f" (part: {c['part']})" if c.get("part") else ""))
        cf = Counter(e["data"]["classification"] for e in cevs if e["type"] == "attempt.finished")
        if cf:
            add(f"- attempts finished: {', '.join(f'{k}={v}' for k, v in sorted(cf.items()))}")
        for e in cevs:
            d, t = e["data"], e["ts"][11:16]
            if e["type"] == "stage.status" and d["status"] in ("running", "succeeded", "failed", "awaiting_approval"):
                add(f"- {t} stage `{d['stage']}` -> {d['status']} ({d.get('reason')})")
            elif e["type"] == "campaign.status":
                add(f"- {t} campaign -> {d['status']} ({d.get('reason')})")
            elif e["type"] == "noop.detected":
                add(f"- {t} NO-OP parameter `{d['param']}` in {d['stage']}: asked {d['asked']}, echoed {d['echoed']}")
            elif e["type"] == "stage.results" and (d.get("table") or {}).get("rows"):
                add(f"- {t} results frozen for `{d['stage']}` ({headline(d['table'])}):")
                add("")
                L.extend(table_md(d["table"]))
                add("")
        failed = [e["data"] for e in cevs if e["type"] == "run.finished" and e["data"]["status"] == "failed"]
        for d in failed:
            add(f"- run failed: {d['run_id']} ({d.get('classification')})")
        add("")

    sections = [
        ("Findings", [e for e in evs if e["type"] == "finding.added"],
         lambda e: f"- {state_finding_id(e, state)} [{e['data'].get('status')}] {e['data']['claim']} "
                   f"(evidence: {', '.join(e['data'].get('evidence') or [])})"),
        ("Finding status changes", [e for e in evs if e["type"] == "finding.updated" and e["data"].get("status")],
         lambda e: f"- {e['ts'][11:16]} {e['data']['id']} -> {e['data']['status']} ({e['data'].get('why')})"),
        ("Decisions", [e for e in evs if e["type"] == "decision.recorded"],
         lambda e: f"- {e['ts'][11:16]} {e['data']['text']} (why: {e['data'].get('why')}; by {_who(e.get('actor'))})"),
        ("Notes", [e for e in evs if e["type"] == "note.added"],
         lambda e: f"- {e['ts'][11:16]} " + (f"[{e['data']['campaign']}] " if e['data'].get('campaign') else "")
                   + f"{e['data']['text']} ({_who(e.get('actor'))})"),
        ("Infrastructure", [e for e in evs if e["type"] in ("node.flagged", "brain.started", "brain.requeue", "brain.error")],
         lambda e: f"- {e['ts'][11:16]} {e['type']}: " + json.dumps(
             {k: v for k, v in e["data"].items() if k in ("node", "status", "reason", "scope", "job", "error")},
             sort_keys=True)),
        ("Tasks", [e for e in evs if e["type"] in ("task.added", "task.done")],
         lambda e: f"- {e['ts'][11:16]} {e['type'].split('.')[1]} {e['data']['task']}"
                   + (f": {e['data'].get('text')}" if e["type"] == "task.added" else "")),
    ]
    for title, items, fmt in sections:
        if items:
            add(f"## {title}")
            for e in items:
                add(fmt(e))
            add("")

    sess = [e for e in evs if e["type"] in ("session.started", "baton.written")]
    if sess:
        add("## Sessions and batons")
        for e in sess:
            d = e["data"]
            if e["type"] == "session.started":
                add(f"- {e['ts'][11:16]} session {d['session']} started ({d.get('agent')}/{d.get('model') or '?'})"
                    + (f": {d['purpose']}" if d.get("purpose") else ""))
            else:
                tag = " [reconstructed]" if d.get("reconstructed") else ""
                add(f"- {e['ts'][11:16]} baton {d.get('session')}{tag}: {d.get('goal') or ''}")
                for key in ("done", "next", "open_questions", "unverified"):
                    for item in _items(d.get(key))[:6]:
                        add(f"  - {key}: {item}")
        add("")
    return "\n".join(L).rstrip() + "\n"


def state_finding_id(ev: Dict[str, Any], state: State) -> str:
    return f"F{ev.get('seq')}"


def render_open_items(state: State) -> List[str]:
    out = []
    for c in state.campaigns.values():
        for s in c["stages"]:
            if s["status"] == "awaiting_approval":
                out.append(f"- approval needed: `autoexp approve {c['name']} {s['name']}`")
        if c["status"] == "blocked":
            out.append(f"- blocked: {c['name']} ({c.get('status_reason')})")
    for t in state.tasks.values():
        if t["status"] != "done":
            out.append(f"- task {t['task']} ({t['tier']}): {t['text']}")
    for f in state.findings.values():
        if f.get("status") == "tentative":
            out.append(f"- finding {f['id']} is tentative: verify or refute it")
    return out


def render_notebook(name: str, cevs: List[Dict[str, Any]], state: State, spec_data: Dict[str, Any]) -> str:
    c = state.campaigns[name]
    L: List[str] = []
    add = L.append
    add(f"# Campaign `{name}`" + (f" — part: {c['part']}" if c.get("part") else ""))
    add(f"_living notebook generated from events up to #{max((e.get('seq', 0) for e in cevs), default=0)}_")
    add("")
    add(f"- status: **{c['status']}**" + (f" ({c.get('status_reason')})" if c.get("status_reason") else ""))
    if c.get("hypothesis"):
        add(f"- hypothesis: {c['hypothesis']}")
    if c.get("success"):
        add(f"- success criterion: {c['success']}")
    prov = c.get("provenance") or {}
    add(f"- created: {c['created'][:16]} by {_who(c.get('actor'))}; spec hash {c.get('spec_hash')}"
        + (f"; code {prov.get('commit', '')[:10]}{' (dirty)' if prov.get('dirty') else ''}" if prov else ""))
    add("")
    add("## Stages")
    add("| stage | status | note | since |")
    add("|---|---|---|---|")
    for s in c["stages"]:
        add(f"| {s['name']} | {s['status']} | {s.get('reason') or ''} | {s['ts'][:16]} |")
    add("")

    add("## Results by stage")
    any_res = False
    for s in c["stages"]:
        snap = c["results"].get(s["name"])
        if snap and (snap.get("table") or {}).get("rows"):
            any_res = True
            add(f"### {s['name']} (frozen {snap['ts'][:16]}, event #{snap.get('seq')}; run outcomes: "
                f"{', '.join(f'{k}={v}' for k, v in sorted((snap.get('classes') or {}).items()))})")
            add("")
            L.extend(table_md(snap["table"]))
            add("")
            for r in snap["table"]["rows"]:
                add(f"- {r['label']}: runs {', '.join(x.split('/')[-1] for x in r.get('runs', []))}")
            add("")
        elif s["status"] in ("running", "succeeded", "failed") and spec_data:
            live = results_table(state, spec_data, name, s["name"])
            if live and live["rows"]:
                any_res = True
                add(f"### {s['name']} (live, not final)")
                add("")
                L.extend(table_md(live))
                add("")
    if not any_res:
        add("(no results yet)")
        add("")

    finds = [f for f in state.findings.values() if f.get("campaign") == name]
    add("## Findings")
    if not finds:
        add("(none recorded; add one with `autoexp finding add`)")
    for f in sorted(finds, key=lambda f: f["ts"]):
        add(f"- **{f['id']}** [{f.get('status')}] {f['claim']}")
    add("")

    add("## Failures")
    fails = Counter()
    examples: Dict[str, str] = {}
    for e in cevs:
        if e["type"] == "attempt.finished" and e["data"]["classification"] != "ok":
            k = e["data"]["classification"]
            fails[k] += 1
            examples.setdefault(k, f"{e['data']['run_id'].split('/', 1)[1]}: {e['data'].get('detail')}")
    if not fails:
        add("(none)")
    for k, n in fails.most_common():
        add(f"- {k} x{n}" + (" (handled automatically)" if class_family(k) == "infra" else "") + f" — e.g. {examples[k]}")
    add("")

    add("## Timeline")
    for e in cevs:
        d, t = e["data"], e["ts"][:16]
        if e["type"] == "stage.status":
            add(f"- {t} `{d['stage']}` -> {d['status']} ({d.get('reason')})")
        elif e["type"] == "campaign.status":
            add(f"- {t} campaign -> {d['status']} ({d.get('reason')})")
        elif e["type"] == "decision.recorded":
            add(f"- {t} decision: {d['text']} (why: {d.get('why')})")
        elif e["type"] == "note.added":
            add(f"- {t} note: {d['text']} ({_who(e.get('actor'))})")
        elif e["type"] == "noop.detected":
            add(f"- {t} no-op parameter `{d['param']}` in {d['stage']}")
    add("")
    return "\n".join(L).rstrip() + "\n"


def render_findings(state: State) -> str:
    L = ["# Findings ledger", "_append-only; a finding is never edited, only re-rated or superseded_", ""]
    for status in ("supported", "tentative", "refuted", "superseded"):
        fs = [f for f in state.findings.values() if f.get("status") == status]
        if not fs:
            continue
        L.append(f"## {status} ({len(fs)})")
        for f in sorted(fs, key=lambda f: int(f["id"][1:]), reverse=True):
            scope = "/".join(x for x in (f.get("part"), f.get("campaign"), f.get("stage")) if x)
            L.append(f"- **{f['id']}** {f['claim']}" + (f" [{scope}]" if scope else ""))
            L.append(f"  - evidence: {', '.join(f.get('evidence') or [])}; recorded {f['ts'][:16]} by {_who(f.get('actor'))}")
            for h in f.get("history") or []:
                L.append(f"  - {h['ts'][:16]} {h['from']} -> {h['to']}: {h.get('why')}")
            if f.get("superseded_by"):
                L.append(f"  - superseded by {f['superseded_by']}")
            for w in f.get("warnings") or []:
                L.append(f"  - warning at entry: {w}")
        L.append("")
    if len(L) == 3:
        L.append("(no findings yet)")
    return "\n".join(L).rstrip() + "\n"


def render_weekly(week: str, days: Dict[str, List[Dict[str, Any]]], state: State) -> str:
    L = [f"# Week {week}", ""]
    for day in sorted(days):
        evs = days[day]
        fin = Counter(e["data"]["classification"] for e in evs if e["type"] == "attempt.finished")
        stages = [f"{e['data']['campaign']}/{e['data']['stage']}={e['data']['status']}" for e in evs
                  if e["type"] == "stage.status" and e["data"]["status"] in ("succeeded", "failed")]
        L.append(f"- **{day}**: {sum(fin.values())} attempts finished"
                 + (f" ({', '.join(f'{k}={v}' for k, v in sorted(fin.items()))})" if fin else "")
                 + (f"; stages: {', '.join(stages)}" if stages else "") + f" → [journal](../{day}.md)")
    added = [e for d in days.values() for e in d if e["type"] == "finding.added"]
    if added:
        L += ["", "## Findings recorded this week"]
        for e in added:
            L.append(f"- F{e['seq']} {e['data']['claim']}")
    dec = [e for d in days.values() for e in d if e["type"] == "decision.recorded"]
    if dec:
        L += ["", "## Decisions this week"]
        for e in dec:
            L.append(f"- {e['ts'][:10]} {e['data']['text']} (why: {e['data'].get('why')})")
    return "\n".join(L).rstrip() + "\n"


def render_memory(state: State, days: List[str], weeks: List[str], journal_lines: Dict[str, str]) -> str:
    L: List[str] = []
    add = L.append
    add("# MEMORY — long-term index")
    add(f"_generated {now()} from events up to #{state.seq}. Read order for a new or compacted session:_")
    add("_this file → `autoexp handoff` (what is happening now) → the notebook of your campaign._")
    add("")
    sup = sorted((f for f in state.findings.values() if f.get("status") == "supported"),
                 key=lambda f: int(f["id"][1:]), reverse=True)
    add("## Established findings (supported)")
    for f in sup[:30]:
        scope = "/".join(x for x in (f.get("campaign"), f.get("stage")) if x)
        add(f"- {f['id']} {f['claim']}" + (f" [{scope}]" if scope else "") + f" ({f['updated'][:10]})")
    if len(sup) > 30:
        add(f"- ... {len(sup) - 30} older supported findings in FINDINGS.md")
    if not sup:
        add("(none yet)")
    tent = [f for f in state.findings.values() if f.get("status") == "tentative"]
    if tent:
        add("")
        add("## Tentative findings (verify before relying on them)")
        for f in sorted(tent, key=lambda f: int(f["id"][1:]), reverse=True)[:15]:
            add(f"- {f['id']} {f['claim']}")
    add("")
    add("## Campaigns")
    by_part: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for c in state.campaigns.values():
        by_part[c.get("part") or "(no part)"].append(c)
    named = any(p != "(no part)" for p in by_part)
    for part in sorted(by_part):
        if named:
            add(f"### {part}")
        for c in sorted(by_part[part], key=lambda c: c["created"], reverse=True)[:20]:
            stages = " -> ".join(f"{s['name']}:{s['status']}" for s in c["stages"])
            latest = None
            for s in reversed(c["stages"]):
                if s["name"] in c["results"]:
                    latest = (s["name"], c["results"][s["name"]].get("table"))
                    break
            res = f" | {latest[0]}: {headline(latest[1])}" if latest else ""
            add(f"- `{c['name']}` [{c['status']}] {stages}{res} → notebooks/{c['name']}.md")
    add("")
    add("## Recent decisions")
    for d in state.decisions[-10:]:
        add(f"- {d['ts'][:10]} {d.get('text')} (why: {d.get('why')})")
    if not state.decisions:
        add("(none)")
    add("")
    add("## Open now")
    opens = render_open_items(state)
    L.extend(opens[:15] or ["(nothing open)"])
    add("")
    add("## Journals")
    for day in sorted(days, reverse=True)[:10]:
        add(f"- {day}: {journal_lines.get(day, '')} → journal/{day}.md")
    if weeks:
        add("- weekly: " + ", ".join(f"journal/weekly/{w}.md" for w in sorted(weeks, reverse=True)[:6]))
    if len(L) > MEMORY_MAX_LINES:
        L = L[:MEMORY_MAX_LINES - 1] + ["... (truncated; see FINDINGS.md and notebooks/)"]
    return "\n".join(L).rstrip() + "\n"


# ---------------------------------------------------------------- update

def update(cfg: Config, log: EventLog, specs: Optional[Dict[str, Dict[str, Any]]] = None,
           days: Optional[List[str]] = None, rebuild: bool = False) -> Dict[str, Any]:
    """Regenerate the archive. By default only today's and yesterday's journals,
    their weeks, all notebooks, FINDINGS.md and MEMORY.md."""
    evs = list(log.iter())
    state = State()
    for e in evs:
        state.apply(e)
    root = archive_dir(cfg)
    (root / "journal" / "weekly").mkdir(parents=True, exist_ok=True)
    (root / "notebooks").mkdir(parents=True, exist_ok=True)

    by_day: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    by_campaign: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for e in evs:
        by_day[day_of(e)].append(e)
        c = campaign_of(e)
        if c:
            by_campaign[c].append(e)
    all_days = sorted(by_day)
    if rebuild:
        targets = all_days
    elif days:
        targets = [d for d in days if d in by_day]
    else:
        today_s = _dt.date.today().isoformat()
        yesterday = (_dt.date.today() - _dt.timedelta(days=1)).isoformat()
        targets = [d for d in (yesterday, today_s) if d in by_day]

    written = []
    journal_lines = {}
    for day in all_days:
        evs_d = by_day[day]
        fin = Counter(e["data"]["classification"] for e in evs_d if e["type"] == "attempt.finished")
        journal_lines[day] = (f"{sum(fin.values())} attempts finished, "
                              f"{sum(1 for e in evs_d if e['type'] == 'finding.added')} findings, "
                              f"{sum(1 for e in evs_d if e['type'] == 'decision.recorded')} decisions")
    for day in targets:
        if _write_if_changed(root / "journal" / f"{day}.md", render_daily(day, by_day[day], state)):
            written.append(f"journal/{day}.md")
    weeks = sorted({week_of(d) for d in all_days})
    for week in sorted({week_of(d) for d in targets}):
        days_w = {d: by_day[d] for d in all_days if week_of(d) == week}
        if _write_if_changed(root / "journal" / "weekly" / f"{week}.md", render_weekly(week, days_w, state)):
            written.append(f"journal/weekly/{week}.md")
    for name in state.campaigns:
        text = render_notebook(name, by_campaign.get(name, []), state, (specs or {}).get(name) or {})
        if _write_if_changed(root / "notebooks" / f"{name}.md", text):
            written.append(f"notebooks/{name}.md")
    if _write_if_changed(root / "FINDINGS.md", render_findings(state)):
        written.append("FINDINGS.md")
    atomic_write(root / "MEMORY.md", render_memory(state, all_days, weeks, journal_lines))
    written.append("MEMORY.md")
    return {"root": str(root), "written": written, "state": state}


# ---------------------------------------------------------------- versioning the state

STATE_GITIGNORE = """\
# auto-experiment state repository: secrets, locks and volatile files stay out
secrets/
codex_home/
logs/
local_backend/
*.lock
brain.lease
brain.sbatch
*.tmp
.*.tmp
"""


def _git(root: Path, *args: str, timeout: int = 120):
    import subprocess
    return subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True, timeout=timeout)


def git_init(cfg: Config) -> str:
    """Make the state directory a git repository (events, archive, specs, batons)."""
    root = cfg.home
    gi = root / ".gitignore"
    if not gi.exists():
        atomic_write(gi, STATE_GITIGNORE)
    if not (root / ".git").exists():
        proc = _git(root, "init", "-q")
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.strip())
        _git(root, "checkout", "-q", "-b", "main")
    return str(root)


def git_commit(cfg: Config, message: Optional[str] = None, push: Optional[bool] = None) -> Dict[str, Any]:
    """Commit the state directory if it is a git repository and something changed.
    Pushing is optional and never raises: a network hiccup must not stop the brain."""
    root = cfg.home
    if not (root / ".git").exists():
        return {"committed": False, "reason": "not a git repository (run `autoexp archive --git-init`)"}
    _git(root, "add", "-A")
    if _git(root, "diff", "--cached", "--quiet").returncode == 0:
        return {"committed": False, "reason": "no changes"}
    seq = (cfg.events_dir / ".seq").read_text().strip() if (cfg.events_dir / ".seq").exists() else "?"
    msg = message or f"autoexp state: events up to #{seq} ({now()})"
    proc = _git(root, "commit", "-q", "-m", msg)
    if proc.returncode != 0:
        return {"committed": False, "reason": proc.stderr.strip()[:300]}
    out: Dict[str, Any] = {"committed": True, "message": msg}
    settings = cfg.get("state_git") or {}
    if push if push is not None else settings.get("push"):
        remotes = _git(root, "remote").stdout.split()
        if remotes:
            pr = _git(root, "push", "-q", remotes[0], "HEAD", timeout=120)
            out["pushed"] = pr.returncode == 0
            if pr.returncode != 0:
                out["push_error"] = pr.stderr.strip()[:300]
    return out


def git_due(cfg: Config) -> bool:
    settings = cfg.get("state_git") or {}
    if not settings.get("enabled") or not (cfg.home / ".git").exists():
        return False
    last = _git(cfg.home, "log", "-1", "--format=%ct").stdout.strip()
    import time
    return not last or time.time() - int(last) >= 60 * float(settings.get("every_minutes", 60))
