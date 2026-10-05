"""Deterministic morning brief: what ran, what broke, what the numbers say.

Every number here is computed by code from result files. A language model may
later add commentary, but it can only cite numbers that appear in these tables.
Statistics are robust by default (median, interquartile mean, bootstrap CI),
because a handful of RL seeds with one outlier can flip a ranking of means.
"""

from __future__ import annotations

import random
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import Config
from .events import State, class_family
from .handoff import brain_status
from .util import age_seconds, atomic_write, is_finite_number, now, read_json, today


def iqm(values: List[float]) -> float:
    v = sorted(values)
    k = int(len(v) * 0.25)
    core = v[k:len(v) - k] or v
    return sum(core) / len(core)


def bootstrap_ci(values: List[float], stat=statistics.mean, n: int = 2000, alpha: float = 0.05,
                 seed: int = 0) -> Tuple[float, float]:
    if len(values) < 2:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    boots = sorted(stat([rng.choice(values) for _ in values]) for _ in range(n))
    lo = boots[int(alpha / 2 * n)]
    hi = boots[min(n - 1, int((1 - alpha / 2) * n))]
    return lo, hi


def results_table(state: State, spec_data: Dict[str, Any], campaign: str, stage: str) -> Optional[Dict[str, Any]]:
    analysis = spec_data.get("analysis") or {}
    metric = analysis.get("primary")
    rfile = analysis.get("result_file")
    if not metric or not rfile:
        return None
    groups: Dict[str, Dict[str, Any]] = {}
    for run in state.runs_of(campaign, stage):
        if run["status"] != "succeeded":
            continue
        data = read_json(Path(run["run_dir"]) / rfile) or {}
        val = data.get(metric)
        g = groups.setdefault(run["label"], {"label": run["label"], "values": [], "seeds": [], "runs": []})
        if is_finite_number(val):
            g["values"].append(float(val))
            g["seeds"].append(run["seed"])
            g["runs"].append(run["run_id"])
    rows = []
    for g in groups.values():
        vals = g["values"]
        if not vals:
            continue
        lo, hi = bootstrap_ci(vals)
        rows.append({"label": g["label"], "n": len(vals), "median": statistics.median(vals), "iqm": iqm(vals),
                     "mean": statistics.mean(vals), "ci_lo": lo, "ci_hi": hi, "min": min(vals), "max": max(vals),
                     "runs": g["runs"], "values": [round(v, 6) for v in vals]})
    higher = analysis.get("higher_is_better", True)
    rows.sort(key=lambda r: r["iqm"], reverse=bool(higher))
    return {"metric": metric, "higher_is_better": higher, "rows": rows}


def _f(x: float) -> str:
    return "-" if x != x else f"{x:.2f}"


def table_md(tab: Dict[str, Any]) -> List[str]:
    """Markdown lines for a results table (as produced by results_table)."""
    arrow = "higher is better" if tab.get("higher_is_better", True) else "lower is better"
    out = [f"`{tab['metric']}` ({arrow}; CI = 95% bootstrap of the mean)", "",
           "| arm | n | median | IQM | mean | 95% CI | min | max |", "|---|---|---|---|---|---|---|---|"]
    for r in tab["rows"]:
        out.append(f"| {r['label']} | {r['n']} | {_f(r['median'])} | {_f(r['iqm'])} | {_f(r['mean'])} | "
                   f"[{_f(r['ci_lo'])}, {_f(r['ci_hi'])}] | {_f(r['min'])} | {_f(r['max'])} |")
    if any(r["n"] < 3 for r in tab["rows"]):
        out += ["", "_Fewer than 3 seeds in some arms: treat differences as anecdotal._"]
    return out


def render_brief(cfg: Config, state: State, specs: Dict[str, Dict[str, Any]], hours: float = 24.0) -> str:
    L: List[str] = []
    add = L.append
    add(f"# Brief {today()} (last {hours:g} h, generated {now()})")
    add("")
    b = brain_status(cfg)
    add(f"Brain: {'alive' if b.get('alive') else 'NOT RUNNING'}"
        + (f" (job {b.get('job')}, last tick {state.brain.get('last_tick', '?')})" if b.get("alive") else ""))
    add("")

    window = hours * 3600
    fin = [a for a in state.attempts.values() if a["finished"] and (age_seconds(a.get("finished_ts")) or 1e12) < window]
    by_class: Dict[str, int] = {}
    for a in fin:
        by_class[a["classification"]] = by_class.get(a["classification"], 0) + 1
    add("## Overview")
    add(f"- attempts finished: {len(fin)} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(by_class.items())) or 'none'})")
    add(f"- jobs in flight: {len(state.active_attempts())}")
    infra = sum(v for k, v in by_class.items() if class_family(k) == "infra")
    if infra:
        add(f"- infrastructure problems handled automatically: {infra} (retried or rescheduled)")
    add("")

    for name, c in sorted(state.campaigns.items(), key=lambda kv: kv[1]["created"]):
        if c["status"] in ("cancelled",) and (age_seconds(c.get("status_ts")) or 1e12) > window:
            continue
        add(f"## {name} [{c['status']}]")
        if c.get("hypothesis"):
            add(f"Hypothesis: {c['hypothesis']}")
        add("Stages: " + " -> ".join(f"{s['name']}:{s['status']}" for s in c["stages"]))
        if c["status"] == "blocked":
            add(f"**Blocked:** {c.get('status_reason')}")
        for s in c["stages"]:
            if s["status"] == "awaiting_approval":
                add(f"**Needs your approval:** `autoexp approve {name} {s['name']}`")
            if s["status"] in ("succeeded", "running", "failed"):
                tab = results_table(state, specs.get(name) or {}, name, s["name"])
                if tab and tab["rows"]:
                    add(f"\nResults, stage `{s['name']}`:")
                    for line in table_md(tab):
                        add(line)
        fails = [a for a in fin if state.runs[a["run_id"]]["campaign"] == name and a["classification"] != "ok"]
        if fails:
            add("\nFailures in the window:")
            for a in sorted(fails, key=lambda a: a["finished_ts"]):
                add(f"- {a['classification']}: {a['run_id']} (job {a['job_id']}, {a.get('node') or '?'}) {a.get('detail')}")
        add("")

    flagged = [f for f in state.node_flags if (age_seconds(f.get("ts")) or 1e12) < window]
    if flagged:
        add("## Nodes flagged")
        for f in flagged:
            add(f"- {f['node']} -> {f['status']} ({f.get('reason')}), until {f.get('expires', '')[:16]}"
                + (f" [scope {f['scope']}]" if f.get("scope") else ""))
        add("")
    if state.noops:
        add("## No-op parameters detected")
        for n in state.noops[-10:]:
            add(f"- {n['campaign']}/{n['stage']}: `{n['param']}` asked {n['asked']} but echoed {n['echoed']}")
        add("")
    open_t = [t for t in state.tasks.values() if t["status"] != "done"]
    if open_t:
        add("## Open tasks")
        for t in open_t:
            add(f"- [{t['task']}] ({t['tier']}) {t['text']}")
    return "\n".join(L).rstrip() + "\n"


def write_brief(cfg: Config, text: str) -> Path:
    path = cfg.briefs_dir / f"{today()}.md"
    atomic_write(path, text)
    return path
