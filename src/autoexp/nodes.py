"""Node health registry, shared by a group.

``nodes.yaml`` (in the shared directory) maps a node name to what is known
about it. One infrastructure failure marks a node *suspect* for a day; a
second one within a day makes it *bad* for a week. Entries carry evidence and
expire on their own, so a list never silently goes stale. Flags raised by test
campaigns are scoped to that campaign and expire quickly, so a canary run can
never poison the group's list.
"""

from __future__ import annotations

import datetime as _dt
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from .util import atomic_write, default_actor, now, parse_ts, short_lock

SUSPECT_HOURS = 24
BAD_DAYS = 7
TEST_HOURS = 2
STRIKE_WINDOW_HOURS = 24


def _load(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"nodes": {}}
    with open(path) as fh:
        data = yaml.safe_load(fh) or {}
    data.setdefault("nodes", {})
    return data


def _save(path: Path, data: Dict[str, Any]) -> None:
    header = "# Node health registry, maintained by auto-experiment. Safe to edit by hand.\n"
    atomic_write(path, header + yaml.safe_dump(data, sort_keys=True))


def _key(node: str, scope: Optional[str]) -> str:
    return node if not scope else f"{node}@{scope}"


def active(entry: Dict[str, Any], at: Optional[_dt.datetime] = None) -> bool:
    at = at or _dt.datetime.now().astimezone()
    exp = parse_ts(entry.get("expires"))
    return exp is None or exp > at


def flag(path: Path, node: str, reason: str, evidence: List[str] = None, scope: Optional[str] = None,
         hours: Optional[float] = None) -> Dict[str, Any]:
    """Record an infrastructure failure on ``node``; returns the updated entry."""
    path = Path(path)
    tnow = _dt.datetime.now().astimezone()
    with short_lock(path.with_name(".nodes.lock")):
        data = _load(path)
        key = _key(node, scope)
        entry = data["nodes"].get(key) or {"node": node, "strikes": 0, "first_seen": now(), "evidence": []}
        last = parse_ts(entry.get("last_seen"))
        recent = last is not None and (tnow - last).total_seconds() < STRIKE_WINDOW_HOURS * 3600
        entry["strikes"] = int(entry.get("strikes", 0)) + 1 if recent or not last else 1
        entry["last_seen"] = now()
        entry["reason"] = reason
        entry["reporter"] = default_actor().get("user")
        if scope:
            entry["scope"] = scope
        entry["evidence"] = (list(entry.get("evidence") or []) + list(evidence or []))[-10:]
        if hours is not None:
            span = _dt.timedelta(hours=hours)
            entry["status"] = "bad"
        elif scope:
            span = _dt.timedelta(hours=TEST_HOURS)
            entry["status"] = "suspect"
        elif entry["strikes"] >= 2:
            span = _dt.timedelta(days=BAD_DAYS)
            entry["status"] = "bad"
        else:
            span = _dt.timedelta(hours=SUSPECT_HOURS)
            entry["status"] = "suspect"
        entry["expires"] = (tnow + span).isoformat(timespec="seconds")
        data["nodes"][key] = entry
        _save(path, data)
        return entry


def clear(path: Path, node: str) -> int:
    path = Path(path)
    with short_lock(path.with_name(".nodes.lock")):
        data = _load(path)
        keys = [k for k in data["nodes"] if k == node or k.startswith(node + "@")]
        for k in keys:
            del data["nodes"][k]
        _save(path, data)
    return len(keys)


def entries(path: Path) -> Dict[str, Dict[str, Any]]:
    return _load(Path(path))["nodes"]


def excludes(path: Path, campaign: Optional[str] = None, known: Optional[set] = None) -> List[str]:
    """Nodes to exclude for a job of ``campaign``. Names unknown to the scheduler
    are dropped: one stale name makes sbatch reject the whole job."""
    out = set()
    for key, entry in entries(path).items():
        if not active(entry):
            continue
        scope = entry.get("scope")
        if scope and scope != campaign:
            continue
        out.add(entry.get("node") or key.split("@")[0])
    if known is not None:
        out &= known
    return sorted(out)
