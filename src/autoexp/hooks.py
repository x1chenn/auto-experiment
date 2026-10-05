"""Agent lifecycle hooks shared by Claude Code and Codex CLI.

Both tools call a command on SessionStart / PreCompact / SessionEnd with a JSON
payload on stdin (``session_id``, sometimes ``model``). One implementation
serves both:

* session-start: register the session, export AUTOEXP_SESSION for later shell
  commands when the tool supports it (Claude Code's CLAUDE_ENV_FILE), and inject
  a digest of HANDOFF.md as additional context.
* pre-compact / session-end: if the session has not written a baton, write a
  reconstructed one so the next agent is never left without a handover.

A hook must never break the agent's session: every error is swallowed.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Optional

from .config import Config
from .events import EventLog, State
from .handoff import guess_tier, render_handoff, session_end, session_start

PROTOCOL_REMINDER = """\
[auto-experiment] You are session {sid}. Protocol: (1) MEMORY below is what we know, HANDOFF is
what is happening now; (2) claim work with `autoexp task claim <id> --session {sid}` before acting;
(3) submit experiments only through `autoexp` (never raw sbatch); (4) record knowledge as you go:
`autoexp finding add` (claims with evidence) and `autoexp note` (anything else); (5) before you stop,
write a baton: `autoexp baton write --session {sid} --goal ... --done ... --next ...`.
"""

COMPACTED = """\
[auto-experiment] Your context was just compacted. Details you half-remember may be wrong: trust
MEMORY, HANDOFF and `autoexp runs/log` over your recollection, and re-check numbers before using them.
"""

MEMORY_DIGEST_LINES = 80


def _memory_digest(cfg: Config) -> str:
    from .archive import archive_dir
    path = archive_dir(cfg) / "MEMORY.md"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return "(no MEMORY.md yet: run `autoexp archive`)\n"
    if len(lines) > MEMORY_DIGEST_LINES:
        lines = lines[:MEMORY_DIGEST_LINES] + [f"... (`autoexp memory` for the rest)"]
    return "\n".join(lines) + "\n"


def _payload() -> Dict[str, Any]:
    if sys.stdin is None or sys.stdin.isatty():
        return {}
    try:
        return json.loads(sys.stdin.read() or "{}")
    except ValueError:
        return {}


def _model(payload: Dict[str, Any]) -> str:
    m = payload.get("model") or os.environ.get("AUTOEXP_MODEL") or ""
    if isinstance(m, dict):
        m = m.get("id") or m.get("display_name") or ""
    return str(m)


def _sid(vendor: str, payload: Dict[str, Any]) -> Optional[str]:
    raw = payload.get("session_id") or payload.get("thread_id") or os.environ.get("AUTOEXP_SESSION")
    if not raw:
        return None
    raw = str(raw)
    return raw if raw.startswith(vendor + "-") else f"{vendor}-{raw[:12]}"


def run_hook(cfg: Config, event: str, vendor: str) -> int:
    try:
        return _run(cfg, event, vendor)
    except Exception as exc:  # never break the agent
        print(f"[autoexp hook] ignored error: {exc}", file=sys.stderr)
        return 0


HANDOFF_FRESH_S = 1800


def _handoff_text(cfg: Config, tier: str) -> str:
    """The brain rewrites HANDOFF.md every tick; reuse it when fresh, render only as a fallback."""
    from .util import age_seconds, now  # noqa: F401
    try:
        age = __import__("time").time() - cfg.handoff_path.stat().st_mtime
        if age < HANDOFF_FRESH_S:
            return cfg.handoff_path.read_text()
    except OSError:
        pass
    return render_handoff(cfg, State.load(EventLog(cfg)), tier=tier)


def _run(cfg: Config, event: str, vendor: str) -> int:
    """Hot paths read two small files and append at most two events: Codex gives
    SessionEnd hooks one to three seconds."""
    from .handoff import marker_path, read_marker
    payload = _payload()
    log = EventLog(cfg)
    sid = _sid(vendor, payload)
    model = _model(payload)
    transcript = payload.get("transcript_path")

    if event == "session-start":
        marker = read_marker(cfg, sid) if sid else None
        if marker is None:
            sid = session_start(log, vendor, model, purpose="", session=sid)
            marker = read_marker(cfg, sid) or {}
        env_file = os.environ.get("CLAUDE_ENV_FILE")
        if env_file:
            with open(env_file, "a") as fh:
                fh.write(f"export AUTOEXP_SESSION={sid}\nexport AUTOEXP_AGENT={vendor}\n")
                if model:
                    fh.write(f"export AUTOEXP_MODEL='{model}'\n")
        tier = marker.get("tier") or guess_tier(model)
        context = PROTOCOL_REMINDER.format(sid=sid)
        if payload.get("source") == "compact":
            context += "\n" + COMPACTED
            log.append("note.added", {"text": f"session {sid} was compacted and re-grounded from MEMORY/HANDOFF",
                                      "kind": "compaction"})
        context += "\n" + _memory_digest(cfg) + "\n" + _handoff_text(cfg, tier)
        print(json.dumps({"hookSpecificOutput": {"hookEventName": "SessionStart",
                                                 "additionalContext": context}}))
        return 0

    if event == "pre-compact":
        if sid and read_marker(cfg, sid) is not None:
            log.append("session.compacting", {"session": sid, "transcript": transcript})
        return 0

    if event == "session-end":
        marker = read_marker(cfg, sid) if sid else None
        if marker is not None and not marker.get("ended"):
            session_end(log, sid, transcript=transcript)  # the brain reconstructs a missing baton
        return 0

    print(f"[autoexp hook] unknown event {event}", file=sys.stderr)
    return 0
