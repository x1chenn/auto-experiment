"""Handover files installed into a user's project by ``autoexp init --project``.

One source of truth (AGENTS.md) and thin per-vendor entry points: Codex reads
AGENTS.md natively; Claude Code reads CLAUDE.md, which only imports AGENTS.md.
Only the block between the markers is managed; the rest of AGENTS.md is the
project's own.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List

BEGIN = "<!-- autoexp:begin (managed by `autoexp init --project`; edit outside these markers) -->"
END = "<!-- autoexp:end -->"

AGENTS_BLOCK = f"""{BEGIN}
## Experiments in this project run through auto-experiment

Every agent (any vendor, any model size) and every person follows this protocol.
Sessions are disposable; the state lives in auto-experiment, not in your memory.

**Boot, every session**
1. `autoexp session start --agent <claude|codex|human> --model <model>` (a hook may already
   have done this; then `$AUTOEXP_SESSION` is set). Keep the session id.
2. Read the current state: `autoexp handoff` (generated from the event log; never edit it).
3. Claim before acting: `autoexp task claim <task-id> --session <sid>`.

**Rules**
- Submit experiments only with `autoexp submit <spec.yaml>`; never call `sbatch` for experiments.
- Never edit a submitted spec to change a running campaign: submit a new campaign instead.
- A run succeeded only if its contract holds (`autoexp runs <campaign>`), not because it exited 0.
- Numbers you report come from `autoexp brief` tables or result files, cited by run id.
- Approving a non-automatic stage (`autoexp approve`) is a human decision: propose, never approve.
- When two sources disagree (two skills, two docs, two notes), ask the user which one is
  authoritative before merging or choosing.
- Record decisions with `autoexp decide "<what>" --why "<why>"`.

**Tiers**
- any: status, brief, follow a checklist, resubmit infrastructure failures.
- standard: diagnose failures, analyze results, draft specs.
- strong: plan campaigns and propose changes to the science (a human still approves).

**Before you stop**
`autoexp baton write --session <sid> --goal "..." --done "..." --next "..." --question "..."`
(if you cannot, a baton is reconstructed from your recorded actions and flagged as such).
{END}
"""


def install_project_files(project: Path) -> List[str]:
    project = Path(project)
    out = []
    agents = project / "AGENTS.md"
    if agents.exists():
        text = agents.read_text()
        if BEGIN in text and END in text:
            text = re.sub(re.escape(BEGIN) + r".*?" + re.escape(END) + r"\n?", AGENTS_BLOCK, text, flags=re.S)
            out.append(f"updated the auto-experiment block in {agents}")
        else:
            text = text.rstrip() + "\n\n" + AGENTS_BLOCK
            out.append(f"appended the auto-experiment block to {agents}")
    else:
        text = f"# {project.resolve().name}\n\n" + AGENTS_BLOCK
        out.append(f"created {agents}")
    agents.write_text(text)

    claude = project / "CLAUDE.md"
    if not claude.exists():
        claude.write_text("@AGENTS.md\n")
        out.append(f"created {claude} (imports AGENTS.md)")
    elif "@AGENTS.md" not in claude.read_text():
        out.append(f"note: {claude} exists and does not import AGENTS.md; add a line '@AGENTS.md' to it")
    return out
