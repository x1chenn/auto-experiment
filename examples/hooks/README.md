# Agent hooks

The same three hooks serve Claude Code and Codex CLI. Both pass a JSON payload on stdin
(`session_id`, `source` = startup/resume/clear/compact, `transcript_path`, sometimes `model`) and
accept `additionalContext` from SessionStart.

| hook | effect | cost |
|---|---|---|
| `session-start` | registers the session (once; resumes are recognised), sets `AUTOEXP_SESSION` for later shell commands where supported (Claude Code's `CLAUDE_ENV_FILE`), injects MEMORY and HANDOFF; after a compaction also warns the agent to trust the archive over its recollection | reads two generated files |
| `pre-compact` | records the transcript path before the context is compacted | appends one event |
| `session-end` | closes the session; the brain writes a reconstructed baton if the agent left none | appends one event |

Each hook takes about a quarter of a second (mostly interpreter start-up), well inside Codex's
1–3 s limit for SessionEnd.

## Claude Code
Merge `claude-settings.json` into `~/.claude/settings.json` (all projects) or a project's
`.claude/settings.json`. Use the absolute path of `bin/autoexp` if it is not on the PATH of the
editor, and set `AUTOEXP_PYTHON` if the default `python3` lacks PyYAML, e.g.
`AUTOEXP_PYTHON=/usr/bin/python3 /path/to/auto-experiment/bin/autoexp hook session-start --vendor claude`.

## Codex CLI
Copy `codex-hooks.json` to `~/.codex/hooks.json` (or `<repo>/.codex/hooks.json`). Codex runs a
non-managed hook only after you trust its exact definition: open Codex and run `/hooks` once
(and again whenever the file changes). Untrusted hooks are skipped silently.

## Headless model calls
Batch jobs that call `claude -p` or `codex exec` for analysis do not need the handover context;
run them with `--bare` (Claude Code) so user-level hooks are not applied.
