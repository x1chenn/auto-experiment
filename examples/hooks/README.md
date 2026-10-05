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

## Headless model calls from batch jobs
Analysis calls made by the brain do not need the handover context and must not share the
interactive login. Give them their own configuration directory and credentials:

- Claude Code: `CLAUDE_CONFIG_DIR=<private dir> CLAUDE_CODE_OAUTH_TOKEN=$(cat <token file>) claude -p ...`
  (the token comes from `claude setup-token`). A separate config directory loads no user-level
  hooks or settings and never refreshes the interactive session's credentials. Do not use
  `--bare` with a subscription token: bare mode ignores OAuth and accepts only an API key.
- Codex: `CODEX_HOME=<private dir> codex exec ...` after a one-time
  `mkdir -p -m 700 <private dir> && CODEX_HOME=<private dir> codex login --device-auth`
  (Codex refuses a `CODEX_HOME` that does not exist yet).
