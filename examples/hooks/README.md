# Agent hooks

The same three hooks serve Claude Code and Codex CLI; both pass a JSON payload with a
`session_id` on stdin and accept `additionalContext` from SessionStart.

| hook | effect |
|---|---|
| `session-start` | registers the session, sets `AUTOEXP_SESSION` for later commands (Claude Code, via `CLAUDE_ENV_FILE`), injects the HANDOFF digest |
| `pre-compact` | writes a reconstructed baton if the session has none yet |
| `session-end` | same, then closes the session |

- **Claude Code**: merge `claude-settings.json` into `.claude/settings.json` (project) or
  `~/.claude/settings.json` (user).
- **Codex CLI**: hooks are a stable feature in recent versions and use the same event names;
  `codex-hooks.json` follows the Claude Code layout. Check the location and format against
  your Codex version (`codex features list`, Codex docs) before relying on it.

`autoexp` must be on the PATH of the agent (or use the absolute path of `bin/autoexp`).
Hooks never fail the session: errors are printed to stderr and ignored.
