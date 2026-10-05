# auto-experiment

Experiment orchestration on Slurm with a handover protocol for humans and AI agents.
Read this file first; it is the single source of truth for every agent (CLAUDE.md only imports it).

## What this repository is

A deterministic core (`src/autoexp/`) that plans runs from a campaign spec, submits them,
watches them, classifies failures, retries infrastructure problems, advances stages
(smoke -> pilot -> full) under an autonomy policy, and records everything in an
append-only event log. Language models are optional consumers of that record, never
part of the control loop. Design: `docs/DESIGN.md`.

```
src/autoexp/
  spec.py      strict campaign specs and run planning
  engine.py    tick: poll, classify, retry, advance stages, submit
  runner.py    in-job wrapper: preflight, heartbeat, signals, contract, result.json
  slurm.py     backends (slurm, local, fake) and sbatch rendering
  events.py    event log and the State replayed from it
  contract.py  artifact contracts and no-op parameter detection
  nodes.py     node health registry (shared by a group)
  handoff.py   sessions, batons, tasks, generated HANDOFF.md
  archive.py   memory consolidation: journals, weekly rollups, notebooks, findings, MEMORY.md
  brief.py     deterministic report with robust statistics
  brain.py     long-lived supervisor job (lease, self-requeue)
  hooks.py     SessionStart/PreCompact/SessionEnd hooks for Claude Code and Codex
  canary.py    synthetic workload with known ground truth and fault injection
tests/         python3 -m unittest discover -s tests
examples/      canary campaigns, hook configs, config example
tools/         leakcheck.py (run before every commit)
```

## Boot sequence (every session, any agent)

1. `autoexp session start --agent <claude|codex|human> --model <model>`; keep the session id
   (with hooks installed this happens automatically and `$AUTOEXP_SESSION` is set).
2. `autoexp memory` - the long-term index (findings, campaigns, journals); then
   `autoexp handoff` - what is happening now (campaigns, jobs, tasks, last batons).
3. Claim before acting: `autoexp task claim <id> --session <sid>`.
4. Record as you go (`autoexp note`, `autoexp finding add`); after a context compaction, redo step 2.
5. Before stopping: `autoexp baton write --session <sid> --goal ... --done ... --next ... --question ...`.

## Rules

- This repository is public. Never commit secrets, tokens, absolute paths of a cluster,
  account or user names, node names, or anything about unpublished research. Cluster- and
  user-specific settings live in `~/.autoexp/config.yaml`, never here.
- Run `python3 tools/leakcheck.py` and the tests before every commit.
- Code must run on Python 3.9 with only the standard library and PyYAML. The in-job runner
  must not import yaml.
- Keep the control loop deterministic: no model calls inside `engine.py`, `runner.py`, `brain.py`.
- Every state change goes through `EventLog.append`; never mutate state files by hand.
- When two sources disagree (two skills, two docs, two notes), ask the user which one is
  authoritative before merging or choosing.
- Approving a stage that is not automatic is a human decision. Agents propose; they do not approve.

## Tiers

- any: run tests, `autoexp status`/`brief`/`handoff`, follow a checklist, resubmit infra failures.
- standard: fix bugs with tests, diagnose failures, draft specs.
- strong: change the design, the state machine or the protocol (update docs/DESIGN.md too).
