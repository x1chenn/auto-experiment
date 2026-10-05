# auto-experiment — design

Status: v0.1 (milestones M0–M1.5 implemented). This document is the reference for anyone,
human or agent, changing the system.

## 1. Purpose

Run the research loop *question → experiments → evidence → next question* on a shared
Slurm cluster with as little babysitting as possible:

1. **Remember.** What was submitted, why, with which code/config/node, and what came out —
   recoverable next morning, next week, by another session, another model, another person.
2. **Report.** A morning brief: what ran, what failed (infrastructure vs. code vs. science),
   result tables with robust statistics, open approvals.
3. **Close the loop.** Smoke-test new work automatically, scale it up through gates, and
   (later) propose the next experiments as concrete specs.
4. **Heal infrastructure overnight.** Bad nodes, OOM, time limits, preemption, hangs.
5. **Hand over seamlessly** between sessions, model sizes and vendors (§9), and
6. **never forget an important result** across long sessions and weeks (§10).

Non-goals: replacing Slurm, replacing experiment dashboards, writing papers, or letting a
model make scientific decisions without a human.

## 2. Lessons that shaped the design

Collected from months of running reinforcement-learning experiments by hand on a university
cluster (thousands of jobs, roughly one in five ending abnormally):

| Lesson | Consequence |
|---|---|
| Some nodes kill jobs within seconds, have ECC errors or no usable CUDA, and keep attracting new jobs; hand-maintained exclude lists go stale; a single unknown node name makes sbatch reject a job | In-job preflight, a shared node registry with evidence and expiry, excludes filtered against `sinfo` |
| Launchers can exit 0 when the program failed; jobs can COMPLETE with no results | Success is defined by an artifact contract, not the exit code |
| A failed CUDA initialisation can fall back to CPU silently and burn days of walltime | Preflight fails hard when requested GPUs are not usable |
| Flags get silently ignored (renamed keys, wrong defaults, a wrapper that runs another algorithm) | Strict specs; config echo checked per run; differential no-op detection across arms |
| Training finishes but evaluation never gets submitted; evaluation picks a stale checkpoint | Stages are explicit and advanced by the system; contracts can require final steps and files |
| `afterok` dependency chains leave zombies when an upstream job fails | Gates are evaluated by the supervisor, not by Slurm dependencies |
| Editing a batch script after submission changed queued jobs | Every attempt runs from a frozen launch bundle with a protocol hash |
| Mass cancellations lose track of work | Each tick reconciles declared runs, scheduler state and files |
| One outlier seed can invert a ranking of means | Median, IQM and bootstrap CIs by default |
| Supervisors written as batch jobs die at their own time limit | Self-requeueing brain with a heartbeat lease and lazy resurrection |
| Knowledge lives in one person's notes or one vendor's memory store | Vendor-neutral event log, generated handoff, batons |

## 3. Principles

1. **Deterministic core, models at the edges.** State machine, scheduler I/O, contracts and
   statistics are plain code. Models (later) compile intent into specs, explain failures,
   comment on tables and propose next steps — and the system works without them.
2. **Event sourcing.** The only truth is an append-only JSONL log with sequence numbers.
   Everything else (state, HANDOFF.md, briefs) is derived and can be regenerated.
3. **Separate infrastructure, code and science.** Only infrastructure failures are retried.
4. **Contracts over exit codes.**
5. **Numbers come from code.** Any prose may only cite numbers present in generated tables.
6. **Autonomy is a policy enforced in code** (`auto_stages`), never a prompt.
7. **Humans can always take over;** every automatic action is recorded with its reason.

## 4. Architecture

```
 login node (light)          shared filesystem                      CPU partition
 ───────────────────         ─────────────────────────────          ───────────────────────
 autoexp CLI  ───────────▶   $AUTOEXP_HOME                   ◀────  brain job (1 per user)
 agents via hooks/CLI        events/*.jsonl  (truth)                 tick every ~2 min:
                             campaigns/<name>/spec.yaml (frozen)      poll · classify · retry
                             HANDOFF.md, briefs/, sessions/           advance stages · submit
                             $AUTOEXP_SHARED/nodes.yaml (group)       self-requeue on SIGUSR1
                                       ▲                                   │ sbatch
                                       │ result.json, heartbeat            ▼
                             run_dir/.autoexp/                   GPU/CPU partitions
                                                                 runner → user command
```

- **Engine** (`engine.py`): one `tick()` polls `sacct` (the truth; `squeue` forgets finished
  jobs within seconds on many clusters), cancels stalled attempts, classifies finished ones,
  flags nodes, schedules retries, advances stages and submits within limits.
- **Runner** (`runner.py`): wraps every job. Records restarts (`SLURM_RESTART_COUNT`, since
  preemption may requeue silently), runs preflight checks, launches the command in its own
  process group, writes a heartbeat with the latest step, forwards SIGUSR1 so the program can
  checkpoint, waits for the whole group, requeues, and finally checks the contract and writes
  `result.json`.
- **Brain** (`brain.py`): the tick loop as a batch job on a CPU partition. A lease file with a
  heartbeat (not a held lock) prevents two brains. Shortly before its time limit it requeues
  itself under the same job id, which also works under a QOS allowing a single submitted job.
  `autoexp brain ensure` resubmits it if it died.
- **Tags**: jobs carry `--job-name=ae:<campaign>:<stage>` and `--extra=ae:<run_id>;plan=<hash>`.
  `--comment` is left alone because some clusters use it for billing.

## 5. Data model

Campaign → stages → runs (grid point × seed) → attempts (one Slurm submission each).

- `run_id = <campaign>/<stage>/<config-hash>-s<seed>`; planning is deterministic, so a run is
  never submitted twice.
- An attempt records job id, resources, excludes, bundle path and protocol hash; its
  classification comes from the scheduler state, `result.json`, `preflight.json` and log
  signatures (built in, extendable per group in `signatures.yaml`).
- Sessions, batons, tasks (with leases) and decisions live in the same log.

## 6. Specs

Strict YAML (see README). Unknown keys, unknown placeholders and unused grid parameters are
errors. Stages may override seeds, parameters (`set`) and resources. Submission freezes the
spec; the original file can change without affecting the campaign.

## 7. Failures and retries

| Class | Typical cause | Action |
|---|---|---|
| `infra/node` | preflight failed, instant death without output, ECC/CUDA signature, NODE_FAIL | flag node (suspect 24 h; second strike within 24 h → bad for 7 days; test campaigns: scoped, 2 h), retry with excludes |
| `infra/oom` | OUT_OF_MEMORY or OOM signature | retry once with memory × 1.5 |
| `infra/timeout` | time limit | with `resume`: checkpoint + requeue (same job); otherwise one retry with time × 1.5 |
| `infra/preempt`, `infra/unknown`, `infra/lost`, `infra/runner` | | retry up to the policy limit |
| `infra/hung` | no heartbeat or no training progress for `stall_minutes` | cancel, retry once |
| `code` | non-zero exit | not retried |
| `contract` | exit 0 but files/metrics/steps missing | not retried |
| `config_mismatch` | echoed config differs from the request | not retried |
| `diverged` | non-finite results | not retried |

A stage succeeds when all runs succeeded (or at most `max_failures` failed) and no no-op
parameter was detected; otherwise the campaign is blocked and the brief says why.

## 8. Stages and autonomy

`smoke → pilot → full` by convention. Submitting a campaign approves its first stage. A later
stage starts automatically only if the spec marks it `auto: true` **and** the user's
`auto_stages` policy lists it (default: smoke, pilot). Otherwise it waits for
`autoexp approve <campaign> <stage>`, which is a human decision.

## 9. Handover protocol

Sessions are disposable; state is external. The protocol must work for the weakest agent we
allow and must not depend on any vendor-private memory.

**Layers**
- `AGENTS.md` (hand-written, short): what the project is, the boot sequence, hard rules, tiers.
  Codex reads it natively; `CLAUDE.md` contains only `@AGENTS.md`.
- `autoexp handoff` / `HANDOFF.md` (generated, ≤ 200 lines, stamped with the last event
  sequence): brain status, campaigns, jobs, recent failures, excluded nodes, open tasks for
  the reader's tier, last batons, recent decisions.
- The event log and per-run files, queried on demand (`autoexp status/runs/log`).

**Boot**: `session start` → `handoff` → `task claim` → work → `baton write`.

**Batons** record goal, done (with evidence), in flight, next (with acceptance criteria),
open questions, unverified claims and touched files. If a session goes quiet without one,
the brain writes a reconstructed baton flagged as such.

**Tiers** (`any`, `standard`, `strong`) label tasks; the handoff shows each reader only what
its tier may take on. Dangerous operations are blocked in code regardless of tier.

**Hooks**: Claude Code and Codex CLI expose the same lifecycle events (SessionStart,
PreCompact, SessionEnd, …) and accept `additionalContext`; one `autoexp hook` command serves
both (examples/hooks).

**Conflicting sources** (two skills, two notes) are never merged silently: the user decides
which is authoritative.

**Measuring it** (planned, `autoexp drill`): fresh headless sessions of several models and
vendors read only AGENTS.md and answer a quiz generated from the state (what is running, the
next task and its acceptance criterion, the last decision and why, what is forbidden), plus
one practical task. Pass rates per model are the adaptability metric.

## 10. Memory consolidation

Long sessions lose detail when their context is compacted, and people forget what was
learned three weeks ago. Conversation memory is therefore never relied upon; knowledge is
written into the event log as it is produced and consolidated into bounded documents.

**What enters memory (events only):** frozen stage results (`stage.results`, emitted by the
engine when a stage completes, so tables survive deleted run directories), findings
(`autoexp finding add`, claims with evidence), notes (`autoexp note`), decisions, batons,
failure classes and node flags.

**Findings ledger.** Each finding has an id from its event sequence (`F<seq>`), a claim, a
scope (part / campaign / stage), evidence that must resolve (a frozen results table, run ids,
`job:`, `commit:`, `file:`, `url:`), and a status: `tentative → supported | refuted`, or
`superseded` by a newer finding. Findings are never edited; re-rating appends history. Numbers
in a claim that do not occur in the cited table (or in the arm labels) produce a warning.

**Generated documents (`$AUTOEXP_HOME/archive`, or `archive_dir`):**

| Document | Scope | Size | Purpose |
|---|---|---|---|
| `MEMORY.md` | everything | ≤ 150 lines | long-term index: supported findings (kept until superseded), tentative ones, campaigns grouped by part with their latest headline result, decisions, what is open, links |
| `FINDINGS.md` | all findings | grows | the ledger with status history |
| `notebooks/<campaign>.md` | one campaign | grows slowly | intent, stages, frozen results per stage with run ids, findings, failures, timeline, notes |
| `journal/YYYY-MM-DD.md` | one day | bounded by activity | progress per campaign, results frozen that day, findings, decisions, notes, infrastructure, tasks, sessions and batons |
| `journal/weekly/YYYY-Www.md` | one week | one line per day | rollup with links to the daily journals |

Generation is deterministic and incremental: the brain regenerates today's and yesterday's
journals, the notebooks, the ledger and the index on every tick; `autoexp archive --rebuild`
regenerates everything. Journals and notebooks contain no wall-clock timestamp, so unchanged
inputs produce byte-identical files and the archive can be versioned in a private repository.

**Reading cost is bounded:** a new or compacted session reads MEMORY (≤ 150 lines) and HANDOFF
(≤ 200 lines), then only the notebook of its campaign. The SessionStart hook injects both;
when the session was just compacted it also warns the agent to trust the archive over its
own recollection, and records the compaction as a note.

**Writing discipline (in AGENTS.md):** record a finding or note at every milestone rather than
at the end; never edit generated documents; agents propose findings as `tentative`, and
promoting one to `supported` needs a strong-tier agent or a human.

## 11. Model roles (planned, M2–M4)

| Role | Input | Output | Guard |
|---|---|---|---|
| Compiler | natural-language intent | spec draft | schema validation, dry run, human confirms |
| Doctor | failure class, log tail, diff since last success | diagnosis, action, optional patch on a branch | patch must pass smoke; human approves |
| Analyst | brief tables, hypothesis, prior insights | insights citing run ids | every number must exist in the tables |
| Critic (other vendor) | analyst output | objections | refuted insights stay tentative |
| Planner | insights, tried config hashes, budget | proposals as spec patches | dedup by hash; cost computed from smoke throughput |

Models are called through official CLIs in headless mode (`claude -p --json-schema`,
`codex exec --output-schema`) with subscription or API credentials kept outside the repo,
only when the deterministic tick saw a change, and with a daily cap.

## 12. Prior art and what we took

- **seml** — explicit scheduler-state mapping, config-hash dedup, reconciliation.
- **submitit** — checkpoint-then-requeue semantics.
- **Hydra** — resolved configs saved per run (our config echo).
- **Snakemake/Nextflow** — resumability, resource escalation per attempt.
- **Optuna/Ray Tune** — ask/tell separation and successive halving (our pilot gate).
- **CodeScientist** — mini-pilot → pilot → full with a human before full.
- **AI Scientist, AIDE, RD-Agent, AgentLaboratory** — experiment journals, bounded debug
  depth, hypothesis/feedback records; and the lesson that fully autonomous science fails often.
- **karpathy/autoresearch** — small program files as skills, read logs with grep only.
- **CheetahClaws, OpenClaw, NanoClaw** — markdown memory with verification dates, single
  writers, waking the model only when there is work, approvals bound to the requesting client.
- **xgenius** — safety limits enforced in code for agent-driven Slurm use.

## 13. Roadmap

| Milestone | Content | Status |
|---|---|---|
| M0 | specs, engine, runner, contracts, node registry, CLI, canary | done |
| M1 | brain, deterministic brief, handoff/batons/tasks, hooks | done (field-tested) |
| M1.5 | memory consolidation: frozen results, findings ledger, journals, notebooks, MEMORY.md | done |
| M2 | analyst + critic, insight ledger, doctor (read-only), `autoexp drill` | next |
| M3 | proposals queue, gates with conditions, MCP server | |
| M4 | planner, numeric search via ask/tell | |
| M5 | group onboarding: shared registry, `init --project` everywhere, docs | |
