# auto-experiment

Submit experiments in the evening; find out in the morning what ran, what broke, what the
numbers say, and what to run next — with every infrastructure hiccup already handled.

auto-experiment is an orchestration layer on top of Slurm. It does not schedule GPUs (Slurm
does); it schedules *research*: a campaign is one question, run as stages
(smoke → pilot → full) of a parameter grid, with a contract that says what success means.

- **Remembers.** Every action goes to an append-only event log. State is rebuilt by replay,
  so nothing is lost when a session, a process or a person goes away.
- **Distrusts exit codes.** A run succeeds only when its artifact contract holds: files exist,
  metrics were logged up to the last step, values are finite, and the program *echoed back*
  the configuration it was asked to run. Arms that ask for different values but echo the
  same configuration are reported as no-op parameters.
- **Heals infrastructure, never science.** Bad nodes (preflight failures, instant deaths,
  ECC/CUDA signatures), out-of-memory, time limits, preemption and hangs are retried with
  exclusions, more memory, or a checkpoint-and-requeue. Code bugs, broken contracts and
  divergence are never retried; they are reported.
- **Gated autonomy.** Smoke and pilot stages may start on their own; full-scale stages wait for
  a human `autoexp approve`. The policy is enforced in code.
- **Hands over.** Sessions are disposable. Any agent (Claude Code, Codex, any model size) or
  person reads `AGENTS.md`, runs `autoexp handoff`, claims a task and leaves a baton.
  Sessions that end abruptly get a reconstructed baton.

Language models are optional consumers of this record. Nothing in the control loop calls one.

## Quick start

```bash
git clone https://github.com/x1chenn/auto-experiment.git
export PATH="$PWD/auto-experiment/bin:$PATH"     # or: pip install -e auto-experiment
autoexp init                                      # creates ~/.autoexp/config.yaml
$EDITOR ~/.autoexp/config.yaml                    # account, default_partition, brain partition
autoexp check

autoexp submit auto-experiment/examples/canary/faults.yaml   # 9 injected failures, ~20 min, 1 CPU each
autoexp brain submit                              # long-lived supervisor job (or run `autoexp tick` yourself)
autoexp status
autoexp runs canary-faults
autoexp brief
```

No cluster at hand? `AUTOEXP_BACKEND=local` runs the same jobs as local processes.

## A campaign spec

```yaml
name: lr-sweep
hypothesis: "lr=3e-4 beats lr=1e-3"
command: "{python} train.py --lr {lr} --seed {seed} --out {run_dir}"
params: {lr: [0.0003, 0.001]}
seeds: [0, 1, 2, 3, 4]
resources: {partition: gpu, gpus: 1, cpus: 8, mem: 32G, time: "12:00:00"}
stages:
  - {name: smoke, seeds: [0], set: {steps: 200}, resources: {time: "00:20:00"}, auto: true}
  - {name: pilot, seeds: [0, 1], auto: true}
  - {name: full, auto: false}
contract:
  files: [final_eval.json]
  metrics_file: metrics.jsonl
  metric_keys: [eval/return]
  final_step_at_least: "{steps}"
  echo_file: config_echo.json      # your program writes the config it actually used
  echo_keys: [lr, seed]
  finite: {file: final_eval.json, keys: [eval/return]}
preflight:
  - {name: cuda, cmd: "{python} -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)'"}
resume: {enabled: true, signal_seconds: 300}   # SIGUSR1 -> checkpoint -> requeue -> resume
analysis: {primary: eval/return, result_file: final_eval.json}
```

Specs are strict: unknown keys, unknown `{placeholders}` and grid parameters that the command
never uses are errors. A submitted spec is frozen; editing the file later changes nothing.

## Handover protocol

```bash
autoexp session start --agent codex --model <model>    # prints the session id and HANDOFF
autoexp task claim T20261005-... --session <sid>
autoexp decide "drop beta from the grid" --why "no-op in smoke"
autoexp baton write --session <sid> --goal "..." --done "..." --next "..." --question "..."
```

`autoexp init --project <dir>` installs the protocol into a project's `AGENTS.md` (read
natively by Codex) and a one-line `CLAUDE.md` that imports it. Hook configurations for Claude
Code and Codex are in `examples/hooks/`.

## Status

Milestone M0/M1 of `docs/DESIGN.md`: spec, engine, runner, failure classification, node
registry, stages with approval, brain, deterministic brief, handover protocol, canary.
Next: model-written analysis with a critic from another vendor, failure diagnosis, proposals.

## Development

```bash
python3 -m unittest discover -s tests
python3 tools/leakcheck.py      # before every commit; private terms come from ~/.autoexp/deny_terms.txt
```

MIT license.
