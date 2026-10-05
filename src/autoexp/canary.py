"""A synthetic, cheap, long-running "training" job with known ground truth.

It exercises everything a real experiment does (metrics, checkpoints, resume,
config echo, final evaluation) on one CPU core and almost no memory, and it
can inject every failure class the system is supposed to handle. Because the
true effects are known, it also tests whether an analysis reached the right
conclusion.

Ground truth: the asymptotic return peaks at lr = 3e-4 (100) and is 8% lower at
lr = 1e-3 (92); ``beta`` has no effect at all.

Faults (``--fault``):
  none              healthy run
  crash             raises at mid-training on every attempt          -> code
  preflight         preflight check fails on attempt 1 only          -> infra/node, then ok
  exit0             exits 0 without writing final_eval.json          -> contract
  nan               evaluation becomes NaN from mid-training          -> diverged
  hang              stops making progress on attempt 1 only          -> infra/hung, then ok
  oom               simulated OOM kill on attempt 1 only (message + exit 137; --real-oom
                    allocates --oom-mb MiB instead)                  -> infra/oom, then ok
  ignore_beta       silently ignores --beta (echoes the default)     -> config_mismatch
  slow              takes longer than its time limit; with resume enabled it is
                    checkpointed and requeued until it finishes      -> ok after restarts
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import signal
import sys
import time
from pathlib import Path

FAULTS = ["none", "crash", "preflight", "exit0", "nan", "hang", "oom", "ignore_beta", "slow"]
DEFAULT_BETA = 0.5


def max_return(lr: float) -> float:
    k = -math.log(0.92) / (math.log10(1e-3) - math.log10(3e-4)) ** 2
    return 100.0 * math.exp(-k * (math.log10(lr) - math.log10(3e-4)) ** 2)


def noise(seed: int, step: int, scale: float) -> float:
    return random.Random(seed * 1_000_003 + step).gauss(0.0, scale)


def expected_return(lr: float, seed: int, step: int, steps: int) -> float:
    tau = max(1.0, steps / 5.0)
    seed_offset = random.Random(seed * 7919).gauss(0.0, 2.0)
    return max_return(lr) * (1.0 - math.exp(-step / tau)) + seed_offset + noise(seed, step, 1.0)


def attempt() -> int:
    return int(os.environ.get("AUTOEXP_ATTEMPT", "1") or 1)


def write_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, path)


class Training:
    def __init__(self, args: argparse.Namespace):
        self.a = args
        self.out = Path(args.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.ckpt = self.out / "ckpt.json"
        self.metrics = self.out / "metrics.jsonl"
        self.step = 0
        self.evals = []
        self.beta = DEFAULT_BETA if args.fault == "ignore_beta" else args.beta

    def save(self) -> None:
        write_json(self.ckpt, {"step": self.step, "evals": self.evals[-20:]})

    def resume(self) -> None:
        if not self.ckpt.exists():
            return
        state = json.loads(self.ckpt.read_text())
        self.step, self.evals = int(state["step"]), list(state.get("evals", []))
        if self.metrics.exists():  # drop metrics logged after the checkpoint
            keep = [ln for ln in self.metrics.read_text().splitlines()
                    if ln.strip() and json.loads(ln).get("step", 0) <= self.step]
            self.metrics.write_text("".join(ln + "\n" for ln in keep))
        print(f"[canary] resumed from step {self.step}", flush=True)

    def on_signal(self, signum, frame) -> None:
        self.save()
        print(f"[canary] signal {signum}: checkpoint saved at step {self.step}", flush=True)
        sys.exit(85 if signum == signal.SIGUSR1 else 143)

    def run(self) -> int:
        a = self.a
        signal.signal(signal.SIGUSR1, self.on_signal)
        signal.signal(signal.SIGTERM, self.on_signal)
        write_json(self.out / "config_echo.json",
                   {"lr": a.lr, "beta": self.beta, "steps": a.steps, "seed": a.seed, "fault": a.fault})
        self.resume()
        half = a.steps // 2
        hog = None
        pace = a.seconds_per_step * (4.0 if a.fault == "slow" else 1.0)
        while self.step < a.steps:
            time.sleep(pace)
            self.step += 1
            if a.fault == "crash" and self.step == half:
                raise RuntimeError(f"injected crash at step {self.step}")
            if a.fault == "hang" and self.step == half and attempt() == 1:
                print("[canary] injected hang", flush=True)
                while True:
                    time.sleep(60)
            if a.fault == "oom" and self.step == half and attempt() == 1:
                # Simulated, never real: on clusters where memory above --mem spills into swap a
                # real overshoot does not die, it only slows down a shared node.
                if a.real_oom:
                    hog = bytearray(a.oom_mb * 1024 * 1024)
                    for i in range(0, len(hog), 4096):
                        hog[i] = 1
                else:
                    print("slurmstepd: error: Detected 1 oom_kill event in StepId=canary.batch. "
                          "Some of the step tasks have been OOM Killed.", file=sys.stderr, flush=True)
                    os._exit(137)
            if self.step % a.log_every == 0 or self.step == a.steps:
                ret = expected_return(a.lr, a.seed, self.step, a.steps)
                if a.fault == "nan" and self.step >= half:
                    ret = float("nan")
                self.evals.append(ret)
                rec = {"step": self.step, "eval/return": ret,
                       "train/loss": 1.0 / (1.0 + self.step / 100.0) + abs(noise(a.seed, -self.step, 0.01))}
                with open(self.metrics, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
            if self.step % a.ckpt_every == 0:
                self.save()
        self.save()
        if a.fault == "exit0":
            print("[canary] injected: exiting 0 without results", flush=True)
            return 0
        last = self.evals[-5:]
        final = sum(last) / len(last) if last else float("nan")
        write_json(self.out / "final_eval.json", {"eval/return": final, "step": self.step})
        print(f"[canary] done: eval/return={final:.3f}", flush=True)
        return 0


def preflight(args: argparse.Namespace) -> int:
    if args.fault == "preflight" and attempt() == 1:
        print("[canary] injected preflight failure: pretending CUDA is unavailable", flush=True)
        return 1
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="synthetic training job for auto-experiment")
    p.add_argument("--out", required=False, default=".")
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--beta", type=float, default=DEFAULT_BETA)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--seconds-per-step", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--ckpt-every", type=int, default=50)
    p.add_argument("--fault", choices=FAULTS, default="none")
    p.add_argument("--oom-mb", type=int, default=2048)
    p.add_argument("--real-oom", action="store_true", help="really allocate --oom-mb instead of simulating")
    p.add_argument("--preflight", action="store_true", help="run the fake preflight check and exit")
    args = p.parse_args(argv)
    if args.preflight:
        return preflight(args)
    return Training(args).run()


if __name__ == "__main__":
    sys.exit(main())
