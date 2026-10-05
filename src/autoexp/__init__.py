"""auto-experiment: an experiment orchestration layer on top of Slurm.

The deterministic core (this package) plans runs, submits them, watches them,
classifies failures, retries infrastructure problems and keeps an append-only
record of everything, so that any person or agent can pick up the work later.
"""

__version__ = "0.1.0"
