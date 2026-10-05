"""User configuration and the on-disk layout of the state directory.

Layout of ``$AUTOEXP_HOME`` (default ``~/.autoexp``; must be on a filesystem
that every node can see):

    config.yaml          user settings (account, partitions, autonomy, brain)
    events/*.jsonl       append-only event log, the single source of truth
    campaigns/<name>/    frozen copy of each submitted spec
    sessions/            agent/human session batons (rendered markdown)
    briefs/              morning briefs
    logs/                brain logs
    HANDOFF.md           generated "what is going on right now" page
    brain.lease          heartbeat lease of the running brain

``$AUTOEXP_SHARED`` (default ``$AUTOEXP_HOME/shared``) holds knowledge that a
whole group can share: the node health registry and failure signatures.
"""

from __future__ import annotations

import copy
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict

import yaml

DEFAULTS: Dict[str, Any] = {
    # Interpreter used for the in-job runner and the brain. Must exist on compute nodes.
    "python": sys.executable,
    "shared": None,
    "backend": "slurm",  # slurm | local (local runs jobs as background processes; for tests/demos)
    # Autonomy: stages whose spec says `auto: true` may only start without a human
    # if their name is listed here. Enforced in code, not by any prompt.
    "auto_stages": ["smoke", "pilot"],
    "max_active_jobs": 50,
    # Partition used when a spec does not name one (keeps specs portable).
    "default_partition": None,
    "sbatch": {
        "account": None,
        "comment": None,  # some clusters use --comment for billing; we never use it for tags
        "lines": [],  # extra raw "#SBATCH ..." options added to every job
    },
    "brain": {
        "partition": None,
        "cpus": 1,
        "mem": "2G",
        "time": "1-00:00:00",
        "tick_seconds": 120,
        "signal_seconds": 600,
        "brief_at": "06:30",
        "lines": [],
    },
    "watch": {
        "heartbeat_stale_minutes": 20,
        "lost_after_minutes": 15,
        "session_unclean_hours": 6,
    },
}


def _merge(base: Dict[str, Any], over: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class Config:
    home: Path
    data: Dict[str, Any] = field(default_factory=dict)

    # -- settings
    def __getitem__(self, key: str) -> Any:
        return self.data[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    @property
    def python(self) -> str:
        return str(self.data["python"])

    @property
    def backend_name(self) -> str:
        return os.environ.get("AUTOEXP_BACKEND") or self.data["backend"]

    # -- paths
    @property
    def shared(self) -> Path:
        env = os.environ.get("AUTOEXP_SHARED")
        if env:
            return Path(env).expanduser()
        if self.data.get("shared"):
            return Path(self.data["shared"]).expanduser()
        return self.home / "shared"

    @property
    def events_dir(self) -> Path:
        return self.home / "events"

    @property
    def campaigns_dir(self) -> Path:
        return self.home / "campaigns"

    @property
    def sessions_dir(self) -> Path:
        return self.home / "sessions"

    @property
    def briefs_dir(self) -> Path:
        return self.home / "briefs"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def handoff_path(self) -> Path:
        return self.home / "HANDOFF.md"

    @property
    def lease_path(self) -> Path:
        return self.home / "brain.lease"

    @property
    def nodes_path(self) -> Path:
        return self.shared / "nodes.yaml"

    @property
    def signatures_path(self) -> Path:
        return self.shared / "signatures.yaml"

    def ensure_dirs(self) -> None:
        for d in (self.home, self.events_dir, self.campaigns_dir, self.sessions_dir,
                  self.briefs_dir, self.logs_dir, self.shared):
            d.mkdir(parents=True, exist_ok=True)


def home_path() -> Path:
    return Path(os.environ.get("AUTOEXP_HOME", "~/.autoexp")).expanduser()


def load_config(home: Path = None) -> Config:
    home = Path(home) if home else home_path()
    user: Dict[str, Any] = {}
    cfg_file = home / "config.yaml"
    if cfg_file.exists():
        with open(cfg_file) as fh:
            user = yaml.safe_load(fh) or {}
    return Config(home=home, data=_merge(DEFAULTS, user))


CONFIG_TEMPLATE = """\
# auto-experiment user configuration. Everything here is optional.
# This file lives outside any repository: account names and paths belong here.

# Interpreter for the in-job runner and the brain (must exist on compute nodes).
python: {python}

# Directory shared with your group (node health registry, failure signatures).
# shared: /path/visible/to/your/group/autoexp_shared

# Stages that may start without a human, if the spec also marks them auto.
auto_stages: [smoke, pilot]

max_active_jobs: 50

# Partition for specs that do not name one, e.g. a CPU partition for the canary.
default_partition: null

sbatch:
  account: null        # e.g. my_lab_account
  comment: null        # only if your cluster wants a --comment (e.g. billing)
  lines: []            # extra raw options, e.g. ["--qos=normal"]

brain:
  partition: null      # a CPU partition that allows long jobs
  cpus: 1
  mem: 2G
  time: "1-00:00:00"
  tick_seconds: 120
  brief_at: "06:30"
"""
