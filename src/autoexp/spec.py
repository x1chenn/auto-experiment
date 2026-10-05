"""Campaign specs: loading, strict validation and run planning.

A campaign is one research question. Its spec declares the command, the grid
of parameters, the seeds, the resources, the success contract and an ordered
list of stages (smoke -> pilot -> full by convention). Validation is strict on
purpose: an unknown key, an unknown placeholder or a grid parameter that the
command never uses is an error, because each of those silently turns an
experiment arm into a copy of another one.
"""

from __future__ import annotations

import copy
import itertools
import re
from pathlib import Path
from typing import Any, Dict, List

import yaml

from .util import stable_hash

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
PLACEHOLDER_RE = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")

# Placeholders always available in command/recipe/contract templates.
BUILTINS = {"seed", "run_dir", "python", "autoexp_src", "campaign", "stage", "run_id", "workdir", "attempt"}

TOP_KEYS = {
    "name", "part", "hypothesis", "success", "notes", "test", "workdir", "run_root", "recipe", "recipe_file",
    "command", "params", "seeds", "fixed", "resources", "stages", "contract", "preflight",
    "progress", "resume", "retry", "analysis", "limits", "allow_unused",
}
STAGE_KEYS = {"name", "seeds", "params", "set", "resources", "auto", "max_failures"}
RESOURCE_KEYS = {"partition", "cpus", "mem", "time", "gpus", "sbatch"}
CONTRACT_KEYS = {"files", "metrics_file", "metric_keys", "final_step_at_least", "step_key",
                 "echo_file", "echo_keys", "finite"}

DEFAULT_STAGES = [{"name": "smoke", "auto": True}, {"name": "full", "auto": False}]


class SpecError(ValueError):
    pass


def _check_keys(where: str, data: Dict[str, Any], allowed: set) -> None:
    unknown = sorted(set(data) - allowed)
    if unknown:
        raise SpecError(f"{where}: unknown key(s) {unknown}; allowed: {sorted(allowed)}")


class Spec:
    def __init__(self, data: Dict[str, Any], source: Path):
        self.raw = copy.deepcopy(data)
        self.source = Path(source).resolve()
        self.data = copy.deepcopy(data)
        self._normalize()
        self.validate()

    # ------------------------------------------------------------ loading
    @classmethod
    def load(cls, path: Path) -> "Spec":
        path = Path(path)
        with open(path) as fh:
            data = yaml.safe_load(fh)
        if not isinstance(data, dict):
            raise SpecError(f"{path}: expected a mapping at top level")
        return cls(data, path)

    def _normalize(self) -> None:
        d = self.data
        base = self.source.parent
        d.setdefault("params", {})
        d.setdefault("fixed", {})
        d.setdefault("seeds", [0])
        d.setdefault("resources", {})
        d.setdefault("contract", {})
        d.setdefault("preflight", [])
        d.setdefault("progress", {})
        d.setdefault("resume", {"enabled": False})
        d.setdefault("retry", {})
        d.setdefault("analysis", {})
        d.setdefault("limits", {})
        d.setdefault("allow_unused", [])
        d.setdefault("stages", copy.deepcopy(DEFAULT_STAGES))
        d["workdir"] = str((base / d.get("workdir", ".")).resolve())
        d["run_root"] = str((base / d.get("run_root", "autoexp_runs")).resolve())
        if d.get("recipe_file"):
            d["recipe_file"] = str((base / d["recipe_file"]).resolve())
        if isinstance(d.get("recipe"), list):
            d["recipe"] = "\n".join(d["recipe"])
        d.setdefault("recipe", "")
        for st in d["stages"]:
            st.setdefault("auto", st.get("name") == "smoke")
        r = d["retry"]
        r.setdefault("infra", 3)
        r.setdefault("preempt", 10)
        r.setdefault("oom_factor", 1.5)
        r.setdefault("oom", 1)
        r.setdefault("timeout_factor", 1.5)
        r.setdefault("timeout", 1)
        r.setdefault("hung", 1)
        res = d["resume"]
        res.setdefault("enabled", False)
        res.setdefault("signal_seconds", 300)
        res.setdefault("max_restarts", 10)

    # ------------------------------------------------------------ validation
    def validate(self) -> None:
        d = self.data
        _check_keys("spec", d, TOP_KEYS)
        name = d.get("name")
        if not name or not NAME_RE.match(str(name)):
            raise SpecError("spec: 'name' is required (letters, digits, _ . -; at most 64 chars)")
        if not d.get("command"):
            raise SpecError("spec: 'command' is required")
        if not isinstance(d["params"], dict) or not all(isinstance(v, list) and v for v in d["params"].values()):
            raise SpecError("spec: 'params' must map each name to a non-empty list of values")
        if not isinstance(d["fixed"], dict):
            raise SpecError("spec: 'fixed' must be a mapping")
        overlap = set(d["params"]) & set(d["fixed"])
        if overlap:
            raise SpecError(f"spec: {sorted(overlap)} appear in both 'params' and 'fixed'")
        if "seed" in d["params"] or "seed" in d["fixed"]:
            raise SpecError("spec: use the top-level 'seeds' list instead of a 'seed' parameter")
        if not isinstance(d["seeds"], list) or not d["seeds"]:
            raise SpecError("spec: 'seeds' must be a non-empty list")
        _check_keys("resources", d["resources"], RESOURCE_KEYS)
        _check_keys("contract", d["contract"], CONTRACT_KEYS)

        names = [st.get("name") for st in d["stages"]]
        if not names or any(not n for n in names) or len(set(names)) != len(names):
            raise SpecError("spec: 'stages' must be a list of uniquely named stages")
        all_keys = set(d["params"]) | set(d["fixed"])
        for st in d["stages"]:
            _check_keys(f"stage {st['name']}", st, STAGE_KEYS)
            _check_keys(f"stage {st['name']}.resources", st.get("resources", {}), RESOURCE_KEYS)
            for key in list(st.get("params", {})) + list(st.get("set", {})):
                if key not in all_keys:
                    raise SpecError(f"stage {st['name']}: '{key}' is not a declared parameter")
            for key, values in st.get("params", {}).items():
                if not isinstance(values, list) or not values:
                    raise SpecError(f"stage {st['name']}: params.{key} must be a non-empty list")

        known = all_keys | BUILTINS
        templates = {"command": d["command"], "recipe": d["recipe"]}
        for i, check in enumerate(d["preflight"]):
            if not isinstance(check, dict) or "cmd" not in check:
                raise SpecError(f"preflight[{i}] needs a 'cmd'")
            templates[f"preflight[{i}]"] = check["cmd"]
        for where, text in templates.items():
            for ph in PLACEHOLDER_RE.findall(str(text)):
                if ph not in known:
                    raise SpecError(f"{where}: unknown placeholder {{{ph}}}")
        used = set(PLACEHOLDER_RE.findall(str(d["command"])))
        unused = [p for p in d["params"] if p not in used and p not in d["allow_unused"]]
        if unused:
            raise SpecError(
                f"grid parameter(s) {unused} never appear in 'command': every arm would run the "
                "same thing. Use them in the command or list them in 'allow_unused'."
            )

    # ------------------------------------------------------------ accessors
    @property
    def name(self) -> str:
        return self.data["name"]

    @property
    def stage_names(self) -> List[str]:
        return [st["name"] for st in self.data["stages"]]

    def stage(self, name: str) -> Dict[str, Any]:
        for st in self.data["stages"]:
            if st["name"] == name:
                return st
        raise KeyError(name)

    def resources_for(self, stage: str, overrides: Dict[str, Any] = None) -> Dict[str, Any]:
        res = {"partition": None, "cpus": 1, "mem": "4G", "time": "01:00:00", "gpus": 0, "sbatch": []}
        res.update(self.data["resources"])
        res.update(self.stage(stage).get("resources", {}))
        res.update(overrides or {})
        return res

    # ------------------------------------------------------------ planning
    def plan(self, stage_name: str) -> List[Dict[str, Any]]:
        """Expand one stage into concrete runs (deterministic run ids)."""
        d = self.data
        st = self.stage(stage_name)
        grid = dict(d["params"])
        grid.update(st.get("params", {}))
        fixed = dict(d["fixed"])
        overrides = dict(st.get("set", {}))
        seeds = st.get("seeds", d["seeds"])
        keys = sorted(grid)
        varying = [k for k in keys if len(grid[k]) > 1 and k not in overrides]
        runs = []
        for combo in itertools.product(*(grid[k] for k in keys)):
            params = dict(fixed)
            params.update(dict(zip(keys, combo)))
            params.update(overrides)
            cfg_hash = stable_hash(params)
            label = ",".join(f"{k}={params[k]}" for k in varying) or "base"
            for seed in seeds:
                run_id = f"{self.name}/{stage_name}/{cfg_hash}-s{seed}"
                run_dir = Path(d["run_root"]) / self.name / stage_name / f"{cfg_hash}-s{seed}"
                runs.append({
                    "run_id": run_id, "campaign": self.name, "stage": stage_name,
                    "params": params, "seed": seed, "config_hash": cfg_hash, "label": label,
                    "varying": varying, "run_dir": str(run_dir),
                })
        return runs

    def to_yaml(self) -> str:
        return yaml.safe_dump(self.data, sort_keys=False)
