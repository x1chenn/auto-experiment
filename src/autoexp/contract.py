"""Artifact contracts: what "this run succeeded" means, independent of exit codes.

A contract can require output files, metric keys in a JSONL metrics file, a
minimum final step, an echo of the resolved configuration that must match
what was asked for, and finite values in a result file. Exit code 0 with a
broken contract is a failure; a launcher that swallows errors cannot fool it.
"""

from __future__ import annotations

import glob
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from .util import is_finite_number, read_json, subst


def _values_equal(expected: Any, actual: Any) -> bool:
    if is_finite_number(expected) and is_finite_number(actual) and not isinstance(expected, bool):
        a, b = float(expected), float(actual)
        return a == b or math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12)
    return str(expected) == str(actual)


def scan_metrics(path: Path, step_key: str = "step") -> Dict[str, Any]:
    """Keys seen and last step in a JSONL metrics file (tolerates a torn last line)."""
    keys, last_step, lines = set(), None, 0
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(rec, dict):
                    continue
                lines += 1
                keys.update(rec)
                if step_key in rec and is_finite_number(rec[step_key]):
                    last_step = rec[step_key]
    except OSError:
        return {"exists": False, "keys": [], "last_step": None, "lines": 0}
    return {"exists": True, "keys": sorted(keys), "last_step": last_step, "lines": lines}


def check(run_dir: Path, contract: Dict[str, Any], mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Evaluate a contract. Returns ``{ok, failures, kind, ...}`` where ``kind`` is
    ``ok``, ``config_mismatch``, ``diverged`` or ``contract``."""
    run_dir = Path(run_dir)
    failures: List[str] = []
    kinds: List[str] = []
    report: Dict[str, Any] = {}

    for pattern in contract.get("files", []) or []:
        pat = str(run_dir / subst(pattern, mapping))
        if not glob.glob(pat):
            failures.append(f"missing file {subst(pattern, mapping)}")
            kinds.append("contract")

    mfile = contract.get("metrics_file")
    if mfile:
        step_key = contract.get("step_key", "step")
        scan = scan_metrics(run_dir / subst(mfile, mapping), step_key)
        report["metrics"] = {k: scan[k] for k in ("exists", "last_step", "lines")}
        if not scan["exists"]:
            failures.append(f"missing metrics file {mfile}")
            kinds.append("contract")
        for key in contract.get("metric_keys", []) or []:
            if scan["exists"] and key not in scan["keys"]:
                failures.append(f"metric '{key}' never logged")
                kinds.append("contract")
        need = contract.get("final_step_at_least")
        if need is not None and scan["exists"]:
            need_v = float(subst(need, mapping))
            if scan["last_step"] is None or float(scan["last_step"]) < need_v:
                failures.append(f"last step {scan['last_step']} < required {need_v:g}")
                kinds.append("contract")

    efile = contract.get("echo_file")
    if efile:
        echo = read_json(run_dir / subst(efile, mapping))
        report["echo"] = echo
        if not isinstance(echo, dict):
            failures.append(f"missing or invalid echo file {efile}")
            kinds.append("contract")
        else:
            for key in contract.get("echo_keys", []) or []:
                if key not in echo:
                    failures.append(f"echo lacks '{key}' (cannot prove it was applied)")
                    kinds.append("config_mismatch")
                elif key in mapping and not _values_equal(mapping[key], echo[key]):
                    failures.append(f"echo {key}={echo[key]!r} but run asked for {mapping[key]!r}")
                    kinds.append("config_mismatch")

    finite = contract.get("finite")
    if finite:
        fpath = run_dir / subst(finite.get("file", ""), mapping)
        data = read_json(fpath)
        if not isinstance(data, dict):
            failures.append(f"missing or invalid {finite.get('file')}")
            kinds.append("contract")
        else:
            for key in finite.get("keys", []) or []:
                if key not in data:
                    failures.append(f"{finite.get('file')} lacks '{key}'")
                    kinds.append("contract")
                elif not is_finite_number(data[key]):
                    failures.append(f"{key}={data[key]!r} is not finite")
                    kinds.append("diverged")

    if not failures:
        kind = "ok"
    elif "config_mismatch" in kinds:
        kind = "config_mismatch"
    elif "diverged" in kinds:
        kind = "diverged"
    else:
        kind = "contract"
    report.update(ok=not failures, failures=failures, kind=kind)
    return report


def detect_noops(runs: List[Dict[str, Any]], echoes: Dict[str, Optional[dict]]) -> List[Dict[str, Any]]:
    """Differential check across arms of one stage.

    For every parameter that varies between runs, the echoed configuration must
    vary too. If arms that asked for different values echo identical configs,
    the parameter was most likely ignored somewhere between the command line
    and the code (a typo'd flag, a renamed key, a wrong default).
    """
    findings = []
    varying = set()
    for r in runs:
        varying.update(r.get("varying") or [])
    for param in sorted(varying):
        asked: Dict[str, set] = {}
        for r in runs:
            echo = echoes.get(r["run_id"])
            if not isinstance(echo, dict):
                continue
            value = json.dumps(r["params"].get(param), sort_keys=True, default=str)
            if param in echo:
                seen = json.dumps(echo[param], sort_keys=True, default=str)
            else:
                others = {k: v for k, v in echo.items() if k != "seed"}
                seen = json.dumps(others, sort_keys=True, default=str)
            asked.setdefault(value, set()).add(seen)
        if len(asked) < 2:
            continue
        all_seen = set().union(*asked.values())
        if len(all_seen) == 1:
            findings.append({"param": param, "asked": sorted(asked), "echoed": sorted(all_seen)})
    return findings
