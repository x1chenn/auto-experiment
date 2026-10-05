"""Small helpers shared by every module: time, hashing, files, locks, templates."""

from __future__ import annotations

import contextlib
import datetime as _dt
import getpass
import hashlib
import json
import math
import os
import random
import re
import socket
import string
import time
from pathlib import Path
from typing import Any, Dict, Iterator, Optional

# ---------------------------------------------------------------- time


def now() -> str:
    """Local time with UTC offset, second precision (sortable within one zone)."""
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def parse_ts(value: Optional[str]) -> Optional[_dt.datetime]:
    """Parse our own timestamps and Slurm's (naive, local) ones."""
    if not value or value in ("Unknown", "None", "N/A"):
        return None
    try:
        ts = _dt.datetime.fromisoformat(value)
    except ValueError:
        return None
    if ts.tzinfo is None:
        ts = ts.astimezone()
    return ts


def age_seconds(value: Optional[str]) -> Optional[float]:
    ts = parse_ts(value)
    if ts is None:
        return None
    return (_dt.datetime.now().astimezone() - ts).total_seconds()


def today() -> str:
    return _dt.date.today().isoformat()


# ---------------------------------------------------------------- ids and hashes


def stable_hash(obj: Any, n: int = 8) -> str:
    blob = json.dumps(obj, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:n]


def file_hash(path: Path, n: int = 12) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:n]


def new_id(prefix: str) -> str:
    stamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    tail = "".join(random.choices(string.ascii_lowercase + string.digits, k=4))
    return f"{prefix}{stamp}-{tail}"


# ---------------------------------------------------------------- files


def atomic_write(path: Path, text: str, mode: Optional[int] = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w") as fh:
        fh.write(text)
        fh.flush()
        os.fsync(fh.fileno())
    if mode is not None:
        os.chmod(tmp, mode)
    os.replace(tmp, path)


def write_json(path: Path, obj: Any) -> None:
    atomic_write(path, json.dumps(obj, indent=2, sort_keys=True, default=str) + "\n")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


def tail_lines(path: Path, n: int = 200, max_bytes: int = 256 * 1024) -> list:
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - max_bytes))
            data = fh.read().decode("utf-8", errors="replace")
    except OSError:
        return []
    return data.splitlines()[-n:]


def last_json_line(path: Path) -> Optional[dict]:
    """Return the last complete JSON object line of a JSONL file, if any."""
    for line in reversed(tail_lines(path, n=50, max_bytes=64 * 1024)):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            return obj
    return None


@contextlib.contextmanager
def short_lock(path: Path, timeout: float = 60.0, stale: float = 120.0) -> Iterator[None]:
    """Mutual exclusion for a few milliseconds of work, safe across nodes.

    Uses O_CREAT|O_EXCL on a lock file instead of flock: a crashed holder can
    never wedge everyone else, because a lock older than ``stale`` seconds is
    broken. Never hold this around anything slow.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.time() + timeout
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            break
        except FileExistsError:
            try:
                if time.time() - path.stat().st_mtime > stale:
                    path.unlink()
                    continue
            except FileNotFoundError:
                continue
            if time.time() > deadline:
                raise TimeoutError(f"could not acquire {path}")
            time.sleep(0.05 + random.random() * 0.1)
    try:
        os.write(fd, f"{socket.gethostname()} {os.getpid()} {now()}\n".encode())
        os.close(fd)
        yield
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass


# ---------------------------------------------------------------- templates

_PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")


def subst(template: Any, mapping: Dict[str, Any]) -> Any:
    """Replace ``{name}`` for known names only; shell ``${VAR}`` is left alone."""
    if not isinstance(template, str):
        return template

    def repl(m: "re.Match[str]") -> str:
        key = m.group(1)
        return str(mapping[key]) if key in mapping else m.group(0)

    return _PLACEHOLDER.sub(repl, template)


# ---------------------------------------------------------------- units


def parse_mem_mb(value: Any) -> int:
    """'1G' -> 1024, '1500M' -> 1500, 2048 -> 2048 (Slurm default unit is MB)."""
    s = str(value).strip().upper()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)\s*([KMGT]?)B?", s)
    if not m:
        raise ValueError(f"bad memory value {value!r}")
    num, unit = float(m.group(1)), m.group(2) or "M"
    factor = {"K": 1 / 1024, "M": 1, "G": 1024, "T": 1024 * 1024}[unit]
    return int(math.ceil(num * factor))


def parse_time_s(value: Any) -> int:
    """Slurm time formats: MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS."""
    s = str(value).strip()
    days = 0
    if "-" in s:
        d, s = s.split("-", 1)
        days = int(d)
        parts = [int(p) for p in s.split(":")]
        while len(parts) < 3:
            parts.append(0)
        h, m, sec = parts
    else:
        parts = [int(p) for p in s.split(":")]
        if len(parts) == 1:
            h, m, sec = 0, parts[0], 0
        elif len(parts) == 2:
            h, m, sec = 0, parts[0], parts[1]
        else:
            h, m, sec = parts
    return ((days * 24 + h) * 60 + m) * 60 + sec


def format_time_s(seconds: int) -> str:
    seconds = int(seconds)
    d, rem = divmod(seconds, 86400)
    h, rem = divmod(rem, 3600)
    m, s = divmod(rem, 60)
    return f"{d}-{h:02d}:{m:02d}:{s:02d}" if d else f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------- identity


def default_actor() -> Dict[str, str]:
    """Who is acting. Agents set AUTOEXP_SESSION/AGENT/MODEL (hooks do it for them)."""
    actor = {"user": getpass.getuser(), "host": socket.gethostname().split(".")[0]}
    for key, env in (("session", "AUTOEXP_SESSION"), ("agent", "AUTOEXP_AGENT"), ("model", "AUTOEXP_MODEL")):
        if os.environ.get(env):
            actor[key] = os.environ[env]
    if os.environ.get("SLURM_JOB_ID"):
        actor["job"] = os.environ["SLURM_JOB_ID"]
    return actor


def is_finite_number(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False
