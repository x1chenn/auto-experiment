#!/usr/bin/env python3
"""Scan files that would be published for secrets and private identifiers.

Generic patterns (tokens, private keys, home/scratch paths, e-mail addresses)
are built in. Private terms (names, account names, project names, hostnames)
are read from a file OUTSIDE the repository, given by --terms or
$AUTOEXP_DENY_TERMS (default ~/.autoexp/deny_terms.txt), one term per line,
case-insensitive. Keeping that list out of the repo matters: a published
deny-list is itself a leak.

Usage:
    python tools/leakcheck.py            # scan files tracked or staged by git
    python tools/leakcheck.py --all      # scan every file under the repo
Exit code 1 if anything is found.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

GENERIC = [
    ("anthropic key", r"sk-ant-[A-Za-z0-9_-]{10,}"),
    ("openai key", r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}"),
    ("github token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}"),
    ("aws key", r"\bAKIA[0-9A-Z]{16}\b"),
    ("private key", r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    ("oauth token assignment", r"(?i)(?:OAUTH_TOKEN|API_KEY|SECRET|PASSWORD)\s*[=:]\s*['\"]?[A-Za-z0-9_\-./+]{12,}"),
    ("absolute home/scratch path", r"(?<![\w.])/(?:home|scratch|data|gpfs|lustre|mnt|nfs)\d*/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+"),
    ("e-mail address", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(?:edu|com|org|net|io|ai)\b"),
]
ALLOWED_EMAILS = {"noreply@anthropic.com"}
SKIP_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".pdf", ".zip", ".gz", ".pyc"}


def repo_files(root: Path, all_files: bool) -> list:
    if all_files:
        return [p for p in root.rglob("*") if p.is_file() and ".git" not in p.parts]
    out = subprocess.run(["git", "-C", str(root), "ls-files", "--cached", "--others", "--exclude-standard"],
                         capture_output=True, text=True, check=True).stdout.split()
    return [root / f for f in out]


def load_terms(path: Path) -> list:
    if not path.exists():
        return []
    terms = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            terms.append(line)
    return terms


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--terms", default=os.environ.get("AUTOEXP_DENY_TERMS", "~/.autoexp/deny_terms.txt"))
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    terms_path = Path(args.terms).expanduser()
    terms = load_terms(terms_path)
    if not terms:
        print(f"warning: no private terms loaded from {terms_path}; only generic patterns are checked")
    patterns = [(name, re.compile(rx)) for name, rx in GENERIC]
    patterns += [(f"private term #{i + 1}", re.compile(re.escape(t), re.IGNORECASE)) for i, t in enumerate(terms)]
    hits = 0
    for path in repo_files(root, args.all):
        if path.suffix.lower() in SKIP_SUFFIXES or not path.is_file():
            continue
        try:
            text = path.read_text(errors="replace")
        except OSError:
            continue
        for lineno, line in enumerate(text.splitlines(), 1):
            for name, rx in patterns:
                for m in rx.finditer(line):
                    if name == "e-mail address" and (m.group(0) in ALLOWED_EMAILS
                                                     or re.search(r"@example\.(org|com|net)$", m.group(0))):
                        continue
                    hits += 1
                    # never echo the private term itself, only where it is
                    shown = m.group(0) if not name.startswith("private term") else "<redacted>"
                    print(f"{path.relative_to(root)}:{lineno}: {name}: {shown}")
    print(f"{hits} finding(s)" if hits else "clean")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main())
