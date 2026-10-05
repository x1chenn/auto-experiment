import os
import sys
import tempfile
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from autoexp.config import load_config  # noqa: E402


class TempHome:
    """A throwaway AUTOEXP_HOME with an optional config."""

    def __init__(self, config: str = ""):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name)
        self.home = self.path / "home"
        self.home.mkdir()
        if config:
            (self.home / "config.yaml").write_text(textwrap.dedent(config))
        self._old = os.environ.get("AUTOEXP_HOME")
        os.environ["AUTOEXP_HOME"] = str(self.home)
        os.environ.pop("AUTOEXP_SHARED", None)
        os.environ.pop("AUTOEXP_SESSION", None)

    def cfg(self):
        cfg = load_config(self.home)
        cfg.ensure_dirs()
        return cfg

    def write(self, name: str, text: str) -> Path:
        p = self.path / name
        p.write_text(textwrap.dedent(text))
        return p

    def close(self):
        if self._old is None:
            os.environ.pop("AUTOEXP_HOME", None)
        else:
            os.environ["AUTOEXP_HOME"] = self._old
        self.tmp.cleanup()


SIMPLE_SPEC = """\
name: demo
test: true
run_root: ./runs
command: "echo {lr} {beta} {seed} > {run_dir}/out.txt"
params:
  lr: [0.1, 0.2]
  beta: [1]
seeds: [0, 1]
resources: {partition: cpu, mem: 1G, time: "00:10:00"}
stages:
  - {name: smoke, seeds: [0], auto: true}
  - {name: pilot, auto: true}
  - {name: full, auto: false}
contract:
  echo_file: echo.json
  echo_keys: [lr, beta]
retry: {infra: 2}
"""
