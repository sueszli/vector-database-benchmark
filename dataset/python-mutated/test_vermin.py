from __future__ import annotations
import pytest
pytest
from os import chdir
from subprocess import run
import toml
from tests.support.util.project import TOP_PATH

def test_vermin() -> None:
    if False:
        print('Hello World!')
    chdir(TOP_PATH)
    pyproject = toml.load(TOP_PATH / 'pyproject.toml')
    minpy = pyproject['project']['requires-python'].lstrip('>=')
    cmd = f'vermin --eval-annotations --no-tips -t={minpy} -vvv --lint src/bokeh'.split()
    proc = run(cmd, capture_output=True)
    assert proc.returncode == 0, f"vermin issues:\n{proc.stdout.decode('utf-8')}"