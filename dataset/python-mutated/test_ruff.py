from __future__ import annotations
import pytest
pytest
from os import chdir
from subprocess import run
from tests.support.util.project import TOP_PATH

def test_ruff() -> None:
    if False:
        for i in range(10):
            print('nop')
    chdir(TOP_PATH)
    proc = run(['ruff', '.'], capture_output=True)
    assert proc.returncode == 0, f"ruff issues:\n{proc.stdout.decode('utf-8')}"