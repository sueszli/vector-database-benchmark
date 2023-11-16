from __future__ import annotations
import sys
from piptools.subprocess_utils import run_python_snippet

def test_run_python_snippet_returns_multilne():
    if False:
        for i in range(10):
            print('nop')
    result = run_python_snippet(sys.executable, 'print("MULTILINE\\nOUTPUT", end="")')
    assert result == 'MULTILINE\nOUTPUT'