from __future__ import annotations
import pytest
pytest
from subprocess import PIPE, Popen
from sys import executable as python
from typing import Sequence
from tests.support.util.project import ls_modules
SKIP: Sequence[str] = []

def test_python_execution_with_OO() -> None:
    if False:
        while True:
            i = 10
    ' Running python with -OO will discard docstrings (__doc__ is None)\n    which can cause problems if docstrings are naively formatted.\n\n    This test ensures that the all modules are importable, even with -OO set.\n\n    If you encounter a new problem with docstrings being formatted, try\n    using format_docstring.\n    '
    imports = [f'import {mod}' for mod in ls_modules(skip_prefixes=SKIP)]
    proc = Popen([python, '-OO', '-'], stdout=PIPE, stdin=PIPE)
    proc.communicate('\n'.join(imports).encode('utf-8'))
    proc.wait()
    assert proc.returncode == 0, 'Execution with -OO failed'