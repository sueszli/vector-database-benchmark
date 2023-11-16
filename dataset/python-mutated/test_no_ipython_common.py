from __future__ import annotations
import pytest
pytest
from subprocess import run
from sys import executable as python
from tests.support.util.project import ls_modules, verify_clean_imports
IPYTHON_ALLOWED = ('tests.support',)
MODULES = ls_modules(skip_prefixes=IPYTHON_ALLOWED)

def test_no_ipython_common_combined() -> None:
    if False:
        for i in range(10):
            print('nop')
    ' Basic usage of Bokeh should not result in any IPython code being\n    imported. This test ensures that importing basic modules does not bring in\n    IPython.\n\n    '
    proc = run([python, '-c', verify_clean_imports('IPython', MODULES)])
    assert proc.returncode == 0, 'IPython imported in common modules'