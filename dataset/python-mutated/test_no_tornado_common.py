from __future__ import annotations
import pytest
pytest
from subprocess import run
from sys import executable as python
from tests.support.util.project import ls_modules, verify_clean_imports
TORNADO_ALLOWED = ('tests.support', 'bokeh.client', 'bokeh.command', 'bokeh.io.notebook', 'bokeh.server', 'bokeh.util.tornado')
MODULES = ls_modules(skip_prefixes=TORNADO_ALLOWED)

def test_no_tornado_common_combined() -> None:
    if False:
        print('Hello World!')
    ' Basic usage of Bokeh should not result in any Tornado code being\n    imported. This test ensures that importing basic modules does not bring in\n    Tornado.\n\n    '
    proc = run([python, '-c', verify_clean_imports('tornado', MODULES)])
    assert proc.returncode == 0, 'Tornado imported in collective common modules'