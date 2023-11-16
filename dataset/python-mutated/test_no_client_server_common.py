from __future__ import annotations
import pytest
pytest
from subprocess import run
from sys import executable as python
from tests.support.util.project import verify_clean_imports
modules = ['bokeh.embed', 'bokeh.io', 'bokeh.models', 'bokeh.plotting']

def test_no_client_common() -> None:
    if False:
        for i in range(10):
            print('nop')
    ' Basic usage of Bokeh should not result in any client code being\n    imported. This test ensures that importing basic modules does not bring in\n    bokeh.client.\n\n    '
    proc = run([python, '-c', verify_clean_imports('bokeh.client', modules)])
    assert proc.returncode == 0, 'bokeh.client imported in common modules'

def test_no_server_common() -> None:
    if False:
        for i in range(10):
            print('nop')
    ' Basic usage of Bokeh should not result in any server code being\n    imported. This test ensures that importing basic modules does not bring in\n    bokeh.server.\n\n    '
    proc = run([python, '-c', verify_clean_imports('bokeh.server', modules)])
    assert proc.returncode == 0, 'bokeh.server imported in common modules'