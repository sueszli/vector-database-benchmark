""" Provide a Pytest plugin for handling tests when networkx may be missing.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from types import ModuleType
import pytest
from bokeh.util.dependencies import import_optional
__all__ = ('nx',)

@pytest.fixture
def nx() -> ModuleType | None:
    if False:
        while True:
            i = 10
    ' A PyTest fixture that will automatically skip a test if networkx is\n    not installed.\n\n    '
    nx = import_optional('networkx')
    if nx is None:
        pytest.skip('networkx is not installed')
    return nx