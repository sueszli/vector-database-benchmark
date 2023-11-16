""" Provide a Pytest plugin for handling tests when IPython may be missing.

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
from types import ModuleType
import pytest
from bokeh.util.dependencies import import_optional
__all__ = ('ipython',)

@pytest.fixture
def ipython() -> ModuleType | None:
    if False:
        while True:
            i = 10
    ' A PyTest fixture that will automatically skip a test if IPython is\n    not installed.\n\n    '
    ipython = import_optional('IPython')
    if ipython is None:
        pytest.skip('IPython is not installed')
    return ipython