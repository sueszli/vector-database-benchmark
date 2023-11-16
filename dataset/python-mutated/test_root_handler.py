from __future__ import annotations
import pytest
pytest
from bokeh.server.views.auth_request_handler import AuthRequestHandler
from bokeh.server.views.root_handler import RootHandler

def test_uses_auth_request_handler() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert issubclass(RootHandler, AuthRequestHandler)