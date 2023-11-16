from __future__ import annotations
import pytest
pytest
from bokeh.server.views.auth_request_handler import AuthRequestHandler
from bokeh.server.views.session_handler import SessionHandler

def test_uses_auth_request_handler() -> None:
    if False:
        for i in range(10):
            print('nop')
    assert issubclass(SessionHandler, AuthRequestHandler)