from __future__ import annotations
import pytest
pytest
from bokeh.server.views.auth_request_handler import AuthRequestHandler
from bokeh.server.views.metadata_handler import MetadataHandler

def test_uses_auth_request_handler() -> None:
    if False:
        print('Hello World!')
    assert issubclass(MetadataHandler, AuthRequestHandler)