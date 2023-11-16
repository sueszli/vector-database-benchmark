from __future__ import annotations
import pytest
pytest
import logging
from tornado.websocket import WebSocketClosedError
from bokeh.server.views.auth_request_handler import AuthRequestHandler
from bokeh.util.logconfig import basicConfig
from bokeh.server.views.ws import WSHandler
basicConfig()

async def test_send_message_raises(caplog: pytest.LogCaptureFixture) -> None:

    class ExcMessage:

        def send(self, handler):
            if False:
                while True:
                    i = 10
            raise WebSocketClosedError()
    with caplog.at_level(logging.WARN):
        ret = await WSHandler.send_message('self', ExcMessage())
        assert ret is None
        assert len([msg for msg in caplog.messages if 'Failed sending message as connection was closed' in msg]) == 1

def test_uses_auth_request_handler() -> None:
    if False:
        while True:
            i = 10
    assert issubclass(WSHandler, AuthRequestHandler)