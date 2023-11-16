from __future__ import annotations
import pytest
pytest
from unittest.mock import MagicMock, patch
import bokeh.client.session as bcs

def test_DEFAULT_SESSION_ID() -> None:
    if False:
        return 10
    assert bcs.DEFAULT_SESSION_ID == 'default'

def test_DEFAULT_SERVER_WEBSOCKET_URL() -> None:
    if False:
        i = 10
        return i + 15
    assert bcs.DEFAULT_SERVER_WEBSOCKET_URL == 'ws://localhost:5006/ws'

class Test_ClientSession:

    def test_creation_defaults(self) -> None:
        if False:
            while True:
                i = 10
        s = bcs.ClientSession()
        assert s.connected is False
        assert s.document is None
        assert s._connection._arguments is None
        assert isinstance(s.id, str)
        assert len(s.id) == 44

    def test_creation_with_session_id(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        s = bcs.ClientSession('sid')
        assert s.connected is False
        assert s.document is None
        assert s._connection._arguments is None
        assert s.id == 'sid'

    def test_creation_with_ws_url(self) -> None:
        if False:
            while True:
                i = 10
        s = bcs.ClientSession(websocket_url='wsurl')
        assert s.connected is False
        assert s.document is None
        assert s._connection._arguments is None
        assert s._connection.url == 'wsurl'
        assert isinstance(s.id, str)
        assert len(s.id) == 44

    def test_creation_with_ioloop(self) -> None:
        if False:
            i = 10
            return i + 15
        s = bcs.ClientSession(io_loop='io_loop')
        assert s.connected is False
        assert s.document is None
        assert s._connection._arguments is None
        assert s._connection.io_loop == 'io_loop'
        assert isinstance(s.id, str)
        assert len(s.id) == 44

    def test_creation_with_arguments(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        s = bcs.ClientSession(arguments='args')
        assert s.connected is False
        assert s.document is None
        assert s._connection._arguments == 'args'
        assert len(s.id) == 44

    @patch('bokeh.client.connection.ClientConnection.connect')
    def test_connect(self, mock_connect: MagicMock) -> None:
        if False:
            return 10
        s = bcs.ClientSession()
        s.connect()
        assert mock_connect.call_count == 1
        assert mock_connect.call_args[0] == ()
        assert mock_connect.call_args[1] == {}

    @patch('bokeh.client.connection.ClientConnection.close')
    def test_close(self, mock_close: MagicMock) -> None:
        if False:
            for i in range(10):
                print('nop')
        s = bcs.ClientSession()
        s.close()
        assert mock_close.call_count == 1
        assert mock_close.call_args[0] == ('closed',)
        assert mock_close.call_args[1] == {}

    @patch('bokeh.client.connection.ClientConnection.close')
    def test_context_manager(self, mock_close: MagicMock) -> None:
        if False:
            return 10
        with bcs.ClientSession() as session:
            assert isinstance(session, bcs.ClientSession)
        assert mock_close.call_count == 1
        assert mock_close.call_args[0] == ('closed',)
        assert mock_close.call_args[1] == {}

    @patch('bokeh.client.connection.ClientConnection.close')
    def test_close_with_why(self, mock_close: MagicMock) -> None:
        if False:
            while True:
                i = 10
        s = bcs.ClientSession()
        s.close('foo')
        assert mock_close.call_count == 1
        assert mock_close.call_args[0] == ('foo',)
        assert mock_close.call_args[1] == {}

    @patch('bokeh.client.connection.ClientConnection.force_roundtrip')
    def test_force_roundtrip(self, mock_force_roundtrip: MagicMock) -> None:
        if False:
            while True:
                i = 10
        s = bcs.ClientSession()
        s.force_roundtrip()
        assert mock_force_roundtrip.call_count == 1
        assert mock_force_roundtrip.call_args[0] == ()
        assert mock_force_roundtrip.call_args[1] == {}

    @patch('bokeh.client.connection.ClientConnection.request_server_info')
    def test_request_server_info(self, mock_request_server_info: MagicMock) -> None:
        if False:
            print('Hello World!')
        s = bcs.ClientSession()
        s.request_server_info()
        assert mock_request_server_info.call_count == 1
        assert mock_request_server_info.call_args[0] == ()
        assert mock_request_server_info.call_args[1] == {}