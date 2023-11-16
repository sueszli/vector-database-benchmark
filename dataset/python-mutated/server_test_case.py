import urllib.parse
from unittest import mock
import tornado.testing
import tornado.web
import tornado.websocket
from tornado.websocket import WebSocketClientConnection
from streamlit.proto.ForwardMsg_pb2 import ForwardMsg
from streamlit.runtime import Runtime
from streamlit.runtime.app_session import AppSession
from streamlit.web.server import Server

class ServerTestCase(tornado.testing.AsyncHTTPTestCase):
    """Base class for async streamlit.server testing.

    Subclasses should patch 'streamlit.server.server.AppSession',
    to prevent AppSessions from being created, and scripts from
    being run. (Script running involves creating new threads, which
    interfere with other tests if not properly terminated.)

    See the "ServerTest" class for example usage.
    """
    _next_session_id = 0

    def tearDown(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        Runtime._instance = None

    def get_app(self) -> tornado.web.Application:
        if False:
            i = 10
            return i + 15
        self.server = Server('/not/a/script.py', 'test command line')
        app = self.server._create_app()
        return app

    def get_ws_url(self, path):
        if False:
            while True:
                i = 10
        'Return a ws:// URL with the given path for our test server.'
        url = self.get_url(path)
        parts = list(urllib.parse.urlparse(url))
        parts[0] = 'ws'
        return urllib.parse.urlunparse(tuple(parts))

    async def ws_connect(self, existing_session_id=None) -> WebSocketClientConnection:
        """Open a websocket connection to the server.

        Returns
        -------
        WebSocketClientConnection
            The connected websocket client.
        """
        if existing_session_id is None:
            subprotocols = ['streamlit', 'PLACEHOLDER_AUTH_TOKEN']
        else:
            subprotocols = ['streamlit', 'PLACEHOLDER_AUTH_TOKEN', existing_session_id]
        return await tornado.websocket.websocket_connect(self.get_ws_url('/_stcore/stream'), subprotocols=subprotocols)

    async def read_forward_msg(self, ws_client: WebSocketClientConnection) -> ForwardMsg:
        """Parse the next message from a Websocket client into a ForwardMsg."""
        data = await ws_client.read_message()
        message = ForwardMsg()
        message.ParseFromString(data)
        return message

    @staticmethod
    def _create_mock_app_session(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Create a mock AppSession. Each mocked instance will have\n        its own unique ID.'
        mock_id = mock.PropertyMock(return_value=f'mock_id:{ServerTestCase._next_session_id}')
        ServerTestCase._next_session_id += 1
        mock_session = mock.MagicMock(AppSession, *args, autospec=True, **kwargs)
        type(mock_session).id = mock_id
        return mock_session

    def _patch_app_session(self):
        if False:
            i = 10
            return i + 15
        "Mock the Server's AppSession import. We don't want\n        actual sessions to be instantiated, or scripts to be run.\n        "
        return mock.patch('streamlit.runtime.websocket_session_manager.AppSession', new_callable=lambda : self._create_mock_app_session)