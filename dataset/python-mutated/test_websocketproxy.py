""" Unit tests for websocketproxy """
import sys
import unittest
import unittest
import socket
from io import StringIO
from io import BytesIO
from unittest.mock import patch, MagicMock
from websockify import websocketproxy
from websockify import token_plugins
from websockify import auth_plugins

class FakeSocket(object):

    def __init__(self, data=b''):
        if False:
            while True:
                i = 10
        self._data = data

    def recv(self, amt, flags=None):
        if False:
            i = 10
            return i + 15
        res = self._data[0:amt]
        if not flags & socket.MSG_PEEK:
            self._data = self._data[amt:]
        return res

    def makefile(self, mode='r', buffsize=None):
        if False:
            for i in range(10):
                print('nop')
        if 'b' in mode:
            return BytesIO(self._data)
        else:
            return StringIO(self._data.decode('latin_1'))

class FakeServer(object):

    class EClose(Exception):
        pass

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.token_plugin = None
        self.auth_plugin = None
        self.wrap_cmd = None
        self.ssl_target = None
        self.unix_target = None

class ProxyRequestHandlerTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(ProxyRequestHandlerTestCase, self).setUp()
        self.handler = websocketproxy.ProxyRequestHandler(FakeSocket(), '127.0.0.1', FakeServer())
        self.handler.path = 'https://localhost:6080/websockify?token=blah'
        self.handler.headers = None
        patch('websockify.websockifyserver.WebSockifyServer.socket').start()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        patch.stopall()
        super(ProxyRequestHandlerTestCase, self).tearDown()

    def test_get_target(self):
        if False:
            print('Hello World!')

        class TestPlugin(token_plugins.BasePlugin):

            def lookup(self, token):
                if False:
                    return 10
                return ('some host', 'some port')
        (host, port) = self.handler.get_target(TestPlugin(None))
        self.assertEqual(host, 'some host')
        self.assertEqual(port, 'some port')

    def test_get_target_unix_socket(self):
        if False:
            i = 10
            return i + 15

        class TestPlugin(token_plugins.BasePlugin):

            def lookup(self, token):
                if False:
                    return 10
                return ('unix_socket', '/tmp/socket')
        (_, socket) = self.handler.get_target(TestPlugin(None))
        self.assertEqual(socket, '/tmp/socket')

    def test_get_target_raises_error_on_unknown_token(self):
        if False:
            return 10

        class TestPlugin(token_plugins.BasePlugin):

            def lookup(self, token):
                if False:
                    i = 10
                    return i + 15
                return None
        with self.assertRaises(FakeServer.EClose):
            self.handler.get_target(TestPlugin(None))

    @patch('websockify.websocketproxy.ProxyRequestHandler.send_auth_error', MagicMock())
    def test_token_plugin(self):
        if False:
            i = 10
            return i + 15

        class TestPlugin(token_plugins.BasePlugin):

            def lookup(self, token):
                if False:
                    i = 10
                    return i + 15
                return (self.source + token).split(',')
        self.handler.server.token_plugin = TestPlugin('somehost,')
        self.handler.validate_connection()
        self.assertEqual(self.handler.server.target_host, 'somehost')
        self.assertEqual(self.handler.server.target_port, 'blah')

    @patch('websockify.websocketproxy.ProxyRequestHandler.send_auth_error', MagicMock())
    def test_auth_plugin(self):
        if False:
            i = 10
            return i + 15

        class TestPlugin(auth_plugins.BasePlugin):

            def authenticate(self, headers, target_host, target_port):
                if False:
                    for i in range(10):
                        print('nop')
                if target_host == self.source:
                    raise auth_plugins.AuthenticationError(response_msg='some_error')
        self.handler.server.auth_plugin = TestPlugin('somehost')
        self.handler.server.target_host = 'somehost'
        self.handler.server.target_port = 'someport'
        with self.assertRaises(auth_plugins.AuthenticationError):
            self.handler.auth_connection()
        self.handler.server.target_host = 'someotherhost'
        self.handler.auth_connection()