""" Unit tests for websocketserver """
import unittest
from unittest.mock import patch, MagicMock
from websockify.websocketserver import HttpWebSocket

class HttpWebSocketTest(unittest.TestCase):

    @patch('websockify.websocketserver.WebSocket.__init__', autospec=True)
    def test_constructor(self, websock):
        if False:
            i = 10
            return i + 15
        req_obj = MagicMock()
        sock = HttpWebSocket(req_obj)
        websock.assert_called_once_with(sock)
        self.assertEqual(sock.request_handler, req_obj)

    @patch('websockify.websocketserver.WebSocket.__init__', MagicMock(autospec=True))
    def test_send_response(self):
        if False:
            print('Hello World!')
        req_obj = MagicMock()
        sock = HttpWebSocket(req_obj)
        sock.send_response(200, 'message')
        req_obj.send_response.assert_called_once_with(200, 'message')

    @patch('websockify.websocketserver.WebSocket.__init__', MagicMock(autospec=True))
    def test_send_response_default_message(self):
        if False:
            for i in range(10):
                print('nop')
        req_obj = MagicMock()
        sock = HttpWebSocket(req_obj)
        sock.send_response(200)
        req_obj.send_response.assert_called_once_with(200, None)

    @patch('websockify.websocketserver.WebSocket.__init__', MagicMock(autospec=True))
    def test_send_header(self):
        if False:
            for i in range(10):
                print('nop')
        req_obj = MagicMock()
        sock = HttpWebSocket(req_obj)
        sock.send_header('keyword', 'value')
        req_obj.send_header.assert_called_once_with('keyword', 'value')

    @patch('websockify.websocketserver.WebSocket.__init__', MagicMock(autospec=True))
    def test_end_headers(self):
        if False:
            for i in range(10):
                print('nop')
        req_obj = MagicMock()
        sock = HttpWebSocket(req_obj)
        sock.end_headers()
        req_obj.end_headers.assert_called_once_with()