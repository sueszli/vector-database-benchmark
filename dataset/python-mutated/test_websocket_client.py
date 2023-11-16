import unittest
from slack_sdk.socket_mode.websocket_client import SocketModeClient

class TestWebSocketClientLibrary(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        pass

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def test_init_close(self):
        if False:
            print('Hello World!')
        client = SocketModeClient(app_token='xapp-A111-222-xyz')
        try:
            self.assertIsNotNone(client)
            self.assertFalse(client.is_connected())
        finally:
            client.close()