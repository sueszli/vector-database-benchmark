import unittest
from slack_sdk.rtm_v2 import RTMClient
from slack_sdk import errors as e
from tests.rtm.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestRTMClient(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        setup_mock_web_api_server(self)
        self.rtm = RTMClient(token='xoxp-1234', base_url='http://localhost:8888', auto_reconnect_enabled=False)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        cleanup_mock_web_api_server(self)

    def test_run_on_returns_callback(self):
        if False:
            i = 10
            return i + 15

        def fn1(client, payload):
            if False:
                print('Hello World!')
            pass

        @self.rtm.on('message')
        def fn2(client, payload):
            if False:
                i = 10
                return i + 15
            pass
        self.assertIsNotNone(fn1)
        self.assertIsNotNone(fn2)
        self.assertEqual(fn2.__name__, 'fn2')

    def test_run_on_annotation_sets_callbacks(self):
        if False:
            while True:
                i = 10

        @self.rtm.on('message')
        def say_run_on(client, payload):
            if False:
                i = 10
                return i + 15
            pass
        self.assertTrue(len(self.rtm.message_listeners) == 2)

    def test_on_sets_callbacks(self):
        if False:
            return 10

        def say_on(client, payload):
            if False:
                print('Hello World!')
            pass
        self.rtm.on('message')(say_on)
        self.assertTrue(len(self.rtm.message_listeners) == 2)

    def test_on_accepts_a_list_of_callbacks(self):
        if False:
            while True:
                i = 10

        def say_on(client, payload):
            if False:
                while True:
                    i = 10
            pass

        def say_off(client, payload):
            if False:
                print('Hello World!')
            pass
        self.rtm.on('message')(say_on)
        self.rtm.on('message')(say_off)
        self.assertEqual(len(self.rtm.message_listeners), 3)

    def test_on_raises_when_not_callable(self):
        if False:
            for i in range(10):
                print('nop')
        invalid_callback = 'a'
        with self.assertRaises(e.SlackClientError) as context:
            self.rtm.on('message')(invalid_callback)
        expected_error = "The listener 'a' is not a Callable (actual: str)"
        error = str(context.exception)
        self.assertIn(expected_error, error)

    def test_on_raises_when_kwargs_not_accepted(self):
        if False:
            for i in range(10):
                print('nop')

        def invalid_cb():
            if False:
                while True:
                    i = 10
            pass
        with self.assertRaises(e.SlackClientError) as context:
            self.rtm.on('message')(invalid_cb)
        error = str(context.exception)
        self.assertIn("The listener 'invalid_cb' must accept two args: client, event (actual: )", error)

    def test_send_over_websocket_raises_when_not_connected(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(e.SlackClientError) as context:
            self.rtm.send(payload={})
        expected_error = 'The RTM client is not connected to the Slack servers'
        error = str(context.exception)
        self.assertIn(expected_error, error)

    def test_start_raises_an_error_if_rtm_ws_url_is_not_returned(self):
        if False:
            return 10
        with self.assertRaises(e.SlackApiError) as context:
            RTMClient(token='xoxp-1234', auto_reconnect_enabled=False).start()
        expected_error = "The request to the Slack API failed. (url: https://www.slack.com/api/auth.test)\nThe server responded with: {'ok': False, 'error': 'invalid_auth'}"
        self.assertIn(expected_error, str(context.exception))