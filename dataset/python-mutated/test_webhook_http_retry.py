import unittest
from slack_sdk.http_retry import RateLimitErrorRetryHandler
from slack_sdk.webhook import WebhookClient
from tests.slack_sdk.webhook.mock_web_api_server import cleanup_mock_web_api_server, setup_mock_web_api_server
from ..my_retry_handler import MyRetryHandler

class TestWebhook_HttpRetries(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        setup_mock_web_api_server(self)

    def tearDown(self):
        if False:
            print('Hello World!')
        cleanup_mock_web_api_server(self)

    def test_send(self):
        if False:
            print('Hello World!')
        retry_handler = MyRetryHandler(max_retry_count=2)
        client = WebhookClient('http://localhost:8888/remote_disconnected', retry_handlers=[retry_handler])
        try:
            client.send(text='hello!')
            self.fail('An exception is expected')
        except Exception as _:
            pass
        self.assertEqual(2, retry_handler.call_count)

    def test_ratelimited(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebhookClient('http://localhost:8888/ratelimited')
        client.retry_handlers.append(RateLimitErrorRetryHandler())
        response = client.send(text='hello!')
        self.assertEqual(429, response.status_code)