import unittest
from slack_sdk.audit_logs import AuditLogsClient
from slack_sdk.http_retry import RateLimitErrorRetryHandler
from tests.slack_sdk.audit_logs.mock_web_api_server import cleanup_mock_web_api_server, setup_mock_web_api_server
from ..my_retry_handler import MyRetryHandler

class TestAuditLogsClient_HttpRetries(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        setup_mock_web_api_server(self)

    def tearDown(self):
        if False:
            while True:
                i = 10
        cleanup_mock_web_api_server(self)

    def test_retries(self):
        if False:
            while True:
                i = 10
        retry_handler = MyRetryHandler(max_retry_count=2)
        client = AuditLogsClient(token='xoxp-remote_disconnected', base_url='http://localhost:8888/', retry_handlers=[retry_handler])
        try:
            client.actions()
            self.fail('An exception is expected')
        except Exception as _:
            pass
        self.assertEqual(2, retry_handler.call_count)

    def test_ratelimited(self):
        if False:
            for i in range(10):
                print('nop')
        client = AuditLogsClient(token='xoxp-ratelimited', base_url='http://localhost:8888/')
        client.retry_handlers.append(RateLimitErrorRetryHandler())
        response = client.actions()
        self.assertEqual(429, response.status_code)