import unittest
import slack
from tests.web.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestWebClientFunctional(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        setup_mock_web_api_server(self)
        self.client = slack.WebClient(token='xoxb-api_test', base_url='http://localhost:8888')

    def tearDown(self):
        if False:
            return 10
        cleanup_mock_web_api_server(self)

    def test_requests_with_use_session_turned_off(self):
        if False:
            return 10
        self.client.use_pooling = False
        resp = self.client.api_test()
        assert resp['ok']

    def test_subsequent_requests_with_a_session_succeeds(self):
        if False:
            print('Hello World!')
        resp = self.client.api_test()
        assert resp['ok']
        resp = self.client.api_test()
        assert resp['ok']