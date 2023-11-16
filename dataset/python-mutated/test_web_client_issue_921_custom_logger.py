import unittest
from logging import Logger
from slack.web import WebClient
from tests.slack_sdk.web.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestWebClient_Issue_921_CustomLogger(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        setup_mock_web_api_server(self)

    def tearDown(self):
        if False:
            return 10
        cleanup_mock_web_api_server(self)

    def test_if_it_uses_custom_logger(self):
        if False:
            return 10
        logger = CustomLogger('test-logger')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', logger=logger)
        client.chat_postMessage(channel='C111', text='hello')
        self.assertTrue(logger.called)

class CustomLogger(Logger):
    called: bool

    def __init__(self, name, level='DEBUG'):
        if False:
            for i in range(10):
                print('nop')
        Logger.__init__(self, name, level)
        self.called = False

    def debug(self, msg, *args, **kwargs):
        if False:
            return 10
        self.called = True