import os
import unittest
import pytest
from slack_sdk.web import WebClient
from tests.slack_sdk.web.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestWebClient(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        setup_mock_web_api_server(self)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        cleanup_mock_web_api_server(self)

    @pytest.mark.skip()
    def test_stars_deprecation(self):
        if False:
            return 10
        env_value = os.environ.get('SLACKCLIENT_SKIP_DEPRECATION')
        try:
            os.environ.pop('SLACKCLIENT_SKIP_DEPRECATION')
            client = WebClient(base_url='http://localhost:8888')
            client.stars_list(token='xoxb-api_test')
        finally:
            if env_value is not None:
                os.environ.update({'SLACKCLIENT_SKIP_DEPRECATION': env_value})