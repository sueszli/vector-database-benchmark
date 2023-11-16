import os
import unittest
from integration_tests.env_variable_names import SLACK_SDK_TEST_BOT_TOKEN
from integration_tests.helpers import async_test
from slack_sdk.errors import SlackApiError
from slack_sdk.web import WebClient
from slack_sdk.web.async_client import AsyncWebClient

class TestWebClient(unittest.TestCase):
    """Runs integration tests with real Slack API

    export SLACK_SDK_TEST_BOT_TOKEN=xoxb-xxx
    python setup.py run_integration_tests --test-target integration_tests/web/test_issue_1143.py

    https://github.com/slackapi/python-slack-sdk/issues/1143
    """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.bot_token = os.environ[SLACK_SDK_TEST_BOT_TOKEN]

    def tearDown(self):
        if False:
            return 10
        pass

    def test_backward_compatible_header(self):
        if False:
            print('Hello World!')
        client: WebClient = WebClient(token=self.bot_token)
        try:
            while True:
                client.users_list()
        except SlackApiError as e:
            self.assertIsNotNone(e.response.headers['Retry-After'])

    @async_test
    async def test_backward_compatible_header_async(self):
        client: AsyncWebClient = AsyncWebClient(token=self.bot_token)
        try:
            while True:
                await client.users_list()
        except SlackApiError as e:
            self.assertIsNotNone(e.response.headers['Retry-After'])