import collections
import logging
import os
import threading
import time
import unittest
import pytest
from integration_tests.env_variable_names import SLACK_SDK_TEST_CLASSIC_APP_BOT_TOKEN, SLACK_SDK_TEST_RTM_TEST_CHANNEL_ID
from integration_tests.helpers import is_not_specified
from slack_sdk.rtm import RTMClient
from slack_sdk.web import WebClient

class TestRTMClient(unittest.TestCase):
    """Runs integration tests with real Slack API

    https://github.com/slackapi/python-slack-sdk/issues/605
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.environ[SLACK_SDK_TEST_CLASSIC_APP_BOT_TOKEN]
        self.channel_id = os.environ[SLACK_SDK_TEST_RTM_TEST_CHANNEL_ID]
        self.rtm_client = RTMClient(token=self.bot_token, run_async=False)

    def tearDown(self):
        if False:
            print('Hello World!')
        RTMClient._callbacks = collections.defaultdict(list)

    @pytest.mark.skipif(condition=is_not_specified(), reason='To avoid rate_limited errors')
    def test_issue_605(self):
        if False:
            i = 10
            return i + 15
        self.text = 'This message was sent to verify issue #605'
        self.called = False

        @RTMClient.run_on(event='message')
        def process_messages(**payload):
            if False:
                return 10
            self.logger.info(payload)
            self.called = True

        def connect():
            if False:
                for i in range(10):
                    print('nop')
            self.logger.debug('Starting RTM Client...')
            self.rtm_client.start()
        t = threading.Thread(target=connect)
        t.daemon = True
        try:
            t.start()
            self.assertFalse(self.called)
            time.sleep(3)
            self.web_client = WebClient(token=self.bot_token, run_async=False)
            new_message = self.web_client.chat_postMessage(channel=self.channel_id, text=self.text)
            self.assertFalse('error' in new_message)
            time.sleep(5)
            self.assertTrue(self.called)
        finally:
            t.join(0.3)