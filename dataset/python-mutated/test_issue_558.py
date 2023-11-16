import asyncio
import collections
import logging
import os
import unittest
import pytest
from integration_tests.env_variable_names import SLACK_SDK_TEST_CLASSIC_APP_BOT_TOKEN, SLACK_SDK_TEST_RTM_TEST_CHANNEL_ID
from integration_tests.helpers import async_test, is_not_specified
from slack_sdk.rtm import RTMClient
from slack_sdk.web import WebClient

class TestRTMClient(unittest.TestCase):
    """Runs integration tests with real Slack API

    https://github.com/slackapi/python-slack-sdk/issues/558
    """

    def setUp(self):
        if False:
            print('Hello World!')
        self.logger = logging.getLogger(__name__)
        self.bot_token = os.environ[SLACK_SDK_TEST_CLASSIC_APP_BOT_TOKEN]

    def tearDown(self):
        if False:
            return 10
        RTMClient._callbacks = collections.defaultdict(list)

    @pytest.mark.skipif(condition=is_not_specified(), reason='Still unfixed')
    @async_test
    async def test_issue_558(self):
        channel_id = os.environ[SLACK_SDK_TEST_RTM_TEST_CHANNEL_ID]
        text = 'This message was sent by <https://slack.dev/python-slackclient/|python-slackclient>! (test_issue_558)'
        (self.message_count, self.reaction_count) = (0, 0)

        async def process_messages(**payload):
            self.logger.debug(payload)
            self.message_count += 1
            await asyncio.sleep(10)

        async def process_reactions(**payload):
            self.logger.debug(payload)
            self.reaction_count += 1
        rtm = RTMClient(token=self.bot_token, run_async=True)
        RTMClient.on(event='message', callback=process_messages)
        RTMClient.on(event='reaction_added', callback=process_reactions)
        web_client = WebClient(token=self.bot_token, run_async=True)
        message = await web_client.chat_postMessage(channel=channel_id, text=text)
        self.assertFalse('error' in message)
        ts = message['ts']
        await asyncio.sleep(3)
        rtm.start()
        await asyncio.sleep(3)
        try:
            first_reaction = await web_client.reactions_add(channel=channel_id, timestamp=ts, name='eyes')
            self.assertFalse('error' in first_reaction)
            await asyncio.sleep(2)
            message = await web_client.chat_postMessage(channel=channel_id, text=text)
            self.assertFalse('error' in message)
            second_reaction = await web_client.reactions_add(channel=channel_id, timestamp=ts, name='tada')
            self.assertFalse('error' in second_reaction)
            await asyncio.sleep(2)
            self.assertEqual(self.message_count, 1)
            self.assertEqual(self.reaction_count, 2)
        finally:
            if not rtm._stopped:
                rtm.stop()