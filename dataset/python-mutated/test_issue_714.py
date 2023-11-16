import asyncio
import os
import unittest
from urllib.error import URLError
from aiohttp import ClientConnectorError
from integration_tests.env_variable_names import SLACK_SDK_TEST_BOT_TOKEN
from integration_tests.helpers import async_test
from slack_sdk.web import WebClient
from slack_sdk.web.async_client import AsyncWebClient

class TestWebClient(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.proxy = 'http://invalid-host:9999'
        self.bot_token = os.environ[SLACK_SDK_TEST_BOT_TOKEN]

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        pass

    def test_proxy_failure(self):
        if False:
            while True:
                i = 10
        client: WebClient = WebClient(token=self.bot_token, proxy=self.proxy)
        with self.assertRaises(URLError):
            client.auth_test()

    @async_test
    async def test_proxy_failure_async(self):
        client: AsyncWebClient = AsyncWebClient(token=self.bot_token, proxy=self.proxy)
        with self.assertRaises(ClientConnectorError):
            await client.auth_test()