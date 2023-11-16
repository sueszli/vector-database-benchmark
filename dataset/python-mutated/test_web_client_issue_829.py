import unittest
import slack.errors as err
from slack import WebClient
from tests.helpers import async_test
from tests.web.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestWebClient_Issue_829(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        setup_mock_web_api_server(self)
        self.client = WebClient(token='xoxp-1234', base_url='http://localhost:8888')
        self.async_client = WebClient(token='xoxp-1234', run_async=True, base_url='http://localhost:8888')

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        cleanup_mock_web_api_server(self)

    def test_html_response_body_issue_829(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebClient(base_url='http://localhost:8888')
        try:
            client.users_list(token='xoxb-error_html_response')
            self.fail('SlackApiError expected here')
        except err.SlackApiError as e:
            self.assertTrue(str(e).startswith('Received a response in a non-JSON format: '), e)

    @async_test
    async def test_html_response_body_issue_829_async(self):
        client = WebClient(base_url='http://localhost:8888', run_async=True)
        try:
            await client.users_list(token='xoxb-error_html_response')
            self.fail('SlackApiError expected here')
        except err.SlackApiError as e:
            self.assertEqual('The request to the Slack API failed.\nThe server responded with: {}', str(e))