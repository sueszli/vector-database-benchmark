import unittest
from slack_sdk.web.async_slack_response import AsyncSlackResponse
from slack_sdk.web.async_client import AsyncWebClient

class TestAsyncSlackResponse(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            while True:
                i = 10
        pass

    def test_issue_1100(self):
        if False:
            return 10
        response = AsyncSlackResponse(client=AsyncWebClient(token='xoxb-dummy'), http_verb='POST', api_url='http://localhost:3000/api.test', req_args={}, data=None, headers={}, status_code=200)
        with self.assertRaises(ValueError):
            response['foo']
        foo = response.get('foo')
        self.assertIsNone(foo)

    def test_issue_1102(self):
        if False:
            while True:
                i = 10
        response = AsyncSlackResponse(client=AsyncWebClient(token='xoxb-dummy'), http_verb='POST', api_url='http://localhost:3000/api.test', req_args={}, data={'ok': True, 'args': {'hello': 'world'}}, headers={}, status_code=200)
        self.assertTrue('ok' in response)
        self.assertTrue('foo' not in response)