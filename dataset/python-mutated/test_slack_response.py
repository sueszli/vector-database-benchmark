import unittest
from slack import WebClient
from slack.web.slack_response import SlackResponse

class TestSlackResponse(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def test_issue_559(self):
        if False:
            print('Hello World!')
        response = SlackResponse(client=WebClient(token='xoxb-dummy'), http_verb='POST', api_url='http://localhost:3000/api.test', req_args={}, data={'ok': True, 'args': {'hello': 'world'}}, headers={}, status_code=200)
        self.assertTrue('ok' in response.data)
        self.assertTrue('args' in response.data)
        self.assertFalse('error' in response.data)