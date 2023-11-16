import unittest
from slack import WebClient
from tests.web.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestWebClient_Issue_891(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        setup_mock_web_api_server(self)

    def tearDown(self):
        if False:
            print('Hello World!')
        cleanup_mock_web_api_server(self)

    def test_missing_text_warning_chat_postMessage(self):
        if False:
            while True:
                i = 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`text` argument is missing'):
            resp = client.chat_postMessage(channel='C111', blocks=[])
        self.assertIsNone(resp['error'])

    def test_missing_text_warning_chat_postEphemeral(self):
        if False:
            while True:
                i = 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`text` argument is missing'):
            resp = client.chat_postEphemeral(channel='C111', user='U111', blocks=[])
        self.assertIsNone(resp['error'])

    def test_missing_text_warning_chat_scheduleMessage(self):
        if False:
            while True:
                i = 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`text` argument is missing'):
            resp = client.chat_scheduleMessage(channel='C111', post_at='299876400', text='', blocks=[])
        self.assertIsNone(resp['error'])

    def test_missing_text_warning_chat_update(self):
        if False:
            i = 10
            return i + 15
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`text` argument is missing'):
            resp = client.chat_update(channel='C111', ts='111.222', blocks=[])
        self.assertIsNone(resp['error'])

    def test_missing_fallback_warning_chat_postMessage(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`fallback` argument is missing'):
            resp = client.chat_postMessage(channel='C111', blocks=[], attachments=[{'text': 'hi'}])
        self.assertIsNone(resp['error'])

    def test_missing_fallback_warning_chat_postEphemeral(self):
        if False:
            print('Hello World!')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`fallback` argument is missing'):
            resp = client.chat_postEphemeral(channel='C111', user='U111', blocks=[], attachments=[{'text': 'hi'}])
        self.assertIsNone(resp['error'])

    def test_missing_fallback_warning_chat_scheduleMessage(self):
        if False:
            print('Hello World!')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`fallback` argument is missing'):
            resp = client.chat_scheduleMessage(channel='C111', post_at='299876400', text='', blocks=[], attachments=[{'text': 'hi'}])
        self.assertIsNone(resp['error'])

    def test_missing_fallback_warning_chat_update(self):
        if False:
            print('Hello World!')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test')
        with self.assertWarnsRegex(UserWarning, '`fallback` argument is missing'):
            resp = client.chat_update(channel='C111', ts='111.222', blocks=[], attachments=[{'text': 'hi'}])
        self.assertIsNone(resp['error'])