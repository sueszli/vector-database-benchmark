import json
import unittest
from slack_sdk.web import WebClient
from tests.slack_sdk.web.mock_web_api_server import setup_mock_web_api_server, cleanup_mock_web_api_server

class TestWebClient_Issue_971(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        setup_mock_web_api_server(self)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        cleanup_mock_web_api_server(self)

    def test_text_arg_only(self):
        if False:
            return 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        resp = client.chat_postMessage(channel='C111', text='test')
        self.assertTrue(resp['ok'])

    def test_blocks_with_text_arg(self):
        if False:
            while True:
                i = 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        resp = client.chat_postMessage(channel='C111', text='test', blocks=[])
        self.assertTrue(resp['ok'])

    def test_blocks_without_text_arg(self):
        if False:
            return 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        with self.assertWarns(UserWarning):
            resp = client.chat_postMessage(channel='C111', blocks=[])
        self.assertTrue(resp['ok'])

    def test_attachments_with_fallback(self):
        if False:
            i = 10
            return i + 15
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        resp = client.chat_postMessage(channel='C111', attachments=[{'fallback': 'test'}])
        self.assertTrue(resp['ok'])

    def test_attachments_with_empty_fallback(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        with self.assertWarns(UserWarning):
            resp = client.chat_postMessage(channel='C111', attachments=[{'fallback': ''}])
        self.assertTrue(resp['ok'])

    def test_attachments_without_fallback(self):
        if False:
            i = 10
            return i + 15
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        with self.assertWarns(UserWarning):
            resp = client.chat_postMessage(channel='C111', attachments=[{}])
        self.assertTrue(resp['ok'])

    def test_multiple_attachments_one_without_fallback(self):
        if False:
            return 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        with self.assertWarns(UserWarning):
            resp = client.chat_postMessage(channel='C111', attachments=[{'fallback': 'test'}, {}])
        self.assertTrue(resp['ok'])

    def test_blocks_as_deserialzed_json_without_text_arg(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        with self.assertWarns(UserWarning):
            resp = client.chat_postMessage(channel='C111', attachments=json.dumps([]))
        self.assertTrue(resp['ok'])

    def test_blocks_as_deserialized_json_with_text_arg(self):
        if False:
            for i in range(10):
                print('nop')
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        resp = client.chat_postMessage(channel='C111', text='test', blocks=json.dumps([]))
        self.assertTrue(resp['ok'])

    def test_attachments_as_deserialzed_json_without_text_arg(self):
        if False:
            i = 10
            return i + 15
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        with self.assertWarns(UserWarning):
            resp = client.chat_postMessage(channel='C111', attachments=json.dumps([{'fallback': 'test'}]))
        self.assertTrue(resp['ok'])

    def test_attachments_as_deserialized_json_with_text_arg(self):
        if False:
            return 10
        client = WebClient(base_url='http://localhost:8888', token='xoxb-api_test', team_id='T111')
        resp = client.chat_postMessage(channel='C111', text='test', attachments=json.dumps([]))
        self.assertTrue(resp['ok'])