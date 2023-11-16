"""
mock_utils.py: useful utils for mocking requests and responses for testing
for observable analyzers, if can customize the behavior based on:
MOCK_CONNECTIONS to True -> connections to external analyzers are faked
"""
from dataclasses import dataclass
from unittest import skip, skipIf
from unittest.mock import MagicMock, patch
from django.conf import settings

class MockUpRequest:

    def __init__(self, user):
        if False:
            for i in range(10):
                print('nop')
        self.user = user

class MockUpResponse:

    @dataclass()
    class Request:

        def __init__(self):
            if False:
                while True:
                    i = 10
            self.url = None

    def __init__(self, json_data, status_code, text='', content=b'', url='', headers=None):
        if False:
            i = 10
            return i + 15
        self.json_data = json_data
        self.status_code = status_code
        self.text = text
        self.content = content
        self.url = url
        self.headers = headers or {}
        self.request = self.Request()

    def json(self):
        if False:
            return 10
        return self.json_data

    @staticmethod
    def raise_for_status():
        if False:
            i = 10
            return i + 15
        pass

class MockResponseNoOp:

    def __init__(self, json_data, status_code):
        if False:
            return 10
        pass

    @staticmethod
    def search(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        return {}

    @staticmethod
    def query(*args, **kwargs):
        if False:
            print('Hello World!')
        return {}

def if_mock_connections(*decorators):
    if False:
        return 10

    def apply_all(f):
        if False:
            return 10
        for d in reversed(decorators):
            f = d(f)
        return f
    return apply_all if settings.MOCK_CONNECTIONS else lambda x: x