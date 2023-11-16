import logging
import unittest
from slack_sdk.aiohttp_version_checker import validate_aiohttp_version

class TestAiohttpVersionChecker(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.logger = logging.getLogger(__name__)

    def tearDown(self):
        if False:
            print('Hello World!')
        pass

    def test_not_recommended_versions(self):
        if False:
            while True:
                i = 10
        state = {'counter': 0}

        def print(message: str):
            if False:
                print('Hello World!')
            state['counter'] = state['counter'] + 1
        validate_aiohttp_version('2.1.3', print)
        self.assertEqual(state['counter'], 1)
        validate_aiohttp_version('3.6.3', print)
        self.assertEqual(state['counter'], 2)
        validate_aiohttp_version('3.7.0', print)
        self.assertEqual(state['counter'], 3)

    def test_recommended_versions(self):
        if False:
            for i in range(10):
                print('nop')
        state = {'counter': 0}

        def print(message: str):
            if False:
                print('Hello World!')
            state['counter'] = state['counter'] + 1
        validate_aiohttp_version('3.7.1', print)
        self.assertEqual(state['counter'], 0)
        validate_aiohttp_version('3.7.3', print)
        self.assertEqual(state['counter'], 0)
        validate_aiohttp_version('3.8.0', print)
        self.assertEqual(state['counter'], 0)
        validate_aiohttp_version('4.0.0', print)
        self.assertEqual(state['counter'], 0)
        validate_aiohttp_version('4.0.0rc1', print)
        self.assertEqual(state['counter'], 0)