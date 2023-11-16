"""Tests for the 'subsonic' plugin."""
import unittest
from test import _common
from test.helper import TestHelper
from urllib.parse import parse_qs, urlparse
import responses
from beets import config
from beetsplug import subsonicupdate

class ArgumentsMock:
    """Argument mocks for tests."""

    def __init__(self, mode, show_failures):
        if False:
            for i in range(10):
                print('nop')
        'Constructs ArgumentsMock.'
        self.mode = mode
        self.show_failures = show_failures
        self.verbose = 1

def _params(url):
    if False:
        i = 10
        return i + 15
    'Get the query parameters from a URL.'
    return parse_qs(urlparse(url).query)

class SubsonicPluginTest(_common.TestCase, TestHelper):
    """Test class for subsonicupdate."""

    @responses.activate
    def setUp(self):
        if False:
            i = 10
            return i + 15
        'Sets up config and plugin for test.'
        config.clear()
        self.setup_beets()
        config['subsonic']['user'] = 'admin'
        config['subsonic']['pass'] = 'admin'
        config['subsonic']['url'] = 'http://localhost:4040'
        responses.add(responses.GET, 'http://localhost:4040/rest/ping.view', status=200, body=self.PING_BODY)
        self.subsonicupdate = subsonicupdate.SubsonicUpdate()
    PING_BODY = '\n{\n    "subsonic-response": {\n        "status": "failed",\n        "version": "1.15.0"\n    }\n}\n'
    SUCCESS_BODY = '\n{\n    "subsonic-response": {\n        "status": "ok",\n        "version": "1.15.0",\n        "scanStatus": {\n            "scanning": true,\n            "count": 1000\n        }\n    }\n}\n'
    FAILED_BODY = '\n{\n    "subsonic-response": {\n        "status": "failed",\n        "version": "1.15.0",\n        "error": {\n            "code": 40,\n            "message": "Wrong username or password."\n        }\n    }\n}\n'
    ERROR_BODY = '\n{\n    "timestamp": 1599185854498,\n    "status": 404,\n    "error": "Not Found",\n    "message": "No message available",\n    "path": "/rest/startScn"\n}\n'

    def tearDown(self):
        if False:
            return 10
        'Tears down tests.'
        self.teardown_beets()

    @responses.activate
    def test_start_scan(self):
        if False:
            return 10
        'Tests success path based on best case scenario.'
        responses.add(responses.GET, 'http://localhost:4040/rest/startScan', status=200, body=self.SUCCESS_BODY)
        self.subsonicupdate.start_scan()

    @responses.activate
    def test_start_scan_failed_bad_credentials(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests failed path based on bad credentials.'
        responses.add(responses.GET, 'http://localhost:4040/rest/startScan', status=200, body=self.FAILED_BODY)
        self.subsonicupdate.start_scan()

    @responses.activate
    def test_start_scan_failed_not_found(self):
        if False:
            i = 10
            return i + 15
        'Tests failed path based on resource not found.'
        responses.add(responses.GET, 'http://localhost:4040/rest/startScan', status=404, body=self.ERROR_BODY)
        self.subsonicupdate.start_scan()

    def test_start_scan_failed_unreachable(self):
        if False:
            return 10
        'Tests failed path based on service not available.'
        self.subsonicupdate.start_scan()

    @responses.activate
    def test_url_with_context_path(self):
        if False:
            while True:
                i = 10
        'Tests success for included with contextPath.'
        config['subsonic']['url'] = 'http://localhost:4040/contextPath/'
        responses.add(responses.GET, 'http://localhost:4040/contextPath/rest/startScan', status=200, body=self.SUCCESS_BODY)
        self.subsonicupdate.start_scan()

    @responses.activate
    def test_url_with_trailing_forward_slash_url(self):
        if False:
            i = 10
            return i + 15
        'Tests success path based on trailing forward slash.'
        config['subsonic']['url'] = 'http://localhost:4040/'
        responses.add(responses.GET, 'http://localhost:4040/rest/startScan', status=200, body=self.SUCCESS_BODY)
        self.subsonicupdate.start_scan()

    @responses.activate
    def test_url_with_missing_port(self):
        if False:
            return 10
        'Tests failed path based on missing port.'
        config['subsonic']['url'] = 'http://localhost/airsonic'
        responses.add(responses.GET, 'http://localhost/airsonic/rest/startScan', status=200, body=self.SUCCESS_BODY)
        self.subsonicupdate.start_scan()

    @responses.activate
    def test_url_with_missing_schema(self):
        if False:
            while True:
                i = 10
        'Tests failed path based on missing schema.'
        config['subsonic']['url'] = 'localhost:4040/airsonic'
        responses.add(responses.GET, 'http://localhost:4040/rest/startScan', status=200, body=self.SUCCESS_BODY)
        self.subsonicupdate.start_scan()

def suite():
    if False:
        i = 10
        return i + 15
    'Default test suite.'
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')