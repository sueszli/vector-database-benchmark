from unittest import mock
from httpie.cli.constants import SEPARATOR_CREDENTIALS
from httpie.plugins import AuthPlugin
from httpie.plugins.registry import plugin_manager
from .utils import http, HTTP_OK
USERNAME = 'user'
PASSWORD = 'password'
BASIC_AUTH_HEADER_VALUE = 'Basic dXNlcjpwYXNzd29yZA=='
BASIC_AUTH_URL = f'/basic-auth/{USERNAME}/{PASSWORD}'
AUTH_OK = {'authenticated': True, 'user': USERNAME}

def basic_auth(header=BASIC_AUTH_HEADER_VALUE):
    if False:
        while True:
            i = 10

    def inner(r):
        if False:
            while True:
                i = 10
        r.headers['Authorization'] = header
        return r
    return inner

def test_auth_plugin_parse_auth_false(httpbin):
    if False:
        i = 10
        return i + 15

    class Plugin(AuthPlugin):
        auth_type = 'test-parse-false'
        auth_parse = False

        def get_auth(self, username=None, password=None):
            if False:
                for i in range(10):
                    print('nop')
            assert username is None
            assert password is None
            assert self.raw_auth == BASIC_AUTH_HEADER_VALUE
            return basic_auth(self.raw_auth)
    plugin_manager.register(Plugin)
    try:
        r = http(httpbin + BASIC_AUTH_URL, '--auth-type', Plugin.auth_type, '--auth', BASIC_AUTH_HEADER_VALUE)
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)

def test_auth_plugin_require_auth_false(httpbin):
    if False:
        print('Hello World!')

    class Plugin(AuthPlugin):
        auth_type = 'test-require-false'
        auth_require = False

        def get_auth(self, username=None, password=None):
            if False:
                for i in range(10):
                    print('nop')
            assert self.raw_auth is None
            assert username is None
            assert password is None
            return basic_auth()
    plugin_manager.register(Plugin)
    try:
        r = http(httpbin + BASIC_AUTH_URL, '--auth-type', Plugin.auth_type)
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)

def test_auth_plugin_require_auth_false_and_auth_provided(httpbin):
    if False:
        return 10

    class Plugin(AuthPlugin):
        auth_type = 'test-require-false-yet-provided'
        auth_require = False

        def get_auth(self, username=None, password=None):
            if False:
                return 10
            assert self.raw_auth == USERNAME + SEPARATOR_CREDENTIALS + PASSWORD
            assert username == USERNAME
            assert password == PASSWORD
            return basic_auth()
    plugin_manager.register(Plugin)
    try:
        r = http(httpbin + BASIC_AUTH_URL, '--auth-type', Plugin.auth_type, '--auth', USERNAME + SEPARATOR_CREDENTIALS + PASSWORD)
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)

@mock.patch('httpie.cli.argtypes.AuthCredentials._getpass', new=lambda self, prompt: 'UNEXPECTED_PROMPT_RESPONSE')
def test_auth_plugin_prompt_password_false(httpbin):
    if False:
        return 10

    class Plugin(AuthPlugin):
        auth_type = 'test-prompt-false'
        prompt_password = False

        def get_auth(self, username=None, password=None):
            if False:
                while True:
                    i = 10
            assert self.raw_auth == USERNAME
            assert username == USERNAME
            assert password is None
            return basic_auth()
    plugin_manager.register(Plugin)
    try:
        r = http(httpbin + BASIC_AUTH_URL, '--auth-type', Plugin.auth_type, '--auth', USERNAME)
        assert HTTP_OK in r
        assert r.json == AUTH_OK
    finally:
        plugin_manager.unregister(Plugin)