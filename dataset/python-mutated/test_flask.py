"""
Unit tests for ``octoprint.server.util.flask``.
"""
__author__ = 'Gina Häußge <osd@foosel.net>'
__license__ = 'GNU Affero General Public License http://www.gnu.org/licenses/agpl.html'
__copyright__ = 'Copyright (C) 2016 The OctoPrint Project - Released under terms of the AGPLv3 License'
import unittest
from unittest import mock
import flask
from ddt import data, ddt, unpack
from octoprint.server.util.flask import OctoPrintFlaskRequest, OctoPrintFlaskResponse, ReverseProxiedEnvironment
standard_environ = {'HTTP_HOST': 'localhost:5000', 'SERVER_NAME': 'localhost', 'SERVER_PORT': '5000', 'SCRIPT_NAME': '', 'PATH_INFO': '/', 'wsgi.url_scheme': 'http'}

@ddt
class ReverseProxiedEnvironmentTest(unittest.TestCase):

    @data(({}, {}), ({'HTTP_X_SCRIPT_NAME': '/octoprint', 'PATH_INFO': '/static/online.gif'}, {'SCRIPT_NAME': '/octoprint'}), ({'HTTP_X_SCRIPT_NAME': '/octoprint', 'PATH_INFO': '/octoprint/static/online.gif'}, {'SCRIPT_NAME': '/octoprint', 'PATH_INFO': '/static/online.gif'}), ({'HTTP_X_FORWARDED_HOST': 'example.com'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}), ({'HTTP_X_FORWARDED_HOST': 'example.com:1234'}, {'HTTP_HOST': 'example.com:1234', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '1234'}), ({'HTTP_X_FORWARDED_HOST': 'example.com', 'HTTP_X_FORWARDED_PROTO': 'https'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '443', 'wsgi.url_scheme': 'https'}), ({'HTTP_X_FORWARDED_HOST': 'example.com', 'HTTP_X_SCHEME': 'https'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '443', 'wsgi.url_scheme': 'https'}), ({'HTTP_X_FORWARDED_HOST': 'example.com', 'HTTP_X_FORWARDED_SERVER': 'example2.com', 'HTTP_X_FORWARDED_PORT': '444', 'HTTP_X_FORWARDED_PROTO': 'https'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '443', 'wsgi.url_scheme': 'https'}), ({'HTTP_HOST': 'example.com', 'wsgi.url_scheme': 'https', 'SERVER_NAME': 'localhost', 'SERVER_PORT': '80'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '443'}), ({'HTTP_HOST': 'example.com:444', 'wsgi.url_scheme': 'https', 'SERVER_NAME': 'localhost', 'SERVER_PORT': '80'}, {'HTTP_HOST': 'example.com:444', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '444'}), ({'HTTP_X_FORWARDED_PROTO': 'https,http'}, {'wsgi.url_scheme': 'https'}), ({'HTTP_HOST': None, 'HTTP_X_FORWARDED_SERVER': 'example.com', 'HTTP_X_FORWARDED_PORT': '80'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}), ({'HTTP_HOST': None, 'SERVER_NAME': '127.0.0.1', 'SERVER_PORT': '80'}, {'HTTP_HOST': '127.0.0.1', 'SERVER_NAME': '127.0.0.1', 'SERVER_PORT': '80'}), ({'HTTP_HOST': None, 'SERVER_NAME': '127.0.0.1', 'SERVER_PORT': '444'}, {'HTTP_HOST': '127.0.0.1:444', 'SERVER_NAME': '127.0.0.1', 'SERVER_PORT': '444'}), ({'HTTP_HOST': None, 'SERVER_NAME': 'fec1::1', 'SERVER_PORT': '80'}, {'HTTP_HOST': 'fec1::1', 'SERVER_NAME': 'fec1::1', 'SERVER_PORT': '80'}), ({'HTTP_HOST': None, 'SERVER_NAME': 'fec1::1', 'SERVER_PORT': '444'}, {'HTTP_HOST': '[fec1::1]:444', 'SERVER_NAME': 'fec1::1', 'SERVER_PORT': '444'}), ({'HTTP_HOST': '127.0.0.1', 'SERVER_NAME': None, 'SERVER_PORT': None}, {'HTTP_HOST': '127.0.0.1', 'SERVER_NAME': '127.0.0.1', 'SERVER_PORT': '80'}), ({'HTTP_HOST': '127.0.0.1:444', 'SERVER_NAME': None, 'SERVER_PORT': None}, {'HTTP_HOST': '127.0.0.1:444', 'SERVER_NAME': '127.0.0.1', 'SERVER_PORT': '444'}), ({'HTTP_HOST': 'fec1::1', 'SERVER_NAME': None, 'SERVER_PORT': None}, {'HTTP_HOST': 'fec1::1', 'SERVER_NAME': 'fec1::1', 'SERVER_PORT': '80'}), ({'HTTP_HOST': '[fec1::1]:444', 'SERVER_NAME': None, 'SERVER_PORT': None}, {'HTTP_HOST': '[fec1::1]:444', 'SERVER_NAME': 'fec1::1', 'SERVER_PORT': '444'}))
    @unpack
    def test_stock(self, environ, expected):
        if False:
            while True:
                i = 10
        reverse_proxied = ReverseProxiedEnvironment()
        merged_environ = dict(standard_environ)
        merged_environ.update(environ)
        actual = reverse_proxied(merged_environ)
        merged_expected = dict(standard_environ)
        merged_expected.update(environ)
        merged_expected.update(expected)
        self.assertDictEqual(merged_expected, actual)

    @data(({'SERVER_NAME': 'example2.com', 'SERVER_PORT': '444', 'HTTP_X_FORWARDED_PROTO': 'https'}, {'HTTP_HOST': 'example2.com:444', 'SERVER_NAME': 'example2.com', 'SERVER_PORT': '444', 'wsgi.url_scheme': 'https'}), ({'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}), ({'HTTP_X_FORWARDED_SERVER': 'example2.com', 'HTTP_X_FORWARDED_PORT': '444', 'HTTP_X_FORWARDED_PROTO': 'https'}, {'HTTP_HOST': 'example2.com:444', 'SERVER_NAME': 'example2.com', 'SERVER_PORT': '444', 'wsgi.url_scheme': 'https'}), ({'HTTP_X_FORWARDED_SERVER': 'example.com', 'HTTP_X_FORWARDED_PORT': '80'}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}))
    @unpack
    def test_nohost(self, environ, expected):
        if False:
            i = 10
            return i + 15
        reverse_proxied = ReverseProxiedEnvironment()
        merged_environ = dict(standard_environ)
        merged_environ.update(environ)
        del merged_environ['HTTP_HOST']
        actual = reverse_proxied(merged_environ)
        merged_expected = dict(standard_environ)
        merged_expected.update(environ)
        merged_expected.update(expected)
        self.assertDictEqual(merged_expected, actual)

    @data(({'prefix': 'fallback_prefix'}, {}, {'SCRIPT_NAME': 'fallback_prefix'}), ({'scheme': 'https'}, {}, {'wsgi.url_scheme': 'https'}), ({'host': 'example.com'}, {}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}), ({'host': 'example.com:81'}, {}, {'HTTP_HOST': 'example.com:81', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '81'}), ({'prefix': '/octoprint'}, {'HTTP_X_SCRIPT_NAME': ''}, {}), ({'scheme': 'https'}, {'HTTP_X_FORWARDED_PROTO': 'http'}, {}), ({'scheme': 'https'}, {'HTTP_X_SCHEME': 'http'}, {}), ({'host': 'example.com:444'}, {'HTTP_X_FORWARDED_HOST': 'localhost:5000'}, {}), ({'server': 'example.com'}, {'HTTP_X_FORWARDED_SERVER': 'localhost'}, {}), ({'port': '444'}, {'HTTP_X_FORWARDED_PORT': '5000'}, {}), ({'server': 'example.com', 'port': '80'}, {}, {}))
    @unpack
    def test_fallbacks(self, fallbacks, environ, expected):
        if False:
            for i in range(10):
                print('nop')
        reverse_proxied = ReverseProxiedEnvironment(**fallbacks)
        merged_environ = dict(standard_environ)
        merged_environ.update(environ)
        actual = reverse_proxied(merged_environ)
        merged_expected = dict(standard_environ)
        merged_expected.update(environ)
        merged_expected.update(expected)
        self.assertDictEqual(merged_expected, actual)

    @data(({'server': 'example.com'}, {}, {'HTTP_HOST': 'example.com:5000', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '5000'}), ({'port': '80'}, {}, {'HTTP_HOST': 'localhost', 'SERVER_PORT': '80'}), ({'port': '81'}, {}, {'HTTP_HOST': 'localhost:81', 'SERVER_PORT': '81'}), ({'server': 'example.com', 'port': '80'}, {}, {'HTTP_HOST': 'example.com', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '80'}), ({'server': 'example.com', 'port': '81'}, {}, {'HTTP_HOST': 'example.com:81', 'SERVER_NAME': 'example.com', 'SERVER_PORT': '81'}))
    @unpack
    def test_fallbacks_nohost(self, fallbacks, environ, expected):
        if False:
            print('Hello World!')
        reverse_proxied = ReverseProxiedEnvironment(**fallbacks)
        merged_environ = dict(standard_environ)
        merged_environ.update(environ)
        del merged_environ['HTTP_HOST']
        actual = reverse_proxied(merged_environ)
        merged_expected = dict(standard_environ)
        merged_expected.update(environ)
        merged_expected.update(expected)
        self.assertDictEqual(merged_expected, actual)

    def test_header_config_ok(self):
        if False:
            return 10
        result = ReverseProxiedEnvironment.to_header_candidates(['prefix-header1', 'prefix-header2'])
        self.assertSetEqual(set(result), {'HTTP_PREFIX_HEADER1', 'HTTP_PREFIX_HEADER2'})

    def test_header_config_string(self):
        if False:
            while True:
                i = 10
        result = ReverseProxiedEnvironment.to_header_candidates('prefix-header')
        self.assertSetEqual(set(result), {'HTTP_PREFIX_HEADER'})

    def test_header_config_none(self):
        if False:
            for i in range(10):
                print('nop')
        result = ReverseProxiedEnvironment.to_header_candidates(None)
        self.assertEqual(result, [])

class OctoPrintFlaskRequestTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.orig_environment_wrapper = OctoPrintFlaskRequest.environment_wrapper
        self.app = flask.Flask('testapp')
        self.app.config['SECRET_KEY'] = 'secret'

    def tearDown(self):
        if False:
            print('Hello World!')
        OctoPrintFlaskRequest.environment_wrapper = staticmethod(self.orig_environment_wrapper)

    def test_environment_wrapper(self):
        if False:
            print('Hello World!')

        def environment_wrapper(environ):
            if False:
                i = 10
                return i + 15
            environ.update({'TEST': 'yes'})
            return environ
        OctoPrintFlaskRequest.environment_wrapper = staticmethod(environment_wrapper)
        request = OctoPrintFlaskRequest(standard_environ)
        self.assertTrue('TEST' in request.environ)

    def test_server_name(self):
        if False:
            while True:
                i = 10
        request = OctoPrintFlaskRequest(standard_environ)
        self.assertEqual('localhost', request.server_name)

    def test_server_port(self):
        if False:
            print('Hello World!')
        request = OctoPrintFlaskRequest(standard_environ)
        self.assertEqual('5000', request.server_port)

    def test_cookie_suffix(self):
        if False:
            for i in range(10):
                print('nop')
        request = OctoPrintFlaskRequest(standard_environ)
        self.assertEqual('_P5000', request.cookie_suffix)

    def test_cookie_suffix_with_root(self):
        if False:
            for i in range(10):
                print('nop')
        script_root_environ = dict(standard_environ)
        script_root_environ['SCRIPT_NAME'] = '/path/to/octoprint'
        request = OctoPrintFlaskRequest(script_root_environ)
        self.assertEqual('_P5000_R|path|to|octoprint', request.cookie_suffix)

    def test_cookies(self):
        if False:
            print('Hello World!')
        environ = dict(standard_environ)
        environ['HTTP_COOKIE'] = 'postfixed_P5000=postfixed_value; postfixed_wrong_P5001=postfixed_wrong_value; unpostfixed=unpostfixed_value; both_P5000=both_postfixed_value; both=both_unpostfixed_value;'
        request = OctoPrintFlaskRequest(environ)
        with self.app.app_context():
            cookies = request.cookies
        self.assertDictEqual({'postfixed': 'postfixed_value', 'postfixed_wrong_P5001': 'postfixed_wrong_value', 'unpostfixed': 'unpostfixed_value', 'both': 'both_postfixed_value'}, cookies)

@ddt
class OctoPrintFlaskResponseTest(unittest.TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        self.settings_patcher = mock.patch('octoprint.settings.settings')
        self.settings_getter = self.settings_patcher.start()
        self.settings = mock.MagicMock()
        self.settings_getter.return_value = self.settings
        self.app = flask.Flask('testapp')
        self.app.config['SECRET_KEY'] = 'secret'

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.settings_patcher.stop()

    @data([None, None, False, None, None], [None, None, False, 'none', 'None'], [None, None, False, 'lax', 'lax'], [None, None, False, 'StRiCt', 'strict'], [None, None, False, 'INVALID', None], [None, None, True, None, None], ['/subfolder/', None, False, None, None], [None, '/some/other/script/root', False, None, None], ['/subfolder/', '/some/other/script/root', False, None, None])
    @unpack
    def test_cookie_set_and_delete(self, path, scriptroot, secure, samesite, expected_samesite):
        if False:
            return 10
        environ = dict(standard_environ)
        expected_suffix = '_P5000'
        if scriptroot is not None:
            environ.update({'SCRIPT_NAME': scriptroot})
            expected_suffix += '_R' + scriptroot.replace('/', '|')
        request = OctoPrintFlaskRequest(environ)
        if path:
            expected_path_set = expected_path_delete = path
        else:
            expected_path_set = expected_path_delete = '/'
        if scriptroot:
            expected_path_set = scriptroot + expected_path_set
        if path is not None:
            kwargs = {'path': path}
        else:
            kwargs = {}
        with mock.patch('flask.request', new=request):
            with mock.patch('octoprint.server.util.flask.settings') as settings_mock:
                settings = mock.MagicMock()
                settings.getBoolean.return_value = secure
                settings.get.return_value = samesite
                settings_mock.return_value = settings
                response = OctoPrintFlaskResponse()
                with mock.patch('flask.Response.set_cookie') as set_cookie_mock:
                    with self.app.app_context():
                        response.set_cookie('some_key', 'some_value', **kwargs)
                    set_cookie_mock.assert_called_once_with(response, 'some_key' + expected_suffix, value='some_value', path=expected_path_set, secure=secure, samesite=expected_samesite)
                with mock.patch('flask.Response.set_cookie') as set_cookie_mock:
                    with mock.patch('flask.Response.delete_cookie') as delete_cookie_mock:
                        with self.app.app_context():
                            response.delete_cookie('some_key', **kwargs)
                        delete_cookie_mock.assert_called_once_with(response, 'some_key', path=expected_path_delete, domain=None)
                        set_cookie_mock.assert_called_once_with(response, 'some_key', expires=0, max_age=0, path=expected_path_delete, domain=None)