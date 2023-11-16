from tornado.concurrent import Future
from tornado import gen
from tornado.escape import json_decode, utf8, to_unicode, recursive_unicode, native_str, to_basestring
from tornado.httpclient import HTTPClientError
from tornado.httputil import format_timestamp
from tornado.iostream import IOStream
from tornado import locale
from tornado.locks import Event
from tornado.log import app_log, gen_log
from tornado.simple_httpclient import SimpleAsyncHTTPClient
from tornado.template import DictLoader
from tornado.testing import AsyncHTTPTestCase, AsyncTestCase, ExpectLog, gen_test
from tornado.test.util import ignore_deprecation
from tornado.util import ObjectDict, unicode_type
from tornado.web import Application, RequestHandler, StaticFileHandler, RedirectHandler as WebRedirectHandler, HTTPError, MissingArgumentError, ErrorHandler, authenticated, url, _create_signature_v1, create_signed_value, decode_signed_value, get_signature_key_version, UIModule, Finish, stream_request_body, removeslash, addslash, GZipContentEncoding
import binascii
import contextlib
import copy
import datetime
import email.utils
import gzip
from io import BytesIO
import itertools
import logging
import os
import re
import socket
import typing
import unittest
import urllib.parse

def relpath(*a):
    if False:
        print('Hello World!')
    return os.path.join(os.path.dirname(__file__), *a)

class WebTestCase(AsyncHTTPTestCase):
    """Base class for web tests that also supports WSGI mode.

    Override get_handlers and get_app_kwargs instead of get_app.
    This class is deprecated since WSGI mode is no longer supported.
    """

    def get_app(self):
        if False:
            for i in range(10):
                print('nop')
        self.app = Application(self.get_handlers(), **self.get_app_kwargs())
        return self.app

    def get_handlers(self):
        if False:
            print('Hello World!')
        raise NotImplementedError()

    def get_app_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return {}

class SimpleHandlerTestCase(WebTestCase):
    """Simplified base class for tests that work with a single handler class.

    To use, define a nested class named ``Handler``.
    """
    Handler = None

    def get_handlers(self):
        if False:
            return 10
        return [('/', self.Handler)]

class HelloHandler(RequestHandler):

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        self.write('hello')

class CookieTestRequestHandler(RequestHandler):

    def __init__(self, cookie_secret='0123456789', key_version=None):
        if False:
            for i in range(10):
                print('nop')
        self._cookies = {}
        if key_version is None:
            self.application = ObjectDict(settings=dict(cookie_secret=cookie_secret))
        else:
            self.application = ObjectDict(settings=dict(cookie_secret=cookie_secret, key_version=key_version))

    def get_cookie(self, name):
        if False:
            i = 10
            return i + 15
        return self._cookies.get(name)

    def set_cookie(self, name, value, expires_days=None):
        if False:
            i = 10
            return i + 15
        self._cookies[name] = value

class SecureCookieV1Test(unittest.TestCase):

    def test_round_trip(self):
        if False:
            for i in range(10):
                print('nop')
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie('foo', b'bar', version=1)
        self.assertEqual(handler.get_signed_cookie('foo', min_version=1), b'bar')

    def test_cookie_tampering_future_timestamp(self):
        if False:
            while True:
                i = 10
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie('foo', binascii.a2b_hex(b'd76df8e7aefc'), version=1)
        cookie = handler._cookies['foo']
        match = re.match(b'12345678\\|([0-9]+)\\|([0-9a-f]+)', cookie)
        assert match is not None
        timestamp = match.group(1)
        sig = match.group(2)
        self.assertEqual(_create_signature_v1(handler.application.settings['cookie_secret'], 'foo', '12345678', timestamp), sig)
        self.assertEqual(_create_signature_v1(handler.application.settings['cookie_secret'], 'foo', '1234', b'5678' + timestamp), sig)
        handler._cookies['foo'] = utf8('1234|5678%s|%s' % (to_basestring(timestamp), to_basestring(sig)))
        with ExpectLog(gen_log, 'Cookie timestamp in future'):
            self.assertTrue(handler.get_signed_cookie('foo', min_version=1) is None)

    def test_arbitrary_bytes(self):
        if False:
            i = 10
            return i + 15
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie('foo', b'\xe9', version=1)
        self.assertEqual(handler.get_signed_cookie('foo', min_version=1), b'\xe9')

class SecureCookieV2Test(unittest.TestCase):
    KEY_VERSIONS = {0: 'ajklasdf0ojaisdf', 1: 'aslkjasaolwkjsdf'}

    def test_round_trip(self):
        if False:
            for i in range(10):
                print('nop')
        handler = CookieTestRequestHandler()
        handler.set_signed_cookie('foo', b'bar', version=2)
        self.assertEqual(handler.get_signed_cookie('foo', min_version=2), b'bar')

    def test_key_version_roundtrip(self):
        if False:
            i = 10
            return i + 15
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=0)
        handler.set_signed_cookie('foo', b'bar')
        self.assertEqual(handler.get_signed_cookie('foo'), b'bar')

    def test_key_version_roundtrip_differing_version(self):
        if False:
            i = 10
            return i + 15
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=1)
        handler.set_signed_cookie('foo', b'bar')
        self.assertEqual(handler.get_signed_cookie('foo'), b'bar')

    def test_key_version_increment_version(self):
        if False:
            return 10
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=0)
        handler.set_signed_cookie('foo', b'bar')
        new_handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=1)
        new_handler._cookies = handler._cookies
        self.assertEqual(new_handler.get_signed_cookie('foo'), b'bar')

    def test_key_version_invalidate_version(self):
        if False:
            for i in range(10):
                print('nop')
        handler = CookieTestRequestHandler(cookie_secret=self.KEY_VERSIONS, key_version=0)
        handler.set_signed_cookie('foo', b'bar')
        new_key_versions = self.KEY_VERSIONS.copy()
        new_key_versions.pop(0)
        new_handler = CookieTestRequestHandler(cookie_secret=new_key_versions, key_version=1)
        new_handler._cookies = handler._cookies
        self.assertEqual(new_handler.get_signed_cookie('foo'), None)

class FinalReturnTest(WebTestCase):
    final_return = None

    def get_handlers(self):
        if False:
            print('Hello World!')
        test = self

        class FinishHandler(RequestHandler):

            @gen.coroutine
            def get(self):
                if False:
                    return 10
                test.final_return = self.finish()
                yield test.final_return

            @gen.coroutine
            def post(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.write('hello,')
                yield self.flush()
                test.final_return = self.finish('world')
                yield test.final_return

        class RenderHandler(RequestHandler):

            def create_template_loader(self, path):
                if False:
                    while True:
                        i = 10
                return DictLoader({'foo.html': 'hi'})

            @gen.coroutine
            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                test.final_return = self.render('foo.html')
        return [('/finish', FinishHandler), ('/render', RenderHandler)]

    def get_app_kwargs(self):
        if False:
            i = 10
            return i + 15
        return dict(template_path='FinalReturnTest')

    def test_finish_method_return_future(self):
        if False:
            print('Hello World!')
        response = self.fetch(self.get_url('/finish'))
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)
        self.assertTrue(self.final_return.done())
        response = self.fetch(self.get_url('/finish'), method='POST', body=b'')
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)
        self.assertTrue(self.final_return.done())

    def test_render_method_return_future(self):
        if False:
            while True:
                i = 10
        response = self.fetch(self.get_url('/render'))
        self.assertEqual(response.code, 200)
        self.assertIsInstance(self.final_return, Future)

class CookieTest(WebTestCase):

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')

        class SetCookieHandler(RequestHandler):

            def get(self):
                if False:
                    return 10
                self.set_cookie('str', 'asdf')
                self.set_cookie('unicode', 'qwer')
                self.set_cookie('bytes', b'zxcv')

        class GetCookieHandler(RequestHandler):

            def get(self):
                if False:
                    print('Hello World!')
                cookie = self.get_cookie('foo', 'default')
                assert cookie is not None
                self.write(cookie)

        class SetCookieDomainHandler(RequestHandler):

            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.set_cookie('unicode_args', 'blah', domain='foo.com', path='/foo')

        class SetCookieSpecialCharHandler(RequestHandler):

            def get(self):
                if False:
                    i = 10
                    return i + 15
                self.set_cookie('equals', 'a=b')
                self.set_cookie('semicolon', 'a;b')
                self.set_cookie('quote', 'a"b')

        class SetCookieOverwriteHandler(RequestHandler):

            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.set_cookie('a', 'b', domain='example.com')
                self.set_cookie('c', 'd', domain='example.com')
                self.set_cookie('a', 'e')

        class SetCookieMaxAgeHandler(RequestHandler):

            def get(self):
                if False:
                    return 10
                self.set_cookie('foo', 'bar', max_age=10)

        class SetCookieExpiresDaysHandler(RequestHandler):

            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.set_cookie('foo', 'bar', expires_days=10)

        class SetCookieFalsyFlags(RequestHandler):

            def get(self):
                if False:
                    print('Hello World!')
                self.set_cookie('a', '1', secure=True)
                self.set_cookie('b', '1', secure=False)
                self.set_cookie('c', '1', httponly=True)
                self.set_cookie('d', '1', httponly=False)

        class SetCookieDeprecatedArgs(RequestHandler):

            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.set_cookie('a', 'b', HttpOnly=True, pATH='/foo')
        return [('/set', SetCookieHandler), ('/get', GetCookieHandler), ('/set_domain', SetCookieDomainHandler), ('/special_char', SetCookieSpecialCharHandler), ('/set_overwrite', SetCookieOverwriteHandler), ('/set_max_age', SetCookieMaxAgeHandler), ('/set_expires_days', SetCookieExpiresDaysHandler), ('/set_falsy_flags', SetCookieFalsyFlags), ('/set_deprecated', SetCookieDeprecatedArgs)]

    def test_set_cookie(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/set')
        self.assertEqual(sorted(response.headers.get_list('Set-Cookie')), ['bytes=zxcv; Path=/', 'str=asdf; Path=/', 'unicode=qwer; Path=/'])

    def test_get_cookie(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/get', headers={'Cookie': 'foo=bar'})
        self.assertEqual(response.body, b'bar')
        response = self.fetch('/get', headers={'Cookie': 'foo="bar"'})
        self.assertEqual(response.body, b'bar')
        response = self.fetch('/get', headers={'Cookie': '/=exception;'})
        self.assertEqual(response.body, b'default')

    def test_set_cookie_domain(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/set_domain')
        self.assertEqual(response.headers.get_list('Set-Cookie'), ['unicode_args=blah; Domain=foo.com; Path=/foo'])

    def test_cookie_special_char(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/special_char')
        headers = sorted(response.headers.get_list('Set-Cookie'))
        self.assertEqual(len(headers), 3)
        self.assertEqual(headers[0], 'equals="a=b"; Path=/')
        self.assertEqual(headers[1], 'quote="a\\"b"; Path=/')
        self.assertTrue(headers[2] in ('semicolon="a;b"; Path=/', 'semicolon="a\\073b"; Path=/'), headers[2])
        data = [('foo=a=b', 'a=b'), ('foo="a=b"', 'a=b'), ('foo="a;b"', '"a'), ('foo=a\\073b', 'a\\073b'), ('foo="a\\073b"', 'a;b'), ('foo="a\\"b"', 'a"b')]
        for (header, expected) in data:
            logging.debug('trying %r', header)
            response = self.fetch('/get', headers={'Cookie': header})
            self.assertEqual(response.body, utf8(expected))

    def test_set_cookie_overwrite(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/set_overwrite')
        headers = response.headers.get_list('Set-Cookie')
        self.assertEqual(sorted(headers), ['a=e; Path=/', 'c=d; Domain=example.com; Path=/'])

    def test_set_cookie_max_age(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/set_max_age')
        headers = response.headers.get_list('Set-Cookie')
        self.assertEqual(sorted(headers), ['foo=bar; Max-Age=10; Path=/'])

    def test_set_cookie_expires_days(self):
        if False:
            print('Hello World!')
        response = self.fetch('/set_expires_days')
        header = response.headers.get('Set-Cookie')
        assert header is not None
        match = re.match('foo=bar; expires=(?P<expires>.+); Path=/', header)
        assert match is not None
        expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=10)
        header_expires = email.utils.parsedate_to_datetime(match.groupdict()['expires'])
        self.assertTrue(abs((expires - header_expires).total_seconds()) < 10)

    def test_set_cookie_false_flags(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/set_falsy_flags')
        headers = sorted(response.headers.get_list('Set-Cookie'))
        self.assertEqual(headers[0].lower(), 'a=1; path=/; secure')
        self.assertEqual(headers[1].lower(), 'b=1; path=/')
        self.assertEqual(headers[2].lower(), 'c=1; httponly; path=/')
        self.assertEqual(headers[3].lower(), 'd=1; path=/')

    def test_set_cookie_deprecated(self):
        if False:
            while True:
                i = 10
        with ignore_deprecation():
            response = self.fetch('/set_deprecated')
        header = response.headers.get('Set-Cookie')
        self.assertEqual(header, 'a=b; HttpOnly; Path=/foo')

class AuthRedirectRequestHandler(RequestHandler):

    def initialize(self, login_url):
        if False:
            return 10
        self.login_url = login_url

    def get_login_url(self):
        if False:
            print('Hello World!')
        return self.login_url

    @authenticated
    def get(self):
        if False:
            return 10
        self.send_error(500)

class AuthRedirectTest(WebTestCase):

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        return [('/relative', AuthRedirectRequestHandler, dict(login_url='/login')), ('/absolute', AuthRedirectRequestHandler, dict(login_url='http://example.com/login'))]

    def test_relative_auth_redirect(self):
        if False:
            print('Hello World!')
        response = self.fetch(self.get_url('/relative'), follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertEqual(response.headers['Location'], '/login?next=%2Frelative')

    def test_absolute_auth_redirect(self):
        if False:
            while True:
                i = 10
        response = self.fetch(self.get_url('/absolute'), follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertTrue(re.match('http://example.com/login\\?next=http%3A%2F%2F127.0.0.1%3A[0-9]+%2Fabsolute', response.headers['Location']), response.headers['Location'])

class ConnectionCloseHandler(RequestHandler):

    def initialize(self, test):
        if False:
            for i in range(10):
                print('nop')
        self.test = test

    @gen.coroutine
    def get(self):
        if False:
            i = 10
            return i + 15
        self.test.on_handler_waiting()
        yield self.test.cleanup_event.wait()

    def on_connection_close(self):
        if False:
            while True:
                i = 10
        self.test.on_connection_close()

class ConnectionCloseTest(WebTestCase):

    def get_handlers(self):
        if False:
            return 10
        self.cleanup_event = Event()
        return [('/', ConnectionCloseHandler, dict(test=self))]

    def test_connection_close(self):
        if False:
            while True:
                i = 10
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect(('127.0.0.1', self.get_http_port()))
        self.stream = IOStream(s)
        self.stream.write(b'GET / HTTP/1.0\r\n\r\n')
        self.wait()
        self.cleanup_event.set()
        self.io_loop.run_sync(lambda : gen.sleep(0))

    def on_handler_waiting(self):
        if False:
            return 10
        logging.debug('handler waiting')
        self.stream.close()

    def on_connection_close(self):
        if False:
            i = 10
            return i + 15
        logging.debug('connection closed')
        self.stop()

class EchoHandler(RequestHandler):

    def get(self, *path_args):
        if False:
            print('Hello World!')
        for key in self.request.arguments:
            if type(key) != str:
                raise Exception('incorrect type for key: %r' % type(key))
            for bvalue in self.request.arguments[key]:
                if type(bvalue) != bytes:
                    raise Exception('incorrect type for value: %r' % type(bvalue))
            for svalue in self.get_arguments(key):
                if type(svalue) != unicode_type:
                    raise Exception('incorrect type for value: %r' % type(svalue))
        for arg in path_args:
            if type(arg) != unicode_type:
                raise Exception('incorrect type for path arg: %r' % type(arg))
        self.write(dict(path=self.request.path, path_args=path_args, args=recursive_unicode(self.request.arguments)))

class RequestEncodingTest(WebTestCase):

    def get_handlers(self):
        if False:
            while True:
                i = 10
        return [('/group/(.*)', EchoHandler), ('/slashes/([^/]*)/([^/]*)', EchoHandler)]

    def fetch_json(self, path):
        if False:
            return 10
        return json_decode(self.fetch(path).body)

    def test_group_question_mark(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.fetch_json('/group/%3F'), dict(path='/group/%3F', path_args=['?'], args={}))
        self.assertEqual(self.fetch_json('/group/%3F?%3F=%3F'), dict(path='/group/%3F', path_args=['?'], args={'?': ['?']}))

    def test_group_encoding(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.fetch_json('/group/%C3%A9?arg=%C3%A9'), {'path': '/group/%C3%A9', 'path_args': ['é'], 'args': {'arg': ['é']}})

    def test_slashes(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.fetch_json('/slashes/foo/bar'), dict(path='/slashes/foo/bar', path_args=['foo', 'bar'], args={}))
        self.assertEqual(self.fetch_json('/slashes/a%2Fb/c%2Fd'), dict(path='/slashes/a%2Fb/c%2Fd', path_args=['a/b', 'c/d'], args={}))

    def test_error(self):
        if False:
            return 10
        with ExpectLog(gen_log, '.*Invalid unicode'):
            self.fetch('/group/?arg=%25%e9')

class TypeCheckHandler(RequestHandler):

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.errors = {}
        self.check_type('status', self.get_status(), int)
        self.check_type('argument', self.get_argument('foo'), unicode_type)
        self.check_type('cookie_key', list(self.cookies.keys())[0], str)
        self.check_type('cookie_value', list(self.cookies.values())[0].value, str)
        if list(self.cookies.keys()) != ['asdf']:
            raise Exception('unexpected values for cookie keys: %r' % self.cookies.keys())
        self.check_type('get_signed_cookie', self.get_signed_cookie('asdf'), bytes)
        self.check_type('get_cookie', self.get_cookie('asdf'), str)
        self.check_type('xsrf_token', self.xsrf_token, bytes)
        self.check_type('xsrf_form_html', self.xsrf_form_html(), str)
        self.check_type('reverse_url', self.reverse_url('typecheck', 'foo'), str)
        self.check_type('request_summary', self._request_summary(), str)

    def get(self, path_component):
        if False:
            i = 10
            return i + 15
        self.check_type('path_component', path_component, unicode_type)
        self.write(self.errors)

    def post(self, path_component):
        if False:
            i = 10
            return i + 15
        self.check_type('path_component', path_component, unicode_type)
        self.write(self.errors)

    def check_type(self, name, obj, expected_type):
        if False:
            i = 10
            return i + 15
        actual_type = type(obj)
        if expected_type != actual_type:
            self.errors[name] = 'expected %s, got %s' % (expected_type, actual_type)

class DecodeArgHandler(RequestHandler):

    def decode_argument(self, value, name=None):
        if False:
            return 10
        if type(value) != bytes:
            raise Exception('unexpected type for value: %r' % type(value))
        if 'encoding' in self.request.arguments:
            return value.decode(to_unicode(self.request.arguments['encoding'][0]))
        else:
            return value

    def get(self, arg):
        if False:
            i = 10
            return i + 15

        def describe(s):
            if False:
                print('Hello World!')
            if type(s) == bytes:
                return ['bytes', native_str(binascii.b2a_hex(s))]
            elif type(s) == unicode_type:
                return ['unicode', s]
            raise Exception('unknown type')
        self.write({'path': describe(arg), 'query': describe(self.get_argument('foo'))})

class LinkifyHandler(RequestHandler):

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        self.render('linkify.html', message='http://example.com')

class UIModuleResourceHandler(RequestHandler):

    def get(self):
        if False:
            print('Hello World!')
        self.render('page.html', entries=[1, 2])

class OptionalPathHandler(RequestHandler):

    def get(self, path):
        if False:
            for i in range(10):
                print('nop')
        self.write({'path': path})

class MultiHeaderHandler(RequestHandler):

    def get(self):
        if False:
            while True:
                i = 10
        self.set_header('x-overwrite', '1')
        self.set_header('X-Overwrite', 2)
        self.add_header('x-multi', 3)
        self.add_header('X-Multi', '4')

class RedirectHandler(RequestHandler):

    def get(self):
        if False:
            return 10
        if self.get_argument('permanent', None) is not None:
            self.redirect('/', permanent=bool(int(self.get_argument('permanent'))))
        elif self.get_argument('status', None) is not None:
            self.redirect('/', status=int(self.get_argument('status')))
        else:
            raise Exception("didn't get permanent or status arguments")

class EmptyFlushCallbackHandler(RequestHandler):

    @gen.coroutine
    def get(self):
        if False:
            return 10
        yield self.flush()
        yield self.flush()
        self.write('o')
        yield self.flush()
        yield self.flush()
        self.finish('k')

class HeaderInjectionHandler(RequestHandler):

    def get(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.set_header('X-Foo', 'foo\r\nX-Bar: baz')
            raise Exception("Didn't get expected exception")
        except ValueError as e:
            if 'Unsafe header value' in str(e):
                self.finish(b'ok')
            else:
                raise

class GetArgumentHandler(RequestHandler):

    def prepare(self):
        if False:
            return 10
        if self.get_argument('source', None) == 'query':
            method = self.get_query_argument
        elif self.get_argument('source', None) == 'body':
            method = self.get_body_argument
        else:
            method = self.get_argument
        self.finish(method('foo', 'default'))

class GetArgumentsHandler(RequestHandler):

    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.finish(dict(default=self.get_arguments('foo'), query=self.get_query_arguments('foo'), body=self.get_body_arguments('foo')))

class WSGISafeWebTest(WebTestCase):
    COOKIE_SECRET = 'WebTest.COOKIE_SECRET'

    def get_app_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        loader = DictLoader({'linkify.html': '{% module linkify(message) %}', 'page.html': '<html><head></head><body>\n{% for e in entries %}\n{% module Template("entry.html", entry=e) %}\n{% end %}\n</body></html>', 'entry.html': '{{ set_resources(embedded_css=".entry { margin-bottom: 1em; }",\n                 embedded_javascript="js_embed()",\n                 css_files=["/base.css", "/foo.css"],\n                 javascript_files="/common.js",\n                 html_head="<meta>",\n                 html_body=\'<script src="/analytics.js"/>\') }}\n<div class="entry">...</div>'})
        return dict(template_loader=loader, autoescape='xhtml_escape', cookie_secret=self.COOKIE_SECRET)

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        urls = [url('/typecheck/(.*)', TypeCheckHandler, name='typecheck'), url('/decode_arg/(.*)', DecodeArgHandler, name='decode_arg'), url('/decode_arg_kw/(?P<arg>.*)', DecodeArgHandler), url('/linkify', LinkifyHandler), url('/uimodule_resources', UIModuleResourceHandler), url('/optional_path/(.+)?', OptionalPathHandler), url('/multi_header', MultiHeaderHandler), url('/redirect', RedirectHandler), url('/web_redirect_permanent', WebRedirectHandler, {'url': '/web_redirect_newpath'}), url('/web_redirect', WebRedirectHandler, {'url': '/web_redirect_newpath', 'permanent': False}), url('//web_redirect_double_slash', WebRedirectHandler, {'url': '/web_redirect_newpath'}), url('/header_injection', HeaderInjectionHandler), url('/get_argument', GetArgumentHandler), url('/get_arguments', GetArgumentsHandler)]
        return urls

    def fetch_json(self, *args, **kwargs):
        if False:
            print('Hello World!')
        response = self.fetch(*args, **kwargs)
        response.rethrow()
        return json_decode(response.body)

    def test_types(self):
        if False:
            i = 10
            return i + 15
        cookie_value = to_unicode(create_signed_value(self.COOKIE_SECRET, 'asdf', 'qwer'))
        response = self.fetch('/typecheck/asdf?foo=bar', headers={'Cookie': 'asdf=' + cookie_value})
        data = json_decode(response.body)
        self.assertEqual(data, {})
        response = self.fetch('/typecheck/asdf?foo=bar', method='POST', headers={'Cookie': 'asdf=' + cookie_value}, body='foo=bar')

    def test_decode_argument(self):
        if False:
            i = 10
            return i + 15
        urls = ['/decode_arg/%C3%A9?foo=%C3%A9&encoding=utf-8', '/decode_arg/%E9?foo=%E9&encoding=latin1', '/decode_arg_kw/%E9?foo=%E9&encoding=latin1']
        for req_url in urls:
            response = self.fetch(req_url)
            response.rethrow()
            data = json_decode(response.body)
            self.assertEqual(data, {'path': ['unicode', 'é'], 'query': ['unicode', 'é']})
        response = self.fetch('/decode_arg/%C3%A9?foo=%C3%A9')
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {'path': ['bytes', 'c3a9'], 'query': ['bytes', 'c3a9']})

    def test_decode_argument_invalid_unicode(self):
        if False:
            for i in range(10):
                print('nop')
        with ExpectLog(gen_log, '.*Invalid unicode.*'):
            response = self.fetch('/typecheck/invalid%FF')
            self.assertEqual(response.code, 400)
            response = self.fetch('/typecheck/invalid?foo=%FF')
            self.assertEqual(response.code, 400)

    def test_decode_argument_plus(self):
        if False:
            print('Hello World!')
        urls = ['/decode_arg/1%20%2B%201?foo=1%20%2B%201&encoding=utf-8', '/decode_arg/1%20+%201?foo=1+%2B+1&encoding=utf-8']
        for req_url in urls:
            response = self.fetch(req_url)
            response.rethrow()
            data = json_decode(response.body)
            self.assertEqual(data, {'path': ['unicode', '1 + 1'], 'query': ['unicode', '1 + 1']})

    def test_reverse_url(self):
        if False:
            return 10
        self.assertEqual(self.app.reverse_url('decode_arg', 'foo'), '/decode_arg/foo')
        self.assertEqual(self.app.reverse_url('decode_arg', 42), '/decode_arg/42')
        self.assertEqual(self.app.reverse_url('decode_arg', b'\xe9'), '/decode_arg/%E9')
        self.assertEqual(self.app.reverse_url('decode_arg', 'é'), '/decode_arg/%C3%A9')
        self.assertEqual(self.app.reverse_url('decode_arg', '1 + 1'), '/decode_arg/1%20%2B%201')

    def test_uimodule_unescaped(self):
        if False:
            print('Hello World!')
        response = self.fetch('/linkify')
        self.assertEqual(response.body, b'<a href="http://example.com">http://example.com</a>')

    def test_uimodule_resources(self):
        if False:
            return 10
        response = self.fetch('/uimodule_resources')
        self.assertEqual(response.body, b'<html><head><link href="/base.css" type="text/css" rel="stylesheet"/><link href="/foo.css" type="text/css" rel="stylesheet"/>\n<style type="text/css">\n.entry { margin-bottom: 1em; }\n</style>\n<meta>\n</head><body>\n\n\n<div class="entry">...</div>\n\n\n<div class="entry">...</div>\n\n<script src="/common.js" type="text/javascript"></script>\n<script type="text/javascript">\n//<![CDATA[\njs_embed()\n//]]>\n</script>\n<script src="/analytics.js"/>\n</body></html>')

    def test_optional_path(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(self.fetch_json('/optional_path/foo'), {'path': 'foo'})
        self.assertEqual(self.fetch_json('/optional_path/'), {'path': None})

    def test_multi_header(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/multi_header')
        self.assertEqual(response.headers['x-overwrite'], '2')
        self.assertEqual(response.headers.get_list('x-multi'), ['3', '4'])

    def test_redirect(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/redirect?permanent=1', follow_redirects=False)
        self.assertEqual(response.code, 301)
        response = self.fetch('/redirect?permanent=0', follow_redirects=False)
        self.assertEqual(response.code, 302)
        response = self.fetch('/redirect?status=307', follow_redirects=False)
        self.assertEqual(response.code, 307)

    def test_web_redirect(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/web_redirect_permanent', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/web_redirect_newpath')
        response = self.fetch('/web_redirect', follow_redirects=False)
        self.assertEqual(response.code, 302)
        self.assertEqual(response.headers['Location'], '/web_redirect_newpath')

    def test_web_redirect_double_slash(self):
        if False:
            return 10
        response = self.fetch('//web_redirect_double_slash', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/web_redirect_newpath')

    def test_header_injection(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/header_injection')
        self.assertEqual(response.body, b'ok')

    def test_get_argument(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/get_argument?foo=bar')
        self.assertEqual(response.body, b'bar')
        response = self.fetch('/get_argument?foo=')
        self.assertEqual(response.body, b'')
        response = self.fetch('/get_argument')
        self.assertEqual(response.body, b'default')
        body = urllib.parse.urlencode(dict(foo='hello'))
        response = self.fetch('/get_argument?foo=bar', method='POST', body=body)
        self.assertEqual(response.body, b'hello')
        response = self.fetch('/get_arguments?foo=bar', method='POST', body=body)
        self.assertEqual(json_decode(response.body), dict(default=['bar', 'hello'], query=['bar'], body=['hello']))

    def test_get_query_arguments(self):
        if False:
            for i in range(10):
                print('nop')
        body = urllib.parse.urlencode(dict(foo='hello'))
        response = self.fetch('/get_argument?source=query&foo=bar', method='POST', body=body)
        self.assertEqual(response.body, b'bar')
        response = self.fetch('/get_argument?source=query&foo=', method='POST', body=body)
        self.assertEqual(response.body, b'')
        response = self.fetch('/get_argument?source=query', method='POST', body=body)
        self.assertEqual(response.body, b'default')

    def test_get_body_arguments(self):
        if False:
            while True:
                i = 10
        body = urllib.parse.urlencode(dict(foo='bar'))
        response = self.fetch('/get_argument?source=body&foo=hello', method='POST', body=body)
        self.assertEqual(response.body, b'bar')
        body = urllib.parse.urlencode(dict(foo=''))
        response = self.fetch('/get_argument?source=body&foo=hello', method='POST', body=body)
        self.assertEqual(response.body, b'')
        body = urllib.parse.urlencode(dict())
        response = self.fetch('/get_argument?source=body&foo=hello', method='POST', body=body)
        self.assertEqual(response.body, b'default')

    def test_no_gzip(self):
        if False:
            print('Hello World!')
        response = self.fetch('/get_argument')
        self.assertNotIn('Accept-Encoding', response.headers.get('Vary', ''))
        self.assertNotIn('gzip', response.headers.get('Content-Encoding', ''))

class NonWSGIWebTests(WebTestCase):

    def get_handlers(self):
        if False:
            return 10
        return [('/empty_flush', EmptyFlushCallbackHandler)]

    def test_empty_flush(self):
        if False:
            return 10
        response = self.fetch('/empty_flush')
        self.assertEqual(response.body, b'ok')

class ErrorResponseTest(WebTestCase):

    def get_handlers(self):
        if False:
            return 10

        class DefaultHandler(RequestHandler):

            def get(self):
                if False:
                    while True:
                        i = 10
                if self.get_argument('status', None):
                    raise HTTPError(int(self.get_argument('status')))
                1 / 0

        class WriteErrorHandler(RequestHandler):

            def get(self):
                if False:
                    i = 10
                    return i + 15
                if self.get_argument('status', None):
                    self.send_error(int(self.get_argument('status')))
                else:
                    1 / 0

            def write_error(self, status_code, **kwargs):
                if False:
                    return 10
                self.set_header('Content-Type', 'text/plain')
                if 'exc_info' in kwargs:
                    self.write('Exception: %s' % kwargs['exc_info'][0].__name__)
                else:
                    self.write('Status: %d' % status_code)

        class FailedWriteErrorHandler(RequestHandler):

            def get(self):
                if False:
                    print('Hello World!')
                1 / 0

            def write_error(self, status_code, **kwargs):
                if False:
                    print('Hello World!')
                raise Exception('exception in write_error')
        return [url('/default', DefaultHandler), url('/write_error', WriteErrorHandler), url('/failed_write_error', FailedWriteErrorHandler)]

    def test_default(self):
        if False:
            return 10
        with ExpectLog(app_log, 'Uncaught exception'):
            response = self.fetch('/default')
            self.assertEqual(response.code, 500)
            self.assertTrue(b'500: Internal Server Error' in response.body)
            response = self.fetch('/default?status=503')
            self.assertEqual(response.code, 503)
            self.assertTrue(b'503: Service Unavailable' in response.body)
            response = self.fetch('/default?status=435')
            self.assertEqual(response.code, 435)
            self.assertTrue(b'435: Unknown' in response.body)

    def test_write_error(self):
        if False:
            for i in range(10):
                print('nop')
        with ExpectLog(app_log, 'Uncaught exception'):
            response = self.fetch('/write_error')
            self.assertEqual(response.code, 500)
            self.assertEqual(b'Exception: ZeroDivisionError', response.body)
            response = self.fetch('/write_error?status=503')
            self.assertEqual(response.code, 503)
            self.assertEqual(b'Status: 503', response.body)

    def test_failed_write_error(self):
        if False:
            for i in range(10):
                print('nop')
        with ExpectLog(app_log, 'Uncaught exception'):
            response = self.fetch('/failed_write_error')
            self.assertEqual(response.code, 500)
            self.assertEqual(b'', response.body)

class StaticFileTest(WebTestCase):
    robots_txt_hash = b'63a36e950e134b5217e33c763e88840c10a07d80e6057d92b9ac97508de7fb1fa6f0e9b7531e169657165ea764e8963399cb6d921ffe6078425aaafe54c04563'
    static_dir = os.path.join(os.path.dirname(__file__), 'static')

    def get_handlers(self):
        if False:
            return 10

        class StaticUrlHandler(RequestHandler):

            def get(self, path):
                if False:
                    while True:
                        i = 10
                with_v = int(self.get_argument('include_version', '1'))
                self.write(self.static_url(path, include_version=with_v))

        class AbsoluteStaticUrlHandler(StaticUrlHandler):
            include_host = True

        class OverrideStaticUrlHandler(RequestHandler):

            def get(self, path):
                if False:
                    while True:
                        i = 10
                do_include = bool(self.get_argument('include_host'))
                self.include_host = not do_include
                regular_url = self.static_url(path)
                override_url = self.static_url(path, include_host=do_include)
                if override_url == regular_url:
                    return self.write(str(False))
                protocol = self.request.protocol + '://'
                protocol_length = len(protocol)
                check_regular = regular_url.find(protocol, 0, protocol_length)
                check_override = override_url.find(protocol, 0, protocol_length)
                if do_include:
                    result = check_override == 0 and check_regular == -1
                else:
                    result = check_override == -1 and check_regular == 0
                self.write(str(result))
        return [('/static_url/(.*)', StaticUrlHandler), ('/abs_static_url/(.*)', AbsoluteStaticUrlHandler), ('/override_static_url/(.*)', OverrideStaticUrlHandler), ('/root_static/(.*)', StaticFileHandler, dict(path='/'))]

    def get_app_kwargs(self):
        if False:
            print('Hello World!')
        return dict(static_path=relpath('static'))

    def test_static_files(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/robots.txt')
        self.assertTrue(b'Disallow: /' in response.body)
        response = self.fetch('/static/robots.txt')
        self.assertTrue(b'Disallow: /' in response.body)
        self.assertEqual(response.headers.get('Content-Type'), 'text/plain')

    def test_static_files_cacheable(self):
        if False:
            return 10
        response = self.fetch('/robots.txt?v=12345')
        self.assertTrue(b'Disallow: /' in response.body)
        self.assertIn('Cache-Control', response.headers)
        self.assertIn('Expires', response.headers)

    def test_static_compressed_files(self):
        if False:
            print('Hello World!')
        response = self.fetch('/static/sample.xml.gz')
        self.assertEqual(response.headers.get('Content-Type'), 'application/gzip')
        response = self.fetch('/static/sample.xml.bz2')
        self.assertEqual(response.headers.get('Content-Type'), 'application/octet-stream')
        response = self.fetch('/static/sample.xml')
        self.assertTrue(response.headers.get('Content-Type') in set(('text/xml', 'application/xml')))

    def test_static_url(self):
        if False:
            print('Hello World!')
        response = self.fetch('/static_url/robots.txt')
        self.assertEqual(response.body, b'/static/robots.txt?v=' + self.robots_txt_hash)

    def test_absolute_static_url(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/abs_static_url/robots.txt')
        self.assertEqual(response.body, utf8(self.get_url('/')) + b'static/robots.txt?v=' + self.robots_txt_hash)

    def test_relative_version_exclusion(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/static_url/robots.txt?include_version=0')
        self.assertEqual(response.body, b'/static/robots.txt')

    def test_absolute_version_exclusion(self):
        if False:
            return 10
        response = self.fetch('/abs_static_url/robots.txt?include_version=0')
        self.assertEqual(response.body, utf8(self.get_url('/') + 'static/robots.txt'))

    def test_include_host_override(self):
        if False:
            print('Hello World!')
        self._trigger_include_host_check(False)
        self._trigger_include_host_check(True)

    def _trigger_include_host_check(self, include_host):
        if False:
            for i in range(10):
                print('nop')
        path = '/override_static_url/robots.txt?include_host=%s'
        response = self.fetch(path % int(include_host))
        self.assertEqual(response.body, utf8(str(True)))

    def get_and_head(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Performs a GET and HEAD request and returns the GET response.\n\n        Fails if any ``Content-*`` headers returned by the two requests\n        differ.\n        '
        head_response = self.fetch(*args, method='HEAD', **kwargs)
        get_response = self.fetch(*args, method='GET', **kwargs)
        content_headers = set()
        for h in itertools.chain(head_response.headers, get_response.headers):
            if h.startswith('Content-'):
                content_headers.add(h)
        for h in content_headers:
            self.assertEqual(head_response.headers.get(h), get_response.headers.get(h), '%s differs between GET (%s) and HEAD (%s)' % (h, head_response.headers.get(h), get_response.headers.get(h)))
        return get_response

    def test_static_304_if_modified_since(self):
        if False:
            i = 10
            return i + 15
        response1 = self.get_and_head('/static/robots.txt')
        response2 = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': response1.headers['Last-Modified']})
        self.assertEqual(response2.code, 304)
        self.assertTrue('Content-Length' not in response2.headers)

    def test_static_304_if_none_match(self):
        if False:
            return 10
        response1 = self.get_and_head('/static/robots.txt')
        response2 = self.get_and_head('/static/robots.txt', headers={'If-None-Match': response1.headers['Etag']})
        self.assertEqual(response2.code, 304)

    def test_static_304_etag_modified_bug(self):
        if False:
            while True:
                i = 10
        response1 = self.get_and_head('/static/robots.txt')
        response2 = self.get_and_head('/static/robots.txt', headers={'If-None-Match': '"MISMATCH"', 'If-Modified-Since': response1.headers['Last-Modified']})
        self.assertEqual(response2.code, 200)

    def test_static_if_modified_since_pre_epoch(self):
        if False:
            print('Hello World!')
        response = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': 'Fri, 01 Jan 1960 00:00:00 GMT'})
        self.assertEqual(response.code, 200)

    def test_static_if_modified_since_time_zone(self):
        if False:
            i = 10
            return i + 15
        stat = os.stat(relpath('static/robots.txt'))
        response = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': format_timestamp(stat.st_mtime - 1)})
        self.assertEqual(response.code, 200)
        response = self.get_and_head('/static/robots.txt', headers={'If-Modified-Since': format_timestamp(stat.st_mtime + 1)})
        self.assertEqual(response.code, 304)

    def test_static_etag(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_and_head('/static/robots.txt')
        self.assertEqual(utf8(response.headers.get('Etag')), b'"' + self.robots_txt_hash + b'"')

    def test_static_with_range(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=0-9'})
        self.assertEqual(response.code, 206)
        self.assertEqual(response.body, b'User-agent')
        self.assertEqual(utf8(response.headers.get('Etag')), b'"' + self.robots_txt_hash + b'"')
        self.assertEqual(response.headers.get('Content-Length'), '10')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 0-9/26')

    def test_static_with_range_full_file(self):
        if False:
            i = 10
            return i + 15
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=0-'})
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get('Content-Length'), '26')
        self.assertEqual(response.headers.get('Content-Range'), None)

    def test_static_with_range_full_past_end(self):
        if False:
            i = 10
            return i + 15
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=0-10000000'})
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get('Content-Length'), '26')
        self.assertEqual(response.headers.get('Content-Range'), None)

    def test_static_with_range_partial_past_end(self):
        if False:
            return 10
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=1-10000000'})
        self.assertEqual(response.code, 206)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()[1:]))
        self.assertEqual(response.headers.get('Content-Length'), '25')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 1-25/26')

    def test_static_with_range_end_edge(self):
        if False:
            while True:
                i = 10
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=22-'})
        self.assertEqual(response.body, b': /\n')
        self.assertEqual(response.headers.get('Content-Length'), '4')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 22-25/26')

    def test_static_with_range_neg_end(self):
        if False:
            return 10
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=-4'})
        self.assertEqual(response.body, b': /\n')
        self.assertEqual(response.headers.get('Content-Length'), '4')
        self.assertEqual(response.headers.get('Content-Range'), 'bytes 22-25/26')

    def test_static_with_range_neg_past_start(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=-1000000'})
        self.assertEqual(response.code, 200)
        robots_file_path = os.path.join(self.static_dir, 'robots.txt')
        with open(robots_file_path, encoding='utf-8') as f:
            self.assertEqual(response.body, utf8(f.read()))
        self.assertEqual(response.headers.get('Content-Length'), '26')
        self.assertEqual(response.headers.get('Content-Range'), None)

    def test_static_invalid_range(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'asdf'})
        self.assertEqual(response.code, 200)

    def test_static_unsatisfiable_range_zero_suffix(self):
        if False:
            while True:
                i = 10
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=-0'})
        self.assertEqual(response.headers.get('Content-Range'), 'bytes */26')
        self.assertEqual(response.code, 416)

    def test_static_unsatisfiable_range_invalid_start(self):
        if False:
            print('Hello World!')
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=26'})
        self.assertEqual(response.code, 416)
        self.assertEqual(response.headers.get('Content-Range'), 'bytes */26')

    def test_static_unsatisfiable_range_end_less_than_start(self):
        if False:
            return 10
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=10-3'})
        self.assertEqual(response.code, 416)
        self.assertEqual(response.headers.get('Content-Range'), 'bytes */26')

    def test_static_head(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/static/robots.txt', method='HEAD')
        self.assertEqual(response.code, 200)
        self.assertEqual(response.body, b'')
        self.assertEqual(response.headers['Content-Length'], '26')
        self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')

    def test_static_head_range(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/static/robots.txt', method='HEAD', headers={'Range': 'bytes=1-4'})
        self.assertEqual(response.code, 206)
        self.assertEqual(response.body, b'')
        self.assertEqual(response.headers['Content-Length'], '4')
        self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')

    def test_static_range_if_none_match(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_and_head('/static/robots.txt', headers={'Range': 'bytes=1-4', 'If-None-Match': b'"' + self.robots_txt_hash + b'"'})
        self.assertEqual(response.code, 304)
        self.assertEqual(response.body, b'')
        self.assertTrue('Content-Length' not in response.headers)
        self.assertEqual(utf8(response.headers['Etag']), b'"' + self.robots_txt_hash + b'"')

    def test_static_404(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.get_and_head('/static/blarg')
        self.assertEqual(response.code, 404)

    def test_path_traversal_protection(self):
        if False:
            print('Hello World!')
        self.http_client.close()
        self.http_client = SimpleAsyncHTTPClient()
        with ExpectLog(gen_log, '.*not in root static directory'):
            response = self.get_and_head('/static/../static_foo.txt')
        self.assertEqual(response.code, 403)

    @unittest.skipIf(os.name != 'posix', 'non-posix OS')
    def test_root_static_path(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/robots.txt')
        response = self.get_and_head('/root_static' + urllib.parse.quote(path))
        self.assertEqual(response.code, 200)

class StaticDefaultFilenameTest(WebTestCase):

    def get_app_kwargs(self):
        if False:
            i = 10
            return i + 15
        return dict(static_path=relpath('static'), static_handler_args=dict(default_filename='index.html'))

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')
        return []

    def test_static_default_filename(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/static/dir/', follow_redirects=False)
        self.assertEqual(response.code, 200)
        self.assertEqual(b'this is the index\n', response.body)

    def test_static_default_redirect(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/static/dir', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertTrue(response.headers['Location'].endswith('/static/dir/'))

class StaticDefaultFilenameRootTest(WebTestCase):

    def get_app_kwargs(self):
        if False:
            i = 10
            return i + 15
        return dict(static_path=os.path.abspath(relpath('static')), static_handler_args=dict(default_filename='index.html'), static_url_prefix='/')

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        return []

    def get_http_client(self):
        if False:
            print('Hello World!')
        return SimpleAsyncHTTPClient()

    def test_no_open_redirect(self):
        if False:
            print('Hello World!')
        test_dir = os.path.dirname(__file__)
        (drive, tail) = os.path.splitdrive(test_dir)
        if os.name == 'posix':
            self.assertEqual(tail, test_dir)
        else:
            test_dir = tail
        with ExpectLog(gen_log, '.*cannot redirect path with two initial slashes'):
            response = self.fetch(f'//evil.com/../{test_dir}/static/dir', follow_redirects=False)
        self.assertEqual(response.code, 403)

class StaticFileWithPathTest(WebTestCase):

    def get_app_kwargs(self):
        if False:
            i = 10
            return i + 15
        return dict(static_path=relpath('static'), static_handler_args=dict(default_filename='index.html'))

    def get_handlers(self):
        if False:
            while True:
                i = 10
        return [('/foo/(.*)', StaticFileHandler, {'path': relpath('templates/')})]

    def test_serve(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/foo/utf8.html')
        self.assertEqual(response.body, b'H\xc3\xa9llo\n')

class CustomStaticFileTest(WebTestCase):

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')

        class MyStaticFileHandler(StaticFileHandler):

            @classmethod
            def make_static_url(cls, settings, path):
                if False:
                    while True:
                        i = 10
                version_hash = cls.get_version(settings, path)
                extension_index = path.rindex('.')
                before_version = path[:extension_index]
                after_version = path[extension_index + 1:]
                return '/static/%s.%s.%s' % (before_version, version_hash, after_version)

            def parse_url_path(self, url_path):
                if False:
                    return 10
                extension_index = url_path.rindex('.')
                version_index = url_path.rindex('.', 0, extension_index)
                return '%s%s' % (url_path[:version_index], url_path[extension_index:])

            @classmethod
            def get_absolute_path(cls, settings, path):
                if False:
                    for i in range(10):
                        print('nop')
                return 'CustomStaticFileTest:' + path

            def validate_absolute_path(self, root, absolute_path):
                if False:
                    for i in range(10):
                        print('nop')
                return absolute_path

            @classmethod
            def get_content(self, path, start=None, end=None):
                if False:
                    i = 10
                    return i + 15
                assert start is None and end is None
                if path == 'CustomStaticFileTest:foo.txt':
                    return b'bar'
                raise Exception('unexpected path %r' % path)

            def get_content_size(self):
                if False:
                    for i in range(10):
                        print('nop')
                if self.absolute_path == 'CustomStaticFileTest:foo.txt':
                    return 3
                raise Exception('unexpected path %r' % self.absolute_path)

            def get_modified_time(self):
                if False:
                    i = 10
                    return i + 15
                return None

            @classmethod
            def get_version(cls, settings, path):
                if False:
                    i = 10
                    return i + 15
                return '42'

        class StaticUrlHandler(RequestHandler):

            def get(self, path):
                if False:
                    return 10
                self.write(self.static_url(path))
        self.static_handler_class = MyStaticFileHandler
        return [('/static_url/(.*)', StaticUrlHandler)]

    def get_app_kwargs(self):
        if False:
            i = 10
            return i + 15
        return dict(static_path='dummy', static_handler_class=self.static_handler_class)

    def test_serve(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/static/foo.42.txt')
        self.assertEqual(response.body, b'bar')

    def test_static_url(self):
        if False:
            while True:
                i = 10
        with ExpectLog(gen_log, 'Could not open static file', required=False):
            response = self.fetch('/static_url/foo.txt')
            self.assertEqual(response.body, b'/static/foo.42.txt')

class HostMatchingTest(WebTestCase):

    class Handler(RequestHandler):

        def initialize(self, reply):
            if False:
                i = 10
                return i + 15
            self.reply = reply

        def get(self):
            if False:
                while True:
                    i = 10
            self.write(self.reply)

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        return [('/foo', HostMatchingTest.Handler, {'reply': 'wildcard'})]

    def test_host_matching(self):
        if False:
            return 10
        self.app.add_handlers('www.example.com', [('/foo', HostMatchingTest.Handler, {'reply': '[0]'})])
        self.app.add_handlers('www\\.example\\.com', [('/bar', HostMatchingTest.Handler, {'reply': '[1]'})])
        self.app.add_handlers('www.example.com', [('/baz', HostMatchingTest.Handler, {'reply': '[2]'})])
        self.app.add_handlers('www.e.*e.com', [('/baz', HostMatchingTest.Handler, {'reply': '[3]'})])
        response = self.fetch('/foo')
        self.assertEqual(response.body, b'wildcard')
        response = self.fetch('/bar')
        self.assertEqual(response.code, 404)
        response = self.fetch('/baz')
        self.assertEqual(response.code, 404)
        response = self.fetch('/foo', headers={'Host': 'www.example.com'})
        self.assertEqual(response.body, b'[0]')
        response = self.fetch('/bar', headers={'Host': 'www.example.com'})
        self.assertEqual(response.body, b'[1]')
        response = self.fetch('/baz', headers={'Host': 'www.example.com'})
        self.assertEqual(response.body, b'[2]')
        response = self.fetch('/baz', headers={'Host': 'www.exe.com'})
        self.assertEqual(response.body, b'[3]')

class DefaultHostMatchingTest(WebTestCase):

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        return []

    def get_app_kwargs(self):
        if False:
            print('Hello World!')
        return {'default_host': 'www.example.com'}

    def test_default_host_matching(self):
        if False:
            print('Hello World!')
        self.app.add_handlers('www.example.com', [('/foo', HostMatchingTest.Handler, {'reply': '[0]'})])
        self.app.add_handlers('www\\.example\\.com', [('/bar', HostMatchingTest.Handler, {'reply': '[1]'})])
        self.app.add_handlers('www.test.com', [('/baz', HostMatchingTest.Handler, {'reply': '[2]'})])
        response = self.fetch('/foo')
        self.assertEqual(response.body, b'[0]')
        response = self.fetch('/bar')
        self.assertEqual(response.body, b'[1]')
        response = self.fetch('/baz')
        self.assertEqual(response.code, 404)
        response = self.fetch('/foo', headers={'X-Real-Ip': '127.0.0.1'})
        self.assertEqual(response.code, 404)
        self.app.default_host = 'www.test.com'
        response = self.fetch('/baz')
        self.assertEqual(response.body, b'[2]')

class NamedURLSpecGroupsTest(WebTestCase):

    def get_handlers(self):
        if False:
            return 10

        class EchoHandler(RequestHandler):

            def get(self, path):
                if False:
                    print('Hello World!')
                self.write(path)
        return [('/str/(?P<path>.*)', EchoHandler), ('/unicode/(?P<path>.*)', EchoHandler)]

    def test_named_urlspec_groups(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/str/foo')
        self.assertEqual(response.body, b'foo')
        response = self.fetch('/unicode/bar')
        self.assertEqual(response.body, b'bar')

class ClearHeaderTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            self.set_header('h1', 'foo')
            self.set_header('h2', 'bar')
            self.clear_header('h1')
            self.clear_header('nonexistent')

    def test_clear_header(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/')
        self.assertTrue('h1' not in response.headers)
        self.assertEqual(response.headers['h2'], 'bar')

class Header204Test(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                return 10
            self.set_status(204)
            self.finish()

    def test_204_headers(self):
        if False:
            print('Hello World!')
        response = self.fetch('/')
        self.assertEqual(response.code, 204)
        self.assertNotIn('Content-Length', response.headers)
        self.assertNotIn('Transfer-Encoding', response.headers)

class Header304Test(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            self.set_header('Content-Language', 'en_US')
            self.write('hello')

    def test_304_headers(self):
        if False:
            i = 10
            return i + 15
        response1 = self.fetch('/')
        self.assertEqual(response1.headers['Content-Length'], '5')
        self.assertEqual(response1.headers['Content-Language'], 'en_US')
        response2 = self.fetch('/', headers={'If-None-Match': response1.headers['Etag']})
        self.assertEqual(response2.code, 304)
        self.assertTrue('Content-Length' not in response2.headers)
        self.assertTrue('Content-Language' not in response2.headers)
        self.assertTrue('Transfer-Encoding' not in response2.headers)

class StatusReasonTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                i = 10
                return i + 15
            reason = self.request.arguments.get('reason', [])
            self.set_status(int(self.get_argument('code')), reason=to_unicode(reason[0]) if reason else None)

    def get_http_client(self):
        if False:
            while True:
                i = 10
        return SimpleAsyncHTTPClient()

    def test_status(self):
        if False:
            return 10
        response = self.fetch('/?code=304')
        self.assertEqual(response.code, 304)
        self.assertEqual(response.reason, 'Not Modified')
        response = self.fetch('/?code=304&reason=Foo')
        self.assertEqual(response.code, 304)
        self.assertEqual(response.reason, 'Foo')
        response = self.fetch('/?code=682&reason=Bar')
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, 'Bar')
        response = self.fetch('/?code=682')
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, 'Unknown')

class DateHeaderTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            self.write('hello')

    def test_date_header(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/')
        header_date = email.utils.parsedate_to_datetime(response.headers['Date'])
        self.assertTrue(header_date - datetime.datetime.now(datetime.timezone.utc) < datetime.timedelta(seconds=2))

class RaiseWithReasonTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            raise HTTPError(682, reason='Foo')

    def get_http_client(self):
        if False:
            while True:
                i = 10
        return SimpleAsyncHTTPClient()

    def test_raise_with_reason(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/')
        self.assertEqual(response.code, 682)
        self.assertEqual(response.reason, 'Foo')
        self.assertIn(b'682: Foo', response.body)

    def test_httperror_str(self):
        if False:
            return 10
        self.assertEqual(str(HTTPError(682, reason='Foo')), 'HTTP 682: Foo')

    def test_httperror_str_from_httputil(self):
        if False:
            print('Hello World!')
        self.assertEqual(str(HTTPError(682)), 'HTTP 682: Unknown')

class ErrorHandlerXSRFTest(WebTestCase):

    def get_handlers(self):
        if False:
            return 10
        return [('/error', ErrorHandler, dict(status_code=417))]

    def get_app_kwargs(self):
        if False:
            while True:
                i = 10
        return dict(xsrf_cookies=True)

    def test_error_xsrf(self):
        if False:
            return 10
        response = self.fetch('/error', method='POST', body='')
        self.assertEqual(response.code, 417)

    def test_404_xsrf(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/404', method='POST', body='')
        self.assertEqual(response.code, 404)

class GzipTestCase(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                for i in range(10):
                    print('nop')
            for v in self.get_arguments('vary'):
                self.add_header('Vary', v)
            self.write('hello world' + '!' * GZipContentEncoding.MIN_LENGTH)

    def get_app_kwargs(self):
        if False:
            while True:
                i = 10
        return dict(gzip=True, static_path=os.path.join(os.path.dirname(__file__), 'static'))

    def assert_compressed(self, response):
        if False:
            while True:
                i = 10
        self.assertEqual(response.headers.get('Content-Encoding', response.headers.get('X-Consumed-Content-Encoding')), 'gzip')

    def test_gzip(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/')
        self.assert_compressed(response)
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')

    def test_gzip_static(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/robots.txt')
        self.assert_compressed(response)
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')

    def test_gzip_not_requested(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/', use_gzip=False)
        self.assertNotIn('Content-Encoding', response.headers)
        self.assertEqual(response.headers['Vary'], 'Accept-Encoding')

    def test_vary_already_present(self):
        if False:
            return 10
        response = self.fetch('/?vary=Accept-Language')
        self.assert_compressed(response)
        self.assertEqual([s.strip() for s in response.headers['Vary'].split(',')], ['Accept-Language', 'Accept-Encoding'])

    def test_vary_already_present_multiple(self):
        if False:
            print('Hello World!')
        response = self.fetch('/?vary=Accept-Language&vary=Cookie')
        self.assert_compressed(response)
        self.assertEqual([s.strip() for s in response.headers['Vary'].split(',')], ['Accept-Language', 'Cookie', 'Accept-Encoding'])

class PathArgsInPrepareTest(WebTestCase):

    class Handler(RequestHandler):

        def prepare(self):
            if False:
                for i in range(10):
                    print('nop')
            self.write(dict(args=self.path_args, kwargs=self.path_kwargs))

        def get(self, path):
            if False:
                return 10
            assert path == 'foo'
            self.finish()

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        return [('/pos/(.*)', self.Handler), ('/kw/(?P<path>.*)', self.Handler)]

    def test_pos(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/pos/foo')
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {'args': ['foo'], 'kwargs': {}})

    def test_kw(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/kw/foo')
        response.rethrow()
        data = json_decode(response.body)
        self.assertEqual(data, {'args': [], 'kwargs': {'path': 'foo'}})

class ClearAllCookiesTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                print('Hello World!')
            self.clear_all_cookies()
            self.write('ok')

    def test_clear_all_cookies(self):
        if False:
            return 10
        response = self.fetch('/', headers={'Cookie': 'foo=bar; baz=xyzzy'})
        set_cookies = sorted(response.headers.get_list('Set-Cookie'))
        self.assertTrue(set_cookies[0].startswith('baz=;') or set_cookies[0].startswith('baz="";'))
        self.assertTrue(set_cookies[1].startswith('foo=;') or set_cookies[1].startswith('foo="";'))

class PermissionError(Exception):
    pass

class ExceptionHandlerTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            exc = self.get_argument('exc')
            if exc == 'http':
                raise HTTPError(410, 'no longer here')
            elif exc == 'zero':
                1 / 0
            elif exc == 'permission':
                raise PermissionError('not allowed')

        def write_error(self, status_code, **kwargs):
            if False:
                while True:
                    i = 10
            if 'exc_info' in kwargs:
                (typ, value, tb) = kwargs['exc_info']
                if isinstance(value, PermissionError):
                    self.set_status(403)
                    self.write('PermissionError')
                    return
            RequestHandler.write_error(self, status_code, **kwargs)

        def log_exception(self, typ, value, tb):
            if False:
                print('Hello World!')
            if isinstance(value, PermissionError):
                app_log.warning('custom logging for PermissionError: %s', value.args[0])
            else:
                RequestHandler.log_exception(self, typ, value, tb)

    def test_http_error(self):
        if False:
            for i in range(10):
                print('nop')
        with ExpectLog(gen_log, '.*no longer here'):
            response = self.fetch('/?exc=http')
            self.assertEqual(response.code, 410)

    def test_unknown_error(self):
        if False:
            print('Hello World!')
        with ExpectLog(app_log, 'Uncaught exception'):
            response = self.fetch('/?exc=zero')
            self.assertEqual(response.code, 500)

    def test_known_error(self):
        if False:
            print('Hello World!')
        with ExpectLog(app_log, 'custom logging for PermissionError: not allowed'):
            response = self.fetch('/?exc=permission')
            self.assertEqual(response.code, 403)

class BuggyLoggingTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                print('Hello World!')
            1 / 0

        def log_exception(self, typ, value, tb):
            if False:
                return 10
            1 / 0

    def test_buggy_log_exception(self):
        if False:
            i = 10
            return i + 15
        with ExpectLog(app_log, '.*'):
            self.fetch('/')

class UIMethodUIModuleTest(SimpleHandlerTestCase):
    """Test that UI methods and modules are created correctly and
    associated with the handler.
    """

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            self.render('foo.html')

        def value(self):
            if False:
                return 10
            return self.get_argument('value')

    def get_app_kwargs(self):
        if False:
            while True:
                i = 10

        def my_ui_method(handler, x):
            if False:
                for i in range(10):
                    print('nop')
            return 'In my_ui_method(%s) with handler value %s.' % (x, handler.value())

        class MyModule(UIModule):

            def render(self, x):
                if False:
                    i = 10
                    return i + 15
                return 'In MyModule(%s) with handler value %s.' % (x, typing.cast(UIMethodUIModuleTest.Handler, self.handler).value())
        loader = DictLoader({'foo.html': '{{ my_ui_method(42) }} {% module MyModule(123) %}'})
        return dict(template_loader=loader, ui_methods={'my_ui_method': my_ui_method}, ui_modules={'MyModule': MyModule})

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def test_ui_method(self):
        if False:
            return 10
        response = self.fetch('/?value=asdf')
        self.assertEqual(response.body, b'In my_ui_method(42) with handler value asdf. In MyModule(123) with handler value asdf.')

class GetArgumentErrorTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            try:
                self.get_argument('foo')
                self.write({})
            except MissingArgumentError as e:
                self.write({'arg_name': e.arg_name, 'log_message': e.log_message})

    def test_catch_error(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/')
        self.assertEqual(json_decode(response.body), {'arg_name': 'foo', 'log_message': 'Missing argument foo'})

class SetLazyPropertiesTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def prepare(self):
            if False:
                while True:
                    i = 10
            self.current_user = 'Ben'
            self.locale = locale.get('en_US')

        def get_user_locale(self):
            if False:
                for i in range(10):
                    print('nop')
            raise NotImplementedError()

        def get_current_user(self):
            if False:
                while True:
                    i = 10
            raise NotImplementedError()

        def get(self):
            if False:
                return 10
            self.write('Hello %s (%s)' % (self.current_user, self.locale.code))

    def test_set_properties(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/')
        self.assertEqual(response.body, b'Hello Ben (en_US)')

class GetCurrentUserTest(WebTestCase):

    def get_app_kwargs(self):
        if False:
            for i in range(10):
                print('nop')

        class WithoutUserModule(UIModule):

            def render(self):
                if False:
                    i = 10
                    return i + 15
                return ''

        class WithUserModule(UIModule):

            def render(self):
                if False:
                    for i in range(10):
                        print('nop')
                return str(self.current_user)
        loader = DictLoader({'without_user.html': '', 'with_user.html': '{{ current_user }}', 'without_user_module.html': '{% module WithoutUserModule() %}', 'with_user_module.html': '{% module WithUserModule() %}'})
        return dict(template_loader=loader, ui_modules={'WithUserModule': WithUserModule, 'WithoutUserModule': WithoutUserModule})

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        RequestHandler._template_loaders.clear()

    def get_handlers(self):
        if False:
            i = 10
            return i + 15

        class CurrentUserHandler(RequestHandler):

            def prepare(self):
                if False:
                    while True:
                        i = 10
                self.has_loaded_current_user = False

            def get_current_user(self):
                if False:
                    return 10
                self.has_loaded_current_user = True
                return ''

        class WithoutUserHandler(CurrentUserHandler):

            def get(self):
                if False:
                    return 10
                self.render_string('without_user.html')
                self.finish(str(self.has_loaded_current_user))

        class WithUserHandler(CurrentUserHandler):

            def get(self):
                if False:
                    print('Hello World!')
                self.render_string('with_user.html')
                self.finish(str(self.has_loaded_current_user))

        class CurrentUserModuleHandler(CurrentUserHandler):

            def get_template_namespace(self):
                if False:
                    print('Hello World!')
                return self.ui

        class WithoutUserModuleHandler(CurrentUserModuleHandler):

            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.render_string('without_user_module.html')
                self.finish(str(self.has_loaded_current_user))

        class WithUserModuleHandler(CurrentUserModuleHandler):

            def get(self):
                if False:
                    i = 10
                    return i + 15
                self.render_string('with_user_module.html')
                self.finish(str(self.has_loaded_current_user))
        return [('/without_user', WithoutUserHandler), ('/with_user', WithUserHandler), ('/without_user_module', WithoutUserModuleHandler), ('/with_user_module', WithUserModuleHandler)]

    @unittest.skip('needs fix')
    def test_get_current_user_is_lazy(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/without_user')
        self.assertEqual(response.body, b'False')

    def test_get_current_user_works(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/with_user')
        self.assertEqual(response.body, b'True')

    def test_get_current_user_from_ui_module_is_lazy(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/without_user_module')
        self.assertEqual(response.body, b'False')

    def test_get_current_user_from_ui_module_works(self):
        if False:
            return 10
        response = self.fetch('/with_user_module')
        self.assertEqual(response.body, b'True')

class UnimplementedHTTPMethodsTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):
        pass

    def test_unimplemented_standard_methods(self):
        if False:
            for i in range(10):
                print('nop')
        for method in ['HEAD', 'GET', 'DELETE', 'OPTIONS']:
            response = self.fetch('/', method=method)
            self.assertEqual(response.code, 405)
        for method in ['POST', 'PUT']:
            response = self.fetch('/', method=method, body=b'')
            self.assertEqual(response.code, 405)

class UnimplementedNonStandardMethodsTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def other(self):
            if False:
                return 10
            self.write('other')

    def test_unimplemented_patch(self):
        if False:
            return 10
        response = self.fetch('/', method='PATCH', body=b'')
        self.assertEqual(response.code, 405)

    def test_unimplemented_other(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/', method='OTHER', allow_nonstandard_methods=True)
        self.assertEqual(response.code, 405)

class AllHTTPMethodsTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def method(self):
            if False:
                for i in range(10):
                    print('nop')
            assert self.request.method is not None
            self.write(self.request.method)
        get = delete = options = post = put = method

    def test_standard_methods(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/', method='HEAD')
        self.assertEqual(response.body, b'')
        for method in ['GET', 'DELETE', 'OPTIONS']:
            response = self.fetch('/', method=method)
            self.assertEqual(response.body, utf8(method))
        for method in ['POST', 'PUT']:
            response = self.fetch('/', method=method, body=b'')
            self.assertEqual(response.body, utf8(method))

class PatchMethodTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):
        SUPPORTED_METHODS = RequestHandler.SUPPORTED_METHODS + ('OTHER',)

        def patch(self):
            if False:
                while True:
                    i = 10
            self.write('patch')

        def other(self):
            if False:
                return 10
            self.write('other')

    def test_patch(self):
        if False:
            print('Hello World!')
        response = self.fetch('/', method='PATCH', body=b'')
        self.assertEqual(response.body, b'patch')

    def test_other(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/', method='OTHER', allow_nonstandard_methods=True)
        self.assertEqual(response.body, b'other')

class FinishInPrepareTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def prepare(self):
            if False:
                return 10
            self.finish('done')

        def get(self):
            if False:
                print('Hello World!')
            raise Exception('should not reach this method')

    def test_finish_in_prepare(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/')
        self.assertEqual(response.body, b'done')

class Default404Test(WebTestCase):

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')
        return [('/foo', RequestHandler)]

    def test_404(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/')
        self.assertEqual(response.code, 404)
        self.assertEqual(response.body, b'<html><title>404: Not Found</title><body>404: Not Found</body></html>')

class Custom404Test(WebTestCase):

    def get_handlers(self):
        if False:
            print('Hello World!')
        return [('/foo', RequestHandler)]

    def get_app_kwargs(self):
        if False:
            print('Hello World!')

        class Custom404Handler(RequestHandler):

            def get(self):
                if False:
                    for i in range(10):
                        print('nop')
                self.set_status(404)
                self.write('custom 404 response')
        return dict(default_handler_class=Custom404Handler)

    def test_404(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/')
        self.assertEqual(response.code, 404)
        self.assertEqual(response.body, b'custom 404 response')

class DefaultHandlerArgumentsTest(WebTestCase):

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')
        return [('/foo', RequestHandler)]

    def get_app_kwargs(self):
        if False:
            i = 10
            return i + 15
        return dict(default_handler_class=ErrorHandler, default_handler_args=dict(status_code=403))

    def test_403(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/')
        self.assertEqual(response.code, 403)

class HandlerByNameTest(WebTestCase):

    def get_handlers(self):
        if False:
            i = 10
            return i + 15
        return [('/hello1', HelloHandler), ('/hello2', 'tornado.test.web_test.HelloHandler'), url('/hello3', 'tornado.test.web_test.HelloHandler')]

    def test_handler_by_name(self):
        if False:
            while True:
                i = 10
        resp = self.fetch('/hello1')
        self.assertEqual(resp.body, b'hello')
        resp = self.fetch('/hello2')
        self.assertEqual(resp.body, b'hello')
        resp = self.fetch('/hello3')
        self.assertEqual(resp.body, b'hello')

class StreamingRequestBodyTest(WebTestCase):

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')

        @stream_request_body
        class StreamingBodyHandler(RequestHandler):

            def initialize(self, test):
                if False:
                    while True:
                        i = 10
                self.test = test

            def prepare(self):
                if False:
                    i = 10
                    return i + 15
                self.test.prepared.set_result(None)

            def data_received(self, data):
                if False:
                    return 10
                self.test.data.set_result(data)

            def get(self):
                if False:
                    i = 10
                    return i + 15
                self.test.finished.set_result(None)
                self.write({})

        @stream_request_body
        class EarlyReturnHandler(RequestHandler):

            def prepare(self):
                if False:
                    return 10
                raise HTTPError(401)

        @stream_request_body
        class CloseDetectionHandler(RequestHandler):

            def initialize(self, test):
                if False:
                    while True:
                        i = 10
                self.test = test

            def on_connection_close(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().on_connection_close()
                self.test.close_future.set_result(None)
        return [('/stream_body', StreamingBodyHandler, dict(test=self)), ('/early_return', EarlyReturnHandler), ('/close_detection', CloseDetectionHandler, dict(test=self))]

    def connect(self, url, connection_close):
        if False:
            i = 10
            return i + 15
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM, 0)
        s.connect(('127.0.0.1', self.get_http_port()))
        stream = IOStream(s)
        stream.write(b'GET ' + url + b' HTTP/1.1\r\n')
        if connection_close:
            stream.write(b'Connection: close\r\n')
        stream.write(b'Transfer-Encoding: chunked\r\n\r\n')
        return stream

    @gen_test
    def test_streaming_body(self):
        if False:
            while True:
                i = 10
        self.prepared = Future()
        self.data = Future()
        self.finished = Future()
        stream = self.connect(b'/stream_body', connection_close=True)
        yield self.prepared
        stream.write(b'4\r\nasdf\r\n')
        data = (yield self.data)
        self.assertEqual(data, b'asdf')
        self.data = Future()
        stream.write(b'4\r\nqwer\r\n')
        data = (yield self.data)
        self.assertEqual(data, b'qwer')
        stream.write(b'0\r\n\r\n')
        yield self.finished
        data = (yield stream.read_until_close())
        self.assertTrue(data.endswith(b'{}'))
        stream.close()

    @gen_test
    def test_early_return(self):
        if False:
            while True:
                i = 10
        stream = self.connect(b'/early_return', connection_close=False)
        data = (yield stream.read_until_close())
        self.assertTrue(data.startswith(b'HTTP/1.1 401'))

    @gen_test
    def test_early_return_with_data(self):
        if False:
            print('Hello World!')
        stream = self.connect(b'/early_return', connection_close=False)
        stream.write(b'4\r\nasdf\r\n')
        data = (yield stream.read_until_close())
        self.assertTrue(data.startswith(b'HTTP/1.1 401'))

    @gen_test
    def test_close_during_upload(self):
        if False:
            while True:
                i = 10
        self.close_future = Future()
        stream = self.connect(b'/close_detection', connection_close=False)
        stream.close()
        yield self.close_future

@stream_request_body
class BaseFlowControlHandler(RequestHandler):

    def initialize(self, test):
        if False:
            while True:
                i = 10
        self.test = test
        self.method = None
        self.methods = []

    @contextlib.contextmanager
    def in_method(self, method):
        if False:
            while True:
                i = 10
        if self.method is not None:
            self.test.fail('entered method %s while in %s' % (method, self.method))
        self.method = method
        self.methods.append(method)
        try:
            yield
        finally:
            self.method = None

    @gen.coroutine
    def prepare(self):
        if False:
            for i in range(10):
                print('nop')
        self.methods.append('prepare')
        yield gen.moment

    @gen.coroutine
    def post(self):
        if False:
            return 10
        with self.in_method('post'):
            yield gen.moment
        self.write(dict(methods=self.methods))

class BaseStreamingRequestFlowControlTest(object):

    def get_httpserver_options(self):
        if False:
            i = 10
            return i + 15
        return dict(chunk_size=10, decompress_request=True)

    def get_http_client(self):
        if False:
            i = 10
            return i + 15
        return SimpleAsyncHTTPClient()

    def test_flow_control_fixed_body(self: typing.Any):
        if False:
            return 10
        response = self.fetch('/', body='abcdefghijklmnopqrstuvwxyz', method='POST')
        response.rethrow()
        self.assertEqual(json_decode(response.body), dict(methods=['prepare', 'data_received', 'data_received', 'data_received', 'post']))

    def test_flow_control_chunked_body(self: typing.Any):
        if False:
            i = 10
            return i + 15
        chunks = [b'abcd', b'efgh', b'ijkl']

        @gen.coroutine
        def body_producer(write):
            if False:
                i = 10
                return i + 15
            for i in chunks:
                yield write(i)
        response = self.fetch('/', body_producer=body_producer, method='POST')
        response.rethrow()
        self.assertEqual(json_decode(response.body), dict(methods=['prepare', 'data_received', 'data_received', 'data_received', 'post']))

    def test_flow_control_compressed_body(self: typing.Any):
        if False:
            print('Hello World!')
        bytesio = BytesIO()
        gzip_file = gzip.GzipFile(mode='w', fileobj=bytesio)
        gzip_file.write(b'abcdefghijklmnopqrstuvwxyz')
        gzip_file.close()
        compressed_body = bytesio.getvalue()
        response = self.fetch('/', body=compressed_body, method='POST', headers={'Content-Encoding': 'gzip'})
        response.rethrow()
        self.assertEqual(json_decode(response.body), dict(methods=['prepare', 'data_received', 'data_received', 'data_received', 'post']))

class DecoratedStreamingRequestFlowControlTest(BaseStreamingRequestFlowControlTest, WebTestCase):

    def get_handlers(self):
        if False:
            print('Hello World!')

        class DecoratedFlowControlHandler(BaseFlowControlHandler):

            @gen.coroutine
            def data_received(self, data):
                if False:
                    i = 10
                    return i + 15
                with self.in_method('data_received'):
                    yield gen.moment
        return [('/', DecoratedFlowControlHandler, dict(test=self))]

class NativeStreamingRequestFlowControlTest(BaseStreamingRequestFlowControlTest, WebTestCase):

    def get_handlers(self):
        if False:
            return 10

        class NativeFlowControlHandler(BaseFlowControlHandler):

            async def data_received(self, data):
                with self.in_method('data_received'):
                    import asyncio
                    await asyncio.sleep(0)
        return [('/', NativeFlowControlHandler, dict(test=self))]

class IncorrectContentLengthTest(SimpleHandlerTestCase):

    def get_handlers(self):
        if False:
            while True:
                i = 10
        test = self
        self.server_error = None

        class TooHigh(RequestHandler):

            def get(self):
                if False:
                    i = 10
                    return i + 15
                self.set_header('Content-Length', '42')
                try:
                    self.finish('ok')
                except Exception as e:
                    test.server_error = e
                    raise

        class TooLow(RequestHandler):

            def get(self):
                if False:
                    print('Hello World!')
                self.set_header('Content-Length', '2')
                try:
                    self.finish('hello')
                except Exception as e:
                    test.server_error = e
                    raise
        return [('/high', TooHigh), ('/low', TooLow)]

    def test_content_length_too_high(self):
        if False:
            for i in range(10):
                print('nop')
        with ExpectLog(app_log, '(Uncaught exception|Exception in callback)'):
            with ExpectLog(gen_log, '(Cannot send error response after headers written|Failed to flush partial response)'):
                with self.assertRaises(HTTPClientError):
                    self.fetch('/high', raise_error=True)
        self.assertEqual(str(self.server_error), 'Tried to write 40 bytes less than Content-Length')

    def test_content_length_too_low(self):
        if False:
            print('Hello World!')
        with ExpectLog(app_log, '(Uncaught exception|Exception in callback)'):
            with ExpectLog(gen_log, '(Cannot send error response after headers written|Failed to flush partial response)'):
                with self.assertRaises(HTTPClientError):
                    self.fetch('/low', raise_error=True)
        self.assertEqual(str(self.server_error), 'Tried to write more data than Content-Length')

class ClientCloseTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                i = 10
                return i + 15
            if self.request.version.startswith('HTTP/1'):
                self.request.connection.stream.close()
                self.write('hello')
            else:
                self.write('requires HTTP/1.x')

    def test_client_close(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaises((HTTPClientError, unittest.SkipTest)):
            response = self.fetch('/', raise_error=True)
            if response.body == b'requires HTTP/1.x':
                self.skipTest('requires HTTP/1.x')
            self.assertEqual(response.code, 599)

class SignedValueTest(unittest.TestCase):
    SECRET = "It's a secret to everybody"
    SECRET_DICT = {0: 'asdfbasdf', 1: '12312312', 2: '2342342'}

    def past(self):
        if False:
            return 10
        return self.present() - 86400 * 32

    def present(self):
        if False:
            return 10
        return 1300000000

    def test_known_values(self):
        if False:
            print('Hello World!')
        signed_v1 = create_signed_value(SignedValueTest.SECRET, 'key', 'value', version=1, clock=self.present)
        self.assertEqual(signed_v1, b'dmFsdWU=|1300000000|31c934969f53e48164c50768b40cbd7e2daaaa4f')
        signed_v2 = create_signed_value(SignedValueTest.SECRET, 'key', 'value', version=2, clock=self.present)
        self.assertEqual(signed_v2, b'2|1:0|10:1300000000|3:key|8:dmFsdWU=|3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152')
        signed_default = create_signed_value(SignedValueTest.SECRET, 'key', 'value', clock=self.present)
        self.assertEqual(signed_default, signed_v2)
        decoded_v1 = decode_signed_value(SignedValueTest.SECRET, 'key', signed_v1, min_version=1, clock=self.present)
        self.assertEqual(decoded_v1, b'value')
        decoded_v2 = decode_signed_value(SignedValueTest.SECRET, 'key', signed_v2, min_version=2, clock=self.present)
        self.assertEqual(decoded_v2, b'value')

    def test_name_swap(self):
        if False:
            print('Hello World!')
        signed1 = create_signed_value(SignedValueTest.SECRET, 'key1', 'value', clock=self.present)
        signed2 = create_signed_value(SignedValueTest.SECRET, 'key2', 'value', clock=self.present)
        decoded1 = decode_signed_value(SignedValueTest.SECRET, 'key2', signed1, clock=self.present)
        self.assertIs(decoded1, None)
        decoded2 = decode_signed_value(SignedValueTest.SECRET, 'key1', signed2, clock=self.present)
        self.assertIs(decoded2, None)

    def test_expired(self):
        if False:
            return 10
        signed = create_signed_value(SignedValueTest.SECRET, 'key1', 'value', clock=self.past)
        decoded_past = decode_signed_value(SignedValueTest.SECRET, 'key1', signed, clock=self.past)
        self.assertEqual(decoded_past, b'value')
        decoded_present = decode_signed_value(SignedValueTest.SECRET, 'key1', signed, clock=self.present)
        self.assertIs(decoded_present, None)

    def test_payload_tampering(self):
        if False:
            i = 10
            return i + 15
        sig = '3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152'

        def validate(prefix):
            if False:
                return 10
            return b'value' == decode_signed_value(SignedValueTest.SECRET, 'key', prefix + sig, clock=self.present)
        self.assertTrue(validate('2|1:0|10:1300000000|3:key|8:dmFsdWU=|'))
        self.assertFalse(validate('2|1:1|10:1300000000|3:key|8:dmFsdWU=|'))
        self.assertFalse(validate('2|1:0|10:130000000|3:key|8:dmFsdWU=|'))
        self.assertFalse(validate('2|1:0|10:1300000000|3:keey|8:dmFsdWU=|'))

    def test_signature_tampering(self):
        if False:
            i = 10
            return i + 15
        prefix = '2|1:0|10:1300000000|3:key|8:dmFsdWU=|'

        def validate(sig):
            if False:
                for i in range(10):
                    print('nop')
            return b'value' == decode_signed_value(SignedValueTest.SECRET, 'key', prefix + sig, clock=self.present)
        self.assertTrue(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152'))
        self.assertFalse(validate('0' * 32))
        self.assertFalse(validate('4d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e152'))
        self.assertFalse(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e153'))
        self.assertFalse(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e15'))
        self.assertFalse(validate('3d4e60b996ff9c5d5788e333a0cba6f238a22c6c0f94788870e1a9ecd482e1538'))

    def test_non_ascii(self):
        if False:
            for i in range(10):
                print('nop')
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET, 'key', value, clock=self.present)
        decoded = decode_signed_value(SignedValueTest.SECRET, 'key', signed, clock=self.present)
        self.assertEqual(value, decoded)

    def test_key_versioning_read_write_default_key(self):
        if False:
            print('Hello World!')
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=0)
        decoded = decode_signed_value(SignedValueTest.SECRET_DICT, 'key', signed, clock=self.present)
        self.assertEqual(value, decoded)

    def test_key_versioning_read_write_non_default_key(self):
        if False:
            return 10
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=1)
        decoded = decode_signed_value(SignedValueTest.SECRET_DICT, 'key', signed, clock=self.present)
        self.assertEqual(value, decoded)

    def test_key_versioning_invalid_key(self):
        if False:
            for i in range(10):
                print('nop')
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=0)
        newkeys = SignedValueTest.SECRET_DICT.copy()
        newkeys.pop(0)
        decoded = decode_signed_value(newkeys, 'key', signed, clock=self.present)
        self.assertEqual(None, decoded)

    def test_key_version_retrieval(self):
        if False:
            return 10
        value = b'\xe9'
        signed = create_signed_value(SignedValueTest.SECRET_DICT, 'key', value, clock=self.present, key_version=1)
        key_version = get_signature_key_version(signed)
        self.assertEqual(1, key_version)

class XSRFTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                print('Hello World!')
            version = int(self.get_argument('version', '2'))
            self.settings['xsrf_cookie_version'] = version
            self.write(self.xsrf_token)

        def post(self):
            if False:
                print('Hello World!')
            self.write('ok')

    def get_app_kwargs(self):
        if False:
            while True:
                i = 10
        return dict(xsrf_cookies=True)

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.xsrf_token = self.get_token()

    def get_token(self, old_token=None, version=None):
        if False:
            print('Hello World!')
        if old_token is not None:
            headers = self.cookie_headers(old_token)
        else:
            headers = None
        response = self.fetch('/' if version is None else '/?version=%d' % version, headers=headers)
        response.rethrow()
        return native_str(response.body)

    def cookie_headers(self, token=None):
        if False:
            print('Hello World!')
        if token is None:
            token = self.xsrf_token
        return {'Cookie': '_xsrf=' + token}

    def test_xsrf_fail_no_token(self):
        if False:
            while True:
                i = 10
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch('/', method='POST', body=b'')
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_body_no_cookie(self):
        if False:
            return 10
        with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)))
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_argument_invalid_format(self):
        if False:
            return 10
        with ExpectLog(gen_log, ".*'_xsrf' argument has invalid format"):
            response = self.fetch('/', method='POST', headers=self.cookie_headers(), body=urllib.parse.urlencode(dict(_xsrf='3|')))
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_cookie_invalid_format(self):
        if False:
            while True:
                i = 10
        with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
            response = self.fetch('/', method='POST', headers=self.cookie_headers(token='3|'), body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)))
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_cookie_no_body(self):
        if False:
            return 10
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch('/', method='POST', body=b'', headers=self.cookie_headers())
        self.assertEqual(response.code, 403)

    def test_xsrf_success_short_token(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf='deadbeef')), headers=self.cookie_headers(token='deadbeef'))
        self.assertEqual(response.code, 200)

    def test_xsrf_success_non_hex_token(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf='xoxo')), headers=self.cookie_headers(token='xoxo'))
        self.assertEqual(response.code, 200)

    def test_xsrf_success_post_body(self):
        if False:
            return 10
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), headers=self.cookie_headers())
        self.assertEqual(response.code, 200)

    def test_xsrf_success_query_string(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/?' + urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), method='POST', body=b'', headers=self.cookie_headers())
        self.assertEqual(response.code, 200)

    def test_xsrf_success_header(self):
        if False:
            print('Hello World!')
        response = self.fetch('/', method='POST', body=b'', headers=dict({'X-Xsrftoken': self.xsrf_token}, **self.cookie_headers()))
        self.assertEqual(response.code, 200)

    def test_distinct_tokens(self):
        if False:
            for i in range(10):
                print('nop')
        NUM_TOKENS = 10
        tokens = set()
        for i in range(NUM_TOKENS):
            tokens.add(self.get_token())
        self.assertEqual(len(tokens), NUM_TOKENS)

    def test_cross_user(self):
        if False:
            while True:
                i = 10
        token2 = self.get_token()
        for token in (self.xsrf_token, token2):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=token)), headers=self.cookie_headers(token))
            self.assertEqual(response.code, 200)
        for (cookie_token, body_token) in ((self.xsrf_token, token2), (token2, self.xsrf_token)):
            with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
                response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=body_token)), headers=self.cookie_headers(cookie_token))
            self.assertEqual(response.code, 403)

    def test_refresh_token(self):
        if False:
            i = 10
            return i + 15
        token = self.xsrf_token
        tokens_seen = set([token])
        for i in range(5):
            token = self.get_token(token)
            tokens_seen.add(token)
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), headers=self.cookie_headers(token))
            self.assertEqual(response.code, 200)
        self.assertEqual(len(tokens_seen), 6)

    def test_versioning(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(self.get_token(version=1), self.get_token(version=1))
        v1_token = self.get_token(version=1)
        for i in range(5):
            self.assertEqual(self.get_token(v1_token, version=1), v1_token)
        v2_token = self.get_token(v1_token)
        self.assertNotEqual(v1_token, v2_token)
        self.assertNotEqual(v2_token, self.get_token(v1_token))
        for (cookie_token, body_token) in ((v1_token, v2_token), (v2_token, v1_token)):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=body_token)), headers=self.cookie_headers(cookie_token))
            self.assertEqual(response.code, 200)

class XSRFCookieNameTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                return 10
            self.write(self.xsrf_token)

        def post(self):
            if False:
                i = 10
                return i + 15
            self.write('ok')

    def get_app_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return dict(xsrf_cookies=True, xsrf_cookie_name='__Host-xsrf', xsrf_cookie_kwargs={'secure': True})

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.xsrf_token = self.get_token()

    def get_token(self, old_token=None):
        if False:
            while True:
                i = 10
        if old_token is not None:
            headers = self.cookie_headers(old_token)
        else:
            headers = None
        response = self.fetch('/', headers=headers)
        response.rethrow()
        return native_str(response.body)

    def cookie_headers(self, token=None):
        if False:
            for i in range(10):
                print('nop')
        if token is None:
            token = self.xsrf_token
        return {'Cookie': '__Host-xsrf=' + token}

    def test_xsrf_fail_no_token(self):
        if False:
            return 10
        with ExpectLog(gen_log, ".*'_xsrf' argument missing"):
            response = self.fetch('/', method='POST', body=b'')
        self.assertEqual(response.code, 403)

    def test_xsrf_fail_body_no_cookie(self):
        if False:
            print('Hello World!')
        with ExpectLog(gen_log, '.*XSRF cookie does not match POST'):
            response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)))
        self.assertEqual(response.code, 403)

    def test_xsrf_success_post_body(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/', method='POST', body=urllib.parse.urlencode(dict(_xsrf=self.xsrf_token)), headers=self.cookie_headers())
        self.assertEqual(response.code, 200)

class XSRFCookieKwargsTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            self.write(self.xsrf_token)

    def get_app_kwargs(self):
        if False:
            for i in range(10):
                print('nop')
        return dict(xsrf_cookies=True, xsrf_cookie_kwargs=dict(httponly=True, expires_days=2))

    def test_xsrf_httponly(self):
        if False:
            for i in range(10):
                print('nop')
        response = self.fetch('/')
        self.assertIn('httponly;', response.headers['Set-Cookie'].lower())
        self.assertIn('expires=', response.headers['Set-Cookie'].lower())
        header = response.headers.get('Set-Cookie')
        assert header is not None
        match = re.match('.*; expires=(?P<expires>.+);.*', header)
        assert match is not None
        expires = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(days=2)
        header_expires = email.utils.parsedate_to_datetime(match.groupdict()['expires'])
        if header_expires.tzinfo is None:
            header_expires = header_expires.replace(tzinfo=datetime.timezone.utc)
        self.assertTrue(abs((expires - header_expires).total_seconds()) < 10)

class FinishExceptionTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                while True:
                    i = 10
            self.set_status(401)
            self.set_header('WWW-Authenticate', 'Basic realm="something"')
            if self.get_argument('finish_value', ''):
                raise Finish('authentication required')
            else:
                self.write('authentication required')
                raise Finish()

    def test_finish_exception(self):
        if False:
            return 10
        for u in ['/', '/?finish_value=1']:
            response = self.fetch(u)
            self.assertEqual(response.code, 401)
            self.assertEqual('Basic realm="something"', response.headers.get('WWW-Authenticate'))
            self.assertEqual(b'authentication required', response.body)

class DecoratorTest(WebTestCase):

    def get_handlers(self):
        if False:
            i = 10
            return i + 15

        class RemoveSlashHandler(RequestHandler):

            @removeslash
            def get(self):
                if False:
                    print('Hello World!')
                pass

        class AddSlashHandler(RequestHandler):

            @addslash
            def get(self):
                if False:
                    return 10
                pass
        return [('/removeslash/', RemoveSlashHandler), ('/addslash', AddSlashHandler)]

    def test_removeslash(self):
        if False:
            return 10
        response = self.fetch('/removeslash/', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/removeslash')
        response = self.fetch('/removeslash/?foo=bar', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/removeslash?foo=bar')

    def test_addslash(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/addslash', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/addslash/')
        response = self.fetch('/addslash?foo=bar', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/addslash/?foo=bar')

class CacheTest(WebTestCase):

    def get_handlers(self):
        if False:
            while True:
                i = 10

        class EtagHandler(RequestHandler):

            def get(self, computed_etag):
                if False:
                    while True:
                        i = 10
                self.write(computed_etag)

            def compute_etag(self):
                if False:
                    for i in range(10):
                        print('nop')
                return self._write_buffer[0]
        return [('/etag/(.*)', EtagHandler)]

    def test_wildcard_etag(self):
        if False:
            while True:
                i = 10
        computed_etag = '"xyzzy"'
        etags = '*'
        self._test_etag(computed_etag, etags, 304)

    def test_strong_etag_match(self):
        if False:
            print('Hello World!')
        computed_etag = '"xyzzy"'
        etags = '"xyzzy"'
        self._test_etag(computed_etag, etags, 304)

    def test_multiple_strong_etag_match(self):
        if False:
            return 10
        computed_etag = '"xyzzy1"'
        etags = '"xyzzy1", "xyzzy2"'
        self._test_etag(computed_etag, etags, 304)

    def test_strong_etag_not_match(self):
        if False:
            for i in range(10):
                print('nop')
        computed_etag = '"xyzzy"'
        etags = '"xyzzy1"'
        self._test_etag(computed_etag, etags, 200)

    def test_multiple_strong_etag_not_match(self):
        if False:
            return 10
        computed_etag = '"xyzzy"'
        etags = '"xyzzy1", "xyzzy2"'
        self._test_etag(computed_etag, etags, 200)

    def test_weak_etag_match(self):
        if False:
            i = 10
            return i + 15
        computed_etag = '"xyzzy1"'
        etags = 'W/"xyzzy1"'
        self._test_etag(computed_etag, etags, 304)

    def test_multiple_weak_etag_match(self):
        if False:
            i = 10
            return i + 15
        computed_etag = '"xyzzy2"'
        etags = 'W/"xyzzy1", W/"xyzzy2"'
        self._test_etag(computed_etag, etags, 304)

    def test_weak_etag_not_match(self):
        if False:
            while True:
                i = 10
        computed_etag = '"xyzzy2"'
        etags = 'W/"xyzzy1"'
        self._test_etag(computed_etag, etags, 200)

    def test_multiple_weak_etag_not_match(self):
        if False:
            for i in range(10):
                print('nop')
        computed_etag = '"xyzzy3"'
        etags = 'W/"xyzzy1", W/"xyzzy2"'
        self._test_etag(computed_etag, etags, 200)

    def _test_etag(self, computed_etag, etags, status_code):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/etag/' + computed_etag, headers={'If-None-Match': etags})
        self.assertEqual(response.code, status_code)

class RequestSummaryTest(SimpleHandlerTestCase):

    class Handler(RequestHandler):

        def get(self):
            if False:
                print('Hello World!')
            self.request.remote_ip = None
            self.finish(self._request_summary())

    def test_missing_remote_ip(self):
        if False:
            i = 10
            return i + 15
        resp = self.fetch('/')
        self.assertEqual(resp.body, b'GET / (None)')

class HTTPErrorTest(unittest.TestCase):

    def test_copy(self):
        if False:
            print('Hello World!')
        e = HTTPError(403, reason='Go away')
        e2 = copy.copy(e)
        self.assertIsNot(e, e2)
        self.assertEqual(e.status_code, e2.status_code)
        self.assertEqual(e.reason, e2.reason)

class ApplicationTest(AsyncTestCase):

    def test_listen(self):
        if False:
            i = 10
            return i + 15
        app = Application([])
        server = app.listen(0, address='127.0.0.1')
        server.stop()

class URLSpecReverseTest(unittest.TestCase):

    def test_reverse(self):
        if False:
            return 10
        self.assertEqual('/favicon.ico', url('/favicon\\.ico', None).reverse())
        self.assertEqual('/favicon.ico', url('^/favicon\\.ico$', None).reverse())

    def test_non_reversible(self):
        if False:
            return 10
        paths = ['^/api/v\\d+/foo/(\\w+)$']
        for path in paths:
            url_spec = url(path, None)
            try:
                result = url_spec.reverse()
                self.fail('did not get expected exception when reversing %s. result: %s' % (path, result))
            except ValueError:
                pass

    def test_reverse_arguments(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual('/api/v1/foo/bar', url('^/api/v1/foo/(\\w+)$', None).reverse('bar'))
        self.assertEqual('/api.v1/foo/5/icon.png', url('/api\\.v1/foo/([0-9]+)/icon\\.png', None).reverse(5))

class RedirectHandlerTest(WebTestCase):

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')
        return [('/src', WebRedirectHandler, {'url': '/dst'}), ('/src2', WebRedirectHandler, {'url': '/dst2?foo=bar'}), ('/(.*?)/(.*?)/(.*)', WebRedirectHandler, {'url': '/{1}/{0}/{2}'})]

    def test_basic_redirect(self):
        if False:
            print('Hello World!')
        response = self.fetch('/src', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/dst')

    def test_redirect_with_argument(self):
        if False:
            return 10
        response = self.fetch('/src?foo=bar', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/dst?foo=bar')

    def test_redirect_with_appending_argument(self):
        if False:
            return 10
        response = self.fetch('/src2?foo2=bar2', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/dst2?foo=bar&foo2=bar2')

    def test_redirect_pattern(self):
        if False:
            i = 10
            return i + 15
        response = self.fetch('/a/b/c', follow_redirects=False)
        self.assertEqual(response.code, 301)
        self.assertEqual(response.headers['Location'], '/b/a/c')

class AcceptLanguageTest(WebTestCase):
    """Test evaluation of Accept-Language header"""

    def get_handlers(self):
        if False:
            for i in range(10):
                print('nop')
        locale.load_gettext_translations(os.path.join(os.path.dirname(__file__), 'gettext_translations'), 'tornado_test')

        class AcceptLanguageHandler(RequestHandler):

            def get(self):
                if False:
                    print('Hello World!')
                self.set_header('Content-Language', self.get_browser_locale().code.replace('_', '-'))
                self.finish(b'')
        return [('/', AcceptLanguageHandler)]

    def test_accept_language(self):
        if False:
            print('Hello World!')
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR;q=0.9'})
        self.assertEqual(response.headers['Content-Language'], 'fr-FR')
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR; q=0.9'})
        self.assertEqual(response.headers['Content-Language'], 'fr-FR')

    def test_accept_language_ignore(self):
        if False:
            while True:
                i = 10
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR;q=0'})
        self.assertEqual(response.headers['Content-Language'], 'en-US')

    def test_accept_language_invalid(self):
        if False:
            return 10
        response = self.fetch('/', headers={'Accept-Language': 'fr-FR;q=-1'})
        self.assertEqual(response.headers['Content-Language'], 'en-US')