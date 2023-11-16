"""Basic tests for the cherrypy.Request object."""
from functools import wraps
import os
import sys
import types
import uuid
from http.client import IncompleteRead
import cherrypy
from cherrypy._cpcompat import ntou
from cherrypy.lib import httputil
from cherrypy.test import helper
localDir = os.path.dirname(__file__)
defined_http_methods = ('OPTIONS', 'GET', 'HEAD', 'POST', 'PUT', 'DELETE', 'TRACE', 'PROPFIND', 'PATCH')

class RequestObjectTests(helper.CPWebCase):

    @staticmethod
    def setup_server():
        if False:
            while True:
                i = 10

        class Root:

            @cherrypy.expose
            def index(self):
                if False:
                    print('Hello World!')
                return 'hello'

            @cherrypy.expose
            def scheme(self):
                if False:
                    while True:
                        i = 10
                return cherrypy.request.scheme

            @cherrypy.expose
            def created_example_com_3128(self):
                if False:
                    return 10
                'Handle CONNECT method.'
                cherrypy.response.status = 204

            @cherrypy.expose
            def body_example_com_3128(self):
                if False:
                    print('Hello World!')
                'Handle CONNECT method.'
                return cherrypy.request.method + 'ed to ' + cherrypy.request.path_info

            @cherrypy.expose
            def request_uuid4(self):
                if False:
                    print('Hello World!')
                return [str(cherrypy.request.unique_id), ' ', str(cherrypy.request.unique_id)]
        root = Root()

        class TestType(type):
            """Metaclass which automatically exposes all functions in each
            subclass, and adds an instance of the subclass as an attribute
            of root.
            """

            def __init__(cls, name, bases, dct):
                if False:
                    while True:
                        i = 10
                type.__init__(cls, name, bases, dct)
                for value in dct.values():
                    if isinstance(value, types.FunctionType):
                        value.exposed = True
                setattr(root, name.lower(), cls())
        Test = TestType('Test', (object,), {})

        class PathInfo(Test):

            def default(self, *args):
                if False:
                    while True:
                        i = 10
                return cherrypy.request.path_info

        class Params(Test):

            def index(self, thing):
                if False:
                    i = 10
                    return i + 15
                return repr(thing)

            def ismap(self, x, y):
                if False:
                    print('Hello World!')
                return 'Coordinates: %s, %s' % (x, y)

            @cherrypy.config(**{'request.query_string_encoding': 'latin1'})
            def default(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                return 'args: %s kwargs: %s' % (args, sorted(kwargs.items()))

        @cherrypy.expose
        class ParamErrorsCallable(object):

            def __call__(self):
                if False:
                    while True:
                        i = 10
                return 'data'

        def handler_dec(f):
            if False:
                while True:
                    i = 10

            @wraps(f)
            def wrapper(handler, *args, **kwargs):
                if False:
                    return 10
                return f(handler, *args, **kwargs)
            return wrapper

        class ParamErrors(Test):

            @cherrypy.expose
            def one_positional(self, param1):
                if False:
                    while True:
                        i = 10
                return 'data'

            @cherrypy.expose
            def one_positional_args(self, param1, *args):
                if False:
                    i = 10
                    return i + 15
                return 'data'

            @cherrypy.expose
            def one_positional_args_kwargs(self, param1, *args, **kwargs):
                if False:
                    print('Hello World!')
                return 'data'

            @cherrypy.expose
            def one_positional_kwargs(self, param1, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return 'data'

            @cherrypy.expose
            def no_positional(self):
                if False:
                    print('Hello World!')
                return 'data'

            @cherrypy.expose
            def no_positional_args(self, *args):
                if False:
                    return 10
                return 'data'

            @cherrypy.expose
            def no_positional_args_kwargs(self, *args, **kwargs):
                if False:
                    return 10
                return 'data'

            @cherrypy.expose
            def no_positional_kwargs(self, **kwargs):
                if False:
                    return 10
                return 'data'
            callable_object = ParamErrorsCallable()

            @cherrypy.expose
            def raise_type_error(self, **kwargs):
                if False:
                    print('Hello World!')
                raise TypeError('Client Error')

            @cherrypy.expose
            def raise_type_error_with_default_param(self, x, y=None):
                if False:
                    while True:
                        i = 10
                return '%d' % 'a'

            @cherrypy.expose
            @handler_dec
            def raise_type_error_decorated(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                raise TypeError('Client Error')

        def callable_error_page(status, **kwargs):
            if False:
                while True:
                    i = 10
            return "Error %s - Well, I'm very sorry but you haven't paid!" % status

        @cherrypy.config(**{'tools.log_tracebacks.on': True})
        class Error(Test):

            def reason_phrase(self):
                if False:
                    for i in range(10):
                        print('nop')
                raise cherrypy.HTTPError("410 Gone fishin'")

            @cherrypy.config(**{'error_page.404': os.path.join(localDir, 'static/index.html'), 'error_page.401': callable_error_page})
            def custom(self, err='404'):
                if False:
                    i = 10
                    return i + 15
                raise cherrypy.HTTPError(int(err), 'No, <b>really</b>, not found!')

            @cherrypy.config(**{'error_page.default': callable_error_page})
            def custom_default(self):
                if False:
                    print('Hello World!')
                return 1 + 'a'

            @cherrypy.config(**{'error_page.404': 'nonexistent.html'})
            def noexist(self):
                if False:
                    i = 10
                    return i + 15
                raise cherrypy.HTTPError(404, 'No, <b>really</b>, not found!')

            def page_method(self):
                if False:
                    i = 10
                    return i + 15
                raise ValueError()

            def page_yield(self):
                if False:
                    print('Hello World!')
                yield 'howdy'
                raise ValueError()

            @cherrypy.config(**{'response.stream': True})
            def page_streamed(self):
                if False:
                    print('Hello World!')
                yield 'word up'
                raise ValueError()
                yield 'very oops'

            @cherrypy.config(**{'request.show_tracebacks': False})
            def cause_err_in_finalize(self):
                if False:
                    for i in range(10):
                        print('nop')
                cherrypy.response.status = 'ZOO OK'

            @cherrypy.config(**{'request.throw_errors': True})
            def rethrow(self):
                if False:
                    while True:
                        i = 10
                'Test that an error raised here will be thrown out to\n                the server.\n                '
                raise ValueError()

        class Expect(Test):

            def expectation_failed(self):
                if False:
                    while True:
                        i = 10
                expect = cherrypy.request.headers.elements('Expect')
                if expect and expect[0].value != '100-continue':
                    raise cherrypy.HTTPError(400)
                raise cherrypy.HTTPError(417, 'Expectation Failed')

        class Headers(Test):

            def default(self, headername):
                if False:
                    while True:
                        i = 10
                'Spit back out the value for the requested header.'
                return cherrypy.request.headers[headername]

            def doubledheaders(self):
                if False:
                    for i in range(10):
                        print('nop')
                hMap = cherrypy.response.headers
                hMap['content-type'] = 'text/html'
                hMap['content-length'] = 18
                hMap['server'] = 'CherryPy headertest'
                hMap['location'] = '%s://%s:%s/headers/' % (cherrypy.request.local.ip, cherrypy.request.local.port, cherrypy.request.scheme)
                hMap['Expires'] = 'Thu, 01 Dec 2194 16:00:00 GMT'
                return 'double header test'

            def ifmatch(self):
                if False:
                    print('Hello World!')
                val = cherrypy.request.headers['If-Match']
                assert isinstance(val, str)
                cherrypy.response.headers['ETag'] = val
                return val

        class HeaderElements(Test):

            def get_elements(self, headername):
                if False:
                    while True:
                        i = 10
                e = cherrypy.request.headers.elements(headername)
                return '\n'.join([str(x) for x in e])

        class Method(Test):

            def index(self):
                if False:
                    return 10
                m = cherrypy.request.method
                if m in defined_http_methods or m == 'CONNECT':
                    return m
                if m == 'LINK':
                    raise cherrypy.HTTPError(405)
                else:
                    raise cherrypy.HTTPError(501)

            def parameterized(self, data):
                if False:
                    i = 10
                    return i + 15
                return data

            def request_body(self):
                if False:
                    for i in range(10):
                        print('nop')
                return cherrypy.request.body

            def reachable(self):
                if False:
                    return 10
                return 'success'

        class Divorce(Test):
            """HTTP Method handlers shouldn't collide with normal method names.
            For example, a GET-handler shouldn't collide with a method named
            'get'.

            If you build HTTP method dispatching into CherryPy, rewrite this
            class to use your new dispatch mechanism and make sure that:
                "GET /divorce HTTP/1.1" maps to divorce.index() and
                "GET /divorce/get?ID=13 HTTP/1.1" maps to divorce.get()
            """
            documents = {}

            @cherrypy.expose
            def index(self):
                if False:
                    print('Hello World!')
                yield '<h1>Choose your document</h1>\n'
                yield '<ul>\n'
                for (id, contents) in self.documents.items():
                    yield ("    <li><a href='/divorce/get?ID=%s'>%s</a>: %s</li>\n" % (id, id, contents))
                yield '</ul>'

            @cherrypy.expose
            def get(self, ID):
                if False:
                    i = 10
                    return i + 15
                return 'Divorce document %s: %s' % (ID, self.documents.get(ID, 'empty'))

        class ThreadLocal(Test):

            def index(self):
                if False:
                    print('Hello World!')
                existing = repr(getattr(cherrypy.request, 'asdf', None))
                cherrypy.request.asdf = 'rassfrassin'
                return existing
        appconf = {'/method': {'request.methods_with_bodies': ('POST', 'PUT', 'PROPFIND', 'PATCH')}}
        cherrypy.tree.mount(root, config=appconf)

    def test_scheme(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/scheme')
        self.assertBody(self.scheme)

    def test_per_request_uuid4(self):
        if False:
            while True:
                i = 10
        self.getPage('/request_uuid4')
        (first_uuid4, _, second_uuid4) = self.body.decode().partition(' ')
        assert uuid.UUID(first_uuid4, version=4) == uuid.UUID(second_uuid4, version=4)
        self.getPage('/request_uuid4')
        (third_uuid4, _, _) = self.body.decode().partition(' ')
        assert uuid.UUID(first_uuid4, version=4) != uuid.UUID(third_uuid4, version=4)

    def testRelativeURIPathInfo(self):
        if False:
            while True:
                i = 10
        self.getPage('/pathinfo/foo/bar')
        self.assertBody('/pathinfo/foo/bar')

    def testAbsoluteURIPathInfo(self):
        if False:
            i = 10
            return i + 15
        self.getPage('http://localhost/pathinfo/foo/bar')
        self.assertBody('/pathinfo/foo/bar')

    def testParams(self):
        if False:
            while True:
                i = 10
        self.getPage('/params/?thing=a')
        self.assertBody(repr(ntou('a')))
        self.getPage('/params/?thing=a&thing=b&thing=c')
        self.assertBody(repr([ntou('a'), ntou('b'), ntou('c')]))
        cherrypy.config.update({'request.show_mismatched_params': True})
        self.getPage('/params/?notathing=meeting')
        self.assertInBody('Missing parameters: thing')
        self.getPage('/params/?thing=meeting&notathing=meeting')
        self.assertInBody('Unexpected query string parameters: notathing')
        cherrypy.config.update({'request.show_mismatched_params': False})
        self.getPage('/params/?notathing=meeting')
        self.assertInBody('Not Found')
        self.getPage('/params/?thing=meeting&notathing=meeting')
        self.assertInBody('Not Found')
        self.getPage('/params/%d4%20%e3/cheese?Gruy%E8re=Bulgn%e9ville')
        self.assertBody('args: %s kwargs: %s' % (('Ô ã', 'cheese'), [('Gruyère', ntou('Bulgnéville'))]))
        self.getPage('/params/code?url=http%3A//cherrypy.dev/index%3Fa%3D1%26b%3D2')
        self.assertBody('args: %s kwargs: %s' % (('code',), [('url', ntou('http://cherrypy.dev/index?a=1&b=2'))]))
        self.getPage('/params/ismap?223,114')
        self.assertBody('Coordinates: 223, 114')
        self.getPage('/params/dictlike?a[1]=1&a[2]=2&b=foo&b[bar]=baz')
        self.assertBody('args: %s kwargs: %s' % (('dictlike',), [('a[1]', ntou('1')), ('a[2]', ntou('2')), ('b', ntou('foo')), ('b[bar]', ntou('baz'))]))

    def testParamErrors(self):
        if False:
            for i in range(10):
                print('nop')
        for uri in ('/paramerrors/one_positional?param1=foo', '/paramerrors/one_positional_args?param1=foo', '/paramerrors/one_positional_args/foo', '/paramerrors/one_positional_args/foo/bar/baz', '/paramerrors/one_positional_args_kwargs?param1=foo&param2=bar', '/paramerrors/one_positional_args_kwargs/foo?param2=bar&param3=baz', '/paramerrors/one_positional_args_kwargs/foo/bar/baz?param2=bar&param3=baz', '/paramerrors/one_positional_kwargs?param1=foo&param2=bar&param3=baz', '/paramerrors/one_positional_kwargs/foo?param4=foo&param2=bar&param3=baz', '/paramerrors/no_positional', '/paramerrors/no_positional_args/foo', '/paramerrors/no_positional_args/foo/bar/baz', '/paramerrors/no_positional_args_kwargs?param1=foo&param2=bar', '/paramerrors/no_positional_args_kwargs/foo?param2=bar', '/paramerrors/no_positional_args_kwargs/foo/bar/baz?param2=bar&param3=baz', '/paramerrors/no_positional_kwargs?param1=foo&param2=bar', '/paramerrors/callable_object'):
            self.getPage(uri)
            self.assertStatus(200)
        error_msgs = ['Missing parameters', 'Nothing matches the given URI', 'Multiple values for parameters', 'Unexpected query string parameters', 'Unexpected body parameters', 'Invalid path in Request-URI', 'Illegal #fragment in Request-URI']
        for (uri, error_idx) in (('invalid/path/without/leading/slash', 5), ('/valid/path#invalid=fragment', 6)):
            self.getPage(uri)
            self.assertStatus(400)
            self.assertInBody(error_msgs[error_idx])
        for (uri, msg) in (('/paramerrors/one_positional', error_msgs[0]), ('/paramerrors/one_positional?foo=foo', error_msgs[0]), ('/paramerrors/one_positional/foo/bar/baz', error_msgs[1]), ('/paramerrors/one_positional/foo?param1=foo', error_msgs[2]), ('/paramerrors/one_positional/foo?param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo?param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo/bar/baz?param2=foo', error_msgs[3]), ('/paramerrors/one_positional_args_kwargs/foo/bar/baz?param1=bar&param3=baz', error_msgs[2]), ('/paramerrors/one_positional_kwargs/foo?param1=foo&param2=bar&param3=baz', error_msgs[2]), ('/paramerrors/no_positional/boo', error_msgs[1]), ('/paramerrors/no_positional?param1=foo', error_msgs[3]), ('/paramerrors/no_positional_args/boo?param1=foo', error_msgs[3]), ('/paramerrors/no_positional_kwargs/boo?param1=foo', error_msgs[1]), ('/paramerrors/callable_object?param1=foo', error_msgs[3]), ('/paramerrors/callable_object/boo', error_msgs[1])):
            for show_mismatched_params in (True, False):
                cherrypy.config.update({'request.show_mismatched_params': show_mismatched_params})
                self.getPage(uri)
                self.assertStatus(404)
                if show_mismatched_params:
                    self.assertInBody(msg)
                else:
                    self.assertInBody('Not Found')
        for (uri, body, msg) in (('/paramerrors/one_positional/foo', 'param1=foo', error_msgs[2]), ('/paramerrors/one_positional/foo', 'param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo', 'param1=foo&param2=foo', error_msgs[2]), ('/paramerrors/one_positional_args/foo/bar/baz', 'param2=foo', error_msgs[4]), ('/paramerrors/one_positional_args_kwargs/foo/bar/baz', 'param1=bar&param3=baz', error_msgs[2]), ('/paramerrors/one_positional_kwargs/foo', 'param1=foo&param2=bar&param3=baz', error_msgs[2]), ('/paramerrors/no_positional', 'param1=foo', error_msgs[4]), ('/paramerrors/no_positional_args/boo', 'param1=foo', error_msgs[4]), ('/paramerrors/callable_object', 'param1=foo', error_msgs[4])):
            for show_mismatched_params in (True, False):
                cherrypy.config.update({'request.show_mismatched_params': show_mismatched_params})
                self.getPage(uri, method='POST', body=body)
                self.assertStatus(400)
                if show_mismatched_params:
                    self.assertInBody(msg)
                else:
                    self.assertInBody('400 Bad')
        for (uri, body, msg) in (('/paramerrors/one_positional?param2=foo', 'param1=foo', error_msgs[3]), ('/paramerrors/one_positional/foo/bar', 'param2=foo', error_msgs[1]), ('/paramerrors/one_positional_args/foo/bar?param2=foo', 'param3=foo', error_msgs[3]), ('/paramerrors/one_positional_kwargs/foo/bar', 'param2=bar&param3=baz', error_msgs[1]), ('/paramerrors/no_positional?param1=foo', 'param2=foo', error_msgs[3]), ('/paramerrors/no_positional_args/boo?param2=foo', 'param1=foo', error_msgs[3]), ('/paramerrors/callable_object?param2=bar', 'param1=foo', error_msgs[3])):
            for show_mismatched_params in (True, False):
                cherrypy.config.update({'request.show_mismatched_params': show_mismatched_params})
                self.getPage(uri, method='POST', body=body)
                self.assertStatus(404)
                if show_mismatched_params:
                    self.assertInBody(msg)
                else:
                    self.assertInBody('Not Found')
        for uri in ('/paramerrors/raise_type_error', '/paramerrors/raise_type_error_with_default_param?x=0', '/paramerrors/raise_type_error_with_default_param?x=0&y=0', '/paramerrors/raise_type_error_decorated'):
            self.getPage(uri, method='GET')
            self.assertStatus(500)
            self.assertTrue('Client Error', self.body)

    def testErrorHandling(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/error/missing')
        self.assertStatus(404)
        self.assertErrorPage(404, "The path '/error/missing' was not found.")
        ignore = helper.webtest.ignored_exceptions
        ignore.append(ValueError)
        try:
            valerr = '\n    raise ValueError()\nValueError'
            self.getPage('/error/page_method')
            self.assertErrorPage(500, pattern=valerr)
            self.getPage('/error/page_yield')
            self.assertErrorPage(500, pattern=valerr)
            if cherrypy.server.protocol_version == 'HTTP/1.0' or getattr(cherrypy.server, 'using_apache', False):
                self.getPage('/error/page_streamed')
                self.assertStatus(200)
                self.assertBody('word up')
            else:
                self.assertRaises((ValueError, IncompleteRead), self.getPage, '/error/page_streamed')
            self.getPage('/error/cause_err_in_finalize')
            msg = "Illegal response status from server ('ZOO' is non-numeric)."
            self.assertErrorPage(500, msg, None)
        finally:
            ignore.pop()
        self.getPage('/error/reason_phrase')
        self.assertStatus("410 Gone fishin'")
        self.getPage('/error/custom')
        self.assertStatus(404)
        self.assertBody('Hello, world\r\n' + ' ' * 499)
        self.getPage('/error/custom?err=401')
        self.assertStatus(401)
        self.assertBody("Error 401 Unauthorized - Well, I'm very sorry but you haven't paid!")
        self.getPage('/error/custom_default')
        self.assertStatus(500)
        self.assertBody("Error 500 Internal Server Error - Well, I'm very sorry but you haven't paid!".ljust(513))
        self.getPage('/error/noexist')
        self.assertStatus(404)
        if sys.version_info >= (3, 3):
            exc_name = 'FileNotFoundError'
        else:
            exc_name = 'IOError'
        msg = "No, &lt;b&gt;really&lt;/b&gt;, not found!<br />In addition, the custom error page failed:\n<br />%s: [Errno 2] No such file or directory: 'nonexistent.html'" % (exc_name,)
        self.assertInBody(msg)
        if getattr(cherrypy.server, 'using_apache', False):
            pass
        else:
            self.getPage('/error/rethrow')
            self.assertInBody('raise ValueError()')

    def testExpect(self):
        if False:
            print('Hello World!')
        e = ('Expect', '100-continue')
        self.getPage('/headerelements/get_elements?headername=Expect', [e])
        self.assertBody('100-continue')
        self.getPage('/expect/expectation_failed', [e])
        self.assertStatus(417)

    def testHeaderElements(self):
        if False:
            return 10
        h = [('Accept', 'audio/*; q=0.2, audio/basic')]
        self.getPage('/headerelements/get_elements?headername=Accept', h)
        self.assertStatus(200)
        self.assertBody('audio/basic\naudio/*;q=0.2')
        h = [('Accept', 'text/plain; q=0.5, text/html, text/x-dvi; q=0.8, text/x-c')]
        self.getPage('/headerelements/get_elements?headername=Accept', h)
        self.assertStatus(200)
        self.assertBody('text/x-c\ntext/html\ntext/x-dvi;q=0.8\ntext/plain;q=0.5')
        h = [('Accept', 'text/*, text/html, text/html;level=1, */*')]
        self.getPage('/headerelements/get_elements?headername=Accept', h)
        self.assertStatus(200)
        self.assertBody('text/html;level=1\ntext/html\ntext/*\n*/*')
        h = [('Accept-Charset', 'iso-8859-5, unicode-1-1;q=0.8')]
        self.getPage('/headerelements/get_elements?headername=Accept-Charset', h)
        self.assertStatus('200 OK')
        self.assertBody('iso-8859-5\nunicode-1-1;q=0.8')
        h = [('Accept-Encoding', 'gzip;q=1.0, identity; q=0.5, *;q=0')]
        self.getPage('/headerelements/get_elements?headername=Accept-Encoding', h)
        self.assertStatus('200 OK')
        self.assertBody('gzip;q=1.0\nidentity;q=0.5\n*;q=0')
        h = [('Accept-Language', 'da, en-gb;q=0.8, en;q=0.7')]
        self.getPage('/headerelements/get_elements?headername=Accept-Language', h)
        self.assertStatus('200 OK')
        self.assertBody('da\nen-gb;q=0.8\nen;q=0.7')
        self.getPage('/headerelements/get_elements?headername=Content-Type', headers=[('Content-Type', 'text/html; charset=utf-8;')])
        self.assertStatus(200)
        self.assertBody('text/html;charset=utf-8')

    def test_repeated_headers(self):
        if False:
            i = 10
            return i + 15
        self.getPage('/headers/Accept-Charset', headers=[('Accept-Charset', 'iso-8859-5'), ('Accept-Charset', 'unicode-1-1;q=0.8')])
        self.assertBody('iso-8859-5, unicode-1-1;q=0.8')
        self.getPage('/headers/doubledheaders')
        self.assertBody('double header test')
        hnames = [name.title() for (name, val) in self.headers]
        for key in ['Content-Length', 'Content-Type', 'Date', 'Expires', 'Location', 'Server']:
            self.assertEqual(hnames.count(key), 1, self.headers)

    def test_encoded_headers(self):
        if False:
            while True:
                i = 10
        self.assertEqual(httputil.decode_TEXT(ntou('=?utf-8?q?f=C3=BCr?=')), ntou('für'))
        if cherrypy.server.protocol_version == 'HTTP/1.1':
            u = ntou('Ångström', 'escape')
            c = ntou('=E2=84=ABngstr=C3=B6m')
            self.getPage('/headers/ifmatch', [('If-Match', ntou('=?utf-8?q?%s?=') % c)])
            self.assertBody(b'\xe2\x84\xabngstr\xc3\xb6m')
            self.assertHeader('ETag', ntou('=?utf-8?b?4oSrbmdzdHLDtm0=?='))
            self.getPage('/headers/ifmatch', [('If-Match', ntou('=?utf-8?q?%s?=') % (c * 10))])
            self.assertBody(b'\xe2\x84\xabngstr\xc3\xb6m' * 10)
            etag = self.assertHeader('ETag', '=?utf-8?b?4oSrbmdzdHLDtm3ihKtuZ3N0csO2beKEq25nc3Ryw7Zt4oSrbmdzdHLDtm3ihKtuZ3N0csO2beKEq25nc3Ryw7Zt4oSrbmdzdHLDtm3ihKtuZ3N0csO2beKEq25nc3Ryw7Zt4oSrbmdzdHLDtm0=?=')
            self.assertEqual(httputil.decode_TEXT(etag), u * 10)

    def test_header_presence(self):
        if False:
            while True:
                i = 10
        self.getPage('/headers/Content-Type', headers=[])
        self.assertStatus(500)
        self.getPage('/headers/Content-Type', headers=[('Content-type', 'application/json')])
        self.assertBody('application/json')

    def test_dangerous_host(self):
        if False:
            i = 10
            return i + 15
        '\n        Dangerous characters like newlines should be elided.\n        Ref #1974.\n        '
        encoded = '=?iso-8859-1?q?foo=0Abar?='
        self.getPage('/headers/Host', headers=[('Host', encoded)])
        self.assertBody('foobar')

    def test_basic_HTTPMethods(self):
        if False:
            while True:
                i = 10
        helper.webtest.methods_with_bodies = ('POST', 'PUT', 'PROPFIND', 'PATCH')
        for m in defined_http_methods:
            self.getPage('/method/', method=m)
            if m == 'HEAD':
                self.assertBody('')
            elif m == 'TRACE':
                self.assertEqual(self.body[:5], b'TRACE')
            else:
                self.assertBody(m)
        self.getPage('/method/parameterized', method='PATCH', body='data=on+top+of+other+things')
        self.assertBody('on top of other things')
        b = 'one thing on top of another'
        h = [('Content-Type', 'text/plain'), ('Content-Length', str(len(b)))]
        self.getPage('/method/request_body', headers=h, method='PATCH', body=b)
        self.assertStatus(200)
        self.assertBody(b)
        b = b'one thing on top of another'
        self.persistent = True
        try:
            conn = self.HTTP_CONN
            conn.putrequest('PATCH', '/method/request_body', skip_host=True)
            conn.putheader('Host', self.HOST)
            conn.putheader('Content-Length', str(len(b)))
            conn.endheaders()
            conn.send(b)
            response = conn.response_class(conn.sock, method='PATCH')
            response.begin()
            self.assertEqual(response.status, 200)
            self.body = response.read()
            self.assertBody(b)
        finally:
            self.persistent = False
        h = [('Content-Type', 'text/plain')]
        self.getPage('/method/reachable', headers=h, method='PATCH')
        self.assertStatus(411)
        self.getPage('/method/parameterized', method='PUT', body='data=on+top+of+other+things')
        self.assertBody('on top of other things')
        b = 'one thing on top of another'
        h = [('Content-Type', 'text/plain'), ('Content-Length', str(len(b)))]
        self.getPage('/method/request_body', headers=h, method='PUT', body=b)
        self.assertStatus(200)
        self.assertBody(b)
        b = b'one thing on top of another'
        self.persistent = True
        try:
            conn = self.HTTP_CONN
            conn.putrequest('PUT', '/method/request_body', skip_host=True)
            conn.putheader('Host', self.HOST)
            conn.putheader('Content-Length', str(len(b)))
            conn.endheaders()
            conn.send(b)
            response = conn.response_class(conn.sock, method='PUT')
            response.begin()
            self.assertEqual(response.status, 200)
            self.body = response.read()
            self.assertBody(b)
        finally:
            self.persistent = False
        h = [('Content-Type', 'text/plain')]
        self.getPage('/method/reachable', headers=h, method='PUT')
        self.assertStatus(411)
        b = '<?xml version="1.0" encoding="utf-8" ?>\n\n<propfind xmlns="DAV:"><prop><getlastmodified/></prop></propfind>'
        h = [('Content-Type', 'text/xml'), ('Content-Length', str(len(b)))]
        self.getPage('/method/request_body', headers=h, method='PROPFIND', body=b)
        self.assertStatus(200)
        self.assertBody(b)
        self.getPage('/method/', method='LINK')
        self.assertStatus(405)
        self.getPage('/method/', method='SEARCH')
        self.assertStatus(501)
        self.getPage('/divorce/get?ID=13')
        self.assertBody('Divorce document 13: empty')
        self.assertStatus(200)
        self.getPage('/divorce/', method='GET')
        self.assertBody('<h1>Choose your document</h1>\n<ul>\n</ul>')
        self.assertStatus(200)

    def test_CONNECT_method(self):
        if False:
            i = 10
            return i + 15
        self.persistent = True
        try:
            conn = self.HTTP_CONN
            conn.request('CONNECT', 'created.example.com:3128')
            response = conn.response_class(conn.sock, method='CONNECT')
            response.begin()
            self.assertEqual(response.status, 204)
        finally:
            self.persistent = False
        self.persistent = True
        try:
            conn = self.HTTP_CONN
            conn.request('CONNECT', 'body.example.com:3128')
            response = conn.response_class(conn.sock, method='CONNECT')
            response.begin()
            self.assertEqual(response.status, 200)
            self.body = response.read()
            self.assertBody(b'CONNECTed to /body.example.com:3128')
        finally:
            self.persistent = False

    def test_CONNECT_method_invalid_authority(self):
        if False:
            for i in range(10):
                print('nop')
        for request_target in ['example.com', 'http://example.com:33', '/path/', 'path/', '/?q=f', '#f']:
            self.persistent = True
            try:
                conn = self.HTTP_CONN
                conn.request('CONNECT', request_target)
                response = conn.response_class(conn.sock, method='CONNECT')
                response.begin()
                self.assertEqual(response.status, 400)
                self.body = response.read()
                self.assertBody(b'Invalid path in Request-URI: request-target must match authority-form.')
            finally:
                self.persistent = False

    def testEmptyThreadlocals(self):
        if False:
            for i in range(10):
                print('nop')
        results = []
        for x in range(20):
            self.getPage('/threadlocal/')
            results.append(self.body)
        self.assertEqual(results, [b'None'] * 20)