import base64
import os
import email
import urllib.parse
import urllib.request
import http.server
import threading
import unittest
import hashlib
from test.support import hashlib_helper
from test.support import threading_helper
from test.support import warnings_helper
try:
    import ssl
except ImportError:
    ssl = None
here = os.path.dirname(__file__)
CERT_localhost = os.path.join(here, 'keycert.pem')
CERT_fakehostname = os.path.join(here, 'keycert2.pem')

class LoopbackHttpServer(http.server.HTTPServer):
    """HTTP server w/ a few modifications that make it useful for
    loopback testing purposes.
    """

    def __init__(self, server_address, RequestHandlerClass):
        if False:
            i = 10
            return i + 15
        http.server.HTTPServer.__init__(self, server_address, RequestHandlerClass)
        self.socket.settimeout(0.1)

    def get_request(self):
        if False:
            print('Hello World!')
        'HTTPServer method, overridden.'
        (request, client_address) = self.socket.accept()
        request.settimeout(10.0)
        return (request, client_address)

class LoopbackHttpServerThread(threading.Thread):
    """Stoppable thread that runs a loopback http server."""

    def __init__(self, request_handler):
        if False:
            i = 10
            return i + 15
        threading.Thread.__init__(self)
        self._stop_server = False
        self.ready = threading.Event()
        request_handler.protocol_version = 'HTTP/1.0'
        self.httpd = LoopbackHttpServer(('127.0.0.1', 0), request_handler)
        self.port = self.httpd.server_port

    def stop(self):
        if False:
            print('Hello World!')
        "Stops the webserver if it's currently running."
        self._stop_server = True
        self.join()
        self.httpd.server_close()

    def run(self):
        if False:
            i = 10
            return i + 15
        self.ready.set()
        while not self._stop_server:
            self.httpd.handle_request()

class DigestAuthHandler:
    """Handler for performing digest authentication."""

    def __init__(self):
        if False:
            while True:
                i = 10
        self._request_num = 0
        self._nonces = []
        self._users = {}
        self._realm_name = 'Test Realm'
        self._qop = 'auth'

    def set_qop(self, qop):
        if False:
            for i in range(10):
                print('nop')
        self._qop = qop

    def set_users(self, users):
        if False:
            return 10
        assert isinstance(users, dict)
        self._users = users

    def set_realm(self, realm):
        if False:
            while True:
                i = 10
        self._realm_name = realm

    def _generate_nonce(self):
        if False:
            print('Hello World!')
        self._request_num += 1
        nonce = hashlib.md5(str(self._request_num).encode('ascii')).hexdigest()
        self._nonces.append(nonce)
        return nonce

    def _create_auth_dict(self, auth_str):
        if False:
            print('Hello World!')
        first_space_index = auth_str.find(' ')
        auth_str = auth_str[first_space_index + 1:]
        parts = auth_str.split(',')
        auth_dict = {}
        for part in parts:
            (name, value) = part.split('=')
            name = name.strip()
            if value[0] == '"' and value[-1] == '"':
                value = value[1:-1]
            else:
                value = value.strip()
            auth_dict[name] = value
        return auth_dict

    def _validate_auth(self, auth_dict, password, method, uri):
        if False:
            for i in range(10):
                print('nop')
        final_dict = {}
        final_dict.update(auth_dict)
        final_dict['password'] = password
        final_dict['method'] = method
        final_dict['uri'] = uri
        HA1_str = '%(username)s:%(realm)s:%(password)s' % final_dict
        HA1 = hashlib.md5(HA1_str.encode('ascii')).hexdigest()
        HA2_str = '%(method)s:%(uri)s' % final_dict
        HA2 = hashlib.md5(HA2_str.encode('ascii')).hexdigest()
        final_dict['HA1'] = HA1
        final_dict['HA2'] = HA2
        response_str = '%(HA1)s:%(nonce)s:%(nc)s:%(cnonce)s:%(qop)s:%(HA2)s' % final_dict
        response = hashlib.md5(response_str.encode('ascii')).hexdigest()
        return response == auth_dict['response']

    def _return_auth_challenge(self, request_handler):
        if False:
            return 10
        request_handler.send_response(407, 'Proxy Authentication Required')
        request_handler.send_header('Content-Type', 'text/html')
        request_handler.send_header('Proxy-Authenticate', 'Digest realm="%s", qop="%s",nonce="%s", ' % (self._realm_name, self._qop, self._generate_nonce()))
        request_handler.end_headers()
        request_handler.wfile.write(b'Proxy Authentication Required.')
        return False

    def handle_request(self, request_handler):
        if False:
            i = 10
            return i + 15
        'Performs digest authentication on the given HTTP request\n        handler.  Returns True if authentication was successful, False\n        otherwise.\n\n        If no users have been set, then digest auth is effectively\n        disabled and this method will always return True.\n        '
        if len(self._users) == 0:
            return True
        if 'Proxy-Authorization' not in request_handler.headers:
            return self._return_auth_challenge(request_handler)
        else:
            auth_dict = self._create_auth_dict(request_handler.headers['Proxy-Authorization'])
            if auth_dict['username'] in self._users:
                password = self._users[auth_dict['username']]
            else:
                return self._return_auth_challenge(request_handler)
            if not auth_dict.get('nonce') in self._nonces:
                return self._return_auth_challenge(request_handler)
            else:
                self._nonces.remove(auth_dict['nonce'])
            auth_validated = False
            for path in [request_handler.path, request_handler.short_path]:
                if self._validate_auth(auth_dict, password, request_handler.command, path):
                    auth_validated = True
            if not auth_validated:
                return self._return_auth_challenge(request_handler)
            return True

class BasicAuthHandler(http.server.BaseHTTPRequestHandler):
    """Handler for performing basic authentication."""
    USER = 'testUser'
    PASSWD = 'testPass'
    REALM = 'Test'
    USER_PASSWD = '%s:%s' % (USER, PASSWD)
    ENCODED_AUTH = base64.b64encode(USER_PASSWD.encode('ascii')).decode('ascii')

    def __init__(self, *args, **kwargs):
        if False:
            return 10
        http.server.BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def log_message(self, format, *args):
        if False:
            while True:
                i = 10
        pass

    def do_HEAD(self):
        if False:
            return 10
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_AUTHHEAD(self):
        if False:
            return 10
        self.send_response(401)
        self.send_header('WWW-Authenticate', 'Basic realm="%s"' % self.REALM)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

    def do_GET(self):
        if False:
            return 10
        if not self.headers.get('Authorization', ''):
            self.do_AUTHHEAD()
            self.wfile.write(b'No Auth header received')
        elif self.headers.get('Authorization', '') == 'Basic ' + self.ENCODED_AUTH:
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'It works')
        else:
            self.do_AUTHHEAD()

class FakeProxyHandler(http.server.BaseHTTPRequestHandler):
    """This is a 'fake proxy' that makes it look like the entire
    internet has gone down due to a sudden zombie invasion.  It main
    utility is in providing us with authentication support for
    testing.
    """

    def __init__(self, digest_auth_handler, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.digest_auth_handler = digest_auth_handler
        http.server.BaseHTTPRequestHandler.__init__(self, *args, **kwargs)

    def log_message(self, format, *args):
        if False:
            return 10
        pass

    def do_GET(self):
        if False:
            for i in range(10):
                print('nop')
        (scm, netloc, path, params, query, fragment) = urllib.parse.urlparse(self.path, 'http')
        self.short_path = path
        if self.digest_auth_handler.handle_request(self):
            self.send_response(200, 'OK')
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(bytes("You've reached %s!<BR>" % self.path, 'ascii'))
            self.wfile.write(b'Our apologies, but our server is down due to a sudden zombie invasion.')

class BasicAuthTests(unittest.TestCase):
    USER = 'testUser'
    PASSWD = 'testPass'
    INCORRECT_PASSWD = 'Incorrect'
    REALM = 'Test'

    def setUp(self):
        if False:
            while True:
                i = 10
        super(BasicAuthTests, self).setUp()

        def http_server_with_basic_auth_handler(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return BasicAuthHandler(*args, **kwargs)
        self.server = LoopbackHttpServerThread(http_server_with_basic_auth_handler)
        self.addCleanup(self.stop_server)
        self.server_url = 'http://127.0.0.1:%s' % self.server.port
        self.server.start()
        self.server.ready.wait()

    def stop_server(self):
        if False:
            print('Hello World!')
        self.server.stop()
        self.server = None

    def tearDown(self):
        if False:
            while True:
                i = 10
        super(BasicAuthTests, self).tearDown()

    def test_basic_auth_success(self):
        if False:
            i = 10
            return i + 15
        ah = urllib.request.HTTPBasicAuthHandler()
        ah.add_password(self.REALM, self.server_url, self.USER, self.PASSWD)
        urllib.request.install_opener(urllib.request.build_opener(ah))
        try:
            self.assertTrue(urllib.request.urlopen(self.server_url))
        except urllib.error.HTTPError:
            self.fail('Basic auth failed for the url: %s' % self.server_url)

    def test_basic_auth_httperror(self):
        if False:
            print('Hello World!')
        ah = urllib.request.HTTPBasicAuthHandler()
        ah.add_password(self.REALM, self.server_url, self.USER, self.INCORRECT_PASSWD)
        urllib.request.install_opener(urllib.request.build_opener(ah))
        self.assertRaises(urllib.error.HTTPError, urllib.request.urlopen, self.server_url)

@hashlib_helper.requires_hashdigest('md5', openssl=True)
class ProxyAuthTests(unittest.TestCase):
    URL = 'http://localhost'
    USER = 'tester'
    PASSWD = 'test123'
    REALM = 'TestRealm'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(ProxyAuthTests, self).setUp()

        def restore_environ(old_environ):
            if False:
                print('Hello World!')
            os.environ.clear()
            os.environ.update(old_environ)
        self.addCleanup(restore_environ, os.environ.copy())
        os.environ['NO_PROXY'] = ''
        os.environ['no_proxy'] = ''
        self.digest_auth_handler = DigestAuthHandler()
        self.digest_auth_handler.set_users({self.USER: self.PASSWD})
        self.digest_auth_handler.set_realm(self.REALM)

        def create_fake_proxy_handler(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            return FakeProxyHandler(self.digest_auth_handler, *args, **kwargs)
        self.server = LoopbackHttpServerThread(create_fake_proxy_handler)
        self.addCleanup(self.stop_server)
        self.server.start()
        self.server.ready.wait()
        proxy_url = 'http://127.0.0.1:%d' % self.server.port
        handler = urllib.request.ProxyHandler({'http': proxy_url})
        self.proxy_digest_handler = urllib.request.ProxyDigestAuthHandler()
        self.opener = urllib.request.build_opener(handler, self.proxy_digest_handler)

    def stop_server(self):
        if False:
            return 10
        self.server.stop()
        self.server = None

    def test_proxy_with_bad_password_raises_httperror(self):
        if False:
            return 10
        self.proxy_digest_handler.add_password(self.REALM, self.URL, self.USER, self.PASSWD + 'bad')
        self.digest_auth_handler.set_qop('auth')
        self.assertRaises(urllib.error.HTTPError, self.opener.open, self.URL)

    def test_proxy_with_no_password_raises_httperror(self):
        if False:
            while True:
                i = 10
        self.digest_auth_handler.set_qop('auth')
        self.assertRaises(urllib.error.HTTPError, self.opener.open, self.URL)

    def test_proxy_qop_auth_works(self):
        if False:
            return 10
        self.proxy_digest_handler.add_password(self.REALM, self.URL, self.USER, self.PASSWD)
        self.digest_auth_handler.set_qop('auth')
        with self.opener.open(self.URL) as result:
            while result.read():
                pass

    def test_proxy_qop_auth_int_works_or_throws_urlerror(self):
        if False:
            return 10
        self.proxy_digest_handler.add_password(self.REALM, self.URL, self.USER, self.PASSWD)
        self.digest_auth_handler.set_qop('auth-int')
        try:
            result = self.opener.open(self.URL)
        except urllib.error.URLError:
            pass
        else:
            with result:
                while result.read():
                    pass

def GetRequestHandler(responses):
    if False:
        for i in range(10):
            print('nop')

    class FakeHTTPRequestHandler(http.server.BaseHTTPRequestHandler):
        server_version = 'TestHTTP/'
        requests = []
        headers_received = []
        port = 80

        def do_GET(self):
            if False:
                while True:
                    i = 10
            body = self.send_head()
            while body:
                done = self.wfile.write(body)
                body = body[done:]

        def do_POST(self):
            if False:
                return 10
            content_length = self.headers['Content-Length']
            post_data = self.rfile.read(int(content_length))
            self.do_GET()
            self.requests.append(post_data)

        def send_head(self):
            if False:
                print('Hello World!')
            FakeHTTPRequestHandler.headers_received = self.headers
            self.requests.append(self.path)
            (response_code, headers, body) = responses.pop(0)
            self.send_response(response_code)
            for (header, value) in headers:
                self.send_header(header, value % {'port': self.port})
            if body:
                self.send_header('Content-type', 'text/plain')
                self.end_headers()
                return body
            self.end_headers()

        def log_message(self, *args):
            if False:
                i = 10
                return i + 15
            pass
    return FakeHTTPRequestHandler

class TestUrlopen(unittest.TestCase):
    """Tests urllib.request.urlopen using the network.

    These tests are not exhaustive.  Assuming that testing using files does a
    good job overall of some of the basic interface features.  There are no
    tests exercising the optional 'data' and 'proxies' arguments.  No tests
    for transparent redirection have been written.
    """

    def setUp(self):
        if False:
            return 10
        super(TestUrlopen, self).setUp()
        self.addCleanup(urllib.request.urlcleanup)

        def restore_environ(old_environ):
            if False:
                return 10
            os.environ.clear()
            os.environ.update(old_environ)
        self.addCleanup(restore_environ, os.environ.copy())
        os.environ['NO_PROXY'] = '*'
        os.environ['no_proxy'] = '*'

    def urlopen(self, url, data=None, **kwargs):
        if False:
            return 10
        l = []
        f = urllib.request.urlopen(url, data, **kwargs)
        try:
            l.extend(f.readlines(200))
            l.append(f.readline())
            l.append(f.read(1024))
            l.append(f.read())
        finally:
            f.close()
        return b''.join(l)

    def stop_server(self):
        if False:
            return 10
        self.server.stop()
        self.server = None

    def start_server(self, responses=None):
        if False:
            for i in range(10):
                print('nop')
        if responses is None:
            responses = [(200, [], b"we don't care")]
        handler = GetRequestHandler(responses)
        self.server = LoopbackHttpServerThread(handler)
        self.addCleanup(self.stop_server)
        self.server.start()
        self.server.ready.wait()
        port = self.server.port
        handler.port = port
        return handler

    def start_https_server(self, responses=None, **kwargs):
        if False:
            while True:
                i = 10
        if not hasattr(urllib.request, 'HTTPSHandler'):
            self.skipTest('ssl support required')
        from test.ssl_servers import make_https_server
        if responses is None:
            responses = [(200, [], b'we care a bit')]
        handler = GetRequestHandler(responses)
        server = make_https_server(self, handler_class=handler, **kwargs)
        handler.port = server.port
        return handler

    def test_redirection(self):
        if False:
            return 10
        expected_response = b'We got here...'
        responses = [(302, [('Location', 'http://localhost:%(port)s/somewhere_else')], ''), (200, [], expected_response)]
        handler = self.start_server(responses)
        data = self.urlopen('http://localhost:%s/' % handler.port)
        self.assertEqual(data, expected_response)
        self.assertEqual(handler.requests, ['/', '/somewhere_else'])

    def test_chunked(self):
        if False:
            return 10
        expected_response = b'hello world'
        chunked_start = b'a\r\nhello worl\r\n1\r\nd\r\n0\r\n'
        response = [(200, [('Transfer-Encoding', 'chunked')], chunked_start)]
        handler = self.start_server(response)
        data = self.urlopen('http://localhost:%s/' % handler.port)
        self.assertEqual(data, expected_response)

    def test_404(self):
        if False:
            return 10
        expected_response = b'Bad bad bad...'
        handler = self.start_server([(404, [], expected_response)])
        try:
            self.urlopen('http://localhost:%s/weeble' % handler.port)
        except urllib.error.URLError as f:
            data = f.read()
            f.close()
        else:
            self.fail('404 should raise URLError')
        self.assertEqual(data, expected_response)
        self.assertEqual(handler.requests, ['/weeble'])

    def test_200(self):
        if False:
            i = 10
            return i + 15
        expected_response = b'pycon 2008...'
        handler = self.start_server([(200, [], expected_response)])
        data = self.urlopen('http://localhost:%s/bizarre' % handler.port)
        self.assertEqual(data, expected_response)
        self.assertEqual(handler.requests, ['/bizarre'])

    def test_200_with_parameters(self):
        if False:
            while True:
                i = 10
        expected_response = b'pycon 2008...'
        handler = self.start_server([(200, [], expected_response)])
        data = self.urlopen('http://localhost:%s/bizarre' % handler.port, b'get=with_feeling')
        self.assertEqual(data, expected_response)
        self.assertEqual(handler.requests, ['/bizarre', b'get=with_feeling'])

    def test_https(self):
        if False:
            print('Hello World!')
        handler = self.start_https_server()
        context = ssl.create_default_context(cafile=CERT_localhost)
        data = self.urlopen('https://localhost:%s/bizarre' % handler.port, context=context)
        self.assertEqual(data, b'we care a bit')

    def test_https_with_cafile(self):
        if False:
            return 10
        handler = self.start_https_server(certfile=CERT_localhost)
        with warnings_helper.check_warnings(('', DeprecationWarning)):
            data = self.urlopen('https://localhost:%s/bizarre' % handler.port, cafile=CERT_localhost)
            self.assertEqual(data, b'we care a bit')
            with self.assertRaises(urllib.error.URLError) as cm:
                self.urlopen('https://localhost:%s/bizarre' % handler.port, cafile=CERT_fakehostname)
            handler = self.start_https_server(certfile=CERT_fakehostname)
            with self.assertRaises(urllib.error.URLError) as cm:
                self.urlopen('https://localhost:%s/bizarre' % handler.port, cafile=CERT_fakehostname)

    def test_https_with_cadefault(self):
        if False:
            i = 10
            return i + 15
        handler = self.start_https_server(certfile=CERT_localhost)
        with warnings_helper.check_warnings(('', DeprecationWarning)):
            with self.assertRaises(urllib.error.URLError) as cm:
                self.urlopen('https://localhost:%s/bizarre' % handler.port, cadefault=True)

    def test_https_sni(self):
        if False:
            i = 10
            return i + 15
        if ssl is None:
            self.skipTest('ssl module required')
        if not ssl.HAS_SNI:
            self.skipTest('SNI support required in OpenSSL')
        sni_name = None

        def cb_sni(ssl_sock, server_name, initial_context):
            if False:
                return 10
            nonlocal sni_name
            sni_name = server_name
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.set_servername_callback(cb_sni)
        handler = self.start_https_server(context=context, certfile=CERT_localhost)
        context = ssl.create_default_context(cafile=CERT_localhost)
        self.urlopen('https://localhost:%s' % handler.port, context=context)
        self.assertEqual(sni_name, 'localhost')

    def test_sending_headers(self):
        if False:
            return 10
        handler = self.start_server()
        req = urllib.request.Request('http://localhost:%s/' % handler.port, headers={'Range': 'bytes=20-39'})
        with urllib.request.urlopen(req):
            pass
        self.assertEqual(handler.headers_received['Range'], 'bytes=20-39')

    def test_sending_headers_camel(self):
        if False:
            while True:
                i = 10
        handler = self.start_server()
        req = urllib.request.Request('http://localhost:%s/' % handler.port, headers={'X-SoMe-hEader': 'foobar'})
        with urllib.request.urlopen(req):
            pass
        self.assertIn('X-Some-Header', handler.headers_received.keys())
        self.assertNotIn('X-SoMe-hEader', handler.headers_received.keys())

    def test_basic(self):
        if False:
            i = 10
            return i + 15
        handler = self.start_server()
        with urllib.request.urlopen('http://localhost:%s' % handler.port) as open_url:
            for attr in ('read', 'close', 'info', 'geturl'):
                self.assertTrue(hasattr(open_url, attr), 'object returned from urlopen lacks the %s attribute' % attr)
            self.assertTrue(open_url.read(), "calling 'read' failed")

    def test_info(self):
        if False:
            i = 10
            return i + 15
        handler = self.start_server()
        open_url = urllib.request.urlopen('http://localhost:%s' % handler.port)
        with open_url:
            info_obj = open_url.info()
        self.assertIsInstance(info_obj, email.message.Message, "object returned by 'info' is not an instance of email.message.Message")
        self.assertEqual(info_obj.get_content_subtype(), 'plain')

    def test_geturl(self):
        if False:
            while True:
                i = 10
        handler = self.start_server()
        open_url = urllib.request.urlopen('http://localhost:%s' % handler.port)
        with open_url:
            url = open_url.geturl()
        self.assertEqual(url, 'http://localhost:%s' % handler.port)

    def test_iteration(self):
        if False:
            print('Hello World!')
        expected_response = b'pycon 2008...'
        handler = self.start_server([(200, [], expected_response)])
        data = urllib.request.urlopen('http://localhost:%s' % handler.port)
        for line in data:
            self.assertEqual(line, expected_response)

    def test_line_iteration(self):
        if False:
            while True:
                i = 10
        lines = [b'We\n', b'got\n', b'here\n', b'verylong ' * 8192 + b'\n']
        expected_response = b''.join(lines)
        handler = self.start_server([(200, [], expected_response)])
        data = urllib.request.urlopen('http://localhost:%s' % handler.port)
        for (index, line) in enumerate(data):
            self.assertEqual(line, lines[index], "Fetched line number %s doesn't match expected:\n    Expected length was %s, got %s" % (index, len(lines[index]), len(line)))
        self.assertEqual(index + 1, len(lines))

    def test_issue16464(self):
        if False:
            return 10
        handler = self.start_server([(200, [], b'any'), (200, [], b'any')])
        opener = urllib.request.build_opener()
        request = urllib.request.Request('http://localhost:%s' % handler.port)
        self.assertEqual(None, request.data)
        opener.open(request, '1'.encode('us-ascii'))
        self.assertEqual(b'1', request.data)
        self.assertEqual('1', request.get_header('Content-length'))
        opener.open(request, '1234567890'.encode('us-ascii'))
        self.assertEqual(b'1234567890', request.data)
        self.assertEqual('10', request.get_header('Content-length'))

def setUpModule():
    if False:
        i = 10
        return i + 15
    thread_info = threading_helper.threading_setup()
    unittest.addModuleCleanup(threading_helper.threading_cleanup, *thread_info)
if __name__ == '__main__':
    unittest.main()