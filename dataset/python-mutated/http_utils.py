from cStringIO import StringIO
import re
import urllib2
from bzrlib import errors, osutils, tests, transport
from bzrlib.smart import medium
from bzrlib.tests import http_server
from bzrlib.transport import chroot

class HTTPServerWithSmarts(http_server.HttpServer):
    """HTTPServerWithSmarts extends the HttpServer with POST methods that will
    trigger a smart server to execute with a transport rooted at the rootdir of
    the HTTP server.
    """

    def __init__(self, protocol_version=None):
        if False:
            i = 10
            return i + 15
        http_server.HttpServer.__init__(self, SmartRequestHandler, protocol_version=protocol_version)

class SmartRequestHandler(http_server.TestingHTTPRequestHandler):
    """Extend TestingHTTPRequestHandler to support smart client POSTs.

    XXX: This duplicates a fair bit of the logic in bzrlib.transport.http.wsgi.
    """

    def do_POST(self):
        if False:
            while True:
                i = 10
        'Hand the request off to a smart server instance.'
        backing = transport.get_transport_from_path(self.server.test_case_server._home_dir)
        chroot_server = chroot.ChrootServer(backing)
        chroot_server.start_server()
        try:
            t = transport.get_transport_from_url(chroot_server.get_url())
            self.do_POST_inner(t)
        finally:
            chroot_server.stop_server()

    def do_POST_inner(self, chrooted_transport):
        if False:
            for i in range(10):
                print('nop')
        self.send_response(200)
        self.send_header('Content-type', 'application/octet-stream')
        if not self.path.endswith('.bzr/smart'):
            raise AssertionError('POST to path not ending in .bzr/smart: %r' % (self.path,))
        t = chrooted_transport.clone(self.path[:-len('.bzr/smart')])
        data_length = int(self.headers['Content-Length'])
        request_bytes = self.rfile.read(data_length)
        (protocol_factory, unused_bytes) = medium._get_protocol_factory_for_bytes(request_bytes)
        out_buffer = StringIO()
        smart_protocol_request = protocol_factory(t, out_buffer.write, '/')
        smart_protocol_request.accept_bytes(unused_bytes)
        if not smart_protocol_request.next_read_size() == 0:
            raise errors.SmartProtocolError('not finished reading, but all data sent to protocol.')
        self.send_header('Content-Length', str(len(out_buffer.getvalue())))
        self.end_headers()
        self.wfile.write(out_buffer.getvalue())

class TestCaseWithWebserver(tests.TestCaseWithTransport):
    """A support class that provides readonly urls that are http://.

    This is done by forcing the readonly server to be an http
    one. This will currently fail if the primary transport is not
    backed by regular disk files.
    """
    _protocol_version = None
    _url_protocol = 'http'

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestCaseWithWebserver, self).setUp()
        self.transport_readonly_server = http_server.HttpServer

    def create_transport_readonly_server(self):
        if False:
            while True:
                i = 10
        server = self.transport_readonly_server(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server

class TestCaseWithTwoWebservers(TestCaseWithWebserver):
    """A support class providing readonly urls on two servers that are http://.

    We set up two webservers to allows various tests involving
    proxies or redirections from one server to the other.
    """

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TestCaseWithTwoWebservers, self).setUp()
        self.transport_secondary_server = http_server.HttpServer
        self.__secondary_server = None

    def create_transport_secondary_server(self):
        if False:
            return 10
        'Create a transport server from class defined at init.\n\n        This is mostly a hook for daughter classes.\n        '
        server = self.transport_secondary_server(protocol_version=self._protocol_version)
        server._url_protocol = self._url_protocol
        return server

    def get_secondary_server(self):
        if False:
            i = 10
            return i + 15
        'Get the server instance for the secondary transport.'
        if self.__secondary_server is None:
            self.__secondary_server = self.create_transport_secondary_server()
            self.start_server(self.__secondary_server)
        return self.__secondary_server

    def get_secondary_url(self, relpath=None):
        if False:
            print('Hello World!')
        base = self.get_secondary_server().get_url()
        return self._adjust_url(base, relpath)

    def get_secondary_transport(self, relpath=None):
        if False:
            while True:
                i = 10
        t = transport.get_transport_from_url(self.get_secondary_url(relpath))
        self.assertTrue(t.is_readonly())
        return t

class ProxyServer(http_server.HttpServer):
    """A proxy test server for http transports."""
    proxy_requests = True

class RedirectRequestHandler(http_server.TestingHTTPRequestHandler):
    """Redirect all request to the specified server"""

    def parse_request(self):
        if False:
            while True:
                i = 10
        'Redirect a single HTTP request to another host'
        valid = http_server.TestingHTTPRequestHandler.parse_request(self)
        if valid:
            tcs = self.server.test_case_server
            (code, target) = tcs.is_redirected(self.path)
            if code is not None and target is not None:
                self.send_response(code)
                self.send_header('Location', target)
                self.send_header('Content-Length', '0')
                self.end_headers()
                return False
            else:
                pass
        return valid

class HTTPServerRedirecting(http_server.HttpServer):
    """An HttpServer redirecting to another server """

    def __init__(self, request_handler=RedirectRequestHandler, protocol_version=None):
        if False:
            print('Hello World!')
        http_server.HttpServer.__init__(self, request_handler, protocol_version=protocol_version)
        self.redirections = []

    def redirect_to(self, host, port):
        if False:
            for i in range(10):
                print('nop')
        'Redirect all requests to a specific host:port'
        self.redirections = [('(.*)', 'http://%s:%s\\1' % (host, port), 301)]

    def is_redirected(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Is the path redirected by this server.\n\n        :param path: the requested relative path\n\n        :returns: a tuple (code, target) if a matching\n             redirection is found, (None, None) otherwise.\n        '
        code = None
        target = None
        for (rsource, rtarget, rcode) in self.redirections:
            (target, match) = re.subn(rsource, rtarget, path)
            if match:
                code = rcode
                break
            else:
                target = None
        return (code, target)

class TestCaseWithRedirectedWebserver(TestCaseWithTwoWebservers):
    """A support class providing redirections from one server to another.

   We set up two webservers to allows various tests involving
   redirections.
   The 'old' server is redirected to the 'new' server.
   """

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super(TestCaseWithRedirectedWebserver, self).setUp()
        self.new_server = self.get_readonly_server()
        self.old_server = self.get_secondary_server()

    def create_transport_secondary_server(self):
        if False:
            for i in range(10):
                print('nop')
        'Create the secondary server redirecting to the primary server'
        new = self.get_readonly_server()
        redirecting = HTTPServerRedirecting(protocol_version=self._protocol_version)
        redirecting.redirect_to(new.host, new.port)
        redirecting._url_protocol = self._url_protocol
        return redirecting

    def get_old_url(self, relpath=None):
        if False:
            i = 10
            return i + 15
        base = self.old_server.get_url()
        return self._adjust_url(base, relpath)

    def get_old_transport(self, relpath=None):
        if False:
            i = 10
            return i + 15
        t = transport.get_transport_from_url(self.get_old_url(relpath))
        self.assertTrue(t.is_readonly())
        return t

    def get_new_url(self, relpath=None):
        if False:
            print('Hello World!')
        base = self.new_server.get_url()
        return self._adjust_url(base, relpath)

    def get_new_transport(self, relpath=None):
        if False:
            while True:
                i = 10
        t = transport.get_transport_from_url(self.get_new_url(relpath))
        self.assertTrue(t.is_readonly())
        return t

class AuthRequestHandler(http_server.TestingHTTPRequestHandler):
    """Requires an authentication to process requests.

    This is intended to be used with a server that always and
    only use one authentication scheme (implemented by daughter
    classes).
    """

    def _require_authentication(self):
        if False:
            for i in range(10):
                print('nop')
        tcs = self.server.test_case_server
        tcs.auth_required_errors += 1
        self.send_response(tcs.auth_error_code)
        self.send_header_auth_reqed()
        self.send_header('Content-Length', '0')
        self.end_headers()
        return

    def do_GET(self):
        if False:
            print('Hello World!')
        if self.authorized():
            return http_server.TestingHTTPRequestHandler.do_GET(self)
        else:
            return self._require_authentication()

    def do_HEAD(self):
        if False:
            return 10
        if self.authorized():
            return http_server.TestingHTTPRequestHandler.do_HEAD(self)
        else:
            return self._require_authentication()

class BasicAuthRequestHandler(AuthRequestHandler):
    """Implements the basic authentication of a request"""

    def authorized(self):
        if False:
            i = 10
            return i + 15
        tcs = self.server.test_case_server
        if tcs.auth_scheme != 'basic':
            return False
        auth_header = self.headers.get(tcs.auth_header_recv, None)
        if auth_header:
            (scheme, raw_auth) = auth_header.split(' ', 1)
            if scheme.lower() == tcs.auth_scheme:
                (user, password) = raw_auth.decode('base64').split(':')
                return tcs.authorized(user, password)
        return False

    def send_header_auth_reqed(self):
        if False:
            i = 10
            return i + 15
        tcs = self.server.test_case_server
        self.send_header(tcs.auth_header_sent, 'Basic realm="%s"' % tcs.auth_realm)

class DigestAuthRequestHandler(AuthRequestHandler):
    """Implements the digest authentication of a request.

    We need persistence for some attributes and that can't be
    achieved here since we get instantiated for each request. We
    rely on the DigestAuthServer to take care of them.
    """

    def authorized(self):
        if False:
            print('Hello World!')
        tcs = self.server.test_case_server
        auth_header = self.headers.get(tcs.auth_header_recv, None)
        if auth_header is None:
            return False
        (scheme, auth) = auth_header.split(None, 1)
        if scheme.lower() == tcs.auth_scheme:
            auth_dict = urllib2.parse_keqv_list(urllib2.parse_http_list(auth))
            return tcs.digest_authorized(auth_dict, self.command)
        return False

    def send_header_auth_reqed(self):
        if False:
            for i in range(10):
                print('nop')
        tcs = self.server.test_case_server
        header = 'Digest realm="%s", ' % tcs.auth_realm
        header += 'nonce="%s", algorithm="%s", qop="auth"' % (tcs.auth_nonce, 'MD5')
        self.send_header(tcs.auth_header_sent, header)

class DigestAndBasicAuthRequestHandler(DigestAuthRequestHandler):
    """Implements a digest and basic authentication of a request.

    I.e. the server proposes both schemes and the client should choose the best
    one it can handle, which, in that case, should be digest, the only scheme
    accepted here.
    """

    def send_header_auth_reqed(self):
        if False:
            while True:
                i = 10
        tcs = self.server.test_case_server
        self.send_header(tcs.auth_header_sent, 'Basic realm="%s"' % tcs.auth_realm)
        header = 'Digest realm="%s", ' % tcs.auth_realm
        header += 'nonce="%s", algorithm="%s", qop="auth"' % (tcs.auth_nonce, 'MD5')
        self.send_header(tcs.auth_header_sent, header)

class AuthServer(http_server.HttpServer):
    """Extends HttpServer with a dictionary of passwords.

    This is used as a base class for various schemes which should
    all use or redefined the associated AuthRequestHandler.

    Note that no users are defined by default, so add_user should
    be called before issuing the first request.
    """
    auth_header_sent = None
    auth_header_recv = None
    auth_error_code = None
    auth_realm = 'Thou should not pass'

    def __init__(self, request_handler, auth_scheme, protocol_version=None):
        if False:
            while True:
                i = 10
        http_server.HttpServer.__init__(self, request_handler, protocol_version=protocol_version)
        self.auth_scheme = auth_scheme
        self.password_of = {}
        self.auth_required_errors = 0

    def add_user(self, user, password):
        if False:
            i = 10
            return i + 15
        "Declare a user with an associated password.\n\n        password can be empty, use an empty string ('') in that\n        case, not None.\n        "
        self.password_of[user] = password

    def authorized(self, user, password):
        if False:
            print('Hello World!')
        'Check that the given user provided the right password'
        expected_password = self.password_of.get(user, None)
        return expected_password is not None and password == expected_password

class DigestAuthServer(AuthServer):
    """A digest authentication server"""
    auth_nonce = 'now!'

    def __init__(self, request_handler, auth_scheme, protocol_version=None):
        if False:
            print('Hello World!')
        AuthServer.__init__(self, request_handler, auth_scheme, protocol_version=protocol_version)

    def digest_authorized(self, auth, command):
        if False:
            for i in range(10):
                print('nop')
        nonce = auth['nonce']
        if nonce != self.auth_nonce:
            return False
        realm = auth['realm']
        if realm != self.auth_realm:
            return False
        user = auth['username']
        if not self.password_of.has_key(user):
            return False
        algorithm = auth['algorithm']
        if algorithm != 'MD5':
            return False
        qop = auth['qop']
        if qop != 'auth':
            return False
        password = self.password_of[user]
        A1 = '%s:%s:%s' % (user, realm, password)
        A2 = '%s:%s' % (command, auth['uri'])
        H = lambda x: osutils.md5(x).hexdigest()
        KD = lambda secret, data: H('%s:%s' % (secret, data))
        nonce_count = int(auth['nc'], 16)
        ncvalue = '%08x' % nonce_count
        cnonce = auth['cnonce']
        noncebit = '%s:%s:%s:%s:%s' % (nonce, ncvalue, cnonce, qop, H(A2))
        response_digest = KD(H(A1), noncebit)
        return response_digest == auth['response']

class HTTPAuthServer(AuthServer):
    """An HTTP server requiring authentication"""

    def init_http_auth(self):
        if False:
            for i in range(10):
                print('nop')
        self.auth_header_sent = 'WWW-Authenticate'
        self.auth_header_recv = 'Authorization'
        self.auth_error_code = 401

class ProxyAuthServer(AuthServer):
    """A proxy server requiring authentication"""

    def init_proxy_auth(self):
        if False:
            return 10
        self.proxy_requests = True
        self.auth_header_sent = 'Proxy-Authenticate'
        self.auth_header_recv = 'Proxy-Authorization'
        self.auth_error_code = 407

class HTTPBasicAuthServer(HTTPAuthServer):
    """An HTTP server requiring basic authentication"""

    def __init__(self, protocol_version=None):
        if False:
            i = 10
            return i + 15
        HTTPAuthServer.__init__(self, BasicAuthRequestHandler, 'basic', protocol_version=protocol_version)
        self.init_http_auth()

class HTTPDigestAuthServer(DigestAuthServer, HTTPAuthServer):
    """An HTTP server requiring digest authentication"""

    def __init__(self, protocol_version=None):
        if False:
            return 10
        DigestAuthServer.__init__(self, DigestAuthRequestHandler, 'digest', protocol_version=protocol_version)
        self.init_http_auth()

class HTTPBasicAndDigestAuthServer(DigestAuthServer, HTTPAuthServer):
    """An HTTP server requiring basic or digest authentication"""

    def __init__(self, protocol_version=None):
        if False:
            for i in range(10):
                print('nop')
        DigestAuthServer.__init__(self, DigestAndBasicAuthRequestHandler, 'basicdigest', protocol_version=protocol_version)
        self.init_http_auth()
        self.auth_scheme = 'digest'

class ProxyBasicAuthServer(ProxyAuthServer):
    """A proxy server requiring basic authentication"""

    def __init__(self, protocol_version=None):
        if False:
            while True:
                i = 10
        ProxyAuthServer.__init__(self, BasicAuthRequestHandler, 'basic', protocol_version=protocol_version)
        self.init_proxy_auth()

class ProxyDigestAuthServer(DigestAuthServer, ProxyAuthServer):
    """A proxy server requiring basic authentication"""

    def __init__(self, protocol_version=None):
        if False:
            while True:
                i = 10
        ProxyAuthServer.__init__(self, DigestAuthRequestHandler, 'digest', protocol_version=protocol_version)
        self.init_proxy_auth()

class ProxyBasicAndDigestAuthServer(DigestAuthServer, ProxyAuthServer):
    """An proxy server requiring basic or digest authentication"""

    def __init__(self, protocol_version=None):
        if False:
            return 10
        DigestAuthServer.__init__(self, DigestAndBasicAuthRequestHandler, 'basicdigest', protocol_version=protocol_version)
        self.init_proxy_auth()
        self.auth_scheme = 'digest'