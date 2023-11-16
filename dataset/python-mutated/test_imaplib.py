from test import support
from test.support import socket_helper
from contextlib import contextmanager
import imaplib
import os.path
import socketserver
import time
import calendar
import threading
import socket
from test.support import verbose, run_with_tz, run_with_locale, cpython_only
from test.support import hashlib_helper
from test.support import threading_helper
from test.support import warnings_helper
import unittest
from unittest import mock
from datetime import datetime, timezone, timedelta
try:
    import ssl
except ImportError:
    ssl = None
CERTFILE = os.path.join(os.path.dirname(__file__) or os.curdir, 'keycert3.pem')
CAFILE = os.path.join(os.path.dirname(__file__) or os.curdir, 'pycacert.pem')

class TestImaplib(unittest.TestCase):

    def test_Internaldate2tuple(self):
        if False:
            i = 10
            return i + 15
        t0 = calendar.timegm((2000, 1, 1, 0, 0, 0, -1, -1, -1))
        tt = imaplib.Internaldate2tuple(b'25 (INTERNALDATE "01-Jan-2000 00:00:00 +0000")')
        self.assertEqual(time.mktime(tt), t0)
        tt = imaplib.Internaldate2tuple(b'25 (INTERNALDATE "01-Jan-2000 11:30:00 +1130")')
        self.assertEqual(time.mktime(tt), t0)
        tt = imaplib.Internaldate2tuple(b'25 (INTERNALDATE "31-Dec-1999 12:30:00 -1130")')
        self.assertEqual(time.mktime(tt), t0)

    @run_with_tz('MST+07MDT,M4.1.0,M10.5.0')
    def test_Internaldate2tuple_issue10941(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertNotEqual(imaplib.Internaldate2tuple(b'25 (INTERNALDATE "02-Apr-2000 02:30:00 +0000")'), imaplib.Internaldate2tuple(b'25 (INTERNALDATE "02-Apr-2000 03:30:00 +0000")'))

    def timevalues(self):
        if False:
            i = 10
            return i + 15
        return [2000000000, 2000000000.0, time.localtime(2000000000), (2033, 5, 18, 5, 33, 20, -1, -1, -1), (2033, 5, 18, 5, 33, 20, -1, -1, 1), datetime.fromtimestamp(2000000000, timezone(timedelta(0, 2 * 60 * 60))), '"18-May-2033 05:33:20 +0200"']

    @run_with_locale('LC_ALL', 'de_DE', 'fr_FR')
    @run_with_tz('STD-1DST,M3.2.0,M11.1.0')
    def test_Time2Internaldate(self):
        if False:
            return 10
        expected = '"18-May-2033 05:33:20 +0200"'
        for t in self.timevalues():
            internal = imaplib.Time2Internaldate(t)
            self.assertEqual(internal, expected)

    def test_that_Time2Internaldate_returns_a_result(self):
        if False:
            i = 10
            return i + 15
        for t in self.timevalues():
            imaplib.Time2Internaldate(t)

    def test_imap4_host_default_value(self):
        if False:
            return 10
        with socket.socket() as s:
            try:
                s.connect(('', imaplib.IMAP4_PORT))
                self.skipTest('Cannot run the test with local IMAP server running.')
            except socket.error:
                pass
        expected_errnos = socket_helper.get_socket_conn_refused_errs()
        with self.assertRaises(OSError) as cm:
            imaplib.IMAP4()
        self.assertIn(cm.exception.errno, expected_errnos)
if ssl:

    class SecureTCPServer(socketserver.TCPServer):

        def get_request(self):
            if False:
                for i in range(10):
                    print('nop')
            (newsocket, fromaddr) = self.socket.accept()
            context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
            context.load_cert_chain(CERTFILE)
            connstream = context.wrap_socket(newsocket, server_side=True)
            return (connstream, fromaddr)
    IMAP4_SSL = imaplib.IMAP4_SSL
else:

    class SecureTCPServer:
        pass
    IMAP4_SSL = None

class SimpleIMAPHandler(socketserver.StreamRequestHandler):
    timeout = support.LOOPBACK_TIMEOUT
    continuation = None
    capabilities = ''

    def setup(self):
        if False:
            while True:
                i = 10
        super().setup()
        self.server.is_selected = False
        self.server.logged = None

    def _send(self, message):
        if False:
            while True:
                i = 10
        if verbose:
            print('SENT: %r' % message.strip())
        self.wfile.write(message)

    def _send_line(self, message):
        if False:
            for i in range(10):
                print('nop')
        self._send(message + b'\r\n')

    def _send_textline(self, message):
        if False:
            for i in range(10):
                print('nop')
        self._send_line(message.encode('ASCII'))

    def _send_tagged(self, tag, code, message):
        if False:
            print('Hello World!')
        self._send_textline(' '.join((tag, code, message)))

    def handle(self):
        if False:
            print('Hello World!')
        self._send_textline('* OK IMAP4rev1')
        while 1:
            line = b''
            while 1:
                try:
                    part = self.rfile.read(1)
                    if part == b'':
                        return
                    line += part
                except OSError:
                    return
                if line.endswith(b'\r\n'):
                    break
            if verbose:
                print('GOT: %r' % line.strip())
            if self.continuation:
                try:
                    self.continuation.send(line)
                except StopIteration:
                    self.continuation = None
                continue
            splitline = line.decode('ASCII').split()
            tag = splitline[0]
            cmd = splitline[1]
            args = splitline[2:]
            if hasattr(self, 'cmd_' + cmd):
                continuation = getattr(self, 'cmd_' + cmd)(tag, args)
                if continuation:
                    self.continuation = continuation
                    next(continuation)
            else:
                self._send_tagged(tag, 'BAD', cmd + ' unknown')

    def cmd_CAPABILITY(self, tag, args):
        if False:
            print('Hello World!')
        caps = 'IMAP4rev1 ' + self.capabilities if self.capabilities else 'IMAP4rev1'
        self._send_textline('* CAPABILITY ' + caps)
        self._send_tagged(tag, 'OK', 'CAPABILITY completed')

    def cmd_LOGOUT(self, tag, args):
        if False:
            for i in range(10):
                print('nop')
        self.server.logged = None
        self._send_textline('* BYE IMAP4ref1 Server logging out')
        self._send_tagged(tag, 'OK', 'LOGOUT completed')

    def cmd_LOGIN(self, tag, args):
        if False:
            return 10
        self.server.logged = args[0]
        self._send_tagged(tag, 'OK', 'LOGIN completed')

    def cmd_SELECT(self, tag, args):
        if False:
            return 10
        self.server.is_selected = True
        self._send_line(b'* 2 EXISTS')
        self._send_tagged(tag, 'OK', '[READ-WRITE] SELECT completed.')

    def cmd_UNSELECT(self, tag, args):
        if False:
            return 10
        if self.server.is_selected:
            self.server.is_selected = False
            self._send_tagged(tag, 'OK', 'Returned to authenticated state. (Success)')
        else:
            self._send_tagged(tag, 'BAD', 'No mailbox selected')

class NewIMAPTestsMixin:
    client = None

    def _setup(self, imap_handler, connect=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Sets up imap_handler for tests. imap_handler should inherit from either:\n        - SimpleIMAPHandler - for testing IMAP commands,\n        - socketserver.StreamRequestHandler - if raw access to stream is needed.\n        Returns (client, server).\n        '

        class TestTCPServer(self.server_class):

            def handle_error(self, request, client_address):
                if False:
                    print('Hello World!')
                '\n                End request and raise the error if one occurs.\n                '
                self.close_request(request)
                self.server_close()
                raise
        self.addCleanup(self._cleanup)
        self.server = self.server_class((socket_helper.HOST, 0), imap_handler)
        self.thread = threading.Thread(name=self._testMethodName + '-server', target=self.server.serve_forever, kwargs={'poll_interval': 0.01})
        self.thread.daemon = True
        self.thread.start()
        if connect:
            self.client = self.imap_class(*self.server.server_address)
        return (self.client, self.server)

    def _cleanup(self):
        if False:
            i = 10
            return i + 15
        '\n        Cleans up the test server. This method should not be called manually,\n        it is added to the cleanup queue in the _setup method already.\n        '
        if self.client is not None and self.client.state != 'LOGOUT':
            self.client.shutdown()
        self.server.shutdown()
        self.server.server_close()
        threading_helper.join_thread(self.thread)
        self.thread = None

    def test_EOF_without_complete_welcome_message(self):
        if False:
            while True:
                i = 10

        class EOFHandler(socketserver.StreamRequestHandler):

            def handle(self):
                if False:
                    while True:
                        i = 10
                self.wfile.write(b'* OK')
        (_, server) = self._setup(EOFHandler, connect=False)
        self.assertRaises(imaplib.IMAP4.abort, self.imap_class, *server.server_address)

    def test_line_termination(self):
        if False:
            print('Hello World!')

        class BadNewlineHandler(SimpleIMAPHandler):

            def cmd_CAPABILITY(self, tag, args):
                if False:
                    return 10
                self._send(b'* CAPABILITY IMAP4rev1 AUTH\n')
                self._send_tagged(tag, 'OK', 'CAPABILITY completed')
        (_, server) = self._setup(BadNewlineHandler, connect=False)
        self.assertRaises(imaplib.IMAP4.abort, self.imap_class, *server.server_address)

    def test_enable_raises_error_if_not_AUTH(self):
        if False:
            print('Hello World!')

        class EnableHandler(SimpleIMAPHandler):
            capabilities = 'AUTH ENABLE UTF8=ACCEPT'
        (client, _) = self._setup(EnableHandler)
        self.assertFalse(client.utf8_enabled)
        with self.assertRaisesRegex(imaplib.IMAP4.error, 'ENABLE.*NONAUTH'):
            client.enable('foo')
        self.assertFalse(client.utf8_enabled)

    def test_enable_raises_error_if_no_capability(self):
        if False:
            return 10
        (client, _) = self._setup(SimpleIMAPHandler)
        with self.assertRaisesRegex(imaplib.IMAP4.error, 'does not support ENABLE'):
            client.enable('foo')

    def test_enable_UTF8_raises_error_if_not_supported(self):
        if False:
            i = 10
            return i + 15
        (client, _) = self._setup(SimpleIMAPHandler)
        (typ, data) = client.login('user', 'pass')
        self.assertEqual(typ, 'OK')
        with self.assertRaisesRegex(imaplib.IMAP4.error, 'does not support ENABLE'):
            client.enable('UTF8=ACCEPT')

    def test_enable_UTF8_True_append(self):
        if False:
            i = 10
            return i + 15

        class UTF8AppendServer(SimpleIMAPHandler):
            capabilities = 'ENABLE UTF8=ACCEPT'

            def cmd_ENABLE(self, tag, args):
                if False:
                    return 10
                self._send_tagged(tag, 'OK', 'ENABLE successful')

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'FAKEAUTH successful')

            def cmd_APPEND(self, tag, args):
                if False:
                    return 10
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'okay')
        (client, server) = self._setup(UTF8AppendServer)
        self.assertEqual(client._encoding, 'ascii')
        (code, _) = client.authenticate('MYAUTH', lambda x: b'fake')
        self.assertEqual(code, 'OK')
        self.assertEqual(server.response, b'ZmFrZQ==\r\n')
        (code, _) = client.enable('UTF8=ACCEPT')
        self.assertEqual(code, 'OK')
        self.assertEqual(client._encoding, 'utf-8')
        msg_string = 'Subject: üñí©öðé'
        (typ, data) = client.append(None, None, None, msg_string.encode('utf-8'))
        self.assertEqual(typ, 'OK')
        self.assertEqual(server.response, ('UTF8 (%s)\r\n' % msg_string).encode('utf-8'))

    def test_search_disallows_charset_in_utf8_mode(self):
        if False:
            return 10

        class UTF8Server(SimpleIMAPHandler):
            capabilities = 'AUTH ENABLE UTF8=ACCEPT'

            def cmd_ENABLE(self, tag, args):
                if False:
                    i = 10
                    return i + 15
                self._send_tagged(tag, 'OK', 'ENABLE successful')

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'FAKEAUTH successful')
        (client, _) = self._setup(UTF8Server)
        (typ, _) = client.authenticate('MYAUTH', lambda x: b'fake')
        self.assertEqual(typ, 'OK')
        (typ, _) = client.enable('UTF8=ACCEPT')
        self.assertEqual(typ, 'OK')
        self.assertTrue(client.utf8_enabled)
        with self.assertRaisesRegex(imaplib.IMAP4.error, 'charset.*UTF8'):
            client.search('foo', 'bar')

    def test_bad_auth_name(self):
        if False:
            i = 10
            return i + 15

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    return 10
                self._send_tagged(tag, 'NO', 'unrecognized authentication type {}'.format(args[0]))
        (client, _) = self._setup(MyServer)
        with self.assertRaisesRegex(imaplib.IMAP4.error, 'unrecognized authentication type METHOD'):
            client.authenticate('METHOD', lambda : 1)

    def test_invalid_authentication(self):
        if False:
            return 10

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    return 10
                self._send_textline('+')
                self.response = (yield)
                self._send_tagged(tag, 'NO', '[AUTHENTICATIONFAILED] invalid')
        (client, _) = self._setup(MyServer)
        with self.assertRaisesRegex(imaplib.IMAP4.error, '\\[AUTHENTICATIONFAILED\\] invalid'):
            client.authenticate('MYAUTH', lambda x: b'fake')

    def test_valid_authentication_bytes(self):
        if False:
            i = 10
            return i + 15

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'FAKEAUTH successful')
        (client, server) = self._setup(MyServer)
        (code, _) = client.authenticate('MYAUTH', lambda x: b'fake')
        self.assertEqual(code, 'OK')
        self.assertEqual(server.response, b'ZmFrZQ==\r\n')

    def test_valid_authentication_plain_text(self):
        if False:
            print('Hello World!')

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    while True:
                        i = 10
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'FAKEAUTH successful')
        (client, server) = self._setup(MyServer)
        (code, _) = client.authenticate('MYAUTH', lambda x: 'fake')
        self.assertEqual(code, 'OK')
        self.assertEqual(server.response, b'ZmFrZQ==\r\n')

    @hashlib_helper.requires_hashdigest('md5', openssl=True)
    def test_login_cram_md5_bytes(self):
        if False:
            for i in range(10):
                print('nop')

        class AuthHandler(SimpleIMAPHandler):
            capabilities = 'LOGINDISABLED AUTH=CRAM-MD5'

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    i = 10
                    return i + 15
                self._send_textline('+ PDE4OTYuNjk3MTcwOTUyQHBvc3RvZmZpY2UucmVzdG9uLm1jaS5uZXQ=')
                r = (yield)
                if r == b'dGltIGYxY2E2YmU0NjRiOWVmYTFjY2E2ZmZkNmNmMmQ5ZjMy\r\n':
                    self._send_tagged(tag, 'OK', 'CRAM-MD5 successful')
                else:
                    self._send_tagged(tag, 'NO', 'No access')
        (client, _) = self._setup(AuthHandler)
        self.assertTrue('AUTH=CRAM-MD5' in client.capabilities)
        (ret, _) = client.login_cram_md5('tim', b'tanstaaftanstaaf')
        self.assertEqual(ret, 'OK')

    @hashlib_helper.requires_hashdigest('md5', openssl=True)
    def test_login_cram_md5_plain_text(self):
        if False:
            print('Hello World!')

        class AuthHandler(SimpleIMAPHandler):
            capabilities = 'LOGINDISABLED AUTH=CRAM-MD5'

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    for i in range(10):
                        print('nop')
                self._send_textline('+ PDE4OTYuNjk3MTcwOTUyQHBvc3RvZmZpY2UucmVzdG9uLm1jaS5uZXQ=')
                r = (yield)
                if r == b'dGltIGYxY2E2YmU0NjRiOWVmYTFjY2E2ZmZkNmNmMmQ5ZjMy\r\n':
                    self._send_tagged(tag, 'OK', 'CRAM-MD5 successful')
                else:
                    self._send_tagged(tag, 'NO', 'No access')
        (client, _) = self._setup(AuthHandler)
        self.assertTrue('AUTH=CRAM-MD5' in client.capabilities)
        (ret, _) = client.login_cram_md5('tim', 'tanstaaftanstaaf')
        self.assertEqual(ret, 'OK')

    def test_aborted_authentication(self):
        if False:
            while True:
                i = 10

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    while True:
                        i = 10
                self._send_textline('+')
                self.response = (yield)
                if self.response == b'*\r\n':
                    self._send_tagged(tag, 'NO', '[AUTHENTICATIONFAILED] aborted')
                else:
                    self._send_tagged(tag, 'OK', 'MYAUTH successful')
        (client, _) = self._setup(MyServer)
        with self.assertRaisesRegex(imaplib.IMAP4.error, '\\[AUTHENTICATIONFAILED\\] aborted'):
            client.authenticate('MYAUTH', lambda x: None)

    @mock.patch('imaplib._MAXLINE', 10)
    def test_linetoolong(self):
        if False:
            return 10

        class TooLongHandler(SimpleIMAPHandler):

            def handle(self):
                if False:
                    while True:
                        i = 10
                self.wfile.write(b'* OK ' + 11 * b'x' + b'\r\n')
        (_, server) = self._setup(TooLongHandler, connect=False)
        with self.assertRaisesRegex(imaplib.IMAP4.error, 'got more than 10 bytes'):
            self.imap_class(*server.server_address)

    def test_simple_with_statement(self):
        if False:
            return 10
        (_, server) = self._setup(SimpleIMAPHandler, connect=False)
        with self.imap_class(*server.server_address):
            pass

    def test_imaplib_timeout_test(self):
        if False:
            for i in range(10):
                print('nop')
        (_, server) = self._setup(SimpleIMAPHandler)
        addr = server.server_address[1]
        client = self.imap_class('localhost', addr, timeout=None)
        self.assertEqual(client.sock.timeout, None)
        client.shutdown()
        client = self.imap_class('localhost', addr, timeout=support.LOOPBACK_TIMEOUT)
        self.assertEqual(client.sock.timeout, support.LOOPBACK_TIMEOUT)
        client.shutdown()
        with self.assertRaises(ValueError):
            client = self.imap_class('localhost', addr, timeout=0)

    def test_imaplib_timeout_functionality_test(self):
        if False:
            return 10

        class TimeoutHandler(SimpleIMAPHandler):

            def handle(self):
                if False:
                    print('Hello World!')
                time.sleep(1)
                SimpleIMAPHandler.handle(self)
        (_, server) = self._setup(TimeoutHandler)
        addr = server.server_address[1]
        with self.assertRaises(TimeoutError):
            client = self.imap_class('localhost', addr, timeout=0.001)

    def test_with_statement(self):
        if False:
            print('Hello World!')
        (_, server) = self._setup(SimpleIMAPHandler, connect=False)
        with self.imap_class(*server.server_address) as imap:
            imap.login('user', 'pass')
            self.assertEqual(server.logged, 'user')
        self.assertIsNone(server.logged)

    def test_with_statement_logout(self):
        if False:
            for i in range(10):
                print('nop')
        (_, server) = self._setup(SimpleIMAPHandler, connect=False)
        with self.imap_class(*server.server_address) as imap:
            imap.login('user', 'pass')
            self.assertEqual(server.logged, 'user')
            imap.logout()
            self.assertIsNone(server.logged)
        self.assertIsNone(server.logged)

    def test_login(self):
        if False:
            return 10
        (client, _) = self._setup(SimpleIMAPHandler)
        (typ, data) = client.login('user', 'pass')
        self.assertEqual(typ, 'OK')
        self.assertEqual(data[0], b'LOGIN completed')
        self.assertEqual(client.state, 'AUTH')

    def test_logout(self):
        if False:
            i = 10
            return i + 15
        (client, _) = self._setup(SimpleIMAPHandler)
        (typ, data) = client.login('user', 'pass')
        self.assertEqual(typ, 'OK')
        self.assertEqual(data[0], b'LOGIN completed')
        (typ, data) = client.logout()
        self.assertEqual(typ, 'BYE', (typ, data))
        self.assertEqual(data[0], b'IMAP4ref1 Server logging out', (typ, data))
        self.assertEqual(client.state, 'LOGOUT')

    def test_lsub(self):
        if False:
            while True:
                i = 10

        class LsubCmd(SimpleIMAPHandler):

            def cmd_LSUB(self, tag, args):
                if False:
                    for i in range(10):
                        print('nop')
                self._send_textline('* LSUB () "." directoryA')
                return self._send_tagged(tag, 'OK', 'LSUB completed')
        (client, _) = self._setup(LsubCmd)
        client.login('user', 'pass')
        (typ, data) = client.lsub()
        self.assertEqual(typ, 'OK')
        self.assertEqual(data[0], b'() "." directoryA')

    def test_unselect(self):
        if False:
            return 10
        (client, _) = self._setup(SimpleIMAPHandler)
        client.login('user', 'pass')
        (typ, data) = client.select()
        self.assertEqual(typ, 'OK')
        self.assertEqual(data[0], b'2')
        (typ, data) = client.unselect()
        self.assertEqual(typ, 'OK')
        self.assertEqual(data[0], b'Returned to authenticated state. (Success)')
        self.assertEqual(client.state, 'AUTH')

class NewIMAPTests(NewIMAPTestsMixin, unittest.TestCase):
    imap_class = imaplib.IMAP4
    server_class = socketserver.TCPServer

@unittest.skipUnless(ssl, 'SSL not available')
class NewIMAPSSLTests(NewIMAPTestsMixin, unittest.TestCase):
    imap_class = IMAP4_SSL
    server_class = SecureTCPServer

    def test_ssl_raises(self):
        if False:
            for i in range(10):
                print('nop')
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        self.assertEqual(ssl_context.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(ssl_context.check_hostname, True)
        ssl_context.load_verify_locations(CAFILE)
        with self.assertRaisesRegex(ssl.CertificateError, "IP address mismatch, certificate is not valid for '127.0.0.1'"):
            (_, server) = self._setup(SimpleIMAPHandler)
            client = self.imap_class(*server.server_address, ssl_context=ssl_context)
            client.shutdown()

    def test_ssl_verified(self):
        if False:
            i = 10
            return i + 15
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.load_verify_locations(CAFILE)
        (_, server) = self._setup(SimpleIMAPHandler)
        client = self.imap_class('localhost', server.server_address[1], ssl_context=ssl_context)
        client.shutdown()

    @cpython_only
    def test_certfile_arg_warn(self):
        if False:
            for i in range(10):
                print('nop')
        with warnings_helper.check_warnings(('', DeprecationWarning)):
            with mock.patch.object(self.imap_class, 'open'):
                with mock.patch.object(self.imap_class, '_connect'):
                    self.imap_class('localhost', 143, certfile=CERTFILE)

class ThreadedNetworkedTests(unittest.TestCase):
    server_class = socketserver.TCPServer
    imap_class = imaplib.IMAP4

    def make_server(self, addr, hdlr):
        if False:
            print('Hello World!')

        class MyServer(self.server_class):

            def handle_error(self, request, client_address):
                if False:
                    print('Hello World!')
                self.close_request(request)
                self.server_close()
                raise
        if verbose:
            print('creating server')
        server = MyServer(addr, hdlr)
        self.assertEqual(server.server_address, server.socket.getsockname())
        if verbose:
            print('server created')
            print('ADDR =', addr)
            print('CLASS =', self.server_class)
            print('HDLR =', server.RequestHandlerClass)
        t = threading.Thread(name='%s serving' % self.server_class, target=server.serve_forever, kwargs={'poll_interval': 0.01})
        t.daemon = True
        t.start()
        if verbose:
            print('server running')
        return (server, t)

    def reap_server(self, server, thread):
        if False:
            return 10
        if verbose:
            print('waiting for server')
        server.shutdown()
        server.server_close()
        thread.join()
        if verbose:
            print('done')

    @contextmanager
    def reaped_server(self, hdlr):
        if False:
            i = 10
            return i + 15
        (server, thread) = self.make_server((socket_helper.HOST, 0), hdlr)
        try:
            yield server
        finally:
            self.reap_server(server, thread)

    @contextmanager
    def reaped_pair(self, hdlr):
        if False:
            print('Hello World!')
        with self.reaped_server(hdlr) as server:
            client = self.imap_class(*server.server_address)
            try:
                yield (server, client)
            finally:
                client.logout()

    @threading_helper.reap_threads
    def test_connect(self):
        if False:
            i = 10
            return i + 15
        with self.reaped_server(SimpleIMAPHandler) as server:
            client = self.imap_class(*server.server_address)
            client.shutdown()

    @threading_helper.reap_threads
    def test_bracket_flags(self):
        if False:
            for i in range(10):
                print('nop')

        class BracketFlagHandler(SimpleIMAPHandler):

            def handle(self):
                if False:
                    print('Hello World!')
                self.flags = ['Answered', 'Flagged', 'Deleted', 'Seen', 'Draft']
                super().handle()

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'FAKEAUTH successful')

            def cmd_SELECT(self, tag, args):
                if False:
                    while True:
                        i = 10
                flag_msg = ' \\'.join(self.flags)
                self._send_line(('* FLAGS (%s)' % flag_msg).encode('ascii'))
                self._send_line(b'* 2 EXISTS')
                self._send_line(b'* 0 RECENT')
                msg = '* OK [PERMANENTFLAGS %s \\*)] Flags permitted.' % flag_msg
                self._send_line(msg.encode('ascii'))
                self._send_tagged(tag, 'OK', '[READ-WRITE] SELECT completed.')

            def cmd_STORE(self, tag, args):
                if False:
                    print('Hello World!')
                new_flags = args[2].strip('(').strip(')').split()
                self.flags.extend(new_flags)
                flags_msg = '(FLAGS (%s))' % ' \\'.join(self.flags)
                msg = '* %s FETCH %s' % (args[0], flags_msg)
                self._send_line(msg.encode('ascii'))
                self._send_tagged(tag, 'OK', 'STORE completed.')
        with self.reaped_pair(BracketFlagHandler) as (server, client):
            (code, data) = client.authenticate('MYAUTH', lambda x: b'fake')
            self.assertEqual(code, 'OK')
            self.assertEqual(server.response, b'ZmFrZQ==\r\n')
            client.select('test')
            (typ, [data]) = client.store(b'1', '+FLAGS', '[test]')
            self.assertIn(b'[test]', data)
            client.select('test')
            (typ, [data]) = client.response('PERMANENTFLAGS')
            self.assertIn(b'[test]', data)

    @threading_helper.reap_threads
    def test_issue5949(self):
        if False:
            while True:
                i = 10

        class EOFHandler(socketserver.StreamRequestHandler):

            def handle(self):
                if False:
                    while True:
                        i = 10
                self.wfile.write(b'* OK')
        with self.reaped_server(EOFHandler) as server:
            self.assertRaises(imaplib.IMAP4.abort, self.imap_class, *server.server_address)

    @threading_helper.reap_threads
    def test_line_termination(self):
        if False:
            i = 10
            return i + 15

        class BadNewlineHandler(SimpleIMAPHandler):

            def cmd_CAPABILITY(self, tag, args):
                if False:
                    return 10
                self._send(b'* CAPABILITY IMAP4rev1 AUTH\n')
                self._send_tagged(tag, 'OK', 'CAPABILITY completed')
        with self.reaped_server(BadNewlineHandler) as server:
            self.assertRaises(imaplib.IMAP4.abort, self.imap_class, *server.server_address)

    class UTF8Server(SimpleIMAPHandler):
        capabilities = 'AUTH ENABLE UTF8=ACCEPT'

        def cmd_ENABLE(self, tag, args):
            if False:
                i = 10
                return i + 15
            self._send_tagged(tag, 'OK', 'ENABLE successful')

        def cmd_AUTHENTICATE(self, tag, args):
            if False:
                print('Hello World!')
            self._send_textline('+')
            self.server.response = (yield)
            self._send_tagged(tag, 'OK', 'FAKEAUTH successful')

    @threading_helper.reap_threads
    def test_enable_raises_error_if_not_AUTH(self):
        if False:
            return 10
        with self.reaped_pair(self.UTF8Server) as (server, client):
            self.assertFalse(client.utf8_enabled)
            self.assertRaises(imaplib.IMAP4.error, client.enable, 'foo')
            self.assertFalse(client.utf8_enabled)

    @threading_helper.reap_threads
    def test_enable_raises_error_if_no_capability(self):
        if False:
            print('Hello World!')

        class NoEnableServer(self.UTF8Server):
            capabilities = 'AUTH'
        with self.reaped_pair(NoEnableServer) as (server, client):
            self.assertRaises(imaplib.IMAP4.error, client.enable, 'foo')

    @threading_helper.reap_threads
    def test_enable_UTF8_raises_error_if_not_supported(self):
        if False:
            while True:
                i = 10

        class NonUTF8Server(SimpleIMAPHandler):
            pass
        with self.assertRaises(imaplib.IMAP4.error):
            with self.reaped_pair(NonUTF8Server) as (server, client):
                (typ, data) = client.login('user', 'pass')
                self.assertEqual(typ, 'OK')
                client.enable('UTF8=ACCEPT')
                pass

    @threading_helper.reap_threads
    def test_enable_UTF8_True_append(self):
        if False:
            for i in range(10):
                print('nop')

        class UTF8AppendServer(self.UTF8Server):

            def cmd_APPEND(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'okay')
        with self.reaped_pair(UTF8AppendServer) as (server, client):
            self.assertEqual(client._encoding, 'ascii')
            (code, _) = client.authenticate('MYAUTH', lambda x: b'fake')
            self.assertEqual(code, 'OK')
            self.assertEqual(server.response, b'ZmFrZQ==\r\n')
            (code, _) = client.enable('UTF8=ACCEPT')
            self.assertEqual(code, 'OK')
            self.assertEqual(client._encoding, 'utf-8')
            msg_string = 'Subject: üñí©öðé'
            (typ, data) = client.append(None, None, None, msg_string.encode('utf-8'))
            self.assertEqual(typ, 'OK')
            self.assertEqual(server.response, ('UTF8 (%s)\r\n' % msg_string).encode('utf-8'))

    @threading_helper.reap_threads
    def test_search_disallows_charset_in_utf8_mode(self):
        if False:
            print('Hello World!')
        with self.reaped_pair(self.UTF8Server) as (server, client):
            (typ, _) = client.authenticate('MYAUTH', lambda x: b'fake')
            self.assertEqual(typ, 'OK')
            (typ, _) = client.enable('UTF8=ACCEPT')
            self.assertEqual(typ, 'OK')
            self.assertTrue(client.utf8_enabled)
            self.assertRaises(imaplib.IMAP4.error, client.search, 'foo', 'bar')

    @threading_helper.reap_threads
    def test_bad_auth_name(self):
        if False:
            i = 10
            return i + 15

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    return 10
                self._send_tagged(tag, 'NO', 'unrecognized authentication type {}'.format(args[0]))
        with self.reaped_pair(MyServer) as (server, client):
            with self.assertRaises(imaplib.IMAP4.error):
                client.authenticate('METHOD', lambda : 1)

    @threading_helper.reap_threads
    def test_invalid_authentication(self):
        if False:
            print('Hello World!')

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.response = (yield)
                self._send_tagged(tag, 'NO', '[AUTHENTICATIONFAILED] invalid')
        with self.reaped_pair(MyServer) as (server, client):
            with self.assertRaises(imaplib.IMAP4.error):
                (code, data) = client.authenticate('MYAUTH', lambda x: b'fake')

    @threading_helper.reap_threads
    def test_valid_authentication(self):
        if False:
            i = 10
            return i + 15

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    i = 10
                    return i + 15
                self._send_textline('+')
                self.server.response = (yield)
                self._send_tagged(tag, 'OK', 'FAKEAUTH successful')
        with self.reaped_pair(MyServer) as (server, client):
            (code, data) = client.authenticate('MYAUTH', lambda x: b'fake')
            self.assertEqual(code, 'OK')
            self.assertEqual(server.response, b'ZmFrZQ==\r\n')
        with self.reaped_pair(MyServer) as (server, client):
            (code, data) = client.authenticate('MYAUTH', lambda x: 'fake')
            self.assertEqual(code, 'OK')
            self.assertEqual(server.response, b'ZmFrZQ==\r\n')

    @threading_helper.reap_threads
    @hashlib_helper.requires_hashdigest('md5', openssl=True)
    def test_login_cram_md5(self):
        if False:
            for i in range(10):
                print('nop')

        class AuthHandler(SimpleIMAPHandler):
            capabilities = 'LOGINDISABLED AUTH=CRAM-MD5'

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    for i in range(10):
                        print('nop')
                self._send_textline('+ PDE4OTYuNjk3MTcwOTUyQHBvc3RvZmZpY2UucmVzdG9uLm1jaS5uZXQ=')
                r = (yield)
                if r == b'dGltIGYxY2E2YmU0NjRiOWVmYTFjY2E2ZmZkNmNmMmQ5ZjMy\r\n':
                    self._send_tagged(tag, 'OK', 'CRAM-MD5 successful')
                else:
                    self._send_tagged(tag, 'NO', 'No access')
        with self.reaped_pair(AuthHandler) as (server, client):
            self.assertTrue('AUTH=CRAM-MD5' in client.capabilities)
            (ret, data) = client.login_cram_md5('tim', 'tanstaaftanstaaf')
            self.assertEqual(ret, 'OK')
        with self.reaped_pair(AuthHandler) as (server, client):
            self.assertTrue('AUTH=CRAM-MD5' in client.capabilities)
            (ret, data) = client.login_cram_md5('tim', b'tanstaaftanstaaf')
            self.assertEqual(ret, 'OK')

    @threading_helper.reap_threads
    def test_aborted_authentication(self):
        if False:
            for i in range(10):
                print('nop')

        class MyServer(SimpleIMAPHandler):

            def cmd_AUTHENTICATE(self, tag, args):
                if False:
                    print('Hello World!')
                self._send_textline('+')
                self.response = (yield)
                if self.response == b'*\r\n':
                    self._send_tagged(tag, 'NO', '[AUTHENTICATIONFAILED] aborted')
                else:
                    self._send_tagged(tag, 'OK', 'MYAUTH successful')
        with self.reaped_pair(MyServer) as (server, client):
            with self.assertRaises(imaplib.IMAP4.error):
                (code, data) = client.authenticate('MYAUTH', lambda x: None)

    def test_linetoolong(self):
        if False:
            return 10

        class TooLongHandler(SimpleIMAPHandler):

            def handle(self):
                if False:
                    print('Hello World!')
                self.wfile.write(b'* OK ' + imaplib._MAXLINE * b'x' + b'\r\n')
        with self.reaped_server(TooLongHandler) as server:
            self.assertRaises(imaplib.IMAP4.error, self.imap_class, *server.server_address)

    @threading_helper.reap_threads
    def test_simple_with_statement(self):
        if False:
            print('Hello World!')
        with self.reaped_server(SimpleIMAPHandler) as server:
            with self.imap_class(*server.server_address):
                pass

    @threading_helper.reap_threads
    def test_with_statement(self):
        if False:
            print('Hello World!')
        with self.reaped_server(SimpleIMAPHandler) as server:
            with self.imap_class(*server.server_address) as imap:
                imap.login('user', 'pass')
                self.assertEqual(server.logged, 'user')
            self.assertIsNone(server.logged)

    @threading_helper.reap_threads
    def test_with_statement_logout(self):
        if False:
            while True:
                i = 10
        with self.reaped_server(SimpleIMAPHandler) as server:
            with self.imap_class(*server.server_address) as imap:
                imap.login('user', 'pass')
                self.assertEqual(server.logged, 'user')
                imap.logout()
                self.assertIsNone(server.logged)
            self.assertIsNone(server.logged)

    @threading_helper.reap_threads
    @cpython_only
    @unittest.skipUnless(__debug__, "Won't work if __debug__ is False")
    def test_dump_ur(self):
        if False:
            return 10
        untagged_resp_dict = {'READ-WRITE': [b'']}
        with self.reaped_server(SimpleIMAPHandler) as server:
            with self.imap_class(*server.server_address) as imap:
                with mock.patch.object(imap, '_mesg') as mock_mesg:
                    imap._dump_ur(untagged_resp_dict)
                    mock_mesg.assert_called_with("untagged responses dump:READ-WRITE: [b'']")

@unittest.skipUnless(ssl, 'SSL not available')
class ThreadedNetworkedTestsSSL(ThreadedNetworkedTests):
    server_class = SecureTCPServer
    imap_class = IMAP4_SSL

    @threading_helper.reap_threads
    def test_ssl_verified(self):
        if False:
            for i in range(10):
                print('nop')
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.load_verify_locations(CAFILE)
        with self.assertRaisesRegex(ssl.CertificateError, "IP address mismatch, certificate is not valid for '127.0.0.1'"):
            with self.reaped_server(SimpleIMAPHandler) as server:
                client = self.imap_class(*server.server_address, ssl_context=ssl_context)
                client.shutdown()
        with self.reaped_server(SimpleIMAPHandler) as server:
            client = self.imap_class('localhost', server.server_address[1], ssl_context=ssl_context)
            client.shutdown()

@unittest.skipUnless(support.is_resource_enabled('network'), 'network resource disabled')
@unittest.skip('cyrus.andrew.cmu.edu blocks connections')
class RemoteIMAPTest(unittest.TestCase):
    host = 'cyrus.andrew.cmu.edu'
    port = 143
    username = 'anonymous'
    password = 'pass'
    imap_class = imaplib.IMAP4

    def setUp(self):
        if False:
            print('Hello World!')
        with socket_helper.transient_internet(self.host):
            self.server = self.imap_class(self.host, self.port)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        if self.server is not None:
            with socket_helper.transient_internet(self.host):
                self.server.logout()

    def test_logincapa(self):
        if False:
            for i in range(10):
                print('nop')
        with socket_helper.transient_internet(self.host):
            for cap in self.server.capabilities:
                self.assertIsInstance(cap, str)
            self.assertIn('LOGINDISABLED', self.server.capabilities)
            self.assertIn('AUTH=ANONYMOUS', self.server.capabilities)
            rs = self.server.login(self.username, self.password)
            self.assertEqual(rs[0], 'OK')

    def test_logout(self):
        if False:
            print('Hello World!')
        with socket_helper.transient_internet(self.host):
            rs = self.server.logout()
            self.server = None
            self.assertEqual(rs[0], 'BYE', rs)

@unittest.skipUnless(ssl, 'SSL not available')
@unittest.skipUnless(support.is_resource_enabled('network'), 'network resource disabled')
@unittest.skip('cyrus.andrew.cmu.edu blocks connections')
class RemoteIMAP_STARTTLSTest(RemoteIMAPTest):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        with socket_helper.transient_internet(self.host):
            rs = self.server.starttls()
            self.assertEqual(rs[0], 'OK')

    def test_logincapa(self):
        if False:
            return 10
        for cap in self.server.capabilities:
            self.assertIsInstance(cap, str)
        self.assertNotIn('LOGINDISABLED', self.server.capabilities)

@unittest.skipUnless(ssl, 'SSL not available')
@unittest.skip('cyrus.andrew.cmu.edu blocks connections')
class RemoteIMAP_SSLTest(RemoteIMAPTest):
    port = 993
    imap_class = IMAP4_SSL

    def setUp(self):
        if False:
            print('Hello World!')
        pass

    def tearDown(self):
        if False:
            return 10
        pass

    def create_ssl_context(self):
        if False:
            i = 10
            return i + 15
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        ssl_context.load_cert_chain(CERTFILE)
        return ssl_context

    def check_logincapa(self, server):
        if False:
            for i in range(10):
                print('nop')
        try:
            for cap in server.capabilities:
                self.assertIsInstance(cap, str)
            self.assertNotIn('LOGINDISABLED', server.capabilities)
            self.assertIn('AUTH=PLAIN', server.capabilities)
            rs = server.login(self.username, self.password)
            self.assertEqual(rs[0], 'OK')
        finally:
            server.logout()

    def test_logincapa(self):
        if False:
            for i in range(10):
                print('nop')
        with socket_helper.transient_internet(self.host):
            _server = self.imap_class(self.host, self.port)
            self.check_logincapa(_server)

    def test_logout(self):
        if False:
            for i in range(10):
                print('nop')
        with socket_helper.transient_internet(self.host):
            _server = self.imap_class(self.host, self.port)
            rs = _server.logout()
            self.assertEqual(rs[0], 'BYE', rs)

    def test_ssl_context_certfile_exclusive(self):
        if False:
            return 10
        with socket_helper.transient_internet(self.host):
            self.assertRaises(ValueError, self.imap_class, self.host, self.port, certfile=CERTFILE, ssl_context=self.create_ssl_context())

    def test_ssl_context_keyfile_exclusive(self):
        if False:
            while True:
                i = 10
        with socket_helper.transient_internet(self.host):
            self.assertRaises(ValueError, self.imap_class, self.host, self.port, keyfile=CERTFILE, ssl_context=self.create_ssl_context())
if __name__ == '__main__':
    unittest.main()