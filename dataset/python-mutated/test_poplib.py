"""Test script for poplib module."""
import poplib
import socket
import os
import errno
import threading
import unittest
from unittest import TestCase, skipUnless
from test import support as test_support
from test.support import hashlib_helper
from test.support import socket_helper
from test.support import threading_helper
import warnings
with warnings.catch_warnings():
    warnings.simplefilter('ignore', DeprecationWarning)
    import asynchat
    import asyncore
HOST = socket_helper.HOST
PORT = 0
SUPPORTS_SSL = False
if hasattr(poplib, 'POP3_SSL'):
    import ssl
    SUPPORTS_SSL = True
    CERTFILE = os.path.join(os.path.dirname(__file__) or os.curdir, 'keycert3.pem')
    CAFILE = os.path.join(os.path.dirname(__file__) or os.curdir, 'pycacert.pem')
requires_ssl = skipUnless(SUPPORTS_SSL, 'SSL not supported')
LIST_RESP = b'1 1\r\n2 2\r\n3 3\r\n4 4\r\n5 5\r\n.\r\n'
RETR_RESP = b'From: postmaster@python.org\r\nContent-Type: text/plain\r\nMIME-Version: 1.0\r\nSubject: Dummy\r\n\r\nline1\r\nline2\r\nline3\r\n.\r\n'

class DummyPOP3Handler(asynchat.async_chat):
    CAPAS = {'UIDL': [], 'IMPLEMENTATION': ['python-testlib-pop-server']}
    enable_UTF8 = False

    def __init__(self, conn):
        if False:
            for i in range(10):
                print('nop')
        asynchat.async_chat.__init__(self, conn)
        self.set_terminator(b'\r\n')
        self.in_buffer = []
        self.push('+OK dummy pop3 server ready. <timestamp>')
        self.tls_active = False
        self.tls_starting = False

    def collect_incoming_data(self, data):
        if False:
            print('Hello World!')
        self.in_buffer.append(data)

    def found_terminator(self):
        if False:
            while True:
                i = 10
        line = b''.join(self.in_buffer)
        line = str(line, 'ISO-8859-1')
        self.in_buffer = []
        cmd = line.split(' ')[0].lower()
        space = line.find(' ')
        if space != -1:
            arg = line[space + 1:]
        else:
            arg = ''
        if hasattr(self, 'cmd_' + cmd):
            method = getattr(self, 'cmd_' + cmd)
            method(arg)
        else:
            self.push('-ERR unrecognized POP3 command "%s".' % cmd)

    def handle_error(self):
        if False:
            i = 10
            return i + 15
        raise

    def push(self, data):
        if False:
            i = 10
            return i + 15
        asynchat.async_chat.push(self, data.encode('ISO-8859-1') + b'\r\n')

    def cmd_echo(self, arg):
        if False:
            i = 10
            return i + 15
        self.push(arg)

    def cmd_user(self, arg):
        if False:
            for i in range(10):
                print('nop')
        if arg != 'guido':
            self.push('-ERR no such user')
        self.push('+OK password required')

    def cmd_pass(self, arg):
        if False:
            i = 10
            return i + 15
        if arg != 'python':
            self.push('-ERR wrong password')
        self.push('+OK 10 messages')

    def cmd_stat(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.push('+OK 10 100')

    def cmd_list(self, arg):
        if False:
            while True:
                i = 10
        if arg:
            self.push('+OK %s %s' % (arg, arg))
        else:
            self.push('+OK')
            asynchat.async_chat.push(self, LIST_RESP)
    cmd_uidl = cmd_list

    def cmd_retr(self, arg):
        if False:
            i = 10
            return i + 15
        self.push('+OK %s bytes' % len(RETR_RESP))
        asynchat.async_chat.push(self, RETR_RESP)
    cmd_top = cmd_retr

    def cmd_dele(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.push('+OK message marked for deletion.')

    def cmd_noop(self, arg):
        if False:
            while True:
                i = 10
        self.push('+OK done nothing.')

    def cmd_rpop(self, arg):
        if False:
            while True:
                i = 10
        self.push('+OK done nothing.')

    def cmd_apop(self, arg):
        if False:
            for i in range(10):
                print('nop')
        self.push('+OK done nothing.')

    def cmd_quit(self, arg):
        if False:
            return 10
        self.push('+OK closing.')
        self.close_when_done()

    def _get_capas(self):
        if False:
            print('Hello World!')
        _capas = dict(self.CAPAS)
        if not self.tls_active and SUPPORTS_SSL:
            _capas['STLS'] = []
        return _capas

    def cmd_capa(self, arg):
        if False:
            return 10
        self.push('+OK Capability list follows')
        if self._get_capas():
            for (cap, params) in self._get_capas().items():
                _ln = [cap]
                if params:
                    _ln.extend(params)
                self.push(' '.join(_ln))
        self.push('.')

    def cmd_utf8(self, arg):
        if False:
            i = 10
            return i + 15
        self.push('+OK I know RFC6856' if self.enable_UTF8 else '-ERR What is UTF8?!')
    if SUPPORTS_SSL:

        def cmd_stls(self, arg):
            if False:
                i = 10
                return i + 15
            if self.tls_active is False:
                self.push('+OK Begin TLS negotiation')
                context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
                context.load_cert_chain(CERTFILE)
                tls_sock = context.wrap_socket(self.socket, server_side=True, do_handshake_on_connect=False, suppress_ragged_eofs=False)
                self.del_channel()
                self.set_socket(tls_sock)
                self.tls_active = True
                self.tls_starting = True
                self.in_buffer = []
                self._do_tls_handshake()
            else:
                self.push('-ERR Command not permitted when TLS active')

        def _do_tls_handshake(self):
            if False:
                i = 10
                return i + 15
            try:
                self.socket.do_handshake()
            except ssl.SSLError as err:
                if err.args[0] in (ssl.SSL_ERROR_WANT_READ, ssl.SSL_ERROR_WANT_WRITE):
                    return
                elif err.args[0] == ssl.SSL_ERROR_EOF:
                    return self.handle_close()
                elif 'SSLV3_ALERT_BAD_CERTIFICATE' in err.args[1] or 'SSLV3_ALERT_CERTIFICATE_UNKNOWN' in err.args[1]:
                    return self.handle_close()
                raise
            except OSError as err:
                if err.args[0] == errno.ECONNABORTED:
                    return self.handle_close()
            else:
                self.tls_active = True
                self.tls_starting = False

        def handle_read(self):
            if False:
                return 10
            if self.tls_starting:
                self._do_tls_handshake()
            else:
                try:
                    asynchat.async_chat.handle_read(self)
                except ssl.SSLEOFError:
                    self.handle_close()

class DummyPOP3Server(asyncore.dispatcher, threading.Thread):
    handler = DummyPOP3Handler

    def __init__(self, address, af=socket.AF_INET):
        if False:
            while True:
                i = 10
        threading.Thread.__init__(self)
        asyncore.dispatcher.__init__(self)
        self.daemon = True
        self.create_socket(af, socket.SOCK_STREAM)
        self.bind(address)
        self.listen(5)
        self.active = False
        self.active_lock = threading.Lock()
        (self.host, self.port) = self.socket.getsockname()[:2]
        self.handler_instance = None

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        assert not self.active
        self.__flag = threading.Event()
        threading.Thread.start(self)
        self.__flag.wait()

    def run(self):
        if False:
            print('Hello World!')
        self.active = True
        self.__flag.set()
        try:
            while self.active and asyncore.socket_map:
                with self.active_lock:
                    asyncore.loop(timeout=0.1, count=1)
        finally:
            asyncore.close_all(ignore_all=True)

    def stop(self):
        if False:
            return 10
        assert self.active
        self.active = False
        self.join()

    def handle_accepted(self, conn, addr):
        if False:
            i = 10
            return i + 15
        self.handler_instance = self.handler(conn)

    def handle_connect(self):
        if False:
            return 10
        self.close()
    handle_read = handle_connect

    def writable(self):
        if False:
            return 10
        return 0

    def handle_error(self):
        if False:
            while True:
                i = 10
        raise

class TestPOP3Class(TestCase):

    def assertOK(self, resp):
        if False:
            i = 10
            return i + 15
        self.assertTrue(resp.startswith(b'+OK'))

    def setUp(self):
        if False:
            return 10
        self.server = DummyPOP3Server((HOST, PORT))
        self.server.start()
        self.client = poplib.POP3(self.server.host, self.server.port, timeout=test_support.LOOPBACK_TIMEOUT)

    def tearDown(self):
        if False:
            while True:
                i = 10
        self.client.close()
        self.server.stop()
        self.server = None

    def test_getwelcome(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.client.getwelcome(), b'+OK dummy pop3 server ready. <timestamp>')

    def test_exceptions(self):
        if False:
            return 10
        self.assertRaises(poplib.error_proto, self.client._shortcmd, 'echo -err')

    def test_user(self):
        if False:
            print('Hello World!')
        self.assertOK(self.client.user('guido'))
        self.assertRaises(poplib.error_proto, self.client.user, 'invalid')

    def test_pass_(self):
        if False:
            i = 10
            return i + 15
        self.assertOK(self.client.pass_('python'))
        self.assertRaises(poplib.error_proto, self.client.user, 'invalid')

    def test_stat(self):
        if False:
            return 10
        self.assertEqual(self.client.stat(), (10, 100))

    def test_list(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(self.client.list()[1:], ([b'1 1', b'2 2', b'3 3', b'4 4', b'5 5'], 25))
        self.assertTrue(self.client.list('1').endswith(b'OK 1 1'))

    def test_retr(self):
        if False:
            return 10
        expected = (b'+OK 116 bytes', [b'From: postmaster@python.org', b'Content-Type: text/plain', b'MIME-Version: 1.0', b'Subject: Dummy', b'', b'line1', b'line2', b'line3'], 113)
        foo = self.client.retr('foo')
        self.assertEqual(foo, expected)

    def test_too_long_lines(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(poplib.error_proto, self.client._shortcmd, 'echo +%s' % ((poplib._MAXLINE + 10) * 'a'))

    def test_dele(self):
        if False:
            while True:
                i = 10
        self.assertOK(self.client.dele('foo'))

    def test_noop(self):
        if False:
            print('Hello World!')
        self.assertOK(self.client.noop())

    def test_rpop(self):
        if False:
            print('Hello World!')
        self.assertOK(self.client.rpop('foo'))

    @hashlib_helper.requires_hashdigest('md5', openssl=True)
    def test_apop_normal(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertOK(self.client.apop('foo', 'dummypassword'))

    @hashlib_helper.requires_hashdigest('md5', openssl=True)
    def test_apop_REDOS(self):
        if False:
            i = 10
            return i + 15
        evil_welcome = b'+OK' + b'<' * 1000000
        with test_support.swap_attr(self.client, 'welcome', evil_welcome):
            self.assertRaises(poplib.error_proto, self.client.apop, 'a', 'kb')

    def test_top(self):
        if False:
            i = 10
            return i + 15
        expected = (b'+OK 116 bytes', [b'From: postmaster@python.org', b'Content-Type: text/plain', b'MIME-Version: 1.0', b'Subject: Dummy', b'', b'line1', b'line2', b'line3'], 113)
        self.assertEqual(self.client.top(1, 1), expected)

    def test_uidl(self):
        if False:
            return 10
        self.client.uidl()
        self.client.uidl('foo')

    def test_utf8_raises_if_unsupported(self):
        if False:
            i = 10
            return i + 15
        self.server.handler.enable_UTF8 = False
        self.assertRaises(poplib.error_proto, self.client.utf8)

    def test_utf8(self):
        if False:
            print('Hello World!')
        self.server.handler.enable_UTF8 = True
        expected = b'+OK I know RFC6856'
        result = self.client.utf8()
        self.assertEqual(result, expected)

    def test_capa(self):
        if False:
            i = 10
            return i + 15
        capa = self.client.capa()
        self.assertTrue('IMPLEMENTATION' in capa.keys())

    def test_quit(self):
        if False:
            return 10
        resp = self.client.quit()
        self.assertTrue(resp)
        self.assertIsNone(self.client.sock)
        self.assertIsNone(self.client.file)

    @requires_ssl
    def test_stls_capa(self):
        if False:
            for i in range(10):
                print('nop')
        capa = self.client.capa()
        self.assertTrue('STLS' in capa.keys())

    @requires_ssl
    def test_stls(self):
        if False:
            for i in range(10):
                print('nop')
        expected = b'+OK Begin TLS negotiation'
        resp = self.client.stls()
        self.assertEqual(resp, expected)

    @requires_ssl
    def test_stls_context(self):
        if False:
            for i in range(10):
                print('nop')
        expected = b'+OK Begin TLS negotiation'
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.load_verify_locations(CAFILE)
        self.assertEqual(ctx.verify_mode, ssl.CERT_REQUIRED)
        self.assertEqual(ctx.check_hostname, True)
        with self.assertRaises(ssl.CertificateError):
            resp = self.client.stls(context=ctx)
        self.client = poplib.POP3('localhost', self.server.port, timeout=test_support.LOOPBACK_TIMEOUT)
        resp = self.client.stls(context=ctx)
        self.assertEqual(resp, expected)
if SUPPORTS_SSL:
    from test.test_ftplib import SSLConnection

    class DummyPOP3_SSLHandler(SSLConnection, DummyPOP3Handler):

        def __init__(self, conn):
            if False:
                return 10
            asynchat.async_chat.__init__(self, conn)
            self.secure_connection()
            self.set_terminator(b'\r\n')
            self.in_buffer = []
            self.push('+OK dummy pop3 server ready. <timestamp>')
            self.tls_active = True
            self.tls_starting = False

@requires_ssl
class TestPOP3_SSLClass(TestPOP3Class):

    def setUp(self):
        if False:
            print('Hello World!')
        self.server = DummyPOP3Server((HOST, PORT))
        self.server.handler = DummyPOP3_SSLHandler
        self.server.start()
        self.client = poplib.POP3_SSL(self.server.host, self.server.port)

    def test__all__(self):
        if False:
            while True:
                i = 10
        self.assertIn('POP3_SSL', poplib.__all__)

    def test_context(self):
        if False:
            print('Hello World!')
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
        self.assertRaises(ValueError, poplib.POP3_SSL, self.server.host, self.server.port, keyfile=CERTFILE, context=ctx)
        self.assertRaises(ValueError, poplib.POP3_SSL, self.server.host, self.server.port, certfile=CERTFILE, context=ctx)
        self.assertRaises(ValueError, poplib.POP3_SSL, self.server.host, self.server.port, keyfile=CERTFILE, certfile=CERTFILE, context=ctx)
        self.client.quit()
        self.client = poplib.POP3_SSL(self.server.host, self.server.port, context=ctx)
        self.assertIsInstance(self.client.sock, ssl.SSLSocket)
        self.assertIs(self.client.sock.context, ctx)
        self.assertTrue(self.client.noop().startswith(b'+OK'))

    def test_stls(self):
        if False:
            return 10
        self.assertRaises(poplib.error_proto, self.client.stls)
    test_stls_context = test_stls

    def test_stls_capa(self):
        if False:
            return 10
        capa = self.client.capa()
        self.assertFalse('STLS' in capa.keys())

@requires_ssl
class TestPOP3_TLSClass(TestPOP3Class):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.server = DummyPOP3Server((HOST, PORT))
        self.server.start()
        self.client = poplib.POP3(self.server.host, self.server.port, timeout=test_support.LOOPBACK_TIMEOUT)
        self.client.stls()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if self.client.file is not None and self.client.sock is not None:
            try:
                self.client.quit()
            except poplib.error_proto:
                self.client.close()
        self.server.stop()
        self.server = None

    def test_stls(self):
        if False:
            print('Hello World!')
        self.assertRaises(poplib.error_proto, self.client.stls)
    test_stls_context = test_stls

    def test_stls_capa(self):
        if False:
            i = 10
            return i + 15
        capa = self.client.capa()
        self.assertFalse(b'STLS' in capa.keys())

class TestTimeouts(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self.evt = threading.Event()
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.settimeout(60)
        self.port = socket_helper.bind_port(self.sock)
        self.thread = threading.Thread(target=self.server, args=(self.evt, self.sock))
        self.thread.daemon = True
        self.thread.start()
        self.evt.wait()

    def tearDown(self):
        if False:
            return 10
        self.thread.join()
        self.thread = None

    def server(self, evt, serv):
        if False:
            while True:
                i = 10
        serv.listen()
        evt.set()
        try:
            (conn, addr) = serv.accept()
            conn.send(b'+ Hola mundo\n')
            conn.close()
        except TimeoutError:
            pass
        finally:
            serv.close()

    def testTimeoutDefault(self):
        if False:
            while True:
                i = 10
        self.assertIsNone(socket.getdefaulttimeout())
        socket.setdefaulttimeout(test_support.LOOPBACK_TIMEOUT)
        try:
            pop = poplib.POP3(HOST, self.port)
        finally:
            socket.setdefaulttimeout(None)
        self.assertEqual(pop.sock.gettimeout(), test_support.LOOPBACK_TIMEOUT)
        pop.close()

    def testTimeoutNone(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsNone(socket.getdefaulttimeout())
        socket.setdefaulttimeout(30)
        try:
            pop = poplib.POP3(HOST, self.port, timeout=None)
        finally:
            socket.setdefaulttimeout(None)
        self.assertIsNone(pop.sock.gettimeout())
        pop.close()

    def testTimeoutValue(self):
        if False:
            while True:
                i = 10
        pop = poplib.POP3(HOST, self.port, timeout=test_support.LOOPBACK_TIMEOUT)
        self.assertEqual(pop.sock.gettimeout(), test_support.LOOPBACK_TIMEOUT)
        pop.close()
        with self.assertRaises(ValueError):
            poplib.POP3(HOST, self.port, timeout=0)

def setUpModule():
    if False:
        for i in range(10):
            print('nop')
    thread_info = threading_helper.threading_setup()
    unittest.addModuleCleanup(threading_helper.threading_cleanup, *thread_info)
if __name__ == '__main__':
    unittest.main()