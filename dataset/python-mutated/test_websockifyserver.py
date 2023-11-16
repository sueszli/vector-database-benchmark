""" Unit tests for websockifyserver """
import errno
import os
import logging
import select
import shutil
import socket
import ssl
from unittest.mock import patch, MagicMock, ANY
import sys
import tempfile
import unittest
import socket
import signal
from http.server import BaseHTTPRequestHandler
from io import StringIO
from io import BytesIO
from websockify import websockifyserver

def raise_oserror(*args, **kwargs):
    if False:
        print('Hello World!')
    raise OSError('fake error')

class FakeSocket(object):

    def __init__(self, data=b''):
        if False:
            i = 10
            return i + 15
        self._data = data

    def recv(self, amt, flags=None):
        if False:
            for i in range(10):
                print('nop')
        res = self._data[0:amt]
        if not flags & socket.MSG_PEEK:
            self._data = self._data[amt:]
        return res

    def makefile(self, mode='r', buffsize=None):
        if False:
            while True:
                i = 10
        if 'b' in mode:
            return BytesIO(self._data)
        else:
            return StringIO(self._data.decode('latin_1'))

class WebSockifyRequestHandlerTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        super(WebSockifyRequestHandlerTestCase, self).setUp()
        self.tmpdir = tempfile.mkdtemp('-websockify-tests')
        patch('os.chdir').start()

    def tearDown(self):
        if False:
            return 10
        'Called automatically after each test.'
        patch.stopall()
        os.rmdir(self.tmpdir)
        super(WebSockifyRequestHandlerTestCase, self).tearDown()

    def _get_server(self, handler_class=websockifyserver.WebSockifyRequestHandler, **kwargs):
        if False:
            print('Hello World!')
        web = kwargs.pop('web', self.tmpdir)
        return websockifyserver.WebSockifyServer(handler_class, listen_host='localhost', listen_port=80, key=self.tmpdir, web=web, record=self.tmpdir, daemon=False, ssl_only=0, idle_timeout=1, **kwargs)

    @patch('websockify.websockifyserver.WebSockifyRequestHandler.send_error')
    def test_normal_get_with_only_upgrade_returns_error(self, send_error):
        if False:
            for i in range(10):
                print('nop')
        server = self._get_server(web=None)
        handler = websockifyserver.WebSockifyRequestHandler(FakeSocket(b'GET /tmp.txt HTTP/1.1'), '127.0.0.1', server)
        handler.do_GET()
        send_error.assert_called_with(405)

    @patch('websockify.websockifyserver.WebSockifyRequestHandler.send_error')
    def test_list_dir_with_file_only_returns_error(self, send_error):
        if False:
            i = 10
            return i + 15
        server = self._get_server(file_only=True)
        handler = websockifyserver.WebSockifyRequestHandler(FakeSocket(b'GET / HTTP/1.1'), '127.0.0.1', server)
        handler.path = '/'
        handler.do_GET()
        send_error.assert_called_with(404)

class WebSockifyServerTestCase(unittest.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(WebSockifyServerTestCase, self).setUp()
        self.tmpdir = tempfile.mkdtemp('-websockify-tests')
        patch('os.chdir').start()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        'Called automatically after each test.'
        patch.stopall()
        os.rmdir(self.tmpdir)
        super(WebSockifyServerTestCase, self).tearDown()

    def _get_server(self, handler_class=websockifyserver.WebSockifyRequestHandler, **kwargs):
        if False:
            print('Hello World!')
        return websockifyserver.WebSockifyServer(handler_class, listen_host='localhost', listen_port=80, key=self.tmpdir, web=self.tmpdir, record=self.tmpdir, **kwargs)

    def test_daemonize_raises_error_while_closing_fds(self):
        if False:
            return 10
        server = self._get_server(daemon=True, ssl_only=1, idle_timeout=1)
        patch('os.fork').start().return_value = 0
        patch('signal.signal').start()
        patch('os.setsid').start()
        patch('os.close').start().side_effect = raise_oserror
        self.assertRaises(OSError, server.daemonize, keepfd=None, chdir='./')

    def test_daemonize_ignores_ebadf_error_while_closing_fds(self):
        if False:
            i = 10
            return i + 15

        def raise_oserror_ebadf(fd):
            if False:
                print('Hello World!')
            raise OSError(errno.EBADF, 'fake error')
        server = self._get_server(daemon=True, ssl_only=1, idle_timeout=1)
        patch('os.fork').start().return_value = 0
        patch('signal.signal').start()
        patch('os.setsid').start()
        patch('os.close').start().side_effect = raise_oserror_ebadf
        patch('os.open').start().side_effect = raise_oserror
        self.assertRaises(OSError, server.daemonize, keepfd=None, chdir='./')

    def test_handshake_fails_on_not_ready(self):
        if False:
            return 10
        server = self._get_server(daemon=True, ssl_only=0, idle_timeout=1)

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                return 10
            return ([], [], [])
        patch('select.select').start().side_effect = fake_select
        self.assertRaises(websockifyserver.WebSockifyServer.EClose, server.do_handshake, FakeSocket(), '127.0.0.1')

    def test_empty_handshake_fails(self):
        if False:
            print('Hello World!')
        server = self._get_server(daemon=True, ssl_only=0, idle_timeout=1)
        sock = FakeSocket('')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                while True:
                    i = 10
            return ([sock], [], [])
        patch('select.select').start().side_effect = fake_select
        self.assertRaises(websockifyserver.WebSockifyServer.EClose, server.do_handshake, sock, '127.0.0.1')

    def test_handshake_policy_request(self):
        if False:
            return 10
        pass

    def test_handshake_ssl_only_without_ssl_raises_error(self):
        if False:
            print('Hello World!')
        server = self._get_server(daemon=True, ssl_only=1, idle_timeout=1)
        sock = FakeSocket(b'some initial data')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                print('Hello World!')
            return ([sock], [], [])
        patch('select.select').start().side_effect = fake_select
        self.assertRaises(websockifyserver.WebSockifyServer.EClose, server.do_handshake, sock, '127.0.0.1')

    def test_do_handshake_no_ssl(self):
        if False:
            while True:
                i = 10

        class FakeHandler(object):
            CALLED = False

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                type(self).CALLED = True
        FakeHandler.CALLED = False
        server = self._get_server(handler_class=FakeHandler, daemon=True, ssl_only=0, idle_timeout=1)
        sock = FakeSocket(b'some initial data')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                print('Hello World!')
            return ([sock], [], [])
        patch('select.select').start().side_effect = fake_select
        self.assertEqual(server.do_handshake(sock, '127.0.0.1'), sock)
        self.assertTrue(FakeHandler.CALLED, True)

    def test_do_handshake_ssl(self):
        if False:
            while True:
                i = 10
        pass

    def test_do_handshake_ssl_without_ssl_raises_error(self):
        if False:
            return 10
        pass

    def test_do_handshake_ssl_without_cert_raises_error(self):
        if False:
            i = 10
            return i + 15
        server = self._get_server(daemon=True, ssl_only=0, idle_timeout=1, cert='afdsfasdafdsafdsafdsafdas')
        sock = FakeSocket(b'\x16some ssl data')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                print('Hello World!')
            return ([sock], [], [])
        patch('select.select').start().side_effect = fake_select
        self.assertRaises(websockifyserver.WebSockifyServer.EClose, server.do_handshake, sock, '127.0.0.1')

    def test_do_handshake_ssl_error_eof_raises_close_error(self):
        if False:
            while True:
                i = 10
        server = self._get_server(daemon=True, ssl_only=0, idle_timeout=1)
        sock = FakeSocket(b'\x16some ssl data')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                return 10
            return ([sock], [], [])

        def fake_wrap_socket(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise ssl.SSLError(ssl.SSL_ERROR_EOF)

        class fake_create_default_context:

            def __init__(self, purpose):
                if False:
                    for i in range(10):
                        print('nop')
                self.verify_mode = None
                self.options = 0

            def load_cert_chain(self, certfile, keyfile, password):
                if False:
                    return 10
                pass

            def set_default_verify_paths(self):
                if False:
                    while True:
                        i = 10
                pass

            def load_verify_locations(self, cafile):
                if False:
                    while True:
                        i = 10
                pass

            def wrap_socket(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                raise ssl.SSLError(ssl.SSL_ERROR_EOF)
        patch('select.select').start().side_effect = fake_select
        patch('ssl.create_default_context').start().side_effect = fake_create_default_context
        self.assertRaises(websockifyserver.WebSockifyServer.EClose, server.do_handshake, sock, '127.0.0.1')

    def test_do_handshake_ssl_sets_ciphers(self):
        if False:
            print('Hello World!')
        test_ciphers = 'TEST-CIPHERS-1:TEST-CIPHER-2'

        class FakeHandler(object):

            def __init__(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                pass
        server = self._get_server(handler_class=FakeHandler, daemon=True, idle_timeout=1, ssl_ciphers=test_ciphers)
        sock = FakeSocket(b'\x16some ssl data')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                while True:
                    i = 10
            return ([sock], [], [])

        class fake_create_default_context:
            CIPHERS = ''

            def __init__(self, purpose):
                if False:
                    print('Hello World!')
                self.verify_mode = None
                self.options = 0

            def load_cert_chain(self, certfile, keyfile, password):
                if False:
                    print('Hello World!')
                pass

            def set_default_verify_paths(self):
                if False:
                    return 10
                pass

            def load_verify_locations(self, cafile):
                if False:
                    print('Hello World!')
                pass

            def wrap_socket(self, *args, **kwargs):
                if False:
                    return 10
                pass

            def set_ciphers(self, ciphers_to_set):
                if False:
                    return 10
                fake_create_default_context.CIPHERS = ciphers_to_set
        patch('select.select').start().side_effect = fake_select
        patch('ssl.create_default_context').start().side_effect = fake_create_default_context
        server.do_handshake(sock, '127.0.0.1')
        self.assertEqual(fake_create_default_context.CIPHERS, test_ciphers)

    def test_do_handshake_ssl_sets_opions(self):
        if False:
            i = 10
            return i + 15
        test_options = 3405692655

        class FakeHandler(object):

            def __init__(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                pass
        server = self._get_server(handler_class=FakeHandler, daemon=True, idle_timeout=1, ssl_options=test_options)
        sock = FakeSocket(b'\x16some ssl data')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                for i in range(10):
                    print('nop')
            return ([sock], [], [])

        class fake_create_default_context(object):
            OPTIONS = 0

            def __init__(self, purpose):
                if False:
                    return 10
                self.verify_mode = None
                self._options = 0

            def load_cert_chain(self, certfile, keyfile, password):
                if False:
                    i = 10
                    return i + 15
                pass

            def set_default_verify_paths(self):
                if False:
                    for i in range(10):
                        print('nop')
                pass

            def load_verify_locations(self, cafile):
                if False:
                    i = 10
                    return i + 15
                pass

            def wrap_socket(self, *args, **kwargs):
                if False:
                    print('Hello World!')
                pass

            def get_options(self):
                if False:
                    i = 10
                    return i + 15
                return self._options

            def set_options(self, val):
                if False:
                    return 10
                fake_create_default_context.OPTIONS = val
            options = property(get_options, set_options)
        patch('select.select').start().side_effect = fake_select
        patch('ssl.create_default_context').start().side_effect = fake_create_default_context
        server.do_handshake(sock, '127.0.0.1')
        self.assertEqual(fake_create_default_context.OPTIONS, test_options)

    def test_fallback_sigchld_handler(self):
        if False:
            print('Hello World!')
        pass

    def test_start_server_error(self):
        if False:
            i = 10
            return i + 15
        server = self._get_server(daemon=False, ssl_only=1, idle_timeout=1)
        sock = server.socket('localhost')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                print('Hello World!')
            raise Exception('fake error')
        patch('websockify.websockifyserver.WebSockifyServer.socket').start()
        patch('websockify.websockifyserver.WebSockifyServer.daemonize').start()
        patch('select.select').start().side_effect = fake_select
        server.start_server()

    def test_start_server_keyboardinterrupt(self):
        if False:
            return 10
        server = self._get_server(daemon=False, ssl_only=0, idle_timeout=1)
        sock = server.socket('localhost')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                i = 10
                return i + 15
            raise KeyboardInterrupt
        patch('websockify.websockifyserver.WebSockifyServer.socket').start()
        patch('websockify.websockifyserver.WebSockifyServer.daemonize').start()
        patch('select.select').start().side_effect = fake_select
        server.start_server()

    def test_start_server_systemexit(self):
        if False:
            i = 10
            return i + 15
        server = self._get_server(daemon=False, ssl_only=0, idle_timeout=1)
        sock = server.socket('localhost')

        def fake_select(rlist, wlist, xlist, timeout=None):
            if False:
                return 10
            sys.exit()
        patch('websockify.websockifyserver.WebSockifyServer.socket').start()
        patch('websockify.websockifyserver.WebSockifyServer.daemonize').start()
        patch('select.select').start().side_effect = fake_select
        server.start_server()

    def test_socket_set_keepalive_options(self):
        if False:
            for i in range(10):
                print('nop')
        keepcnt = 12
        keepidle = 34
        keepintvl = 56
        server = self._get_server(daemon=False, ssl_only=0, idle_timeout=1)
        sock = server.socket('localhost', tcp_keepcnt=keepcnt, tcp_keepidle=keepidle, tcp_keepintvl=keepintvl)
        if hasattr(socket, 'TCP_KEEPCNT'):
            self.assertEqual(sock.getsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT), keepcnt)
        self.assertEqual(sock.getsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE), keepidle)
        self.assertEqual(sock.getsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL), keepintvl)
        sock = server.socket('localhost', tcp_keepalive=False, tcp_keepcnt=keepcnt, tcp_keepidle=keepidle, tcp_keepintvl=keepintvl)
        if hasattr(socket, 'TCP_KEEPCNT'):
            self.assertNotEqual(sock.getsockopt(socket.SOL_TCP, socket.TCP_KEEPCNT), keepcnt)
        self.assertNotEqual(sock.getsockopt(socket.SOL_TCP, socket.TCP_KEEPIDLE), keepidle)
        self.assertNotEqual(sock.getsockopt(socket.SOL_TCP, socket.TCP_KEEPINTVL), keepintvl)