from __future__ import print_function, division, absolute_import
from gevent import monkey
monkey.patch_all()
import os
import socket
import gevent.testing as greentest
from gevent.tests import test__socket
import ssl

def ssl_listener(private_key, certificate):
    if False:
        for i in range(10):
            print('nop')
    raw_listener = socket.socket()
    greentest.bind_and_listen(raw_listener)
    sock = wrap_socket(raw_listener, keyfile=private_key, certfile=certificate, server_side=True)
    return (sock, raw_listener)

def wrap_socket(sock, *, keyfile=None, certfile=None, server_side=False):
    if False:
        print('Hello World!')
    context = ssl.SSLContext(protocol=ssl.PROTOCOL_TLS)
    context.verify_mode = ssl.CERT_NONE
    context.check_hostname = False
    context.load_default_certs()
    if keyfile is not None or certfile is not None:
        context.load_cert_chain(certfile=certfile, keyfile=keyfile)
    return context.wrap_socket(sock, server_side=server_side)

class TestSSL(test__socket.TestTCP):
    certfile = os.path.join(os.path.dirname(__file__), 'test_server.crt')
    privfile = os.path.join(os.path.dirname(__file__), 'test_server.key')
    TIMEOUT_ERROR = socket.timeout

    def _setup_listener(self):
        if False:
            while True:
                i = 10
        (listener, raw_listener) = ssl_listener(self.privfile, self.certfile)
        self._close_on_teardown(raw_listener)
        return listener

    def create_connection(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return self._close_on_teardown(wrap_socket(super(TestSSL, self).create_connection(*args, **kwargs)))
    _test_sendall_timeout_check_time = False
    _test_sendall_data = data_sent = b'hello' * 100000000
    test_sendall_array = greentest.skipOnMacOnCI('Sometimes misses data')(greentest.skipOnManylinux('Sometimes misses data')(test__socket.TestTCP.test_sendall_array))
    test_sendall_str = greentest.skipOnMacOnCI('Sometimes misses data')(greentest.skipOnManylinux('Sometimes misses data')(test__socket.TestTCP.test_sendall_str))

    @greentest.skipOnWindows("Not clear why we're skipping")
    def test_ssl_sendall_timeout0(self):
        if False:
            print('Hello World!')
        server_sock = []
        acceptor = test__socket.Thread(target=lambda : server_sock.append(self.listener.accept()))
        client = self.create_connection()
        client.setblocking(False)
        try:
            expected = getattr(ssl, 'SSLWantWriteError', ssl.SSLError)
            with self.assertRaises(expected):
                client.sendall(self._test_sendall_data)
        finally:
            acceptor.join()
            client.close()
            server_sock[0][0].close()

    @greentest.ignores_leakcheck
    @greentest.skipOnPy310('No longer raises SSLError')
    def test_empty_send(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ssl.SSLError):
            super(TestSSL, self).test_empty_send()

    @greentest.ignores_leakcheck
    def test_sendall_nonblocking(self):
        if False:
            return 10
        pass

    @greentest.ignores_leakcheck
    def test_connect_with_type_flags_ignored(self):
        if False:
            return 10
        pass
if __name__ == '__main__':
    greentest.main()