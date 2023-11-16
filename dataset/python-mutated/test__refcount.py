"""This test checks that underlying socket instances (gevent.socket.socket._sock)
are not leaked by the hub.
"""
from __future__ import print_function
from _socket import socket as c_socket
import sys
if sys.version_info[0] >= 3:
    __import__('socket')
    Socket = c_socket
else:

    class Socket(c_socket):
        """Something we can have a weakref to"""
import _socket
_socket.socket = Socket
from gevent import monkey
monkey.patch_all()
import gevent.testing as greentest
from gevent.testing import support
from gevent.testing import params
try:
    from thread import start_new_thread
except ImportError:
    from _thread import start_new_thread
from time import sleep
import weakref
import gc
import socket
socket._realsocket = Socket
SOCKET_TIMEOUT = 0.1
if greentest.RESOLVER_DNSPYTHON:
    SOCKET_TIMEOUT *= 2
if greentest.RUNNING_ON_CI:
    SOCKET_TIMEOUT *= 2

class Server(object):
    listening = False
    client_data = None
    server_port = None

    def __init__(self, raise_on_timeout):
        if False:
            print('Hello World!')
        self.raise_on_timeout = raise_on_timeout
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            self.server_port = support.bind_port(self.socket, params.DEFAULT_BIND_ADDR)
        except:
            self.close()
            raise

    def close(self):
        if False:
            for i in range(10):
                print('nop')
        self.socket.close()
        self.socket = None

    def handle_request(self):
        if False:
            print('Hello World!')
        try:
            self.socket.settimeout(SOCKET_TIMEOUT)
            self.socket.listen(5)
            self.listening = True
            try:
                (conn, _) = self.socket.accept()
            except socket.timeout:
                if self.raise_on_timeout:
                    raise
                return
            try:
                self.client_data = conn.recv(100)
                conn.send(b'bye')
            finally:
                conn.close()
        finally:
            self.close()

class Client(object):
    server_data = None

    def __init__(self, server_port):
        if False:
            return 10
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_port = server_port

    def close(self):
        if False:
            return 10
        self.socket.close()
        self.socket = None

    def make_request(self):
        if False:
            i = 10
            return i + 15
        try:
            self.socket.connect((params.DEFAULT_CONNECT, self.server_port))
            self.socket.send(b'hello')
            self.server_data = self.socket.recv(100)
        finally:
            self.close()

class Test(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT

    def run_interaction(self, run_client):
        if False:
            i = 10
            return i + 15
        server = Server(raise_on_timeout=run_client)
        wref_to_hidden_server_socket = weakref.ref(server.socket._sock)
        client = None
        start_new_thread(server.handle_request)
        if run_client:
            client = Client(server.server_port)
            start_new_thread(client.make_request)
        for obj in (server, client):
            if obj is None:
                continue
            while obj.socket is not None:
                sleep(0.01)
        if run_client:
            self.assertEqual(server.client_data, b'hello')
            self.assertEqual(client.server_data, b'bye')
        return wref_to_hidden_server_socket

    def run_and_check(self, run_client):
        if False:
            print('Hello World!')
        wref_to_hidden_server_socket = self.run_interaction(run_client=run_client)
        greentest.gc_collect_if_needed()
        if wref_to_hidden_server_socket():
            from pprint import pformat
            print(pformat(gc.get_referrers(wref_to_hidden_server_socket())))
            for x in gc.get_referrers(wref_to_hidden_server_socket()):
                print(pformat(x))
                for y in gc.get_referrers(x):
                    print('-', pformat(y))
            self.fail('server socket should be dead by now')

    def test_clean_exit(self):
        if False:
            i = 10
            return i + 15
        self.run_and_check(True)
        self.run_and_check(True)

    def test_timeout_exit(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_and_check(False)
        self.run_and_check(False)
if __name__ == '__main__':
    greentest.main()