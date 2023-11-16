import socket
import ssl
import threading
from contextlib import contextmanager
BUFSIZE = 4096
server_ctx = ssl.create_default_context(ssl.Purpose.CLIENT_AUTH)
server_ctx.load_cert_chain('trio-test-1.pem')

def _ssl_echo_serve_sync(sock):
    if False:
        while True:
            i = 10
    try:
        wrapped = server_ctx.wrap_socket(sock, server_side=True)
        while True:
            data = wrapped.recv(BUFSIZE)
            if not data:
                wrapped.unwrap()
                return
            wrapped.sendall(data)
    except BrokenPipeError:
        pass

@contextmanager
def echo_server_connection():
    if False:
        while True:
            i = 10
    (client_sock, server_sock) = socket.socketpair()
    with client_sock, server_sock:
        t = threading.Thread(target=_ssl_echo_serve_sync, args=(server_sock,), daemon=True)
        t.start()
        yield client_sock

class ManuallyWrappedSocket:

    def __init__(self, ctx, sock, **kwargs):
        if False:
            while True:
                i = 10
        self.incoming = ssl.MemoryBIO()
        self.outgoing = ssl.MemoryBIO()
        self.obj = ctx.wrap_bio(self.incoming, self.outgoing, **kwargs)
        self.sock = sock

    def _retry(self, fn, *args):
        if False:
            print('Hello World!')
        finished = False
        while not finished:
            want_read = False
            try:
                ret = fn(*args)
            except ssl.SSLWantReadError:
                want_read = True
            except ssl.SSLWantWriteError:
                pass
            else:
                finished = True
            data = self.outgoing.read()
            if data:
                self.sock.sendall(data)
            if want_read:
                data = self.sock.recv(BUFSIZE)
                if not data:
                    self.incoming.write_eof()
                else:
                    self.incoming.write(data)
        return ret

    def do_handshake(self):
        if False:
            i = 10
            return i + 15
        self._retry(self.obj.do_handshake)

    def recv(self, bufsize):
        if False:
            for i in range(10):
                print('nop')
        return self._retry(self.obj.read, bufsize)

    def sendall(self, data):
        if False:
            return 10
        self._retry(self.obj.write, data)

    def unwrap(self):
        if False:
            print('Hello World!')
        self._retry(self.obj.unwrap)
        return self.sock

def wrap_socket_via_wrap_socket(ctx, sock, **kwargs):
    if False:
        i = 10
        return i + 15
    return ctx.wrap_socket(sock, do_handshake_on_connect=False, **kwargs)

def wrap_socket_via_wrap_bio(ctx, sock, **kwargs):
    if False:
        return 10
    return ManuallyWrappedSocket(ctx, sock, **kwargs)
for wrap_socket in [wrap_socket_via_wrap_socket, wrap_socket_via_wrap_bio]:
    print(f'\n--- checking {wrap_socket.__name__} ---\n')
    print('checking with do_handshake + correct hostname...')
    with echo_server_connection() as client_sock:
        client_ctx = ssl.create_default_context(cafile='trio-test-CA.pem')
        wrapped = wrap_socket(client_ctx, client_sock, server_hostname='trio-test-1.example.org')
        wrapped.do_handshake()
        wrapped.sendall(b'x')
        assert wrapped.recv(1) == b'x'
        wrapped.unwrap()
    print('...success')
    print('checking with do_handshake + wrong hostname...')
    with echo_server_connection() as client_sock:
        client_ctx = ssl.create_default_context(cafile='trio-test-CA.pem')
        wrapped = wrap_socket(client_ctx, client_sock, server_hostname='trio-test-2.example.org')
        try:
            wrapped.do_handshake()
        except Exception:
            print('...got error as expected')
        else:
            print('??? no error ???')
    print('checking withOUT do_handshake + wrong hostname...')
    with echo_server_connection() as client_sock:
        client_ctx = ssl.create_default_context(cafile='trio-test-CA.pem')
        wrapped = wrap_socket(client_ctx, client_sock, server_hostname='trio-test-2.example.org')
        sent = b'x'
        print('sending', sent)
        wrapped.sendall(sent)
        got = wrapped.recv(1)
        print('got:', got)
        assert got == sent
        print('!!!! successful chat with invalid host! we have been haxored!')