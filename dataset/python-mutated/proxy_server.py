import base64
import http.server as SimpleHTTPServer
import select
import socket
import socketserver as SocketServer
import threading
from urllib.request import urlopen

def pipe_sockets(sock1, sock2, timeout):
    if False:
        i = 10
        return i + 15
    'Pipe data from one socket to another and vice-versa.'
    sockets = [sock1, sock2]
    try:
        while True:
            (rlist, _, xlist) = select.select(sockets, [], sockets, timeout)
            if xlist:
                break
            for sock in rlist:
                data = sock.recv(8096)
                if not data:
                    break
                other = next((s for s in sockets if s is not sock))
                other.sendall(data)
    except OSError:
        pass
    finally:
        for sock in sockets:
            sock.close()

class Future:

    def __init__(self):
        if False:
            while True:
                i = 10
        self._event = threading.Event()
        self._result = None
        self._exc = None

    def result(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        if not self._event.wait(timeout):
            raise AssertionError('Future timed out')
        if self._exc is not None:
            raise self._exc
        return self._result

    def set_result(self, result):
        if False:
            i = 10
            return i + 15
        self._result = result
        self._event.set()

    def set_exception(self, exception):
        if False:
            for i in range(10):
                print('nop')
        self._exc = exception
        self._event.set()

class ProxyServerThread(threading.Thread):
    spinup_timeout = 10

    def __init__(self, timeout=None):
        if False:
            i = 10
            return i + 15
        self.proxy_host = 'localhost'
        self.proxy_port = None
        self.timeout = timeout
        self.proxy_server = None
        self.socket_created_future = Future()
        self.requests = []
        self.auth = None
        super().__init__()
        self.daemon = True

    def reset(self):
        if False:
            print('Hello World!')
        self.requests.clear()
        self.auth = None

    def __enter__(self):
        if False:
            print('Hello World!')
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        self.stop()
        self.join()

    def set_auth(self, username, password):
        if False:
            return 10
        self.auth = '%s:%s' % (username, password)

    def get_proxy_url(self, with_scheme=True):
        if False:
            while True:
                i = 10
        assert self.socket_created_future.result(self.spinup_timeout)
        if self.auth:
            auth = '%s@' % self.auth
        else:
            auth = ''
        if with_scheme:
            scheme = 'http://'
        else:
            scheme = ''
        return '%s%s%s:%s' % (scheme, auth, self.proxy_host, self.proxy_port)

    def run(self):
        if False:
            while True:
                i = 10
        assert not self.proxy_server, 'This class is not reentrable. Please create a new instance.'
        requests = self.requests
        proxy_thread = self

        class Proxy(SimpleHTTPServer.SimpleHTTPRequestHandler):
            timeout = self.timeout

            def check_auth(self):
                if False:
                    print('Hello World!')
                if proxy_thread.auth is not None:
                    auth_header = self.headers.get('Proxy-Authorization')
                    b64_auth = base64.standard_b64encode(proxy_thread.auth.encode()).decode()
                    expected_auth = 'Basic %s' % b64_auth
                    if auth_header != expected_auth:
                        self.send_response(401)
                        self.send_header('Connection', 'close')
                        self.end_headers()
                        self.wfile.write(('not authenticated. Expected %r, received %r' % (expected_auth, auth_header)).encode())
                        self.connection.close()
                        return False
                return True

            def do_GET(self):
                if False:
                    print('Hello World!')
                if not self.check_auth():
                    return
                requests.append(self.path)
                req = urlopen(self.path, timeout=self.timeout)
                self.send_response(req.getcode())
                content_type = req.info().get('content-type', None)
                if content_type:
                    self.send_header('Content-Type', content_type)
                self.send_header('Connection', 'close')
                self.end_headers()
                self.copyfile(req, self.wfile)
                self.connection.close()
                req.close()

            def do_CONNECT(self):
                if False:
                    i = 10
                    return i + 15
                if not self.check_auth():
                    return
                requests.append(self.path)
                (host, port) = self.path.split(':')
                try:
                    addr = (host, int(port))
                    other_connection = socket.create_connection(addr, timeout=self.timeout)
                except OSError:
                    self.send_error(502, 'Bad gateway')
                    return
                self.send_response(200)
                self.send_header('Connection', 'close')
                self.end_headers()
                pipe_sockets(self.connection, other_connection, self.timeout)
        try:
            self.proxy_server = SocketServer.ThreadingTCPServer((self.proxy_host, 0), Proxy)
            self.proxy_server.daemon_threads = True
            self.proxy_port = self.proxy_server.server_address[1]
        except Exception as e:
            self.socket_created_future.set_exception(e)
            raise
        else:
            self.socket_created_future.set_result(True)
        self.proxy_server.serve_forever()

    def stop(self):
        if False:
            return 10
        self.proxy_server.shutdown()
        self.proxy_server.server_close()

class HttpServerThread(threading.Thread):
    spinup_timeout = 10

    def __init__(self, timeout=None):
        if False:
            i = 10
            return i + 15
        self.server_host = 'localhost'
        self.server_port = None
        self.timeout = timeout
        self.http_server = None
        self.socket_created_future = Future()
        super().__init__()
        self.daemon = True

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        self.stop()
        self.join()

    def get_server_url(self):
        if False:
            while True:
                i = 10
        assert self.socket_created_future.result(self.spinup_timeout)
        return 'http://%s:%s' % (self.server_host, self.server_port)

    def run(self):
        if False:
            i = 10
            return i + 15
        assert not self.http_server, 'This class is not reentrable. Please create a new instance.'

        class Server(SimpleHTTPServer.SimpleHTTPRequestHandler):
            timeout = self.timeout

            def do_GET(self):
                if False:
                    i = 10
                    return i + 15
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.write(b'Hello world')
                elif self.path == '/json':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.write(b'{"hello":"world"}')
                elif self.path == '/json/plain':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/plain;charset=utf-8')
                    self.send_header('Connection', 'close')
                    self.end_headers()
                    self.wfile.write(b'{"hello":"world"}')
                else:
                    self.send_response(404)
                    self.send_header('Connection', 'close')
                    self.send_header('X-test-header', 'hello')
                    self.end_headers()
                    self.wfile.write(b'Not found')
                self.connection.close()
        try:
            self.http_server = SocketServer.ThreadingTCPServer((self.server_host, 0), Server)
            self.http_server.daemon_threads = True
            self.server_port = self.http_server.server_address[1]
        except Exception as e:
            self.socket_created_future.set_exception(e)
            raise
        else:
            self.socket_created_future.set_result(True)
        self.http_server.serve_forever()

    def stop(self):
        if False:
            while True:
                i = 10
        self.http_server.shutdown()
        self.http_server.server_close()