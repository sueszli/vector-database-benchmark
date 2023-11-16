import sys
try:
    from urllib import request as urllib2
except ImportError:
    import urllib2
import socket
import ssl
import gevent.testing as greentest
from gevent.testing import DEFAULT_XPC_SOCKET_TIMEOUT
from gevent.testing import util
from gevent.testing import params

@greentest.skipOnCI('Timing issues sometimes lead to a connection refused')
class Test_wsgiserver(util.TestServer):
    example = 'wsgiserver.py'
    URL = 'http://%s:8088' % (params.DEFAULT_LOCAL_HOST_ADDR,)
    PORT = 8088
    not_found_message = b'<h1>Not Found</h1>'
    ssl_ctx = None
    _use_ssl = False

    def read(self, path='/'):
        if False:
            return 10
        url = self.URL + path
        try:
            kwargs = {}
            if self.ssl_ctx is not None:
                kwargs = {'context': self.ssl_ctx}
            response = urllib2.urlopen(url, None, DEFAULT_XPC_SOCKET_TIMEOUT, **kwargs)
        except urllib2.HTTPError:
            response = sys.exc_info()[1]
        result = ('%s %s' % (response.code, response.msg), response.read())
        response.close()
        return result

    def _test_hello(self):
        if False:
            for i in range(10):
                print('nop')
        (status, data) = self.read('/')
        self.assertEqual(status, '200 OK')
        self.assertEqual(data, b'<b>hello world</b>')

    def _test_not_found(self):
        if False:
            while True:
                i = 10
        (status, data) = self.read('/xxx')
        self.assertEqual(status, '404 Not Found')
        self.assertEqual(data, self.not_found_message)

    def _do_test_a_blocking_client(self):
        if False:
            while True:
                i = 10
        with self.running_server():
            self._test_hello()
            sock = socket.create_connection((params.DEFAULT_LOCAL_HOST_ADDR, self.PORT))
            ssl_sock = None
            if self._use_ssl:
                context = ssl.SSLContext()
                ssl_sock = context.wrap_socket(sock)
                sock_file = ssl_sock.makefile(mode='rwb')
            else:
                sock_file = sock.makefile(mode='rwb')
            sock_file.write(b'GET /xxx HTTP/1.0\r\n')
            sock_file.flush()
            self._test_hello()
            sock_file.write(b'\r\n')
            sock_file.flush()
            line = sock_file.readline()
            self.assertEqual(line, b'HTTP/1.1 404 Not Found\r\n')
            sock_file.close()
            if ssl_sock is not None:
                ssl_sock.close()
            sock.close()

    def test_a_blocking_client(self):
        if False:
            while True:
                i = 10
        self._do_test_a_blocking_client()
if __name__ == '__main__':
    greentest.main()