"""
Trivial test that a single process (and single thread) can both read
and write from green sockets (when monkey patched).
"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from gevent import monkey
monkey.patch_all()
import gevent.testing as greentest
try:
    from urllib import request as urllib2
    from http.server import HTTPServer
    from http.server import SimpleHTTPRequestHandler
except ImportError:
    import urllib2
    from BaseHTTPServer import HTTPServer
    from SimpleHTTPServer import SimpleHTTPRequestHandler
import gevent
from gevent.testing import params

class QuietHandler(SimpleHTTPRequestHandler, object):

    def log_message(self, *args):
        if False:
            while True:
                i = 10
        self.server.messages += ((args,),)

class Server(HTTPServer, object):
    messages = ()
    requests_handled = 0

    def __init__(self):
        if False:
            return 10
        HTTPServer.__init__(self, params.DEFAULT_BIND_ADDR_TUPLE, QuietHandler)

    def handle_request(self):
        if False:
            i = 10
            return i + 15
        HTTPServer.handle_request(self)
        self.requests_handled += 1

class TestGreenness(greentest.TestCase):
    check_totalrefcount = False

    def test_urllib2(self):
        if False:
            for i in range(10):
                print('nop')
        httpd = Server()
        server_greenlet = gevent.spawn(httpd.handle_request)
        port = httpd.socket.getsockname()[1]
        rsp = urllib2.urlopen('http://127.0.0.1:%s' % port)
        rsp.read()
        rsp.close()
        server_greenlet.join()
        self.assertEqual(httpd.requests_handled, 1)
        httpd.server_close()
if __name__ == '__main__':
    greentest.main()