from gevent import monkey
monkey.patch_all()
try:
    import httplib
except ImportError:
    from http import client as httplib
import socket
import gevent.testing as greentest

@greentest.skipUnless(hasattr(socket, 'ssl'), 'Needs socket.ssl (Python 2)')
@greentest.skipWithoutExternalNetwork('Tries to access amazon.com')
class AmazonHTTPSTests(greentest.TestCase):
    __timeout__ = 30

    def test_amazon_response(self):
        if False:
            i = 10
            return i + 15
        conn = httplib.HTTPSConnection('sdb.amazonaws.com')
        conn.request('GET', '/')
        conn.getresponse()

    def test_str_and_repr(self):
        if False:
            for i in range(10):
                print('nop')
        conn = socket.socket()
        conn.connect(('sdb.amazonaws.com', 443))
        ssl_conn = socket.ssl(conn)
        assert str(ssl_conn)
        assert repr(ssl_conn)
if __name__ == '__main__':
    greentest.main()