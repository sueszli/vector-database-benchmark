import gevent.testing as greentest
from gevent.testing import support
from gevent.testing import sysinfo
from gevent.socket import socket, error
from gevent.exceptions import LoopExit

class TestSocketErrors(greentest.TestCase):
    __timeout__ = 5

    def test_connection_refused(self):
        if False:
            for i in range(10):
                print('nop')
        port = support.find_unused_port()
        with socket() as s:
            try:
                with self.assertRaises(error) as exc:
                    s.connect((greentest.DEFAULT_CONNECT_HOST, port))
            except LoopExit:
                return
        ex = exc.exception
        self.assertIn(ex.args[0], sysinfo.CONN_REFUSED_ERRORS, ex)
        self.assertIn('refused', str(ex).lower())
if __name__ == '__main__':
    greentest.main()