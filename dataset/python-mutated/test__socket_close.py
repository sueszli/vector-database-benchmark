import gevent
from gevent import socket
from gevent import server
import gevent.testing as greentest

def readall(sock, _):
    if False:
        return 10
    while sock.recv(1024):
        pass
    sock.close()

class Test(greentest.TestCase):
    error_fatal = False

    def setUp(self):
        if False:
            return 10
        self.server = server.StreamServer(greentest.DEFAULT_BIND_ADDR_TUPLE, readall)
        self.server.start()

    def tearDown(self):
        if False:
            print('Hello World!')
        self.server.stop()

    def test_recv_closed(self):
        if False:
            print('Hello World!')
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((greentest.DEFAULT_CONNECT_HOST, self.server.server_port))
        receiver = gevent.spawn(sock.recv, 25)
        try:
            gevent.sleep(0.001)
            sock.close()
            receiver.join(timeout=0.1)
            self.assertTrue(receiver.ready(), receiver)
            self.assertEqual(receiver.value, None)
            self.assertIsInstance(receiver.exception, socket.error)
            self.assertEqual(receiver.exception.errno, socket.EBADF)
        finally:
            receiver.kill()

    @greentest.skipOnLibuvOnCI('Sometimes randomly times out')
    def test_recv_twice(self):
        if False:
            while True:
                i = 10
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((greentest.DEFAULT_CONNECT_HOST, self.server.server_port))
        receiver = gevent.spawn(sock.recv, 25)
        try:
            gevent.sleep(0.001)
            self.assertRaises(AssertionError, sock.recv, 25)
            self.assertRaises(AssertionError, sock.recv, 25)
        finally:
            receiver.kill()
            sock.close()
if __name__ == '__main__':
    greentest.main()