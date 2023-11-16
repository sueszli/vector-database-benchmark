import gevent.testing as greentest
from gevent import socket
import errno
import sys

class TestClosedSocket(greentest.TestCase):
    switch_expected = False

    def test(self):
        if False:
            while True:
                i = 10
        sock = socket.socket()
        sock.close()
        try:
            sock.send(b'a', timeout=1)
            self.fail('Should raise socket error')
        except OSError as ex:
            if ex.args[0] != errno.EBADF:
                if sys.platform.startswith('win'):
                    pass
                else:
                    raise

class TestRef(greentest.TestCase):
    switch_expected = False

    def test(self):
        if False:
            for i in range(10):
                print('nop')
        sock = socket.socket()
        self.assertTrue(sock.ref)
        sock.ref = False
        self.assertFalse(sock.ref)
        self.assertFalse(sock._read_event.ref)
        self.assertFalse(sock._write_event.ref)
        sock.close()
if __name__ == '__main__':
    greentest.main()