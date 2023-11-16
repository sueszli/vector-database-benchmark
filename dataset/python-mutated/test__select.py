from gevent.testing import six
import sys
import os
import errno
from gevent import select, socket
import gevent.core
import gevent.testing as greentest
import gevent.testing.timing
import unittest

class TestSelect(gevent.testing.timing.AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            print('Hello World!')
        select.select([], [], [], timeout)

@greentest.skipOnWindows('Cant select on files')
class TestSelectRead(gevent.testing.timing.AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            while True:
                i = 10
        (r, w) = os.pipe()
        try:
            select.select([r], [], [], timeout)
        finally:
            os.close(r)
            os.close(w)

    @unittest.skipIf(sys.platform.startswith('freebsd'), 'skip because of a FreeBSD bug: kern/155606')
    def test_errno(self):
        if False:
            print('Hello World!')
        with open(__file__, 'rb') as fp:
            fd = fp.fileno()
            fp.close()
            try:
                select.select([fd], [], [], 0)
            except OSError as err:
                self.assertEqual(err.errno, errno.EBADF)
            except select.error as err:
                self.assertEqual(err.args[0], errno.EBADF)
            else:
                self.fail('exception not raised')

@unittest.skipUnless(hasattr(select, 'poll'), 'Needs poll')
@greentest.skipOnWindows('Cant poll on files')
class TestPollRead(gevent.testing.timing.AbstractGenericWaitTestCase):

    def wait(self, timeout):
        if False:
            for i in range(10):
                print('nop')
        (r, w) = os.pipe()
        try:
            poll = select.poll()
            poll.register(r, select.POLLIN)
            poll.poll(timeout * 1000)
        finally:
            poll.unregister(r)
            os.close(r)
            os.close(w)

    def test_unregister_never_registered(self):
        if False:
            for i in range(10):
                print('nop')
        poll = select.poll()
        self.assertRaises(KeyError, poll.unregister, 5)

    def test_poll_invalid(self):
        if False:
            i = 10
            return i + 15
        self.skipTest('libev >= 4.27 aborts the process if built with EV_VERIFY >= 2. For libuv, depending on whether the fileno is reused or not this either crashes or does nothing.')
        with open(__file__, 'rb') as fp:
            fd = fp.fileno()
            poll = select.poll()
            poll.register(fd, select.POLLIN)
            fp.close()
            result = poll.poll(0)
            self.assertEqual(result, [(fd, select.POLLNVAL)])

class TestSelectTypes(greentest.TestCase):

    def test_int(self):
        if False:
            while True:
                i = 10
        sock = socket.socket()
        try:
            select.select([int(sock.fileno())], [], [], 0.001)
        finally:
            sock.close()
    if hasattr(six.builtins, 'long'):

        def test_long(self):
            if False:
                for i in range(10):
                    print('nop')
            sock = socket.socket()
            try:
                select.select([six.builtins.long(sock.fileno())], [], [], 0.001)
            finally:
                sock.close()

    def test_iterable(self):
        if False:
            return 10
        sock = socket.socket()

        def fileno_iter():
            if False:
                print('Hello World!')
            yield int(sock.fileno())
        try:
            select.select(fileno_iter(), [], [], 0.001)
        finally:
            sock.close()

    def test_string(self):
        if False:
            while True:
                i = 10
        self.switch_expected = False
        self.assertRaises(TypeError, select.select, ['hello'], [], [], 0.001)
if __name__ == '__main__':
    greentest.main()