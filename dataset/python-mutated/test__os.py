from __future__ import print_function, absolute_import, division
import sys
from os import pipe
import gevent
from gevent import os
from gevent import Greenlet, joinall
from gevent import testing as greentest
from gevent.testing import mock
from gevent.testing import six
from gevent.testing.skipping import skipOnLibuvOnPyPyOnWin

class TestOS_tp(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT

    def pipe(self):
        if False:
            while True:
                i = 10
        return pipe()
    read = staticmethod(os.tp_read)
    write = staticmethod(os.tp_write)

    @skipOnLibuvOnPyPyOnWin('Sometimes times out')
    def _test_if_pipe_blocks(self, buffer_class):
        if False:
            while True:
                i = 10
        (r, w) = self.pipe()
        nbytes = 1000000
        block = b'x' * 4096
        buf = buffer_class(block)
        bytesread = [0]
        byteswritten = [0]

        def produce():
            if False:
                while True:
                    i = 10
            while byteswritten[0] != nbytes:
                bytesleft = nbytes - byteswritten[0]
                byteswritten[0] += self.write(w, buf[:min(bytesleft, 4096)])

        def consume():
            if False:
                while True:
                    i = 10
            while bytesread[0] != nbytes:
                bytesleft = nbytes - bytesread[0]
                bytesread[0] += len(self.read(r, min(bytesleft, 4096)))
        producer = Greenlet(produce)
        producer.start()
        consumer = Greenlet(consume)
        consumer.start_later(1)
        joinall([producer, consumer])
        self.assertEqual(bytesread[0], nbytes)
        self.assertEqual(bytesread[0], byteswritten[0])
    if sys.version_info[0] < 3:

        def test_if_pipe_blocks_buffer(self):
            if False:
                for i in range(10):
                    print('nop')
            self._test_if_pipe_blocks(six.builtins.buffer)
    if sys.version_info[:2] >= (2, 7):

        def test_if_pipe_blocks_memoryview(self):
            if False:
                while True:
                    i = 10
            self._test_if_pipe_blocks(six.builtins.memoryview)

@greentest.skipUnless(hasattr(os, 'make_nonblocking'), 'Only on POSIX')
class TestOS_nb(TestOS_tp):

    def read(self, fd, count):
        if False:
            return 10
        return os.nb_read(fd, count)

    def write(self, fd, count):
        if False:
            print('Hello World!')
        return os.nb_write(fd, count)

    def pipe(self):
        if False:
            print('Hello World!')
        (r, w) = super(TestOS_nb, self).pipe()
        os.make_nonblocking(r)
        os.make_nonblocking(w)
        return (r, w)

    def _make_ignored_oserror(self):
        if False:
            i = 10
            return i + 15
        import errno
        ignored_oserror = OSError()
        ignored_oserror.errno = errno.EINTR
        return ignored_oserror

    def _check_hub_event_closed(self, mock_get_hub, fd, event):
        if False:
            i = 10
            return i + 15
        mock_get_hub.assert_called_once_with()
        hub = mock_get_hub.return_value
        io = hub.loop.io
        io.assert_called_once_with(fd, event)
        event = io.return_value
        event.close.assert_called_once_with()

    def _test_event_closed_on_normal_io(self, nb_func, nb_arg, mock_io, mock_get_hub, event):
        if False:
            while True:
                i = 10
        mock_io.side_effect = [self._make_ignored_oserror(), 42]
        fd = 100
        result = nb_func(fd, nb_arg)
        self.assertEqual(result, 42)
        self._check_hub_event_closed(mock_get_hub, fd, event)

    def _test_event_closed_on_io_error(self, nb_func, nb_arg, mock_io, mock_get_hub, event):
        if False:
            return 10
        mock_io.side_effect = [self._make_ignored_oserror(), ValueError()]
        fd = 100
        with self.assertRaises(ValueError):
            nb_func(fd, nb_arg)
        self._check_hub_event_closed(mock_get_hub, fd, event)

    @mock.patch('gevent.os.get_hub')
    @mock.patch('gevent.os._write')
    def test_event_closed_on_write(self, mock_write, mock_get_hub):
        if False:
            for i in range(10):
                print('nop')
        self._test_event_closed_on_normal_io(os.nb_write, b'buf', mock_write, mock_get_hub, 2)

    @mock.patch('gevent.os.get_hub')
    @mock.patch('gevent.os._write')
    def test_event_closed_on_write_error(self, mock_write, mock_get_hub):
        if False:
            print('Hello World!')
        self._test_event_closed_on_io_error(os.nb_write, b'buf', mock_write, mock_get_hub, 2)

    @mock.patch('gevent.os.get_hub')
    @mock.patch('gevent.os._read')
    def test_event_closed_on_read(self, mock_read, mock_get_hub):
        if False:
            i = 10
            return i + 15
        self._test_event_closed_on_normal_io(os.nb_read, b'buf', mock_read, mock_get_hub, 1)

    @mock.patch('gevent.os.get_hub')
    @mock.patch('gevent.os._read')
    def test_event_closed_on_read_error(self, mock_read, mock_get_hub):
        if False:
            print('Hello World!')
        self._test_event_closed_on_io_error(os.nb_read, b'buf', mock_read, mock_get_hub, 1)

@greentest.skipUnless(hasattr(os, 'fork_and_watch'), 'Only on POSIX')
class TestForkAndWatch(greentest.TestCase):
    __timeout__ = greentest.LARGE_TIMEOUT

    def test_waitpid_all(self):
        if False:
            print('Hello World!')
        pid = os.fork_and_watch()
        if pid:
            os.waitpid(-1, 0)
            os._reap_children(0)
        else:
            gevent.sleep(2)
            os._exit(0)

    def test_waitpid_wrong_neg(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(OSError, os.waitpid, -2, 0)

    def test_waitpid_wrong_pos(self):
        if False:
            i = 10
            return i + 15
        self.assertRaises(OSError, os.waitpid, 1, 0)
if __name__ == '__main__':
    greentest.main()