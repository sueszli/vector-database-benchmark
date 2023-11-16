from __future__ import absolute_import, print_function, division
import unittest
import sys
import gevent.testing as greentest
from gevent._config import Loop
available_loops = Loop().get_options()
available_loops.pop('libuv', None)

def not_available(name):
    if False:
        print('Hello World!')
    return isinstance(available_loops[name], ImportError)

class WatcherTestMixin(object):
    kind = None

    def _makeOne(self):
        if False:
            for i in range(10):
                print('nop')
        return self.kind(default=False)

    def destroyOne(self, loop):
        if False:
            for i in range(10):
                print('nop')
        loop.destroy()

    def setUp(self):
        if False:
            print('Hello World!')
        self.loop = self._makeOne()
        self.core = sys.modules[self.kind.__module__]

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.destroyOne(self.loop)
        del self.loop

    def test_get_version(self):
        if False:
            print('Hello World!')
        version = self.core.get_version()
        self.assertIsInstance(version, str)
        self.assertTrue(version)
        header_version = self.core.get_header_version()
        self.assertIsInstance(header_version, str)
        self.assertTrue(header_version)
        self.assertEqual(version, header_version)

    def test_events_conversion(self):
        if False:
            while True:
                i = 10
        self.assertEqual(self.core._events_to_str(self.core.READ | self.core.WRITE), 'READ|WRITE')

    def test_EVENTS(self):
        if False:
            print('Hello World!')
        self.assertEqual(str(self.core.EVENTS), 'gevent.core.EVENTS')
        self.assertEqual(repr(self.core.EVENTS), 'gevent.core.EVENTS')

    def test_io(self):
        if False:
            for i in range(10):
                print('nop')
        if greentest.WIN:
            Error = (IOError, ValueError)
        else:
            Error = ValueError
        with self.assertRaises(Error):
            self.loop.io(-1, 1)
        if hasattr(self.core, 'TIMER'):
            with self.assertRaises(ValueError):
                self.loop.io(1, self.core.TIMER)
        if not greentest.WIN:
            io = self.loop.io(1, self.core.READ)
            io.fd = 2
            self.assertEqual(io.fd, 2)
            io.events = self.core.WRITE
            if not hasattr(self.core, 'libuv'):
                self.assertEqual(self.core._events_to_str(io.events), 'WRITE|_IOFDSET')
            else:
                self.assertEqual(self.core._events_to_str(io.events), 'WRITE')
            io.start(lambda : None)
            io.close()

    def test_timer_constructor(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self.loop.timer(1, -1)

    def test_signal_constructor(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            self.loop.signal(1000)

class LibevTestMixin(WatcherTestMixin):

    def test_flags_conversion(self):
        if False:
            return 10
        core = self.core
        if not greentest.WIN:
            self.assertEqual(core.loop(2, default=False).backend_int, 2)
        self.assertEqual(core.loop('select', default=False).backend, 'select')
        self.assertEqual(core._flags_to_int(None), 0)
        self.assertEqual(core._flags_to_int(['kqueue', 'SELECT']), core.BACKEND_KQUEUE | core.BACKEND_SELECT)
        self.assertEqual(core._flags_to_list(core.BACKEND_PORT | core.BACKEND_POLL), ['port', 'poll'])
        self.assertRaises(ValueError, core.loop, ['port', 'blabla'])
        self.assertRaises(TypeError, core.loop, object())

@unittest.skipIf(not_available('libev-cext'), 'Needs libev-cext')
class TestLibevCext(LibevTestMixin, unittest.TestCase):
    kind = available_loops['libev-cext']

@unittest.skipIf(not_available('libev-cffi'), 'Needs libev-cffi')
class TestLibevCffi(LibevTestMixin, unittest.TestCase):
    kind = available_loops['libev-cffi']

@unittest.skipIf(not_available('libuv-cffi'), 'Needs libuv-cffi')
class TestLibuvCffi(WatcherTestMixin, unittest.TestCase):
    kind = available_loops['libuv-cffi']

    @greentest.skipOnLibev('libuv-specific')
    @greentest.skipOnWindows('Destroying the loop somehow fails')
    def test_io_multiplex_events(self):
        if False:
            return 10
        import socket
        sock = socket.socket()
        fd = sock.fileno()
        core = self.core
        read = self.loop.io(fd, core.READ)
        write = self.loop.io(fd, core.WRITE)
        try:
            real_watcher = read._watcher_ref
            read.start(lambda : None)
            self.assertEqual(real_watcher.events, core.READ)
            write.start(lambda : None)
            self.assertEqual(real_watcher.events, core.READ | core.WRITE)
            write.stop()
            self.assertEqual(real_watcher.events, core.READ)
            write.start(lambda : None)
            self.assertEqual(real_watcher.events, core.READ | core.WRITE)
            read.stop()
            self.assertEqual(real_watcher.events, core.WRITE)
            write.stop()
            self.assertEqual(real_watcher.events, 0)
        finally:
            read.close()
            write.close()
            sock.close()
if __name__ == '__main__':
    greentest.main()