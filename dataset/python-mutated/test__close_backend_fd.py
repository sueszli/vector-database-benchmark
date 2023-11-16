from __future__ import print_function
import os
import unittest
import gevent
from gevent import core
from gevent.hub import Hub
from gevent.testing import sysinfo

@unittest.skipUnless(getattr(core, 'LIBEV_EMBED', False), 'Needs embedded libev. hub.loop.fileno is only defined when we embed libev for some reason. Choosing specific backends is also only supported by libev (not libuv), and besides, libuv has a nasty tendency to abort() the process if its FD gets closed. ')
class Test(unittest.TestCase):
    BACKENDS_THAT_SUCCEED_WHEN_FD_CLOSED = ('kqueue', 'epoll', 'linux_aio', 'linux_iouring')
    BACKENDS_THAT_WILL_FAIL_TO_CREATE_AT_RUNTIME = ('linux_iouring',) if not sysinfo.libev_supports_linux_iouring() else ()
    BACKENDS_THAT_WILL_FAIL_TO_CREATE_AT_RUNTIME += ('linux_aio',) if not sysinfo.libev_supports_linux_aio() else ()

    def _check_backend(self, backend):
        if False:
            i = 10
            return i + 15
        hub = Hub(backend, default=False)
        try:
            self.assertEqual(hub.loop.backend, backend)
            gevent.sleep(0.001)
            fileno = hub.loop.fileno()
            if fileno is None:
                return
            os.close(fileno)
            if backend in self.BACKENDS_THAT_SUCCEED_WHEN_FD_CLOSED:
                gevent.sleep(0.001)
            else:
                with self.assertRaisesRegex(SystemError, '(libev)'):
                    gevent.sleep(0.001)
            hub.destroy()
            self.assertIn('destroyed', repr(hub))
        finally:
            if hub.loop is not None:
                hub.destroy()

    @classmethod
    def _make_test(cls, count, backend):
        if False:
            while True:
                i = 10
        if backend in cls.BACKENDS_THAT_WILL_FAIL_TO_CREATE_AT_RUNTIME:

            def test(self):
                if False:
                    return 10
                with self.assertRaisesRegex(SystemError, 'ev_loop_new'):
                    Hub(backend, default=False)
        else:

            def test(self):
                if False:
                    i = 10
                    return i + 15
                self._check_backend(backend)
        test.__name__ = 'test_' + backend + '_' + str(count)
        return (test.__name__, test)

    @classmethod
    def _make_tests(cls):
        if False:
            i = 10
            return i + 15
        count = backend = None
        for count in range(2):
            for backend in core.supported_backends():
                (name, func) = cls._make_test(count, backend)
                setattr(cls, name, func)
                name = func = None
Test._make_tests()
if __name__ == '__main__':
    unittest.main()