"""
Test stubs for running tests against all supported adaptations of
nbio_interface.AbstractIOServices and variations such as without SSL and
with SSL.

Usage example:

```
import unittest

from ..io_services_test_stubs import IOServicesTestStubs


class TestGetNativeIOLoop(unittest.TestCase,
                          IOServicesTestStubs):

    def start(self):
        native_loop = self.create_nbio().get_native_ioloop()
        self.assertIsNotNone(self._native_loop)
        self.assertIs(native_loop, self._native_loop)
```

"""
import sys
import unittest
from tests.wrappers.threaded_test_wrapper import run_in_thread_with_timeout

class IOServicesTestStubs(object):
    """Provides a stub test method for each combination of parameters we wish to
    test

    """
    _nbio_factory = None
    _native_loop = None
    _use_ssl = None

    def start(self):
        if False:
            print('Hello World!')
        ' Subclasses must override to run the test. This method is called\n        from a thread.\n\n        '
        raise NotImplementedError

    def create_nbio(self):
        if False:
            for i in range(10):
                print('nop')
        'Create the configured AbstractIOServices adaptation and schedule\n        it to be closed automatically when the test terminates.\n\n        :param unittest.TestCase self:\n        :rtype: pika.adapters.utils.nbio_interface.AbstractIOServices\n\n        '
        nbio = self._nbio_factory()
        self.addCleanup(nbio.close)
        return nbio

    def _run_start(self, nbio_factory, native_loop, use_ssl=False):
        if False:
            i = 10
            return i + 15
        'Called by framework-specific test stubs to initialize test paramters\n        and execute the `self.start()` method.\n\n        :param nbio_interface.AbstractIOServices _() nbio_factory: function\n            to call to create an instance of `AbstractIOServices` adaptation.\n        :param native_loop: native loop implementation instance\n        :param bool use_ssl: Whether to test with SSL instead of Plaintext\n            transport. Defaults to Plaintext.\n        '
        self._nbio_factory = nbio_factory
        self._native_loop = native_loop
        self._use_ssl = use_ssl
        self.start()

    @run_in_thread_with_timeout
    def test_with_select_connection_io_services(self):
        if False:
            print('Hello World!')
        from pika.adapters.select_connection import IOLoop
        from pika.adapters.utils.selector_ioloop_adapter import SelectorIOServicesAdapter
        native_loop = IOLoop()
        self._run_start(nbio_factory=lambda : SelectorIOServicesAdapter(native_loop), native_loop=native_loop)

    @run_in_thread_with_timeout
    def test_with_tornado_io_services(self):
        if False:
            while True:
                i = 10
        from tornado.ioloop import IOLoop
        from pika.adapters.utils.selector_ioloop_adapter import SelectorIOServicesAdapter
        native_loop = IOLoop()
        self._run_start(nbio_factory=lambda : SelectorIOServicesAdapter(native_loop), native_loop=native_loop)

    @unittest.skipIf(sys.version_info < (3, 4), 'Asyncio is available only with Python 3.4+')
    @run_in_thread_with_timeout
    def test_with_asyncio_io_services(self):
        if False:
            print('Hello World!')
        import asyncio
        from pika.adapters.asyncio_connection import _AsyncioIOServicesAdapter
        native_loop = asyncio.new_event_loop()
        self._run_start(nbio_factory=lambda : _AsyncioIOServicesAdapter(native_loop), native_loop=native_loop)