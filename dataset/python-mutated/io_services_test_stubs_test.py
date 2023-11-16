"""
Test for io_services_test_stubs.py
"""
import pika.compat
try:
    import asyncio
except ImportError:
    asyncio = None
import sys
import threading
import unittest
import tornado.ioloop
import twisted.internet.reactor
from pika.adapters import select_connection
from tests.stubs.io_services_test_stubs import IOServicesTestStubs
_TORNADO_IO_LOOP = tornado.ioloop.IOLoop()
_TORNADO_IOLOOP_CLASS = _TORNADO_IO_LOOP.__class__
_TORNADO_IO_LOOP.close()
del _TORNADO_IO_LOOP
_SUPPORTED_LOOP_CLASSES = {select_connection.IOLoop, _TORNADO_IOLOOP_CLASS}
if asyncio is not None:
    if pika.compat.ON_WINDOWS:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    _SUPPORTED_LOOP_CLASSES.add(asyncio.get_event_loop().__class__)

class TestStartCalledFromOtherThreadAndWithVaryingNativeLoops(unittest.TestCase, IOServicesTestStubs):
    _native_loop_classes = None

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        cls._native_loop_classes = set()

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        if cls._native_loop_classes != _SUPPORTED_LOOP_CLASSES:
            raise AssertionError('Expected these {} native I/O loop classes from IOServicesTestStubs: {!r}, but got these {}: {!r}'.format(len(_SUPPORTED_LOOP_CLASSES), _SUPPORTED_LOOP_CLASSES, len(cls._native_loop_classes), cls._native_loop_classes))

    def setUp(self):
        if False:
            return 10
        self._runner_thread_id = threading.current_thread().ident

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        nbio = self.create_nbio()
        native_loop = nbio.get_native_ioloop()
        self.assertIsNotNone(self._native_loop)
        self.assertIs(native_loop, self._native_loop)
        self._native_loop_classes.add(native_loop.__class__)
        self.assertNotEqual(threading.current_thread().ident, self._runner_thread_id)
        nbio.add_callback_threadsafe(nbio.stop)
        nbio.run()