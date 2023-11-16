import errno
import os
import signal
import socket
from subprocess import Popen
import sys
import time
import unittest
from tornado.netutil import BlockingResolver, OverrideResolver, ThreadedResolver, is_valid_ip, bind_sockets
from tornado.testing import AsyncTestCase, gen_test, bind_unused_port
from tornado.test.util import skipIfNoNetwork
import typing
if typing.TYPE_CHECKING:
    from typing import List
try:
    import pycares
except ImportError:
    pycares = None
else:
    from tornado.platform.caresresolver import CaresResolver
try:
    import twisted
    import twisted.names
except ImportError:
    twisted = None
else:
    from tornado.platform.twisted import TwistedResolver

class _ResolverTestMixin(object):
    resolver = None

    @gen_test
    def test_localhost(self: typing.Any):
        if False:
            for i in range(10):
                print('nop')
        addrinfo = (yield self.resolver.resolve('localhost', 80, socket.AF_UNSPEC))
        self.assertTrue((socket.AF_INET, ('127.0.0.1', 80)) in addrinfo or (socket.AF_INET6, ('::1', 80)) in addrinfo, f'loopback address not found in {addrinfo}')

class _ResolverErrorTestMixin(object):
    resolver = None

    @gen_test
    def test_bad_host(self: typing.Any):
        if False:
            while True:
                i = 10
        with self.assertRaises(IOError):
            yield self.resolver.resolve('an invalid domain', 80, socket.AF_UNSPEC)

def _failing_getaddrinfo(*args):
    if False:
        return 10
    'Dummy implementation of getaddrinfo for use in mocks'
    raise socket.gaierror(errno.EIO, 'mock: lookup failed')

@skipIfNoNetwork
class BlockingResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.resolver = BlockingResolver()

class BlockingResolverErrorTest(AsyncTestCase, _ResolverErrorTestMixin):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.resolver = BlockingResolver()
        self.real_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = _failing_getaddrinfo

    def tearDown(self):
        if False:
            print('Hello World!')
        socket.getaddrinfo = self.real_getaddrinfo
        super().tearDown()

class OverrideResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        mapping = {('google.com', 80): ('1.2.3.4', 80), ('google.com', 80, socket.AF_INET): ('1.2.3.4', 80), ('google.com', 80, socket.AF_INET6): ('2a02:6b8:7c:40c:c51e:495f:e23a:3', 80)}
        self.resolver = OverrideResolver(BlockingResolver(), mapping)

    @gen_test
    def test_resolve_multiaddr(self):
        if False:
            for i in range(10):
                print('nop')
        result = (yield self.resolver.resolve('google.com', 80, socket.AF_INET))
        self.assertIn((socket.AF_INET, ('1.2.3.4', 80)), result)
        result = (yield self.resolver.resolve('google.com', 80, socket.AF_INET6))
        self.assertIn((socket.AF_INET6, ('2a02:6b8:7c:40c:c51e:495f:e23a:3', 80, 0, 0)), result)

@skipIfNoNetwork
class ThreadedResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.resolver = ThreadedResolver()

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        self.resolver.close()
        super().tearDown()

class ThreadedResolverErrorTest(AsyncTestCase, _ResolverErrorTestMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.resolver = BlockingResolver()
        self.real_getaddrinfo = socket.getaddrinfo
        socket.getaddrinfo = _failing_getaddrinfo

    def tearDown(self):
        if False:
            print('Hello World!')
        socket.getaddrinfo = self.real_getaddrinfo
        super().tearDown()

@skipIfNoNetwork
@unittest.skipIf(sys.platform == 'win32', 'preexec_fn not available on win32')
class ThreadedResolverImportTest(unittest.TestCase):

    def test_import(self):
        if False:
            while True:
                i = 10
        TIMEOUT = 5
        command = [sys.executable, '-c', 'import tornado.test.resolve_test_helper']
        start = time.time()
        popen = Popen(command, preexec_fn=lambda : signal.alarm(TIMEOUT))
        while time.time() - start < TIMEOUT:
            return_code = popen.poll()
            if return_code is not None:
                self.assertEqual(0, return_code)
                return
            time.sleep(0.05)
        self.fail('import timed out')

@skipIfNoNetwork
@unittest.skipIf(pycares is None, 'pycares module not present')
@unittest.skipIf(sys.platform == 'win32', "pycares doesn't return loopback on windows")
@unittest.skipIf(sys.platform == 'darwin', "pycares doesn't return 127.0.0.1 on darwin")
class CaresResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.resolver = CaresResolver()

@skipIfNoNetwork
@unittest.skipIf(twisted is None, 'twisted module not present')
@unittest.skipIf(getattr(twisted, '__version__', '0.0') < '12.1', 'old version of twisted')
@unittest.skipIf(sys.platform == 'win32', 'twisted resolver hangs on windows')
class TwistedResolverTest(AsyncTestCase, _ResolverTestMixin):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.resolver = TwistedResolver()

class IsValidIPTest(unittest.TestCase):

    def test_is_valid_ip(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(is_valid_ip('127.0.0.1'))
        self.assertTrue(is_valid_ip('4.4.4.4'))
        self.assertTrue(is_valid_ip('::1'))
        self.assertTrue(is_valid_ip('2620:0:1cfe:face:b00c::3'))
        self.assertTrue(not is_valid_ip('www.google.com'))
        self.assertTrue(not is_valid_ip('localhost'))
        self.assertTrue(not is_valid_ip('4.4.4.4<'))
        self.assertTrue(not is_valid_ip(' 127.0.0.1'))
        self.assertTrue(not is_valid_ip(''))
        self.assertTrue(not is_valid_ip(' '))
        self.assertTrue(not is_valid_ip('\n'))
        self.assertTrue(not is_valid_ip('\x00'))
        self.assertTrue(not is_valid_ip('a' * 100))

class TestPortAllocation(unittest.TestCase):

    def test_same_port_allocation(self):
        if False:
            return 10
        if 'TRAVIS' in os.environ:
            self.skipTest('dual-stack servers often have port conflicts on travis')
        sockets = bind_sockets(0, 'localhost')
        try:
            port = sockets[0].getsockname()[1]
            self.assertTrue(all((s.getsockname()[1] == port for s in sockets[1:])))
        finally:
            for sock in sockets:
                sock.close()

    @unittest.skipIf(not hasattr(socket, 'SO_REUSEPORT'), 'SO_REUSEPORT is not supported')
    def test_reuse_port(self):
        if False:
            print('Hello World!')
        sockets = []
        (socket, port) = bind_unused_port(reuse_port=True)
        try:
            sockets = bind_sockets(port, '127.0.0.1', reuse_port=True)
            self.assertTrue(all((s.getsockname()[1] == port for s in sockets)))
        finally:
            socket.close()
            for sock in sockets:
                sock.close()