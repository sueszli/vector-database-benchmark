import asyncio
import os
import socket
import sys
import tempfile
import unittest
import uuid
from uvloop import _testbase as tb

class MyDatagramProto(asyncio.DatagramProtocol):
    done = None

    def __init__(self, loop=None):
        if False:
            return 10
        self.state = 'INITIAL'
        self.nbytes = 0
        if loop is not None:
            self.done = asyncio.Future(loop=loop)

    def connection_made(self, transport):
        if False:
            while True:
                i = 10
        self.transport = transport
        assert self.state == 'INITIAL', self.state
        self.state = 'INITIALIZED'

    def datagram_received(self, data, addr):
        if False:
            for i in range(10):
                print('nop')
        assert self.state == 'INITIALIZED', self.state
        self.nbytes += len(data)

    def error_received(self, exc):
        if False:
            for i in range(10):
                print('nop')
        assert self.state == 'INITIALIZED', self.state
        raise exc

    def connection_lost(self, exc):
        if False:
            return 10
        assert self.state == 'INITIALIZED', self.state
        self.state = 'CLOSED'
        if self.done:
            self.done.set_result(None)

class _TestUDP:

    def _test_create_datagram_endpoint_addrs(self, family, lc_addr):
        if False:
            for i in range(10):
                print('nop')

        class TestMyDatagramProto(MyDatagramProto):

            def __init__(inner_self):
                if False:
                    while True:
                        i = 10
                super().__init__(loop=self.loop)

            def datagram_received(self, data, addr):
                if False:
                    i = 10
                    return i + 15
                super().datagram_received(data, addr)
                self.transport.sendto(b'resp:' + data, addr)
        coro = self.loop.create_datagram_endpoint(TestMyDatagramProto, local_addr=lc_addr, family=family)
        (s_transport, server) = self.loop.run_until_complete(coro)
        remote_addr = s_transport.get_extra_info('sockname')
        (host, port, *_) = remote_addr
        self.assertIsInstance(server, TestMyDatagramProto)
        self.assertEqual('INITIALIZED', server.state)
        self.assertIs(server.transport, s_transport)
        extra = {}
        if hasattr(socket, 'SO_REUSEPORT'):
            extra['reuse_port'] = True
        coro = self.loop.create_datagram_endpoint(lambda : MyDatagramProto(loop=self.loop), family=family, remote_addr=(host, port), **extra)
        (transport, client) = self.loop.run_until_complete(coro)
        self.assertIsInstance(client, MyDatagramProto)
        self.assertEqual('INITIALIZED', client.state)
        self.assertIs(client.transport, transport)
        transport.sendto(b'xxx')
        tb.run_until(self.loop, lambda : server.nbytes)
        self.assertEqual(3, server.nbytes)
        tb.run_until(self.loop, lambda : client.nbytes)
        self.assertEqual(8, client.nbytes)
        transport.sendto(b'xxx', remote_addr)
        tb.run_until(self.loop, lambda : server.nbytes > 3 or client.done.done())
        self.assertEqual(6, server.nbytes)
        tb.run_until(self.loop, lambda : client.nbytes > 8)
        self.assertEqual(16, client.nbytes)
        with self.assertRaisesRegex(ValueError, 'Invalid address.*' + repr(remote_addr)):
            bad_addr = list(remote_addr)
            bad_addr[1] += 1
            bad_addr = tuple(bad_addr)
            transport.sendto(b'xxx', bad_addr)
        if remote_addr[0] != lc_addr[0]:
            with self.assertRaisesRegex(ValueError, 'Invalid address.*' + repr(remote_addr)):
                bad_addr = list(remote_addr)
                bad_addr[0] = lc_addr[0]
                bad_addr = tuple(bad_addr)
                transport.sendto(b'xxx', bad_addr)
        self.assertIsNotNone(transport.get_extra_info('sockname'))
        transport.close()
        self.loop.run_until_complete(client.done)
        self.assertEqual('CLOSED', client.state)
        server.transport.close()
        self.loop.run_until_complete(server.done)

    def test_create_datagram_endpoint_addrs_ipv4(self):
        if False:
            return 10
        self._test_create_datagram_endpoint_addrs(socket.AF_INET, ('127.0.0.1', 0))

    def test_create_datagram_endpoint_addrs_ipv4_nameaddr(self):
        if False:
            print('Hello World!')
        self._test_create_datagram_endpoint_addrs(socket.AF_INET, ('localhost', 0))

    def _test_create_datagram_endpoint_addrs_ipv6(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_create_datagram_endpoint_addrs(socket.AF_INET6, ('::1', 0))

    def test_create_datagram_endpoint_ipv6_family(self):
        if False:
            print('Hello World!')

        class TestMyDatagramProto(MyDatagramProto):

            def __init__(inner_self):
                if False:
                    i = 10
                    return i + 15
                super().__init__(loop=self.loop)

            def datagram_received(self, data, addr):
                if False:
                    print('Hello World!')
                super().datagram_received(data, addr)
                self.transport.sendto(b'resp:' + data, addr)
        coro = self.loop.create_datagram_endpoint(TestMyDatagramProto, local_addr=None, family=socket.AF_INET6)
        s_transport = None
        try:
            (s_transport, server) = self.loop.run_until_complete(coro)
        finally:
            if s_transport:
                s_transport.close()
                self.loop.run_until_complete(asyncio.sleep(0.1))

    def test_create_datagram_endpoint_sock(self):
        if False:
            i = 10
            return i + 15
        sock = None
        local_address = ('127.0.0.1', 0)
        infos = self.loop.run_until_complete(self.loop.getaddrinfo(*local_address, type=socket.SOCK_DGRAM))
        for (family, type, proto, cname, address) in infos:
            try:
                sock = socket.socket(family=family, type=type, proto=proto)
                sock.setblocking(False)
                sock.bind(address)
            except Exception:
                pass
            else:
                break
        else:
            assert False, 'Can not create socket.'
        with sock:
            f = self.loop.create_datagram_endpoint(lambda : MyDatagramProto(loop=self.loop), sock=sock)
            (tr, pr) = self.loop.run_until_complete(f)
            self.assertIsInstance(pr, MyDatagramProto)
            tr.close()
            self.loop.run_until_complete(pr.done)

    def test_create_datagram_endpoint_sock_unix_domain(self):
        if False:
            while True:
                i = 10

        class Proto(asyncio.DatagramProtocol):
            done = None

            def __init__(self, loop):
                if False:
                    for i in range(10):
                        print('nop')
                self.state = 'INITIAL'
                self.addrs = set()
                self.done = asyncio.Future(loop=loop)
                self.data = b''

            def connection_made(self, transport):
                if False:
                    for i in range(10):
                        print('nop')
                self.transport = transport
                assert self.state == 'INITIAL', self.state
                self.state = 'INITIALIZED'

            def datagram_received(self, data, addr):
                if False:
                    i = 10
                    return i + 15
                assert self.state == 'INITIALIZED', self.state
                self.addrs.add(addr)
                self.data += data
                if self.data == b'STOP' and (not self.done.done()):
                    self.done.set_result(True)

            def error_received(self, exc):
                if False:
                    print('Hello World!')
                assert self.state == 'INITIALIZED', self.state
                if not self.done.done():
                    self.done.set_exception(exc or RuntimeError())

            def connection_lost(self, exc):
                if False:
                    print('Hello World!')
                assert self.state == 'INITIALIZED', self.state
                self.state = 'CLOSED'
                if self.done and (not self.done.done()):
                    self.done.set_result(None)
        tmp_file = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
        sock = socket.socket(socket.AF_UNIX, type=socket.SOCK_DGRAM)
        sock.bind(tmp_file)
        with sock:
            pr = Proto(loop=self.loop)
            f = self.loop.create_datagram_endpoint(lambda : pr, sock=sock)
            (tr, pr_prime) = self.loop.run_until_complete(f)
            self.assertIs(pr, pr_prime)
            tmp_file2 = os.path.join(tempfile.gettempdir(), str(uuid.uuid4()))
            sock2 = socket.socket(socket.AF_UNIX, type=socket.SOCK_DGRAM)
            sock2.bind(tmp_file2)
            with sock2:
                f2 = self.loop.create_datagram_endpoint(asyncio.DatagramProtocol, sock=sock2)
                (tr2, pr2) = self.loop.run_until_complete(f2)
                tr2.sendto(b'STOP', tmp_file)
                self.loop.run_until_complete(pr.done)
                tr.close()
                tr2.close()
                self.loop.run_until_complete(asyncio.sleep(0.2))
                self.assertIn(tmp_file2, pr.addrs)

    def test_create_datagram_1(self):
        if False:
            i = 10
            return i + 15
        server_addr = ('127.0.0.1', 8888)
        client_addr = ('127.0.0.1', 0)

        async def run():
            (server_transport, client_protocol) = await self.loop.create_datagram_endpoint(asyncio.DatagramProtocol, local_addr=server_addr)
            (client_transport, client_conn) = await self.loop.create_datagram_endpoint(asyncio.DatagramProtocol, remote_addr=server_addr, local_addr=client_addr)
            client_transport.close()
            server_transport.close()
            await asyncio.sleep(0.1)
        self.loop.run_until_complete(run())

    def test_socketpair(self):
        if False:
            print('Hello World!')
        peername = asyncio.Future(loop=self.loop)

        class Proto(MyDatagramProto):

            def datagram_received(self, data, addr):
                if False:
                    while True:
                        i = 10
                super().datagram_received(data, addr)
                peername.set_result(addr)
        (s1, s2) = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM, 0)
        with s1, s2:
            f = self.loop.create_datagram_endpoint(lambda : Proto(loop=self.loop), sock=s1)
            (tr, pr) = self.loop.run_until_complete(f)
            self.assertIsInstance(pr, Proto)
            s2.send(b'hello, socketpair')
            addr = self.loop.run_until_complete(asyncio.wait_for(peername, 1))
            if sys.platform.startswith('linux'):
                self.assertEqual(addr, None)
            else:
                self.assertEqual(addr, '')
            self.assertEqual(pr.nbytes, 17)
            if not self.is_asyncio_loop():
                data = b'from uvloop'
                tr.sendto(data)
                result = self.loop.run_until_complete(asyncio.wait_for(self.loop.run_in_executor(None, s2.recv, 1024), 1))
                self.assertEqual(data, result)
            tr.close()
            self.loop.run_until_complete(pr.done)

    def _skip_create_datagram_endpoint_reuse_addr(self):
        if False:
            i = 10
            return i + 15
        if self.implementation == 'asyncio':
            if sys.version_info[:2] >= (3, 11):
                raise unittest.SkipTest()
            if (3, 8, 0) <= sys.version_info < (3, 8, 1):
                raise unittest.SkipTest()

    def test_create_datagram_endpoint_reuse_address_error(self):
        if False:
            i = 10
            return i + 15
        self._skip_create_datagram_endpoint_reuse_addr()
        coro = self.loop.create_datagram_endpoint(lambda : MyDatagramProto(loop=self.loop), local_addr=('127.0.0.1', 0), reuse_address=True)
        with self.assertRaises(ValueError):
            self.loop.run_until_complete(coro)

    def test_create_datagram_endpoint_reuse_address_warning(self):
        if False:
            while True:
                i = 10
        self._skip_create_datagram_endpoint_reuse_addr()
        coro = self.loop.create_datagram_endpoint(lambda : MyDatagramProto(loop=self.loop), local_addr=('127.0.0.1', 0), reuse_address=False)
        with self.assertWarns(DeprecationWarning):
            (tr, pr) = self.loop.run_until_complete(coro)
        tr.close()
        self.loop.run_until_complete(pr.done)

class Test_UV_UDP(_TestUDP, tb.UVTestCase):

    def test_create_datagram_endpoint_wrong_sock(self):
        if False:
            for i in range(10):
                print('nop')
        sock = socket.socket(socket.AF_INET)
        with sock:
            coro = self.loop.create_datagram_endpoint(lambda : None, sock=sock)
            with self.assertRaisesRegex(ValueError, 'A UDP Socket was expected'):
                self.loop.run_until_complete(coro)

    def test_udp_sendto_dns(self):
        if False:
            for i in range(10):
                print('nop')
        coro = self.loop.create_datagram_endpoint(asyncio.DatagramProtocol, local_addr=('127.0.0.1', 0), family=socket.AF_INET)
        (s_transport, server) = self.loop.run_until_complete(coro)
        with self.assertRaisesRegex(ValueError, 'DNS lookup'):
            s_transport.sendto(b'aaaa', ('example.com', 80))
        with self.assertRaisesRegex(ValueError, 'socket family mismatch'):
            s_transport.sendto(b'aaaa', ('::1', 80))
        s_transport.close()
        self.loop.run_until_complete(asyncio.sleep(0.01))

    def test_send_after_close(self):
        if False:
            while True:
                i = 10
        coro = self.loop.create_datagram_endpoint(asyncio.DatagramProtocol, local_addr=('127.0.0.1', 0), family=socket.AF_INET)
        (s_transport, _) = self.loop.run_until_complete(coro)
        s_transport.close()
        s_transport.sendto(b'aaaa', ('127.0.0.1', 80))
        self.loop.run_until_complete(asyncio.sleep(0.01))
        s_transport.sendto(b'aaaa', ('127.0.0.1', 80))

    @unittest.skipUnless(tb.has_IPv6, 'no IPv6')
    def test_create_datagram_endpoint_addrs_ipv6(self):
        if False:
            return 10
        self._test_create_datagram_endpoint_addrs_ipv6()

class Test_AIO_UDP(_TestUDP, tb.AIOTestCase):

    @unittest.skipUnless(tb.has_IPv6, 'no IPv6')
    def test_create_datagram_endpoint_addrs_ipv6(self):
        if False:
            for i in range(10):
                print('nop')
        self._test_create_datagram_endpoint_addrs_ipv6()