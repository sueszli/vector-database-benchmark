import socket
import subprocess
import sys
import textwrap
import unittest
from tornado import gen
from tornado.iostream import IOStream
from tornado.log import app_log
from tornado.tcpserver import TCPServer
from tornado.test.util import skipIfNonUnix
from tornado.testing import AsyncTestCase, ExpectLog, bind_unused_port, gen_test
from typing import Tuple

class TCPServerTest(AsyncTestCase):

    @gen_test
    def test_handle_stream_coroutine_logging(self):
        if False:
            i = 10
            return i + 15

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                if False:
                    while True:
                        i = 10
                yield stream.read_bytes(len(b'hello'))
                stream.close()
                1 / 0
        server = client = None
        try:
            (sock, port) = bind_unused_port()
            server = TestServer()
            server.add_socket(sock)
            client = IOStream(socket.socket())
            with ExpectLog(app_log, 'Exception in callback'):
                yield client.connect(('localhost', port))
                yield client.write(b'hello')
                yield client.read_until_close()
                yield gen.moment
        finally:
            if server is not None:
                server.stop()
            if client is not None:
                client.close()

    @gen_test
    def test_handle_stream_native_coroutine(self):
        if False:
            for i in range(10):
                print('nop')

        class TestServer(TCPServer):

            async def handle_stream(self, stream, address):
                stream.write(b'data')
                stream.close()
        (sock, port) = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        client = IOStream(socket.socket())
        yield client.connect(('localhost', port))
        result = (yield client.read_until_close())
        self.assertEqual(result, b'data')
        server.stop()
        client.close()

    def test_stop_twice(self):
        if False:
            i = 10
            return i + 15
        (sock, port) = bind_unused_port()
        server = TCPServer()
        server.add_socket(sock)
        server.stop()
        server.stop()

    @gen_test
    def test_stop_in_callback(self):
        if False:
            print('Hello World!')

        class TestServer(TCPServer):

            @gen.coroutine
            def handle_stream(self, stream, address):
                if False:
                    print('Hello World!')
                server.stop()
                yield stream.read_until_close()
        (sock, port) = bind_unused_port()
        server = TestServer()
        server.add_socket(sock)
        server_addr = ('localhost', port)
        N = 40
        clients = [IOStream(socket.socket()) for i in range(N)]
        connected_clients = []

        @gen.coroutine
        def connect(c):
            if False:
                i = 10
                return i + 15
            try:
                yield c.connect(server_addr)
            except EnvironmentError:
                pass
            else:
                connected_clients.append(c)
        yield [connect(c) for c in clients]
        self.assertGreater(len(connected_clients), 0, 'all clients failed connecting')
        try:
            if len(connected_clients) == N:
                self.skipTest('at least one client should fail connecting for the test to be meaningful')
        finally:
            for c in connected_clients:
                c.close()

@skipIfNonUnix
class TestMultiprocess(unittest.TestCase):

    def run_subproc(self, code: str) -> Tuple[str, str]:
        if False:
            for i in range(10):
                print('nop')
        try:
            result = subprocess.run([sys.executable, '-Werror::DeprecationWarning'], capture_output=True, input=code, encoding='utf8', check=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f'Process returned {e.returncode} stdout={e.stdout} stderr={e.stderr}') from e
        return (result.stdout, result.stderr)

    def test_listen_single(self):
        if False:
            while True:
                i = 10
        code = textwrap.dedent("\n            import asyncio\n            from tornado.tcpserver import TCPServer\n\n            async def main():\n                server = TCPServer()\n                server.listen(0, address='127.0.0.1')\n\n            asyncio.run(main())\n            print('012', end='')\n        ")
        (out, err) = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')

    def test_bind_start(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent('\n            import warnings\n\n            from tornado.ioloop import IOLoop\n            from tornado.process import task_id\n            from tornado.tcpserver import TCPServer\n\n            warnings.simplefilter("ignore", DeprecationWarning)\n\n            server = TCPServer()\n            server.bind(0, address=\'127.0.0.1\')\n            server.start(3)\n            IOLoop.current().run_sync(lambda: None)\n            print(task_id(), end=\'\')\n        ')
        (out, err) = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')

    def test_add_sockets(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("\n            import asyncio\n            from tornado.netutil import bind_sockets\n            from tornado.process import fork_processes, task_id\n            from tornado.ioloop import IOLoop\n            from tornado.tcpserver import TCPServer\n\n            sockets = bind_sockets(0, address='127.0.0.1')\n            fork_processes(3)\n            async def post_fork_main():\n                server = TCPServer()\n                server.add_sockets(sockets)\n            asyncio.run(post_fork_main())\n            print(task_id(), end='')\n        ")
        (out, err) = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')

    def test_listen_multi_reuse_port(self):
        if False:
            print('Hello World!')
        code = textwrap.dedent("\n            import asyncio\n            import socket\n            from tornado.netutil import bind_sockets\n            from tornado.process import task_id, fork_processes\n            from tornado.tcpserver import TCPServer\n\n            # Pick an unused port which we will be able to bind to multiple times.\n            (sock,) = bind_sockets(0, address='127.0.0.1',\n                family=socket.AF_INET, reuse_port=True)\n            port = sock.getsockname()[1]\n\n            fork_processes(3)\n\n            async def main():\n                server = TCPServer()\n                server.listen(port, address='127.0.0.1', reuse_port=True)\n            asyncio.run(main())\n            print(task_id(), end='')\n            ")
        (out, err) = self.run_subproc(code)
        self.assertEqual(''.join(sorted(out)), '012')
        self.assertEqual(err, '')