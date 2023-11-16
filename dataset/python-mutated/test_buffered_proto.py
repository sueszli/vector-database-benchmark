import asyncio
import unittest
from test.test_asyncio import functional as func_tests

def tearDownModule():
    if False:
        print('Hello World!')
    asyncio.set_event_loop_policy(None)

class ReceiveStuffProto(asyncio.BufferedProtocol):

    def __init__(self, cb, con_lost_fut):
        if False:
            for i in range(10):
                print('nop')
        self.cb = cb
        self.con_lost_fut = con_lost_fut

    def get_buffer(self, sizehint):
        if False:
            for i in range(10):
                print('nop')
        self.buffer = bytearray(100)
        return self.buffer

    def buffer_updated(self, nbytes):
        if False:
            for i in range(10):
                print('nop')
        self.cb(self.buffer[:nbytes])

    def connection_lost(self, exc):
        if False:
            i = 10
            return i + 15
        if exc is None:
            self.con_lost_fut.set_result(None)
        else:
            self.con_lost_fut.set_exception(exc)

class BaseTestBufferedProtocol(func_tests.FunctionalTestCaseMixin):

    def new_loop(self):
        if False:
            print('Hello World!')
        raise NotImplementedError

    def test_buffered_proto_create_connection(self):
        if False:
            while True:
                i = 10
        NOISE = b'12345678+' * 1024

        async def client(addr):
            data = b''

            def on_buf(buf):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal data
                data += buf
                if data == NOISE:
                    tr.write(b'1')
            conn_lost_fut = self.loop.create_future()
            (tr, pr) = await self.loop.create_connection(lambda : ReceiveStuffProto(on_buf, conn_lost_fut), *addr)
            await conn_lost_fut

        async def on_server_client(reader, writer):
            writer.write(NOISE)
            await reader.readexactly(1)
            writer.close()
            await writer.wait_closed()
        srv = self.loop.run_until_complete(asyncio.start_server(on_server_client, '127.0.0.1', 0))
        addr = srv.sockets[0].getsockname()
        self.loop.run_until_complete(asyncio.wait_for(client(addr), 5))
        srv.close()
        self.loop.run_until_complete(srv.wait_closed())

class BufferedProtocolSelectorTests(BaseTestBufferedProtocol, unittest.TestCase):

    def new_loop(self):
        if False:
            while True:
                i = 10
        return asyncio.SelectorEventLoop()

@unittest.skipUnless(hasattr(asyncio, 'ProactorEventLoop'), 'Windows only')
class BufferedProtocolProactorTests(BaseTestBufferedProtocol, unittest.TestCase):

    def new_loop(self):
        if False:
            print('Hello World!')
        return asyncio.ProactorEventLoop()
if __name__ == '__main__':
    unittest.main()