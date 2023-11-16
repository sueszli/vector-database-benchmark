import asyncio
import io
import os
import socket
from uvloop import _testbase as tb

class MyReadPipeProto(asyncio.Protocol):
    done = None

    def __init__(self, loop=None):
        if False:
            print('Hello World!')
        self.state = ['INITIAL']
        self.nbytes = 0
        self.transport = None
        if loop is not None:
            self.done = asyncio.Future(loop=loop)

    def connection_made(self, transport):
        if False:
            while True:
                i = 10
        self.transport = transport
        assert self.state == ['INITIAL'], self.state
        self.state.append('CONNECTED')

    def data_received(self, data):
        if False:
            while True:
                i = 10
        assert self.state == ['INITIAL', 'CONNECTED'], self.state
        self.nbytes += len(data)

    def eof_received(self):
        if False:
            print('Hello World!')
        assert self.state == ['INITIAL', 'CONNECTED'], self.state
        self.state.append('EOF')

    def connection_lost(self, exc):
        if False:
            print('Hello World!')
        if 'EOF' not in self.state:
            self.state.append('EOF')
        assert self.state == ['INITIAL', 'CONNECTED', 'EOF'], self.state
        self.state.append('CLOSED')
        if self.done:
            self.done.set_result(None)

class MyWritePipeProto(asyncio.BaseProtocol):
    done = None
    paused = False

    def __init__(self, loop=None):
        if False:
            print('Hello World!')
        self.state = 'INITIAL'
        self.transport = None
        if loop is not None:
            self.done = asyncio.Future(loop=loop)

    def connection_made(self, transport):
        if False:
            print('Hello World!')
        self.transport = transport
        assert self.state == 'INITIAL', self.state
        self.state = 'CONNECTED'

    def connection_lost(self, exc):
        if False:
            print('Hello World!')
        assert self.state == 'CONNECTED', self.state
        self.state = 'CLOSED'
        if self.done:
            self.done.set_result(None)

    def pause_writing(self):
        if False:
            print('Hello World!')
        self.paused = True

    def resume_writing(self):
        if False:
            for i in range(10):
                print('nop')
        self.paused = False

class _BasePipeTest:

    def test_read_pipe(self):
        if False:
            i = 10
            return i + 15
        proto = MyReadPipeProto(loop=self.loop)
        (rpipe, wpipe) = os.pipe()
        pipeobj = io.open(rpipe, 'rb', 1024)

        async def connect():
            (t, p) = await self.loop.connect_read_pipe(lambda : proto, pipeobj)
            self.assertIs(p, proto)
            self.assertIs(t, proto.transport)
            self.assertEqual(['INITIAL', 'CONNECTED'], proto.state)
            self.assertEqual(0, proto.nbytes)
        self.loop.run_until_complete(connect())
        os.write(wpipe, b'1')
        tb.run_until(self.loop, lambda : proto.nbytes >= 1)
        self.assertEqual(1, proto.nbytes)
        os.write(wpipe, b'2345')
        tb.run_until(self.loop, lambda : proto.nbytes >= 5)
        self.assertEqual(['INITIAL', 'CONNECTED'], proto.state)
        self.assertEqual(5, proto.nbytes)
        os.close(wpipe)
        self.loop.run_until_complete(proto.done)
        self.assertEqual(['INITIAL', 'CONNECTED', 'EOF', 'CLOSED'], proto.state)
        self.assertIsNotNone(proto.transport.get_extra_info('pipe'))

    def test_read_pty_output(self):
        if False:
            for i in range(10):
                print('nop')
        proto = MyReadPipeProto(loop=self.loop)
        (master, slave) = os.openpty()
        master_read_obj = io.open(master, 'rb', 0)

        async def connect():
            (t, p) = await self.loop.connect_read_pipe(lambda : proto, master_read_obj)
            self.assertIs(p, proto)
            self.assertIs(t, proto.transport)
            self.assertEqual(['INITIAL', 'CONNECTED'], proto.state)
            self.assertEqual(0, proto.nbytes)
        self.loop.run_until_complete(connect())
        os.write(slave, b'1')
        tb.run_until(self.loop, lambda : proto.nbytes)
        self.assertEqual(1, proto.nbytes)
        os.write(slave, b'2345')
        tb.run_until(self.loop, lambda : proto.nbytes >= 5)
        self.assertEqual(['INITIAL', 'CONNECTED'], proto.state)
        self.assertEqual(5, proto.nbytes)
        self.loop.set_exception_handler(lambda loop, ctx: None)
        os.close(slave)
        proto.transport.close()
        self.loop.run_until_complete(proto.done)
        self.assertEqual(['INITIAL', 'CONNECTED', 'EOF', 'CLOSED'], proto.state)
        self.assertIsNotNone(proto.transport.get_extra_info('pipe'))

    def test_write_pipe(self):
        if False:
            i = 10
            return i + 15
        (rpipe, wpipe) = os.pipe()
        os.set_blocking(rpipe, False)
        pipeobj = io.open(wpipe, 'wb', 1024)
        proto = MyWritePipeProto(loop=self.loop)
        connect = self.loop.connect_write_pipe(lambda : proto, pipeobj)
        (transport, p) = self.loop.run_until_complete(connect)
        self.assertIs(p, proto)
        self.assertIs(transport, proto.transport)
        self.assertEqual('CONNECTED', proto.state)
        transport.write(b'1')
        data = bytearray()

        def reader(data):
            if False:
                for i in range(10):
                    print('nop')
            try:
                chunk = os.read(rpipe, 1024)
            except BlockingIOError:
                return len(data)
            data += chunk
            return len(data)
        tb.run_until(self.loop, lambda : reader(data) >= 1)
        self.assertEqual(b'1', data)
        transport.write(b'2345')
        tb.run_until(self.loop, lambda : reader(data) >= 5)
        self.assertEqual(b'12345', data)
        self.assertEqual('CONNECTED', proto.state)
        os.close(rpipe)
        self.assertIsNotNone(proto.transport.get_extra_info('pipe'))
        proto.transport.close()
        self.loop.run_until_complete(proto.done)
        self.assertEqual('CLOSED', proto.state)

    def test_write_pipe_disconnect_on_close(self):
        if False:
            print('Hello World!')
        (rsock, wsock) = socket.socketpair()
        rsock.setblocking(False)
        pipeobj = io.open(wsock.detach(), 'wb', 1024)
        proto = MyWritePipeProto(loop=self.loop)
        connect = self.loop.connect_write_pipe(lambda : proto, pipeobj)
        (transport, p) = self.loop.run_until_complete(connect)
        self.assertIs(p, proto)
        self.assertIs(transport, proto.transport)
        self.assertEqual('CONNECTED', proto.state)
        transport.write(b'1')
        data = self.loop.run_until_complete(self.loop.sock_recv(rsock, 1024))
        self.assertEqual(b'1', data)
        rsock.close()
        self.loop.run_until_complete(proto.done)
        self.assertEqual('CLOSED', proto.state)

    def test_write_pty(self):
        if False:
            print('Hello World!')
        (master, slave) = os.openpty()
        os.set_blocking(master, False)
        slave_write_obj = io.open(slave, 'wb', 0)
        proto = MyWritePipeProto(loop=self.loop)
        connect = self.loop.connect_write_pipe(lambda : proto, slave_write_obj)
        (transport, p) = self.loop.run_until_complete(connect)
        self.assertIs(p, proto)
        self.assertIs(transport, proto.transport)
        self.assertEqual('CONNECTED', proto.state)
        transport.write(b'1')
        data = bytearray()

        def reader(data):
            if False:
                print('Hello World!')
            try:
                chunk = os.read(master, 1024)
            except BlockingIOError:
                return len(data)
            data += chunk
            return len(data)
        tb.run_until(self.loop, lambda : reader(data) >= 1, timeout=10)
        self.assertEqual(b'1', data)
        transport.write(b'2345')
        tb.run_until(self.loop, lambda : reader(data) >= 5, timeout=10)
        self.assertEqual(b'12345', data)
        self.assertEqual('CONNECTED', proto.state)
        os.close(master)
        self.assertIsNotNone(proto.transport.get_extra_info('pipe'))
        proto.transport.close()
        self.loop.run_until_complete(proto.done)
        self.assertEqual('CLOSED', proto.state)

    def test_write_buffer_full(self):
        if False:
            for i in range(10):
                print('nop')
        (rpipe, wpipe) = os.pipe()
        pipeobj = io.open(wpipe, 'wb', 1024)
        proto = MyWritePipeProto(loop=self.loop)
        connect = self.loop.connect_write_pipe(lambda : proto, pipeobj)
        (transport, p) = self.loop.run_until_complete(connect)
        self.assertIs(p, proto)
        self.assertIs(transport, proto.transport)
        self.assertEqual('CONNECTED', proto.state)
        for i in range(32):
            transport.write(b'x' * 32768)
            if proto.paused:
                transport.write(b'x' * 32768)
                break
        else:
            self.fail("Didn't reach a full buffer")
        os.close(rpipe)
        self.loop.run_until_complete(asyncio.wait_for(proto.done, 1))
        self.assertEqual('CLOSED', proto.state)

class Test_UV_Pipes(_BasePipeTest, tb.UVTestCase):
    pass

class Test_AIO_Pipes(_BasePipeTest, tb.AIOTestCase):
    pass