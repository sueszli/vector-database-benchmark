import errno
import re
import socket
import sys
import pytest
import trio
from trio.testing._fake_net import FakeNet
if sys.platform == 'linux':
    ENOTCONN_MSG = 'Transport endpoint is not connected'
elif sys.platform == 'darwin':
    ENOTCONN_MSG = 'Socket is not connected'
else:
    ENOTCONN_MSG = 'Unknown error'

def fn() -> FakeNet:
    if False:
        return 10
    fn = FakeNet()
    fn.enable()
    return fn

async def test_basic_udp() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    await s1.bind(('127.0.0.1', 0))
    (ip, port) = s1.getsockname()
    assert ip == '127.0.0.1'
    assert port != 0
    with pytest.raises(OSError) as exc:
        await s1.bind(('192.0.2.1', 0))
    assert exc.value.errno == errno.EINVAL
    with pytest.raises(OSError) as exc:
        await s2.bind(('127.0.0.1', port))
    assert exc.value.errno == errno.EADDRINUSE
    await s2.sendto(b'xyz', s1.getsockname())
    (data, addr) = await s1.recvfrom(10)
    assert data == b'xyz'
    assert addr == s2.getsockname()
    await s1.sendto(b'abc', s2.getsockname())
    (data, addr) = await s2.recvfrom(10)
    assert data == b'abc'
    assert addr == s1.getsockname()

async def test_msg_trunc() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    await s1.bind(('127.0.0.1', 0))
    await s2.sendto(b'xyz', s1.getsockname())
    (data, addr) = await s1.recvfrom(10)

async def test_recv_methods() -> None:
    """Test all recv methods for codecov"""
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    with pytest.raises(NotImplementedError, match='code will most likely hang'):
        await s2.recv(10)
    await s1.bind(('127.0.0.1', 0))
    (ip, port) = s1.getsockname()
    assert ip == '127.0.0.1'
    assert port != 0
    await s2.sendto(b'abc', s1.getsockname())
    (data, addr) = await s1.recvfrom(10)
    assert data == b'abc'
    assert addr == s2.getsockname()
    await s1.sendto(b'def', s2.getsockname())
    data = await s2.recv(10)
    assert data == b'def'
    assert await s1.sendto(b'ghi', s2.getsockname()) == 3
    buf = bytearray(10)
    with pytest.raises(NotImplementedError, match='partial recvfrom_into'):
        (nbytes, addr) = await s2.recvfrom_into(buf, nbytes=2)
    (nbytes, addr) = await s2.recvfrom_into(buf)
    assert nbytes == 3
    assert buf == b'ghi' + b'\x00' * 7
    assert addr == s1.getsockname()
    assert await s1.sendto(b'jkl', s2.getsockname()) == 3
    buf2 = bytearray(10)
    nbytes = await s2.recv_into(buf2)
    assert nbytes == 3
    assert buf2 == b'jkl' + b'\x00' * 7
    if sys.platform == 'linux' and sys.implementation.name == 'cpython':
        flags: int = socket.MSG_MORE
    else:
        flags = 1
    with pytest.raises(OSError, match=ENOTCONN_MSG) as exc:
        await s2.send(b'mno')
    assert exc.value.errno == errno.ENOTCONN
    with pytest.raises(NotImplementedError, match='FakeNet send flags must be 0, not'):
        await s2.send(b'mno', flags)
    with pytest.raises(NotImplementedError, match='FakeNet send flags must be 0, not'):
        await s2.sendto(b'mno', flags, s1.getsockname())
    with pytest.raises(TypeError, match='wrong number of arguments'):
        await s2.sendto(b'mno', flags, s1.getsockname(), 'extra arg')

@pytest.mark.skipif(sys.platform == 'win32', reason='functions not in socket on windows')
async def test_nonwindows_functionality() -> None:
    if sys.platform != 'win32':
        fn()
        s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        await s2.bind(('127.0.0.1', 0))
        with pytest.raises(OSError, match=ENOTCONN_MSG) as exc:
            await s2.sendmsg([b'mno'])
        assert exc.value.errno == errno.ENOTCONN
        assert await s1.sendmsg([b'jkl'], (), 0, s2.getsockname()) == 3
        (data, ancdata, msg_flags, addr) = await s2.recvmsg(10)
        assert data == b'jkl'
        assert ancdata == []
        assert msg_flags == 0
        assert addr == s1.getsockname()
        assert await s1.sendto(b'xyzw', s2.getsockname()) == 4
        buf1 = bytearray(2)
        buf2 = bytearray(3)
        ret = await s2.recvmsg_into([buf1, buf2])
        (nbytes, ancdata, msg_flags, addr) = ret
        assert nbytes == 4
        assert buf1 == b'xy'
        assert buf2 == b'zw' + b'\x00'
        assert ancdata == []
        assert msg_flags == 0
        assert addr == s1.getsockname()
        assert await s1.sendto(b'xyzwv', s2.getsockname()) == 5
        buf1 = bytearray(2)
        ret = await s2.recvmsg_into([buf1])
        (nbytes, ancdata, msg_flags, addr) = ret
        assert nbytes == 2
        assert buf1 == b'xy'
        assert ancdata == []
        assert msg_flags == socket.MSG_TRUNC
        assert addr == s1.getsockname()
        with pytest.raises(AttributeError, match="'FakeSocket' object has no attribute 'share'"):
            await s1.share(0)

@pytest.mark.skipif(sys.platform != 'win32', reason='windows-specific fakesocket testing')
async def test_windows_functionality() -> None:
    if sys.platform == 'win32':
        fn()
        s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        s2 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
        await s1.bind(('127.0.0.1', 0))
        with pytest.raises(AttributeError, match="'FakeSocket' object has no attribute 'sendmsg'"):
            await s1.sendmsg([b'jkl'], (), 0, s2.getsockname())
        with pytest.raises(AttributeError, match="'FakeSocket' object has no attribute 'recvmsg'"):
            s2.recvmsg(0)
        with pytest.raises(AttributeError, match="'FakeSocket' object has no attribute 'recvmsg_into'"):
            s2.recvmsg_into([])
        with pytest.raises(NotImplementedError):
            s1.share(0)

async def test_basic_tcp() -> None:
    fn()
    with pytest.raises(NotImplementedError):
        trio.socket.socket()

async def test_not_implemented_functions() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    with pytest.raises(OSError, match="FakeNet doesn't implement getsockopt"):
        s1.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
    with pytest.raises(NotImplementedError, match='FakeNet always has IPV6_V6ONLY=True'):
        s1.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, False)
    with pytest.raises(OSError, match="FakeNet doesn't implement setsockopt"):
        s1.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, True)
    with pytest.raises(OSError, match="FakeNet doesn't implement setsockopt"):
        s1.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s1.set_inheritable(False)
    with pytest.raises(NotImplementedError, match="FakeNet can't make inheritable sockets"):
        s1.set_inheritable(True)
    assert not s1.get_inheritable()

async def test_getpeername() -> None:
    fn()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    with pytest.raises(OSError, match=ENOTCONN_MSG) as exc:
        s1.getpeername()
    assert exc.value.errno == errno.ENOTCONN
    await s1.bind(('127.0.0.1', 0))
    with pytest.raises(AssertionError, match='This method seems to assume that self._binding has a remote UDPEndpoint'):
        s1.getpeername()

async def test_init() -> None:
    fn()
    with pytest.raises(NotImplementedError, match=re.escape(f"FakeNet doesn't (yet) support type={trio.socket.SOCK_STREAM}")):
        s1 = trio.socket.socket()
    s1 = trio.socket.socket(type=trio.socket.SOCK_DGRAM)
    assert s1.getsockname() == ('0.0.0.0', 0)
    await s1.bind(('0.0.0.0', 0))
    (ip, port) = s1.getsockname()
    assert ip == '127.0.0.1'
    assert port != 0
    s2 = trio.socket.socket(family=socket.AF_INET6, type=socket.SOCK_DGRAM)
    assert s2.getsockname() == ('::', 0)
    await s2.bind(('::', 0))
    (ip, port, *_) = s2.getsockname()
    assert ip == '::1'
    assert port != 0
    assert _ == [0, 0]