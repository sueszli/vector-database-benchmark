import socket
from unittest import mock
import pytest
from aiohttp.tcp_helpers import tcp_nodelay
has_ipv6: bool = socket.has_ipv6
if has_ipv6:
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_STREAM):
            pass
    except OSError:
        has_ipv6 = False

def test_tcp_nodelay_exception() -> None:
    if False:
        while True:
            i = 10
    transport = mock.Mock()
    s = mock.Mock()
    s.setsockopt = mock.Mock()
    s.family = socket.AF_INET
    s.setsockopt.side_effect = OSError
    transport.get_extra_info.return_value = s
    tcp_nodelay(transport, True)
    s.setsockopt.assert_called_with(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)

def test_tcp_nodelay_enable() -> None:
    if False:
        for i in range(10):
            print('nop')
    transport = mock.Mock()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        transport.get_extra_info.return_value = s
        tcp_nodelay(transport, True)
        assert s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

def test_tcp_nodelay_enable_and_disable() -> None:
    if False:
        for i in range(10):
            print('nop')
    transport = mock.Mock()
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        transport.get_extra_info.return_value = s
        tcp_nodelay(transport, True)
        assert s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)
        tcp_nodelay(transport, False)
        assert not s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

@pytest.mark.skipif(not has_ipv6, reason='IPv6 is not available')
def test_tcp_nodelay_enable_ipv6() -> None:
    if False:
        while True:
            i = 10
    transport = mock.Mock()
    with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s:
        transport.get_extra_info.return_value = s
        tcp_nodelay(transport, True)
        assert s.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY)

@pytest.mark.skipif(not hasattr(socket, 'AF_UNIX'), reason='requires unix sockets')
def test_tcp_nodelay_enable_unix() -> None:
    if False:
        for i in range(10):
            print('nop')
    transport = mock.Mock()
    s = mock.Mock(family=socket.AF_UNIX, type=socket.SOCK_STREAM)
    transport.get_extra_info.return_value = s
    tcp_nodelay(transport, True)
    assert not s.setsockopt.called

def test_tcp_nodelay_enable_no_socket() -> None:
    if False:
        print('Hello World!')
    transport = mock.Mock()
    transport.get_extra_info.return_value = None
    tcp_nodelay(transport, True)