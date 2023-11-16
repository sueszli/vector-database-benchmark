from __future__ import annotations
import socket
import sys
from socket import AddressFamily, SocketKind
from typing import TYPE_CHECKING, Any, Sequence
import attr
import pytest
import trio
from trio._highlevel_open_tcp_stream import close_all, format_host_port, open_tcp_stream, reorder_for_rfc_6555_section_5_4
from trio.socket import AF_INET, AF_INET6, IPPROTO_TCP, SOCK_STREAM, SocketType
if TYPE_CHECKING:
    from trio.testing import MockClock
if sys.version_info < (3, 11):
    from exceptiongroup import BaseExceptionGroup

def test_close_all() -> None:
    if False:
        print('Hello World!')

    class CloseMe(SocketType):
        closed = False

        def close(self) -> None:
            if False:
                return 10
            self.closed = True

    class CloseKiller(SocketType):

        def close(self) -> None:
            if False:
                print('Hello World!')
            raise OSError
    c: CloseMe = CloseMe()
    with close_all() as to_close:
        to_close.add(c)
    assert c.closed
    c = CloseMe()
    with pytest.raises(RuntimeError):
        with close_all() as to_close:
            to_close.add(c)
            raise RuntimeError
    assert c.closed
    c = CloseMe()
    with pytest.raises(OSError):
        with close_all() as to_close:
            to_close.add(CloseKiller())
            to_close.add(c)
    assert c.closed

def test_reorder_for_rfc_6555_section_5_4() -> None:
    if False:
        return 10

    def fake4(i: int) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
        if False:
            print('Hello World!')
        return (AF_INET, SOCK_STREAM, IPPROTO_TCP, '', (f'10.0.0.{i}', 80))

    def fake6(i: int) -> tuple[socket.AddressFamily, socket.SocketKind, int, str, tuple[str, int]]:
        if False:
            while True:
                i = 10
        return (AF_INET6, SOCK_STREAM, IPPROTO_TCP, '', (f'::{i}', 80))
    for fake in (fake4, fake6):
        targets = [fake(0), fake(1), fake(2)]
        reorder_for_rfc_6555_section_5_4(targets)
        assert targets == [fake(0), fake(1), fake(2)]
        targets = [fake(0)]
        reorder_for_rfc_6555_section_5_4(targets)
        assert targets == [fake(0)]
    orig = [fake4(0), fake6(0), fake4(1), fake6(1)]
    targets = list(orig)
    reorder_for_rfc_6555_section_5_4(targets)
    assert targets == orig
    targets = [fake4(0), fake4(1), fake4(2), fake6(0), fake6(1)]
    reorder_for_rfc_6555_section_5_4(targets)
    assert targets == [fake4(0), fake6(0), fake4(1), fake4(2), fake6(1)]

def test_format_host_port() -> None:
    if False:
        while True:
            i = 10
    assert format_host_port('127.0.0.1', 80) == '127.0.0.1:80'
    assert format_host_port(b'127.0.0.1', 80) == '127.0.0.1:80'
    assert format_host_port('example.com', 443) == 'example.com:443'
    assert format_host_port(b'example.com', 443) == 'example.com:443'
    assert format_host_port('::1', 'http') == '[::1]:http'
    assert format_host_port(b'::1', 'http') == '[::1]:http'

async def test_open_tcp_stream_real_socket_smoketest() -> None:
    listen_sock = trio.socket.socket()
    await listen_sock.bind(('127.0.0.1', 0))
    (_, listen_port) = listen_sock.getsockname()
    listen_sock.listen(1)
    client_stream = await open_tcp_stream('127.0.0.1', listen_port)
    (server_sock, _) = await listen_sock.accept()
    await client_stream.send_all(b'x')
    assert await server_sock.recv(1) == b'x'
    await client_stream.aclose()
    server_sock.close()
    listen_sock.close()

async def test_open_tcp_stream_input_validation() -> None:
    with pytest.raises(ValueError):
        await open_tcp_stream(None, 80)
    with pytest.raises(TypeError):
        await open_tcp_stream('127.0.0.1', b'80')

def can_bind_127_0_0_2() -> bool:
    if False:
        return 10
    with socket.socket() as s:
        try:
            s.bind(('127.0.0.2', 0))
        except OSError:
            return False
        return s.getsockname()[0] == '127.0.0.2'

async def test_local_address_real() -> None:
    with trio.socket.socket() as listener:
        await listener.bind(('127.0.0.1', 0))
        listener.listen()
        local_address = '127.0.0.2' if can_bind_127_0_0_2() else '127.0.0.1'
        async with await open_tcp_stream(*listener.getsockname(), local_address=local_address) as client_stream:
            assert client_stream.socket.getsockname()[0] == local_address
            if hasattr(trio.socket, 'IP_BIND_ADDRESS_NO_PORT'):
                assert client_stream.socket.getsockopt(trio.socket.IPPROTO_IP, trio.socket.IP_BIND_ADDRESS_NO_PORT)
            (server_sock, remote_addr) = await listener.accept()
            await client_stream.aclose()
            server_sock.close()
            assert remote_addr[0] == local_address
        with pytest.raises(OSError):
            await open_tcp_stream(*listener.getsockname(), local_address='::')
        async with await open_tcp_stream(*listener.getsockname(), local_address='0.0.0.0') as client_stream:
            (server_sock, remote_addr) = await listener.accept()
            server_sock.close()
            assert remote_addr == client_stream.socket.getsockname()

@attr.s(eq=False)
class FakeSocket(trio.socket.SocketType):
    scenario: Scenario = attr.ib()
    _family: AddressFamily = attr.ib()
    _type: SocketKind = attr.ib()
    _proto: int = attr.ib()
    ip: str | int | None = attr.ib(default=None)
    port: str | int | None = attr.ib(default=None)
    succeeded: bool = attr.ib(default=False)
    closed: bool = attr.ib(default=False)
    failing: bool = attr.ib(default=False)

    @property
    def type(self) -> SocketKind:
        if False:
            for i in range(10):
                print('nop')
        return self._type

    @property
    def family(self) -> AddressFamily:
        if False:
            while True:
                i = 10
        return self._family

    @property
    def proto(self) -> int:
        if False:
            while True:
                i = 10
        return self._proto

    async def connect(self, sockaddr: tuple[str | int, str | int | None]) -> None:
        self.ip = sockaddr[0]
        self.port = sockaddr[1]
        assert self.ip not in self.scenario.sockets
        self.scenario.sockets[self.ip] = self
        self.scenario.connect_times[self.ip] = trio.current_time()
        (delay, result) = self.scenario.ip_dict[self.ip]
        await trio.sleep(delay)
        if result == 'error':
            raise OSError('sorry')
        if result == 'postconnect_fail':
            self.failing = True
        self.succeeded = True

    def close(self) -> None:
        if False:
            print('Hello World!')
        self.closed = True

    def setsockopt(self, *args: object, **kwargs: object) -> None:
        if False:
            return 10
        if self.failing:
            raise KeyboardInterrupt

class Scenario(trio.abc.SocketFactory, trio.abc.HostnameResolver):

    def __init__(self, port: int, ip_list: Sequence[tuple[str, float, str]], supported_families: set[AddressFamily]) -> None:
        if False:
            print('Hello World!')
        ip_order = [ip for (ip, _, _) in ip_list]
        assert len(set(ip_order)) == len(ip_list)
        ip_dict: dict[str | int, tuple[float, str]] = {}
        for (ip, delay, result) in ip_list:
            assert delay >= 0
            assert result in ['error', 'success', 'postconnect_fail']
            ip_dict[ip] = (delay, result)
        self.port = port
        self.ip_order = ip_order
        self.ip_dict = ip_dict
        self.supported_families = supported_families
        self.socket_count = 0
        self.sockets: dict[str | int, FakeSocket] = {}
        self.connect_times: dict[str | int, float] = {}

    def socket(self, family: AddressFamily | int | None=None, type: SocketKind | int | None=None, proto: int | None=None) -> SocketType:
        if False:
            i = 10
            return i + 15
        assert isinstance(family, AddressFamily)
        assert isinstance(type, SocketKind)
        assert proto is not None
        if family not in self.supported_families:
            raise OSError('pretending not to support this family')
        self.socket_count += 1
        return FakeSocket(self, family, type, proto)

    def _ip_to_gai_entry(self, ip: str) -> tuple[AddressFamily, SocketKind, int, str, tuple[str, int, int, int] | tuple[str, int]]:
        if False:
            print('Hello World!')
        sockaddr: tuple[str, int] | tuple[str, int, int, int]
        if ':' in ip:
            family = trio.socket.AF_INET6
            sockaddr = (ip, self.port, 0, 0)
        else:
            family = trio.socket.AF_INET
            sockaddr = (ip, self.port)
        return (family, SOCK_STREAM, IPPROTO_TCP, '', sockaddr)

    async def getaddrinfo(self, host: str | bytes | None, port: bytes | str | int | None, family: int=-1, type: int=-1, proto: int=-1, flags: int=-1) -> list[tuple[AddressFamily, SocketKind, int, str, tuple[str, int, int, int] | tuple[str, int]]]:
        assert host == b'test.example.com'
        assert port == self.port
        assert family == trio.socket.AF_UNSPEC
        assert type == trio.socket.SOCK_STREAM
        assert proto == 0
        assert flags == 0
        return [self._ip_to_gai_entry(ip) for ip in self.ip_order]

    async def getnameinfo(self, sockaddr: tuple[str, int] | tuple[str, int, int, int], flags: int) -> tuple[str, str]:
        raise NotImplementedError

    def check(self, succeeded: SocketType | None) -> None:
        if False:
            i = 10
            return i + 15
        assert self.socket_count == len(self.sockets)
        for (ip, socket_) in self.sockets.items():
            assert ip in self.ip_dict
            if socket_ is not succeeded:
                assert socket_.closed
            assert socket_.port == self.port

async def run_scenario(port: int, ip_list: Sequence[tuple[str, float, str]], *, ipv4_supported: bool=True, ipv6_supported: bool=True, expect_error: tuple[type[BaseException], ...] | type[BaseException]=(), **kwargs: Any) -> tuple[SocketType, Scenario] | tuple[BaseException, Scenario]:
    supported_families = set()
    if ipv4_supported:
        supported_families.add(trio.socket.AF_INET)
    if ipv6_supported:
        supported_families.add(trio.socket.AF_INET6)
    scenario = Scenario(port, ip_list, supported_families)
    trio.socket.set_custom_hostname_resolver(scenario)
    trio.socket.set_custom_socket_factory(scenario)
    try:
        stream = await open_tcp_stream('test.example.com', port, **kwargs)
        assert expect_error == ()
        scenario.check(stream.socket)
        return (stream.socket, scenario)
    except AssertionError:
        raise
    except expect_error as exc:
        scenario.check(None)
        return (exc, scenario)

async def test_one_host_quick_success(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.2.3.4', 0.123, 'success')])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '1.2.3.4'
    assert trio.current_time() == 0.123

async def test_one_host_slow_success(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(81, [('1.2.3.4', 100, 'success')])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '1.2.3.4'
    assert trio.current_time() == 100

async def test_one_host_quick_fail(autojump_clock: MockClock) -> None:
    (exc, scenario) = await run_scenario(82, [('1.2.3.4', 0.123, 'error')], expect_error=OSError)
    assert isinstance(exc, OSError)
    assert trio.current_time() == 0.123

async def test_one_host_slow_fail(autojump_clock: MockClock) -> None:
    (exc, scenario) = await run_scenario(83, [('1.2.3.4', 100, 'error')], expect_error=OSError)
    assert isinstance(exc, OSError)
    assert trio.current_time() == 100

async def test_one_host_failed_after_connect(autojump_clock: MockClock) -> None:
    (exc, scenario) = await run_scenario(83, [('1.2.3.4', 1, 'postconnect_fail')], expect_error=KeyboardInterrupt)
    assert isinstance(exc, KeyboardInterrupt)

async def test_basic_fallthrough(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.1.1.1', 1, 'success'), ('2.2.2.2', 1, 'success'), ('3.3.3.3', 0.2, 'success')])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '3.3.3.3'
    assert trio.current_time() == 0.25 + 0.25 + 0.2
    assert scenario.connect_times == {'1.1.1.1': 0, '2.2.2.2': 0.25, '3.3.3.3': 0.5}

async def test_early_success(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.1.1.1', 1, 'success'), ('2.2.2.2', 0.1, 'success'), ('3.3.3.3', 0.2, 'success')])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '2.2.2.2'
    assert trio.current_time() == 0.25 + 0.1
    assert scenario.connect_times == {'1.1.1.1': 0, '2.2.2.2': 0.25}

async def test_custom_delay(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.1.1.1', 1, 'success'), ('2.2.2.2', 1, 'success'), ('3.3.3.3', 0.2, 'success')], happy_eyeballs_delay=0.45)
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '1.1.1.1'
    assert trio.current_time() == 1
    assert scenario.connect_times == {'1.1.1.1': 0, '2.2.2.2': 0.45, '3.3.3.3': 0.9}

async def test_custom_errors_expedite(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.1.1.1', 0.1, 'error'), ('2.2.2.2', 0.2, 'error'), ('3.3.3.3', 10, 'success'), ('4.4.4.4', 0.25, 'success')])
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '4.4.4.4'
    assert trio.current_time() == 0.1 + 0.2 + 0.25 + 0.25
    assert scenario.connect_times == {'1.1.1.1': 0, '2.2.2.2': 0.1, '3.3.3.3': 0.1 + 0.2, '4.4.4.4': 0.1 + 0.2 + 0.25}

async def test_all_fail(autojump_clock: MockClock) -> None:
    (exc, scenario) = await run_scenario(80, [('1.1.1.1', 0.1, 'error'), ('2.2.2.2', 0.2, 'error'), ('3.3.3.3', 10, 'error'), ('4.4.4.4', 0.25, 'error')], expect_error=OSError)
    assert isinstance(exc, OSError)
    assert isinstance(exc.__cause__, BaseExceptionGroup)
    assert len(exc.__cause__.exceptions) == 4
    assert trio.current_time() == 0.1 + 0.2 + 10
    assert scenario.connect_times == {'1.1.1.1': 0, '2.2.2.2': 0.1, '3.3.3.3': 0.1 + 0.2, '4.4.4.4': 0.1 + 0.2 + 0.25}

async def test_multi_success(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.1.1.1', 0.5, 'error'), ('2.2.2.2', 10, 'success'), ('3.3.3.3', 10 - 1, 'success'), ('4.4.4.4', 10 - 2, 'success'), ('5.5.5.5', 0.5, 'error')], happy_eyeballs_delay=1)
    assert not scenario.sockets['1.1.1.1'].succeeded
    assert scenario.sockets['2.2.2.2'].succeeded or scenario.sockets['3.3.3.3'].succeeded or scenario.sockets['4.4.4.4'].succeeded
    assert not scenario.sockets['5.5.5.5'].succeeded
    assert isinstance(sock, FakeSocket)
    assert sock.ip in ['2.2.2.2', '3.3.3.3', '4.4.4.4']
    assert trio.current_time() == 0.5 + 10
    assert scenario.connect_times == {'1.1.1.1': 0, '2.2.2.2': 0.5, '3.3.3.3': 1.5, '4.4.4.4': 2.5, '5.5.5.5': 3.5}

async def test_does_reorder(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('1.1.1.1', 10, 'error'), ('2.2.2.2', 1, 'success'), ('::3', 0.5, 'success')], happy_eyeballs_delay=1)
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '::3'
    assert trio.current_time() == 1 + 0.5
    assert scenario.connect_times == {'1.1.1.1': 0, '::3': 1}

async def test_handles_no_ipv4(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('::1', 10, 'success'), ('2.2.2.2', 0, 'success'), ('::3', 0.1, 'success'), ('4.4.4.4', 0, 'success')], happy_eyeballs_delay=1, ipv4_supported=False)
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '::3'
    assert trio.current_time() == 1 + 0.1
    assert scenario.connect_times == {'::1': 0, '::3': 1.0}

async def test_handles_no_ipv6(autojump_clock: MockClock) -> None:
    (sock, scenario) = await run_scenario(80, [('::1', 0, 'success'), ('2.2.2.2', 10, 'success'), ('::3', 0, 'success'), ('4.4.4.4', 0.1, 'success')], happy_eyeballs_delay=1, ipv6_supported=False)
    assert isinstance(sock, FakeSocket)
    assert sock.ip == '4.4.4.4'
    assert trio.current_time() == 1 + 0.1
    assert scenario.connect_times == {'2.2.2.2': 0, '4.4.4.4': 1.0}

async def test_no_hosts(autojump_clock: MockClock) -> None:
    (exc, scenario) = await run_scenario(80, [], expect_error=OSError)
    assert 'no results found' in str(exc)

async def test_cancel(autojump_clock: MockClock) -> None:
    with trio.move_on_after(5) as cancel_scope:
        (exc, scenario) = await run_scenario(80, [('1.1.1.1', 10, 'success'), ('2.2.2.2', 10, 'success'), ('3.3.3.3', 10, 'success'), ('4.4.4.4', 10, 'success')], expect_error=BaseExceptionGroup)
        assert isinstance(exc, BaseException)
        raise exc
    assert cancel_scope.cancelled_caught
    assert trio.current_time() == 5
    scenario.check(succeeded=None)