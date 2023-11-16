import socket
from unittest.mock import MagicMock
import pytest as pytest
from localstack import config
from localstack.constants import LOCALHOST
from localstack.testing.pytest import markers
from localstack.utils import net
from localstack.utils.common import short_uid
from localstack.utils.net import Port, PortNotAvailableException, PortRange, dynamic_port_range, get_addressable_container_host, get_free_tcp_port, get_free_tcp_port_range, get_free_udp_port, is_ip_address, port_can_be_bound, resolve_hostname

@markers.skip_offline
def test_resolve_hostname():
    if False:
        return 10
    assert '127.' in resolve_hostname(LOCALHOST)
    assert resolve_hostname('example.com')
    assert resolve_hostname(f'non-existing-host-{short_uid()}') is None

@pytest.mark.parametrize('protocol', ['tcp', 'udp'])
def test_port_open(protocol):
    if False:
        for i in range(10):
            print('nop')
    if protocol == 'tcp':
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    else:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('', 0))
    (addr, port) = sock.getsockname()
    port = Port(port, protocol=protocol)
    assert not port_can_be_bound(port)
    sock.close()
    assert port_can_be_bound(port)

def test_get_free_udp_port():
    if False:
        return 10
    port = get_free_udp_port()
    assert port_can_be_bound(Port(port, 'udp'))

def test_free_tcp_port_blocklist_raises_exception():
    if False:
        i = 10
        return i + 15
    blocklist = range(0, 70000)
    with pytest.raises(Exception) as ctx:
        get_free_tcp_port(blocklist)
    assert 'Unable to determine free TCP' in str(ctx.value)

def test_port_can_be_bound():
    if False:
        while True:
            i = 10
    port = get_free_tcp_port()
    assert port_can_be_bound(port)

def test_port_can_be_bound_illegal_port():
    if False:
        return 10
    assert not port_can_be_bound(9999999999)

def test_port_can_be_bound_already_bound():
    if False:
        print('Hello World!')
    tcp = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        tcp.bind(('', 0))
        (addr, port) = tcp.getsockname()
        assert not port_can_be_bound(port)
    finally:
        tcp.close()
    assert port_can_be_bound(port)

def test_get_free_tcp_port_range():
    if False:
        i = 10
        return i + 15
    port_range = get_free_tcp_port_range(20)
    assert len(port_range) == 20
    for port in port_range:
        assert dynamic_port_range.is_port_reserved(port)
    for port in port_range:
        assert port_can_be_bound(port)

def test_get_free_tcp_port_range_fails_if_reserved(monkeypatch):
    if False:
        print('Hello World!')
    mock = MagicMock()
    mock.return_value = True
    monkeypatch.setattr(dynamic_port_range, 'is_port_reserved', mock)
    with pytest.raises(PortNotAvailableException):
        get_free_tcp_port_range(20)
    assert mock.call_count == 50

def test_get_free_tcp_port_range_fails_if_cannot_be_bound(monkeypatch):
    if False:
        print('Hello World!')
    mock = MagicMock()
    mock.return_value = False
    monkeypatch.setattr(net, 'port_can_be_bound', mock)
    with pytest.raises(PortNotAvailableException):
        get_free_tcp_port_range(20, max_attempts=10)
    assert mock.call_count == 10

def test_port_range_iter():
    if False:
        return 10
    ports = PortRange(10, 13)
    assert list(ports) == [10, 11, 12, 13]

def test_get_addressable_container_host(monkeypatch):
    if False:
        i = 10
        return i + 15
    if not config.is_in_docker:
        monkeypatch.setattr(config, 'is_in_docker', True)
        monkeypatch.setattr(config, 'in_docker', lambda : True)
        assert is_ip_address(get_addressable_container_host())
    monkeypatch.setattr(config, 'is_in_docker', False)
    monkeypatch.setattr(config, 'in_docker', lambda : False)
    assert get_addressable_container_host(default_local_hostname='test.abc') == 'test.abc'