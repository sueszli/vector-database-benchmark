import pytest
from tribler.core.components.socks_servers.socks5.udp_connection import SocksUDPConnection

@pytest.fixture
async def connection():
    connection = SocksUDPConnection(None, ('1.1.1.1', 1234))
    await connection.open()
    yield connection
    connection.close()

def test_datagram_received(connection):
    if False:
        print('Hello World!')
    '\n    Test whether the right operations happen when a datagram is received\n    '
    assert not connection.datagram_received(b'aaa\x04', ('1.1.1.1', 1234))
    assert not connection.datagram_received(b'aa\x01aaa', ('1.1.1.1', 1234))
    assert not connection.datagram_received(b'aaaaaa', ('1.2.3.4', 1234))
    invalid_udp_packet = b'\x00\x00\x00\x03\x1etracker1.invalid-tracker\xc4\xe95\x11$\x00\x1f\x940x000'
    assert not connection.datagram_received(invalid_udp_packet, ('1.1.1.1', 1234))

def test_send_diagram(connection):
    if False:
        i = 10
        return i + 15
    '\n    Test sending a diagram over the SOCKS5 UDP connection\n    '
    assert connection.send_datagram(b'a')
    connection.remote_udp_address = None
    assert not connection.send_datagram(b'a')