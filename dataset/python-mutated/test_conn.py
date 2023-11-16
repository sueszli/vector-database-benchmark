from __future__ import absolute_import
from errno import EALREADY, EINPROGRESS, EISCONN, ECONNRESET
import socket
import mock
import pytest
from kafka.conn import BrokerConnection, ConnectionStates, collect_hosts
from kafka.protocol.api import RequestHeader
from kafka.protocol.metadata import MetadataRequest
from kafka.protocol.produce import ProduceRequest
import kafka.errors as Errors

@pytest.fixture
def dns_lookup(mocker):
    if False:
        i = 10
        return i + 15
    return mocker.patch('kafka.conn.dns_lookup', return_value=[(socket.AF_INET, None, None, None, ('localhost', 9092))])

@pytest.fixture
def _socket(mocker):
    if False:
        i = 10
        return i + 15
    socket = mocker.MagicMock()
    socket.connect_ex.return_value = 0
    mocker.patch('socket.socket', return_value=socket)
    return socket

@pytest.fixture
def conn(_socket, dns_lookup):
    if False:
        for i in range(10):
            print('nop')
    conn = BrokerConnection('localhost', 9092, socket.AF_INET)
    return conn

@pytest.mark.parametrize('states', [(([EINPROGRESS, EALREADY], ConnectionStates.CONNECTING),), (([EALREADY, EALREADY], ConnectionStates.CONNECTING),), (([0], ConnectionStates.CONNECTED),), (([EINPROGRESS, EALREADY], ConnectionStates.CONNECTING), ([ECONNRESET], ConnectionStates.DISCONNECTED)), (([EINPROGRESS, EALREADY], ConnectionStates.CONNECTING), ([EALREADY], ConnectionStates.CONNECTING), ([EISCONN], ConnectionStates.CONNECTED))])
def test_connect(_socket, conn, states):
    if False:
        while True:
            i = 10
    assert conn.state is ConnectionStates.DISCONNECTED
    for (errno, state) in states:
        _socket.connect_ex.side_effect = errno
        conn.connect()
        assert conn.state is state

def test_connect_timeout(_socket, conn):
    if False:
        print('Hello World!')
    assert conn.state is ConnectionStates.DISCONNECTED
    _socket.connect_ex.side_effect = [EINPROGRESS, EALREADY, EALREADY, EALREADY]
    conn.connect()
    assert conn.state is ConnectionStates.CONNECTING
    conn.connect()
    assert conn.state is ConnectionStates.CONNECTING
    conn.last_attempt = 0
    conn.connect()
    assert conn.state is ConnectionStates.DISCONNECTED

def test_blacked_out(conn):
    if False:
        while True:
            i = 10
    with mock.patch('time.time', return_value=1000):
        conn.last_attempt = 0
        assert conn.blacked_out() is False
        conn.last_attempt = 1000
        assert conn.blacked_out() is True

def test_connection_delay(conn):
    if False:
        return 10
    with mock.patch('time.time', return_value=1000):
        conn.last_attempt = 1000
        assert conn.connection_delay() == conn.config['reconnect_backoff_ms']
        conn.state = ConnectionStates.CONNECTING
        assert conn.connection_delay() == float('inf')
        conn.state = ConnectionStates.CONNECTED
        assert conn.connection_delay() == float('inf')

def test_connected(conn):
    if False:
        while True:
            i = 10
    assert conn.connected() is False
    conn.state = ConnectionStates.CONNECTED
    assert conn.connected() is True

def test_connecting(conn):
    if False:
        print('Hello World!')
    assert conn.connecting() is False
    conn.state = ConnectionStates.CONNECTING
    assert conn.connecting() is True
    conn.state = ConnectionStates.CONNECTED
    assert conn.connecting() is False

def test_send_disconnected(conn):
    if False:
        print('Hello World!')
    conn.state = ConnectionStates.DISCONNECTED
    f = conn.send('foobar')
    assert f.failed() is True
    assert isinstance(f.exception, Errors.KafkaConnectionError)

def test_send_connecting(conn):
    if False:
        for i in range(10):
            print('nop')
    conn.state = ConnectionStates.CONNECTING
    f = conn.send('foobar')
    assert f.failed() is True
    assert isinstance(f.exception, Errors.NodeNotReadyError)

def test_send_max_ifr(conn):
    if False:
        return 10
    conn.state = ConnectionStates.CONNECTED
    max_ifrs = conn.config['max_in_flight_requests_per_connection']
    for i in range(max_ifrs):
        conn.in_flight_requests[i] = 'foo'
    f = conn.send('foobar')
    assert f.failed() is True
    assert isinstance(f.exception, Errors.TooManyInFlightRequests)

def test_send_no_response(_socket, conn):
    if False:
        return 10
    conn.connect()
    assert conn.state is ConnectionStates.CONNECTED
    req = ProduceRequest[0](required_acks=0, timeout=0, topics=())
    header = RequestHeader(req, client_id=conn.config['client_id'])
    payload_bytes = len(header.encode()) + len(req.encode())
    third = payload_bytes // 3
    remainder = payload_bytes % 3
    _socket.send.side_effect = [4, third, third, third, remainder]
    assert len(conn.in_flight_requests) == 0
    f = conn.send(req)
    assert f.succeeded() is True
    assert f.value is None
    assert len(conn.in_flight_requests) == 0

def test_send_response(_socket, conn):
    if False:
        print('Hello World!')
    conn.connect()
    assert conn.state is ConnectionStates.CONNECTED
    req = MetadataRequest[0]([])
    header = RequestHeader(req, client_id=conn.config['client_id'])
    payload_bytes = len(header.encode()) + len(req.encode())
    third = payload_bytes // 3
    remainder = payload_bytes % 3
    _socket.send.side_effect = [4, third, third, third, remainder]
    assert len(conn.in_flight_requests) == 0
    f = conn.send(req)
    assert f.is_done is False
    assert len(conn.in_flight_requests) == 1

def test_send_error(_socket, conn):
    if False:
        i = 10
        return i + 15
    conn.connect()
    assert conn.state is ConnectionStates.CONNECTED
    req = MetadataRequest[0]([])
    try:
        _socket.send.side_effect = ConnectionError
    except NameError:
        _socket.send.side_effect = socket.error
    f = conn.send(req)
    assert f.failed() is True
    assert isinstance(f.exception, Errors.KafkaConnectionError)
    assert _socket.close.call_count == 1
    assert conn.state is ConnectionStates.DISCONNECTED

def test_can_send_more(conn):
    if False:
        print('Hello World!')
    assert conn.can_send_more() is True
    max_ifrs = conn.config['max_in_flight_requests_per_connection']
    for i in range(max_ifrs):
        assert conn.can_send_more() is True
        conn.in_flight_requests[i] = 'foo'
    assert conn.can_send_more() is False

def test_recv_disconnected(_socket, conn):
    if False:
        while True:
            i = 10
    conn.connect()
    assert conn.connected()
    req = MetadataRequest[0]([])
    header = RequestHeader(req, client_id=conn.config['client_id'])
    payload_bytes = len(header.encode()) + len(req.encode())
    _socket.send.side_effect = [4, payload_bytes]
    conn.send(req)
    _socket.recv.return_value = b''
    assert conn.connected()
    conn.recv()
    assert conn.disconnected()

def test_recv(_socket, conn):
    if False:
        print('Hello World!')
    pass

def test_close(conn):
    if False:
        i = 10
        return i + 15
    pass

def test_collect_hosts__happy_path():
    if False:
        for i in range(10):
            print('nop')
    hosts = '127.0.0.1:1234,127.0.0.1'
    results = collect_hosts(hosts)
    assert set(results) == set([('127.0.0.1', 1234, socket.AF_INET), ('127.0.0.1', 9092, socket.AF_INET)])

def test_collect_hosts__ipv6():
    if False:
        for i in range(10):
            print('nop')
    hosts = '[localhost]:1234,[2001:1000:2000::1],[2001:1000:2000::1]:1234'
    results = collect_hosts(hosts)
    assert set(results) == set([('localhost', 1234, socket.AF_INET6), ('2001:1000:2000::1', 9092, socket.AF_INET6), ('2001:1000:2000::1', 1234, socket.AF_INET6)])

def test_collect_hosts__string_list():
    if False:
        for i in range(10):
            print('nop')
    hosts = ['localhost:1234', 'localhost', '[localhost]', '2001::1', '[2001::1]', '[2001::1]:1234']
    results = collect_hosts(hosts)
    assert set(results) == set([('localhost', 1234, socket.AF_UNSPEC), ('localhost', 9092, socket.AF_UNSPEC), ('localhost', 9092, socket.AF_INET6), ('2001::1', 9092, socket.AF_INET6), ('2001::1', 9092, socket.AF_INET6), ('2001::1', 1234, socket.AF_INET6)])

def test_collect_hosts__with_spaces():
    if False:
        print('Hello World!')
    hosts = 'localhost:1234, localhost'
    results = collect_hosts(hosts)
    assert set(results) == set([('localhost', 1234, socket.AF_UNSPEC), ('localhost', 9092, socket.AF_UNSPEC)])

def test_lookup_on_connect():
    if False:
        print('Hello World!')
    hostname = 'example.org'
    port = 9092
    conn = BrokerConnection(hostname, port, socket.AF_UNSPEC)
    assert conn.host == hostname
    assert conn.port == port
    assert conn.afi == socket.AF_UNSPEC
    afi1 = socket.AF_INET
    sockaddr1 = ('127.0.0.1', 9092)
    mock_return1 = [(afi1, socket.SOCK_STREAM, 6, '', sockaddr1)]
    with mock.patch('socket.getaddrinfo', return_value=mock_return1) as m:
        conn.connect()
        m.assert_called_once_with(hostname, port, 0, socket.SOCK_STREAM)
        assert conn._sock_afi == afi1
        assert conn._sock_addr == sockaddr1
        conn.close()
    afi2 = socket.AF_INET6
    sockaddr2 = ('::1', 9092, 0, 0)
    mock_return2 = [(afi2, socket.SOCK_STREAM, 6, '', sockaddr2)]
    with mock.patch('socket.getaddrinfo', return_value=mock_return2) as m:
        conn.last_attempt = 0
        conn.connect()
        m.assert_called_once_with(hostname, port, 0, socket.SOCK_STREAM)
        assert conn._sock_afi == afi2
        assert conn._sock_addr == sockaddr2
        conn.close()

def test_relookup_on_failure():
    if False:
        return 10
    hostname = 'example.org'
    port = 9092
    conn = BrokerConnection(hostname, port, socket.AF_UNSPEC)
    assert conn.host == hostname
    mock_return1 = []
    with mock.patch('socket.getaddrinfo', return_value=mock_return1) as m:
        last_attempt = conn.last_attempt
        conn.connect()
        m.assert_called_once_with(hostname, port, 0, socket.SOCK_STREAM)
        assert conn.disconnected()
        assert conn.last_attempt > last_attempt
    afi2 = socket.AF_INET
    sockaddr2 = ('127.0.0.2', 9092)
    mock_return2 = [(afi2, socket.SOCK_STREAM, 6, '', sockaddr2)]
    with mock.patch('socket.getaddrinfo', return_value=mock_return2) as m:
        conn.last_attempt = 0
        conn.connect()
        m.assert_called_once_with(hostname, port, 0, socket.SOCK_STREAM)
        assert conn._sock_afi == afi2
        assert conn._sock_addr == sockaddr2
        conn.close()

def test_requests_timed_out(conn):
    if False:
        return 10
    with mock.patch('time.time', return_value=0):
        assert not conn.requests_timed_out()
        conn.in_flight_requests[0] = ('foo', 0)
        assert not conn.requests_timed_out()
        request_timeout = conn.config['request_timeout_ms']
        expired_timestamp = 0 - request_timeout - 1
        conn.in_flight_requests[1] = ('bar', expired_timestamp)
        assert conn.requests_timed_out()
        conn.in_flight_requests.pop(1)
        assert not conn.requests_timed_out()