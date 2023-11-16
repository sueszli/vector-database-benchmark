from __future__ import absolute_import, division
try:
    import selectors
except ImportError:
    import kafka.vendor.selectors34 as selectors
import socket
import time
import pytest
from kafka.client_async import KafkaClient, IdleConnectionManager
from kafka.cluster import ClusterMetadata
from kafka.conn import ConnectionStates
import kafka.errors as Errors
from kafka.future import Future
from kafka.protocol.metadata import MetadataRequest
from kafka.protocol.produce import ProduceRequest
from kafka.structs import BrokerMetadata

@pytest.fixture
def cli(mocker, conn):
    if False:
        for i in range(10):
            print('nop')
    client = KafkaClient(api_version=(0, 9))
    mocker.patch.object(client, '_selector')
    client.poll(future=client.cluster.request_update())
    return client

def test_bootstrap(mocker, conn):
    if False:
        return 10
    conn.state = ConnectionStates.CONNECTED
    cli = KafkaClient(api_version=(0, 9))
    mocker.patch.object(cli, '_selector')
    future = cli.cluster.request_update()
    cli.poll(future=future)
    assert future.succeeded()
    (args, kwargs) = conn.call_args
    assert args == ('localhost', 9092, socket.AF_UNSPEC)
    kwargs.pop('state_change_callback')
    kwargs.pop('node_id')
    assert kwargs == cli.config
    conn.send.assert_called_once_with(MetadataRequest[0]([]), blocking=False)
    assert cli._bootstrap_fails == 0
    assert cli.cluster.brokers() == set([BrokerMetadata(0, 'foo', 12, None), BrokerMetadata(1, 'bar', 34, None)])

def test_can_connect(cli, conn):
    if False:
        while True:
            i = 10
    assert not cli._can_connect(2)
    assert 0 not in cli._conns
    assert cli._can_connect(0)
    assert cli._maybe_connect(0) is True
    assert not cli._can_connect(0)
    cli._conns[0].state = ConnectionStates.DISCONNECTED
    assert cli._can_connect(0)
    conn.blacked_out.return_value = True
    assert not cli._can_connect(0)

def test_maybe_connect(cli, conn):
    if False:
        while True:
            i = 10
    try:
        cli._maybe_connect(2)
    except AssertionError:
        pass
    else:
        assert False, 'Exception not raised'
    assert 0 not in cli._conns
    conn.state = ConnectionStates.DISCONNECTED
    conn.connect.side_effect = lambda : conn._set_conn_state(ConnectionStates.CONNECTING)
    assert cli._maybe_connect(0) is False
    assert cli._conns[0] is conn

def test_conn_state_change(mocker, cli, conn):
    if False:
        print('Hello World!')
    sel = cli._selector
    node_id = 0
    cli._conns[node_id] = conn
    conn.state = ConnectionStates.CONNECTING
    sock = conn._sock
    cli._conn_state_change(node_id, sock, conn)
    assert node_id in cli._connecting
    sel.register.assert_called_with(sock, selectors.EVENT_WRITE, conn)
    conn.state = ConnectionStates.CONNECTED
    cli._conn_state_change(node_id, sock, conn)
    assert node_id not in cli._connecting
    sel.modify.assert_called_with(sock, selectors.EVENT_READ, conn)
    assert cli.cluster._need_update is False
    conn.state = ConnectionStates.DISCONNECTED
    cli._conn_state_change(node_id, sock, conn)
    assert node_id not in cli._connecting
    assert cli.cluster._need_update is True
    sel.unregister.assert_called_with(sock)
    conn.state = ConnectionStates.CONNECTING
    cli._conn_state_change(node_id, sock, conn)
    assert node_id in cli._connecting
    conn.state = ConnectionStates.DISCONNECTED
    cli._conn_state_change(node_id, sock, conn)
    assert node_id not in cli._connecting

def test_ready(mocker, cli, conn):
    if False:
        return 10
    maybe_connect = mocker.patch.object(cli, 'maybe_connect')
    node_id = 1
    cli.ready(node_id)
    maybe_connect.assert_called_with(node_id)

def test_is_ready(mocker, cli, conn):
    if False:
        i = 10
        return i + 15
    cli._maybe_connect(0)
    cli._maybe_connect(1)
    assert cli.is_ready(0)
    assert cli.is_ready(1)
    cli._metadata_refresh_in_progress = True
    assert not cli.is_ready(0)
    assert not cli.is_ready(1)
    cli._metadata_refresh_in_progress = False
    assert cli.is_ready(0)
    assert cli.is_ready(1)
    cli.cluster.request_update()
    cli.cluster.config['retry_backoff_ms'] = 0
    assert not cli._metadata_refresh_in_progress
    assert not cli.is_ready(0)
    assert not cli.is_ready(1)
    cli.cluster._need_update = False
    assert cli.is_ready(0)
    conn.can_send_more.return_value = False
    assert not cli.is_ready(0)
    conn.can_send_more.return_value = True
    assert cli.is_ready(0)
    conn.state = ConnectionStates.DISCONNECTED
    assert not cli.is_ready(0)

def test_close(mocker, cli, conn):
    if False:
        for i in range(10):
            print('nop')
    mocker.patch.object(cli, '_selector')
    call_count = conn.close.call_count
    cli.close(2)
    call_count += 0
    assert conn.close.call_count == call_count
    cli._maybe_connect(0)
    assert conn.close.call_count == call_count
    cli.close(0)
    call_count += 1
    assert conn.close.call_count == call_count
    cli._maybe_connect(1)
    cli.close()
    call_count += 2
    assert conn.close.call_count == call_count

def test_is_disconnected(cli, conn):
    if False:
        i = 10
        return i + 15
    conn.state = ConnectionStates.DISCONNECTED
    assert not cli.is_disconnected(0)
    cli._maybe_connect(0)
    assert cli.is_disconnected(0)
    conn.state = ConnectionStates.CONNECTING
    assert not cli.is_disconnected(0)
    conn.state = ConnectionStates.CONNECTED
    assert not cli.is_disconnected(0)

def test_send(cli, conn):
    if False:
        return 10
    try:
        cli.send(2, None)
        assert False, 'Exception not raised'
    except AssertionError:
        pass
    conn.state = ConnectionStates.DISCONNECTED
    f = cli.send(0, None)
    assert f.failed()
    assert isinstance(f.exception, Errors.NodeNotReadyError)
    conn.state = ConnectionStates.CONNECTED
    cli._maybe_connect(0)
    request = ProduceRequest[0](0, 0, [])
    assert request.expect_response() is False
    ret = cli.send(0, request)
    conn.send.assert_called_with(request, blocking=False)
    assert isinstance(ret, Future)
    request = MetadataRequest[0]([])
    cli.send(0, request)
    conn.send.assert_called_with(request, blocking=False)

def test_poll(mocker):
    if False:
        return 10
    metadata = mocker.patch.object(KafkaClient, '_maybe_refresh_metadata')
    _poll = mocker.patch.object(KafkaClient, '_poll')
    ifrs = mocker.patch.object(KafkaClient, 'in_flight_request_count')
    ifrs.return_value = 1
    cli = KafkaClient(api_version=(0, 9))
    metadata.return_value = 1000
    cli.poll()
    _poll.assert_called_with(1.0)
    cli.poll(250)
    _poll.assert_called_with(0.25)
    metadata.return_value = 1000000
    cli.poll()
    _poll.assert_called_with(cli.config['request_timeout_ms'] / 1000.0)
    ifrs.return_value = 0
    cli.poll()
    _poll.assert_called_with(cli.config['retry_backoff_ms'] / 1000.0)

def test__poll():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_in_flight_request_count():
    if False:
        while True:
            i = 10
    pass

def test_least_loaded_node():
    if False:
        for i in range(10):
            print('nop')
    pass

def test_set_topics(mocker):
    if False:
        for i in range(10):
            print('nop')
    request_update = mocker.patch.object(ClusterMetadata, 'request_update')
    request_update.side_effect = lambda : Future()
    cli = KafkaClient(api_version=(0, 10))
    request_update.reset_mock()
    fut = cli.set_topics(['t1', 't2'])
    assert not fut.is_done
    request_update.assert_called_with()
    request_update.reset_mock()
    fut = cli.set_topics(['t1', 't2'])
    assert fut.is_done
    assert fut.value == set(['t1', 't2'])
    request_update.assert_not_called()
    request_update.reset_mock()
    fut = cli.set_topics([])
    assert fut.is_done
    assert fut.value == set()
    request_update.assert_not_called()

@pytest.fixture
def client(mocker):
    if False:
        i = 10
        return i + 15
    _poll = mocker.patch.object(KafkaClient, '_poll')
    cli = KafkaClient(request_timeout_ms=9999999, reconnect_backoff_ms=2222, connections_max_idle_ms=float('inf'), api_version=(0, 9))
    ttl = mocker.patch.object(cli.cluster, 'ttl')
    ttl.return_value = 0
    return cli

def test_maybe_refresh_metadata_ttl(mocker, client):
    if False:
        i = 10
        return i + 15
    client.cluster.ttl.return_value = 1234
    mocker.patch.object(KafkaClient, 'in_flight_request_count', return_value=1)
    client.poll(timeout_ms=12345678)
    client._poll.assert_called_with(1.234)

def test_maybe_refresh_metadata_backoff(mocker, client):
    if False:
        while True:
            i = 10
    mocker.patch.object(KafkaClient, 'in_flight_request_count', return_value=1)
    now = time.time()
    t = mocker.patch('time.time')
    t.return_value = now
    client.poll(timeout_ms=12345678)
    client._poll.assert_called_with(2.222)

def test_maybe_refresh_metadata_in_progress(mocker, client):
    if False:
        return 10
    client._metadata_refresh_in_progress = True
    mocker.patch.object(KafkaClient, 'in_flight_request_count', return_value=1)
    client.poll(timeout_ms=12345678)
    client._poll.assert_called_with(9999.999)

def test_maybe_refresh_metadata_update(mocker, client):
    if False:
        return 10
    mocker.patch.object(client, 'least_loaded_node', return_value='foobar')
    mocker.patch.object(client, '_can_send_request', return_value=True)
    mocker.patch.object(KafkaClient, 'in_flight_request_count', return_value=1)
    send = mocker.patch.object(client, 'send')
    client.poll(timeout_ms=12345678)
    client._poll.assert_called_with(9999.999)
    assert client._metadata_refresh_in_progress
    request = MetadataRequest[0]([])
    send.assert_called_once_with('foobar', request, wakeup=False)

def test_maybe_refresh_metadata_cant_send(mocker, client):
    if False:
        print('Hello World!')
    mocker.patch.object(client, 'least_loaded_node', return_value='foobar')
    mocker.patch.object(client, '_can_connect', return_value=True)
    mocker.patch.object(client, '_maybe_connect', return_value=True)
    mocker.patch.object(client, 'maybe_connect', return_value=True)
    mocker.patch.object(KafkaClient, 'in_flight_request_count', return_value=1)
    now = time.time()
    t = mocker.patch('time.time')
    t.return_value = now
    client.poll(timeout_ms=12345678)
    client._poll.assert_called_with(2.222)
    client.maybe_connect.assert_called_once_with('foobar', wakeup=False)
    client._connecting.add('foobar')
    client._can_connect.reset_mock()
    client.poll(timeout_ms=12345678)
    client._poll.assert_called_with(2.222)
    assert not client._can_connect.called
    assert not client._metadata_refresh_in_progress

def test_schedule():
    if False:
        while True:
            i = 10
    pass

def test_unschedule():
    if False:
        return 10
    pass

def test_idle_connection_manager(mocker):
    if False:
        for i in range(10):
            print('nop')
    t = mocker.patch.object(time, 'time')
    t.return_value = 0
    idle = IdleConnectionManager(100)
    assert idle.next_check_ms() == float('inf')
    idle.update('foo')
    assert not idle.is_expired('foo')
    assert idle.poll_expired_connection() is None
    assert idle.next_check_ms() == 100
    t.return_value = 90 / 1000
    assert not idle.is_expired('foo')
    assert idle.poll_expired_connection() is None
    assert idle.next_check_ms() == 10
    t.return_value = 100 / 1000
    assert idle.is_expired('foo')
    assert idle.next_check_ms() == 0
    (conn_id, conn_ts) = idle.poll_expired_connection()
    assert conn_id == 'foo'
    assert conn_ts == 0
    idle.remove('foo')
    assert idle.next_check_ms() == float('inf')