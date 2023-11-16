import uuid
import eventlet
import pytest
from eventlet.event import Event
from mock import ANY, Mock, call, patch
from nameko.events import event_handler
from nameko.rpc import RpcProxy, rpc
from nameko.standalone.events import event_dispatcher
from nameko.standalone.rpc import ServiceRpcProxy
from nameko.testing.services import dummy, entrypoint_hook
from nameko.testing.utils import assert_stops_raising, get_rabbit_connections, reset_rabbit_connections
disconnect_now = Event()
disconnected = Event()
method_called = Mock()
handle_called = Mock()
long_called = Event()

@pytest.fixture(autouse=True)
def reset():
    if False:
        for i in range(10):
            print('nop')
    yield
    method_called.reset_mock()
    handle_called.reset_mock()
    for event in (disconnect_now, disconnected):
        if event.ready():
            event.reset()

@pytest.fixture
def logger():
    if False:
        i = 10
        return i + 15
    with patch('nameko.rpc._log', autospec=True) as patched:
        yield patched
    patched.reset_mock()

class ExampleService(object):
    name = 'exampleservice'

    @rpc
    def echo(self, arg):
        if False:
            print('Hello World!')
        return arg

    @rpc
    def method(self, arg):
        if False:
            for i in range(10):
                print('nop')
        method_called(arg)
        if not disconnect_now.ready():
            disconnect_now.send(True)
            disconnected.wait()
            return arg
        return 'duplicate-call-result'

    @event_handler('srcservice', 'exampleevent')
    def handle(self, evt_data):
        if False:
            while True:
                i = 10
        handle_called(evt_data)
        if not disconnect_now.ready():
            disconnect_now.send(True)
            disconnected.wait()

class ProxyService(object):
    name = 'proxyservice'
    example_rpc = RpcProxy('exampleservice')

    @dummy
    def entrypoint(self, arg):
        if False:
            print('Hello World!')
        return self.example_rpc.method(arg)

def disconnect_on_event(rabbit_manager, connection_name):
    if False:
        i = 10
        return i + 15
    disconnect_now.wait()
    rabbit_manager.delete_connection(connection_name)
    disconnected.send(True)

def test_idle_disconnect(container_factory, rabbit_manager, rabbit_config):
    if False:
        return 10
    ' Break the connection to rabbit while a service is started but idle\n    (i.e. without active workers)\n    '
    container = container_factory(ExampleService, rabbit_config)
    container.start()
    vhost = rabbit_config['vhost']
    reset_rabbit_connections(vhost, rabbit_manager)
    with ServiceRpcProxy('exampleservice', rabbit_config) as proxy:
        assert proxy.echo('hello') == 'hello'

def test_proxy_disconnect_with_active_worker(container_factory, rabbit_manager, rabbit_config):
    if False:
        print('Hello World!')
    " Break the connection to rabbit while a service's queue consumer and\n    rabbit while the service has an in-flight rpc request (i.e. it is waiting\n    on a reply).\n    "
    proxy_container = container_factory(ProxyService, rabbit_config)
    example_container = container_factory(ExampleService, rabbit_config)
    proxy_container.start()
    vhost = rabbit_config['vhost']
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 1
    proxy_consumer_conn = connections[0]['name']
    example_container.start()
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 2
    eventlet.spawn(disconnect_on_event, rabbit_manager, proxy_consumer_conn)
    with entrypoint_hook(proxy_container, 'entrypoint') as entrypoint:
        assert entrypoint('hello') == 'hello'
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert proxy_consumer_conn not in [conn['name'] for conn in connections]

def test_service_disconnect_with_active_async_worker(container_factory, rabbit_manager, rabbit_config):
    if False:
        return 10
    " Break the connection between a service's queue consumer and rabbit\n    while the service has an active async worker (e.g. event handler).\n    "
    container = container_factory(ExampleService, rabbit_config)
    container.start()
    vhost = rabbit_config['vhost']
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 1
    queue_consumer_conn = connections[0]['name']
    eventlet.spawn(disconnect_on_event, rabbit_manager, queue_consumer_conn)
    data = uuid.uuid4().hex
    dispatch = event_dispatcher(rabbit_config)
    dispatch('srcservice', 'exampleevent', data)

    def event_handled_twice():
        if False:
            while True:
                i = 10
        assert handle_called.call_args_list == [call(data), call(data)]
    assert_stops_raising(event_handled_twice)
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert queue_consumer_conn not in [conn['name'] for conn in connections]

def test_service_disconnect_with_active_rpc_worker(container_factory, rabbit_manager, rabbit_config):
    if False:
        print('Hello World!')
    " Break the connection between a service's queue consumer and rabbit\n    while the service has an active rpc worker (i.e. response required).\n    "
    container = container_factory(ExampleService, rabbit_config)
    container.start()
    vhost = rabbit_config['vhost']
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 1
    queue_consumer_conn = connections[0]['name']
    rpc_proxy = ServiceRpcProxy('exampleservice', rabbit_config)
    proxy = rpc_proxy.start()
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 2
    eventlet.spawn(disconnect_on_event, rabbit_manager, queue_consumer_conn)
    arg = uuid.uuid4().hex
    assert proxy.method(arg) == arg

    def method_called_twice():
        if False:
            i = 10
            return i + 15
        assert method_called.call_args_list == [call(arg), call(arg)]
    assert_stops_raising(method_called_twice)
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert queue_consumer_conn not in [conn['name'] for conn in connections]
    rpc_proxy.stop()

def test_service_disconnect_with_active_rpc_worker_via_service_proxy(logger, container_factory, rabbit_manager, rabbit_config):
    if False:
        for i in range(10):
            print('nop')
    " Break the connection between a service's queue consumer and rabbit\n    while the service has an active rpc worker (i.e. response required).\n\n    Make the rpc call from a nameko service. We expect the service to see\n    the duplicate response and discard it.\n    "
    proxy_container = container_factory(ProxyService, rabbit_config)
    service_container = container_factory(ExampleService, rabbit_config)
    service_container.start()
    vhost = rabbit_config['vhost']
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 1
    service_consumer_conn = connections[0]['name']
    proxy_container.start()
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert len(connections) == 2
    eventlet.spawn(disconnect_on_event, rabbit_manager, service_consumer_conn)
    arg = uuid.uuid4().hex
    with entrypoint_hook(proxy_container, 'entrypoint') as entrypoint:
        assert entrypoint(arg) == arg

    def duplicate_response_received():
        if False:
            return 10
        correlation_warning = call('Unknown correlation id: %s', ANY)
        assert correlation_warning in logger.debug.call_args_list
    assert_stops_raising(duplicate_response_received)

    def method_called_twice():
        if False:
            print('Hello World!')
        assert method_called.call_args_list == [call(arg), call(arg)]
    assert_stops_raising(method_called_twice)
    connections = get_rabbit_connections(vhost, rabbit_manager)
    assert service_consumer_conn not in [conn['name'] for conn in connections]