import itertools
import socket
import time
import uuid
import eventlet
import pytest
from eventlet.event import Event
from kombu import Connection, Exchange, Queue
from kombu.exceptions import TimeoutError
from mock import ANY, Mock, call, patch
from nameko.constants import AMQP_URI_CONFIG_KEY, DEFAULT_HEARTBEAT, HEARTBEAT_CONFIG_KEY
from nameko.events import event_handler
from nameko.messaging import Consumer, QueueConsumer
from nameko.rpc import RpcProxy, rpc
from nameko.standalone.events import event_dispatcher
from nameko.standalone.rpc import ServiceRpcProxy
from nameko.testing.services import entrypoint_waiter
from nameko.testing.utils import assert_stops_raising, get_extension, get_rabbit_connections
from nameko.utils.retry import retry
TIMEOUT = 5
exchange = Exchange('spam')
ham_queue = Queue('ham', exchange=exchange, auto_delete=False)

@pytest.fixture
def logger():
    if False:
        while True:
            i = 10
    with patch('nameko.messaging._log') as logger:
        yield logger

class MessageHandler(object):
    queue = ham_queue

    def __init__(self):
        if False:
            while True:
                i = 10
        self.handle_message_called = Event()

    def handle_message(self, body, message):
        if False:
            return 10
        self.handle_message_called.send(message)

    def wait(self):
        if False:
            print('Hello World!')
        return self.handle_message_called.wait()

def spawn_managed_thread(method, identifier=None):
    if False:
        while True:
            i = 10
    return eventlet.spawn(method)

def test_lifecycle(rabbit_manager, rabbit_config, mock_container):
    if False:
        print('Hello World!')
    container = mock_container
    container.shared_extensions = {}
    container.config = rabbit_config
    container.max_workers = 3
    container.spawn_managed_thread.side_effect = spawn_managed_thread
    content_type = 'application/data'
    container.accept = [content_type]
    queue_consumer = QueueConsumer().bind(container)
    handler = MessageHandler()
    queue_consumer.register_provider(handler)
    queue_consumer.setup()
    queue_consumer.start()
    container.spawn_managed_thread.assert_called_once_with(ANY)
    vhost = rabbit_config['vhost']
    rabbit_manager.publish(vhost, 'spam', '', 'shrub', properties=dict(content_type=content_type))
    message = handler.wait()
    gt = eventlet.spawn(queue_consumer.unregister_provider, handler)
    with eventlet.Timeout(TIMEOUT):
        while len(queue_consumer._consumers):
            eventlet.sleep()
    assert not gt.dead
    rabbit_manager.publish(vhost, 'spam', '', 'ni')
    queue_consumer.ack_message(message)
    with eventlet.Timeout(TIMEOUT):
        gt.wait()
    messages = rabbit_manager.get_messages(vhost, 'ham')
    assert ['ni'] == [msg['payload'] for msg in messages]
    queue_consumer.kill()

def test_reentrant_start_stops(mock_container):
    if False:
        while True:
            i = 10
    container = mock_container
    container.shared_extensions = {}
    container.config = {AMQP_URI_CONFIG_KEY: 'memory://'}
    container.max_workers = 3
    container.spawn_managed_thread = spawn_managed_thread
    queue_consumer = QueueConsumer().bind(container)
    queue_consumer.setup()
    queue_consumer.start()
    gt = queue_consumer._gt
    queue_consumer.start()
    assert gt is queue_consumer._gt
    queue_consumer.kill()

def test_stop_while_starting(rabbit_config, mock_container):
    if False:
        i = 10
        return i + 15
    started = Event()
    container = mock_container
    container.shared_extensions = {}
    container.config = rabbit_config
    container.max_workers = 3
    container.spawn_managed_thread = spawn_managed_thread

    class BrokenConnConsumer(QueueConsumer):

        def consume(self, *args, **kwargs):
            if False:
                return 10
            started.send(None)
            started.reset()
            return super(BrokenConnConsumer, self).consume(*args, **kwargs)
    queue_consumer = BrokenConnConsumer().bind(container)
    queue_consumer.setup()
    handler = MessageHandler()
    queue_consumer.register_provider(handler)
    with eventlet.Timeout(TIMEOUT):
        with patch.object(Connection, 'connect', autospec=True) as connect:
            connect.side_effect = TimeoutError('test')
            gt = eventlet.spawn(queue_consumer.start)
            started.wait()
    with eventlet.Timeout(TIMEOUT):
        queue_consumer.unregister_provider(handler)
        queue_consumer.stop()
    with eventlet.Timeout(TIMEOUT):
        while not gt.dead:
            eventlet.sleep()
        assert queue_consumer._gt.dead

def test_error_stops_consumer_thread(mock_container):
    if False:
        return 10
    container = mock_container
    container.shared_extensions = {}
    container.config = {AMQP_URI_CONFIG_KEY: 'memory://'}
    container.max_workers = 3
    container.spawn_managed_thread = spawn_managed_thread
    queue_consumer = QueueConsumer().bind(container)
    queue_consumer.setup()
    handler = MessageHandler()
    queue_consumer.register_provider(handler)
    with eventlet.Timeout(TIMEOUT):
        with patch.object(Connection, 'drain_events', autospec=True) as drain_events:
            drain_events.side_effect = Exception('test')
            queue_consumer.start()
    with pytest.raises(Exception) as exc_info:
        queue_consumer._gt.wait()
    assert exc_info.value.args == ('test',)

def test_on_consume_error_kills_consumer(mock_container):
    if False:
        i = 10
        return i + 15
    container = mock_container
    container.shared_extensions = {}
    container.config = {AMQP_URI_CONFIG_KEY: 'memory://'}
    container.max_workers = 1
    container.spawn_managed_thread = spawn_managed_thread
    queue_consumer = QueueConsumer().bind(container)
    queue_consumer.setup()
    handler = MessageHandler()
    queue_consumer.register_provider(handler)
    with patch.object(queue_consumer, 'on_consume_ready') as on_consume_ready:
        on_consume_ready.side_effect = Exception('err')
        queue_consumer.start()
        with pytest.raises(Exception):
            queue_consumer._gt.wait()

def test_reconnect_on_socket_error(rabbit_config, mock_container):
    if False:
        i = 10
        return i + 15
    container = mock_container
    container.shared_extensions = {}
    container.config = rabbit_config
    container.max_workers = 1
    container.spawn_managed_thread = spawn_managed_thread
    connection_revived = Mock()
    queue_consumer = QueueConsumer().bind(container)
    queue_consumer.setup()
    queue_consumer.on_connection_revived = connection_revived
    handler = MessageHandler()
    queue_consumer.register_provider(handler)
    queue_consumer.start()
    with patch.object(Connection, 'drain_events', autospec=True) as drain_events:
        drain_events.side_effect = socket.error('test-error')

        def check_reconnected():
            if False:
                i = 10
                return i + 15
            assert connection_revived.call_count > 1
        assert_stops_raising(check_reconnected)
    queue_consumer.unregister_provider(handler)
    queue_consumer.stop()

def test_prefetch_count(rabbit_manager, rabbit_config, container_factory):
    if False:
        while True:
            i = 10

    class NonShared(QueueConsumer):

        @property
        def sharing_key(self):
            if False:
                i = 10
                return i + 15
            return uuid.uuid4()
    messages = []

    class SelfishConsumer1(Consumer):
        queue_consumer = NonShared()

        def handle_message(self, body, message):
            if False:
                return 10
            consumer_continue.wait()
            super(SelfishConsumer1, self).handle_message(body, message)

    class SelfishConsumer2(Consumer):
        queue_consumer = NonShared()

        def handle_message(self, body, message):
            if False:
                print('Hello World!')
            messages.append(body)
            super(SelfishConsumer2, self).handle_message(body, message)

    class Service(object):
        name = 'service'

        @SelfishConsumer1.decorator(queue=ham_queue)
        @SelfishConsumer2.decorator(queue=ham_queue)
        def handle(self, payload):
            if False:
                i = 10
                return i + 15
            pass
    rabbit_config['max_workers'] = 1
    container = container_factory(Service, rabbit_config)
    container.start()
    consumer_continue = Event()

    def wait_for_expected(worker_ctx, res, exc_info):
        if False:
            print('Hello World!')
        return {'m3', 'm4', 'm5'}.issubset(set(messages))
    with entrypoint_waiter(container, 'handle', callback=wait_for_expected):
        vhost = rabbit_config['vhost']
        properties = {'content_type': 'application/data'}
        for message in ('m1', 'm2', 'm3', 'm4', 'm5'):
            rabbit_manager.publish(vhost, 'spam', '', message, properties=properties)
    assert messages[-3:] == ['m3', 'm4', 'm5']
    consumer_continue.send(None)

def test_kill_closes_connections(rabbit_manager, rabbit_config, mock_container):
    if False:
        print('Hello World!')
    container = mock_container
    container.shared_extensions = {}
    container.config = rabbit_config
    container.max_workers = 1
    container.spawn_managed_thread = spawn_managed_thread
    queue_consumer = QueueConsumer().bind(container)
    queue_consumer.setup()

    class Handler(object):
        queue = ham_queue

        def handle_message(self, body, message):
            if False:
                while True:
                    i = 10
            pass
    queue_consumer.register_provider(Handler())
    queue_consumer.start()
    queue_consumer.kill()
    vhost = rabbit_config['vhost']

    @retry
    def check_connections_closed():
        if False:
            for i in range(10):
                print('nop')
        connections = get_rabbit_connections(vhost, rabbit_manager)
        if connections:
            for connection in connections:
                assert connection['vhost'] != vhost
    check_connections_closed()

class TestHeartbeats(object):

    @pytest.fixture
    def service_cls(self):
        if False:
            return 10

        class Service(object):
            name = 'service'

            @rpc
            def echo(self, arg):
                if False:
                    i = 10
                    return i + 15
                return arg
        return Service

    def test_default(self, service_cls, container_factory, rabbit_config):
        if False:
            return 10
        container = container_factory(service_cls, rabbit_config)
        container.start()
        queue_consumer = get_extension(container, QueueConsumer)
        assert queue_consumer.connection.heartbeat == DEFAULT_HEARTBEAT

    @pytest.mark.parametrize('heartbeat', [30, None])
    def test_config_value(self, heartbeat, service_cls, container_factory, rabbit_config):
        if False:
            while True:
                i = 10
        rabbit_config[HEARTBEAT_CONFIG_KEY] = heartbeat
        container = container_factory(service_cls, rabbit_config)
        container.start()
        queue_consumer = get_extension(container, QueueConsumer)
        assert queue_consumer.connection.heartbeat == heartbeat

class TestDeadlockRegression(object):
    """ Regression test for https://github.com/nameko/nameko/issues/428
    """

    @pytest.fixture
    def config(self, rabbit_config):
        if False:
            return 10
        config = rabbit_config.copy()
        config['max_workers'] = 2
        return config

    @pytest.fixture
    def upstream(self, container_factory, config):
        if False:
            return 10

        class Service(object):
            name = 'upstream'

            @rpc
            def method(self):
                if False:
                    while True:
                        i = 10
                time.sleep(0.5)
        container = container_factory(Service, config)
        container.start()

    @pytest.fixture
    def service_cls(self):
        if False:
            return 10

        class Service(object):
            name = 'downsteam'
            upstream_rpc = RpcProxy('upstream')

            @event_handler('service', 'event1')
            def handle_event1(self, event_data):
                if False:
                    return 10
                self.upstream_rpc.method()

            @event_handler('service', 'event2')
            def handle_event2(self, event_data):
                if False:
                    i = 10
                    return i + 15
                self.upstream_rpc.method()
        return Service

    @pytest.mark.usefixtures('upstream')
    def test_deadlock_due_to_slow_workers(self, service_cls, container_factory, config):
        if False:
            return 10
        " Deadlock will occur if the unack'd messages grows beyond the\n        size of the worker pool at any point. The QueueConsumer will block\n        waiting for a worker and pending RPC replies will not be ack'd.\n        Any running workers therefore never complete, and the worker pool\n        remains exhausted.\n        "
        container = container_factory(service_cls, config)
        container.start()
        count = 2
        dispatch = event_dispatcher(config)
        for _ in range(count):
            dispatch('service', 'event1', 1)
            dispatch('service', 'event2', 1)
        counter = itertools.count(start=1)

        def cb(worker_ctx, res, exc_info):
            if False:
                for i in range(10):
                    print('nop')
            if next(counter) == count:
                return True
        with entrypoint_waiter(container, 'handle_event1', timeout=5, callback=cb):
            pass

def test_greenthread_raise_in_kill(container_factory, rabbit_config, logger):
    if False:
        for i in range(10):
            print('nop')

    class Service(object):
        name = 'service'

        @rpc
        def echo(self, arg):
            if False:
                return 10
            return arg
    container = container_factory(Service, rabbit_config)
    queue_consumer = get_extension(container, QueueConsumer)
    exc = Exception('error cancelling consumers')
    with patch.object(queue_consumer, '_cancel_consumers_if_requested') as cancel_consumers:
        cancel_consumers.side_effect = exc
        container.start()
        with ServiceRpcProxy('service', rabbit_config) as service_rpc:
            service_rpc.echo.call_async('foo')
    with pytest.raises(Exception) as exc_info:
        container.wait()
    assert str(exc_info.value) == str(exc)
    assert logger.warn.call_args_list == [call('QueueConsumer %s raised `%s` during kill', queue_consumer, exc)]