"""
Provides core messaging decorators and dependency providers.
"""
from __future__ import absolute_import
import warnings
from functools import partial
from logging import getLogger
import six
from amqp.exceptions import ConnectionError
from eventlet.event import Event
from kombu import Connection
from kombu.common import maybe_declare
from kombu.mixins import ConsumerMixin
from nameko.amqp.publish import Publisher as PublisherCore
from nameko.amqp.publish import get_connection
from nameko.constants import AMQP_SSL_CONFIG_KEY, AMQP_URI_CONFIG_KEY, DEFAULT_HEARTBEAT, DEFAULT_TRANSPORT_OPTIONS, HEADER_PREFIX, HEARTBEAT_CONFIG_KEY, LOGIN_METHOD_CONFIG_KEY, TRANSPORT_OPTIONS_CONFIG_KEY
from nameko.exceptions import ContainerBeingKilled
from nameko.extensions import DependencyProvider, Entrypoint, ProviderCollector, SharedExtension
from nameko.utils import sanitize_url
_log = getLogger(__name__)

class HeaderEncoder(object):
    header_prefix = HEADER_PREFIX

    def _get_header_name(self, key):
        if False:
            return 10
        return '{}.{}'.format(self.header_prefix, key)

    def get_message_headers(self, worker_ctx):
        if False:
            for i in range(10):
                print('nop')
        data = worker_ctx.context_data
        if None in data.values():
            warnings.warn('Attempted to publish unserialisable header value. Headers with a value of `None` will be dropped from the payload.', UserWarning)
        headers = {self._get_header_name(key): value for (key, value) in data.items() if value is not None}
        return headers

class HeaderDecoder(object):
    header_prefix = HEADER_PREFIX

    def _strip_header_name(self, key):
        if False:
            print('Hello World!')
        full_prefix = '{}.'.format(self.header_prefix)
        if key.startswith(full_prefix):
            return key[len(full_prefix):]
        return key

    def unpack_message_headers(self, message):
        if False:
            i = 10
            return i + 15
        stripped = {self._strip_header_name(k): v for (k, v) in six.iteritems(message.headers)}
        return stripped

class Publisher(DependencyProvider, HeaderEncoder):
    publisher_cls = PublisherCore

    def __init__(self, exchange=None, queue=None, declare=None, **options):
        if False:
            return 10
        " Provides an AMQP message publisher method via dependency injection.\n\n        In AMQP, messages are published to *exchanges* and routed to bound\n        *queues*. This dependency accepts the `exchange` to publish to and\n        will ensure that it is declared before publishing.\n\n        Optionally, you may use the `declare` keyword argument to pass a list\n        of other :class:`kombu.Exchange` or :class:`kombu.Queue` objects to\n        declare before publishing.\n\n        :Parameters:\n            exchange : :class:`kombu.Exchange`\n                Destination exchange\n            queue : :class:`kombu.Queue`\n                **Deprecated**: Bound queue. The event will be published to\n                this queue's exchange.\n            declare : list\n                List of :class:`kombu.Exchange` or :class:`kombu.Queue` objects\n                to declare before publishing.\n\n        If `exchange` is not provided, the message will be published to the\n        default exchange.\n\n        Example::\n\n            class Foobar(object):\n\n                publish = Publisher(exchange=...)\n\n                def spam(self, data):\n                    self.publish('spam:' + data)\n        "
        self.exchange = exchange
        self.options = options
        self.declare = declare[:] if declare is not None else []
        if self.exchange:
            self.declare.append(self.exchange)
        if queue is not None:
            warnings.warn('The signature of `Publisher` has changed. The `queue` kwarg is now deprecated. You can use the `declare` kwarg to provide a list of Kombu queues to be declared. See CHANGES, version 2.7.0 for more details. This warning will be removed in version 2.9.0.', DeprecationWarning)
            if exchange is None:
                self.exchange = queue.exchange
            self.declare.append(queue)
        compat_attrs = ('retry', 'retry_policy', 'use_confirms')
        for compat_attr in compat_attrs:
            if hasattr(self, compat_attr):
                warnings.warn("'{}' should be specified at instantiation time rather than as a class attribute. See CHANGES, version 2.7.0 for more details. This warning will be removed in version 2.9.0.".format(compat_attr), DeprecationWarning)
                self.options[compat_attr] = getattr(self, compat_attr)

    @property
    def amqp_uri(self):
        if False:
            i = 10
            return i + 15
        return self.container.config[AMQP_URI_CONFIG_KEY]

    @property
    def serializer(self):
        if False:
            while True:
                i = 10
        ' Default serializer to use when publishing messages.\n\n        Must be registered as a\n        `kombu serializer <http://bit.do/kombu_serialization>`_.\n        '
        return self.container.serializer

    def setup(self):
        if False:
            i = 10
            return i + 15
        ssl = self.container.config.get(AMQP_SSL_CONFIG_KEY)
        login_method = self.container.config.get(LOGIN_METHOD_CONFIG_KEY)
        with get_connection(self.amqp_uri, ssl) as conn:
            for entity in self.declare:
                maybe_declare(entity, conn.channel())
        serializer = self.options.pop('serializer', self.serializer)
        self.publisher = self.publisher_cls(self.amqp_uri, serializer=serializer, exchange=self.exchange, declare=self.declare, ssl=ssl, login_method=login_method, **self.options)

    def get_dependency(self, worker_ctx):
        if False:
            print('Hello World!')
        extra_headers = self.get_message_headers(worker_ctx)

        def publish(msg, **kwargs):
            if False:
                print('Hello World!')
            self.publisher.publish(msg, extra_headers=extra_headers, **kwargs)
        return publish

class QueueConsumer(SharedExtension, ProviderCollector, ConsumerMixin):

    def __init__(self):
        if False:
            return 10
        self._consumers = {}
        self._pending_remove_providers = {}
        self._gt = None
        self._starting = False
        self._consumers_ready = Event()
        super(QueueConsumer, self).__init__()

    @property
    def amqp_uri(self):
        if False:
            i = 10
            return i + 15
        return self.container.config[AMQP_URI_CONFIG_KEY]

    @property
    def prefetch_count(self):
        if False:
            print('Hello World!')
        return self.container.max_workers

    @property
    def accept(self):
        if False:
            i = 10
            return i + 15
        return self.container.accept

    def _handle_thread_exited(self, gt):
        if False:
            print('Hello World!')
        exc = None
        try:
            gt.wait()
        except Exception as e:
            exc = e
        if not self._consumers_ready.ready():
            self._consumers_ready.send_exception(exc)

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        if not self._starting:
            self._starting = True
            _log.debug('starting %s', self)
            self._gt = self.container.spawn_managed_thread(self.run)
            self._gt.link(self._handle_thread_exited)
        try:
            _log.debug('waiting for consumer ready %s', self)
            self._consumers_ready.wait()
        except QueueConsumerStopped:
            _log.debug('consumer was stopped before it started %s', self)
        except Exception as exc:
            _log.debug('consumer failed to start %s (%s)', self, exc)
        else:
            _log.debug('started %s', self)

    def stop(self):
        if False:
            i = 10
            return i + 15
        " Stop the queue-consumer gracefully.\n\n        Wait until the last provider has been unregistered and for\n        the ConsumerMixin's greenthread to exit (i.e. until all pending\n        messages have been acked or requeued and all consumers stopped).\n        "
        if not self._consumers_ready.ready():
            _log.debug('stopping while consumer is starting %s', self)
            stop_exc = QueueConsumerStopped()
            self._gt.kill(stop_exc)
        self.wait_for_providers()
        try:
            _log.debug('waiting for consumer death %s', self)
            self._gt.wait()
        except QueueConsumerStopped:
            pass
        super(QueueConsumer, self).stop()
        _log.debug('stopped %s', self)

    def kill(self):
        if False:
            while True:
                i = 10
        ' Kill the queue-consumer.\n\n        Unlike `stop()` any pending message ack or requeue-requests,\n        requests to remove providers, etc are lost and the consume thread is\n        asked to terminate as soon as possible.\n        '
        if self._gt is not None and (not self._gt.dead):
            self._providers = set()
            self._pending_remove_providers = {}
            self.should_stop = True
            try:
                self._gt.wait()
            except Exception as exc:
                _log.warn('QueueConsumer %s raised `%s` during kill', self, exc)
            super(QueueConsumer, self).kill()
            _log.debug('killed %s', self)

    def unregister_provider(self, provider):
        if False:
            for i in range(10):
                print('nop')
        if not self._consumers_ready.ready():
            self._last_provider_unregistered.send()
            return
        removed_event = Event()
        self._pending_remove_providers[provider] = removed_event
        removed_event.wait()
        super(QueueConsumer, self).unregister_provider(provider)

    def ack_message(self, message):
        if False:
            i = 10
            return i + 15
        if message.channel.connection:
            try:
                message.ack()
            except ConnectionError:
                pass

    def requeue_message(self, message):
        if False:
            return 10
        if message.channel.connection:
            try:
                message.requeue()
            except ConnectionError:
                pass

    def _cancel_consumers_if_requested(self):
        if False:
            print('Hello World!')
        provider_remove_events = self._pending_remove_providers.items()
        self._pending_remove_providers = {}
        for (provider, removed_event) in provider_remove_events:
            consumer = self._consumers.pop(provider)
            _log.debug('cancelling consumer [%s]: %s', provider, consumer)
            consumer.cancel()
            removed_event.send()

    @property
    def connection(self):
        if False:
            i = 10
            return i + 15
        " Provide the connection parameters for kombu's ConsumerMixin.\n\n        The `Connection` object is a declaration of connection parameters\n        that is lazily evaluated. It doesn't represent an established\n        connection to the broker at this point.\n        "
        heartbeat = self.container.config.get(HEARTBEAT_CONFIG_KEY, DEFAULT_HEARTBEAT)
        transport_options = self.container.config.get(TRANSPORT_OPTIONS_CONFIG_KEY, DEFAULT_TRANSPORT_OPTIONS)
        ssl = self.container.config.get(AMQP_SSL_CONFIG_KEY)
        login_method = self.container.config.get(LOGIN_METHOD_CONFIG_KEY)
        conn = Connection(self.amqp_uri, transport_options=transport_options, heartbeat=heartbeat, ssl=ssl, login_method=login_method)
        return conn

    def handle_message(self, provider, body, message):
        if False:
            return 10
        ident = u'{}.handle_message[{}]'.format(type(provider).__name__, message.delivery_info['routing_key'])
        self.container.spawn_managed_thread(partial(provider.handle_message, body, message), identifier=ident)

    def get_consumers(self, consumer_cls, channel):
        if False:
            for i in range(10):
                print('nop')
        ' Kombu callback to set up consumers.\n\n        Called after any (re)connection to the broker.\n        '
        _log.debug('setting up consumers %s', self)
        for provider in self._providers:
            callbacks = [partial(self.handle_message, provider)]
            consumer = consumer_cls(queues=[provider.queue], callbacks=callbacks, accept=self.accept)
            consumer.qos(prefetch_count=self.prefetch_count)
            self._consumers[provider] = consumer
        return self._consumers.values()

    def on_iteration(self):
        if False:
            i = 10
            return i + 15
        ' Kombu callback for each `drain_events` loop iteration.'
        self._cancel_consumers_if_requested()
        if len(self._consumers) == 0:
            _log.debug('requesting stop after iteration')
            self.should_stop = True

    def on_connection_error(self, exc, interval):
        if False:
            for i in range(10):
                print('nop')
        _log.warning('Error connecting to broker at {} ({}).\nRetrying in {} seconds.'.format(sanitize_url(self.amqp_uri), exc, interval))

    def on_consume_ready(self, connection, channel, consumers, **kwargs):
        if False:
            return 10
        ' Kombu callback when consumers are ready to accept messages.\n\n        Called after any (re)connection to the broker.\n        '
        if not self._consumers_ready.ready():
            _log.debug('consumer started %s', self)
            self._consumers_ready.send(None)

class Consumer(Entrypoint, HeaderDecoder):
    queue_consumer = QueueConsumer()

    def __init__(self, queue, requeue_on_error=False, **kwargs):
        if False:
            print('Hello World!')
        "\n        Decorates a method as a message consumer.\n\n        Messages from the queue will be deserialized depending on their content\n        type and passed to the the decorated method.\n        When the consumer method returns without raising any exceptions,\n        the message will automatically be acknowledged.\n        If any exceptions are raised during the consumption and\n        `requeue_on_error` is True, the message will be requeued.\n\n        If `requeue_on_error` is true, handlers will return the event to the\n        queue if an error occurs while handling it. Defaults to false.\n\n        Example::\n\n            @consume(...)\n            def handle_message(self, body):\n\n                if not self.spam(body):\n                    raise Exception('message will be requeued')\n\n                self.shrub(body)\n\n        Args:\n            queue: The queue to consume from.\n        "
        self.queue = queue
        self.requeue_on_error = requeue_on_error
        super(Consumer, self).__init__(**kwargs)

    def setup(self):
        if False:
            for i in range(10):
                print('nop')
        self.queue_consumer.register_provider(self)

    def stop(self):
        if False:
            print('Hello World!')
        self.queue_consumer.unregister_provider(self)

    def handle_message(self, body, message):
        if False:
            print('Hello World!')
        args = (body,)
        kwargs = {}
        context_data = self.unpack_message_headers(message)
        handle_result = partial(self.handle_result, message)
        try:
            self.container.spawn_worker(self, args, kwargs, context_data=context_data, handle_result=handle_result)
        except ContainerBeingKilled:
            self.queue_consumer.requeue_message(message)

    def handle_result(self, message, worker_ctx, result=None, exc_info=None):
        if False:
            return 10
        self.handle_message_processed(message, result, exc_info)
        return (result, exc_info)

    def handle_message_processed(self, message, result=None, exc_info=None):
        if False:
            print('Hello World!')
        if exc_info is not None and self.requeue_on_error:
            self.queue_consumer.requeue_message(message)
        else:
            self.queue_consumer.ack_message(message)
consume = Consumer.decorator

class QueueConsumerStopped(Exception):
    pass