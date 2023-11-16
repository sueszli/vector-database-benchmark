import logging
import os
import time
import warnings
from functools import partial
from itertools import chain
from threading import Event, local
import pika
from ..broker import Broker, Consumer, MessageProxy
from ..common import current_millis, dq_name, q_name, xq_name
from ..errors import ConnectionClosed, DecodeError, QueueJoinTimeout
from ..logging import get_logger
from ..message import Message, get_encoder
DEAD_MESSAGE_TTL = int(os.getenv('dramatiq_dead_message_ttl', 86400000 * 7))
MAX_ENQUEUE_ATTEMPTS = 6
MAX_DECLARE_ATTEMPTS = 2

class RabbitmqBroker(Broker):
    """A broker that can be used with RabbitMQ.

    Examples:

      If you want to specify connection parameters individually:

      >>> RabbitmqBroker(host="127.0.0.1", port=5672)

      Alternatively, if you want to use a connection URL:

      >>> RabbitmqBroker(url="amqp://guest:guest@127.0.0.1:5672")

      To support message priorities, provide a ``max_priority``...

      >>> broker = RabbitmqBroker(url="...", max_priority=255)

      ... then enqueue messages with the ``broker_priority`` option:

      >>> broker.enqueue(an_actor.message_with_options(
      ...    broker_priority=255,
      ... ))

    See also:
      ConnectionParameters_ for a list of all the available connection
      parameters.

    Parameters:
      confirm_delivery(bool): Wait for RabbitMQ to confirm that
        messages have been committed on every call to enqueue.
        Defaults to False.
      url(str|list[str]): An optional connection URL.  If both a URL
        and connection parameters are provided, the URL is used.
      middleware(list[Middleware]): The set of middleware that apply
        to this broker.
      max_priority(int): Configure queues with ``x-max-priority`` to
        support queue-global priority queueing.
      parameters(list[dict]): A sequence of (pika) connection parameters
        to determine which Rabbit server(s) to connect to.
      **kwargs: The (pika) connection parameters to use to
        determine which Rabbit server to connect to.

    .. _ConnectionParameters: https://pika.readthedocs.io/en/0.12.0/modules/parameters.html
    """

    def __init__(self, *, confirm_delivery=False, url=None, middleware=None, max_priority=None, parameters=None, **kwargs):
        if False:
            return 10
        super().__init__(middleware=middleware)
        if max_priority is not None and (not 0 < max_priority <= 255):
            raise ValueError('max_priority must be a value between 0 and 255')
        if url is not None:
            if parameters is not None or kwargs:
                raise RuntimeError("the 'url' argument cannot be used in conjunction with pika parameters")
            if isinstance(url, str) and ';' in url:
                self.parameters = [pika.URLParameters(u) for u in url.split(';')]
            elif isinstance(url, list):
                self.parameters = [pika.URLParameters(u) for u in url]
            else:
                self.parameters = pika.URLParameters(url)
        elif parameters is not None:
            if kwargs:
                raise RuntimeError("the 'parameters' argument cannot be used in conjunction with other pika parameters")
            self.parameters = [pika.ConnectionParameters(**p) for p in parameters]
        else:
            self.parameters = pika.ConnectionParameters(**kwargs)
        self.confirm_delivery = confirm_delivery
        self.max_priority = max_priority
        self.connections = set()
        self.channels = set()
        self.queues = set()
        self.queues_pending = set()
        self.state = local()

    @property
    def consumer_class(self):
        if False:
            for i in range(10):
                print('nop')
        return _RabbitmqConsumer

    @property
    def connection(self):
        if False:
            while True:
                i = 10
        'The :class:`pika.BlockingConnection` for the current\n        thread.  This property may change without notice.\n        '
        connection = getattr(self.state, 'connection', None)
        if connection is None:
            connection = self.state.connection = pika.BlockingConnection(parameters=self.parameters)
            self.connections.add(connection)
        return connection

    @connection.deleter
    def connection(self):
        if False:
            while True:
                i = 10
        del self.channel
        try:
            connection = self.state.connection
        except AttributeError:
            return
        del self.state.connection
        self.connections.remove(connection)
        if connection.is_open:
            try:
                connection.close()
            except Exception:
                self.logger.exception('Encountered exception while closing Connection.')

    @property
    def channel(self):
        if False:
            for i in range(10):
                print('nop')
        'The :class:`pika.BlockingChannel` for the current thread.\n        This property may change without notice.\n        '
        channel = getattr(self.state, 'channel', None)
        if channel is None:
            channel = self.state.channel = self.connection.channel()
            if self.confirm_delivery:
                channel.confirm_delivery()
            self.channels.add(channel)
        return channel

    @channel.deleter
    def channel(self):
        if False:
            while True:
                i = 10
        try:
            channel = self.state.channel
        except AttributeError:
            return
        del self.state.channel
        self.channels.remove(channel)
        if channel.is_open:
            try:
                channel.close()
            except Exception:
                self.logger.exception('Encountered exception while closing Channel.')

    def close(self):
        if False:
            print('Hello World!')
        'Close all open RabbitMQ connections.\n        '
        logging_filter = _IgnoreScaryLogs()
        logging.getLogger('pika.adapters.base_connection').addFilter(logging_filter)
        logging.getLogger('pika.adapters.blocking_connection').addFilter(logging_filter)
        self.logger.debug('Closing channels and connections...')
        for channel_or_conn in chain(self.channels, self.connections):
            try:
                channel_or_conn.close()
            except pika.exceptions.AMQPError:
                pass
            except Exception:
                self.logger.debug('Encountered an error while closing %r.', channel_or_conn, exc_info=True)
        self.logger.debug('Channels and connections closed.')

    def consume(self, queue_name, prefetch=1, timeout=5000):
        if False:
            while True:
                i = 10
        'Create a new consumer for a queue.\n\n        Parameters:\n          queue_name(str): The queue to consume.\n          prefetch(int): The number of messages to prefetch.\n          timeout(int): The idle timeout in milliseconds.\n\n        Returns:\n          Consumer: A consumer that retrieves messages from RabbitMQ.\n        '
        self.declare_queue(queue_name, ensure=True)
        return self.consumer_class(self.parameters, queue_name, prefetch, timeout)

    def declare_queue(self, queue_name, *, ensure=False):
        if False:
            return 10
        'Declare a queue.  Has no effect if a queue with the given\n        name already exists.\n\n        Parameters:\n          queue_name(str): The name of the new queue.\n          ensure(bool): When True, the queue is created immediately on\n            the server.\n\n        Raises:\n          ConnectionClosed: When ensure=True if the underlying channel\n            or connection fails.\n        '
        if q_name(queue_name) not in self.queues:
            self.emit_before('declare_queue', queue_name)
            self.queues.add(queue_name)
            self.queues_pending.add(queue_name)
            self.emit_after('declare_queue', queue_name)
            delayed_name = dq_name(queue_name)
            self.delay_queues.add(delayed_name)
            self.emit_after('declare_delay_queue', delayed_name)
        if ensure:
            self._ensure_queue(queue_name)

    def _ensure_queue(self, queue_name):
        if False:
            for i in range(10):
                print('nop')
        attempts = 1
        while True:
            try:
                if queue_name in self.queues_pending:
                    self._declare_queue(queue_name)
                    self._declare_dq_queue(queue_name)
                    self._declare_xq_queue(queue_name)
                    self.queues_pending.discard(queue_name)
                break
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
                del self.connection
                attempts += 1
                if attempts > MAX_DECLARE_ATTEMPTS:
                    raise ConnectionClosed(e) from None
                self.logger.debug('Retrying declare due to closed connection. [%d/%d]', attempts, MAX_DECLARE_ATTEMPTS)

    def _build_queue_arguments(self, queue_name):
        if False:
            for i in range(10):
                print('nop')
        arguments = {'x-dead-letter-exchange': '', 'x-dead-letter-routing-key': xq_name(queue_name)}
        if self.max_priority:
            arguments['x-max-priority'] = self.max_priority
        return arguments

    def _declare_queue(self, queue_name):
        if False:
            while True:
                i = 10
        arguments = self._build_queue_arguments(queue_name)
        return self.channel.queue_declare(queue=queue_name, durable=True, arguments=arguments)

    def _declare_dq_queue(self, queue_name):
        if False:
            i = 10
            return i + 15
        arguments = self._build_queue_arguments(queue_name)
        return self.channel.queue_declare(queue=dq_name(queue_name), durable=True, arguments=arguments)

    def _declare_xq_queue(self, queue_name):
        if False:
            print('Hello World!')
        return self.channel.queue_declare(queue=xq_name(queue_name), durable=True, arguments={'x-message-ttl': DEAD_MESSAGE_TTL})

    def enqueue(self, message, *, delay=None):
        if False:
            return 10
        'Enqueue a message.\n\n        Parameters:\n          message(Message): The message to enqueue.\n          delay(int): The minimum amount of time, in milliseconds, to\n            delay the message by.\n\n        Raises:\n          ConnectionClosed: If the underlying channel or connection\n            has been closed.\n        '
        queue_name = message.queue_name
        self.declare_queue(queue_name, ensure=True)
        if delay is not None:
            queue_name = dq_name(queue_name)
            message_eta = current_millis() + delay
            message = message.copy(queue_name=queue_name, options={'eta': message_eta})
        attempts = 1
        while True:
            try:
                self.logger.debug('Enqueueing message %r on queue %r.', message.message_id, queue_name)
                self.emit_before('enqueue', message, delay)
                self.channel.basic_publish(exchange='', routing_key=queue_name, body=message.encode(), properties=pika.BasicProperties(delivery_mode=2, priority=message.options.get('broker_priority')))
                self.emit_after('enqueue', message, delay)
                return message
            except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
                del self.connection
                attempts += 1
                if attempts > MAX_ENQUEUE_ATTEMPTS:
                    raise ConnectionClosed(e) from None
                self.logger.debug('Retrying enqueue due to closed connection. [%d/%d]', attempts, MAX_ENQUEUE_ATTEMPTS)

    def get_declared_queues(self):
        if False:
            return 10
        'Get all declared queues.\n\n        Returns:\n          set[str]: The names of all the queues declared so far on\n          this Broker.\n        '
        return self.queues.copy()

    def get_queue_message_counts(self, queue_name):
        if False:
            i = 10
            return i + 15
        'Get the number of messages in a queue.  This method is only\n        meant to be used in unit and integration tests.\n\n        Parameters:\n          queue_name(str): The queue whose message counts to get.\n\n        Returns:\n          tuple: A triple representing the number of messages in the\n          queue, its delayed queue and its dead letter queue.\n        '
        queue_response = self._declare_queue(queue_name)
        dq_queue_response = self._declare_dq_queue(queue_name)
        xq_queue_response = self._declare_xq_queue(queue_name)
        return (queue_response.method.message_count, dq_queue_response.method.message_count, xq_queue_response.method.message_count)

    def flush(self, queue_name):
        if False:
            print('Hello World!')
        'Drop all the messages from a queue.\n\n        Parameters:\n          queue_name(str): The queue to flush.\n        '
        for name in (queue_name, dq_name(queue_name), xq_name(queue_name)):
            if queue_name not in self.queues_pending:
                self.channel.queue_purge(name)

    def flush_all(self):
        if False:
            for i in range(10):
                print('nop')
        'Drop all messages from all declared queues.\n        '
        for queue_name in self.queues:
            self.flush(queue_name)

    def join(self, queue_name, min_successes=10, idle_time=100, *, timeout=None):
        if False:
            print('Hello World!')
        "Wait for all the messages on the given queue to be\n        processed.  This method is only meant to be used in tests to\n        wait for all the messages in a queue to be processed.\n\n        Warning:\n          This method doesn't wait for unacked messages so it may not\n          be completely reliable.  Use the stub broker in your unit\n          tests and only use this for simple integration tests.\n\n        Parameters:\n          queue_name(str): The queue to wait on.\n          min_successes(int): The minimum number of times all the\n            polled queues should be empty.\n          idle_time(int): The number of milliseconds to wait between\n            counts.\n          timeout(Optional[int]): The max amount of time, in\n            milliseconds, to wait on this queue.\n        "
        deadline = timeout and time.monotonic() + timeout / 1000
        successes = 0
        while successes < min_successes:
            if deadline and time.monotonic() >= deadline:
                raise QueueJoinTimeout(queue_name)
            total_messages = sum(self.get_queue_message_counts(queue_name)[:-1])
            if total_messages == 0:
                successes += 1
            else:
                successes = 0
            self.connection.sleep(idle_time / 1000)

def URLRabbitmqBroker(url, *, middleware=None):
    if False:
        while True:
            i = 10
    'Alias for the RabbitMQ broker that takes a connection URL as a\n    positional argument.\n\n    Parameters:\n      url(str): A connection string.\n      middleware(list[Middleware]): The middleware to add to this\n        broker.\n    '
    warnings.warn("Use RabbitmqBroker with the 'url' parameter instead of URLRabbitmqBroker.", DeprecationWarning, stacklevel=2)
    return RabbitmqBroker(url=url, middleware=middleware)

class _IgnoreScaryLogs(logging.Filter):

    def filter(self, record):
        if False:
            return 10
        return 'Broken pipe' not in record.getMessage()

class _RabbitmqConsumer(Consumer):

    def __init__(self, parameters, queue_name, prefetch, timeout):
        if False:
            print('Hello World!')
        try:
            self.logger = get_logger(__name__, type(self))
            self.connection = pika.BlockingConnection(parameters=parameters)
            self.channel = self.connection.channel()
            self.channel.basic_qos(prefetch_count=prefetch)
            self.iterator = self.channel.consume(queue_name, inactivity_timeout=timeout / 1000)
            self.known_tags = set()
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
            raise ConnectionClosed(e) from None

    def ack(self, message):
        if False:
            i = 10
            return i + 15
        try:
            self.known_tags.remove(message._tag)
            self.connection.add_callback_threadsafe(partial(self.channel.basic_ack, message._tag))
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
            raise ConnectionClosed(e) from None
        except KeyError:
            self.logger.warning('Failed to ack message: not in known tags.')
        except Exception:
            self.logger.warning('Failed to ack message.', exc_info=True)

    def nack(self, message):
        if False:
            for i in range(10):
                print('nop')
        try:
            self.known_tags.remove(message._tag)
            self._nack(message._tag)
        except (pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
            raise ConnectionClosed(e) from None
        except KeyError:
            self.logger.warning('Failed to nack message: not in known tags.')
        except Exception:
            self.logger.warning('Failed to nack message.', exc_info=True)

    def _nack(self, tag):
        if False:
            for i in range(10):
                print('nop')
        self.connection.add_callback_threadsafe(partial(self.channel.basic_nack, tag, requeue=False))

    def requeue(self, messages):
        if False:
            i = 10
            return i + 15
        'RabbitMQ automatically re-enqueues unacked messages when\n        consumers disconnect so this is a no-op.\n        '

    def __next__(self):
        if False:
            while True:
                i = 10
        try:
            (method, properties, body) = next(self.iterator)
            if method is None:
                return None
        except (AssertionError, pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
            raise ConnectionClosed(e) from None
        try:
            message = Message.decode(body)
        except DecodeError:
            self.logger.exception('Failed to decode message using encoder %r.', get_encoder())
            self._nack(method.delivery_tag)
            return None
        rmq_message = _RabbitmqMessage(method.redelivered, method.delivery_tag, message)
        self.known_tags.add(method.delivery_tag)
        return rmq_message

    def close(self):
        if False:
            while True:
                i = 10
        try:
            all_callbacks_handled = Event()
            self.connection.add_callback_threadsafe(all_callbacks_handled.set)
            while not all_callbacks_handled.is_set():
                self.connection.sleep(0)
        except Exception:
            self.logger.exception('Failed to wait for all callbacks to complete.  This can happen when the RabbitMQ server is suddenly restarted.')
        try:
            self.channel.close()
            self.connection.close()
        except (AssertionError, pika.exceptions.AMQPConnectionError, pika.exceptions.AMQPChannelError) as e:
            raise ConnectionClosed(e) from None

class _RabbitmqMessage(MessageProxy):

    def __init__(self, redelivered, tag, message):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(message)
        self.redelivered = redelivered
        self._tag = tag