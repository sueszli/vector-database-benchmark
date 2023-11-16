"""Event dispatcher sends events."""
import os
import threading
import time
from collections import defaultdict, deque
from kombu import Producer
from celery.app import app_or_default
from celery.utils.nodenames import anon_nodename
from celery.utils.time import utcoffset
from .event import Event, get_exchange, group_from
__all__ = ('EventDispatcher',)

class EventDispatcher:
    """Dispatches event messages.

    Arguments:
        connection (kombu.Connection): Connection to the broker.

        hostname (str): Hostname to identify ourselves as,
            by default uses the hostname returned by
            :func:`~celery.utils.anon_nodename`.

        groups (Sequence[str]): List of groups to send events for.
            :meth:`send` will ignore send requests to groups not in this list.
            If this is :const:`None`, all events will be sent.
            Example groups include ``"task"`` and ``"worker"``.

        enabled (bool): Set to :const:`False` to not actually publish any
            events, making :meth:`send` a no-op.

        channel (kombu.Channel): Can be used instead of `connection` to specify
            an exact channel to use when sending events.

        buffer_while_offline (bool): If enabled events will be buffered
            while the connection is down. :meth:`flush` must be called
            as soon as the connection is re-established.

    Note:
        You need to :meth:`close` this after use.
    """
    DISABLED_TRANSPORTS = {'sql'}
    app = None
    on_enabled = None
    on_disabled = None

    def __init__(self, connection=None, hostname=None, enabled=True, channel=None, buffer_while_offline=True, app=None, serializer=None, groups=None, delivery_mode=1, buffer_group=None, buffer_limit=24, on_send_buffered=None):
        if False:
            print('Hello World!')
        self.app = app_or_default(app or self.app)
        self.connection = connection
        self.channel = channel
        self.hostname = hostname or anon_nodename()
        self.buffer_while_offline = buffer_while_offline
        self.buffer_group = buffer_group or frozenset()
        self.buffer_limit = buffer_limit
        self.on_send_buffered = on_send_buffered
        self._group_buffer = defaultdict(list)
        self.mutex = threading.Lock()
        self.producer = None
        self._outbound_buffer = deque()
        self.serializer = serializer or self.app.conf.event_serializer
        self.on_enabled = set()
        self.on_disabled = set()
        self.groups = set(groups or [])
        self.tzoffset = [-time.timezone, -time.altzone]
        self.clock = self.app.clock
        self.delivery_mode = delivery_mode
        if not connection and channel:
            self.connection = channel.connection.client
        self.enabled = enabled
        conninfo = self.connection or self.app.connection_for_write()
        self.exchange = get_exchange(conninfo, name=self.app.conf.event_exchange)
        if conninfo.transport.driver_type in self.DISABLED_TRANSPORTS:
            self.enabled = False
        if self.enabled:
            self.enable()
        self.headers = {'hostname': self.hostname}
        self.pid = os.getpid()

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, *exc_info):
        if False:
            while True:
                i = 10
        self.close()

    def enable(self):
        if False:
            return 10
        self.producer = Producer(self.channel or self.connection, exchange=self.exchange, serializer=self.serializer, auto_declare=False)
        self.enabled = True
        for callback in self.on_enabled:
            callback()

    def disable(self):
        if False:
            print('Hello World!')
        if self.enabled:
            self.enabled = False
            self.close()
            for callback in self.on_disabled:
                callback()

    def publish(self, type, fields, producer, blind=False, Event=Event, **kwargs):
        if False:
            i = 10
            return i + 15
        "Publish event using custom :class:`~kombu.Producer`.\n\n        Arguments:\n            type (str): Event type name, with group separated by dash (`-`).\n                fields: Dictionary of event fields, must be json serializable.\n            producer (kombu.Producer): Producer instance to use:\n                only the ``publish`` method will be called.\n            retry (bool): Retry in the event of connection failure.\n            retry_policy (Mapping): Map of custom retry policy options.\n                See :meth:`~kombu.Connection.ensure`.\n            blind (bool): Don't set logical clock value (also don't forward\n                the internal logical clock).\n            Event (Callable): Event type used to create event.\n                Defaults to :func:`Event`.\n            utcoffset (Callable): Function returning the current\n                utc offset in hours.\n        "
        clock = None if blind else self.clock.forward()
        event = Event(type, hostname=self.hostname, utcoffset=utcoffset(), pid=self.pid, clock=clock, **fields)
        with self.mutex:
            return self._publish(event, producer, routing_key=type.replace('-', '.'), **kwargs)

    def _publish(self, event, producer, routing_key, retry=False, retry_policy=None, utcoffset=utcoffset):
        if False:
            print('Hello World!')
        exchange = self.exchange
        try:
            producer.publish(event, routing_key=routing_key, exchange=exchange.name, retry=retry, retry_policy=retry_policy, declare=[exchange], serializer=self.serializer, headers=self.headers, delivery_mode=self.delivery_mode)
        except Exception as exc:
            if not self.buffer_while_offline:
                raise
            self._outbound_buffer.append((event, routing_key, exc))

    def send(self, type, blind=False, utcoffset=utcoffset, retry=False, retry_policy=None, Event=Event, **fields):
        if False:
            for i in range(10):
                print('nop')
        "Send event.\n\n        Arguments:\n            type (str): Event type name, with group separated by dash (`-`).\n            retry (bool): Retry in the event of connection failure.\n            retry_policy (Mapping): Map of custom retry policy options.\n                See :meth:`~kombu.Connection.ensure`.\n            blind (bool): Don't set logical clock value (also don't forward\n                the internal logical clock).\n            Event (Callable): Event type used to create event,\n                defaults to :func:`Event`.\n            utcoffset (Callable): unction returning the current utc offset\n                in hours.\n            **fields (Any): Event fields -- must be json serializable.\n        "
        if self.enabled:
            (groups, group) = (self.groups, group_from(type))
            if groups and group not in groups:
                return
            if group in self.buffer_group:
                clock = self.clock.forward()
                event = Event(type, hostname=self.hostname, utcoffset=utcoffset(), pid=self.pid, clock=clock, **fields)
                buf = self._group_buffer[group]
                buf.append(event)
                if len(buf) >= self.buffer_limit:
                    self.flush()
                elif self.on_send_buffered:
                    self.on_send_buffered()
            else:
                return self.publish(type, fields, self.producer, blind=blind, Event=Event, retry=retry, retry_policy=retry_policy)

    def flush(self, errors=True, groups=True):
        if False:
            return 10
        'Flush the outbound buffer.'
        if errors:
            buf = list(self._outbound_buffer)
            try:
                with self.mutex:
                    for (event, routing_key, _) in buf:
                        self._publish(event, self.producer, routing_key)
            finally:
                self._outbound_buffer.clear()
        if groups:
            with self.mutex:
                for (group, events) in self._group_buffer.items():
                    self._publish(events, self.producer, '%s.multi' % group)
                    events[:] = []

    def extend_buffer(self, other):
        if False:
            return 10
        'Copy the outbound buffer of another instance.'
        self._outbound_buffer.extend(other._outbound_buffer)

    def close(self):
        if False:
            while True:
                i = 10
        'Close the event dispatcher.'
        self.mutex.locked() and self.mutex.release()
        self.producer = None

    def _get_publisher(self):
        if False:
            i = 10
            return i + 15
        return self.producer

    def _set_publisher(self, producer):
        if False:
            for i in range(10):
                print('nop')
        self.producer = producer
    publisher = property(_get_publisher, _set_publisher)