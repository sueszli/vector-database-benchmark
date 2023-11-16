"""The blocking connection adapter module implements blocking semantics on top
of Pika's core AMQP driver. While most of the asynchronous expectations are
removed when using the blocking connection adapter, it attempts to remain true
to the asynchronous RPC nature of the AMQP protocol, supporting server sent
RPC commands.

The user facing classes in the module consist of the
:py:class:`~pika.adapters.blocking_connection.BlockingConnection`
and the :class:`~pika.adapters.blocking_connection.BlockingChannel`
classes.

"""
from collections import namedtuple, deque
import contextlib
import functools
import logging
import threading
import pika.compat as compat
import pika.exceptions as exceptions
import pika.spec
import pika.validators as validators
from pika.adapters.utils import connection_workflow
from pika.adapters import select_connection
from pika.exchange_type import ExchangeType
LOGGER = logging.getLogger(__name__)

class _CallbackResult:
    """ CallbackResult is a non-thread-safe implementation for receiving
    callback results; INTERNAL USE ONLY!
    """
    __slots__ = ('_value_class', '_ready', '_values')

    def __init__(self, value_class=None):
        if False:
            print('Hello World!')
        '\n        :param callable value_class: only needed if the CallbackResult\n                                     instance will be used with\n                                     `set_value_once` and `append_element`.\n                                     *args and **kwargs of the value setter\n                                     methods will be passed to this class.\n\n        '
        self._value_class = value_class
        self._ready = None
        self._values = None
        self.reset()

    def reset(self):
        if False:
            while True:
                i = 10
        'Reset value, but not _value_class'
        self._ready = False
        self._values = None

    def __bool__(self):
        if False:
            return 10
        ' Called by python runtime to implement truth value testing and the\n        built-in operation bool(); NOTE: python 3.x\n        '
        return self.is_ready()
    __nonzero__ = __bool__

    def __enter__(self):
        if False:
            return 10
        ' Entry into context manager that automatically resets the object\n        on exit; this usage pattern helps garbage-collection by eliminating\n        potential circular references.\n        '
        return self

    def __exit__(self, *args, **kwargs):
        if False:
            return 10
        'Reset value'
        self.reset()

    def is_ready(self):
        if False:
            while True:
                i = 10
        '\n        :returns: True if the object is in a signaled state\n        :rtype: bool\n        '
        return self._ready

    @property
    def ready(self):
        if False:
            while True:
                i = 10
        'True if the object is in a signaled state'
        return self._ready

    def signal_once(self, *_args, **_kwargs):
        if False:
            i = 10
            return i + 15
        ' Set as ready\n\n        :raises AssertionError: if result was already signalled\n        '
        assert not self._ready, '_CallbackResult was already set'
        self._ready = True

    def set_value_once(self, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Set as ready with value; the value may be retrieved via the `value`\n        property getter\n\n        :raises AssertionError: if result was already set\n        '
        self.signal_once()
        try:
            self._values = (self._value_class(*args, **kwargs),)
        except Exception:
            LOGGER.error('set_value_once failed: value_class=%r; args=%r; kwargs=%r', self._value_class, args, kwargs)
            raise

    def append_element(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Append an element to values'
        assert not self._ready or isinstance(self._values, list), '_CallbackResult state is incompatible with append_element: ready=%r; values=%r' % (self._ready, self._values)
        try:
            value = self._value_class(*args, **kwargs)
        except Exception:
            LOGGER.error('append_element failed: value_class=%r; args=%r; kwargs=%r', self._value_class, args, kwargs)
            raise
        if self._values is None:
            self._values = [value]
        else:
            self._values.append(value)
        self._ready = True

    @property
    def value(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        :returns: a reference to the value that was set via `set_value_once`\n        :rtype: object\n        :raises AssertionError: if result was not set or value is incompatible\n                                with `set_value_once`\n        '
        assert self._ready, '_CallbackResult was not set'
        assert isinstance(self._values, tuple) and len(self._values) == 1, '_CallbackResult value is incompatible with set_value_once: %r' % (self._values,)
        return self._values[0]

    @property
    def elements(self):
        if False:
            i = 10
            return i + 15
        '\n        :returns: a reference to the list containing one or more elements that\n            were added via `append_element`\n        :rtype: list\n        :raises AssertionError: if result was not set or value is incompatible\n                                with `append_element`\n        '
        assert self._ready, '_CallbackResult was not set'
        assert isinstance(self._values, list) and self._values, '_CallbackResult value is incompatible with append_element: %r' % (self._values,)
        return self._values

class _IoloopTimerContext:
    """Context manager for registering and safely unregistering a
    SelectConnection ioloop-based timer
    """

    def __init__(self, duration, connection):
        if False:
            return 10
        '\n        :param float duration: non-negative timer duration in seconds\n        :param select_connection.SelectConnection connection:\n        '
        assert hasattr(connection, '_adapter_call_later'), connection
        self._duration = duration
        self._connection = connection
        self._callback_result = _CallbackResult()
        self._timer_handle = None

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        'Register a timer'
        self._timer_handle = self._connection._adapter_call_later(self._duration, self._callback_result.signal_once)
        return self

    def __exit__(self, *_args, **_kwargs):
        if False:
            return 10
        "Unregister timer if it hasn't fired yet"
        if not self._callback_result:
            self._connection._adapter_remove_timeout(self._timer_handle)
            self._timer_handle = None

    def is_ready(self):
        if False:
            while True:
                i = 10
        '\n        :returns: True if timer has fired, False otherwise\n        :rtype: bool\n        '
        return self._callback_result.is_ready()

class _TimerEvt:
    """Represents a timer created via `BlockingConnection.call_later`"""
    __slots__ = ('timer_id', '_callback')

    def __init__(self, callback):
        if False:
            for i in range(10):
                print('nop')
        '\n        :param callback: see callback in `BlockingConnection.call_later`\n        '
        self._callback = callback
        self.timer_id = None

    def __repr__(self):
        if False:
            return 10
        return '<{} timer_id={} callback={}>'.format(self.__class__.__name__, self.timer_id, self._callback)

    def dispatch(self):
        if False:
            while True:
                i = 10
        "Dispatch the user's callback method"
        LOGGER.debug('_TimerEvt.dispatch: invoking callback=%r', self._callback)
        self._callback()

class _ConnectionBlockedUnblockedEvtBase:
    """Base class for `_ConnectionBlockedEvt` and `_ConnectionUnblockedEvt`"""
    __slots__ = ('_callback', '_method_frame')

    def __init__(self, callback, method_frame):
        if False:
            while True:
                i = 10
        '\n        :param callback: see callback parameter in\n          `BlockingConnection.add_on_connection_blocked_callback` and\n          `BlockingConnection.add_on_connection_unblocked_callback`\n        :param pika.frame.Method method_frame: with method_frame.method of type\n          `pika.spec.Connection.Blocked` or `pika.spec.Connection.Unblocked`\n        '
        self._callback = callback
        self._method_frame = method_frame

    def __repr__(self):
        if False:
            return 10
        return '<{} callback={}, frame={}>'.format(self.__class__.__name__, self._callback, self._method_frame)

    def dispatch(self):
        if False:
            print('Hello World!')
        "Dispatch the user's callback method"
        self._callback(self._method_frame)

class _ConnectionBlockedEvt(_ConnectionBlockedUnblockedEvtBase):
    """Represents a Connection.Blocked notification from RabbitMQ broker`"""

class _ConnectionUnblockedEvt(_ConnectionBlockedUnblockedEvtBase):
    """Represents a Connection.Unblocked notification from RabbitMQ broker`"""

class BlockingConnection:
    """The BlockingConnection creates a layer on top of Pika's asynchronous core
    providing methods that will block until their expected response has
    returned. Due to the asynchronous nature of the `Basic.Deliver` and
    `Basic.Return` calls from RabbitMQ to your application, you can still
    implement continuation-passing style asynchronous methods if you'd like to
    receive messages from RabbitMQ using
    :meth:`basic_consume <BlockingChannel.basic_consume>` or if you want to be
    notified of a delivery failure when using
    :meth:`basic_publish <BlockingChannel.basic_publish>`.

    For more information about communicating with the blocking_connection
    adapter, be sure to check out the
    :class:`BlockingChannel <BlockingChannel>` class which implements the
    :class:`Channel <pika.channel.Channel>` based communication for the
    blocking_connection adapter.

    To prevent recursion/reentrancy, the blocking connection and channel
    implementations queue asynchronously-delivered events received
    in nested context (e.g., while waiting for `BlockingConnection.channel` or
    `BlockingChannel.queue_declare` to complete), dispatching them synchronously
    once nesting returns to the desired context. This concerns all callbacks,
    such as those registered via `BlockingConnection.call_later`,
    `BlockingConnection.add_on_connection_blocked_callback`,
    `BlockingConnection.add_on_connection_unblocked_callback`,
    `BlockingChannel.basic_consume`, etc.

    Blocked Connection deadlock avoidance: when RabbitMQ becomes low on
    resources, it emits Connection.Blocked (AMQP extension) to the client
    connection when client makes a resource-consuming request on that connection
    or its channel (e.g., `Basic.Publish`); subsequently, RabbitMQ suspsends
    processing requests from that connection until the affected resources are
    restored. See http://www.rabbitmq.com/connection-blocked.html. This
    may impact `BlockingConnection` and `BlockingChannel` operations in a
    way that users might not be expecting. For example, if the user dispatches
    `BlockingChannel.basic_publish` in non-publisher-confirmation mode while
    RabbitMQ is in this low-resource state followed by a synchronous request
    (e.g., `BlockingConnection.channel`, `BlockingChannel.consume`,
    `BlockingChannel.basic_consume`, etc.), the synchronous request will block
    indefinitely (until Connection.Unblocked) waiting for RabbitMQ to reply. If
    the blocked state persists for a long time, the blocking operation will
    appear to hang. In this state, `BlockingConnection` instance and its
    channels will not dispatch user callbacks. SOLUTION: To break this potential
    deadlock, applications may configure the `blocked_connection_timeout`
    connection parameter when instantiating `BlockingConnection`. Upon blocked
    connection timeout, this adapter will raise ConnectionBlockedTimeout
    exception`. See `pika.connection.ConnectionParameters` documentation to
    learn more about the `blocked_connection_timeout` configuration.

    """
    _OnClosedArgs = namedtuple('BlockingConnection__OnClosedArgs', 'connection error')
    _OnChannelOpenedArgs = namedtuple('BlockingConnection__OnChannelOpenedArgs', 'channel')

    def __init__(self, parameters=None, _impl_class=None):
        if False:
            while True:
                i = 10
        'Create a new instance of the Connection object.\n\n        :param None | pika.connection.Parameters | sequence parameters:\n            Connection parameters instance or non-empty sequence of them. If\n            None, a `pika.connection.Parameters` instance will be created with\n            default settings. See `pika.AMQPConnectionWorkflow` for more\n            details about multiple parameter configurations and retries.\n        :param _impl_class: for tests/debugging only; implementation class;\n            None=default\n\n        :raises RuntimeError:\n\n        '
        self._cleanup_mutex = threading.Lock()
        self._event_dispatch_suspend_depth = 0
        self._ready_events = deque()
        self._channels_pending_dispatch = set()
        self._closed_result = _CallbackResult(self._OnClosedArgs)
        self._impl = None
        self._impl = self._create_connection(parameters, _impl_class)
        self._impl.add_on_close_callback(self._closed_result.set_value_once)

    def __repr__(self):
        if False:
            return 10
        return '<{} impl={!r}>'.format(self.__class__.__name__, self._impl)

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, value, traceback):
        if False:
            print('Hello World!')
        if self.is_open:
            self.close()

    def _cleanup(self):
        if False:
            i = 10
            return i + 15
        'Clean up members that might inhibit garbage collection\n\n        '
        with self._cleanup_mutex:
            if self._impl is not None:
                self._impl.ioloop.close()
            self._ready_events.clear()
            self._closed_result.reset()

    @contextlib.contextmanager
    def _acquire_event_dispatch(self):
        if False:
            for i in range(10):
                print('nop')
        ' Context manager that controls access to event dispatcher for\n        preventing reentrancy.\n\n        The "as" value is True if the managed code block owns the event\n        dispatcher and False if caller higher up in the call stack already owns\n        it. Only managed code that gets ownership (got True) is permitted to\n        dispatch\n        '
        try:
            self._event_dispatch_suspend_depth += 1
            yield (self._event_dispatch_suspend_depth == 1)
        finally:
            self._event_dispatch_suspend_depth -= 1

    def _create_connection(self, configs, impl_class):
        if False:
            for i in range(10):
                print('nop')
        'Run connection workflow, blocking until it completes.\n\n        :param None | pika.connection.Parameters | sequence configs: Connection\n            parameters instance or non-empty sequence of them.\n        :param None | SelectConnection impl_class: for tests/debugging only;\n            implementation class;\n\n        :rtype: impl_class\n\n        :raises: exception on failure\n        '
        if configs is None:
            configs = (pika.connection.Parameters(),)
        if isinstance(configs, pika.connection.Parameters):
            configs = (configs,)
        if not configs:
            raise ValueError('Expected a non-empty sequence of connection parameters, but got {!r}.'.format(configs))
        on_cw_done_result = _CallbackResult(namedtuple('BlockingConnection_OnConnectionWorkflowDoneArgs', 'result'))
        impl_class = impl_class or select_connection.SelectConnection
        ioloop = select_connection.IOLoop()
        ioloop.activate_poller()
        try:
            impl_class.create_connection(configs, on_done=on_cw_done_result.set_value_once, custom_ioloop=ioloop)
            while not on_cw_done_result.ready:
                ioloop.poll()
                ioloop.process_timeouts()
            if isinstance(on_cw_done_result.value.result, BaseException):
                error = on_cw_done_result.value.result
                LOGGER.error('Connection workflow failed: %r', error)
                raise self._reap_last_connection_workflow_error(error)
            else:
                LOGGER.info('Connection workflow succeeded: %r', on_cw_done_result.value.result)
                return on_cw_done_result.value.result
        except Exception:
            LOGGER.exception('Error in _create_connection().')
            ioloop.close()
            self._cleanup()
            raise

    @staticmethod
    def _reap_last_connection_workflow_error(error):
        if False:
            return 10
        'Extract exception value from the last connection attempt\n\n        :param Exception error: error passed by the `AMQPConnectionWorkflow`\n            completion callback.\n\n        :returns: Exception value from the last connection attempt\n        :rtype: Exception\n        '
        if isinstance(error, connection_workflow.AMQPConnectionWorkflowFailed):
            error = error.exceptions[-1]
            if isinstance(error, connection_workflow.AMQPConnectorSocketConnectError):
                error = exceptions.AMQPConnectionError(error)
            elif isinstance(error, connection_workflow.AMQPConnectorPhaseErrorBase):
                error = error.exception
        return error

    def _flush_output(self, *waiters):
        if False:
            for i in range(10):
                print('nop')
        " Flush output and process input while waiting for any of the given\n        callbacks to return true. The wait is aborted upon connection-close.\n        Otherwise, processing continues until the output is flushed AND at least\n        one of the callbacks returns true. If there are no callbacks, then\n        processing ends when all output is flushed.\n\n        :param waiters: sequence of zero or more callables taking no args and\n                        returning true when it's time to stop processing.\n                        Their results are OR'ed together.\n        :raises: exceptions passed by impl if opening of connection fails or\n            connection closes.\n        "
        if self.is_closed:
            raise exceptions.ConnectionWrongStateError()
        is_done = lambda : self._closed_result.ready or ((not self._impl._transport or self._impl._get_write_buffer_size() == 0) and (not waiters or any((ready() for ready in waiters))))
        while not is_done():
            self._impl.ioloop.poll()
            self._impl.ioloop.process_timeouts()
        if self._closed_result.ready:
            try:
                if not isinstance(self._closed_result.value.error, exceptions.ConnectionClosedByClient):
                    LOGGER.error('Unexpected connection close detected: %r', self._closed_result.value.error)
                    raise self._closed_result.value.error
                else:
                    LOGGER.info('User-initiated close: result=%r', self._closed_result.value)
            finally:
                self._cleanup()

    def _request_channel_dispatch(self, channel_number):
        if False:
            print('Hello World!')
        "Called by BlockingChannel instances to request a call to their\n        _dispatch_events method or to terminate `process_data_events`;\n        BlockingConnection will honor these requests from a safe context.\n\n        :param int channel_number: positive channel number to request a call\n            to the channel's `_dispatch_events`; a negative channel number to\n            request termination of `process_data_events`\n        "
        self._channels_pending_dispatch.add(channel_number)

    def _dispatch_channel_events(self):
        if False:
            for i in range(10):
                print('nop')
        'Invoke the `_dispatch_events` method on open channels that requested\n        it\n        '
        if not self._channels_pending_dispatch:
            return
        with self._acquire_event_dispatch() as dispatch_acquired:
            if not dispatch_acquired:
                return
            candidates = list(self._channels_pending_dispatch)
            self._channels_pending_dispatch.clear()
            for channel_number in candidates:
                if channel_number < 0:
                    continue
                try:
                    impl_channel = self._impl._channels[channel_number]
                except KeyError:
                    continue
                if impl_channel.is_open:
                    impl_channel._get_cookie()._dispatch_events()

    def _on_timer_ready(self, evt):
        if False:
            for i in range(10):
                print('nop')
        'Handle expiry of a timer that was registered via\n        `_adapter_call_later()`\n\n        :param _TimerEvt evt:\n\n        '
        self._ready_events.append(evt)

    def _on_threadsafe_callback(self, user_callback):
        if False:
            print('Hello World!')
        'Handle callback that was registered via\n        `self._impl._adapter_add_callback_threadsafe`.\n\n        :param user_callback: callback passed to our\n            `add_callback_threadsafe` by the application.\n\n        '
        self.call_later(0, user_callback)

    def _on_connection_blocked(self, user_callback, _impl, method_frame):
        if False:
            i = 10
            return i + 15
        'Handle Connection.Blocked notification from RabbitMQ broker\n\n        :param callable user_callback: callback passed to\n           `add_on_connection_blocked_callback`\n        :param select_connection.SelectConnection _impl:\n        :param pika.frame.Method method_frame: method frame having `method`\n            member of type `pika.spec.Connection.Blocked`\n        '
        self._ready_events.append(_ConnectionBlockedEvt(user_callback, method_frame))

    def _on_connection_unblocked(self, user_callback, _impl, method_frame):
        if False:
            while True:
                i = 10
        'Handle Connection.Unblocked notification from RabbitMQ broker\n\n        :param callable user_callback: callback passed to\n           `add_on_connection_unblocked_callback`\n        :param select_connection.SelectConnection _impl:\n        :param pika.frame.Method method_frame: method frame having `method`\n            member of type `pika.spec.Connection.Blocked`\n        '
        self._ready_events.append(_ConnectionUnblockedEvt(user_callback, method_frame))

    def _dispatch_connection_events(self):
        if False:
            for i in range(10):
                print('nop')
        'Dispatch ready connection events'
        if not self._ready_events:
            return
        with self._acquire_event_dispatch() as dispatch_acquired:
            if not dispatch_acquired:
                return
            for _ in compat.xrange(len(self._ready_events)):
                try:
                    evt = self._ready_events.popleft()
                except IndexError:
                    break
                evt.dispatch()

    def add_on_connection_blocked_callback(self, callback):
        if False:
            while True:
                i = 10
        "RabbitMQ AMQP extension - Add a callback to be notified when the\n        connection gets blocked (`Connection.Blocked` received from RabbitMQ)\n        due to the broker running low on resources (memory or disk). In this\n        state RabbitMQ suspends processing incoming data until the connection\n        is unblocked, so it's a good idea for publishers receiving this\n        notification to suspend publishing until the connection becomes\n        unblocked.\n\n        NOTE: due to the blocking nature of BlockingConnection, if it's sending\n        outbound data while the connection is/becomes blocked, the call may\n        remain blocked until the connection becomes unblocked, if ever. You\n        may use `ConnectionParameters.blocked_connection_timeout` to abort a\n        BlockingConnection method call with an exception when the connection\n        remains blocked longer than the given timeout value.\n\n        See also `Connection.add_on_connection_unblocked_callback()`\n\n        See also `ConnectionParameters.blocked_connection_timeout`.\n\n        :param callable callback: Callback to call on `Connection.Blocked`,\n            having the signature `callback(connection, pika.frame.Method)`,\n            where connection is the `BlockingConnection` instance and the method\n            frame's `method` member is of type `pika.spec.Connection.Blocked`\n\n        "
        self._impl.add_on_connection_blocked_callback(functools.partial(self._on_connection_blocked, functools.partial(callback, self)))

    def add_on_connection_unblocked_callback(self, callback):
        if False:
            while True:
                i = 10
        "RabbitMQ AMQP extension - Add a callback to be notified when the\n        connection gets unblocked (`Connection.Unblocked` frame is received from\n        RabbitMQ) letting publishers know it's ok to start publishing again.\n\n        :param callable callback: Callback to call on Connection.Unblocked`,\n            having the signature `callback(connection, pika.frame.Method)`,\n            where connection is the `BlockingConnection` instance and the method\n            frame's `method` member is of type `pika.spec.Connection.Unblocked`\n\n        "
        self._impl.add_on_connection_unblocked_callback(functools.partial(self._on_connection_unblocked, functools.partial(callback, self)))

    def call_later(self, delay, callback):
        if False:
            for i in range(10):
                print('nop')
        "Create a single-shot timer to fire after delay seconds. Do not\n        confuse with Tornado's timeout where you pass in the time you want to\n        have your callback called. Only pass in the seconds until it's to be\n        called.\n\n        NOTE: the timer callbacks are dispatched only in the scope of\n        specially-designated methods: see\n        `BlockingConnection.process_data_events()` and\n        `BlockingChannel.start_consuming()`.\n\n        :param float delay: The number of seconds to wait to call callback\n        :param callable callback: The callback method with the signature\n            callback()\n        :returns: Opaque timer id\n        :rtype: int\n\n        "
        validators.require_callback(callback)
        evt = _TimerEvt(callback=callback)
        timer_id = self._impl._adapter_call_later(delay, functools.partial(self._on_timer_ready, evt))
        evt.timer_id = timer_id
        return timer_id

    def add_callback_threadsafe(self, callback):
        if False:
            i = 10
            return i + 15
        "Requests a call to the given function as soon as possible in the\n        context of this connection's thread.\n\n        NOTE: This is the only thread-safe method in `BlockingConnection`. All\n        other manipulations of `BlockingConnection` must be performed from the\n        connection's thread.\n\n        NOTE: the callbacks are dispatched only in the scope of\n        specially-designated methods: see\n        `BlockingConnection.process_data_events()` and\n        `BlockingChannel.start_consuming()`.\n\n        For example, a thread may request a call to the\n        `BlockingChannel.basic_ack` method of a `BlockingConnection` that is\n        running in a different thread via::\n\n            connection.add_callback_threadsafe(\n                functools.partial(channel.basic_ack, delivery_tag=...))\n\n        NOTE: if you know that the requester is running on the same thread as\n        the connection it is more efficient to use the\n        `BlockingConnection.call_later()` method with a delay of 0.\n\n        :param callable callback: The callback method; must be callable\n        :raises pika.exceptions.ConnectionWrongStateError: if connection is\n            closed\n        "
        with self._cleanup_mutex:
            if self.is_closed:
                raise exceptions.ConnectionWrongStateError('BlockingConnection.add_callback_threadsafe() called on closed or closing connection.')
            self._impl._adapter_add_callback_threadsafe(functools.partial(self._on_threadsafe_callback, callback))

    def remove_timeout(self, timeout_id):
        if False:
            return 10
        "Remove a timer if it's still in the timeout stack\n\n        :param timeout_id: The opaque timer id to remove\n\n        "
        self._impl._adapter_remove_timeout(timeout_id)
        for (i, evt) in enumerate(self._ready_events):
            if isinstance(evt, _TimerEvt) and evt.timer_id == timeout_id:
                index_to_remove = i
                break
        else:
            return
        del self._ready_events[index_to_remove]

    def update_secret(self, new_secret, reason):
        if False:
            print('Hello World!')
        'RabbitMQ AMQP extension - This method updates the secret used to authenticate this connection. \n        It is used when secrets have an expiration date and need to be renewed, like OAuth 2 tokens.\n\n        :param string new_secret: The new secret\n        :param string reason: The reason for the secret update\n\n        :raises pika.exceptions.ConnectionWrongStateError: if connection is\n            not open.\n        '
        result = _CallbackResult()
        self._impl.update_secret(new_secret, reason, result.signal_once)
        self._flush_output(result.is_ready)

    def close(self, reply_code=200, reply_text='Normal shutdown'):
        if False:
            for i in range(10):
                print('nop')
        'Disconnect from RabbitMQ. If there are any open channels, it will\n        attempt to close them prior to fully disconnecting. Channels which\n        have active consumers will attempt to send a Basic.Cancel to RabbitMQ\n        to cleanly stop the delivery of messages prior to closing the channel.\n\n        :param int reply_code: The code number for the close\n        :param str reply_text: The text reason for the close\n\n        :raises pika.exceptions.ConnectionWrongStateError: if called on a closed\n            connection (NEW in v1.0.0)\n        '
        if not self.is_open:
            msg = '{}.close({}, {!r}) called on closed connection.'.format(self.__class__.__name__, reply_code, reply_text)
            LOGGER.error(msg)
            raise exceptions.ConnectionWrongStateError(msg)
        LOGGER.info('Closing connection (%s): %s', reply_code, reply_text)
        for impl_channel in compat.dictvalues(self._impl._channels):
            channel = impl_channel._get_cookie()
            if channel.is_open:
                try:
                    channel.close(reply_code, reply_text)
                except exceptions.ChannelClosed as exc:
                    LOGGER.warning('Got ChannelClosed while closing channel from connection.close: %r', exc)
        self._impl.close(reply_code, reply_text)
        self._flush_output(self._closed_result.is_ready)

    def process_data_events(self, time_limit=0):
        if False:
            for i in range(10):
                print('nop')
        'Will make sure that data events are processed. Dispatches timer and\n        channel callbacks if not called from the scope of BlockingConnection or\n        BlockingChannel callback. Your app can block on this method. If your\n        application maintains a long-lived publisher connection, this method\n        should be called periodically in order to respond to heartbeats and other\n        data events. See `examples/long_running_publisher.py` for an example.\n\n        :param float time_limit: suggested upper bound on processing time in\n            seconds. The actual blocking time depends on the granularity of the\n            underlying ioloop. Zero means return as soon as possible. None means\n            there is no limit on processing time and the function will block\n            until I/O produces actionable events. Defaults to 0 for backward\n            compatibility. This parameter is NEW in pika 0.10.0.\n        '
        with self._acquire_event_dispatch() as dispatch_acquired:
            common_terminator = lambda : bool(dispatch_acquired and (self._channels_pending_dispatch or self._ready_events))
            if time_limit is None:
                self._flush_output(common_terminator)
            else:
                with _IoloopTimerContext(time_limit, self._impl) as timer:
                    self._flush_output(timer.is_ready, common_terminator)
        if self._ready_events:
            self._dispatch_connection_events()
        if self._channels_pending_dispatch:
            self._dispatch_channel_events()

    def sleep(self, duration):
        if False:
            return 10
        'A safer way to sleep than calling time.sleep() directly that would\n        keep the adapter from ignoring frames sent from the broker. The\n        connection will "sleep" or block the number of seconds specified in\n        duration in small intervals.\n\n        :param float duration: The time to sleep in seconds\n\n        '
        assert duration >= 0, duration
        deadline = compat.time_now() + duration
        time_limit = duration
        while True:
            self.process_data_events(time_limit)
            time_limit = deadline - compat.time_now()
            if time_limit <= 0:
                break

    def channel(self, channel_number=None):
        if False:
            i = 10
            return i + 15
        'Create a new channel with the next available channel number or pass\n        in a channel number to use. Must be non-zero if you would like to\n        specify but it is recommended that you let Pika manage the channel\n        numbers.\n\n        :rtype: pika.adapters.blocking_connection.BlockingChannel\n        '
        with _CallbackResult(self._OnChannelOpenedArgs) as opened_args:
            impl_channel = self._impl.channel(channel_number=channel_number, on_open_callback=opened_args.set_value_once)
            channel = BlockingChannel(impl_channel, self)
            impl_channel._set_cookie(channel)
            channel._flush_output(opened_args.is_ready)
        return channel

    @property
    def is_closed(self):
        if False:
            i = 10
            return i + 15
        '\n        Returns a boolean reporting the current connection state.\n        '
        return self._impl.is_closed

    @property
    def is_open(self):
        if False:
            return 10
        '\n        Returns a boolean reporting the current connection state.\n        '
        return self._impl.is_open

    @property
    def basic_nack_supported(self):
        if False:
            i = 10
            return i + 15
        'Specifies if the server supports basic.nack on the active connection.\n\n        :rtype: bool\n\n        '
        return self._impl.basic_nack

    @property
    def consumer_cancel_notify_supported(self):
        if False:
            i = 10
            return i + 15
        'Specifies if the server supports consumer cancel notification on the\n        active connection.\n\n        :rtype: bool\n\n        '
        return self._impl.consumer_cancel_notify

    @property
    def exchange_exchange_bindings_supported(self):
        if False:
            print('Hello World!')
        'Specifies if the active connection supports exchange to exchange\n        bindings.\n\n        :rtype: bool\n\n        '
        return self._impl.exchange_exchange_bindings

    @property
    def publisher_confirms_supported(self):
        if False:
            return 10
        'Specifies if the active connection can use publisher confirmations.\n\n        :rtype: bool\n\n        '
        return self._impl.publisher_confirms
    basic_nack = basic_nack_supported
    consumer_cancel_notify = consumer_cancel_notify_supported
    exchange_exchange_bindings = exchange_exchange_bindings_supported
    publisher_confirms = publisher_confirms_supported

class _ChannelPendingEvt:
    """Base class for BlockingChannel pending events"""

class _ConsumerDeliveryEvt(_ChannelPendingEvt):
    """This event represents consumer message delivery `Basic.Deliver`; it
    contains method, properties, and body of the delivered message.
    """
    __slots__ = ('method', 'properties', 'body')

    def __init__(self, method, properties, body):
        if False:
            i = 10
            return i + 15
        '\n        :param spec.Basic.Deliver method: NOTE: consumer_tag and delivery_tag\n          are valid only within source channel\n        :param spec.BasicProperties properties: message properties\n        :param bytes body: message body; empty string if no body\n        '
        self.method = method
        self.properties = properties
        self.body = body

class _ConsumerCancellationEvt(_ChannelPendingEvt):
    """This event represents server-initiated consumer cancellation delivered to
    client via Basic.Cancel. After receiving Basic.Cancel, there will be no
    further deliveries for the consumer identified by `consumer_tag` in
    `Basic.Cancel`
    """
    __slots__ = ('method_frame',)

    def __init__(self, method_frame):
        if False:
            return 10
        '\n        :param pika.frame.Method method_frame: method frame with method of type\n            `spec.Basic.Cancel`\n        '
        self.method_frame = method_frame

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{} method_frame={!r}>'.format(self.__class__.__name__, self.method_frame)

    @property
    def method(self):
        if False:
            for i in range(10):
                print('nop')
        'method of type spec.Basic.Cancel'
        return self.method_frame.method

class _ReturnedMessageEvt(_ChannelPendingEvt):
    """This event represents a message returned by broker via `Basic.Return`"""
    __slots__ = ('callback', 'channel', 'method', 'properties', 'body')

    def __init__(self, callback, channel, method, properties, body):
        if False:
            print('Hello World!')
        "\n        :param callable callback: user's callback, having the signature\n            callback(channel, method, properties, body), where\n             - channel: pika.Channel\n             - method: pika.spec.Basic.Return\n             - properties: pika.spec.BasicProperties\n             - body: bytes\n        :param pika.Channel channel:\n        :param pika.spec.Basic.Return method:\n        :param pika.spec.BasicProperties properties:\n        :param bytes body:\n        "
        self.callback = callback
        self.channel = channel
        self.method = method
        self.properties = properties
        self.body = body

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<%s callback=%r channel=%r method=%r properties=%r body=%.300r>' % (self.__class__.__name__, self.callback, self.channel, self.method, self.properties, self.body)

    def dispatch(self):
        if False:
            i = 10
            return i + 15
        "Dispatch user's callback"
        self.callback(self.channel, self.method, self.properties, self.body)

class ReturnedMessage:
    """Represents a message returned via Basic.Return in publish-acknowledgments
    mode
    """
    __slots__ = ('method', 'properties', 'body')

    def __init__(self, method, properties, body):
        if False:
            print('Hello World!')
        '\n        :param spec.Basic.Return method:\n        :param spec.BasicProperties properties: message properties\n        :param bytes body: message body; empty string if no body\n        '
        self.method = method
        self.properties = properties
        self.body = body

class _ConsumerInfo:
    """Information about an active consumer"""
    __slots__ = ('consumer_tag', 'auto_ack', 'on_message_callback', 'alternate_event_sink', 'state')
    SETTING_UP = 1
    ACTIVE = 2
    TEARING_DOWN = 3
    CANCELLED_BY_BROKER = 4

    def __init__(self, consumer_tag, auto_ack, on_message_callback=None, alternate_event_sink=None):
        if False:
            return 10
        "\n        NOTE: exactly one of callback/alternate_event_sink musts be non-None.\n\n        :param str consumer_tag:\n        :param bool auto_ack: the no-ack value for the consumer\n        :param callable on_message_callback: The function for dispatching messages to\n            user, having the signature:\n            on_message_callback(channel, method, properties, body)\n             - channel: BlockingChannel\n             - method: spec.Basic.Deliver\n             - properties: spec.BasicProperties\n             - body: bytes\n        :param callable alternate_event_sink: if specified, _ConsumerDeliveryEvt\n            and _ConsumerCancellationEvt objects will be diverted to this\n            callback instead of being deposited in the channel's\n            `_pending_events` container. Signature:\n            alternate_event_sink(evt)\n        "
        assert (on_message_callback is None) != (alternate_event_sink is None), ('exactly one of on_message_callback/alternate_event_sink must be non-None', on_message_callback, alternate_event_sink)
        self.consumer_tag = consumer_tag
        self.auto_ack = auto_ack
        self.on_message_callback = on_message_callback
        self.alternate_event_sink = alternate_event_sink
        self.state = self.SETTING_UP

    @property
    def setting_up(self):
        if False:
            for i in range(10):
                print('nop')
        'True if in SETTING_UP state'
        return self.state == self.SETTING_UP

    @property
    def active(self):
        if False:
            return 10
        'True if in ACTIVE state'
        return self.state == self.ACTIVE

    @property
    def tearing_down(self):
        if False:
            while True:
                i = 10
        'True if in TEARING_DOWN state'
        return self.state == self.TEARING_DOWN

    @property
    def cancelled_by_broker(self):
        if False:
            print('Hello World!')
        'True if in CANCELLED_BY_BROKER state'
        return self.state == self.CANCELLED_BY_BROKER

class _QueueConsumerGeneratorInfo:
    """Container for information about the active queue consumer generator """
    __slots__ = ('params', 'consumer_tag', 'pending_events')

    def __init__(self, params, consumer_tag):
        if False:
            for i in range(10):
                print('nop')
        '\n        :params tuple params: a three-tuple (queue, auto_ack, exclusive) that were\n           used to create the queue consumer\n        :param str consumer_tag: consumer tag\n        '
        self.params = params
        self.consumer_tag = consumer_tag
        self.pending_events = deque()

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return '<{} params={!r} consumer_tag={!r}>'.format(self.__class__.__name__, self.params, self.consumer_tag)

class BlockingChannel:
    """The BlockingChannel implements blocking semantics for most things that
    one would use callback-passing-style for with the
    :py:class:`~pika.channel.Channel` class. In addition,
    the `BlockingChannel` class implements a :term:`generator` that allows
    you to :doc:`consume messages </examples/blocking_consumer_generator>`
    without using callbacks.

    Example of creating a BlockingChannel::

        import pika

        # Create our connection object
        connection = pika.BlockingConnection()

        # The returned object will be a synchronous channel
        channel = connection.channel()

    """
    _RxMessageArgs = namedtuple('BlockingChannel__RxMessageArgs', ['channel', 'method', 'properties', 'body'])
    _MethodFrameCallbackResultArgs = namedtuple('BlockingChannel__MethodFrameCallbackResultArgs', 'method_frame')
    _OnMessageConfirmationReportArgs = namedtuple('BlockingChannel__OnMessageConfirmationReportArgs', 'method_frame')
    _FlowOkCallbackResultArgs = namedtuple('BlockingChannel__FlowOkCallbackResultArgs', 'active')
    _CONSUMER_CANCELLED_CB_KEY = 'blocking_channel_consumer_cancelled'

    def __init__(self, channel_impl, connection):
        if False:
            print('Hello World!')
        'Create a new instance of the Channel\n\n        :param pika.channel.Channel channel_impl: Channel implementation object\n            as returned from SelectConnection.channel()\n        :param BlockingConnection connection: The connection object\n\n        '
        self._impl = channel_impl
        self._connection = connection
        self._consumer_infos = dict()
        self._queue_consumer_generator = None
        self._delivery_confirmation = False
        self._message_confirmation_result = _CallbackResult(self._OnMessageConfirmationReportArgs)
        self._pending_events = deque()
        self._puback_return = None
        self._closing_reason = None
        self._basic_consume_ok_result = _CallbackResult()
        self._basic_getempty_result = _CallbackResult(self._MethodFrameCallbackResultArgs)
        self._impl.add_on_cancel_callback(self._on_consumer_cancelled_by_broker)
        self._impl.add_callback(self._basic_consume_ok_result.signal_once, replies=[pika.spec.Basic.ConsumeOk], one_shot=False)
        self._impl.add_on_close_callback(self._on_channel_closed)
        self._impl.add_callback(self._basic_getempty_result.set_value_once, replies=[pika.spec.Basic.GetEmpty], one_shot=False)
        LOGGER.info('Created channel=%s', self.channel_number)

    def __int__(self):
        if False:
            i = 10
            return i + 15
        'Return the channel object as its channel number\n\n        NOTE: inherited from legacy BlockingConnection; might be error-prone;\n        use `channel_number` property instead.\n\n        :rtype: int\n\n        '
        return self.channel_number

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<{} impl={!r}>'.format(self.__class__.__name__, self._impl)

    def __enter__(self):
        if False:
            while True:
                i = 10
        return self

    def __exit__(self, exc_type, value, traceback):
        if False:
            print('Hello World!')
        if self.is_open:
            self.close()

    def _cleanup(self):
        if False:
            print('Hello World!')
        'Clean up members that might inhibit garbage collection'
        self._message_confirmation_result.reset()
        self._pending_events = deque()
        self._consumer_infos = dict()
        self._queue_consumer_generator = None

    @property
    def channel_number(self):
        if False:
            for i in range(10):
                print('nop')
        'Channel number'
        return self._impl.channel_number

    @property
    def connection(self):
        if False:
            return 10
        "The channel's BlockingConnection instance"
        return self._connection

    @property
    def is_closed(self):
        if False:
            print('Hello World!')
        'Returns True if the channel is closed.\n\n        :rtype: bool\n\n        '
        return self._impl.is_closed

    @property
    def is_open(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the channel is open.\n\n        :rtype: bool\n\n        '
        return self._impl.is_open

    @property
    def consumer_tags(self):
        if False:
            for i in range(10):
                print('nop')
        'Property method that returns a list of consumer tags for active\n        consumers\n\n        :rtype: list\n\n        '
        return compat.dictkeys(self._consumer_infos)
    _ALWAYS_READY_WAITERS = (lambda : True,)

    def _flush_output(self, *waiters):
        if False:
            print('Hello World!')
        " Flush output and process input while waiting for any of the given\n        callbacks to return true. The wait is aborted upon channel-close or\n        connection-close.\n        Otherwise, processing continues until the output is flushed AND at least\n        one of the callbacks returns true. If there are no callbacks, then\n        processing ends when all output is flushed.\n\n        :param waiters: sequence of zero or more callables taking no args and\n                        returning true when it's time to stop processing.\n                        Their results are OR'ed together. An empty sequence is\n                        treated as equivalent to a waiter always returning true.\n        "
        if self.is_closed:
            self._impl._raise_if_not_open()
        if not waiters:
            waiters = self._ALWAYS_READY_WAITERS
        self._connection._flush_output(lambda : self.is_closed, *waiters)
        if self.is_closed and isinstance(self._closing_reason, exceptions.ChannelClosedByBroker):
            raise self._closing_reason

    def _on_puback_message_returned(self, channel, method, properties, body):
        if False:
            for i in range(10):
                print('nop')
        'Called as the result of Basic.Return from broker in\n        publisher-acknowledgements mode. Saves the info as a ReturnedMessage\n        instance in self._puback_return.\n\n        :param pika.Channel channel: our self._impl channel\n        :param pika.spec.Basic.Return method:\n        :param pika.spec.BasicProperties properties: message properties\n        :param bytes body: returned message body; empty string if no body\n        '
        assert channel is self._impl, (channel.channel_number, self.channel_number)
        assert isinstance(method, pika.spec.Basic.Return), method
        assert isinstance(properties, pika.spec.BasicProperties), properties
        LOGGER.warning('Published message was returned: _delivery_confirmation=%s; channel=%s; method=%r; properties=%r; body_size=%d; body_prefix=%.255r', self._delivery_confirmation, channel.channel_number, method, properties, len(body) if body is not None else None, body)
        self._puback_return = ReturnedMessage(method, properties, body)

    def _add_pending_event(self, evt):
        if False:
            i = 10
            return i + 15
        "Append an event to the channel's list of events that are ready for\n        dispatch to user and signal our connection that this channel is ready\n        for event dispatch\n\n        :param _ChannelPendingEvt evt: an event derived from _ChannelPendingEvt\n        "
        self._pending_events.append(evt)
        self.connection._request_channel_dispatch(self.channel_number)

    def _on_channel_closed(self, _channel, reason):
        if False:
            return 10
        "Callback from impl notifying us that the channel has been closed.\n        This may be as the result of user-, broker-, or internal connection\n        clean-up initiated closing or meta-closing of the channel.\n\n        If it resulted from receiving `Channel.Close` from broker, we will\n        expedite waking up of the event subsystem so that it may respond by\n        raising `ChannelClosed` from user's context.\n\n        NOTE: We can't raise exceptions in callbacks in order to protect\n        the integrity of the underlying implementation. BlockingConnection's\n        underlying asynchronous connection adapter (SelectConnection) uses\n        callbacks to communicate with us. If BlockingConnection leaks exceptions\n        back into the I/O loop or the asynchronous connection adapter, we\n        interrupt their normal workflow and introduce a high likelihood of state\n        inconsistency.\n\n        See `pika.Channel.add_on_close_callback()` for additional documentation.\n\n        :param pika.Channel _channel: (unused)\n        :param Exception reason:\n\n        "
        LOGGER.debug('_on_channel_closed: %r; %r', reason, self)
        self._closing_reason = reason
        if isinstance(reason, exceptions.ChannelClosedByBroker):
            self._cleanup()
            self.connection._request_channel_dispatch(-self.channel_number)

    def _on_consumer_cancelled_by_broker(self, method_frame):
        if False:
            for i in range(10):
                print('nop')
        'Called by impl when broker cancels consumer via Basic.Cancel.\n\n        This is a RabbitMQ-specific feature. The circumstances include deletion\n        of queue being consumed as well as failure of a HA node responsible for\n        the queue being consumed.\n\n        :param pika.frame.Method method_frame: method frame with the\n            `spec.Basic.Cancel` method\n\n        '
        evt = _ConsumerCancellationEvt(method_frame)
        consumer = self._consumer_infos[method_frame.method.consumer_tag]
        if not consumer.tearing_down:
            consumer.state = _ConsumerInfo.CANCELLED_BY_BROKER
        if consumer.alternate_event_sink is not None:
            consumer.alternate_event_sink(evt)
        else:
            self._add_pending_event(evt)

    def _on_consumer_message_delivery(self, _channel, method, properties, body):
        if False:
            while True:
                i = 10
        'Called by impl when a message is delivered for a consumer\n\n        :param Channel channel: The implementation channel object\n        :param spec.Basic.Deliver method:\n        :param pika.spec.BasicProperties properties: message properties\n        :param bytes body: delivered message body; empty string if no body\n        '
        evt = _ConsumerDeliveryEvt(method, properties, body)
        consumer = self._consumer_infos[method.consumer_tag]
        if consumer.alternate_event_sink is not None:
            consumer.alternate_event_sink(evt)
        else:
            self._add_pending_event(evt)

    def _on_consumer_generator_event(self, evt):
        if False:
            for i in range(10):
                print('nop')
        "Sink for the queue consumer generator's consumer events; append the\n        event to queue consumer generator's pending events buffer.\n\n        :param evt: an object of type _ConsumerDeliveryEvt or\n          _ConsumerCancellationEvt\n        "
        self._queue_consumer_generator.pending_events.append(evt)
        self.connection._request_channel_dispatch(-self.channel_number)

    def _cancel_all_consumers(self):
        if False:
            i = 10
            return i + 15
        'Cancel all consumers.\n\n        NOTE: pending non-ackable messages will be lost; pending ackable\n        messages will be rejected.\n\n        '
        if self._consumer_infos:
            LOGGER.debug('Cancelling %i consumers', len(self._consumer_infos))
            if self._queue_consumer_generator is not None:
                self.cancel()
            for consumer_tag in compat.dictkeys(self._consumer_infos):
                self.basic_cancel(consumer_tag)

    def _dispatch_events(self):
        if False:
            i = 10
            return i + 15
        'Called by BlockingConnection to dispatch pending events.\n\n        `BlockingChannel` schedules this callback via\n        `BlockingConnection._request_channel_dispatch`\n        '
        while self._pending_events:
            evt = self._pending_events.popleft()
            if type(evt) is _ConsumerDeliveryEvt:
                consumer_info = self._consumer_infos[evt.method.consumer_tag]
                consumer_info.on_message_callback(self, evt.method, evt.properties, evt.body)
            elif type(evt) is _ConsumerCancellationEvt:
                del self._consumer_infos[evt.method_frame.method.consumer_tag]
                self._impl.callbacks.process(self.channel_number, self._CONSUMER_CANCELLED_CB_KEY, self, evt.method_frame)
            else:
                evt.dispatch()

    def close(self, reply_code=0, reply_text='Normal shutdown'):
        if False:
            while True:
                i = 10
        'Will invoke a clean shutdown of the channel with the AMQP Broker.\n\n        :param int reply_code: The reply code to close the channel with\n        :param str reply_text: The reply text to close the channel with\n\n        '
        LOGGER.debug('Channel.close(%s, %s)', reply_code, reply_text)
        self._impl._raise_if_not_open()
        try:
            self._cancel_all_consumers()
            self._impl.close(reply_code=reply_code, reply_text=reply_text)
            self._flush_output(lambda : self.is_closed)
        finally:
            self._cleanup()

    def flow(self, active):
        if False:
            i = 10
            return i + 15
        'Turn Channel flow control off and on.\n\n        NOTE: RabbitMQ doesn\'t support active=False; per\n        https://www.rabbitmq.com/specification.html: "active=false is not\n        supported by the server. Limiting prefetch with basic.qos provides much\n        better control"\n\n        For more information, please reference:\n\n        http://www.rabbitmq.com/amqp-0-9-1-reference.html#channel.flow\n\n        :param bool active: Turn flow on (True) or off (False)\n        :returns: True if broker will start or continue sending; False if not\n        :rtype: bool\n\n        '
        with _CallbackResult(self._FlowOkCallbackResultArgs) as flow_ok_result:
            self._impl.flow(active=active, callback=flow_ok_result.set_value_once)
            self._flush_output(flow_ok_result.is_ready)
            return flow_ok_result.value.active

    def add_on_cancel_callback(self, callback):
        if False:
            i = 10
            return i + 15
        "Pass a callback function that will be called when Basic.Cancel\n        is sent by the broker. The callback function should receive a method\n        frame parameter.\n\n        :param callable callback: a callable for handling broker's Basic.Cancel\n            notification with the call signature: callback(method_frame)\n            where method_frame is of type `pika.frame.Method` with method of\n            type `spec.Basic.Cancel`\n\n        "
        self._impl.callbacks.add(self.channel_number, self._CONSUMER_CANCELLED_CB_KEY, callback, one_shot=False)

    def add_on_return_callback(self, callback):
        if False:
            return 10
        'Pass a callback function that will be called when a published\n        message is rejected and returned by the server via `Basic.Return`.\n\n        :param callable callback: The method to call on callback with the\n            signature callback(channel, method, properties, body), where\n            - channel: pika.Channel\n            - method: pika.spec.Basic.Return\n            - properties: pika.spec.BasicProperties\n            - body: bytes\n\n        '
        self._impl.add_on_return_callback(lambda _channel, method, properties, body: self._add_pending_event(_ReturnedMessageEvt(callback, self, method, properties, body)))

    def basic_consume(self, queue, on_message_callback, auto_ack=False, exclusive=False, consumer_tag=None, arguments=None):
        if False:
            return 10
        "Sends the AMQP command Basic.Consume to the broker and binds messages\n        for the consumer_tag to the consumer callback. If you do not pass in\n        a consumer_tag, one will be automatically generated for you. Returns\n        the consumer tag.\n\n        NOTE: the consumer callbacks are dispatched only in the scope of\n        specially-designated methods: see\n        `BlockingConnection.process_data_events` and\n        `BlockingChannel.start_consuming`.\n\n        For more information about Basic.Consume, see:\n        http://www.rabbitmq.com/amqp-0-9-1-reference.html#basic.consume\n\n        :param str queue: The queue from which to consume\n        :param callable on_message_callback: Required function for dispatching messages\n            to user, having the signature:\n            on_message_callback(channel, method, properties, body)\n            - channel: BlockingChannel\n            - method: spec.Basic.Deliver\n            - properties: spec.BasicProperties\n            - body: bytes\n        :param bool auto_ack: if set to True, automatic acknowledgement mode will be used\n                              (see http://www.rabbitmq.com/confirms.html). This corresponds\n                              with the 'no_ack' parameter in the basic.consume AMQP 0.9.1\n                              method\n        :param bool exclusive: Don't allow other consumers on the queue\n        :param str consumer_tag: You may specify your own consumer tag; if left\n          empty, a consumer tag will be generated automatically\n        :param dict arguments: Custom key/value pair arguments for the consumer\n        :returns: consumer tag\n        :rtype: str\n        :raises pika.exceptions.DuplicateConsumerTag: if consumer with given\n            consumer_tag is already present.\n\n        "
        validators.require_string(queue, 'queue')
        validators.require_callback(on_message_callback, 'on_message_callback')
        return self._basic_consume_impl(queue=queue, on_message_callback=on_message_callback, auto_ack=auto_ack, exclusive=exclusive, consumer_tag=consumer_tag, arguments=arguments)

    def _basic_consume_impl(self, queue, auto_ack, exclusive, consumer_tag, arguments=None, on_message_callback=None, alternate_event_sink=None):
        if False:
            print('Hello World!')
        "The low-level implementation used by `basic_consume` and `consume`.\n        See `basic_consume` docstring for more info.\n\n        NOTE: exactly one of on_message_callback/alternate_event_sink musts be\n        non-None.\n\n        This method has one additional parameter alternate_event_sink over the\n        args described in `basic_consume`.\n\n        :param callable alternate_event_sink: if specified, _ConsumerDeliveryEvt\n            and _ConsumerCancellationEvt objects will be diverted to this\n            callback instead of being deposited in the channel's\n            `_pending_events` container. Signature:\n            alternate_event_sink(evt)\n\n        :raises pika.exceptions.DuplicateConsumerTag: if consumer with given\n            consumer_tag is already present.\n\n        "
        if (on_message_callback is None) == (alternate_event_sink is None):
            raise ValueError(('exactly one of on_message_callback/alternate_event_sink must be non-None', on_message_callback, alternate_event_sink))
        if not consumer_tag:
            consumer_tag = self._impl._generate_consumer_tag()
        if consumer_tag in self._consumer_infos:
            raise exceptions.DuplicateConsumerTag(consumer_tag)
        self._consumer_infos[consumer_tag] = _ConsumerInfo(consumer_tag, auto_ack=auto_ack, on_message_callback=on_message_callback, alternate_event_sink=alternate_event_sink)
        try:
            with self._basic_consume_ok_result as ok_result:
                tag = self._impl.basic_consume(on_message_callback=self._on_consumer_message_delivery, queue=queue, auto_ack=auto_ack, exclusive=exclusive, consumer_tag=consumer_tag, arguments=arguments)
                assert tag == consumer_tag, (tag, consumer_tag)
                self._flush_output(ok_result.is_ready)
        except Exception:
            if consumer_tag in self._consumer_infos:
                del self._consumer_infos[consumer_tag]
                self.connection._request_channel_dispatch(-self.channel_number)
            raise
        if self._consumer_infos[consumer_tag].setting_up:
            self._consumer_infos[consumer_tag].state = _ConsumerInfo.ACTIVE
        return consumer_tag

    def basic_cancel(self, consumer_tag):
        if False:
            return 10
        "This method cancels a consumer. This does not affect already\n        delivered messages, but it does mean the server will not send any more\n        messages for that consumer. The client may receive an arbitrary number\n        of messages in between sending the cancel method and receiving the\n        cancel-ok reply.\n\n        NOTE: When cancelling an auto_ack=False consumer, this implementation\n        automatically Nacks and suppresses any incoming messages that have not\n        yet been dispatched to the consumer's callback. However, when cancelling\n        a auto_ack=True consumer, this method will return any pending messages\n        that arrived before broker confirmed the cancellation.\n\n        :param str consumer_tag: Identifier for the consumer; the result of\n            passing a consumer_tag that was created on another channel is\n            undefined (bad things will happen)\n        :returns: (NEW IN pika 0.10.0) empty sequence for a auto_ack=False\n            consumer; for a auto_ack=True consumer, returns a (possibly empty)\n            sequence of pending messages that arrived before broker confirmed\n            the cancellation (this is done instead of via consumer's callback in\n            order to prevent reentrancy/recursion. Each message is four-tuple:\n            (channel, method, properties, body)\n            - channel: BlockingChannel\n            - method: spec.Basic.Deliver\n            - properties: spec.BasicProperties\n            - body: bytes\n        :rtype: list\n        "
        try:
            consumer_info = self._consumer_infos[consumer_tag]
        except KeyError:
            LOGGER.warning('User is attempting to cancel an unknown consumer=%s; already cancelled by user or broker?', consumer_tag)
            return []
        try:
            assert consumer_info.active or consumer_info.cancelled_by_broker, consumer_info.state
            assert consumer_info.cancelled_by_broker or consumer_tag in self._impl._consumers, consumer_tag
            auto_ack = consumer_info.auto_ack
            consumer_info.state = _ConsumerInfo.TEARING_DOWN
            with _CallbackResult() as cancel_ok_result:
                if not auto_ack:
                    pending_messages = self._remove_pending_deliveries(consumer_tag)
                    if pending_messages:
                        for message in pending_messages:
                            self._impl.basic_reject(message.method.delivery_tag, requeue=True)
                self._impl.basic_cancel(consumer_tag=consumer_tag, callback=cancel_ok_result.signal_once)
                self._flush_output(cancel_ok_result.is_ready, lambda : consumer_tag not in self._impl._consumers)
            if auto_ack:
                return [(evt.method, evt.properties, evt.body) for evt in self._remove_pending_deliveries(consumer_tag)]
            else:
                messages = self._remove_pending_deliveries(consumer_tag)
                assert not messages, messages
                return []
        finally:
            if consumer_tag in self._consumer_infos:
                del self._consumer_infos[consumer_tag]
                self.connection._request_channel_dispatch(-self.channel_number)

    def _remove_pending_deliveries(self, consumer_tag):
        if False:
            print('Hello World!')
        'Extract _ConsumerDeliveryEvt objects destined for the given consumer\n        from pending events, discarding the _ConsumerCancellationEvt, if any\n\n        :param str consumer_tag:\n\n        :returns: a (possibly empty) sequence of _ConsumerDeliveryEvt destined\n            for the given consumer tag\n        :rtype: list\n        '
        remaining_events = deque()
        unprocessed_messages = []
        while self._pending_events:
            evt = self._pending_events.popleft()
            if type(evt) is _ConsumerDeliveryEvt:
                if evt.method.consumer_tag == consumer_tag:
                    unprocessed_messages.append(evt)
                    continue
            if type(evt) is _ConsumerCancellationEvt:
                if evt.method_frame.method.consumer_tag == consumer_tag:
                    continue
            remaining_events.append(evt)
        self._pending_events = remaining_events
        return unprocessed_messages

    def start_consuming(self):
        if False:
            while True:
                i = 10
        'Processes I/O events and dispatches timers and `basic_consume`\n        callbacks until all consumers are cancelled.\n\n        NOTE: this blocking function may not be called from the scope of a\n        pika callback, because dispatching `basic_consume` callbacks from this\n        context would constitute recursion.\n\n        :raises pika.exceptions.ReentrancyError: if called from the scope of a\n            `BlockingConnection` or `BlockingChannel` callback\n        :raises ChannelClosed: when this channel is closed by broker.\n        '
        with self.connection._acquire_event_dispatch() as dispatch_allowed:
            if not dispatch_allowed:
                raise exceptions.ReentrancyError('start_consuming may not be called from the scope of another BlockingConnection or BlockingChannel callback')
        self._impl._raise_if_not_open()
        while self._consumer_infos:
            self._process_data_events(time_limit=None)

    def stop_consuming(self, consumer_tag=None):
        if False:
            return 10
        ' Cancels all consumers, signalling the `start_consuming` loop to\n        exit.\n\n        NOTE: pending non-ackable messages will be lost; pending ackable\n        messages will be rejected.\n\n        '
        if consumer_tag:
            self.basic_cancel(consumer_tag)
        else:
            self._cancel_all_consumers()

    def consume(self, queue, auto_ack=False, exclusive=False, arguments=None, inactivity_timeout=None):
        if False:
            print('Hello World!')
        "Blocking consumption of a queue instead of via a callback. This\n        method is a generator that yields each message as a tuple of method,\n        properties, and body. The active generator iterator terminates when the\n        consumer is cancelled by client via `BlockingChannel.cancel()` or by\n        broker.\n\n        Example:\n        ::\n            for method, properties, body in channel.consume('queue'):\n                print(body)\n                channel.basic_ack(method.delivery_tag)\n\n        You should call `BlockingChannel.cancel()` when you escape out of the\n        generator loop.\n\n        If you don't cancel this consumer, then next call on the same channel\n        to `consume()` with the exact same (queue, auto_ack, exclusive) parameters\n        will resume the existing consumer generator; however, calling with\n        different parameters will result in an exception.\n\n        :param str queue: The queue name to consume\n        :param bool auto_ack: Tell the broker to not expect a ack/nack response\n        :param bool exclusive: Don't allow other consumers on the queue\n        :param dict arguments: Custom key/value pair arguments for the consumer\n        :param float inactivity_timeout: if a number is given (in\n            seconds), will cause the method to yield (None, None, None) after the\n            given period of inactivity; this permits for pseudo-regular maintenance\n            activities to be carried out by the user while waiting for messages\n            to arrive. If None is given (default), then the method blocks until\n            the next event arrives. NOTE that timing granularity is limited by\n            the timer resolution of the underlying implementation.\n            NEW in pika 0.10.0.\n\n        :yields: tuple(spec.Basic.Deliver, spec.BasicProperties, str or unicode)\n\n        :raises ValueError: if consumer-creation parameters don't match those\n            of the existing queue consumer generator, if any.\n            NEW in pika 0.10.0\n        :raises ChannelClosed: when this channel is closed by broker.\n\n        "
        self._impl._raise_if_not_open()
        params = (queue, auto_ack, exclusive)
        if self._queue_consumer_generator is not None:
            if params != self._queue_consumer_generator.params:
                raise ValueError('Consume with different params not allowed on existing queue consumer generator; previous params: %r; new params: %r' % (self._queue_consumer_generator.params, (queue, auto_ack, exclusive)))
        else:
            LOGGER.debug('Creating new queue consumer generator; params: %r', params)
            consumer_tag = self._impl._generate_consumer_tag()
            self._queue_consumer_generator = _QueueConsumerGeneratorInfo(params, consumer_tag)
            try:
                self._basic_consume_impl(queue=queue, auto_ack=auto_ack, exclusive=exclusive, consumer_tag=consumer_tag, arguments=arguments, alternate_event_sink=self._on_consumer_generator_event)
            except Exception:
                self._queue_consumer_generator = None
                raise
            LOGGER.info('Created new queue consumer generator %r', self._queue_consumer_generator)
        while self._queue_consumer_generator is not None:
            if self._queue_consumer_generator.pending_events:
                evt = self._queue_consumer_generator.pending_events.popleft()
                if type(evt) is _ConsumerCancellationEvt:
                    self._queue_consumer_generator = None
                    break
                else:
                    yield (evt.method, evt.properties, evt.body)
                    continue
            if inactivity_timeout is None:
                self._process_data_events(time_limit=None)
                continue
            wait_start_time = compat.time_now()
            wait_deadline = wait_start_time + inactivity_timeout
            delta = inactivity_timeout
            while self._queue_consumer_generator is not None and (not self._queue_consumer_generator.pending_events):
                self._process_data_events(time_limit=delta)
                if not self._queue_consumer_generator:
                    break
                if self._queue_consumer_generator.pending_events:
                    break
                delta = wait_deadline - compat.time_now()
                if delta <= 0.0:
                    yield (None, None, None)
                    break

    def _process_data_events(self, time_limit):
        if False:
            for i in range(10):
                print('nop')
        "Wrapper for `BlockingConnection.process_data_events()` with common\n        channel-specific logic that raises ChannelClosed if broker closed this\n        channel.\n\n        NOTE: We need to raise an exception in the context of user's call into\n        our API to protect the integrity of the underlying implementation.\n        BlockingConnection's underlying asynchronous connection adapter\n        (SelectConnection) uses callbacks to communicate with us. If\n        BlockingConnection leaks exceptions back into the I/O loop or the\n        asynchronous connection adapter, we interrupt their normal workflow and\n        introduce a high likelihood of state inconsistency.\n\n        See `BlockingConnection.process_data_events()` for documentation of args\n        and behavior.\n\n        :param float time_limit:\n\n        "
        self.connection.process_data_events(time_limit=time_limit)
        if self.is_closed and isinstance(self._closing_reason, exceptions.ChannelClosedByBroker):
            LOGGER.debug('Channel close by broker detected, raising %r; %r', self._closing_reason, self)
            raise self._closing_reason

    def get_waiting_message_count(self):
        if False:
            while True:
                i = 10
        'Returns the number of messages that may be retrieved from the current\n        queue consumer generator via `BlockingChannel.consume` without blocking.\n        NEW in pika 0.10.0\n\n        :returns: The number of waiting messages\n        :rtype: int\n        '
        if self._queue_consumer_generator is not None:
            pending_events = self._queue_consumer_generator.pending_events
            count = len(pending_events)
            if count and type(pending_events[-1]) is _ConsumerCancellationEvt:
                count -= 1
        else:
            count = 0
        return count

    def cancel(self):
        if False:
            i = 10
            return i + 15
        "Cancel the queue consumer created by `BlockingChannel.consume`,\n        rejecting all pending ackable messages.\n\n        NOTE: If you're looking to cancel a consumer issued with\n        BlockingChannel.basic_consume then you should call\n        BlockingChannel.basic_cancel.\n\n        :returns: The number of messages requeued by Basic.Nack.\n            NEW in 0.10.0: returns 0\n        :rtype: int\n\n        "
        if self._queue_consumer_generator is None:
            LOGGER.warning('cancel: queue consumer generator is inactive (already cancelled by client or broker?)')
            return 0
        try:
            (_, auto_ack, _) = self._queue_consumer_generator.params
            if not auto_ack:
                pending_events = self._queue_consumer_generator.pending_events
                for _ in compat.xrange(self.get_waiting_message_count()):
                    evt = pending_events.popleft()
                    self._impl.basic_reject(evt.method.delivery_tag, requeue=True)
            self.basic_cancel(self._queue_consumer_generator.consumer_tag)
        finally:
            self._queue_consumer_generator = None
        return 0

    def basic_ack(self, delivery_tag=0, multiple=False):
        if False:
            for i in range(10):
                print('nop')
        'Acknowledge one or more messages. When sent by the client, this\n        method acknowledges one or more messages delivered via the Deliver or\n        Get-Ok methods. When sent by server, this method acknowledges one or\n        more messages published with the Publish method on a channel in\n        confirm mode. The acknowledgement can be for a single message or a\n        set of messages up to and including a specific message.\n\n        :param int delivery_tag: The server-assigned delivery tag\n        :param bool multiple: If set to True, the delivery tag is treated as\n                              "up to and including", so that multiple messages\n                              can be acknowledged with a single method. If set\n                              to False, the delivery tag refers to a single\n                              message. If the multiple field is 1, and the\n                              delivery tag is zero, this indicates\n                              acknowledgement of all outstanding messages.\n        '
        self._impl.basic_ack(delivery_tag=delivery_tag, multiple=multiple)
        self._flush_output()

    def basic_nack(self, delivery_tag=0, multiple=False, requeue=True):
        if False:
            print('Hello World!')
        'This method allows a client to reject one or more incoming messages.\n        It can be used to interrupt and cancel large incoming messages, or\n        return untreatable messages to their original queue.\n\n        :param int delivery_tag: The server-assigned delivery tag\n        :param bool multiple: If set to True, the delivery tag is treated as\n                              "up to and including", so that multiple messages\n                              can be acknowledged with a single method. If set\n                              to False, the delivery tag refers to a single\n                              message. If the multiple field is 1, and the\n                              delivery tag is zero, this indicates\n                              acknowledgement of all outstanding messages.\n        :param bool requeue: If requeue is true, the server will attempt to\n                             requeue the message. If requeue is false or the\n                             requeue attempt fails the messages are discarded or\n                             dead-lettered.\n\n        '
        self._impl.basic_nack(delivery_tag=delivery_tag, multiple=multiple, requeue=requeue)
        self._flush_output()

    def basic_get(self, queue, auto_ack=False):
        if False:
            i = 10
            return i + 15
        'Get a single message from the AMQP broker. Returns a sequence with\n        the method frame, message properties, and body.\n\n        :param str queue: Name of queue from which to get a message\n        :param bool auto_ack: Tell the broker to not expect a reply\n        :returns: a three-tuple; (None, None, None) if the queue was empty;\n            otherwise (method, properties, body); NOTE: body may be None\n        :rtype: (spec.Basic.GetOk|None, spec.BasicProperties|None, bytes|None)\n        '
        assert not self._basic_getempty_result
        validators.require_string(queue, 'queue')
        with _CallbackResult(self._RxMessageArgs) as get_ok_result:
            with self._basic_getempty_result:
                self._impl.basic_get(queue=queue, auto_ack=auto_ack, callback=get_ok_result.set_value_once)
                self._flush_output(get_ok_result.is_ready, self._basic_getempty_result.is_ready)
                if get_ok_result:
                    evt = get_ok_result.value
                    return (evt.method, evt.properties, evt.body)
                else:
                    assert self._basic_getempty_result, 'wait completed without GetOk and GetEmpty'
                    return (None, None, None)

    def basic_publish(self, exchange, routing_key, body, properties=None, mandatory=False):
        if False:
            return 10
        "Publish to the channel with the given exchange, routing key, and\n        body.\n\n        For more information on basic_publish and what the parameters do, see:\n\n            http://www.rabbitmq.com/amqp-0-9-1-reference.html#basic.publish\n\n        NOTE: mandatory may be enabled even without delivery\n          confirmation, but in the absence of delivery confirmation the\n          synchronous implementation has no way to know how long to wait for\n          the Basic.Return.\n\n        :param str exchange: The exchange to publish to\n        :param str routing_key: The routing key to bind on\n        :param bytes body: The message body; empty string if no body\n        :param pika.spec.BasicProperties properties: message properties\n        :param bool mandatory: The mandatory flag\n\n        :raises UnroutableError: raised when a message published in\n            publisher-acknowledgments mode (see\n            `BlockingChannel.confirm_delivery`) is returned via `Basic.Return`\n            followed by `Basic.Ack`.\n        :raises NackError: raised when a message published in\n            publisher-acknowledgements mode is Nack'ed by the broker. See\n            `BlockingChannel.confirm_delivery`.\n\n        "
        if self._delivery_confirmation:
            with self._message_confirmation_result:
                self._impl.basic_publish(exchange=exchange, routing_key=routing_key, body=body, properties=properties, mandatory=mandatory)
                self._flush_output(self._message_confirmation_result.is_ready)
                conf_method = self._message_confirmation_result.value.method_frame.method
                if isinstance(conf_method, pika.spec.Basic.Nack):
                    LOGGER.warning("Message was Nack'ed by broker: nack=%r; channel=%s; exchange=%s; routing_key=%s; mandatory=%r; ", conf_method, self.channel_number, exchange, routing_key, mandatory)
                    if self._puback_return is not None:
                        returned_messages = [self._puback_return]
                        self._puback_return = None
                    else:
                        returned_messages = []
                    raise exceptions.NackError(returned_messages)
                else:
                    assert isinstance(conf_method, pika.spec.Basic.Ack), conf_method
                    if self._puback_return is not None:
                        messages = [self._puback_return]
                        self._puback_return = None
                        raise exceptions.UnroutableError(messages)
        else:
            self._impl.basic_publish(exchange=exchange, routing_key=routing_key, body=body, properties=properties, mandatory=mandatory)
            self._flush_output()

    def basic_qos(self, prefetch_size=0, prefetch_count=0, global_qos=False):
        if False:
            while True:
                i = 10
        'Specify quality of service. This method requests a specific quality\n        of service. The QoS can be specified for the current channel or for all\n        channels on the connection. The client can request that messages be sent\n        in advance so that when the client finishes processing a message, the\n        following message is already held locally, rather than needing to be\n        sent down the channel. Prefetching gives a performance improvement.\n\n        :param int prefetch_size:  This field specifies the prefetch window\n                                   size. The server will send a message in\n                                   advance if it is equal to or smaller in size\n                                   than the available prefetch size (and also\n                                   falls into other prefetch limits). May be set\n                                   to zero, meaning "no specific limit",\n                                   although other prefetch limits may still\n                                   apply. The prefetch-size is ignored if the\n                                   no-ack option is set in the consumer.\n        :param int prefetch_count: Specifies a prefetch window in terms of whole\n                                   messages. This field may be used in\n                                   combination with the prefetch-size field; a\n                                   message will only be sent in advance if both\n                                   prefetch windows (and those at the channel\n                                   and connection level) allow it. The\n                                   prefetch-count is ignored if the no-ack\n                                   option is set in the consumer.\n        :param bool global_qos:    Should the QoS apply to all channels on the\n                                   connection.\n\n        '
        with _CallbackResult() as qos_ok_result:
            self._impl.basic_qos(callback=qos_ok_result.signal_once, prefetch_size=prefetch_size, prefetch_count=prefetch_count, global_qos=global_qos)
            self._flush_output(qos_ok_result.is_ready)

    def basic_recover(self, requeue=False):
        if False:
            print('Hello World!')
        'This method asks the server to redeliver all unacknowledged messages\n        on a specified channel. Zero or more messages may be redelivered. This\n        method replaces the asynchronous Recover.\n\n        :param bool requeue: If False, the message will be redelivered to the\n                             original recipient. If True, the server will\n                             attempt to requeue the message, potentially then\n                             delivering it to an alternative subscriber.\n\n        '
        with _CallbackResult() as recover_ok_result:
            self._impl.basic_recover(requeue=requeue, callback=recover_ok_result.signal_once)
            self._flush_output(recover_ok_result.is_ready)

    def basic_reject(self, delivery_tag=0, requeue=True):
        if False:
            i = 10
            return i + 15
        'Reject an incoming message. This method allows a client to reject a\n        message. It can be used to interrupt and cancel large incoming messages,\n        or return untreatable messages to their original queue.\n\n        :param int delivery_tag: The server-assigned delivery tag\n        :param bool requeue: If requeue is true, the server will attempt to\n                             requeue the message. If requeue is false or the\n                             requeue attempt fails the messages are discarded or\n                             dead-lettered.\n\n        '
        self._impl.basic_reject(delivery_tag=delivery_tag, requeue=requeue)
        self._flush_output()

    def confirm_delivery(self):
        if False:
            for i in range(10):
                print('nop')
        'Turn on RabbitMQ-proprietary Confirm mode in the channel.\n\n        For more information see:\n            https://www.rabbitmq.com/confirms.html\n        '
        if self._delivery_confirmation:
            LOGGER.error('confirm_delivery: confirmation was already enabled on channel=%s', self.channel_number)
            return
        with _CallbackResult() as select_ok_result:
            self._impl.confirm_delivery(ack_nack_callback=self._message_confirmation_result.set_value_once, callback=select_ok_result.signal_once)
            self._flush_output(select_ok_result.is_ready)
        self._delivery_confirmation = True
        self._impl.add_on_return_callback(self._on_puback_message_returned)

    def exchange_declare(self, exchange, exchange_type=ExchangeType.direct, passive=False, durable=False, auto_delete=False, internal=False, arguments=None):
        if False:
            print('Hello World!')
        'This method creates an exchange if it does not already exist, and if\n        the exchange exists, verifies that it is of the correct and expected\n        class.\n\n        If passive set, the server will reply with Declare-Ok if the exchange\n        already exists with the same name, and raise an error if not and if the\n        exchange does not already exist, the server MUST raise a channel\n        exception with reply code 404 (not found).\n\n        :param str exchange: The exchange name consists of a non-empty sequence of\n                          these characters: letters, digits, hyphen, underscore,\n                          period, or colon.\n        :param str exchange_type: The exchange type to use\n        :param bool passive: Perform a declare or just check to see if it exists\n        :param bool durable: Survive a reboot of RabbitMQ\n        :param bool auto_delete: Remove when no more queues are bound to it\n        :param bool internal: Can only be published to by other exchanges\n        :param dict arguments: Custom key/value pair arguments for the exchange\n        :returns: Method frame from the Exchange.Declare-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Exchange.DeclareOk`\n\n        '
        validators.require_string(exchange, 'exchange')
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as declare_ok_result:
            self._impl.exchange_declare(exchange=exchange, exchange_type=exchange_type, passive=passive, durable=durable, auto_delete=auto_delete, internal=internal, arguments=arguments, callback=declare_ok_result.set_value_once)
            self._flush_output(declare_ok_result.is_ready)
            return declare_ok_result.value.method_frame

    def exchange_delete(self, exchange=None, if_unused=False):
        if False:
            for i in range(10):
                print('nop')
        'Delete the exchange.\n\n        :param str exchange: The exchange name\n        :param bool if_unused: only delete if the exchange is unused\n        :returns: Method frame from the Exchange.Delete-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Exchange.DeleteOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as delete_ok_result:
            self._impl.exchange_delete(exchange=exchange, if_unused=if_unused, callback=delete_ok_result.set_value_once)
            self._flush_output(delete_ok_result.is_ready)
            return delete_ok_result.value.method_frame

    def exchange_bind(self, destination, source, routing_key='', arguments=None):
        if False:
            while True:
                i = 10
        'Bind an exchange to another exchange.\n\n        :param str destination: The destination exchange to bind\n        :param str source: The source exchange to bind to\n        :param str routing_key: The routing key to bind on\n        :param dict arguments: Custom key/value pair arguments for the binding\n        :returns: Method frame from the Exchange.Bind-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n          `spec.Exchange.BindOk`\n\n        '
        validators.require_string(destination, 'destination')
        validators.require_string(source, 'source')
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as bind_ok_result:
            self._impl.exchange_bind(destination=destination, source=source, routing_key=routing_key, arguments=arguments, callback=bind_ok_result.set_value_once)
            self._flush_output(bind_ok_result.is_ready)
            return bind_ok_result.value.method_frame

    def exchange_unbind(self, destination=None, source=None, routing_key='', arguments=None):
        if False:
            return 10
        'Unbind an exchange from another exchange.\n\n        :param str destination: The destination exchange to unbind\n        :param str source: The source exchange to unbind from\n        :param str routing_key: The routing key to unbind\n        :param dict arguments: Custom key/value pair arguments for the binding\n        :returns: Method frame from the Exchange.Unbind-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Exchange.UnbindOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as unbind_ok_result:
            self._impl.exchange_unbind(destination=destination, source=source, routing_key=routing_key, arguments=arguments, callback=unbind_ok_result.set_value_once)
            self._flush_output(unbind_ok_result.is_ready)
            return unbind_ok_result.value.method_frame

    def queue_declare(self, queue, passive=False, durable=False, exclusive=False, auto_delete=False, arguments=None):
        if False:
            return 10
        "Declare queue, create if needed. This method creates or checks a\n        queue. When creating a new queue the client can specify various\n        properties that control the durability of the queue and its contents,\n        and the level of sharing for the queue.\n\n        Use an empty string as the queue name for the broker to auto-generate\n        one. Retrieve this auto-generated queue name from the returned\n        `spec.Queue.DeclareOk` method frame.\n\n        :param str queue: The queue name; if empty string, the broker will\n            create a unique queue name\n        :param bool passive: Only check to see if the queue exists and raise\n          `ChannelClosed` if it doesn't\n        :param bool durable: Survive reboots of the broker\n        :param bool exclusive: Only allow access by the current connection\n        :param bool auto_delete: Delete after consumer cancels or disconnects\n        :param dict arguments: Custom key/value arguments for the queue\n        :returns: Method frame from the Queue.Declare-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Queue.DeclareOk`\n\n        "
        validators.require_string(queue, 'queue')
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as declare_ok_result:
            self._impl.queue_declare(queue=queue, passive=passive, durable=durable, exclusive=exclusive, auto_delete=auto_delete, arguments=arguments, callback=declare_ok_result.set_value_once)
            self._flush_output(declare_ok_result.is_ready)
            return declare_ok_result.value.method_frame

    def queue_delete(self, queue, if_unused=False, if_empty=False):
        if False:
            while True:
                i = 10
        "Delete a queue from the broker.\n\n        :param str queue: The queue to delete\n        :param bool if_unused: only delete if it's unused\n        :param bool if_empty: only delete if the queue is empty\n        :returns: Method frame from the Queue.Delete-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Queue.DeleteOk`\n\n        "
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as delete_ok_result:
            self._impl.queue_delete(queue=queue, if_unused=if_unused, if_empty=if_empty, callback=delete_ok_result.set_value_once)
            self._flush_output(delete_ok_result.is_ready)
            return delete_ok_result.value.method_frame

    def queue_purge(self, queue):
        if False:
            return 10
        'Purge all of the messages from the specified queue\n\n        :param str queue: The queue to purge\n        :returns: Method frame from the Queue.Purge-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Queue.PurgeOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as purge_ok_result:
            self._impl.queue_purge(queue=queue, callback=purge_ok_result.set_value_once)
            self._flush_output(purge_ok_result.is_ready)
            return purge_ok_result.value.method_frame

    def queue_bind(self, queue, exchange, routing_key=None, arguments=None):
        if False:
            i = 10
            return i + 15
        'Bind the queue to the specified exchange\n\n        :param str queue: The queue to bind to the exchange\n        :param str exchange: The source exchange to bind to\n        :param str routing_key: The routing key to bind on\n        :param dict arguments: Custom key/value pair arguments for the binding\n\n        :returns: Method frame from the Queue.Bind-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Queue.BindOk`\n\n        '
        validators.require_string(queue, 'queue')
        validators.require_string(exchange, 'exchange')
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as bind_ok_result:
            self._impl.queue_bind(queue=queue, exchange=exchange, routing_key=routing_key, arguments=arguments, callback=bind_ok_result.set_value_once)
            self._flush_output(bind_ok_result.is_ready)
            return bind_ok_result.value.method_frame

    def queue_unbind(self, queue, exchange=None, routing_key=None, arguments=None):
        if False:
            i = 10
            return i + 15
        'Unbind a queue from an exchange.\n\n        :param str queue: The queue to unbind from the exchange\n        :param str exchange: The source exchange to bind from\n        :param str routing_key: The routing key to unbind\n        :param dict arguments: Custom key/value pair arguments for the binding\n\n        :returns: Method frame from the Queue.Unbind-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Queue.UnbindOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as unbind_ok_result:
            self._impl.queue_unbind(queue=queue, exchange=exchange, routing_key=routing_key, arguments=arguments, callback=unbind_ok_result.set_value_once)
            self._flush_output(unbind_ok_result.is_ready)
            return unbind_ok_result.value.method_frame

    def tx_select(self):
        if False:
            while True:
                i = 10
        'Select standard transaction mode. This method sets the channel to use\n        standard transactions. The client must use this method at least once on\n        a channel before using the Commit or Rollback methods.\n\n        :returns: Method frame from the Tx.Select-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Tx.SelectOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as select_ok_result:
            self._impl.tx_select(select_ok_result.set_value_once)
            self._flush_output(select_ok_result.is_ready)
            return select_ok_result.value.method_frame

    def tx_commit(self):
        if False:
            i = 10
            return i + 15
        'Commit a transaction.\n\n        :returns: Method frame from the Tx.Commit-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Tx.CommitOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as commit_ok_result:
            self._impl.tx_commit(commit_ok_result.set_value_once)
            self._flush_output(commit_ok_result.is_ready)
            return commit_ok_result.value.method_frame

    def tx_rollback(self):
        if False:
            for i in range(10):
                print('nop')
        'Rollback a transaction.\n\n        :returns: Method frame from the Tx.Commit-ok response\n        :rtype: `pika.frame.Method` having `method` attribute of type\n            `spec.Tx.CommitOk`\n\n        '
        with _CallbackResult(self._MethodFrameCallbackResultArgs) as rollback_ok_result:
            self._impl.tx_rollback(rollback_ok_result.set_value_once)
            self._flush_output(rollback_ok_result.is_ready)
            return rollback_ok_result.value.method_frame