"""A utility class for event-based messaging on a zmq socket using tornado.

.. seealso::

    - :mod:`zmq.asyncio`
    - :mod:`zmq.eventloop.future`
"""
import asyncio
import pickle
import warnings
from queue import Queue
from typing import Any, Awaitable, Callable, List, Optional, Sequence, Union, cast, overload
from tornado.ioloop import IOLoop
from tornado.log import gen_log
import zmq
import zmq._future
from zmq import POLLIN, POLLOUT
from zmq._typing import Literal
from zmq.utils import jsonapi

class ZMQStream:
    """A utility class to register callbacks when a zmq socket sends and receives

    For use with tornado IOLoop.

    There are three main methods

    Methods:

    * **on_recv(callback, copy=True):**
        register a callback to be run every time the socket has something to receive
    * **on_send(callback):**
        register a callback to be run every time you call send
    * **send_multipart(self, msg, flags=0, copy=False, callback=None):**
        perform a send that will trigger the callback
        if callback is passed, on_send is also called.

        There are also send_multipart(), send_json(), send_pyobj()

    Three other methods for deactivating the callbacks:

    * **stop_on_recv():**
        turn off the recv callback
    * **stop_on_send():**
        turn off the send callback

    which simply call ``on_<evt>(None)``.

    The entire socket interface, excluding direct recv methods, is also
    provided, primarily through direct-linking the methods.
    e.g.

    >>> stream.bind is stream.socket.bind
    True


    .. versionadded:: 25

        send/recv callbacks can be coroutines.

    .. versionchanged:: 25

        ZMQStreams only support base zmq.Socket classes (this has always been true, but not enforced).
        If ZMQStreams are created with e.g. async Socket subclasses,
        a RuntimeWarning will be shown,
        and the socket cast back to the default zmq.Socket
        before connecting events.

        Previously, using async sockets (or any zmq.Socket subclass) would result in undefined behavior for the
        arguments passed to callback functions.
        Now, the callback functions reliably get the return value of the base `zmq.Socket` send/recv_multipart methods
        (the list of message frames).
    """
    socket: zmq.Socket
    io_loop: IOLoop
    poller: zmq.Poller
    _send_queue: Queue
    _recv_callback: Optional[Callable]
    _send_callback: Optional[Callable]
    _close_callback: Optional[Callable]
    _state: int = 0
    _flushed: bool = False
    _recv_copy: bool = False
    _fd: int

    def __init__(self, socket: 'zmq.Socket', io_loop: Optional[IOLoop]=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(socket, zmq._future._AsyncSocket):
            warnings.warn(f'ZMQStream only supports the base zmq.Socket class.\n\n                Use zmq.Socket(shadow=other_socket)\n                or `ctx.socket(zmq.{socket._type_name}, socket_class=zmq.Socket)`\n                to create a base zmq.Socket object,\n                no matter what other kind of socket your Context creates.\n                ', RuntimeWarning, stacklevel=2)
            socket = zmq.Socket(shadow=socket)
        self.socket = socket
        self.io_loop = io_loop or IOLoop.current()
        self.poller = zmq.Poller()
        self._fd = cast(int, self.socket.FD)
        self._send_queue = Queue()
        self._recv_callback = None
        self._send_callback = None
        self._close_callback = None
        self._recv_copy = False
        self._flushed = False
        self._state = 0
        self._init_io_state()
        self.bind = self.socket.bind
        self.bind_to_random_port = self.socket.bind_to_random_port
        self.connect = self.socket.connect
        self.setsockopt = self.socket.setsockopt
        self.getsockopt = self.socket.getsockopt
        self.setsockopt_string = self.socket.setsockopt_string
        self.getsockopt_string = self.socket.getsockopt_string
        self.setsockopt_unicode = self.socket.setsockopt_unicode
        self.getsockopt_unicode = self.socket.getsockopt_unicode

    def stop_on_recv(self):
        if False:
            while True:
                i = 10
        'Disable callback and automatic receiving.'
        return self.on_recv(None)

    def stop_on_send(self):
        if False:
            return 10
        'Disable callback on sending.'
        return self.on_send(None)

    def stop_on_err(self):
        if False:
            return 10
        'DEPRECATED, does nothing'
        gen_log.warn('on_err does nothing, and will be removed')

    def on_err(self, callback: Callable):
        if False:
            print('Hello World!')
        'DEPRECATED, does nothing'
        gen_log.warn('on_err does nothing, and will be removed')

    @overload
    def on_recv(self, callback: Callable[[List[bytes]], Any]) -> None:
        if False:
            i = 10
            return i + 15
        ...

    @overload
    def on_recv(self, callback: Callable[[List[bytes]], Any], copy: Literal[True]) -> None:
        if False:
            while True:
                i = 10
        ...

    @overload
    def on_recv(self, callback: Callable[[List[zmq.Frame]], Any], copy: Literal[False]) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def on_recv(self, callback: Union[Callable[[List[zmq.Frame]], Any], Callable[[List[bytes]], Any]], copy: bool=...):
        if False:
            i = 10
            return i + 15
        ...

    def on_recv(self, callback: Union[Callable[[List[zmq.Frame]], Any], Callable[[List[bytes]], Any]], copy: bool=True) -> None:
        if False:
            i = 10
            return i + 15
        'Register a callback for when a message is ready to recv.\n\n        There can be only one callback registered at a time, so each\n        call to `on_recv` replaces previously registered callbacks.\n\n        on_recv(None) disables recv event polling.\n\n        Use on_recv_stream(callback) instead, to register a callback that will receive\n        both this ZMQStream and the message, instead of just the message.\n\n        Parameters\n        ----------\n\n        callback : callable\n            callback must take exactly one argument, which will be a\n            list, as returned by socket.recv_multipart()\n            if callback is None, recv callbacks are disabled.\n        copy : bool\n            copy is passed directly to recv, so if copy is False,\n            callback will receive Message objects. If copy is True,\n            then callback will receive bytes/str objects.\n\n        Returns : None\n        '
        self._check_closed()
        assert callback is None or callable(callback)
        self._recv_callback = callback
        self._recv_copy = copy
        if callback is None:
            self._drop_io_state(zmq.POLLIN)
        else:
            self._add_io_state(zmq.POLLIN)

    @overload
    def on_recv_stream(self, callback: Callable[['ZMQStream', List[bytes]], Any]) -> None:
        if False:
            for i in range(10):
                print('nop')
        ...

    @overload
    def on_recv_stream(self, callback: Callable[['ZMQStream', List[bytes]], Any], copy: Literal[True]) -> None:
        if False:
            print('Hello World!')
        ...

    @overload
    def on_recv_stream(self, callback: Callable[['ZMQStream', List[zmq.Frame]], Any], copy: Literal[False]) -> None:
        if False:
            print('Hello World!')
        ...

    @overload
    def on_recv_stream(self, callback: Union[Callable[['ZMQStream', List[zmq.Frame]], Any], Callable[['ZMQStream', List[bytes]], Any]], copy: bool=...):
        if False:
            print('Hello World!')
        ...

    def on_recv_stream(self, callback: Union[Callable[['ZMQStream', List[zmq.Frame]], Any], Callable[['ZMQStream', List[bytes]], Any]], copy: bool=True):
        if False:
            while True:
                i = 10
        'Same as on_recv, but callback will get this stream as first argument\n\n        callback must take exactly two arguments, as it will be called as::\n\n            callback(stream, msg)\n\n        Useful when a single callback should be used with multiple streams.\n        '
        if callback is None:
            self.stop_on_recv()
        else:

            def stream_callback(msg):
                if False:
                    for i in range(10):
                        print('nop')
                return callback(self, msg)
            self.on_recv(stream_callback, copy=copy)

    def on_send(self, callback: Callable[[Sequence[Any], Optional[zmq.MessageTracker]], Any]):
        if False:
            i = 10
            return i + 15
        'Register a callback to be called on each send\n\n        There will be two arguments::\n\n            callback(msg, status)\n\n        * `msg` will be the list of sendable objects that was just sent\n        * `status` will be the return result of socket.send_multipart(msg) -\n          MessageTracker or None.\n\n        Non-copying sends return a MessageTracker object whose\n        `done` attribute will be True when the send is complete.\n        This allows users to track when an object is safe to write to\n        again.\n\n        The second argument will always be None if copy=True\n        on the send.\n\n        Use on_send_stream(callback) to register a callback that will be passed\n        this ZMQStream as the first argument, in addition to the other two.\n\n        on_send(None) disables recv event polling.\n\n        Parameters\n        ----------\n\n        callback : callable\n            callback must take exactly two arguments, which will be\n            the message being sent (always a list),\n            and the return result of socket.send_multipart(msg) -\n            MessageTracker or None.\n\n            if callback is None, send callbacks are disabled.\n        '
        self._check_closed()
        assert callback is None or callable(callback)
        self._send_callback = callback

    def on_send_stream(self, callback: Callable[['ZMQStream', Sequence[Any], Optional[zmq.MessageTracker]], Any]):
        if False:
            i = 10
            return i + 15
        'Same as on_send, but callback will get this stream as first argument\n\n        Callback will be passed three arguments::\n\n            callback(stream, msg, status)\n\n        Useful when a single callback should be used with multiple streams.\n        '
        if callback is None:
            self.stop_on_send()
        else:
            self.on_send(lambda msg, status: callback(self, msg, status))

    def send(self, msg, flags=0, copy=True, track=False, callback=None, **kwargs):
        if False:
            return 10
        'Send a message, optionally also register a new callback for sends.\n        See zmq.socket.send for details.\n        '
        return self.send_multipart([msg], flags=flags, copy=copy, track=track, callback=callback, **kwargs)

    def send_multipart(self, msg: Sequence[Any], flags: int=0, copy: bool=True, track: bool=False, callback: Optional[Callable]=None, **kwargs: Any) -> None:
        if False:
            i = 10
            return i + 15
        'Send a multipart message, optionally also register a new callback for sends.\n        See zmq.socket.send_multipart for details.\n        '
        kwargs.update(dict(flags=flags, copy=copy, track=track))
        self._send_queue.put((msg, kwargs))
        callback = callback or self._send_callback
        if callback is not None:
            self.on_send(callback)
        else:
            self.on_send(lambda *args: None)
        self._add_io_state(zmq.POLLOUT)

    def send_string(self, u: str, flags: int=0, encoding: str='utf-8', callback: Optional[Callable]=None, **kwargs: Any):
        if False:
            print('Hello World!')
        'Send a unicode message with an encoding.\n        See zmq.socket.send_unicode for details.\n        '
        if not isinstance(u, str):
            raise TypeError('unicode/str objects only')
        return self.send(u.encode(encoding), flags=flags, callback=callback, **kwargs)
    send_unicode = send_string

    def send_json(self, obj: Any, flags: int=0, callback: Optional[Callable]=None, **kwargs: Any):
        if False:
            while True:
                i = 10
        'Send json-serialized version of an object.\n        See zmq.socket.send_json for details.\n        '
        msg = jsonapi.dumps(obj)
        return self.send(msg, flags=flags, callback=callback, **kwargs)

    def send_pyobj(self, obj: Any, flags: int=0, protocol: int=-1, callback: Optional[Callable]=None, **kwargs: Any):
        if False:
            return 10
        'Send a Python object as a message using pickle to serialize.\n\n        See zmq.socket.send_json for details.\n        '
        msg = pickle.dumps(obj, protocol)
        return self.send(msg, flags, callback=callback, **kwargs)

    def _finish_flush(self):
        if False:
            for i in range(10):
                print('nop')
        'callback for unsetting _flushed flag.'
        self._flushed = False

    def flush(self, flag: int=zmq.POLLIN | zmq.POLLOUT, limit: Optional[int]=None):
        if False:
            while True:
                i = 10
        'Flush pending messages.\n\n        This method safely handles all pending incoming and/or outgoing messages,\n        bypassing the inner loop, passing them to the registered callbacks.\n\n        A limit can be specified, to prevent blocking under high load.\n\n        flush will return the first time ANY of these conditions are met:\n            * No more events matching the flag are pending.\n            * the total number of events handled reaches the limit.\n\n        Note that if ``flag|POLLIN != 0``, recv events will be flushed even if no callback\n        is registered, unlike normal IOLoop operation. This allows flush to be\n        used to remove *and ignore* incoming messages.\n\n        Parameters\n        ----------\n        flag : int, default=POLLIN|POLLOUT\n                0MQ poll flags.\n                If flag|POLLIN,  recv events will be flushed.\n                If flag|POLLOUT, send events will be flushed.\n                Both flags can be set at once, which is the default.\n        limit : None or int, optional\n                The maximum number of messages to send or receive.\n                Both send and recv count against this limit.\n\n        Returns\n        -------\n        int : count of events handled (both send and recv)\n        '
        self._check_closed()
        already_flushed = self._flushed
        self._flushed = False
        count = 0

        def update_flag():
            if False:
                print('Hello World!')
            "Update the poll flag, to prevent registering POLLOUT events\n            if we don't have pending sends."
            return flag & zmq.POLLIN | (self.sending() and flag & zmq.POLLOUT)
        flag = update_flag()
        if not flag:
            return 0
        self.poller.register(self.socket, flag)
        events = self.poller.poll(0)
        while events and (not limit or count < limit):
            (s, event) = events[0]
            if event & POLLIN:
                self._handle_recv()
                count += 1
                if self.socket is None:
                    break
            if event & POLLOUT and self.sending():
                self._handle_send()
                count += 1
                if self.socket is None:
                    break
            flag = update_flag()
            if flag:
                self.poller.register(self.socket, flag)
                events = self.poller.poll(0)
            else:
                events = []
        if count:
            self._flushed = True
            if not already_flushed:
                self.io_loop.add_callback(self._finish_flush)
        elif already_flushed:
            self._flushed = True
        self._rebuild_io_state()
        return count

    def set_close_callback(self, callback: Optional[Callable]):
        if False:
            print('Hello World!')
        'Call the given callback when the stream is closed.'
        self._close_callback = callback

    def close(self, linger: Optional[int]=None) -> None:
        if False:
            while True:
                i = 10
        'Close this stream.'
        if self.socket is not None:
            if self.socket.closed:
                warnings.warn('Unregistering FD %s after closing socket. This could result in unregistering handlers for the wrong socket. Please use stream.close() instead of closing the socket directly.' % self._fd, stacklevel=2)
                self.io_loop.remove_handler(self._fd)
            else:
                self.io_loop.remove_handler(self.socket)
                self.socket.close(linger)
            self.socket = None
            if self._close_callback:
                self._run_callback(self._close_callback)

    def receiving(self) -> bool:
        if False:
            print('Hello World!')
        'Returns True if we are currently receiving from the stream.'
        return self._recv_callback is not None

    def sending(self) -> bool:
        if False:
            while True:
                i = 10
        'Returns True if we are currently sending to the stream.'
        return not self._send_queue.empty()

    def closed(self) -> bool:
        if False:
            while True:
                i = 10
        if self.socket is None:
            return True
        if self.socket.closed:
            self.close()
            return True
        return False

    def _run_callback(self, callback, *args, **kwargs):
        if False:
            while True:
                i = 10
        'Wrap running callbacks in try/except to allow us to\n        close our socket.'
        try:
            f = callback(*args, **kwargs)
            if isinstance(f, Awaitable):
                f = asyncio.ensure_future(f)
            else:
                f = None
        except Exception:
            gen_log.error('Uncaught exception in ZMQStream callback', exc_info=True)
            raise
        if f is not None:

            def _log_error(f):
                if False:
                    while True:
                        i = 10
                try:
                    f.result()
                except Exception:
                    gen_log.error('Uncaught exception in ZMQStream callback', exc_info=True)
            f.add_done_callback(_log_error)

    def _handle_events(self, fd, events):
        if False:
            i = 10
            return i + 15
        'This method is the actual handler for IOLoop, that gets called whenever\n        an event on my socket is posted. It dispatches to _handle_recv, etc.'
        if not self.socket:
            gen_log.warning('Got events for closed stream %s', self)
            return
        try:
            zmq_events = self.socket.EVENTS
        except zmq.ContextTerminated:
            gen_log.warning('Got events for stream %s after terminating context', self)
            self.closed()
            return
        except zmq.ZMQError as e:
            if self.closed():
                gen_log.warning('Got events for stream %s attached to closed socket: %s', self, e)
            else:
                gen_log.error('Error getting events for %s: %s', self, e)
            return
        try:
            if zmq_events & zmq.POLLIN and self.receiving():
                self._handle_recv()
                if not self.socket:
                    return
            if zmq_events & zmq.POLLOUT and self.sending():
                self._handle_send()
                if not self.socket:
                    return
            self._rebuild_io_state()
        except Exception:
            gen_log.error('Uncaught exception in zmqstream callback', exc_info=True)
            raise

    def _handle_recv(self):
        if False:
            for i in range(10):
                print('nop')
        'Handle a recv event.'
        if self._flushed:
            return
        try:
            msg = self.socket.recv_multipart(zmq.NOBLOCK, copy=self._recv_copy)
        except zmq.ZMQError as e:
            if e.errno == zmq.EAGAIN:
                pass
            else:
                raise
        else:
            if self._recv_callback:
                callback = self._recv_callback
                self._run_callback(callback, msg)

    def _handle_send(self):
        if False:
            while True:
                i = 10
        'Handle a send event.'
        if self._flushed:
            return
        if not self.sending():
            gen_log.error("Shouldn't have handled a send event")
            return
        (msg, kwargs) = self._send_queue.get()
        try:
            status = self.socket.send_multipart(msg, **kwargs)
        except zmq.ZMQError as e:
            gen_log.error('SEND Error: %s', e)
            status = e
        if self._send_callback:
            callback = self._send_callback
            self._run_callback(callback, msg, status)

    def _check_closed(self):
        if False:
            while True:
                i = 10
        if not self.socket:
            raise OSError('Stream is closed')

    def _rebuild_io_state(self):
        if False:
            print('Hello World!')
        'rebuild io state based on self.sending() and receiving()'
        if self.socket is None:
            return
        state = 0
        if self.receiving():
            state |= zmq.POLLIN
        if self.sending():
            state |= zmq.POLLOUT
        self._state = state
        self._update_handler(state)

    def _add_io_state(self, state):
        if False:
            return 10
        'Add io_state to poller.'
        self._state = self._state | state
        self._update_handler(self._state)

    def _drop_io_state(self, state):
        if False:
            while True:
                i = 10
        'Stop poller from watching an io_state.'
        self._state = self._state & ~state
        self._update_handler(self._state)

    def _update_handler(self, state):
        if False:
            i = 10
            return i + 15
        'Update IOLoop handler with state.'
        if self.socket is None:
            return
        if state & self.socket.events:
            self.io_loop.add_callback(lambda : self._handle_events(self.socket, 0))

    def _init_io_state(self):
        if False:
            for i in range(10):
                print('nop')
        'initialize the ioloop event handler'
        self.io_loop.add_handler(self.socket, self._handle_events, self.io_loop.READ)