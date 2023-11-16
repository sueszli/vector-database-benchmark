"""Extremely simple pure-Python implementation of coroutine-style
asynchronous socket I/O. Inspired by, but inferior to, Eventlet.
Bluelet can also be thought of as a less-terrible replacement for
asyncore.

Bluelet: easy concurrency without all the messy parallelism.
"""
import collections
import errno
import select
import socket
import sys
import time
import traceback
import types

class Event:
    """Just a base class identifying Bluelet events. An event is an
    object yielded from a Bluelet thread coroutine to suspend operation
    and communicate with the scheduler.
    """
    pass

class WaitableEvent(Event):
    """A waitable event is one encapsulating an action that can be
    waited for using a select() call. That is, it's an event with an
    associated file descriptor.
    """

    def waitables(self):
        if False:
            return 10
        'Return "waitable" objects to pass to select(). Should return\n        three iterables for input readiness, output readiness, and\n        exceptional conditions (i.e., the three lists passed to\n        select()).\n        '
        return ((), (), ())

    def fire(self):
        if False:
            print('Hello World!')
        'Called when an associated file descriptor becomes ready\n        (i.e., is returned from a select() call).\n        '
        pass

class ValueEvent(Event):
    """An event that does nothing but return a fixed value."""

    def __init__(self, value):
        if False:
            return 10
        self.value = value

class ExceptionEvent(Event):
    """Raise an exception at the yield point. Used internally."""

    def __init__(self, exc_info):
        if False:
            print('Hello World!')
        self.exc_info = exc_info

class SpawnEvent(Event):
    """Add a new coroutine thread to the scheduler."""

    def __init__(self, coro):
        if False:
            for i in range(10):
                print('nop')
        self.spawned = coro

class JoinEvent(Event):
    """Suspend the thread until the specified child thread has
    completed.
    """

    def __init__(self, child):
        if False:
            while True:
                i = 10
        self.child = child

class KillEvent(Event):
    """Unschedule a child thread."""

    def __init__(self, child):
        if False:
            i = 10
            return i + 15
        self.child = child

class DelegationEvent(Event):
    """Suspend execution of the current thread, start a new thread and,
    once the child thread finished, return control to the parent
    thread.
    """

    def __init__(self, coro):
        if False:
            return 10
        self.spawned = coro

class ReturnEvent(Event):
    """Return a value the current thread's delegator at the point of
    delegation. Ends the current (delegate) thread.
    """

    def __init__(self, value):
        if False:
            print('Hello World!')
        self.value = value

class SleepEvent(WaitableEvent):
    """Suspend the thread for a given duration."""

    def __init__(self, duration):
        if False:
            print('Hello World!')
        self.wakeup_time = time.time() + duration

    def time_left(self):
        if False:
            return 10
        return max(self.wakeup_time - time.time(), 0.0)

class ReadEvent(WaitableEvent):
    """Reads from a file-like object."""

    def __init__(self, fd, bufsize):
        if False:
            i = 10
            return i + 15
        self.fd = fd
        self.bufsize = bufsize

    def waitables(self):
        if False:
            print('Hello World!')
        return ((self.fd,), (), ())

    def fire(self):
        if False:
            while True:
                i = 10
        return self.fd.read(self.bufsize)

class WriteEvent(WaitableEvent):
    """Writes to a file-like object."""

    def __init__(self, fd, data):
        if False:
            while True:
                i = 10
        self.fd = fd
        self.data = data

    def waitable(self):
        if False:
            i = 10
            return i + 15
        return ((), (self.fd,), ())

    def fire(self):
        if False:
            while True:
                i = 10
        self.fd.write(self.data)

def _event_select(events):
    if False:
        while True:
            i = 10
    'Perform a select() over all the Events provided, returning the\n    ones ready to be fired. Only WaitableEvents (including SleepEvents)\n    matter here; all other events are ignored (and thus postponed).\n    '
    waitable_to_event = {}
    (rlist, wlist, xlist) = ([], [], [])
    earliest_wakeup = None
    for event in events:
        if isinstance(event, SleepEvent):
            if not earliest_wakeup:
                earliest_wakeup = event.wakeup_time
            else:
                earliest_wakeup = min(earliest_wakeup, event.wakeup_time)
        elif isinstance(event, WaitableEvent):
            (r, w, x) = event.waitables()
            rlist += r
            wlist += w
            xlist += x
            for waitable in r:
                waitable_to_event['r', waitable] = event
            for waitable in w:
                waitable_to_event['w', waitable] = event
            for waitable in x:
                waitable_to_event['x', waitable] = event
    if earliest_wakeup:
        timeout = max(earliest_wakeup - time.time(), 0.0)
    else:
        timeout = None
    if rlist or wlist or xlist:
        (rready, wready, xready) = select.select(rlist, wlist, xlist, timeout)
    else:
        (rready, wready, xready) = ((), (), ())
        if timeout:
            time.sleep(timeout)
    ready_events = set()
    for ready in rready:
        ready_events.add(waitable_to_event['r', ready])
    for ready in wready:
        ready_events.add(waitable_to_event['w', ready])
    for ready in xready:
        ready_events.add(waitable_to_event['x', ready])
    for event in events:
        if isinstance(event, SleepEvent) and event.time_left() == 0.0:
            ready_events.add(event)
    return ready_events

class ThreadException(Exception):

    def __init__(self, coro, exc_info):
        if False:
            i = 10
            return i + 15
        self.coro = coro
        self.exc_info = exc_info

    def reraise(self):
        if False:
            i = 10
            return i + 15
        raise self.exc_info[1].with_traceback(self.exc_info[2])
SUSPENDED = Event()

class Delegated(Event):
    """Placeholder indicating that a thread has delegated execution to a
    different thread.
    """

    def __init__(self, child):
        if False:
            while True:
                i = 10
        self.child = child

def run(root_coro):
    if False:
        print('Hello World!')
    'Schedules a coroutine, running it to completion. This\n    encapsulates the Bluelet scheduler, which the root coroutine can\n    add to by spawning new coroutines.\n    '
    threads = {root_coro: ValueEvent(None)}
    delegators = {}
    joiners = collections.defaultdict(list)

    def complete_thread(coro, return_value):
        if False:
            i = 10
            return i + 15
        'Remove a coroutine from the scheduling pool, awaking\n        delegators and joiners as necessary and returning the specified\n        value to any delegating parent.\n        '
        del threads[coro]
        if coro in delegators:
            threads[delegators[coro]] = ValueEvent(return_value)
            del delegators[coro]
        if coro in joiners:
            for parent in joiners[coro]:
                threads[parent] = ValueEvent(None)
            del joiners[coro]

    def advance_thread(coro, value, is_exc=False):
        if False:
            for i in range(10):
                print('nop')
        'After an event is fired, run a given coroutine associated with\n        it in the threads dict until it yields again. If the coroutine\n        exits, then the thread is removed from the pool. If the coroutine\n        raises an exception, it is reraised in a ThreadException. If\n        is_exc is True, then the value must be an exc_info tuple and the\n        exception is thrown into the coroutine.\n        '
        try:
            if is_exc:
                next_event = coro.throw(*value)
            else:
                next_event = coro.send(value)
        except StopIteration:
            complete_thread(coro, None)
        except BaseException:
            del threads[coro]
            raise ThreadException(coro, sys.exc_info())
        else:
            if isinstance(next_event, types.GeneratorType):
                next_event = DelegationEvent(next_event)
            threads[coro] = next_event

    def kill_thread(coro):
        if False:
            for i in range(10):
                print('nop')
        'Unschedule this thread and its (recursive) delegates.'
        coros = [coro]
        while isinstance(threads[coro], Delegated):
            coro = threads[coro].child
            coros.append(coro)
        for coro in reversed(coros):
            complete_thread(coro, None)
    exit_te = None
    while threads:
        try:
            while True:
                have_ready = False
                for (coro, event) in list(threads.items()):
                    if isinstance(event, SpawnEvent):
                        threads[event.spawned] = ValueEvent(None)
                        advance_thread(coro, None)
                        have_ready = True
                    elif isinstance(event, ValueEvent):
                        advance_thread(coro, event.value)
                        have_ready = True
                    elif isinstance(event, ExceptionEvent):
                        advance_thread(coro, event.exc_info, True)
                        have_ready = True
                    elif isinstance(event, DelegationEvent):
                        threads[coro] = Delegated(event.spawned)
                        threads[event.spawned] = ValueEvent(None)
                        delegators[event.spawned] = coro
                        have_ready = True
                    elif isinstance(event, ReturnEvent):
                        complete_thread(coro, event.value)
                        have_ready = True
                    elif isinstance(event, JoinEvent):
                        threads[coro] = SUSPENDED
                        joiners[event.child].append(coro)
                        have_ready = True
                    elif isinstance(event, KillEvent):
                        threads[coro] = ValueEvent(None)
                        kill_thread(event.child)
                        have_ready = True
                if not have_ready:
                    break
            event2coro = {v: k for (k, v) in threads.items()}
            for event in _event_select(threads.values()):
                try:
                    value = event.fire()
                except OSError as exc:
                    if isinstance(exc.args, tuple) and exc.args[0] == errno.EPIPE:
                        pass
                    elif isinstance(exc.args, tuple) and exc.args[0] == errno.ECONNRESET:
                        pass
                    else:
                        traceback.print_exc()
                    threads[event2coro[event]] = ReturnEvent(None)
                else:
                    advance_thread(event2coro[event], value)
        except ThreadException as te:
            event = ExceptionEvent(te.exc_info)
            if te.coro in delegators:
                threads[delegators[te.coro]] = event
                del delegators[te.coro]
            else:
                exit_te = te
                break
        except BaseException:
            threads = {root_coro: ExceptionEvent(sys.exc_info())}
    for coro in threads:
        coro.close()
    if exit_te:
        exit_te.reraise()

class SocketClosedError(Exception):
    pass

class Listener:
    """A socket wrapper object for listening sockets."""

    def __init__(self, host, port):
        if False:
            print('Hello World!')
        'Create a listening socket on the given hostname and port.'
        self._closed = False
        self.host = host
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind((host, port))
        self.sock.listen(5)

    def accept(self):
        if False:
            print('Hello World!')
        'An event that waits for a connection on the listening socket.\n        When a connection is made, the event returns a Connection\n        object.\n        '
        if self._closed:
            raise SocketClosedError()
        return AcceptEvent(self)

    def close(self):
        if False:
            print('Hello World!')
        'Immediately close the listening socket. (Not an event.)'
        self._closed = True
        self.sock.close()

class Connection:
    """A socket wrapper object for connected sockets."""

    def __init__(self, sock, addr):
        if False:
            while True:
                i = 10
        self.sock = sock
        self.addr = addr
        self._buf = b''
        self._closed = False

    def close(self):
        if False:
            i = 10
            return i + 15
        'Close the connection.'
        self._closed = True
        self.sock.close()

    def recv(self, size):
        if False:
            print('Hello World!')
        'Read at most size bytes of data from the socket.'
        if self._closed:
            raise SocketClosedError()
        if self._buf:
            out = self._buf[:size]
            self._buf = self._buf[size:]
            return ValueEvent(out)
        else:
            return ReceiveEvent(self, size)

    def send(self, data):
        if False:
            return 10
        'Sends data on the socket, returning the number of bytes\n        successfully sent.\n        '
        if self._closed:
            raise SocketClosedError()
        return SendEvent(self, data)

    def sendall(self, data):
        if False:
            return 10
        'Send all of data on the socket.'
        if self._closed:
            raise SocketClosedError()
        return SendEvent(self, data, True)

    def readline(self, terminator=b'\n', bufsize=1024):
        if False:
            while True:
                i = 10
        'Reads a line (delimited by terminator) from the socket.'
        if self._closed:
            raise SocketClosedError()
        while True:
            if terminator in self._buf:
                (line, self._buf) = self._buf.split(terminator, 1)
                line += terminator
                yield ReturnEvent(line)
                break
            data = (yield ReceiveEvent(self, bufsize))
            if data:
                self._buf += data
            else:
                line = self._buf
                self._buf = b''
                yield ReturnEvent(line)
                break

class AcceptEvent(WaitableEvent):
    """An event for Listener objects (listening sockets) that suspends
    execution until the socket gets a connection.
    """

    def __init__(self, listener):
        if False:
            i = 10
            return i + 15
        self.listener = listener

    def waitables(self):
        if False:
            while True:
                i = 10
        return ((self.listener.sock,), (), ())

    def fire(self):
        if False:
            print('Hello World!')
        (sock, addr) = self.listener.sock.accept()
        return Connection(sock, addr)

class ReceiveEvent(WaitableEvent):
    """An event for Connection objects (connected sockets) for
    asynchronously reading data.
    """

    def __init__(self, conn, bufsize):
        if False:
            for i in range(10):
                print('nop')
        self.conn = conn
        self.bufsize = bufsize

    def waitables(self):
        if False:
            print('Hello World!')
        return ((self.conn.sock,), (), ())

    def fire(self):
        if False:
            print('Hello World!')
        return self.conn.sock.recv(self.bufsize)

class SendEvent(WaitableEvent):
    """An event for Connection objects (connected sockets) for
    asynchronously writing data.
    """

    def __init__(self, conn, data, sendall=False):
        if False:
            i = 10
            return i + 15
        self.conn = conn
        self.data = data
        self.sendall = sendall

    def waitables(self):
        if False:
            while True:
                i = 10
        return ((), (self.conn.sock,), ())

    def fire(self):
        if False:
            print('Hello World!')
        if self.sendall:
            return self.conn.sock.sendall(self.data)
        else:
            return self.conn.sock.send(self.data)

def null():
    if False:
        for i in range(10):
            print('nop')
    'Event: yield to the scheduler without doing anything special.'
    return ValueEvent(None)

def spawn(coro):
    if False:
        i = 10
        return i + 15
    'Event: add another coroutine to the scheduler. Both the parent\n    and child coroutines run concurrently.\n    '
    if not isinstance(coro, types.GeneratorType):
        raise ValueError('%s is not a coroutine' % coro)
    return SpawnEvent(coro)

def call(coro):
    if False:
        while True:
            i = 10
    'Event: delegate to another coroutine. The current coroutine\n    is resumed once the sub-coroutine finishes. If the sub-coroutine\n    returns a value using end(), then this event returns that value.\n    '
    if not isinstance(coro, types.GeneratorType):
        raise ValueError('%s is not a coroutine' % coro)
    return DelegationEvent(coro)

def end(value=None):
    if False:
        print('Hello World!')
    'Event: ends the coroutine and returns a value to its\n    delegator.\n    '
    return ReturnEvent(value)

def read(fd, bufsize=None):
    if False:
        for i in range(10):
            print('nop')
    'Event: read from a file descriptor asynchronously.'
    if bufsize is None:

        def reader():
            if False:
                i = 10
                return i + 15
            buf = []
            while True:
                data = (yield read(fd, 1024))
                if not data:
                    break
                buf.append(data)
            yield ReturnEvent(''.join(buf))
        return DelegationEvent(reader())
    else:
        return ReadEvent(fd, bufsize)

def write(fd, data):
    if False:
        return 10
    'Event: write to a file descriptor asynchronously.'
    return WriteEvent(fd, data)

def connect(host, port):
    if False:
        while True:
            i = 10
    'Event: connect to a network address and return a Connection\n    object for communicating on the socket.\n    '
    addr = (host, port)
    sock = socket.create_connection(addr)
    return ValueEvent(Connection(sock, addr))

def sleep(duration):
    if False:
        for i in range(10):
            print('nop')
    'Event: suspend the thread for ``duration`` seconds.'
    return SleepEvent(duration)

def join(coro):
    if False:
        for i in range(10):
            print('nop')
    'Suspend the thread until another, previously `spawn`ed thread\n    completes.\n    '
    return JoinEvent(coro)

def kill(coro):
    if False:
        return 10
    'Halt the execution of a different `spawn`ed thread.'
    return KillEvent(coro)

def server(host, port, func):
    if False:
        print('Hello World!')
    'A coroutine that runs a network server. Host and port specify the\n    listening address. func should be a coroutine that takes a single\n    parameter, a Connection object. The coroutine is invoked for every\n    incoming connection on the listening socket.\n    '

    def handler(conn):
        if False:
            i = 10
            return i + 15
        try:
            yield func(conn)
        finally:
            conn.close()
    listener = Listener(host, port)
    try:
        while True:
            conn = (yield listener.accept())
            yield spawn(handler(conn))
    except KeyboardInterrupt:
        pass
    finally:
        listener.close()