"""Base class for implementing servers"""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
import sys
import _socket
import errno
from gevent.greenlet import Greenlet
from gevent.event import Event
from gevent.hub import get_hub
from gevent._compat import string_types
from gevent._compat import integer_types
from gevent._compat import xrange
__all__ = ['BaseServer']

def _handle_and_close_when_done(handle, close, args_tuple):
    if False:
        while True:
            i = 10
    try:
        return handle(*args_tuple)
    finally:
        close(*args_tuple)

class BaseServer(object):
    """
    An abstract base class that implements some common functionality for the servers in gevent.

    :param listener: Either be an address that the server should bind
        on or a :class:`gevent.socket.socket` instance that is already
        bound (and put into listening mode in case of TCP socket).

    :keyword handle: If given, the request handler. The request
        handler can be defined in a few ways. Most commonly,
        subclasses will implement a ``handle`` method as an
        instance method. Alternatively, a function can be passed
        as the ``handle`` argument to the constructor. In either
        case, the handler can later be changed by calling
        :meth:`set_handle`.

        When the request handler returns, the socket used for the
        request will be closed. Therefore, the handler must not return if
        the socket is still in use (for example, by manually spawned greenlets).

    :keyword spawn: If provided, is called to create a new
        greenlet to run the handler. By default,
        :func:`gevent.spawn` is used (meaning there is no
        artificial limit on the number of concurrent requests). Possible values for *spawn*:

        - a :class:`gevent.pool.Pool` instance -- ``handle`` will be executed
          using :meth:`gevent.pool.Pool.spawn` only if the pool is not full.
          While it is full, no new connections are accepted;
        - :func:`gevent.spawn_raw` -- ``handle`` will be executed in a raw
          greenlet which has a little less overhead then :class:`gevent.Greenlet` instances spawned by default;
        - ``None`` -- ``handle`` will be executed right away, in the :class:`Hub` greenlet.
          ``handle`` cannot use any blocking functions as it would mean switching to the :class:`Hub`.
        - an integer -- a shortcut for ``gevent.pool.Pool(integer)``

    .. versionchanged:: 1.1a1
       When the *handle* function returns from processing a connection,
       the client socket will be closed. This resolves the non-deterministic
       closing of the socket, fixing ResourceWarnings under Python 3 and PyPy.
    .. versionchanged:: 1.5
       Now a context manager that returns itself and calls :meth:`stop` on exit.

    """
    min_delay = 0.01
    max_delay = 1
    max_accept = 100
    _spawn = Greenlet.spawn
    stop_timeout = 1
    fatal_errors = (errno.EBADF, errno.EINVAL, errno.ENOTSOCK)

    def __init__(self, listener, handle=None, spawn='default'):
        if False:
            for i in range(10):
                print('nop')
        self._stop_event = Event()
        self._stop_event.set()
        self._watcher = None
        self._timer = None
        self._handle = None
        self.pool = None
        try:
            self.set_listener(listener)
            self.set_spawn(spawn)
            self.set_handle(handle)
            self.delay = self.min_delay
            self.loop = get_hub().loop
            if self.max_accept < 1:
                raise ValueError('max_accept must be positive int: %r' % (self.max_accept,))
        except:
            self.close()
            raise

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, *args):
        if False:
            return 10
        self.stop()

    def set_listener(self, listener):
        if False:
            i = 10
            return i + 15
        if hasattr(listener, 'accept'):
            if hasattr(listener, 'do_handshake'):
                raise TypeError('Expected a regular socket, not SSLSocket: %r' % (listener,))
            self.family = listener.family
            self.address = listener.getsockname()
            self.socket = listener
        else:
            (self.family, self.address) = parse_address(listener)

    def set_spawn(self, spawn):
        if False:
            for i in range(10):
                print('nop')
        if spawn == 'default':
            self.pool = None
            self._spawn = self._spawn
        elif hasattr(spawn, 'spawn'):
            self.pool = spawn
            self._spawn = spawn.spawn
        elif isinstance(spawn, integer_types):
            from gevent.pool import Pool
            self.pool = Pool(spawn)
            self._spawn = self.pool.spawn
        else:
            self.pool = None
            self._spawn = spawn
        if hasattr(self.pool, 'full'):
            self.full = self.pool.full
        if self.pool is not None:
            self.pool._semaphore.rawlink(self._start_accepting_if_started)

    def set_handle(self, handle):
        if False:
            return 10
        if handle is not None:
            self.handle = handle
        if hasattr(self, 'handle'):
            self._handle = self.handle
        else:
            raise TypeError("'handle' must be provided")

    def _start_accepting_if_started(self, _event=None):
        if False:
            return 10
        if self.started:
            self.start_accepting()

    def start_accepting(self):
        if False:
            i = 10
            return i + 15
        if self._watcher is None:
            self._watcher = self.loop.io(self.socket.fileno(), 1)
            self._watcher.start(self._do_read)

    def stop_accepting(self):
        if False:
            i = 10
            return i + 15
        if self._watcher is not None:
            self._watcher.stop()
            self._watcher.close()
            self._watcher = None
        if self._timer is not None:
            self._timer.stop()
            self._timer.close()
            self._timer = None

    def do_handle(self, *args):
        if False:
            return 10
        spawn = self._spawn
        handle = self._handle
        close = self.do_close
        try:
            if spawn is None:
                _handle_and_close_when_done(handle, close, args)
            else:
                spawn(_handle_and_close_when_done, handle, close, args)
        except:
            close(*args)
            raise

    def do_close(self, *args):
        if False:
            return 10
        pass

    def do_read(self):
        if False:
            return 10
        raise NotImplementedError()

    def _do_read(self):
        if False:
            return 10
        for _ in xrange(self.max_accept):
            if self.full():
                self.stop_accepting()
                if self.pool is not None:
                    self.pool._semaphore.rawlink(self._start_accepting_if_started)
                return
            try:
                args = self.do_read()
                self.delay = self.min_delay
                if not args:
                    return
            except:
                self.loop.handle_error(self, *sys.exc_info())
                ex = sys.exc_info()[1]
                if self.is_fatal_error(ex):
                    self.close()
                    sys.stderr.write('ERROR: %s failed with %s\n' % (self, str(ex) or repr(ex)))
                    return
                if self.delay >= 0:
                    self.stop_accepting()
                    self._timer = self.loop.timer(self.delay)
                    self._timer.start(self._start_accepting_if_started)
                    self.delay = min(self.max_delay, self.delay * 2)
                break
            else:
                try:
                    self.do_handle(*args)
                except:
                    self.loop.handle_error((args[1:], self), *sys.exc_info())
                    if self.delay >= 0:
                        self.stop_accepting()
                        self._timer = self.loop.timer(self.delay)
                        self._timer.start(self._start_accepting_if_started)
                        self.delay = min(self.max_delay, self.delay * 2)
                    break

    def full(self):
        if False:
            print('Hello World!')
        return False

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '<%s at %s %s>' % (type(self).__name__, hex(id(self)), self._formatinfo())

    def __str__(self):
        if False:
            return 10
        return '<%s %s>' % (type(self).__name__, self._formatinfo())

    def _formatinfo(self):
        if False:
            print('Hello World!')
        if hasattr(self, 'socket'):
            try:
                fileno = self.socket.fileno()
            except Exception as ex:
                fileno = str(ex)
            result = 'fileno=%s ' % fileno
        else:
            result = ''
        try:
            if isinstance(self.address, tuple) and len(self.address) == 2:
                result += 'address=%s:%s' % self.address
            else:
                result += 'address=%s' % (self.address,)
        except Exception as ex:
            result += str(ex) or '<error>'
        handle = self.__dict__.get('handle')
        if handle is not None:
            fself = getattr(handle, '__self__', None)
            try:
                if fself is self:
                    handle_repr = '<bound method %s.%s of self>' % (self.__class__.__name__, handle.__name__)
                else:
                    handle_repr = repr(handle)
                result += ' handle=' + handle_repr
            except Exception as ex:
                result += str(ex) or '<error>'
        return result

    @property
    def server_host(self):
        if False:
            print('Hello World!')
        'IP address that the server is bound to (string).'
        if isinstance(self.address, tuple):
            return self.address[0]

    @property
    def server_port(self):
        if False:
            while True:
                i = 10
        'Port that the server is bound to (an integer).'
        if isinstance(self.address, tuple):
            return self.address[1]

    def init_socket(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        If the user initialized the server with an address rather than\n        socket, then this function must create a socket, bind it, and\n        put it into listening mode.\n\n        It is not supposed to be called by the user, it is called by :meth:`start` before starting\n        the accept loop.\n        '

    @property
    def started(self):
        if False:
            return 10
        return not self._stop_event.is_set()

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        'Start accepting the connections.\n\n        If an address was provided in the constructor, then also create a socket,\n        bind it and put it into the listening mode.\n        '
        self.init_socket()
        self._stop_event.clear()
        try:
            self.start_accepting()
        except:
            self.close()
            raise

    def close(self):
        if False:
            print('Hello World!')
        'Close the listener socket and stop accepting.'
        self._stop_event.set()
        try:
            self.stop_accepting()
        finally:
            try:
                self.socket.close()
            except Exception:
                pass
            finally:
                self.__dict__.pop('socket', None)
                self.__dict__.pop('handle', None)
                self.__dict__.pop('_handle', None)
                self.__dict__.pop('_spawn', None)
                self.__dict__.pop('full', None)
                if self.pool is not None:
                    self.pool._semaphore.unlink(self._start_accepting_if_started)

    @property
    def closed(self):
        if False:
            return 10
        return not hasattr(self, 'socket')

    def stop(self, timeout=None):
        if False:
            i = 10
            return i + 15
        '\n        Stop accepting the connections and close the listening socket.\n\n        If the server uses a pool to spawn the requests, then\n        :meth:`stop` also waits for all the handlers to exit. If there\n        are still handlers executing after *timeout* has expired\n        (default 1 second, :attr:`stop_timeout`), then the currently\n        running handlers in the pool are killed.\n\n        If the server does not use a pool, then this merely stops accepting connections;\n        any spawned greenlets that are handling requests continue running until\n        they naturally complete.\n        '
        self.close()
        if timeout is None:
            timeout = self.stop_timeout
        if self.pool:
            self.pool.join(timeout=timeout)
            self.pool.kill(block=True, timeout=1)

    def serve_forever(self, stop_timeout=None):
        if False:
            return 10
        "Start the server if it hasn't been already started and wait until it's stopped."
        if not self.started:
            self.start()
        try:
            self._stop_event.wait()
        finally:
            Greenlet.spawn(self.stop, timeout=stop_timeout).join()

    def is_fatal_error(self, ex):
        if False:
            return 10
        return isinstance(ex, _socket.error) and ex.args[0] in self.fatal_errors

def _extract_family(host):
    if False:
        return 10
    if host.startswith('[') and host.endswith(']'):
        host = host[1:-1]
        return (_socket.AF_INET6, host)
    return (_socket.AF_INET, host)

def _parse_address(address):
    if False:
        i = 10
        return i + 15
    if isinstance(address, tuple):
        if not address[0] or ':' in address[0]:
            return (_socket.AF_INET6, address)
        return (_socket.AF_INET, address)
    if isinstance(address, string_types) and ':' not in address or isinstance(address, integer_types):
        return (_socket.AF_INET6, ('', int(address)))
    if not isinstance(address, string_types):
        raise TypeError('Expected tuple or string, got %s' % type(address))
    (host, port) = address.rsplit(':', 1)
    (family, host) = _extract_family(host)
    if host == '*':
        host = ''
    return (family, (host, int(port)))

def parse_address(address):
    if False:
        for i in range(10):
            print('nop')
    try:
        return _parse_address(address)
    except ValueError as ex:
        raise ValueError('Failed to parse address %r: %s' % (address, ex))