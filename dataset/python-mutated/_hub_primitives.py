"""
A collection of primitives used by the hub, and suitable for
compilation with Cython because of their frequency of use.


"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import traceback
from gevent.exceptions import InvalidSwitchError
from gevent.exceptions import ConcurrentObjectUseError
from gevent import _greenlet_primitives
from gevent import _waiter
from gevent._util import _NONE
from gevent._hub_local import get_hub_noargs as get_hub
from gevent.timeout import Timeout
locals()['getcurrent'] = __import__('greenlet').getcurrent
locals()['greenlet_init'] = lambda : None
locals()['Waiter'] = _waiter.Waiter
locals()['MultipleWaiter'] = _waiter.MultipleWaiter
locals()['SwitchOutGreenletWithLoop'] = _greenlet_primitives.SwitchOutGreenletWithLoop
__all__ = ['WaitOperationsGreenlet', 'iwait_on_objects', 'wait_on_objects', 'wait_read', 'wait_write', 'wait_readwrite']

class WaitOperationsGreenlet(SwitchOutGreenletWithLoop):

    def wait(self, watcher):
        if False:
            i = 10
            return i + 15
        '\n        Wait until the *watcher* (which must not be started) is ready.\n\n        The current greenlet will be unscheduled during this time.\n        '
        waiter = Waiter(self)
        watcher.start(waiter.switch, waiter)
        try:
            result = waiter.get()
            if result is not waiter:
                raise InvalidSwitchError('Invalid switch into %s: got %r (expected %r; waiting on %r with %r)' % (getcurrent(), result, waiter, self, watcher))
        finally:
            watcher.stop()

    def cancel_waits_close_and_then(self, watchers, exc_kind, then, *then_args):
        if False:
            for i in range(10):
                print('nop')
        deferred = []
        for watcher in watchers:
            if watcher is None:
                continue
            if watcher.callback is None:
                watcher.close()
            else:
                deferred.append(watcher)
        if deferred:
            self.loop.run_callback(self._cancel_waits_then, deferred, exc_kind, then, then_args)
        else:
            then(*then_args)

    def _cancel_waits_then(self, watchers, exc_kind, then, then_args):
        if False:
            print('Hello World!')
        for watcher in watchers:
            self._cancel_wait(watcher, exc_kind, True)
        then(*then_args)

    def cancel_wait(self, watcher, error, close_watcher=False):
        if False:
            return 10
        '\n        Cancel an in-progress call to :meth:`wait` by throwing the given *error*\n        in the waiting greenlet.\n\n        .. versionchanged:: 1.3a1\n           Added the *close_watcher* parameter. If true, the watcher\n           will be closed after the exception is thrown. The watcher should then\n           be discarded. Closing the watcher is important to release native resources.\n        .. versionchanged:: 1.3a2\n           Allow the *watcher* to be ``None``. No action is taken in that case.\n\n        '
        if watcher is None:
            return
        if watcher.callback is not None:
            self.loop.run_callback(self._cancel_wait, watcher, error, close_watcher)
            return
        if close_watcher:
            watcher.close()

    def _cancel_wait(self, watcher, error, close_watcher):
        if False:
            print('Hello World!')
        active = watcher.active
        cb = watcher.callback
        if close_watcher:
            watcher.close()
        if active:
            glet = getattr(cb, '__self__', None)
            if glet is not None:
                glet.throw(error)

class _WaitIterator(object):

    def __init__(self, objects, hub, timeout, count):
        if False:
            return 10
        self._hub = hub
        self._waiter = MultipleWaiter(hub)
        self._switch = self._waiter.switch
        self._timeout = timeout
        self._objects = objects
        self._timer = None
        self._begun = False
        self._count = len(objects) if count is None else min(count, len(objects))

    def _begin(self):
        if False:
            print('Hello World!')
        if self._begun:
            return
        self._begun = True
        for obj in self._objects:
            obj.rawlink(self._switch)
        if self._timeout is not None:
            self._timer = self._hub.loop.timer(self._timeout, priority=-1)
            self._timer.start(self._switch, self)

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

    def __next__(self):
        if False:
            i = 10
            return i + 15
        self._begin()
        if self._count == 0:
            self._cleanup()
            raise StopIteration()
        self._count -= 1
        try:
            item = self._waiter.get()
            self._waiter.clear()
            if item is self:
                self._cleanup()
                raise StopIteration()
            return item
        except:
            self._cleanup()
            raise
    next = __next__

    def _cleanup(self):
        if False:
            print('Hello World!')
        if self._timer is not None:
            self._timer.close()
            self._timer = None
        objs = self._objects
        self._objects = ()
        for aobj in objs:
            unlink = getattr(aobj, 'unlink', None)
            if unlink is not None:
                try:
                    unlink(self._switch)
                except:
                    traceback.print_exc()

    def __enter__(self):
        if False:
            print('Hello World!')
        return self

    def __exit__(self, typ, value, tb):
        if False:
            return 10
        self._cleanup()

def iwait_on_objects(objects, timeout=None, count=None):
    if False:
        return 10
    '\n    Iteratively yield *objects* as they are ready, until all (or *count*) are ready\n    or *timeout* expired.\n\n    If you will only be consuming a portion of the *objects*, you should\n    do so inside a ``with`` block on this object to avoid leaking resources::\n\n        with gevent.iwait((a, b, c)) as it:\n            for i in it:\n                if i is a:\n                    break\n\n    :param objects: A sequence (supporting :func:`len`) containing objects\n        implementing the wait protocol (rawlink() and unlink()).\n    :keyword int count: If not `None`, then a number specifying the maximum number\n        of objects to wait for. If ``None`` (the default), all objects\n        are waited for.\n    :keyword float timeout: If given, specifies a maximum number of seconds\n        to wait. If the timeout expires before the desired waited-for objects\n        are available, then this method returns immediately.\n\n    .. seealso:: :func:`wait`\n\n    .. versionchanged:: 1.1a1\n       Add the *count* parameter.\n    .. versionchanged:: 1.1a2\n       No longer raise :exc:`LoopExit` if our caller switches greenlets\n       in between items yielded by this function.\n    .. versionchanged:: 1.4\n       Add support to use the returned object as a context manager.\n    '
    hub = get_hub()
    if objects is None:
        return [hub.join(timeout=timeout)]
    return _WaitIterator(objects, hub, timeout, count)

def wait_on_objects(objects=None, timeout=None, count=None):
    if False:
        return 10
    '\n    Wait for *objects* to become ready or for event loop to finish.\n\n    If *objects* is provided, it must be a list containing objects\n    implementing the wait protocol (rawlink() and unlink() methods):\n\n    - :class:`gevent.Greenlet` instance\n    - :class:`gevent.event.Event` instance\n    - :class:`gevent.lock.Semaphore` instance\n    - :class:`gevent.subprocess.Popen` instance\n\n    If *objects* is ``None`` (the default), ``wait()`` blocks until\n    the current event loop has nothing to do (or until *timeout* passes):\n\n    - all greenlets have finished\n    - all servers were stopped\n    - all event loop watchers were stopped.\n\n    If *count* is ``None`` (the default), wait for all *objects*\n    to become ready.\n\n    If *count* is a number, wait for (up to) *count* objects to become\n    ready. (For example, if count is ``1`` then the function exits\n    when any object in the list is ready).\n\n    If *timeout* is provided, it specifies the maximum number of\n    seconds ``wait()`` will block.\n\n    Returns the list of ready objects, in the order in which they were\n    ready.\n\n    .. seealso:: :func:`iwait`\n    '
    if objects is None:
        hub = get_hub()
        return hub.join(timeout=timeout)
    return list(iwait_on_objects(objects, timeout, count))
_timeout_error = Exception

def set_default_timeout_error(e):
    if False:
        print('Hello World!')
    global _timeout_error
    _timeout_error = e

def _primitive_wait(watcher, timeout, timeout_exc, hub):
    if False:
        i = 10
        return i + 15
    if watcher.callback is not None:
        raise ConcurrentObjectUseError('This socket is already used by another greenlet: %r' % (watcher.callback,))
    if hub is None:
        hub = get_hub()
    if timeout is None:
        hub.wait(watcher)
        return
    timeout = Timeout._start_new_or_dummy(timeout, timeout_exc if timeout_exc is not _NONE or timeout is None else _timeout_error('timed out'))
    with timeout:
        hub.wait(watcher)

def wait_on_socket(socket, watcher, timeout_exc=None):
    if False:
        return 10
    if socket is None or watcher is None:
        raise ConcurrentObjectUseError('The socket has already been closed by another greenlet')
    _primitive_wait(watcher, socket.timeout, timeout_exc if timeout_exc is not None else _NONE, socket.hub)

def wait_on_watcher(watcher, timeout=None, timeout_exc=_NONE, hub=None):
    if False:
        print('Hello World!')
    "\n    wait(watcher, timeout=None, [timeout_exc=None]) -> None\n\n    Block the current greenlet until *watcher* is ready.\n\n    If *timeout* is non-negative, then *timeout_exc* is raised after\n    *timeout* second has passed.\n\n    If :func:`cancel_wait` is called on *watcher* by another greenlet,\n    raise an exception in this blocking greenlet\n    (``socket.error(EBADF, 'File descriptor was closed in another\n    greenlet')`` by default).\n\n    :param watcher: An event loop watcher, most commonly an IO watcher obtained from\n        :meth:`gevent.core.loop.io`\n    :keyword timeout_exc: The exception to raise if the timeout expires.\n        By default, a :class:`socket.timeout` exception is raised.\n        If you pass a value for this keyword, it is interpreted as for\n        :class:`gevent.timeout.Timeout`.\n\n    :raises ~gevent.hub.ConcurrentObjectUseError: If the *watcher* is\n        already started.\n    "
    _primitive_wait(watcher, timeout, timeout_exc, hub)

def wait_read(fileno, timeout=None, timeout_exc=_NONE):
    if False:
        return 10
    '\n    wait_read(fileno, timeout=None, [timeout_exc=None]) -> None\n\n    Block the current greenlet until *fileno* is ready to read.\n\n    For the meaning of the other parameters and possible exceptions,\n    see :func:`wait`.\n\n    .. seealso:: :func:`cancel_wait`\n    '
    hub = get_hub()
    io = hub.loop.io(fileno, 1)
    try:
        return wait_on_watcher(io, timeout, timeout_exc, hub)
    finally:
        io.close()

def wait_write(fileno, timeout=None, timeout_exc=_NONE, event=_NONE):
    if False:
        print('Hello World!')
    '\n    wait_write(fileno, timeout=None, [timeout_exc=None]) -> None\n\n    Block the current greenlet until *fileno* is ready to write.\n\n    For the meaning of the other parameters and possible exceptions,\n    see :func:`wait`.\n\n    .. deprecated:: 1.1\n       The keyword argument *event* is ignored. Applications should not pass this parameter.\n       In the future, doing so will become an error.\n\n    .. seealso:: :func:`cancel_wait`\n    '
    hub = get_hub()
    io = hub.loop.io(fileno, 2)
    try:
        return wait_on_watcher(io, timeout, timeout_exc, hub)
    finally:
        io.close()

def wait_readwrite(fileno, timeout=None, timeout_exc=_NONE, event=_NONE):
    if False:
        i = 10
        return i + 15
    '\n    wait_readwrite(fileno, timeout=None, [timeout_exc=None]) -> None\n\n    Block the current greenlet until *fileno* is ready to read or\n    write.\n\n    For the meaning of the other parameters and possible exceptions,\n    see :func:`wait`.\n\n    .. deprecated:: 1.1\n       The keyword argument *event* is ignored. Applications should not pass this parameter.\n       In the future, doing so will become an error.\n\n    .. seealso:: :func:`cancel_wait`\n    '
    hub = get_hub()
    io = hub.loop.io(fileno, 3)
    try:
        return wait_on_watcher(io, timeout, timeout_exc, hub)
    finally:
        io.close()

def _init():
    if False:
        print('Hello World!')
    greenlet_init()
_init()
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__hub_primitives')