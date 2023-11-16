from __future__ import print_function, absolute_import, division
__all__ = ['Semaphore', 'BoundedSemaphore']
from time import sleep as _native_sleep
from gevent._compat import monotonic
from gevent.exceptions import InvalidThreadUseError
from gevent.exceptions import LoopExit
from gevent.timeout import Timeout

def _get_linkable():
    if False:
        while True:
            i = 10
    x = __import__('gevent._abstract_linkable')
    return x._abstract_linkable.AbstractLinkable
locals()['AbstractLinkable'] = _get_linkable()
del _get_linkable
from gevent._hub_local import get_hub_if_exists
from gevent._hub_local import get_hub
from gevent.hub import spawn_raw

class _LockReleaseLink(object):
    __slots__ = ('lock',)

    def __init__(self, lock):
        if False:
            print('Hello World!')
        self.lock = lock

    def __call__(self, _):
        if False:
            for i in range(10):
                print('nop')
        self.lock.release()
_UNSET = object()
_MULTI = object()

class Semaphore(AbstractLinkable):
    """
    Semaphore(value=1) -> Semaphore

    .. seealso:: :class:`BoundedSemaphore` for a safer version that prevents
       some classes of bugs. If unsure, most users should opt for `BoundedSemaphore`.

    A semaphore manages a counter representing the number of `release`
    calls minus the number of `acquire` calls, plus an initial value.
    The `acquire` method blocks if necessary until it can return
    without making the counter negative. A semaphore does not track ownership
    by greenlets; any greenlet can call `release`, whether or not it has previously
    called `acquire`.

    If not given, ``value`` defaults to 1.

    The semaphore is a context manager and can be used in ``with`` statements.

    This Semaphore's ``__exit__`` method does not call the trace function
    on CPython, but does under PyPy.

    .. versionchanged:: 1.4.0
        Document that the order in which waiters are awakened is not specified. It was not
        specified previously, but due to CPython implementation quirks usually went in FIFO order.
    .. versionchanged:: 1.5a3
       Waiting greenlets are now awakened in the order in which they waited.
    .. versionchanged:: 1.5a3
       The low-level ``rawlink`` method (most users won't use this) now automatically
       unlinks waiters before calling them.
    .. versionchanged:: 20.12.0
       Improved support for multi-threaded usage. When multi-threaded usage is detected,
       instances will no longer create the thread's hub if it's not present.
    """
    __slots__ = ('counter', '_multithreaded')

    def __init__(self, value=1, hub=None):
        if False:
            for i in range(10):
                print('nop')
        self.counter = value
        if self.counter < 0:
            raise ValueError('semaphore initial value must be >= 0')
        super(Semaphore, self).__init__(hub)
        self._notify_all = False
        self._multithreaded = _UNSET

    def __str__(self):
        if False:
            return 10
        return '<%s at 0x%x counter=%s _links[%s]>' % (self.__class__.__name__, id(self), self.counter, self.linkcount())

    def locked(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a boolean indicating whether the semaphore can be\n        acquired (`False` if the semaphore *can* be acquired). Most\n        useful with binary semaphores (those with an initial value of 1).\n\n        :rtype: bool\n        '
        return self.counter <= 0

    def release(self):
        if False:
            i = 10
            return i + 15
        '\n        Release the semaphore, notifying any waiters if needed. There\n        is no return value.\n\n        .. note::\n\n            This can be used to over-release the semaphore.\n            (Release more times than it has been acquired or was initially\n            created with.)\n\n            This is usually a sign of a bug, but under some circumstances it can be\n            used deliberately, for example, to model the arrival of additional\n            resources.\n\n        :rtype: None\n        '
        self.counter += 1
        self._check_and_notify()
        return self.counter

    def ready(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a boolean indicating whether the semaphore can be\n        acquired (`True` if the semaphore can be acquired).\n\n        :rtype: bool\n        '
        return self.counter > 0

    def _start_notify(self):
        if False:
            while True:
                i = 10
        self._check_and_notify()

    def _wait_return_value(self, waited, wait_success):
        if False:
            for i in range(10):
                print('nop')
        if waited:
            return wait_success
        return True

    def wait(self, timeout=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Wait until it is possible to acquire this semaphore, or until the optional\n        *timeout* elapses.\n\n        .. note:: If this semaphore was initialized with a *value* of 0,\n           this method will block forever if no timeout is given.\n\n        :keyword float timeout: If given, specifies the maximum amount of seconds\n           this method will block.\n        :return: A number indicating how many times the semaphore can be acquired\n            before blocking. *This could be 0,* if other waiters acquired\n            the semaphore.\n        :rtype: int\n        '
        if self.counter > 0:
            return self.counter
        self._wait(timeout)
        return self.counter

    def acquire(self, blocking=True, timeout=None):
        if False:
            print('Hello World!')
        '\n        acquire(blocking=True, timeout=None) -> bool\n\n        Acquire the semaphore.\n\n        .. note:: If this semaphore was initialized with a *value* of 0,\n           this method will block forever (unless a timeout is given or blocking is\n           set to false).\n\n        :keyword bool blocking: If True (the default), this function will block\n           until the semaphore is acquired.\n        :keyword float timeout: If given, and *blocking* is true,\n           specifies the maximum amount of seconds\n           this method will block.\n        :return: A `bool` indicating whether the semaphore was acquired.\n           If ``blocking`` is True and ``timeout`` is None (the default), then\n           (so long as this semaphore was initialized with a size greater than 0)\n           this will always return True. If a timeout was given, and it expired before\n           the semaphore was acquired, False will be returned. (Note that this can still\n           raise a ``Timeout`` exception, if some other caller had already started a timer.)\n        '
        if self._multithreaded is _UNSET:
            self._multithreaded = self._get_thread_ident()
        elif self._multithreaded != self._get_thread_ident():
            self._multithreaded = _MULTI
        invalid_thread_use = None
        try:
            self._capture_hub(False)
        except InvalidThreadUseError as e:
            invalid_thread_use = e.args
            e = None
            if not self.counter and blocking:
                return self.__acquire_from_other_thread(invalid_thread_use, blocking, timeout)
        if self.counter > 0:
            self.counter -= 1
            return True
        if not blocking:
            return False
        if self._multithreaded is not _MULTI and self.hub is None:
            self.hub = get_hub()
        if self.hub is None and (not invalid_thread_use):
            return self.__acquire_from_other_thread((None, None, self._getcurrent(), 'NoHubs'), blocking, timeout)
        try:
            success = self._wait(timeout)
        except LoopExit as ex:
            args = ex.args
            ex = None
            if self.counter:
                success = True
            else:
                if len(args) == 3 and args[1].main_hub:
                    raise
                return self.__acquire_from_other_thread((self.hub, get_hub_if_exists(), self._getcurrent(), 'LoopExit'), blocking, timeout)
        if not success:
            assert timeout is not None
            return False
        assert self.counter > 0, (self.counter, blocking, timeout, success)
        self.counter -= 1
        return True
    _py3k_acquire = acquire

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        self.acquire()

    def __exit__(self, t, v, tb):
        if False:
            while True:
                i = 10
        self.release()

    def _handle_unswitched_notifications(self, unswitched):
        if False:
            while True:
                i = 10
        self._links.extend(unswitched)

    def __add_link(self, link):
        if False:
            for i in range(10):
                print('nop')
        if not self._notifier:
            self.rawlink(link)
        else:
            self._notifier.args[0].append(link)

    def __acquire_from_other_thread(self, ex_args, blocking, timeout):
        if False:
            return 10
        assert blocking
        owning_hub = ex_args[0]
        hub_for_this_thread = ex_args[1]
        current_greenlet = ex_args[2]
        if owning_hub is None and hub_for_this_thread is None:
            return self.__acquire_without_hubs(timeout)
        if hub_for_this_thread is None:
            return self.__acquire_using_other_hub(owning_hub, timeout)
        return self.__acquire_using_two_hubs(hub_for_this_thread, current_greenlet, timeout)

    def __acquire_using_two_hubs(self, hub_for_this_thread, current_greenlet, timeout):
        if False:
            i = 10
            return i + 15
        watcher = hub_for_this_thread.loop.async_()
        send = watcher.send_ignoring_arg
        watcher.start(current_greenlet.switch, self)
        try:
            with Timeout._start_new_or_dummy(timeout) as timer:
                try:
                    while 1:
                        if self.counter > 0:
                            self.counter -= 1
                            assert self.counter >= 0, (self,)
                            return True
                        self.__add_link(send)
                        self._switch_to_hub(hub_for_this_thread)
                        result = self.acquire(0)
                        if result:
                            return result
                except Timeout as tex:
                    if tex is not timer:
                        raise
                    return False
        finally:
            self._quiet_unlink_all(send)
            watcher.stop()
            watcher.close()

    def __acquire_from_other_thread_cb(self, results, blocking, timeout, thread_lock):
        if False:
            for i in range(10):
                print('nop')
        try:
            result = self.acquire(blocking, timeout)
            results.append(result)
        finally:
            thread_lock.release()
        return result

    def __acquire_using_other_hub(self, owning_hub, timeout):
        if False:
            for i in range(10):
                print('nop')
        assert owning_hub is not get_hub_if_exists()
        thread_lock = self._allocate_lock()
        thread_lock.acquire()
        results = []
        owning_hub.loop.run_callback_threadsafe(spawn_raw, self.__acquire_from_other_thread_cb, results, 1, timeout, thread_lock)
        self.__spin_on_native_lock(thread_lock, None)
        return results[0]

    def __acquire_without_hubs(self, timeout):
        if False:
            return 10
        thread_lock = self._allocate_lock()
        thread_lock.acquire()
        absolute_expiration = 0
        begin = 0
        if timeout:
            absolute_expiration = monotonic() + timeout
        link = _LockReleaseLink(thread_lock)
        while 1:
            self.__add_link(link)
            if absolute_expiration:
                begin = monotonic()
            got_native = self.__spin_on_native_lock(thread_lock, timeout)
            self._quiet_unlink_all(link)
            if got_native:
                if self.acquire(0):
                    return True
            if absolute_expiration:
                now = monotonic()
                if now >= absolute_expiration:
                    return False
                duration = now - begin
                timeout -= duration
                if timeout <= 0:
                    return False

    def __spin_on_native_lock(self, thread_lock, timeout):
        if False:
            for i in range(10):
                print('nop')
        expiration = 0
        if timeout:
            expiration = monotonic() + timeout
        self._drop_lock_for_switch_out()
        try:
            while not thread_lock.acquire(0):
                if expiration and monotonic() >= expiration:
                    return False
                _native_sleep(0.001)
            return True
        finally:
            self._acquire_lock_for_switch_in()

class BoundedSemaphore(Semaphore):
    """
    BoundedSemaphore(value=1) -> BoundedSemaphore

    A bounded semaphore checks to make sure its current value doesn't
    exceed its initial value. If it does, :class:`ValueError` is
    raised. In most situations semaphores are used to guard resources
    with limited capacity. If the semaphore is released too many times
    it's a sign of a bug.

    If not given, *value* defaults to 1.
    """
    __slots__ = ('_initial_value',)
    _OVER_RELEASE_ERROR = ValueError

    def __init__(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        Semaphore.__init__(self, *args, **kwargs)
        self._initial_value = self.counter

    def release(self):
        if False:
            while True:
                i = 10
        '\n        Like :meth:`Semaphore.release`, but raises :class:`ValueError`\n        if the semaphore is being over-released.\n        '
        if self.counter >= self._initial_value:
            raise self._OVER_RELEASE_ERROR('Semaphore released too many times')
        counter = Semaphore.release(self)
        if counter == self._initial_value:
            self.hub = None
        return counter

    def _at_fork_reinit(self):
        if False:
            while True:
                i = 10
        super(BoundedSemaphore, self)._at_fork_reinit()
        self.counter = self._initial_value
from gevent._util import import_c_accel
import_c_accel(globals(), 'gevent.__semaphore')