import threading
import warnings
from threading import Thread
from time import monotonic, sleep
from typing import TYPE_CHECKING, Optional
from ..logging import get_logger
from .middleware import Middleware
from .threading import Interrupt, current_platform, is_gevent_active, raise_thread_exception, supported_platforms
if TYPE_CHECKING:
    import gevent

class TimeLimitExceeded(Interrupt):
    """Exception used to interrupt worker threads when actors exceed
    their time limits.
    """

class TimeLimit(Middleware):
    """Middleware that cancels actors that run for too long.
    Currently, this is only available on CPython.

    Note:
      This works by setting an async exception in the worker thread
      that runs the actor.  This means that the exception will only get
      called the next time that thread acquires the GIL.  Concretely,
      this means that this middleware can't cancel system calls.

    Parameters:
      time_limit(float): The maximum number of milliseconds actors may
        run for. Use `float("inf")` to avoid setting a timeout for the
        actor.
      interval(int): The interval (in milliseconds) with which to
        check for actors that have exceeded the limit. This does not take
        effect when using gevent because the timers are managed by gevent.
    """

    def __init__(self, *, time_limit=600000, interval=1000):
        if False:
            i = 10
            return i + 15
        self.logger = get_logger(__name__, type(self))
        self.time_limit = time_limit
        if is_gevent_active():
            self.manager = _GeventTimeoutManager(logger=self.logger)
        else:
            self.manager = _CtypesTimeoutManager(interval, logger=self.logger)

    @property
    def actor_options(self):
        if False:
            i = 10
            return i + 15
        return {'time_limit'}

    def after_process_boot(self, broker):
        if False:
            return 10
        if is_gevent_active() or current_platform in supported_platforms:
            self.manager.start()
        else:
            msg = 'TimeLimit cannot kill threads on your current platform (%r).'
            warnings.warn(msg % current_platform, category=RuntimeWarning, stacklevel=2)

    def before_process_message(self, broker, message):
        if False:
            i = 10
            return i + 15
        actor = broker.get_actor(message.actor_name)
        limit = message.options.get('time_limit') or actor.options.get('time_limit', self.time_limit)
        self.manager.add_timeout(threading.get_ident(), limit)

    def after_process_message(self, broker, message, *, result=None, exception=None):
        if False:
            i = 10
            return i + 15
        self.manager.remove_timeout(threading.get_ident())
    after_skip_message = after_process_message

class _CtypesTimeoutManager(Thread):

    def __init__(self, interval, logger=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(daemon=True)
        self.deadlines = {}
        self.interval = interval / 1000
        self.logger = logger or get_logger(__name__, type(self))
        self.mu = threading.RLock()

    def _handle(self):
        if False:
            while True:
                i = 10
        current_time = monotonic()
        threads_to_kill = []
        with self.mu:
            for (thread_id, deadline) in self.deadlines.items():
                if deadline and current_time >= deadline:
                    self.logger.warning('Time limit exceeded. Raising exception in worker thread %r.', thread_id)
                    self.deadlines[thread_id] = None
                    threads_to_kill.append(thread_id)
        for thread_id in threads_to_kill:
            raise_thread_exception(thread_id, TimeLimitExceeded)

    def run(self):
        if False:
            print('Hello World!')
        while True:
            try:
                self._handle()
            except Exception:
                self.logger.exception('Unhandled error while running the time limit handler.')
            sleep(self.interval)

    def add_timeout(self, thread_id, ttl):
        if False:
            print('Hello World!')
        with self.mu:
            self.deadlines[thread_id] = monotonic() + ttl / 1000

    def remove_timeout(self, thread_id):
        if False:
            while True:
                i = 10
        with self.mu:
            self.deadlines[thread_id] = None

class _GeventTimeoutManager:

    def __init__(self, logger=None):
        if False:
            while True:
                i = 10
        self.timers = {}
        self.logger = logger or get_logger(__name__, type(self))

    def start(self):
        if False:
            for i in range(10):
                print('nop')
        pass

    def add_timeout(self, thread_id, ttl):
        if False:
            i = 10
            return i + 15
        self.timers[thread_id] = _GeventTimeout(logger=self.logger, thread_id=thread_id, after_expiration=lambda : self.timers.pop(thread_id, None), seconds=None if ttl == float('inf') else ttl / 1000, exception=TimeLimitExceeded)
        self.timers[thread_id].start()

    def remove_timeout(self, thread_id):
        if False:
            return 10
        timer = self.timers.pop(thread_id, None)
        if timer is not None:
            timer.close()
_GeventTimeout: Optional['gevent.Timeout'] = None
if is_gevent_active():
    from gevent import Timeout

    class __GeventTimeout(Timeout):
        """Cooperative timeout class for gevent with logging on timeouts."""

        def __init__(self, *args, logger=None, thread_id=None, after_expiration=None, **kwargs):
            if False:
                print('Hello World!')
            super().__init__(*args, **kwargs)
            self.logger = logger or get_logger(__name__, type(self))
            self.thread_id = thread_id
            self.after_expiration = after_expiration

        def _on_expiration(self, prev_greenlet, ex):
            if False:
                i = 10
                return i + 15
            self.logger.warning('Time limit exceeded. Raising exception in worker thread %r.', self.thread_id)
            res = super()._on_expiration(prev_greenlet, ex)
            if self.after_expiration is not None:
                self.after_expiration()
            return res
    _GeventTimeout = __GeventTimeout