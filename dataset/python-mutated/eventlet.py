"""Eventlet execution pool."""
import sys
from time import monotonic
from greenlet import GreenletExit
from kombu.asynchronous import timer as _timer
from celery import signals
from . import base
__all__ = ('TaskPool',)
W_RACE = 'Celery module with %s imported before eventlet patched'
RACE_MODS = ('billiard.', 'celery.', 'kombu.')
for mod in (mod for mod in sys.modules if mod.startswith(RACE_MODS)):
    for side in ('thread', 'threading', 'socket'):
        if getattr(mod, side, None):
            import warnings
            warnings.warn(RuntimeWarning(W_RACE % side))

def apply_target(target, args=(), kwargs=None, callback=None, accept_callback=None, getpid=None):
    if False:
        for i in range(10):
            print('nop')
    kwargs = {} if not kwargs else kwargs
    return base.apply_target(target, args, kwargs, callback, accept_callback, pid=getpid())

class Timer(_timer.Timer):
    """Eventlet Timer."""

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from eventlet.greenthread import spawn_after
        from greenlet import GreenletExit
        super().__init__(*args, **kwargs)
        self.GreenletExit = GreenletExit
        self._spawn_after = spawn_after
        self._queue = set()

    def _enter(self, eta, priority, entry, **kwargs):
        if False:
            i = 10
            return i + 15
        secs = max(eta - monotonic(), 0)
        g = self._spawn_after(secs, entry)
        self._queue.add(g)
        g.link(self._entry_exit, entry)
        g.entry = entry
        g.eta = eta
        g.priority = priority
        g.canceled = False
        return g

    def _entry_exit(self, g, entry):
        if False:
            return 10
        try:
            try:
                g.wait()
            except self.GreenletExit:
                entry.cancel()
                g.canceled = True
        finally:
            self._queue.discard(g)

    def clear(self):
        if False:
            return 10
        queue = self._queue
        while queue:
            try:
                queue.pop().cancel()
            except (KeyError, self.GreenletExit):
                pass

    def cancel(self, tref):
        if False:
            while True:
                i = 10
        try:
            tref.cancel()
        except self.GreenletExit:
            pass

    @property
    def queue(self):
        if False:
            for i in range(10):
                print('nop')
        return self._queue

class TaskPool(base.BasePool):
    """Eventlet Task Pool."""
    Timer = Timer
    signal_safe = False
    is_green = True
    task_join_will_block = False
    _pool = None
    _pool_map = None
    _quick_put = None

    def __init__(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        from eventlet import greenthread
        from eventlet.greenpool import GreenPool
        self.Pool = GreenPool
        self.getcurrent = greenthread.getcurrent
        self.getpid = lambda : id(greenthread.getcurrent())
        self.spawn_n = greenthread.spawn_n
        super().__init__(*args, **kwargs)

    def on_start(self):
        if False:
            for i in range(10):
                print('nop')
        self._pool = self.Pool(self.limit)
        self._pool_map = {}
        signals.eventlet_pool_started.send(sender=self)
        self._quick_put = self._pool.spawn
        self._quick_apply_sig = signals.eventlet_pool_apply.send

    def on_stop(self):
        if False:
            while True:
                i = 10
        signals.eventlet_pool_preshutdown.send(sender=self)
        if self._pool is not None:
            self._pool.waitall()
        signals.eventlet_pool_postshutdown.send(sender=self)

    def on_apply(self, target, args=None, kwargs=None, callback=None, accept_callback=None, **_):
        if False:
            i = 10
            return i + 15
        target = TaskPool._make_killable_target(target)
        self._quick_apply_sig(sender=self, target=target, args=args, kwargs=kwargs)
        greenlet = self._quick_put(apply_target, target, args, kwargs, callback, accept_callback, self.getpid)
        self._add_to_pool_map(id(greenlet), greenlet)

    def grow(self, n=1):
        if False:
            while True:
                i = 10
        limit = self.limit + n
        self._pool.resize(limit)
        self.limit = limit

    def shrink(self, n=1):
        if False:
            return 10
        limit = self.limit - n
        self._pool.resize(limit)
        self.limit = limit

    def terminate_job(self, pid, signal=None):
        if False:
            i = 10
            return i + 15
        if pid in self._pool_map.keys():
            greenlet = self._pool_map[pid]
            greenlet.kill()
            greenlet.wait()

    def _get_info(self):
        if False:
            print('Hello World!')
        info = super()._get_info()
        info.update({'max-concurrency': self.limit, 'free-threads': self._pool.free(), 'running-threads': self._pool.running()})
        return info

    @staticmethod
    def _make_killable_target(target):
        if False:
            while True:
                i = 10

        def killable_target(*args, **kwargs):
            if False:
                for i in range(10):
                    print('nop')
            try:
                return target(*args, **kwargs)
            except GreenletExit:
                return (False, None, None)
        return killable_target

    def _add_to_pool_map(self, pid, greenlet):
        if False:
            while True:
                i = 10
        self._pool_map[pid] = greenlet
        greenlet.link(TaskPool._cleanup_after_job_finish, self._pool_map, pid)

    @staticmethod
    def _cleanup_after_job_finish(greenlet, pool_map, pid):
        if False:
            return 10
        del pool_map[pid]