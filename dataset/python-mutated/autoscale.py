"""Pool Autoscaling.

This module implements the internal thread responsible
for growing and shrinking the pool according to the
current autoscale settings.

The autoscale thread is only enabled if
the :option:`celery worker --autoscale` option is used.
"""
import os
import threading
from time import monotonic, sleep
from kombu.asynchronous.semaphore import DummyLock
from celery import bootsteps
from celery.utils.log import get_logger
from celery.utils.threads import bgThread
from . import state
from .components import Pool
__all__ = ('Autoscaler', 'WorkerComponent')
logger = get_logger(__name__)
(debug, info, error) = (logger.debug, logger.info, logger.error)
AUTOSCALE_KEEPALIVE = float(os.environ.get('AUTOSCALE_KEEPALIVE', 30))

class WorkerComponent(bootsteps.StartStopStep):
    """Bootstep that starts the autoscaler thread/timer in the worker."""
    label = 'Autoscaler'
    conditional = True
    requires = (Pool,)

    def __init__(self, w, **kwargs):
        if False:
            while True:
                i = 10
        self.enabled = w.autoscale
        w.autoscaler = None

    def create(self, w):
        if False:
            print('Hello World!')
        scaler = w.autoscaler = self.instantiate(w.autoscaler_cls, w.pool, w.max_concurrency, w.min_concurrency, worker=w, mutex=DummyLock() if w.use_eventloop else None)
        return scaler if not w.use_eventloop else None

    def register_with_event_loop(self, w, hub):
        if False:
            i = 10
            return i + 15
        w.consumer.on_task_message.add(w.autoscaler.maybe_scale)
        hub.call_repeatedly(w.autoscaler.keepalive, w.autoscaler.maybe_scale)

    def info(self, w):
        if False:
            print('Hello World!')
        'Return `Autoscaler` info.'
        return {'autoscaler': w.autoscaler.info()}

class Autoscaler(bgThread):
    """Background thread to autoscale pool workers."""

    def __init__(self, pool, max_concurrency, min_concurrency=0, worker=None, keepalive=AUTOSCALE_KEEPALIVE, mutex=None):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.pool = pool
        self.mutex = mutex or threading.Lock()
        self.max_concurrency = max_concurrency
        self.min_concurrency = min_concurrency
        self.keepalive = keepalive
        self._last_scale_up = None
        self.worker = worker
        assert self.keepalive, 'cannot scale down too fast.'

    def body(self):
        if False:
            print('Hello World!')
        with self.mutex:
            self.maybe_scale()
        sleep(1.0)

    def _maybe_scale(self, req=None):
        if False:
            for i in range(10):
                print('nop')
        procs = self.processes
        cur = min(self.qty, self.max_concurrency)
        if cur > procs:
            self.scale_up(cur - procs)
            return True
        cur = max(self.qty, self.min_concurrency)
        if cur < procs:
            self.scale_down(procs - cur)
            return True

    def maybe_scale(self, req=None):
        if False:
            return 10
        if self._maybe_scale(req):
            self.pool.maintain_pool()

    def update(self, max=None, min=None):
        if False:
            while True:
                i = 10
        with self.mutex:
            if max is not None:
                if max < self.processes:
                    self._shrink(self.processes - max)
                self._update_consumer_prefetch_count(max)
                self.max_concurrency = max
            if min is not None:
                if min > self.processes:
                    self._grow(min - self.processes)
                self.min_concurrency = min
            return (self.max_concurrency, self.min_concurrency)

    def scale_up(self, n):
        if False:
            for i in range(10):
                print('nop')
        self._last_scale_up = monotonic()
        return self._grow(n)

    def scale_down(self, n):
        if False:
            return 10
        if self._last_scale_up and monotonic() - self._last_scale_up > self.keepalive:
            return self._shrink(n)

    def _grow(self, n):
        if False:
            return 10
        info('Scaling up %s processes.', n)
        self.pool.grow(n)

    def _shrink(self, n):
        if False:
            print('Hello World!')
        info('Scaling down %s processes.', n)
        try:
            self.pool.shrink(n)
        except ValueError:
            debug("Autoscaler won't scale down: all processes busy.")
        except Exception as exc:
            error('Autoscaler: scale_down: %r', exc, exc_info=True)

    def _update_consumer_prefetch_count(self, new_max):
        if False:
            print('Hello World!')
        diff = new_max - self.max_concurrency
        if diff:
            self.worker.consumer._update_prefetch_count(diff)

    def info(self):
        if False:
            print('Hello World!')
        return {'max': self.max_concurrency, 'min': self.min_concurrency, 'current': self.processes, 'qty': self.qty}

    @property
    def qty(self):
        if False:
            while True:
                i = 10
        return len(state.reserved_requests)

    @property
    def processes(self):
        if False:
            for i in range(10):
                print('nop')
        return self.pool.num_processes