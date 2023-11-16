"""Prefork execution pool.

Pool implementation using :mod:`multiprocessing`.
"""
import os
from billiard import forking_enable
from billiard.common import REMAP_SIGTERM, TERM_SIGNAME
from billiard.pool import CLOSE, RUN
from billiard.pool import Pool as BlockingPool
from celery import platforms, signals
from celery._state import _set_task_join_will_block, set_default_app
from celery.app import trace
from celery.concurrency.base import BasePool
from celery.utils.functional import noop
from celery.utils.log import get_logger
from .asynpool import AsynPool
__all__ = ('TaskPool', 'process_initializer', 'process_destructor')
WORKER_SIGRESET = {'SIGTERM', 'SIGHUP', 'SIGTTIN', 'SIGTTOU', 'SIGUSR1'}
if REMAP_SIGTERM:
    WORKER_SIGIGNORE = {'SIGINT', TERM_SIGNAME}
else:
    WORKER_SIGIGNORE = {'SIGINT'}
logger = get_logger(__name__)
(warning, debug) = (logger.warning, logger.debug)

def process_initializer(app, hostname):
    if False:
        for i in range(10):
            print('nop')
    'Pool child process initializer.\n\n    Initialize the child pool process to ensure the correct\n    app instance is used and things like logging works.\n    '
    platforms.set_pdeathsig('SIGKILL')
    _set_task_join_will_block(True)
    platforms.signals.reset(*WORKER_SIGRESET)
    platforms.signals.ignore(*WORKER_SIGIGNORE)
    platforms.set_mp_process_title('celeryd', hostname=hostname)
    app.loader.init_worker()
    app.loader.init_worker_process()
    logfile = os.environ.get('CELERY_LOG_FILE') or None
    if logfile and '%i' in logfile.lower():
        app.log.already_setup = False
    app.log.setup(int(os.environ.get('CELERY_LOG_LEVEL', 0) or 0), logfile, bool(os.environ.get('CELERY_LOG_REDIRECT', False)), str(os.environ.get('CELERY_LOG_REDIRECT_LEVEL')), hostname=hostname)
    if os.environ.get('FORKED_BY_MULTIPROCESSING'):
        trace.setup_worker_optimizations(app, hostname)
    else:
        app.set_current()
        set_default_app(app)
        app.finalize()
        trace._tasks = app._tasks
    from celery.app.trace import build_tracer
    for (name, task) in app.tasks.items():
        task.__trace__ = build_tracer(name, task, app.loader, hostname, app=app)
    from celery.worker import state as worker_state
    worker_state.reset_state()
    signals.worker_process_init.send(sender=None)

def process_destructor(pid, exitcode):
    if False:
        i = 10
        return i + 15
    'Pool child process destructor.\n\n    Dispatch the :signal:`worker_process_shutdown` signal.\n    '
    signals.worker_process_shutdown.send(sender=None, pid=pid, exitcode=exitcode)

class TaskPool(BasePool):
    """Multiprocessing Pool implementation."""
    Pool = AsynPool
    BlockingPool = BlockingPool
    uses_semaphore = True
    write_stats = None

    def on_start(self):
        if False:
            while True:
                i = 10
        forking_enable(self.forking_enable)
        Pool = self.BlockingPool if self.options.get('threads', True) else self.Pool
        proc_alive_timeout = self.app.conf.worker_proc_alive_timeout if self.app else None
        P = self._pool = Pool(processes=self.limit, initializer=process_initializer, on_process_exit=process_destructor, enable_timeouts=True, synack=False, proc_alive_timeout=proc_alive_timeout, **self.options)
        self.on_apply = P.apply_async
        self.maintain_pool = P.maintain_pool
        self.terminate_job = P.terminate_job
        self.grow = P.grow
        self.shrink = P.shrink
        self.flush = getattr(P, 'flush', None)

    def restart(self):
        if False:
            for i in range(10):
                print('nop')
        self._pool.restart()
        self._pool.apply_async(noop)

    def did_start_ok(self):
        if False:
            for i in range(10):
                print('nop')
        return self._pool.did_start_ok()

    def register_with_event_loop(self, loop):
        if False:
            print('Hello World!')
        try:
            reg = self._pool.register_with_event_loop
        except AttributeError:
            return
        return reg(loop)

    def on_stop(self):
        if False:
            while True:
                i = 10
        'Gracefully stop the pool.'
        if self._pool is not None and self._pool._state in (RUN, CLOSE):
            self._pool.close()
            self._pool.join()
            self._pool = None

    def on_terminate(self):
        if False:
            i = 10
            return i + 15
        'Force terminate the pool.'
        if self._pool is not None:
            self._pool.terminate()
            self._pool = None

    def on_close(self):
        if False:
            while True:
                i = 10
        if self._pool is not None and self._pool._state == RUN:
            self._pool.close()

    def _get_info(self):
        if False:
            print('Hello World!')
        write_stats = getattr(self._pool, 'human_write_stats', None)
        info = super()._get_info()
        info.update({'max-concurrency': self.limit, 'processes': [p.pid for p in self._pool._pool], 'max-tasks-per-child': self._pool._maxtasksperchild or 'N/A', 'put-guarded-by-semaphore': self.putlocks, 'timeouts': (self._pool.soft_timeout or 0, self._pool.timeout or 0), 'writes': write_stats() if write_stats is not None else 'N/A'})
        return info

    @property
    def num_processes(self):
        if False:
            print('Hello World!')
        return self._pool._processes