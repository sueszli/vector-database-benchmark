"""Version of multiprocessing.Pool using Async I/O.

.. note::

    This module will be moved soon, so don't use it directly.

This is a non-blocking version of :class:`multiprocessing.Pool`.

This code deals with three major challenges:

#. Starting up child processes and keeping them running.
#. Sending jobs to the processes and receiving results back.
#. Safely shutting down this system.
"""
import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
try:
    from _billiard import read as __read__
    readcanbuf = True
except ImportError:

    def __read__(fd, buf, size, read=os.read):
        if False:
            for i in range(10):
                print('nop')
        chunk = read(fd, size)
        n = len(chunk)
        if n != 0:
            buf.write(chunk)
        return n
    readcanbuf = False

    def unpack_from(fmt, iobuf, unpack=unpack):
        if False:
            return 10
        return unpack(fmt, iobuf.getvalue())
__all__ = ('AsynPool',)
logger = get_logger(__name__)
(error, debug) = (logger.error, logger.debug)
UNAVAIL = frozenset({errno.EAGAIN, errno.EINTR})
WORKER_UP = 15
PROC_ALIVE_TIMEOUT = 4.0
SCHED_STRATEGY_FCFS = 1
SCHED_STRATEGY_FAIR = 4
SCHED_STRATEGIES = {None: SCHED_STRATEGY_FAIR, 'default': SCHED_STRATEGY_FAIR, 'fast': SCHED_STRATEGY_FCFS, 'fcfs': SCHED_STRATEGY_FCFS, 'fair': SCHED_STRATEGY_FAIR}
SCHED_STRATEGY_TO_NAME = {v: k for (k, v) in SCHED_STRATEGIES.items()}
Ack = namedtuple('Ack', ('id', 'fd', 'payload'))

def gen_not_started(gen):
    if False:
        i = 10
        return i + 15
    'Return true if generator is not started.'
    return inspect.getgeneratorstate(gen) == 'GEN_CREATED'

def _get_job_writer(job):
    if False:
        return 10
    try:
        writer = job._writer
    except AttributeError:
        pass
    else:
        return writer()
if hasattr(select, 'poll'):

    def _select_imp(readers=None, writers=None, err=None, timeout=0, poll=select.poll, POLLIN=select.POLLIN, POLLOUT=select.POLLOUT, POLLERR=select.POLLERR):
        if False:
            for i in range(10):
                print('nop')
        poller = poll()
        register = poller.register
        if readers:
            [register(fd, POLLIN) for fd in readers]
        if writers:
            [register(fd, POLLOUT) for fd in writers]
        if err:
            [register(fd, POLLERR) for fd in err]
        (R, W) = (set(), set())
        timeout = 0 if timeout and timeout < 0 else round(timeout * 1000.0)
        events = poller.poll(timeout)
        for (fd, event) in events:
            if not isinstance(fd, Integral):
                fd = fd.fileno()
            if event & POLLIN:
                R.add(fd)
            if event & POLLOUT:
                W.add(fd)
            if event & POLLERR:
                R.add(fd)
        return (R, W, 0)
else:

    def _select_imp(readers=None, writers=None, err=None, timeout=0):
        if False:
            return 10
        (r, w, e) = select.select(readers, writers, err, timeout)
        if e:
            r = list(set(r) | set(e))
        return (r, w, 0)

def _select(readers=None, writers=None, err=None, timeout=0, poll=_select_imp):
    if False:
        for i in range(10):
            print('nop')
    "Simple wrapper to :class:`~select.select`, using :`~select.poll`.\n\n    Arguments:\n        readers (Set[Fd]): Set of reader fds to test if readable.\n        writers (Set[Fd]): Set of writer fds to test if writable.\n        err (Set[Fd]): Set of fds to test for error condition.\n\n    All fd sets passed must be mutable as this function\n    will remove non-working fds from them, this also means\n    the caller must make sure there are still fds in the sets\n    before calling us again.\n\n    Returns:\n        Tuple[Set, Set, Set]: of ``(readable, writable, again)``, where\n        ``readable`` is a set of fds that have data available for read,\n        ``writable`` is a set of fds that's ready to be written to\n        and ``again`` is a flag that if set means the caller must\n        throw away the result and call us again.\n    "
    readers = set() if readers is None else readers
    writers = set() if writers is None else writers
    err = set() if err is None else err
    try:
        return poll(readers, writers, err, timeout)
    except OSError as exc:
        _errno = exc.errno
        if _errno == errno.EINTR:
            return (set(), set(), 1)
        elif _errno in SELECT_BAD_FD:
            for fd in readers | writers | err:
                try:
                    select.select([fd], [], [], 0)
                except OSError as exc:
                    _errno = exc.errno
                    if _errno not in SELECT_BAD_FD:
                        raise
                    readers.discard(fd)
                    writers.discard(fd)
                    err.discard(fd)
            return (set(), set(), 1)
        else:
            raise

def iterate_file_descriptors_safely(fds_iter, source_data, hub_method, *args, **kwargs):
    if False:
        while True:
            i = 10
    "Apply hub method to fds in iter, remove from list if failure.\n\n    Some file descriptors may become stale through OS reasons\n    or possibly other reasons, so safely manage our lists of FDs.\n    :param fds_iter: the file descriptors to iterate and apply hub_method\n    :param source_data: data source to remove FD if it renders OSError\n    :param hub_method: the method to call with with each fd and kwargs\n    :*args to pass through to the hub_method;\n    with a special syntax string '*fd*' represents a substitution\n    for the current fd object in the iteration (for some callers).\n    :**kwargs to pass through to the hub method (no substitutions needed)\n    "

    def _meta_fd_argument_maker():
        if False:
            return 10
        call_args = args
        if '*fd*' in call_args:
            call_args = [fd if arg == '*fd*' else arg for arg in args]
        return call_args
    stale_fds = []
    for fd in fds_iter:
        (hub_args, hub_kwargs) = (_meta_fd_argument_maker(), kwargs)
        try:
            hub_method(fd, *hub_args, **hub_kwargs)
        except (OSError, FileNotFoundError):
            logger.warning('Encountered OSError when accessing fd %s ', fd, exc_info=True)
            stale_fds.append(fd)
    if source_data:
        for fd in stale_fds:
            try:
                if hasattr(source_data, 'remove'):
                    source_data.remove(fd)
                else:
                    source_data.pop(fd, None)
            except ValueError:
                logger.warning('ValueError trying to invalidate %s from %s', fd, source_data)

class Worker(_pool.Worker):
    """Pool worker process."""

    def on_loop_start(self, pid):
        if False:
            for i in range(10):
                print('nop')
        self.outq.put((WORKER_UP, (pid,)))

class ResultHandler(_pool.ResultHandler):
    """Handles messages from the pool processes."""

    def __init__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.fileno_to_outq = kwargs.pop('fileno_to_outq')
        self.on_process_alive = kwargs.pop('on_process_alive')
        super().__init__(*args, **kwargs)
        self.state_handlers[WORKER_UP] = self.on_process_alive

    def _recv_message(self, add_reader, fd, callback, __read__=__read__, readcanbuf=readcanbuf, BytesIO=BytesIO, unpack_from=unpack_from, load=_pickle.load):
        if False:
            for i in range(10):
                print('nop')
        Hr = Br = 0
        if readcanbuf:
            buf = bytearray(4)
            bufv = memoryview(buf)
        else:
            buf = bufv = BytesIO()
        while Hr < 4:
            try:
                n = __read__(fd, bufv[Hr:] if readcanbuf else bufv, 4 - Hr)
            except OSError as exc:
                if exc.errno not in UNAVAIL:
                    raise
                yield
            else:
                if n == 0:
                    raise OSError('End of file during message') if Hr else EOFError()
                Hr += n
        (body_size,) = unpack_from('>i', bufv)
        if readcanbuf:
            buf = bytearray(body_size)
            bufv = memoryview(buf)
        else:
            buf = bufv = BytesIO()
        while Br < body_size:
            try:
                n = __read__(fd, bufv[Br:] if readcanbuf else bufv, body_size - Br)
            except OSError as exc:
                if exc.errno not in UNAVAIL:
                    raise
                yield
            else:
                if n == 0:
                    raise OSError('End of file during message') if Br else EOFError()
                Br += n
        add_reader(fd, self.handle_event, fd)
        if readcanbuf:
            message = load(BytesIO(bufv))
        else:
            bufv.seek(0)
            message = load(bufv)
        if message:
            callback(message)

    def _make_process_result(self, hub):
        if False:
            while True:
                i = 10
        'Coroutine reading messages from the pool processes.'
        fileno_to_outq = self.fileno_to_outq
        on_state_change = self.on_state_change
        add_reader = hub.add_reader
        remove_reader = hub.remove_reader
        recv_message = self._recv_message

        def on_result_readable(fileno):
            if False:
                return 10
            try:
                fileno_to_outq[fileno]
            except KeyError:
                return remove_reader(fileno)
            it = recv_message(add_reader, fileno, on_state_change)
            try:
                next(it)
            except StopIteration:
                pass
            except (OSError, EOFError):
                remove_reader(fileno)
            else:
                add_reader(fileno, it)
        return on_result_readable

    def register_with_event_loop(self, hub):
        if False:
            for i in range(10):
                print('nop')
        self.handle_event = self._make_process_result(hub)

    def handle_event(self, *args):
        if False:
            return 10
        raise RuntimeError('Not registered with event loop')

    def on_stop_not_started(self):
        if False:
            while True:
                i = 10
        cache = self.cache
        check_timeouts = self.check_timeouts
        fileno_to_outq = self.fileno_to_outq
        on_state_change = self.on_state_change
        join_exited_workers = self.join_exited_workers
        outqueues = set(fileno_to_outq)
        while cache and outqueues and (self._state != TERMINATE):
            if check_timeouts is not None:
                check_timeouts()
            pending_remove_fd = set()
            for fd in outqueues:
                iterate_file_descriptors_safely([fd], self.fileno_to_outq, self._flush_outqueue, pending_remove_fd.add, fileno_to_outq, on_state_change)
                try:
                    join_exited_workers(shutdown=True)
                except WorkersJoined:
                    debug('result handler: all workers terminated')
                    return
            outqueues.difference_update(pending_remove_fd)

    def _flush_outqueue(self, fd, remove, process_index, on_state_change):
        if False:
            return 10
        try:
            proc = process_index[fd]
        except KeyError:
            return remove(fd)
        reader = proc.outq._reader
        try:
            setblocking(reader, 1)
        except OSError:
            return remove(fd)
        try:
            if reader.poll(0):
                task = reader.recv()
            else:
                task = None
                sleep(0.5)
        except (OSError, EOFError):
            return remove(fd)
        else:
            if task:
                on_state_change(task)
        finally:
            try:
                setblocking(reader, 0)
            except OSError:
                return remove(fd)

class AsynPool(_pool.Pool):
    """AsyncIO Pool (no threads)."""
    ResultHandler = ResultHandler
    Worker = Worker
    _registered_with_event_loop = False

    def WorkerProcess(self, worker):
        if False:
            while True:
                i = 10
        worker = super().WorkerProcess(worker)
        worker.dead = False
        return worker

    def __init__(self, processes=None, synack=False, sched_strategy=None, proc_alive_timeout=None, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.sched_strategy = SCHED_STRATEGIES.get(sched_strategy, sched_strategy)
        processes = self.cpu_count() if processes is None else processes
        self.synack = synack
        self._queues = {self.create_process_queues(): None for _ in range(processes)}
        self._fileno_to_inq = {}
        self._fileno_to_outq = {}
        self._fileno_to_synq = {}
        self._proc_alive_timeout = PROC_ALIVE_TIMEOUT if proc_alive_timeout is None else proc_alive_timeout
        self._waiting_to_start = set()
        self._all_inqueues = set()
        self._active_writes = set()
        self._active_writers = set()
        self._busy_workers = set()
        self._mark_worker_as_available = self._busy_workers.discard
        self.outbound_buffer = deque()
        self.write_stats = Counter()
        super().__init__(processes, *args, **kwargs)
        for proc in self._pool:
            self._fileno_to_outq[proc.outqR_fd] = proc
            self._fileno_to_synq[proc.synqW_fd] = proc
        self.on_soft_timeout = getattr(self._timeout_handler, 'on_soft_timeout', noop)
        self.on_hard_timeout = getattr(self._timeout_handler, 'on_hard_timeout', noop)

    def _create_worker_process(self, i):
        if False:
            return 10
        worker_before_create_process.send(sender=self)
        gc.collect()
        return super()._create_worker_process(i)

    def _event_process_exit(self, hub, proc):
        if False:
            while True:
                i = 10
        self._untrack_child_process(proc, hub)
        self.maintain_pool()

    def _track_child_process(self, proc, hub):
        if False:
            i = 10
            return i + 15
        'Helper method determines appropriate fd for process.'
        try:
            fd = proc._sentinel_poll
        except AttributeError:
            fd = proc._sentinel_poll = os.dup(proc._popen.sentinel)
        iterate_file_descriptors_safely([fd], None, hub.add_reader, self._event_process_exit, hub, proc)

    def _untrack_child_process(self, proc, hub):
        if False:
            while True:
                i = 10
        if proc._sentinel_poll is not None:
            (fd, proc._sentinel_poll) = (proc._sentinel_poll, None)
            hub.remove(fd)
            os.close(fd)

    def register_with_event_loop(self, hub):
        if False:
            return 10
        'Register the async pool with the current event loop.'
        self._result_handler.register_with_event_loop(hub)
        self.handle_result_event = self._result_handler.handle_event
        self._create_timelimit_handlers(hub)
        self._create_process_handlers(hub)
        self._create_write_handlers(hub)
        [self._track_child_process(w, hub) for w in self._pool]
        iterate_file_descriptors_safely(self._fileno_to_outq, self._fileno_to_outq, hub.add_reader, self.handle_result_event, '*fd*')
        for (handler, interval) in self.timers.items():
            hub.call_repeatedly(interval, handler)
        if not self._registered_with_event_loop:
            hub.on_tick.add(self.on_poll_start)
            self._registered_with_event_loop = True

    def _create_timelimit_handlers(self, hub):
        if False:
            for i in range(10):
                print('nop')
        'Create handlers used to implement time limits.'
        call_later = hub.call_later
        trefs = self._tref_for_id = WeakValueDictionary()

        def on_timeout_set(R, soft, hard):
            if False:
                for i in range(10):
                    print('nop')
            if soft:
                trefs[R._job] = call_later(soft, self._on_soft_timeout, R._job, soft, hard, hub)
            elif hard:
                trefs[R._job] = call_later(hard, self._on_hard_timeout, R._job)
        self.on_timeout_set = on_timeout_set

        def _discard_tref(job):
            if False:
                while True:
                    i = 10
            try:
                tref = trefs.pop(job)
                tref.cancel()
                del tref
            except (KeyError, AttributeError):
                pass
        self._discard_tref = _discard_tref

        def on_timeout_cancel(R):
            if False:
                print('Hello World!')
            _discard_tref(R._job)
        self.on_timeout_cancel = on_timeout_cancel

    def _on_soft_timeout(self, job, soft, hard, hub):
        if False:
            return 10
        if hard:
            self._tref_for_id[job] = hub.call_later(hard - soft, self._on_hard_timeout, job)
        try:
            result = self._cache[job]
        except KeyError:
            pass
        else:
            self.on_soft_timeout(result)
        finally:
            if not hard:
                self._discard_tref(job)

    def _on_hard_timeout(self, job):
        if False:
            return 10
        try:
            result = self._cache[job]
        except KeyError:
            pass
        else:
            self.on_hard_timeout(result)
        finally:
            self._discard_tref(job)

    def on_job_ready(self, job, i, obj, inqW_fd):
        if False:
            for i in range(10):
                print('nop')
        self._mark_worker_as_available(inqW_fd)

    def _create_process_handlers(self, hub):
        if False:
            return 10
        'Create handlers called on process up/down, etc.'
        (add_reader, remove_reader, remove_writer) = (hub.add_reader, hub.remove_reader, hub.remove_writer)
        cache = self._cache
        all_inqueues = self._all_inqueues
        fileno_to_inq = self._fileno_to_inq
        fileno_to_outq = self._fileno_to_outq
        fileno_to_synq = self._fileno_to_synq
        busy_workers = self._busy_workers
        handle_result_event = self.handle_result_event
        process_flush_queues = self.process_flush_queues
        waiting_to_start = self._waiting_to_start

        def verify_process_alive(proc):
            if False:
                print('Hello World!')
            proc = proc()
            if proc is not None and proc._is_alive() and (proc in waiting_to_start):
                assert proc.outqR_fd in fileno_to_outq
                assert fileno_to_outq[proc.outqR_fd] is proc
                assert proc.outqR_fd in hub.readers
                error('Timed out waiting for UP message from %r', proc)
                os.kill(proc.pid, 9)

        def on_process_up(proc):
            if False:
                i = 10
                return i + 15
            'Called when a process has started.'
            infd = proc.inqW_fd
            for job in cache.values():
                if job._write_to and job._write_to.inqW_fd == infd:
                    job._write_to = proc
                if job._scheduled_for and job._scheduled_for.inqW_fd == infd:
                    job._scheduled_for = proc
            fileno_to_outq[proc.outqR_fd] = proc
            self._track_child_process(proc, hub)
            assert not isblocking(proc.outq._reader)
            add_reader(proc.outqR_fd, handle_result_event, proc.outqR_fd)
            waiting_to_start.add(proc)
            hub.call_later(self._proc_alive_timeout, verify_process_alive, ref(proc))
        self.on_process_up = on_process_up

        def _remove_from_index(obj, proc, index, remove_fun, callback=None):
            if False:
                return 10
            try:
                fd = obj.fileno()
            except OSError:
                return
            try:
                if index[fd] is proc:
                    index.pop(fd, None)
            except KeyError:
                pass
            else:
                remove_fun(fd)
                if callback is not None:
                    callback(fd)
            return fd

        def on_process_down(proc):
            if False:
                for i in range(10):
                    print('nop')
            'Called when a worker process exits.'
            if getattr(proc, 'dead', None):
                return
            process_flush_queues(proc)
            _remove_from_index(proc.outq._reader, proc, fileno_to_outq, remove_reader)
            if proc.synq:
                _remove_from_index(proc.synq._writer, proc, fileno_to_synq, remove_writer)
            inq = _remove_from_index(proc.inq._writer, proc, fileno_to_inq, remove_writer, callback=all_inqueues.discard)
            if inq:
                busy_workers.discard(inq)
            self._untrack_child_process(proc, hub)
            waiting_to_start.discard(proc)
            self._active_writes.discard(proc.inqW_fd)
            remove_writer(proc.inq._writer)
            remove_reader(proc.outq._reader)
            if proc.synqR_fd:
                remove_reader(proc.synq._reader)
            if proc.synqW_fd:
                self._active_writes.discard(proc.synqW_fd)
                remove_reader(proc.synq._writer)
        self.on_process_down = on_process_down

    def _create_write_handlers(self, hub, pack=pack, dumps=_pickle.dumps, protocol=HIGHEST_PROTOCOL):
        if False:
            for i in range(10):
                print('nop')
        'Create handlers used to write data to child processes.'
        fileno_to_inq = self._fileno_to_inq
        fileno_to_synq = self._fileno_to_synq
        outbound = self.outbound_buffer
        pop_message = outbound.popleft
        put_message = outbound.append
        all_inqueues = self._all_inqueues
        active_writes = self._active_writes
        active_writers = self._active_writers
        busy_workers = self._busy_workers
        diff = all_inqueues.difference
        add_writer = hub.add_writer
        (hub_add, hub_remove) = (hub.add, hub.remove)
        mark_write_fd_as_active = active_writes.add
        mark_write_gen_as_active = active_writers.add
        mark_worker_as_busy = busy_workers.add
        write_generator_done = active_writers.discard
        get_job = self._cache.__getitem__
        write_stats = self.write_stats
        is_fair_strategy = self.sched_strategy == SCHED_STRATEGY_FAIR
        revoked_tasks = worker_state.revoked
        getpid = os.getpid
        precalc = {ACK: self._create_payload(ACK, (0,)), NACK: self._create_payload(NACK, (0,))}

        def _put_back(job, _time=time.time):
            if False:
                while True:
                    i = 10
            if job._terminated is not None or job.correlation_id in revoked_tasks:
                if not job._accepted:
                    job._ack(None, _time(), getpid(), None)
                job._set_terminated(job._terminated)
            elif job not in outbound:
                outbound.appendleft(job)
        self._put_back = _put_back

        def on_poll_start():
            if False:
                i = 10
                return i + 15
            inactive = diff(active_writes)
            if is_fair_strategy:
                add_cond = outbound and len(busy_workers) < len(all_inqueues)
            else:
                add_cond = outbound
            if add_cond:
                iterate_file_descriptors_safely(inactive, all_inqueues, hub_add, None, WRITE | ERR, consolidate=True)
            else:
                iterate_file_descriptors_safely(inactive, all_inqueues, hub_remove)
        self.on_poll_start = on_poll_start

        def on_inqueue_close(fd, proc):
            if False:
                print('Hello World!')
            busy_workers.discard(fd)
            try:
                if fileno_to_inq[fd] is proc:
                    fileno_to_inq.pop(fd, None)
                    active_writes.discard(fd)
                    all_inqueues.discard(fd)
            except KeyError:
                pass
        self.on_inqueue_close = on_inqueue_close
        self.hub_remove = hub_remove

        def schedule_writes(ready_fds, total_write_count=None):
            if False:
                i = 10
                return i + 15
            if not total_write_count:
                total_write_count = [0]
            num_ready = len(ready_fds)
            for _ in range(num_ready):
                ready_fd = ready_fds[total_write_count[0] % num_ready]
                total_write_count[0] += 1
                if ready_fd in active_writes:
                    continue
                if is_fair_strategy and ready_fd in busy_workers:
                    continue
                if ready_fd not in all_inqueues:
                    hub_remove(ready_fd)
                    continue
                try:
                    job = pop_message()
                except IndexError:
                    for inqfd in diff(active_writes):
                        hub_remove(inqfd)
                    break
                else:
                    if not job._accepted:
                        try:
                            proc = job._scheduled_for = fileno_to_inq[ready_fd]
                        except KeyError:
                            put_message(job)
                            continue
                        cor = _write_job(proc, ready_fd, job)
                        job._writer = ref(cor)
                        mark_write_gen_as_active(cor)
                        mark_write_fd_as_active(ready_fd)
                        mark_worker_as_busy(ready_fd)
                        try:
                            next(cor)
                        except StopIteration:
                            pass
                        except OSError as exc:
                            if exc.errno != errno.EBADF:
                                raise
                        else:
                            add_writer(ready_fd, cor)
        hub.consolidate_callback = schedule_writes

        def send_job(tup):
            if False:
                i = 10
                return i + 15
            body = dumps(tup, protocol=protocol)
            body_size = len(body)
            header = pack('>I', body_size)
            job = get_job(tup[1][0])
            job._payload = (memoryview(header), memoryview(body), body_size)
            put_message(job)
        self._quick_put = send_job

        def on_not_recovering(proc, fd, job, exc):
            if False:
                return 10
            logger.exception('Process inqueue damaged: %r %r: %r', proc, proc.exitcode, exc)
            if proc._is_alive():
                proc.terminate()
            hub.remove(fd)
            self._put_back(job)

        def _write_job(proc, fd, job):
            if False:
                for i in range(10):
                    print('nop')
            (header, body, body_size) = job._payload
            errors = 0
            try:
                job._write_to = proc
                send = proc.send_job_offset
                Hw = Bw = 0
                while Hw < 4:
                    try:
                        Hw += send(header, Hw)
                    except Exception as exc:
                        if getattr(exc, 'errno', None) not in UNAVAIL:
                            raise
                        errors += 1
                        if errors > 100:
                            on_not_recovering(proc, fd, job, exc)
                            raise StopIteration()
                        yield
                    else:
                        errors = 0
                while Bw < body_size:
                    try:
                        Bw += send(body, Bw)
                    except Exception as exc:
                        if getattr(exc, 'errno', None) not in UNAVAIL:
                            raise
                        errors += 1
                        if errors > 100:
                            on_not_recovering(proc, fd, job, exc)
                            raise StopIteration()
                        yield
                    else:
                        errors = 0
            finally:
                hub_remove(fd)
                write_stats[proc.index] += 1
                active_writes.discard(fd)
                write_generator_done(job._writer())

        def send_ack(response, pid, job, fd):
            if False:
                return 10
            msg = Ack(job, fd, precalc[response])
            callback = promise(write_generator_done)
            cor = _write_ack(fd, msg, callback=callback)
            mark_write_gen_as_active(cor)
            mark_write_fd_as_active(fd)
            callback.args = (cor,)
            add_writer(fd, cor)
        self.send_ack = send_ack

        def _write_ack(fd, ack, callback=None):
            if False:
                for i in range(10):
                    print('nop')
            (header, body, body_size) = ack[2]
            try:
                try:
                    proc = fileno_to_synq[fd]
                except KeyError:
                    raise StopIteration()
                send = proc.send_syn_offset
                Hw = Bw = 0
                while Hw < 4:
                    try:
                        Hw += send(header, Hw)
                    except Exception as exc:
                        if getattr(exc, 'errno', None) not in UNAVAIL:
                            raise
                        yield
                while Bw < body_size:
                    try:
                        Bw += send(body, Bw)
                    except Exception as exc:
                        if getattr(exc, 'errno', None) not in UNAVAIL:
                            raise
                        yield
            finally:
                if callback:
                    callback()
                active_writes.discard(fd)

    def flush(self):
        if False:
            for i in range(10):
                print('nop')
        if self._state == TERMINATE:
            return
        if self.synack:
            for job in self._cache.values():
                if not job._accepted:
                    job._cancel()
        if self.outbound_buffer:
            self.outbound_buffer.clear()
        self.maintain_pool()
        try:
            if self._state == RUN:
                intervals = fxrange(0.01, 0.1, 0.01, repeatlast=True)
                owned_by = {}
                for job in self._cache.values():
                    writer = _get_job_writer(job)
                    if writer is not None:
                        owned_by[writer] = job
                if not self._active_writers:
                    self._cache.clear()
                else:
                    while self._active_writers:
                        writers = list(self._active_writers)
                        for gen in writers:
                            if gen.__name__ == '_write_job' and gen_not_started(gen):
                                try:
                                    job = owned_by[gen]
                                except KeyError:
                                    pass
                                else:
                                    job.discard()
                                self._active_writers.discard(gen)
                            else:
                                try:
                                    job = owned_by[gen]
                                except KeyError:
                                    pass
                                else:
                                    job_proc = job._write_to
                                    if job_proc._is_alive():
                                        self._flush_writer(job_proc, gen)
                                    job.discard()
                    self.maintain_pool()
                    sleep(next(intervals))
        finally:
            self.outbound_buffer.clear()
            self._active_writers.clear()
            self._active_writes.clear()
            self._busy_workers.clear()

    def _flush_writer(self, proc, writer):
        if False:
            while True:
                i = 10
        fds = {proc.inq._writer}
        try:
            while fds:
                if not proc._is_alive():
                    break
                (readable, writable, again) = _select(writers=fds, err=fds, timeout=0.5)
                if not again and (writable or readable):
                    try:
                        next(writer)
                    except (StopIteration, OSError, EOFError):
                        break
        finally:
            self._active_writers.discard(writer)

    def get_process_queues(self):
        if False:
            i = 10
            return i + 15
        "Get queues for a new process.\n\n        Here we'll find an unused slot, as there should always\n        be one available when we start a new process.\n        "
        return next((q for (q, owner) in self._queues.items() if owner is None))

    def on_grow(self, n):
        if False:
            print('Hello World!')
        'Grow the pool by ``n`` processes.'
        diff = max(self._processes - len(self._queues), 0)
        if diff:
            self._queues.update({self.create_process_queues(): None for _ in range(diff)})

    def on_shrink(self, n):
        if False:
            i = 10
            return i + 15
        'Shrink the pool by ``n`` processes.'

    def create_process_queues(self):
        if False:
            for i in range(10):
                print('nop')
        'Create new in, out, etc. queues, returned as a tuple.'
        inq = _SimpleQueue(wnonblock=True)
        outq = _SimpleQueue(rnonblock=True)
        synq = None
        assert isblocking(inq._reader)
        assert not isblocking(inq._writer)
        assert not isblocking(outq._reader)
        assert isblocking(outq._writer)
        if self.synack:
            synq = _SimpleQueue(wnonblock=True)
            assert isblocking(synq._reader)
            assert not isblocking(synq._writer)
        return (inq, outq, synq)

    def on_process_alive(self, pid):
        if False:
            for i in range(10):
                print('nop')
        'Called when receiving the :const:`WORKER_UP` message.\n\n        Marks the process as ready to receive work.\n        '
        try:
            proc = next((w for w in self._pool if w.pid == pid))
        except StopIteration:
            return logger.warning('process with pid=%s already exited', pid)
        assert proc.inqW_fd not in self._fileno_to_inq
        assert proc.inqW_fd not in self._all_inqueues
        self._waiting_to_start.discard(proc)
        self._fileno_to_inq[proc.inqW_fd] = proc
        self._fileno_to_synq[proc.synqW_fd] = proc
        self._all_inqueues.add(proc.inqW_fd)

    def on_job_process_down(self, job, pid_gone):
        if False:
            i = 10
            return i + 15
        'Called for each job when the process assigned to it exits.'
        if job._write_to and (not job._write_to._is_alive()):
            self.on_partial_read(job, job._write_to)
        elif job._scheduled_for and (not job._scheduled_for._is_alive()):
            self._put_back(job)

    def on_job_process_lost(self, job, pid, exitcode):
        if False:
            return 10
        "Called when the process executing job' exits.\n\n        This happens when the process job'\n        was assigned to exited by mysterious means (error exitcodes and\n        signals).\n        "
        self.mark_as_worker_lost(job, exitcode)

    def human_write_stats(self):
        if False:
            i = 10
            return i + 15
        if self.write_stats is None:
            return 'N/A'
        vals = list(self.write_stats.values())
        total = sum(vals)

        def per(v, total):
            if False:
                return 10
            return f'{(float(v) / total if v else 0):.2f}'
        return {'total': total, 'avg': per(total / len(self.write_stats) if total else 0, total), 'all': ', '.join((per(v, total) for v in vals)), 'raw': ', '.join(map(str, vals)), 'strategy': SCHED_STRATEGY_TO_NAME.get(self.sched_strategy, self.sched_strategy), 'inqueues': {'total': len(self._all_inqueues), 'active': len(self._active_writes)}}

    def _process_cleanup_queues(self, proc):
        if False:
            print('Hello World!')
        'Called to clean up queues after process exit.'
        if not proc.dead:
            try:
                self._queues[self._find_worker_queues(proc)] = None
            except (KeyError, ValueError):
                pass

    @staticmethod
    def _stop_task_handler(task_handler):
        if False:
            i = 10
            return i + 15
        "Called at shutdown to tell processes that we're shutting down."
        for proc in task_handler.pool:
            try:
                setblocking(proc.inq._writer, 1)
            except OSError:
                pass
            else:
                try:
                    proc.inq.put(None)
                except OSError as exc:
                    if exc.errno != errno.EBADF:
                        raise

    def create_result_handler(self):
        if False:
            print('Hello World!')
        return super().create_result_handler(fileno_to_outq=self._fileno_to_outq, on_process_alive=self.on_process_alive)

    def _process_register_queues(self, proc, queues):
        if False:
            i = 10
            return i + 15
        'Mark new ownership for ``queues`` to update fileno indices.'
        assert queues in self._queues
        b = len(self._queues)
        self._queues[queues] = proc
        assert b == len(self._queues)

    def _find_worker_queues(self, proc):
        if False:
            while True:
                i = 10
        'Find the queues owned by ``proc``.'
        try:
            return next((q for (q, owner) in self._queues.items() if owner == proc))
        except StopIteration:
            raise ValueError(proc)

    def _setup_queues(self):
        if False:
            i = 10
            return i + 15
        self._quick_put = None
        self._inqueue = self._outqueue = self._quick_get = self._poll_result = None

    def process_flush_queues(self, proc):
        if False:
            while True:
                i = 10
        "Flush all queues.\n\n        Including the outbound buffer, so that\n        all tasks that haven't been started will be discarded.\n\n        In Celery this is called whenever the transport connection is lost\n        (consumer restart), and when a process is terminated.\n        "
        resq = proc.outq._reader
        on_state_change = self._result_handler.on_state_change
        fds = {resq}
        while fds and (not resq.closed) and (self._state != TERMINATE):
            (readable, _, _) = _select(fds, None, fds, timeout=0.01)
            if readable:
                try:
                    task = resq.recv()
                except (OSError, EOFError) as exc:
                    _errno = getattr(exc, 'errno', None)
                    if _errno == errno.EINTR:
                        continue
                    elif _errno == errno.EAGAIN:
                        break
                    elif _errno not in UNAVAIL:
                        debug('got %r while flushing process %r', exc, proc, exc_info=1)
                    break
                else:
                    if task is None:
                        debug('got sentinel while flushing process %r', proc)
                        break
                    else:
                        on_state_change(task)
            else:
                break

    def on_partial_read(self, job, proc):
        if False:
            while True:
                i = 10
        'Called when a job was partially written to exited child.'
        if not job._accepted:
            self._put_back(job)
        writer = _get_job_writer(job)
        if writer:
            self._active_writers.discard(writer)
            del writer
        if not proc.dead:
            proc.dead = True
            before = len(self._queues)
            try:
                queues = self._find_worker_queues(proc)
                if self.destroy_queues(queues, proc):
                    self._queues[self.create_process_queues()] = None
            except ValueError:
                pass
            assert len(self._queues) == before

    def destroy_queues(self, queues, proc):
        if False:
            print('Hello World!')
        'Destroy queues that can no longer be used.\n\n        This way they can be replaced by new usable sockets.\n        '
        assert not proc._is_alive()
        self._waiting_to_start.discard(proc)
        removed = 1
        try:
            self._queues.pop(queues)
        except KeyError:
            removed = 0
        try:
            self.on_inqueue_close(queues[0]._writer.fileno(), proc)
        except OSError:
            pass
        for queue in queues:
            if queue:
                for sock in (queue._reader, queue._writer):
                    if not sock.closed:
                        self.hub_remove(sock)
                        try:
                            sock.close()
                        except OSError:
                            pass
        return removed

    def _create_payload(self, type_, args, dumps=_pickle.dumps, pack=pack, protocol=HIGHEST_PROTOCOL):
        if False:
            print('Hello World!')
        body = dumps((type_, args), protocol=protocol)
        size = len(body)
        header = pack('>I', size)
        return (header, body, size)

    @classmethod
    def _set_result_sentinel(cls, _outqueue, _pool):
        if False:
            i = 10
            return i + 15
        pass

    def _help_stuff_finish_args(self):
        if False:
            print('Hello World!')
        return (self._pool,)

    @classmethod
    def _help_stuff_finish(cls, pool):
        if False:
            i = 10
            return i + 15
        debug('removing tasks from inqueue until task handler finished')
        fileno_to_proc = {}
        inqR = set()
        for w in pool:
            try:
                fd = w.inq._reader.fileno()
                inqR.add(fd)
                fileno_to_proc[fd] = w
            except OSError:
                pass
        while inqR:
            (readable, _, again) = _select(inqR, timeout=0.5)
            if again:
                continue
            if not readable:
                break
            for fd in readable:
                fileno_to_proc[fd].inq._reader.recv()
            sleep(0)

    @property
    def timers(self):
        if False:
            i = 10
            return i + 15
        return {self.maintain_pool: 5.0}