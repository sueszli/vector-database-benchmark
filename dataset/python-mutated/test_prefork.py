import errno
import os
import socket
from itertools import cycle
from unittest.mock import Mock, patch
import pytest
import t.skip
from celery.app.defaults import DEFAULTS
from celery.concurrency.asynpool import iterate_file_descriptors_safely
from celery.utils.collections import AttributeDict
from celery.utils.functional import noop
from celery.utils.objects import Bunch
try:
    from celery.concurrency import asynpool
    from celery.concurrency import prefork as mp
except ImportError:

    class _mp:
        RUN = 1

        class TaskPool:
            _pool = Mock()

            def __init__(self, *args, **kwargs):
                if False:
                    return 10
                pass

            def start(self):
                if False:
                    return 10
                pass

            def stop(self):
                if False:
                    return 10
                pass

            def apply_async(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                pass
    mp = _mp()
    asynpool = None

class MockResult:

    def __init__(self, value, pid):
        if False:
            for i in range(10):
                print('nop')
        self.value = value
        self.pid = pid

    def worker_pids(self):
        if False:
            print('Hello World!')
        return [self.pid]

    def get(self):
        if False:
            print('Hello World!')
        return self.value

@patch('celery.platforms.set_mp_process_title')
class test_process_initializer:

    @staticmethod
    def Loader(*args, **kwargs):
        if False:
            while True:
                i = 10
        loader = Mock(*args, **kwargs)
        loader.conf = {}
        loader.override_backends = {}
        return loader

    @patch('celery.platforms.signals')
    def test_process_initializer(self, _signals, set_mp_process_title, restore_logging):
        if False:
            print('Hello World!')
        from celery import signals
        from celery._state import _tls
        from celery.concurrency.prefork import WORKER_SIGIGNORE, WORKER_SIGRESET, process_initializer
        on_worker_process_init = Mock()
        signals.worker_process_init.connect(on_worker_process_init)
        with self.Celery(loader=self.Loader) as app:
            app.conf = AttributeDict(DEFAULTS)
            process_initializer(app, 'awesome.worker.com')
            _signals.ignore.assert_any_call(*WORKER_SIGIGNORE)
            _signals.reset.assert_any_call(*WORKER_SIGRESET)
            assert app.loader.init_worker.call_count
            on_worker_process_init.assert_called()
            assert _tls.current_app is app
            set_mp_process_title.assert_called_with('celeryd', hostname='awesome.worker.com')
            with patch('celery.app.trace.setup_worker_optimizations') as S:
                os.environ['FORKED_BY_MULTIPROCESSING'] = '1'
                try:
                    process_initializer(app, 'luke.worker.com')
                    S.assert_called_with(app, 'luke.worker.com')
                finally:
                    os.environ.pop('FORKED_BY_MULTIPROCESSING', None)
            os.environ['CELERY_LOG_FILE'] = 'worker%I.log'
            app.log.setup = Mock(name='log_setup')
            try:
                process_initializer(app, 'luke.worker.com')
            finally:
                os.environ.pop('CELERY_LOG_FILE', None)

    @patch('celery.platforms.set_pdeathsig')
    def test_pdeath_sig(self, _set_pdeathsig, set_mp_process_title, restore_logging):
        if False:
            print('Hello World!')
        from celery import signals
        on_worker_process_init = Mock()
        signals.worker_process_init.connect(on_worker_process_init)
        from celery.concurrency.prefork import process_initializer
        with self.Celery(loader=self.Loader) as app:
            app.conf = AttributeDict(DEFAULTS)
            process_initializer(app, 'awesome.worker.com')
        _set_pdeathsig.assert_called_once_with('SIGKILL')

class test_process_destructor:

    @patch('celery.concurrency.prefork.signals')
    def test_process_destructor(self, signals):
        if False:
            print('Hello World!')
        mp.process_destructor(13, -3)
        signals.worker_process_shutdown.send.assert_called_with(sender=None, pid=13, exitcode=-3)

class MockPool:
    started = False
    closed = False
    joined = False
    terminated = False
    _state = None

    def __init__(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        self.started = True
        self._timeout_handler = Mock()
        self._result_handler = Mock()
        self.maintain_pool = Mock()
        self._state = mp.RUN
        self._processes = kwargs.get('processes')
        self._proc_alive_timeout = kwargs.get('proc_alive_timeout')
        self._pool = [Bunch(pid=i, inqW_fd=1, outqR_fd=2) for i in range(self._processes)]
        self._current_proc = cycle(range(self._processes))

    def close(self):
        if False:
            return 10
        self.closed = True
        self._state = 'CLOSE'

    def join(self):
        if False:
            while True:
                i = 10
        self.joined = True

    def terminate(self):
        if False:
            return 10
        self.terminated = True

    def terminate_job(self, *args, **kwargs):
        if False:
            return 10
        pass

    def restart(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def handle_result_event(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        pass

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def grow(self, n=1):
        if False:
            i = 10
            return i + 15
        self._processes += n

    def shrink(self, n=1):
        if False:
            print('Hello World!')
        self._processes -= n

    def apply_async(self, *args, **kwargs):
        if False:
            print('Hello World!')
        pass

    def register_with_event_loop(self, loop):
        if False:
            while True:
                i = 10
        pass

class ExeMockPool(MockPool):

    def apply_async(self, target, args=(), kwargs={}, callback=noop):
        if False:
            while True:
                i = 10
        from threading import Timer
        res = target(*args, **kwargs)
        Timer(0.1, callback, (res,)).start()
        return MockResult(res, next(self._current_proc))

class TaskPool(mp.TaskPool):
    Pool = BlockingPool = MockPool

class ExeMockTaskPool(mp.TaskPool):
    Pool = BlockingPool = ExeMockPool

@t.skip.if_win32
class test_AsynPool:

    def setup_method(self):
        if False:
            print('Hello World!')
        pytest.importorskip('multiprocessing')

    def test_gen_not_started(self):
        if False:
            while True:
                i = 10

        def gen():
            if False:
                print('Hello World!')
            yield 1
            assert not asynpool.gen_not_started(g)
            yield 2
        g = gen()
        assert asynpool.gen_not_started(g)
        next(g)
        assert not asynpool.gen_not_started(g)
        list(g)
        assert not asynpool.gen_not_started(g)

        def gen2():
            if False:
                for i in range(10):
                    print('nop')
            yield 1
            raise RuntimeError('generator error')
        g = gen2()
        assert asynpool.gen_not_started(g)
        next(g)
        assert not asynpool.gen_not_started(g)
        with pytest.raises(RuntimeError):
            next(g)
        assert not asynpool.gen_not_started(g)

    @patch('select.select', create=True)
    def test_select(self, __select):
        if False:
            while True:
                i = 10
        ebadf = socket.error()
        ebadf.errno = errno.EBADF
        with patch('select.poll', create=True) as poller:
            poll = poller.return_value = Mock(name='poll.poll')
            poll.return_value = ({3}, set(), 0)
            assert asynpool._select({3}, poll=poll) == ({3}, set(), 0)
            poll.return_value = ({3}, set(), 0)
            assert asynpool._select({3}, None, {3}, poll=poll) == ({3}, set(), 0)
            eintr = socket.error()
            eintr.errno = errno.EINTR
            poll.side_effect = eintr
            readers = {3}
            assert asynpool._select(readers, poll=poll) == (set(), set(), 1)
            assert 3 in readers
        with patch('select.poll', create=True) as poller:
            poll = poller.return_value = Mock(name='poll.poll')
            poll.side_effect = ebadf
            with patch('select.select') as selcheck:
                selcheck.side_effect = ebadf
                readers = {3}
                assert asynpool._select(readers, poll=poll) == (set(), set(), 1)
                assert 3 not in readers
        with patch('select.poll', create=True) as poller:
            poll = poller.return_value = Mock(name='poll.poll')
            poll.side_effect = MemoryError()
            with pytest.raises(MemoryError):
                asynpool._select({1}, poll=poll)
        with patch('select.poll', create=True) as poller:
            poll = poller.return_value = Mock(name='poll.poll')
            with patch('select.select') as selcheck:

                def se(*args):
                    if False:
                        while True:
                            i = 10
                    selcheck.side_effect = MemoryError()
                    raise ebadf
                poll.side_effect = se
                with pytest.raises(MemoryError):
                    asynpool._select({3}, poll=poll)
        with patch('select.poll', create=True) as poller:
            poll = poller.return_value = Mock(name='poll.poll')
            with patch('select.select') as selcheck:

                def se2(*args):
                    if False:
                        return 10
                    selcheck.side_effect = socket.error()
                    selcheck.side_effect.errno = 1321
                    raise ebadf
                poll.side_effect = se2
                with pytest.raises(socket.error):
                    asynpool._select({3}, poll=poll)
        with patch('select.poll', create=True) as poller:
            poll = poller.return_value = Mock(name='poll.poll')
            poll.side_effect = socket.error()
            poll.side_effect.errno = 34134
            with pytest.raises(socket.error):
                asynpool._select({3}, poll=poll)

    def test_promise(self):
        if False:
            print('Hello World!')
        fun = Mock()
        x = asynpool.promise(fun, (1,), {'foo': 1})
        x()
        assert x.ready
        fun.assert_called_with(1, foo=1)

    def test_Worker(self):
        if False:
            for i in range(10):
                print('nop')
        w = asynpool.Worker(Mock(), Mock())
        w.on_loop_start(1234)
        w.outq.put.assert_called_with((asynpool.WORKER_UP, (1234,)))

    def test_iterate_file_descriptors_safely_source_data_list(self):
        if False:
            while True:
                i = 10
        fd_iter = [1, 2, 3, 4, 5]

        def _fake_hub(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            raise OSError
        iterate_file_descriptors_safely(fd_iter, fd_iter, _fake_hub, 'arg1', 'arg2', kw1='kw1', kw2='kw2')
        assert fd_iter == [], 'Expected all items removed from managed list'

    def test_iterate_file_descriptors_safely_source_data_set(self):
        if False:
            return 10
        fd_iter = {1, 2, 3, 4, 5}

        def _fake_hub(*args, **kwargs):
            if False:
                print('Hello World!')
            raise OSError
        iterate_file_descriptors_safely(fd_iter, fd_iter, _fake_hub, 'arg1', 'arg2', kw1='kw1', kw2='kw2')
        assert fd_iter == set(), 'Expected all items removed from managed set'

    def test_iterate_file_descriptors_safely_source_data_dict(self):
        if False:
            while True:
                i = 10
        fd_iter = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5}

        def _fake_hub(*args, **kwargs):
            if False:
                return 10
            raise OSError
        iterate_file_descriptors_safely(fd_iter, fd_iter, _fake_hub, 'arg1', 'arg2', kw1='kw1', kw2='kw2')
        assert fd_iter == {}, 'Expected all items removed from managed dict'

    def test_register_with_event_loop__no_on_tick_dupes(self):
        if False:
            return 10
        "Ensure AsynPool's register_with_event_loop only registers\n        on_poll_start in the event loop the first time it's called. This\n        prevents a leak when the Consumer is restarted.\n        "
        pool = asynpool.AsynPool(threads=False)
        hub = Mock(name='hub')
        pool.register_with_event_loop(hub)
        pool.register_with_event_loop(hub)
        hub.on_tick.add.assert_called_once()

    @patch('billiard.pool.Pool._create_worker_process')
    def test_before_create_process_signal(self, create_process):
        if False:
            i = 10
            return i + 15
        from celery import signals
        on_worker_before_create_process = Mock()
        signals.worker_before_create_process.connect(on_worker_before_create_process)
        pool = asynpool.AsynPool(processes=1, threads=False)
        create_process.assert_called_once_with(0)
        on_worker_before_create_process.assert_any_call(signal=signals.worker_before_create_process, sender=pool)

@t.skip.if_win32
class test_ResultHandler:

    def setup_method(self):
        if False:
            while True:
                i = 10
        pytest.importorskip('multiprocessing')

    def test_process_result(self):
        if False:
            print('Hello World!')
        x = asynpool.ResultHandler(Mock(), Mock(), {}, Mock(), Mock(), Mock(), Mock(), Mock(), fileno_to_outq={}, on_process_alive=Mock(), on_job_ready=Mock())
        assert x
        hub = Mock(name='hub')
        recv = x._recv_message = Mock(name='recv_message')
        recv.return_value = iter([])
        x.on_state_change = Mock()
        x.register_with_event_loop(hub)
        proc = x.fileno_to_outq[3] = Mock()
        reader = proc.outq._reader
        reader.poll.return_value = False
        x.handle_event(6)
        x.handle_event(3)
        x._recv_message.assert_called_with(hub.add_reader, 3, x.on_state_change)

class test_TaskPool:

    def test_start(self):
        if False:
            return 10
        pool = TaskPool(10)
        pool.start()
        assert pool._pool.started
        assert pool._pool._state == asynpool.RUN
        _pool = pool._pool
        pool.stop()
        assert _pool.closed
        assert _pool.joined
        pool.stop()
        pool.start()
        _pool = pool._pool
        pool.terminate()
        pool.terminate()
        assert _pool.terminated

    def test_restart(self):
        if False:
            return 10
        pool = TaskPool(10)
        pool._pool = Mock(name='pool')
        pool.restart()
        pool._pool.restart.assert_called_with()
        pool._pool.apply_async.assert_called_with(mp.noop)

    def test_did_start_ok(self):
        if False:
            while True:
                i = 10
        pool = TaskPool(10)
        pool._pool = Mock(name='pool')
        assert pool.did_start_ok() is pool._pool.did_start_ok()

    def test_register_with_event_loop(self):
        if False:
            i = 10
            return i + 15
        pool = TaskPool(10)
        pool._pool = Mock(name='pool')
        loop = Mock(name='loop')
        pool.register_with_event_loop(loop)
        pool._pool.register_with_event_loop.assert_called_with(loop)

    def test_on_close(self):
        if False:
            print('Hello World!')
        pool = TaskPool(10)
        pool._pool = Mock(name='pool')
        pool._pool._state = mp.RUN
        pool.on_close()
        pool._pool.close.assert_called_with()

    def test_on_close__pool_not_running(self):
        if False:
            print('Hello World!')
        pool = TaskPool(10)
        pool._pool = Mock(name='pool')
        pool._pool._state = mp.CLOSE
        pool.on_close()
        pool._pool.close.assert_not_called()

    def test_apply_async(self):
        if False:
            while True:
                i = 10
        pool = TaskPool(10)
        pool.start()
        pool.apply_async(lambda x: x, (2,), {})

    def test_grow_shrink(self):
        if False:
            for i in range(10):
                print('nop')
        pool = TaskPool(10)
        pool.start()
        assert pool._pool._processes == 10
        pool.grow()
        assert pool._pool._processes == 11
        pool.shrink(2)
        assert pool._pool._processes == 9

    def test_info(self):
        if False:
            while True:
                i = 10
        pool = TaskPool(10)
        procs = [Bunch(pid=i) for i in range(pool.limit)]

        class _Pool:
            _pool = procs
            _maxtasksperchild = None
            timeout = 10
            soft_timeout = 5

            def human_write_stats(self, *args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return {}
        pool._pool = _Pool()
        info = pool.info
        assert info['max-concurrency'] == pool.limit
        assert info['max-tasks-per-child'] == 'N/A'
        assert info['timeouts'] == (5, 10)

    def test_num_processes(self):
        if False:
            for i in range(10):
                print('nop')
        pool = TaskPool(7)
        pool.start()
        assert pool.num_processes == 7

    @patch('billiard.forking_enable')
    def test_on_start_proc_alive_timeout_default(self, __forking_enable):
        if False:
            print('Hello World!')
        app = Mock(conf=AttributeDict(DEFAULTS))
        pool = TaskPool(4, app=app)
        pool.on_start()
        assert pool._pool._proc_alive_timeout == 4.0

    @patch('billiard.forking_enable')
    def test_on_start_proc_alive_timeout_custom(self, __forking_enable):
        if False:
            for i in range(10):
                print('nop')
        app = Mock(conf=AttributeDict(DEFAULTS))
        app.conf.worker_proc_alive_timeout = 8.0
        pool = TaskPool(4, app=app)
        pool.on_start()
        assert pool._pool._proc_alive_timeout == 8.0