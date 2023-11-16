import sys
from unittest.mock import Mock, patch
import pytest
pytest.importorskip('eventlet')
from greenlet import GreenletExit
import t.skip
from celery.concurrency.eventlet import TaskPool, Timer, apply_target
eventlet_modules = ('eventlet', 'eventlet.debug', 'eventlet.greenthread', 'eventlet.greenpool', 'greenlet')

@t.skip.if_pypy
class EventletCase:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.patching.modules(*eventlet_modules)

    def teardown_method(self):
        if False:
            print('Hello World!')
        for mod in [mod for mod in sys.modules if mod.startswith('eventlet')]:
            try:
                del sys.modules[mod]
            except KeyError:
                pass

class test_aaa_eventlet_patch(EventletCase):

    def test_aaa_is_patched(self):
        if False:
            i = 10
            return i + 15
        with patch('eventlet.monkey_patch', create=True) as monkey_patch:
            from celery import maybe_patch_concurrency
            maybe_patch_concurrency(['x', '-P', 'eventlet'])
            monkey_patch.assert_called_with()

    @patch('eventlet.debug.hub_blocking_detection', create=True)
    @patch('eventlet.monkey_patch', create=True)
    def test_aaa_blockdetecet(self, monkey_patch, hub_blocking_detection, patching):
        if False:
            while True:
                i = 10
        patching.setenv('EVENTLET_NOBLOCK', '10.3')
        from celery import maybe_patch_concurrency
        maybe_patch_concurrency(['x', '-P', 'eventlet'])
        monkey_patch.assert_called_with()
        hub_blocking_detection.assert_called_with(10.3, 10.3)

class test_Timer(EventletCase):

    @pytest.fixture(autouse=True)
    def setup_patches(self, patching):
        if False:
            for i in range(10):
                print('nop')
        self.spawn_after = patching('eventlet.greenthread.spawn_after')
        self.GreenletExit = patching('greenlet.GreenletExit')

    def test_sched(self):
        if False:
            i = 10
            return i + 15
        x = Timer()
        x.GreenletExit = KeyError
        entry = Mock()
        g = x._enter(1, 0, entry)
        assert x.queue
        x._entry_exit(g, entry)
        g.wait.side_effect = KeyError()
        x._entry_exit(g, entry)
        entry.cancel.assert_called_with()
        assert not x._queue
        x._queue.add(g)
        x.clear()
        x._queue.add(g)
        g.cancel.side_effect = KeyError()
        x.clear()

    def test_cancel(self):
        if False:
            return 10
        x = Timer()
        tref = Mock(name='tref')
        x.cancel(tref)
        tref.cancel.assert_called_with()
        x.GreenletExit = KeyError
        tref.cancel.side_effect = KeyError()
        x.cancel(tref)

class test_TaskPool(EventletCase):

    @pytest.fixture(autouse=True)
    def setup_patches(self, patching):
        if False:
            for i in range(10):
                print('nop')
        self.GreenPool = patching('eventlet.greenpool.GreenPool')
        self.greenthread = patching('eventlet.greenthread')

    def test_pool(self):
        if False:
            for i in range(10):
                print('nop')
        x = TaskPool()
        x.on_start()
        x.on_stop()
        x.on_apply(Mock())
        x._pool = None
        x.on_stop()
        assert len(x._pool_map.keys()) == 1
        assert x.getpid()

    @patch('celery.concurrency.eventlet.base')
    def test_apply_target(self, base):
        if False:
            i = 10
            return i + 15
        apply_target(Mock(), getpid=Mock())
        base.apply_target.assert_called()

    def test_grow(self):
        if False:
            for i in range(10):
                print('nop')
        x = TaskPool(10)
        x._pool = Mock(name='_pool')
        x.grow(2)
        assert x.limit == 12
        x._pool.resize.assert_called_with(12)

    def test_shrink(self):
        if False:
            for i in range(10):
                print('nop')
        x = TaskPool(10)
        x._pool = Mock(name='_pool')
        x.shrink(2)
        assert x.limit == 8
        x._pool.resize.assert_called_with(8)

    def test_get_info(self):
        if False:
            while True:
                i = 10
        x = TaskPool(10)
        x._pool = Mock(name='_pool')
        assert x._get_info() == {'implementation': 'celery.concurrency.eventlet:TaskPool', 'max-concurrency': 10, 'free-threads': x._pool.free(), 'running-threads': x._pool.running()}

    def test_terminate_job(self):
        if False:
            return 10
        func = Mock()
        pool = TaskPool(10)
        pool.on_start()
        pool.on_apply(func)
        assert len(pool._pool_map.keys()) == 1
        pid = list(pool._pool_map.keys())[0]
        greenlet = pool._pool_map[pid]
        pool.terminate_job(pid)
        greenlet.link.assert_called_once()
        greenlet.kill.assert_called_once()

    def test_make_killable_target(self):
        if False:
            print('Hello World!')

        def valid_target():
            if False:
                for i in range(10):
                    print('nop')
            return 'some result...'

        def terminating_target():
            if False:
                print('Hello World!')
            raise GreenletExit()
        assert TaskPool._make_killable_target(valid_target)() == 'some result...'
        assert TaskPool._make_killable_target(terminating_target)() == (False, None, None)

    def test_cleanup_after_job_finish(self):
        if False:
            i = 10
            return i + 15
        testMap = {'1': None}
        TaskPool._cleanup_after_job_finish(None, testMap, '1')
        assert len(testMap) == 0