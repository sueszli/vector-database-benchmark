import importlib
import os
import sys
from itertools import count
from unittest.mock import Mock, patch
import pytest
from celery import concurrency
from celery.concurrency.base import BasePool, apply_target
from celery.exceptions import WorkerShutdown, WorkerTerminate

class test_BasePool:

    def test_apply_target(self):
        if False:
            i = 10
            return i + 15
        scratch = {}
        counter = count(0)

        def gen_callback(name, retval=None):
            if False:
                return 10

            def callback(*args):
                if False:
                    return 10
                scratch[name] = (next(counter), args)
                return retval
            return callback
        apply_target(gen_callback('target', 42), args=(8, 16), callback=gen_callback('callback'), accept_callback=gen_callback('accept_callback'))
        assert scratch['target'] == (1, (8, 16))
        assert scratch['callback'] == (2, (42,))
        pa1 = scratch['accept_callback']
        assert pa1[0] == 0
        assert pa1[1][0] == os.getpid()
        assert pa1[1][1]
        scratch.clear()
        apply_target(gen_callback('target', 42), args=(8, 16), callback=gen_callback('callback'), accept_callback=None)
        assert scratch == {'target': (3, (8, 16)), 'callback': (4, (42,))}

    def test_apply_target__propagate(self):
        if False:
            return 10
        target = Mock(name='target')
        target.side_effect = KeyError()
        with pytest.raises(KeyError):
            apply_target(target, propagate=(KeyError,))

    def test_apply_target__raises(self):
        if False:
            while True:
                i = 10
        target = Mock(name='target')
        target.side_effect = KeyError()
        with pytest.raises(KeyError):
            apply_target(target)

    def test_apply_target__raises_WorkerShutdown(self):
        if False:
            for i in range(10):
                print('nop')
        target = Mock(name='target')
        target.side_effect = WorkerShutdown()
        with pytest.raises(WorkerShutdown):
            apply_target(target)

    def test_apply_target__raises_WorkerTerminate(self):
        if False:
            while True:
                i = 10
        target = Mock(name='target')
        target.side_effect = WorkerTerminate()
        with pytest.raises(WorkerTerminate):
            apply_target(target)

    def test_apply_target__raises_BaseException(self):
        if False:
            for i in range(10):
                print('nop')
        target = Mock(name='target')
        callback = Mock(name='callback')
        target.side_effect = BaseException()
        apply_target(target, callback=callback)
        callback.assert_called()

    @patch('celery.concurrency.base.reraise')
    def test_apply_target__raises_BaseException_raises_else(self, reraise):
        if False:
            for i in range(10):
                print('nop')
        target = Mock(name='target')
        callback = Mock(name='callback')
        reraise.side_effect = KeyError()
        target.side_effect = BaseException()
        with pytest.raises(KeyError):
            apply_target(target, callback=callback)
        callback.assert_not_called()

    def test_does_not_debug(self):
        if False:
            i = 10
            return i + 15
        x = BasePool(10)
        x._does_debug = False
        x.apply_async(object)

    def test_num_processes(self):
        if False:
            print('Hello World!')
        assert BasePool(7).num_processes == 7

    def test_interface_on_start(self):
        if False:
            i = 10
            return i + 15
        BasePool(10).on_start()

    def test_interface_on_stop(self):
        if False:
            return 10
        BasePool(10).on_stop()

    def test_interface_on_apply(self):
        if False:
            for i in range(10):
                print('nop')
        BasePool(10).on_apply()

    def test_interface_info(self):
        if False:
            i = 10
            return i + 15
        assert BasePool(10).info == {'implementation': 'celery.concurrency.base:BasePool', 'max-concurrency': 10}

    def test_interface_flush(self):
        if False:
            print('Hello World!')
        assert BasePool(10).flush() is None

    def test_active(self):
        if False:
            return 10
        p = BasePool(10)
        assert not p.active
        p._state = p.RUN
        assert p.active

    def test_restart(self):
        if False:
            while True:
                i = 10
        p = BasePool(10)
        with pytest.raises(NotImplementedError):
            p.restart()

    def test_interface_on_terminate(self):
        if False:
            for i in range(10):
                print('nop')
        p = BasePool(10)
        p.on_terminate()

    def test_interface_terminate_job(self):
        if False:
            print('Hello World!')
        with pytest.raises(NotImplementedError):
            BasePool(10).terminate_job(101)

    def test_interface_did_start_ok(self):
        if False:
            while True:
                i = 10
        assert BasePool(10).did_start_ok()

    def test_interface_register_with_event_loop(self):
        if False:
            for i in range(10):
                print('nop')
        assert BasePool(10).register_with_event_loop(Mock()) is None

    def test_interface_on_soft_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        assert BasePool(10).on_soft_timeout(Mock()) is None

    def test_interface_on_hard_timeout(self):
        if False:
            for i in range(10):
                print('nop')
        assert BasePool(10).on_hard_timeout(Mock()) is None

    def test_interface_close(self):
        if False:
            while True:
                i = 10
        p = BasePool(10)
        p.on_close = Mock()
        p.close()
        assert p._state == p.CLOSE
        p.on_close.assert_called_with()

    def test_interface_no_close(self):
        if False:
            print('Hello World!')
        assert BasePool(10).on_close() is None

class test_get_available_pool_names:

    def test_no_concurrent_futures__returns_no_threads_pool_name(self):
        if False:
            return 10
        expected_pool_names = ('prefork', 'eventlet', 'gevent', 'solo', 'processes', 'custom')
        with patch.dict(sys.modules, {'concurrent.futures': None}):
            importlib.reload(concurrency)
            assert concurrency.get_available_pool_names() == expected_pool_names

    def test_concurrent_futures__returns_threads_pool_name(self):
        if False:
            return 10
        expected_pool_names = ('prefork', 'eventlet', 'gevent', 'solo', 'processes', 'threads', 'custom')
        with patch.dict(sys.modules, {'concurrent.futures': Mock()}):
            importlib.reload(concurrency)
            assert concurrency.get_available_pool_names() == expected_pool_names