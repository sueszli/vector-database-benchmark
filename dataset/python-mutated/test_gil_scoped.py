import multiprocessing
import sys
import threading
import time
import pytest
import env
from pybind11_tests import gil_scoped as m

class ExtendedVirtClass(m.VirtClass):

    def virtual_func(self):
        if False:
            while True:
                i = 10
        pass

    def pure_virtual_func(self):
        if False:
            i = 10
            return i + 15
        pass

def test_callback_py_obj():
    if False:
        while True:
            i = 10
    m.test_callback_py_obj(lambda : None)

def test_callback_std_func():
    if False:
        i = 10
        return i + 15
    m.test_callback_std_func(lambda : None)

def test_callback_virtual_func():
    if False:
        i = 10
        return i + 15
    extended = ExtendedVirtClass()
    m.test_callback_virtual_func(extended)

def test_callback_pure_virtual_func():
    if False:
        while True:
            i = 10
    extended = ExtendedVirtClass()
    m.test_callback_pure_virtual_func(extended)

def test_cross_module_gil_released():
    if False:
        print('Hello World!')
    'Makes sure that the GIL can be acquired by another module from a GIL-released state.'
    m.test_cross_module_gil_released()

def test_cross_module_gil_acquired():
    if False:
        while True:
            i = 10
    'Makes sure that the GIL can be acquired by another module from a GIL-acquired state.'
    m.test_cross_module_gil_acquired()

def test_cross_module_gil_inner_custom_released():
    if False:
        for i in range(10):
            print('nop')
    'Makes sure that the GIL can be acquired/released by another module\n    from a GIL-released state using custom locking logic.'
    m.test_cross_module_gil_inner_custom_released()

def test_cross_module_gil_inner_custom_acquired():
    if False:
        for i in range(10):
            print('nop')
    'Makes sure that the GIL can be acquired/acquired by another module\n    from a GIL-acquired state using custom locking logic.'
    m.test_cross_module_gil_inner_custom_acquired()

def test_cross_module_gil_inner_pybind11_released():
    if False:
        i = 10
        return i + 15
    'Makes sure that the GIL can be acquired/released by another module\n    from a GIL-released state using pybind11 locking logic.'
    m.test_cross_module_gil_inner_pybind11_released()

def test_cross_module_gil_inner_pybind11_acquired():
    if False:
        i = 10
        return i + 15
    'Makes sure that the GIL can be acquired/acquired by another module\n    from a GIL-acquired state using pybind11 locking logic.'
    m.test_cross_module_gil_inner_pybind11_acquired()

def test_cross_module_gil_nested_custom_released():
    if False:
        for i in range(10):
            print('nop')
    'Makes sure that the GIL can be nested acquired/released by another module\n    from a GIL-released state using custom locking logic.'
    m.test_cross_module_gil_nested_custom_released()

def test_cross_module_gil_nested_custom_acquired():
    if False:
        while True:
            i = 10
    'Makes sure that the GIL can be nested acquired/acquired by another module\n    from a GIL-acquired state using custom locking logic.'
    m.test_cross_module_gil_nested_custom_acquired()

def test_cross_module_gil_nested_pybind11_released():
    if False:
        while True:
            i = 10
    'Makes sure that the GIL can be nested acquired/released by another module\n    from a GIL-released state using pybind11 locking logic.'
    m.test_cross_module_gil_nested_pybind11_released()

def test_cross_module_gil_nested_pybind11_acquired():
    if False:
        return 10
    'Makes sure that the GIL can be nested acquired/acquired by another module\n    from a GIL-acquired state using pybind11 locking logic.'
    m.test_cross_module_gil_nested_pybind11_acquired()

def test_release_acquire():
    if False:
        for i in range(10):
            print('nop')
    assert m.test_release_acquire(171) == '171'

def test_nested_acquire():
    if False:
        while True:
            i = 10
    assert m.test_nested_acquire(171) == '171'

def test_multi_acquire_release_cross_module():
    if False:
        for i in range(10):
            print('nop')
    for bits in range(16 * 8):
        internals_ids = m.test_multi_acquire_release_cross_module(bits)
        assert len(internals_ids) == 2 if bits % 8 else 1
VARS_BEFORE_ALL_BASIC_TESTS = dict(vars())
ALL_BASIC_TESTS = (test_callback_py_obj, test_callback_std_func, test_callback_virtual_func, test_callback_pure_virtual_func, test_cross_module_gil_released, test_cross_module_gil_acquired, test_cross_module_gil_inner_custom_released, test_cross_module_gil_inner_custom_acquired, test_cross_module_gil_inner_pybind11_released, test_cross_module_gil_inner_pybind11_acquired, test_cross_module_gil_nested_custom_released, test_cross_module_gil_nested_custom_acquired, test_cross_module_gil_nested_pybind11_released, test_cross_module_gil_nested_pybind11_acquired, test_release_acquire, test_nested_acquire, test_multi_acquire_release_cross_module)

def test_all_basic_tests_completeness():
    if False:
        for i in range(10):
            print('nop')
    num_found = 0
    for (key, value) in VARS_BEFORE_ALL_BASIC_TESTS.items():
        if not key.startswith('test_'):
            continue
        assert value in ALL_BASIC_TESTS
        num_found += 1
    assert len(ALL_BASIC_TESTS) == num_found

def _intentional_deadlock():
    if False:
        for i in range(10):
            print('nop')
    m.intentional_deadlock()
ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK = ALL_BASIC_TESTS + (_intentional_deadlock,)

def _run_in_process(target, *args, **kwargs):
    if False:
        print('Hello World!')
    test_fn = target if len(args) == 0 else args[0]
    timeout = 0.1 if test_fn is _intentional_deadlock else 10
    process = multiprocessing.Process(target=target, args=args, kwargs=kwargs)
    process.daemon = True
    try:
        t_start = time.time()
        process.start()
        if timeout >= 100:
            print('\nprocess.pid STARTED', process.pid, (sys.argv, target, args, kwargs))
            print(f'COPY-PASTE-THIS: gdb {sys.argv[0]} -p {process.pid}', flush=True)
        process.join(timeout=timeout)
        if timeout >= 100:
            print('\nprocess.pid JOINED', process.pid, flush=True)
        t_delta = time.time() - t_start
        if process.exitcode == 66 and m.defined_THREAD_SANITIZER:
            pytest.skip('ThreadSanitizer: starting new threads after multi-threaded fork is not supported.')
        elif test_fn is _intentional_deadlock:
            assert process.exitcode is None
            return 0
        if process.exitcode is None:
            assert t_delta > 0.9 * timeout
            msg = 'DEADLOCK, most likely, exactly what this test is meant to detect.'
            if env.PYPY and env.WIN:
                pytest.skip(msg)
            raise RuntimeError(msg)
        return process.exitcode
    finally:
        if process.is_alive():
            process.terminate()

def _run_in_threads(test_fn, num_threads, parallel):
    if False:
        while True:
            i = 10
    threads = []
    for _ in range(num_threads):
        thread = threading.Thread(target=test_fn)
        thread.daemon = True
        thread.start()
        if parallel:
            threads.append(thread)
        else:
            thread.join()
    for thread in threads:
        thread.join()

@pytest.mark.parametrize('test_fn', ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_one_thread(test_fn):
    if False:
        i = 10
        return i + 15
    'Makes sure there is no GIL deadlock when running in a thread.\n\n    It runs in a separate process to be able to stop and assert if it deadlocks.\n    '
    assert _run_in_process(_run_in_threads, test_fn, num_threads=1, parallel=False) == 0

@pytest.mark.parametrize('test_fn', ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_multiple_threads_parallel(test_fn):
    if False:
        return 10
    'Makes sure there is no GIL deadlock when running in a thread multiple times in parallel.\n\n    It runs in a separate process to be able to stop and assert if it deadlocks.\n    '
    assert _run_in_process(_run_in_threads, test_fn, num_threads=8, parallel=True) == 0

@pytest.mark.parametrize('test_fn', ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_multiple_threads_sequential(test_fn):
    if False:
        i = 10
        return i + 15
    'Makes sure there is no GIL deadlock when running in a thread multiple times sequentially.\n\n    It runs in a separate process to be able to stop and assert if it deadlocks.\n    '
    assert _run_in_process(_run_in_threads, test_fn, num_threads=8, parallel=False) == 0

@pytest.mark.parametrize('test_fn', ALL_BASIC_TESTS_PLUS_INTENTIONAL_DEADLOCK)
def test_run_in_process_direct(test_fn):
    if False:
        print('Hello World!')
    'Makes sure there is no GIL deadlock when using processes.\n\n    This test is for completion, but it was never an issue.\n    '
    assert _run_in_process(test_fn) == 0