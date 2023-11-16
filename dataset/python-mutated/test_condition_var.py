from panda3d.core import Mutex, ConditionVar
from panda3d import core
from direct.stdpy import thread
import pytest

def yield_thread():
    if False:
        return 10
    core.Thread.sleep(0.002)

def test_cvar_notify():
    if False:
        for i in range(10):
            print('nop')
    m = Mutex()
    cv = ConditionVar(m)
    cv.notify()
    cv.notify_all()
    del cv

def test_cvar_notify_locked():
    if False:
        for i in range(10):
            print('nop')
    m = Mutex()
    cv = ConditionVar(m)
    with m:
        cv.notify()
    with m:
        cv.notify_all()
    del cv

@pytest.mark.parametrize('num_threads', [1, 2, 3, 4])
@pytest.mark.skipif(not core.Thread.is_threading_supported(), reason='Threading support disabled')
def test_cvar_notify_thread(num_threads):
    if False:
        print('Hello World!')
    m = Mutex()
    cv = ConditionVar(m)
    m.acquire()
    cv.notify()
    state = {'waiting': 0}

    def wait_thread():
        if False:
            return 10
        m.acquire()
        state['waiting'] += 1
        cv.wait()
        state['waiting'] -= 1
        m.release()
    threads = []
    for i in range(num_threads):
        thread = core.PythonThread(wait_thread, (), '', '')
        thread.start(core.TP_high, True)
    for i in range(1000):
        m.release()
        yield_thread()
        m.acquire()
        if state['waiting'] == num_threads:
            break
    assert state['waiting'] == num_threads
    for i in range(num_threads):
        cv.notify()
        expected_waiters = num_threads - i - 1
        for j in range(1000):
            m.release()
            yield_thread()
            m.acquire()
            if state['waiting'] == expected_waiters:
                break
        assert state['waiting'] == expected_waiters
    m.release()
    for thread in threads:
        thread.join()
    cv = None

@pytest.mark.parametrize('num_threads', [1, 2, 3, 4])
@pytest.mark.skipif(not core.Thread.is_threading_supported(), reason='Threading support disabled')
def test_cvar_notify_all_threads(num_threads):
    if False:
        while True:
            i = 10
    m = Mutex()
    cv = ConditionVar(m)
    m.acquire()
    cv.notify_all()
    state = {'waiting': 0}

    def wait_thread():
        if False:
            return 10
        m.acquire()
        state['waiting'] += 1
        cv.wait()
        state['waiting'] -= 1
        m.release()
    threads = []
    for i in range(num_threads):
        thread = core.PythonThread(wait_thread, (), '', '')
        thread.start(core.TP_high, True)
    for i in range(1000):
        m.release()
        yield_thread()
        m.acquire()
        if state['waiting'] == num_threads:
            break
    assert state['waiting'] == num_threads
    cv.notify_all()
    for i in range(1000):
        m.release()
        yield_thread()
        m.acquire()
        if state['waiting'] == 0:
            break
    assert state['waiting'] == 0
    m.release()
    for thread in threads:
        thread.join()
    cv = None