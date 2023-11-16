from panda3d.core import Mutex, ReMutex
from panda3d import core
from random import random
import pytest
import sys

def test_mutex_acquire_release():
    if False:
        return 10
    m = Mutex()
    m.acquire()
    assert m.debug_is_locked()
    m.release()
    assert m.try_acquire()
    m.release()

def test_mutex_try_acquire():
    if False:
        while True:
            i = 10
    m = Mutex()
    assert m.try_acquire()
    assert m.debug_is_locked()
    m.release()

def test_mutex_with():
    if False:
        return 10
    m = Mutex()
    rc = sys.getrefcount(m)
    with m:
        assert m.debug_is_locked()
    with m:
        assert m.debug_is_locked()
    assert rc == sys.getrefcount(m)

@pytest.mark.skipif(not core.Thread.is_threading_supported(), reason='Threading support disabled')
def test_mutex_contention():
    if False:
        while True:
            i = 10
    m1 = Mutex()
    m2 = Mutex()
    m3 = Mutex()
    m4 = Mutex()

    def thread_acq_rel(m):
        if False:
            while True:
                i = 10
        for i in range(5000):
            m.acquire()
            m.release()

    def thread_nested():
        if False:
            return 10
        for i in range(5000):
            m1.acquire()
            m4.acquire()
            m4.release()
            m1.release()

    def thread_hand_over_hand():
        if False:
            i = 10
            return i + 15
        m1.acquire()
        for i in range(5000):
            m2.acquire()
            m1.release()
            m3.acquire()
            m2.release()
            m1.acquire()
            m3.release()
        m1.release()

    def thread_sleep(m):
        if False:
            while True:
                i = 10
        for i in range(250):
            m.acquire()
            core.Thread.sleep(random() * 0.003)
            m.release()
    threads = [core.PythonThread(thread_acq_rel, (m1,), '', ''), core.PythonThread(thread_acq_rel, (m2,), '', ''), core.PythonThread(thread_acq_rel, (m3,), '', ''), core.PythonThread(thread_acq_rel, (m4,), '', ''), core.PythonThread(thread_nested, (), '', ''), core.PythonThread(thread_nested, (), '', ''), core.PythonThread(thread_nested, (), '', ''), core.PythonThread(thread_hand_over_hand, (), '', ''), core.PythonThread(thread_hand_over_hand, (), '', ''), core.PythonThread(thread_sleep, (m1,), '', ''), core.PythonThread(thread_sleep, (m2,), '', ''), core.PythonThread(thread_sleep, (m3,), '', ''), core.PythonThread(thread_sleep, (m4,), '', '')]
    for thread in threads:
        thread.start(core.TP_normal, True)
    for thread in threads:
        thread.join()

def test_remutex_acquire_release():
    if False:
        return 10
    m = ReMutex()
    m.acquire()
    m.acquire()
    m.release()
    m.release()

def test_remutex_try_acquire():
    if False:
        i = 10
        return i + 15
    m = ReMutex()
    assert m.try_acquire()
    assert m.debug_is_locked()
    assert m.try_acquire()
    assert m.debug_is_locked()
    m.release()
    m.release()

def test_remutex_with():
    if False:
        for i in range(10):
            print('nop')
    m = ReMutex()
    rc = sys.getrefcount(m)
    with m:
        assert m.debug_is_locked()
        with m:
            assert m.debug_is_locked()
        assert m.debug_is_locked()
    assert rc == sys.getrefcount(m)