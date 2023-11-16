from __future__ import annotations
import signal
import threading
import time
import typing
from socket import socket, socketpair
from types import FrameType
import pytest
from urllib3.util.wait import _have_working_poll, poll_wait_for_socket, select_wait_for_socket, wait_for_read, wait_for_socket, wait_for_write
TYPE_SOCKET_PAIR = typing.Tuple[socket, socket]
TYPE_WAIT_FOR = typing.Callable[..., bool]

@pytest.fixture
def spair() -> typing.Generator[TYPE_SOCKET_PAIR, None, None]:
    if False:
        while True:
            i = 10
    (a, b) = socketpair()
    yield (a, b)
    a.close()
    b.close()
variants: list[TYPE_WAIT_FOR] = [wait_for_socket, select_wait_for_socket]
if _have_working_poll():
    variants.append(poll_wait_for_socket)

@pytest.mark.parametrize('wfs', variants)
def test_wait_for_socket(wfs: TYPE_WAIT_FOR, spair: TYPE_SOCKET_PAIR) -> None:
    if False:
        while True:
            i = 10
    (a, b) = spair
    with pytest.raises(RuntimeError):
        wfs(a, read=False, write=False)
    assert not wfs(a, read=True, timeout=0)
    assert wfs(a, write=True, timeout=0)
    b.send(b'x')
    assert wfs(a, read=True, timeout=0)
    assert wfs(a, read=True, timeout=10)
    assert wfs(a, read=True, timeout=None)
    a.setblocking(False)
    try:
        while True:
            a.send(b'x' * 999999)
    except OSError:
        pass
    assert not wfs(a, write=True, timeout=0)
    assert wfs(a, read=True, write=True, timeout=0)
    assert a.recv(1) == b'x'
    assert not wfs(a, read=True, write=True, timeout=0)
    b.close()
    assert wfs(a, read=True, timeout=0)
    with pytest.raises(Exception):
        wfs(b, read=True)

def test_wait_for_read_write(spair: TYPE_SOCKET_PAIR) -> None:
    if False:
        while True:
            i = 10
    (a, b) = spair
    assert not wait_for_read(a, 0)
    assert wait_for_write(a, 0)
    b.send(b'x')
    assert wait_for_read(a, 0)
    assert wait_for_write(a, 0)
    a.setblocking(False)
    try:
        while True:
            a.send(b'x' * 999999)
    except OSError:
        pass
    assert not wait_for_write(a, 0)

@pytest.mark.skipif(not hasattr(signal, 'setitimer'), reason='need setitimer() support')
@pytest.mark.parametrize('wfs', variants)
def test_eintr(wfs: TYPE_WAIT_FOR, spair: TYPE_SOCKET_PAIR) -> None:
    if False:
        i = 10
        return i + 15
    (a, b) = spair
    interrupt_count = [0]

    def handler(sig: int, frame: FrameType | None) -> typing.Any:
        if False:
            print('Hello World!')
        assert sig == signal.SIGALRM
        interrupt_count[0] += 1
    old_handler = signal.signal(signal.SIGALRM, handler)
    try:
        assert not wfs(a, read=True, timeout=0)
        start = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
            wfs(a, read=True, timeout=1)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
        end = time.monotonic()
        dur = end - start
        assert 0.9 < dur < 3
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    assert interrupt_count[0] > 0

@pytest.mark.skipif(not hasattr(signal, 'setitimer'), reason='need setitimer() support')
@pytest.mark.parametrize('wfs', variants)
def test_eintr_zero_timeout(wfs: TYPE_WAIT_FOR, spair: TYPE_SOCKET_PAIR) -> None:
    if False:
        print('Hello World!')
    (a, b) = spair
    interrupt_count = [0]

    def handler(sig: int, frame: FrameType | None) -> typing.Any:
        if False:
            while True:
                i = 10
        assert sig == signal.SIGALRM
        interrupt_count[0] += 1
    old_handler = signal.signal(signal.SIGALRM, handler)
    try:
        assert not wfs(a, read=True, timeout=0)
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.001, 0.001)
            end = time.monotonic() + 5
            for i in range(100000):
                wfs(a, read=True, timeout=0)
                if time.monotonic() >= end:
                    break
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    assert interrupt_count[0] > 0

@pytest.mark.skipif(not hasattr(signal, 'setitimer'), reason='need setitimer() support')
@pytest.mark.parametrize('wfs', variants)
def test_eintr_infinite_timeout(wfs: TYPE_WAIT_FOR, spair: TYPE_SOCKET_PAIR) -> None:
    if False:
        while True:
            i = 10
    (a, b) = spair
    interrupt_count = [0]

    def handler(sig: int, frame: FrameType | None) -> typing.Any:
        if False:
            i = 10
            return i + 15
        assert sig == signal.SIGALRM
        interrupt_count[0] += 1

    def make_a_readable_after_one_second() -> None:
        if False:
            for i in range(10):
                print('nop')
        time.sleep(1)
        b.send(b'x')
    old_handler = signal.signal(signal.SIGALRM, handler)
    try:
        assert not wfs(a, read=True, timeout=0)
        start = time.monotonic()
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.1, 0.1)
            thread = threading.Thread(target=make_a_readable_after_one_second)
            thread.start()
            wfs(a, read=True)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            thread.join()
        end = time.monotonic()
        dur = end - start
        assert 0.9 < dur < 3
    finally:
        signal.signal(signal.SIGALRM, old_handler)
    assert interrupt_count[0] > 0