from time import sleep
import pytest
from ding.framework import EventLoop
from threading import Lock

@pytest.mark.unittest
def test_event_loop():
    if False:
        i = 10
        return i + 15
    loop = EventLoop.get_event_loop('test')
    try:
        counter = 0
        lock = Lock()

        def callback(n, lock):
            if False:
                i = 10
                return i + 15
            nonlocal counter
            with lock:
                counter += n
        loop.on('count', callback)
        for i in range(5):
            loop.emit('count', i, lock)
        sleep(0.1)
        assert counter == 10
        loop.off('count')
        loop.emit('count', 10, lock)
        sleep(0.1)
        assert counter == 10
        counter = 0
        loop.once('count', callback)
        loop.once('count', callback)
        loop.emit('count', 10, lock)
        sleep(0.1)
        assert counter == 20
        loop.emit('count', 10, lock)
        assert counter == 20

        def except_callback():
            if False:
                print('Hello World!')
            raise Exception('error')
        loop.on('error', except_callback)
        loop.emit('error')
        sleep(0.1)
        assert loop._exception is not None
        with pytest.raises(Exception):
            loop.emit('error')
    finally:
        loop.stop()