import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from dramatiq.rate_limits import Barrier

def test_barrier(rate_limiter_backend):
    if False:
        while True:
            i = 10
    barrier = Barrier(rate_limiter_backend, 'sequential-barrier', ttl=30000)
    assert barrier.create(parties=2)
    assert not barrier.create(parties=10)
    assert not barrier.wait(block=False)
    assert barrier.wait(block=False)

def test_barriers_can_block(rate_limiter_backend):
    if False:
        for i in range(10):
            print('nop')
    barrier = Barrier(rate_limiter_backend, 'sequential-barrier', ttl=30000)
    assert barrier.create(parties=2)
    times = []

    def worker():
        if False:
            while True:
                i = 10
        time.sleep(0.1)
        assert barrier.wait(timeout=1000)
        times.append(time.monotonic())
    try:
        with ThreadPoolExecutor(max_workers=8) as e:
            for future in [e.submit(worker), e.submit(worker)]:
                future.result()
    except NotImplementedError:
        pytest.skip('Waiting is not supported under this backend.')
    assert abs(times[0] - times[1]) <= 0.01

def test_barriers_can_timeout(rate_limiter_backend):
    if False:
        for i in range(10):
            print('nop')
    barrier = Barrier(rate_limiter_backend, 'sequential-barrier', ttl=30000)
    assert barrier.create(parties=2)
    try:
        assert not barrier.wait(timeout=1000)
    except NotImplementedError:
        pytest.skip('Waiting is not supported under this backend.')