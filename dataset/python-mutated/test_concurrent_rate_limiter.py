import time
from concurrent.futures import ThreadPoolExecutor
from dramatiq.rate_limits import ConcurrentRateLimiter, RateLimitExceeded

def test_concurrent_rate_limiter_releases_the_lock_after_each_call(rate_limiter_backend):
    if False:
        for i in range(10):
            print('nop')
    mutex = ConcurrentRateLimiter(rate_limiter_backend, 'sequential-test', limit=1)
    calls = 0
    for _ in range(8):
        with mutex.acquire(raise_on_failure=False) as acquired:
            if not acquired:
                continue
            calls += 1
    assert calls == 8

def test_concurrent_rate_limiter_can_act_as_a_mutex(rate_limiter_backend):
    if False:
        for i in range(10):
            print('nop')
    mutex = ConcurrentRateLimiter(rate_limiter_backend, 'concurrent-test', limit=1)
    calls = []

    def work():
        if False:
            for i in range(10):
                print('nop')
        with mutex.acquire(raise_on_failure=False) as acquired:
            if not acquired:
                return
            calls.append(1)
            time.sleep(0.3)
    with ThreadPoolExecutor(max_workers=8) as e:
        futures = []
        for _ in range(8):
            futures.append(e.submit(work))
        for future in futures:
            future.result()
    assert sum(calls) == 1

def test_concurrent_rate_limiter_limits_concurrency(rate_limiter_backend):
    if False:
        for i in range(10):
            print('nop')
    mutex = ConcurrentRateLimiter(rate_limiter_backend, 'concurrent-test', limit=4)
    calls = []

    def work():
        if False:
            print('Hello World!')
        try:
            with mutex.acquire():
                calls.append(1)
                time.sleep(0.3)
        except RateLimitExceeded:
            pass
    with ThreadPoolExecutor(max_workers=32) as e:
        futures = []
        for _ in range(32):
            futures.append(e.submit(work))
        for future in futures:
            future.result()
    assert 3 <= sum(calls) <= 4