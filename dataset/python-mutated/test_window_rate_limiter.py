import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dramatiq.rate_limits import WindowRateLimiter
from .common import skip_in_ci

@skip_in_ci
def test_window_rate_limiter_limits_per_window(rate_limiter_backend):
    if False:
        print('Hello World!')
    limiter = WindowRateLimiter(rate_limiter_backend, 'window-test', limit=2, window=5)
    calls = defaultdict(lambda : 0)

    def work():
        if False:
            print('Hello World!')
        for _ in range(20):
            for _ in range(8):
                with limiter.acquire(raise_on_failure=False) as acquired:
                    if not acquired:
                        continue
                    calls[int(time.time())] += 1
            time.sleep(1)
    with ThreadPoolExecutor(max_workers=8) as e:
        futures = []
        for _ in range(8):
            futures.append(e.submit(work))
        for future in futures:
            future.result()
    assert 8 <= sum(calls.values()) <= 10