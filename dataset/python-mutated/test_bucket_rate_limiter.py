import time
from dramatiq.rate_limits import BucketRateLimiter
from .common import skip_in_ci

@skip_in_ci
def test_bucket_rate_limiter_limits_per_bucket(rate_limiter_backend):
    if False:
        i = 10
        return i + 15
    limiter = BucketRateLimiter(rate_limiter_backend, 'sequential-test', limit=2)
    calls = 0
    for _ in range(2):
        now = time.time()
        time.sleep(1 - (now - int(now)))
        for _ in range(8):
            with limiter.acquire(raise_on_failure=False) as acquired:
                if not acquired:
                    continue
                calls += 1
    assert calls == 4