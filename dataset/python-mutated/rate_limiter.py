from contextlib import contextmanager
from ..errors import RateLimitExceeded

class RateLimiter:
    """ABC for rate limiters.

    Examples:

      >>> from dramatiq.rate_limits.backends import RedisBackend

      >>> backend = RedisBackend()
      >>> limiter = ConcurrentRateLimiter(backend, "distributed-mutex", limit=1)

      >>> with limiter.acquire(raise_on_failure=False) as acquired:
      ...     if not acquired:
      ...         print("Mutex not acquired.")
      ...         return
      ...
      ...     print("Mutex acquired.")

    Parameters:
      backend(RateLimiterBackend): The rate limiting backend to use.
      key(str): The key to rate limit on.
    """

    def __init__(self, backend, key):
        if False:
            for i in range(10):
                print('nop')
        self.backend = backend
        self.key = key

    def _acquire(self):
        if False:
            return 10
        raise NotImplementedError

    def _release(self):
        if False:
            return 10
        raise NotImplementedError

    @contextmanager
    def acquire(self, *, raise_on_failure=True):
        if False:
            while True:
                i = 10
        'Attempt to acquire a slot under this rate limiter.\n\n        Parameters:\n          raise_on_failure(bool): Whether or not failures should raise an\n            exception.  If this is false, the context manager will instead\n            return a boolean value representing whether or not the rate\n            limit slot was acquired.\n\n        Returns:\n          bool: Whether or not the slot could be acquired.\n        '
        acquired = False
        try:
            acquired = self._acquire()
            if raise_on_failure and (not acquired):
                raise RateLimitExceeded('rate limit exceeded for key %(key)r' % vars(self))
            yield acquired
        finally:
            if acquired:
                self._release()