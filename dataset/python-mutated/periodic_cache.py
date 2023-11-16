from datetime import datetime, timezone
from cachetools import TTLCache

class PeriodicCache(TTLCache):
    """
    Special cache that expires at "straight" times
    A timer with ttl of 3600 (1h) will expire at every full hour (:00).
    """

    def __init__(self, maxsize, ttl, getsizeof=None):
        if False:
            return 10

        def local_timer():
            if False:
                for i in range(10):
                    print('nop')
            ts = datetime.now(timezone.utc).timestamp()
            offset = ts % ttl
            return ts - offset
        super().__init__(maxsize=maxsize, ttl=ttl - 1e-05, timer=local_timer, getsizeof=getsizeof)