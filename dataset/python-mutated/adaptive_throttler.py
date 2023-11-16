import random
from apache_beam.io.components import util

class AdaptiveThrottler(object):
    """Implements adaptive throttling.

  See
  https://landing.google.com/sre/book/chapters/handling-overload.html#client-side-throttling-a7sYUg
  for a full discussion of the use case and algorithm applied.
  """
    MIN_REQUESTS = 1

    def __init__(self, window_ms, bucket_ms, overload_ratio):
        if False:
            i = 10
            return i + 15
        'Initializes AdaptiveThrottler.\n\n      Args:\n        window_ms: int, length of history to consider, in ms, to set\n                   throttling.\n        bucket_ms: int, granularity of time buckets that we store data in, in\n                   ms.\n        overload_ratio: float, the target ratio between requests sent and\n                        successful requests. This is "K" in the formula in\n                        https://landing.google.com/sre/book/chapters/handling-overload.html.\n    '
        self._all_requests = util.MovingSum(window_ms, bucket_ms)
        self._successful_requests = util.MovingSum(window_ms, bucket_ms)
        self._overload_ratio = float(overload_ratio)
        self._random = random.Random()

    def _throttling_probability(self, now):
        if False:
            return 10
        if not self._all_requests.has_data(now):
            return 0
        all_requests = self._all_requests.sum(now)
        successful_requests = self._successful_requests.sum(now)
        return max(0, (all_requests - self._overload_ratio * successful_requests) / (all_requests + AdaptiveThrottler.MIN_REQUESTS))

    def throttle_request(self, now):
        if False:
            for i in range(10):
                print('nop')
        'Determines whether one RPC attempt should be throttled.\n\n    This should be called once each time the caller intends to send an RPC; if\n    it returns true, drop or delay that request (calling this function again\n    after the delay).\n\n    Args:\n      now: int, time in ms since the epoch\n    Returns:\n      bool, True if the caller should throttle or delay the request.\n    '
        throttling_probability = self._throttling_probability(now)
        self._all_requests.add(now, 1)
        return self._random.uniform(0, 1) < throttling_probability

    def successful_request(self, now):
        if False:
            for i in range(10):
                print('nop')
        'Notifies the throttler of a successful request.\n\n    Must be called once for each request (for which throttle_request was\n    previously called) that succeeded.\n\n    Args:\n      now: int, time in ms since the epoch\n    '
        self._successful_requests.add(now, 1)