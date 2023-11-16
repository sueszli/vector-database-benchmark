"""Internal class for resource throttle retry policy implementation in the Azure
Cosmos database service.
"""
from . import http_constants

class ResourceThrottleRetryPolicy(object):

    def __init__(self, max_retry_attempt_count, fixed_retry_interval_in_milliseconds, max_wait_time_in_seconds):
        if False:
            i = 10
            return i + 15
        self._max_retry_attempt_count = max_retry_attempt_count
        self._fixed_retry_interval_in_milliseconds = fixed_retry_interval_in_milliseconds
        self._max_wait_time_in_milliseconds = max_wait_time_in_seconds * 1000
        self.current_retry_attempt_count = 0
        self.cumulative_wait_time_in_milliseconds = 0

    def ShouldRetry(self, exception):
        if False:
            while True:
                i = 10
        'Returns true if the request should retry based on the passed-in exception.\n\n        :param exceptions.CosmosHttpResponseError exception:\n        :returns: a boolean stating whether the request should be retried\n        :rtype: bool\n        '
        if self.current_retry_attempt_count < self._max_retry_attempt_count:
            self.current_retry_attempt_count += 1
            self.retry_after_in_milliseconds = 0
            if self._fixed_retry_interval_in_milliseconds:
                self.retry_after_in_milliseconds = self._fixed_retry_interval_in_milliseconds
            elif http_constants.HttpHeaders.RetryAfterInMilliseconds in exception.headers:
                self.retry_after_in_milliseconds = int(exception.headers[http_constants.HttpHeaders.RetryAfterInMilliseconds])
            if self.cumulative_wait_time_in_milliseconds < self._max_wait_time_in_milliseconds:
                self.cumulative_wait_time_in_milliseconds += self.retry_after_in_milliseconds
                return True
        return False