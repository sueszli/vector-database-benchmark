"""Class for retry options in the Azure Cosmos database service.
"""

class RetryOptions(object):
    """The retry options to be applied to all requests when retrying.

    :ivar int MaxRetryAttemptCount:
        Max number of retries to be performed for a request. Default value 9.
    :ivar int FixedRetryIntervalInMilliseconds:
        Fixed retry interval in milliseconds to wait between each retry ignoring
        the retryAfter returned as part of the response.
    :ivar int MaxWaitTimeInSeconds:
        Max wait time in seconds to wait for a request while the retries are happening.
        Default value 30 seconds.
    """

    def __init__(self, max_retry_attempt_count=9, fixed_retry_interval_in_milliseconds=None, max_wait_time_in_seconds=30):
        if False:
            i = 10
            return i + 15
        self._max_retry_attempt_count = max_retry_attempt_count
        self._fixed_retry_interval_in_milliseconds = fixed_retry_interval_in_milliseconds
        self._max_wait_time_in_seconds = max_wait_time_in_seconds

    @property
    def MaxRetryAttemptCount(self):
        if False:
            while True:
                i = 10
        return self._max_retry_attempt_count

    @property
    def FixedRetryIntervalInMilliseconds(self):
        if False:
            return 10
        return self._fixed_retry_interval_in_milliseconds

    @property
    def MaxWaitTimeInSeconds(self):
        if False:
            return 10
        return self._max_wait_time_in_seconds