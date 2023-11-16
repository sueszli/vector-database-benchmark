import logging
import math
import time
from apache_beam.metrics.metric import Metrics
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
_LOGGER = logging.getLogger(__name__)

class GcsIOOverrides(object):
    """Functions for overriding Google Cloud Storage I/O client."""
    _THROTTLED_SECS = Metrics.counter('StorageV1', 'cumulativeThrottlingSeconds')

    @classmethod
    def retry_func(cls, retry_args):
        if False:
            print('Hello World!')
        if isinstance(retry_args.exc, exceptions.BadStatusCodeError) and retry_args.exc.status_code == http_wrapper.TOO_MANY_REQUESTS:
            _LOGGER.debug('Caught GCS quota error (%s), retrying.', retry_args.exc.status_code)
        else:
            return http_wrapper.HandleExceptionsAndRebuildHttpConnections(retry_args)
        http_wrapper.RebuildHttpConnections(retry_args.http)
        _LOGGER.debug('Retrying request to url %s after exception %s', retry_args.http_request.url, retry_args.exc)
        sleep_seconds = util.CalculateWaitForRetry(retry_args.num_retries, max_wait=retry_args.max_retry_wait)
        cls._THROTTLED_SECS.inc(math.ceil(sleep_seconds))
        time.sleep(sleep_seconds)