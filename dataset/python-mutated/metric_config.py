from __future__ import absolute_import
import sys

class MetricConfig(object):
    """Configuration values for metrics"""

    def __init__(self, quota=None, samples=2, event_window=sys.maxsize, time_window_ms=30 * 1000, tags=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Arguments:\n            quota (Quota, optional): Upper or lower bound of a value.\n            samples (int, optional): Max number of samples kept per metric.\n            event_window (int, optional): Max number of values per sample.\n            time_window_ms (int, optional): Max age of an individual sample.\n            tags (dict of {str: str}, optional): Tags for each metric.\n        '
        self.quota = quota
        self._samples = samples
        self.event_window = event_window
        self.time_window_ms = time_window_ms
        self.tags = tags if tags else {}

    @property
    def samples(self):
        if False:
            for i in range(10):
                print('nop')
        return self._samples

    @samples.setter
    def samples(self, value):
        if False:
            return 10
        if value < 1:
            raise ValueError('The number of samples must be at least 1.')
        self._samples = value