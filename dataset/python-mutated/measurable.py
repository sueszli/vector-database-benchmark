from __future__ import absolute_import
import abc

class AbstractMeasurable(object):
    """A measurable quantity that can be registered as a metric"""

    @abc.abstractmethod
    def measure(self, config, now):
        if False:
            i = 10
            return i + 15
        '\n        Measure this quantity and return the result\n\n        Arguments:\n            config (MetricConfig): The configuration for this metric\n            now (int): The POSIX time in milliseconds the measurement\n                is being taken\n\n        Returns:\n            The measured value\n        '
        raise NotImplementedError

class AnonMeasurable(AbstractMeasurable):

    def __init__(self, measure_fn):
        if False:
            i = 10
            return i + 15
        self._measure_fn = measure_fn

    def measure(self, config, now):
        if False:
            i = 10
            return i + 15
        return float(self._measure_fn(config, now))