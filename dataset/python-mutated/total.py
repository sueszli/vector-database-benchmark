from __future__ import absolute_import
from kafka.metrics.measurable_stat import AbstractMeasurableStat

class Total(AbstractMeasurableStat):
    """An un-windowed cumulative total maintained over all time."""

    def __init__(self, value=0.0):
        if False:
            return 10
        self._total = value

    def record(self, config, value, now):
        if False:
            i = 10
            return i + 15
        self._total += value

    def measure(self, config, now):
        if False:
            return 10
        return float(self._total)