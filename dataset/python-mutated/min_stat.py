from __future__ import absolute_import
import sys
from kafka.metrics.stats.sampled_stat import AbstractSampledStat

class Min(AbstractSampledStat):
    """An AbstractSampledStat that gives the min over its samples."""

    def __init__(self):
        if False:
            while True:
                i = 10
        super(Min, self).__init__(float(sys.maxsize))

    def update(self, sample, config, value, now):
        if False:
            i = 10
            return i + 15
        sample.value = min(sample.value, value)

    def combine(self, samples, config, now):
        if False:
            for i in range(10):
                print('nop')
        if not samples:
            return float(sys.maxsize)
        return float(min((sample.value for sample in samples)))