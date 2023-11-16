from __future__ import absolute_import
from kafka.metrics.stats.sampled_stat import AbstractSampledStat

class Avg(AbstractSampledStat):
    """
    An AbstractSampledStat that maintains a simple average over its samples.
    """

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super(Avg, self).__init__(0.0)

    def update(self, sample, config, value, now):
        if False:
            print('Hello World!')
        sample.value += value

    def combine(self, samples, config, now):
        if False:
            for i in range(10):
                print('nop')
        total_sum = 0
        total_count = 0
        for sample in samples:
            total_sum += sample.value
            total_count += sample.event_count
        if not total_count:
            return 0
        return float(total_sum) / total_count