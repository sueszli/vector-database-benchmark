from __future__ import absolute_import
import abc
from kafka.metrics.stat import AbstractStat

class AbstractCompoundStat(AbstractStat):
    """
    A compound stat is a stat where a single measurement and associated
    data structure feeds many metrics. This is the example for a
    histogram which has many associated percentiles.
    """
    __metaclass__ = abc.ABCMeta

    def stats(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return list of NamedMeasurable\n        '
        raise NotImplementedError

class NamedMeasurable(object):

    def __init__(self, metric_name, measurable_stat):
        if False:
            print('Hello World!')
        self._name = metric_name
        self._stat = measurable_stat

    @property
    def name(self):
        if False:
            for i in range(10):
                print('nop')
        return self._name

    @property
    def stat(self):
        if False:
            return 10
        return self._stat