from __future__ import absolute_import
import abc

class AbstractMetricsReporter(object):
    """
    An abstract class to allow things to listen as new metrics
    are created so they can be reported.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def init(self, metrics):
        if False:
            print('Hello World!')
        '\n        This is called when the reporter is first registered\n        to initially register all existing metrics\n\n        Arguments:\n            metrics (list of KafkaMetric): All currently existing metrics\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def metric_change(self, metric):
        if False:
            i = 10
            return i + 15
        '\n        This is called whenever a metric is updated or added\n\n        Arguments:\n            metric (KafkaMetric)\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def metric_removal(self, metric):
        if False:
            i = 10
            return i + 15
        '\n        This is called whenever a metric is removed\n\n        Arguments:\n            metric (KafkaMetric)\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def configure(self, configs):
        if False:
            print('Hello World!')
        '\n        Configure this class with the given key-value pairs\n\n        Arguments:\n            configs (dict of {str, ?})\n        '
        raise NotImplementedError

    @abc.abstractmethod
    def close(self):
        if False:
            return 10
        'Called when the metrics repository is closed.'
        raise NotImplementedError