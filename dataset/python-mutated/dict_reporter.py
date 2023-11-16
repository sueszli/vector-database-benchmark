from __future__ import absolute_import
import logging
import threading
from kafka.metrics.metrics_reporter import AbstractMetricsReporter
logger = logging.getLogger(__name__)

class DictReporter(AbstractMetricsReporter):
    """A basic dictionary based metrics reporter.

    Store all metrics in a two level dictionary of category > name > metric.
    """

    def __init__(self, prefix=''):
        if False:
            print('Hello World!')
        self._lock = threading.Lock()
        self._prefix = prefix if prefix else ''
        self._store = {}

    def snapshot(self):
        if False:
            while True:
                i = 10
        "\n        Return a nested dictionary snapshot of all metrics and their\n        values at this time. Example:\n        {\n            'category': {\n                'metric1_name': 42.0,\n                'metric2_name': 'foo'\n            }\n        }\n        "
        return dict(((category, dict(((name, metric.value()) for (name, metric) in list(metrics.items())))) for (category, metrics) in list(self._store.items())))

    def init(self, metrics):
        if False:
            for i in range(10):
                print('nop')
        for metric in metrics:
            self.metric_change(metric)

    def metric_change(self, metric):
        if False:
            i = 10
            return i + 15
        with self._lock:
            category = self.get_category(metric)
            if category not in self._store:
                self._store[category] = {}
            self._store[category][metric.metric_name.name] = metric

    def metric_removal(self, metric):
        if False:
            i = 10
            return i + 15
        with self._lock:
            category = self.get_category(metric)
            metrics = self._store.get(category, {})
            removed = metrics.pop(metric.metric_name.name, None)
            if not metrics:
                self._store.pop(category, None)
            return removed

    def get_category(self, metric):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return a string category for the metric.\n\n        The category is made up of this reporter's prefix and the\n        metric's group and tags.\n\n        Examples:\n            prefix = 'foo', group = 'bar', tags = {'a': 1, 'b': 2}\n            returns: 'foo.bar.a=1,b=2'\n\n            prefix = 'foo', group = 'bar', tags = None\n            returns: 'foo.bar'\n\n            prefix = None, group = 'bar', tags = None\n            returns: 'bar'\n        "
        tags = ','.join(('%s=%s' % (k, v) for (k, v) in sorted(metric.metric_name.tags.items())))
        return '.'.join((x for x in [self._prefix, metric.metric_name.group, tags] if x))

    def configure(self, configs):
        if False:
            print('Hello World!')
        pass

    def close(self):
        if False:
            print('Hello World!')
        pass