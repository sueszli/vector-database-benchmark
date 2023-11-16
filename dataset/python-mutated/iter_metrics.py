import collections
from typing import List
from ray.util.annotations import Deprecated
from ray.util.timer import _Timer

@Deprecated
class MetricsContext:
    """Metrics context object for a local iterator.

    This object is accessible by all operators of a local iterator. It can be
    used to store and retrieve global execution metrics for the iterator.
    It can be accessed by calling LocalIterator.get_metrics(), which is only
    allowable inside iterator functions.

    Attributes:
        counters: dict storing increasing metrics.
        timers: dict storing latency timers.
        info: dict storing misc metric values.
        current_actor: reference to the actor handle that
            produced the current iterator output. This is automatically set
            for gather_async().
    """

    def __init__(self):
        if False:
            while True:
                i = 10
        self.counters = collections.defaultdict(int)
        self.timers = collections.defaultdict(_Timer)
        self.info = {}
        self.current_actor = None

    def save(self):
        if False:
            return 10
        'Return a serializable copy of this context.'
        return {'counters': dict(self.counters), 'info': dict(self.info), 'timers': None}

    def restore(self, values):
        if False:
            i = 10
            return i + 15
        'Restores state given the output of save().'
        self.counters.clear()
        self.counters.update(values['counters'])
        self.timers.clear()
        self.info = values['info']

@Deprecated
class SharedMetrics:
    """Holds an indirect reference to a (shared) metrics context.

    This is used by LocalIterator.union() to point the metrics contexts of
    entirely separate iterator chains to the same underlying context."""

    def __init__(self, metrics: MetricsContext=None, parents: List['SharedMetrics']=None):
        if False:
            while True:
                i = 10
        self.metrics = metrics or MetricsContext()
        self.parents = parents or []
        self.set(self.metrics)

    def set(self, metrics):
        if False:
            for i in range(10):
                print('nop')
        'Recursively set self and parents to point to the same metrics.'
        self.metrics = metrics
        for parent in self.parents:
            parent.set(metrics)

    def get(self):
        if False:
            while True:
                i = 10
        return self.metrics