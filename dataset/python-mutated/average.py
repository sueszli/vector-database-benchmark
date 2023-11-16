from allennlp.training.metrics.metric import Metric
from allennlp.nn.util import dist_reduce_sum

@Metric.register('average')
class Average(Metric):
    """
    This [`Metric`](./metric.md) breaks with the typical `Metric` API and just stores values that were
    computed in some fashion outside of a `Metric`.  If you have some external code that computes
    the metric for you, for instance, you can use this to report the average result using our
    `Metric` API.
    """

    def __init__(self) -> None:
        if False:
            return 10
        self._total_value = 0.0
        self._count = 0

    def __call__(self, value):
        if False:
            print('Hello World!')
        '\n        # Parameters\n\n        value : `float`\n            The value to average.\n        '
        self._count += dist_reduce_sum(1)
        self._total_value += dist_reduce_sum(float(list(self.detach_tensors(value))[0]))

    def get_metric(self, reset: bool=False):
        if False:
            i = 10
            return i + 15
        '\n        # Returns\n\n        The average of all values that were passed to `__call__`.\n        '
        average_value = self._total_value / self._count if self._count > 0 else 0.0
        if reset:
            self.reset()
        return float(average_value)

    def reset(self):
        if False:
            print('Hello World!')
        self._total_value = 0.0
        self._count = 0