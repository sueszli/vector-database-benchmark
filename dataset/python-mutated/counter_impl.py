from pyflink.metrics import Counter

class CounterImpl(Counter):

    def __init__(self, inner_counter):
        if False:
            while True:
                i = 10
        self._inner_counter = inner_counter

    def inc(self, n: int=1):
        if False:
            return 10
        '\n        Increment the current count by the given value.\n\n        .. versionadded:: 1.11.0\n        '
        self._inner_counter.inc(n)

    def dec(self, n: int=1):
        if False:
            while True:
                i = 10
        '\n        Decrement the current count by 1.\n\n        .. versionadded:: 1.11.0\n        '
        self.inc(-n)

    def get_count(self) -> int:
        if False:
            return 10
        '\n        Returns the current count.\n\n        .. versionadded:: 1.11.0\n        '
        from apache_beam.metrics.execution import MetricsEnvironment
        container = MetricsEnvironment.current_container()
        return container.get_counter(self._inner_counter.metric_name).get_cumulative()