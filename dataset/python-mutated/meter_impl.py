from pyflink.metrics import Meter

class MeterImpl(Meter):

    def __init__(self, inner_counter):
        if False:
            for i in range(10):
                print('nop')
        self._inner_counter = inner_counter

    def mark_event(self, value: int=1):
        if False:
            while True:
                i = 10
        '\n        Mark occurrence of the specified number of events.\n\n        .. versionadded:: 1.11.0\n        '
        self._inner_counter.inc(value)

    def get_count(self) -> int:
        if False:
            i = 10
            return i + 15
        '\n        Get number of events marked on the meter.\n\n        .. versionadded:: 1.11.0\n        '
        from apache_beam.metrics.execution import MetricsEnvironment
        container = MetricsEnvironment.current_container()
        return container.get_counter(self._inner_counter.metric_name).get_cumulative()