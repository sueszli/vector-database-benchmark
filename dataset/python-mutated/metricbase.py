from abc import ABC, abstractmethod
from typing import Callable

class MetricGroup(ABC):
    """
    A MetricGroup is a named container for metrics and further metric subgroups.

    Instances of this class can be used to register new metrics with Flink and to create a nested
    hierarchy based on the group names.

    A MetricGroup is uniquely identified by it's place in the hierarchy and name.

    .. versionadded:: 1.11.0
    """

    @abstractmethod
    def add_group(self, name: str, extra: str=None) -> 'MetricGroup':
        if False:
            i = 10
            return i + 15
        "\n        Creates a new MetricGroup and adds it to this groups sub-groups.\n\n        If extra is not None, creates a new key-value MetricGroup pair.\n        The key group is added to this group's sub-groups, while the value\n        group is added to the key group's sub-groups. In this case,\n        the value group will be returned and a user variable will be defined.\n\n        .. versionadded:: 1.11.0\n        "
        pass

    @abstractmethod
    def counter(self, name: str) -> 'Counter':
        if False:
            for i in range(10):
                print('nop')
        '\n        Registers a new `Counter` with Flink.\n\n        .. versionadded:: 1.11.0\n        '
        pass

    @abstractmethod
    def gauge(self, name: str, obj: Callable[[], int]) -> None:
        if False:
            while True:
                i = 10
        '\n        Registers a new `Gauge` with Flink.\n\n        .. versionadded:: 1.11.0\n        '
        pass

    @abstractmethod
    def meter(self, name: str, time_span_in_seconds: int=60) -> 'Meter':
        if False:
            i = 10
            return i + 15
        '\n        Registers a new `Meter` with Flink.\n\n        .. versionadded:: 1.11.0\n        '
        pass

    @abstractmethod
    def distribution(self, name: str) -> 'Distribution':
        if False:
            return 10
        '\n        Registers a new `Distribution` with Flink.\n\n        .. versionadded:: 1.11.0\n        '
        pass

class Metric(ABC):
    """
    Base interface of a metric object.

    .. versionadded:: 1.11.0
    """
    pass

class Counter(Metric, ABC):
    """
    Counter metric interface. Allows a count to be incremented/decremented
    during pipeline execution.

    .. versionadded:: 1.11.0
    """

    @abstractmethod
    def inc(self, n: int=1):
        if False:
            while True:
                i = 10
        '\n        Increment the current count by the given value.\n\n        .. versionadded:: 1.11.0\n        '
        pass

    @abstractmethod
    def dec(self, n: int=1):
        if False:
            i = 10
            return i + 15
        '\n        Decrement the current count by 1.\n\n        .. versionadded:: 1.11.0\n        '
        pass

    @abstractmethod
    def get_count(self) -> int:
        if False:
            while True:
                i = 10
        '\n        Returns the current count.\n\n        .. versionadded:: 1.11.0\n        '
        pass

class Meter(Metric, ABC):
    """
    Meter Metric interface.

    Metric for measuring throughput.

    .. versionadded:: 1.11.0
    """

    @abstractmethod
    def mark_event(self, value: int=1):
        if False:
            print('Hello World!')
        '\n        Mark occurrence of the specified number of events.\n\n        .. versionadded:: 1.11.0\n        '
        pass

    @abstractmethod
    def get_count(self) -> int:
        if False:
            print('Hello World!')
        '\n        Get number of events marked on the meter.\n\n        .. versionadded:: 1.11.0\n        '
        pass

class Distribution(Metric, ABC):
    """
    Distribution Metric interface.

    Allows statistics about the distribution of a variable to be collected during
    pipeline execution.

    .. versionadded:: 1.11.0
    """

    @abstractmethod
    def update(self, value):
        if False:
            print('Hello World!')
        pass