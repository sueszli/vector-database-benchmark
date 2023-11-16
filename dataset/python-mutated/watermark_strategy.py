import abc
from typing import Any, Optional
from pyflink.common.time import Duration
from pyflink.java_gateway import get_gateway

class WatermarkStrategy(object):
    """
    The WatermarkStrategy defines how to generate Watermarks in the stream sources. The
    WatermarkStrategy is a builder/factory for the WatermarkGenerator that generates the watermarks
    and the TimestampAssigner which assigns the internal timestamp of a record.

    The convenience methods, for example forBoundedOutOfOrderness(Duration), create a
    WatermarkStrategy for common built in strategies.
    """

    def __init__(self, j_watermark_strategy):
        if False:
            print('Hello World!')
        self._j_watermark_strategy = j_watermark_strategy
        self._timestamp_assigner = None

    def with_timestamp_assigner(self, timestamp_assigner: 'TimestampAssigner') -> 'WatermarkStrategy':
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new WatermarkStrategy that wraps this strategy but instead uses the given a\n        TimestampAssigner by implementing TimestampAssigner interface.\n\n        Example:\n        ::\n\n            >>> watermark_strategy = WatermarkStrategy.for_monotonous_timestamps() \\\n            >>>     .with_timestamp_assigner(MyTimestampAssigner())\n\n        :param timestamp_assigner: The given TimestampAssigner.\n        :return: A WaterMarkStrategy that wraps a TimestampAssigner.\n        '
        self._timestamp_assigner = timestamp_assigner
        return self

    def with_idleness(self, idle_timeout: Duration) -> 'WatermarkStrategy':
        if False:
            return 10
        '\n        Creates a new enriched WatermarkStrategy that also does idleness detection in the created\n        WatermarkGenerator.\n\n        Example:\n        ::\n\n            >>> WatermarkStrategy \\\n            ...     .for_bounded_out_of_orderness(Duration.of_seconds(20)) \\\n            ...     .with_idleness(Duration.of_minutes(1))\n\n        :param idle_timeout: The idle timeout.\n        :return: A new WatermarkStrategy with idle detection configured.\n        '
        return WatermarkStrategy(self._j_watermark_strategy.withIdleness(idle_timeout._j_duration))

    def with_watermark_alignment(self, watermark_group: str, max_allowed_watermark_drift: Duration, update_interval: Optional[Duration]=None) -> 'WatermarkStrategy':
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a new :class:`WatermarkStrategy` that configures the maximum watermark drift from\n        other sources/tasks/partitions in the same watermark group. The group may contain completely\n        independent sources (e.g. File and Kafka).\n\n        Once configured Flink will "pause" consuming from a source/task/partition that is ahead\n        of the emitted watermark in the group by more than the maxAllowedWatermarkDrift.\n\n        Example:\n        ::\n\n            >>> WatermarkStrategy \\\n            ...     .for_bounded_out_of_orderness(Duration.of_seconds(20)) \\\n            ...     .with_watermark_alignment("alignment-group-1", Duration.of_seconds(20),\n            ...         Duration.of_seconds(1))\n\n        :param watermark_group: A group of sources to align watermarks\n        :param max_allowed_watermark_drift: Maximal drift, before we pause consuming from the\n            source/task/partition\n        :param update_interval: How often tasks should notify coordinator about the current\n            watermark and how often the coordinator should announce the maximal aligned watermark.\n            If is None, default update interval (1000ms) is used.\n        :return: A new WatermarkStrategy with watermark alignment configured.\n\n        .. versionadded:: 1.16.0\n        '
        if update_interval is None:
            return WatermarkStrategy(self._j_watermark_strategy.withWatermarkAlignment(watermark_group, max_allowed_watermark_drift._j_duration))
        else:
            return WatermarkStrategy(self._j_watermark_strategy.withWatermarkAlignment(watermark_group, max_allowed_watermark_drift._j_duration, update_interval._j_duration))

    @staticmethod
    def for_monotonous_timestamps() -> 'WatermarkStrategy':
        if False:
            for i in range(10):
                print('nop')
        '\n        Creates a watermark strategy for situations with monotonously ascending timestamps.\n\n        The watermarks are generated periodically and tightly follow the latest timestamp in the\n        data. The delay introduced by this strategy is mainly the periodic interval in which the\n        watermarks are generated.\n        '
        JWaterMarkStrategy = get_gateway().jvm.org.apache.flink.api.common.eventtime.WatermarkStrategy
        return WatermarkStrategy(JWaterMarkStrategy.forMonotonousTimestamps())

    @staticmethod
    def for_bounded_out_of_orderness(max_out_of_orderness: Duration) -> 'WatermarkStrategy':
        if False:
            return 10
        '\n        Creates a watermark strategy for situations where records are out of order, but you can\n        place an upper bound on how far the events are out of order. An out-of-order bound B means\n        that once the an event with timestamp T was encountered, no events older than (T - B) will\n        follow any more.\n        '
        JWaterMarkStrategy = get_gateway().jvm.org.apache.flink.api.common.eventtime.WatermarkStrategy
        return WatermarkStrategy(JWaterMarkStrategy.forBoundedOutOfOrderness(max_out_of_orderness._j_duration))

    @staticmethod
    def no_watermarks() -> 'WatermarkStrategy':
        if False:
            print('Hello World!')
        '\n        Creates a watermark strategy that generates no watermarks at all. This may be useful in\n        scenarios that do pure processing-time based stream processing.\n\n        .. versionadded:: 1.16.0\n        '
        JWaterMarkStrategy = get_gateway().jvm.org.apache.flink.api.common.eventtime.WatermarkStrategy
        return WatermarkStrategy(JWaterMarkStrategy.noWatermarks())

class TimestampAssigner(abc.ABC):
    """
    A TimestampAssigner assigns event time timestamps to elements. These timestamps are used by all
    functions that operate on event time, for example event time windows.

    Timestamps can be an arbitrary int value, but all built-in implementations represent it as the
    milliseconds since the Epoch (midnight, January 1, 1970 UTC), the same way as time.time() does
    it.
    """

    @abc.abstractmethod
    def extract_timestamp(self, value: Any, record_timestamp: int) -> int:
        if False:
            return 10
        '\n        Assigns a timestamp to an element, in milliseconds since the Epoch. This is independent of\n        any particular time zone or calendar.\n\n        The method is passed the previously assigned timestamp of the element.\n        That previous timestamp may have been assigned from a previous assigner. If the element did\n        not carry a timestamp before, this value is the minimum value of int type.\n\n        :param value: The element that the timestamp will be assigned to.\n        :param record_timestamp: The current internal timestamp of the element, or a negative value,\n                                 if no timestamp has been assigned yet.\n        :return: The new timestamp.\n        '
        pass

class AssignerWithPeriodicWatermarksWrapper(object):
    """
    The AssignerWithPeriodicWatermarks assigns event time timestamps to elements, and generates
    low watermarks that signal event time progress within the stream. These timestamps and
    watermarks are used by functions and operators that operate on event time, for example event
    time windows.
    """

    def __init__(self, j_assigner_with_periodic_watermarks):
        if False:
            return 10
        self._j_assigner_with_periodic_watermarks = j_assigner_with_periodic_watermarks