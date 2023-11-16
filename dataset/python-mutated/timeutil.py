"""Timestamp utilities."""
from abc import ABCMeta
from abc import abstractmethod
from apache_beam.portability.api import beam_runner_api_pb2
__all__ = ['TimeDomain']

class TimeDomain(object):
    """Time domain for streaming timers."""
    WATERMARK = 'WATERMARK'
    REAL_TIME = 'REAL_TIME'
    DEPENDENT_REAL_TIME = 'DEPENDENT_REAL_TIME'
    _RUNNER_API_MAPPING = {WATERMARK: beam_runner_api_pb2.TimeDomain.EVENT_TIME, REAL_TIME: beam_runner_api_pb2.TimeDomain.PROCESSING_TIME}

    @staticmethod
    def from_string(domain):
        if False:
            while True:
                i = 10
        if domain in (TimeDomain.WATERMARK, TimeDomain.REAL_TIME, TimeDomain.DEPENDENT_REAL_TIME):
            return domain
        raise ValueError('Unknown time domain: %s' % domain)

    @staticmethod
    def to_runner_api(domain):
        if False:
            print('Hello World!')
        return TimeDomain._RUNNER_API_MAPPING[domain]

    @staticmethod
    def is_event_time(domain):
        if False:
            i = 10
            return i + 15
        return TimeDomain.from_string(domain) == TimeDomain.WATERMARK

class TimestampCombinerImpl(metaclass=ABCMeta):
    """Implementation of TimestampCombiner."""

    @abstractmethod
    def assign_output_time(self, window, input_timestamp):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    @abstractmethod
    def combine(self, output_timestamp, other_output_timestamp):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError

    def combine_all(self, merging_timestamps):
        if False:
            return 10
        'Apply combine to list of timestamps.'
        combined_output_time = None
        for output_time in merging_timestamps:
            if combined_output_time is None:
                combined_output_time = output_time
            elif output_time is not None:
                combined_output_time = self.combine(combined_output_time, output_time)
        return combined_output_time

    def merge(self, unused_result_window, merging_timestamps):
        if False:
            i = 10
            return i + 15
        'Default to returning the result of combine_all.'
        return self.combine_all(merging_timestamps)

class DependsOnlyOnWindow(TimestampCombinerImpl, metaclass=ABCMeta):
    """TimestampCombinerImpl that only depends on the window."""

    def merge(self, result_window, unused_merging_timestamps):
        if False:
            i = 10
            return i + 15
        return self.assign_output_time(result_window, None)

class OutputAtEarliestInputTimestampImpl(TimestampCombinerImpl):
    """TimestampCombinerImpl outputting at earliest input timestamp."""

    def assign_output_time(self, window, input_timestamp):
        if False:
            i = 10
            return i + 15
        return input_timestamp

    def combine(self, output_timestamp, other_output_timestamp):
        if False:
            print('Hello World!')
        'Default to returning the earlier of two timestamps.'
        return min(output_timestamp, other_output_timestamp)

class OutputAtEarliestTransformedInputTimestampImpl(TimestampCombinerImpl):
    """TimestampCombinerImpl outputting at earliest input timestamp."""

    def __init__(self, window_fn):
        if False:
            while True:
                i = 10
        self.window_fn = window_fn

    def assign_output_time(self, window, input_timestamp):
        if False:
            for i in range(10):
                print('nop')
        return self.window_fn.get_transformed_output_time(window, input_timestamp)

    def combine(self, output_timestamp, other_output_timestamp):
        if False:
            i = 10
            return i + 15
        return min(output_timestamp, other_output_timestamp)

class OutputAtLatestInputTimestampImpl(TimestampCombinerImpl):
    """TimestampCombinerImpl outputting at latest input timestamp."""

    def assign_output_time(self, window, input_timestamp):
        if False:
            for i in range(10):
                print('nop')
        return input_timestamp

    def combine(self, output_timestamp, other_output_timestamp):
        if False:
            for i in range(10):
                print('nop')
        return max(output_timestamp, other_output_timestamp)

class OutputAtEndOfWindowImpl(DependsOnlyOnWindow):
    """TimestampCombinerImpl outputting at end of window."""

    def assign_output_time(self, window, unused_input_timestamp):
        if False:
            print('Hello World!')
        return window.max_timestamp()

    def combine(self, output_timestamp, other_output_timestamp):
        if False:
            i = 10
            return i + 15
        return max(output_timestamp, other_output_timestamp)