import dataclasses
import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple
import pendulum
from pendulum.datetime import DateTime, Period

@dataclass
class StreamSlice:
    start_date: DateTime
    end_date: DateTime

    def __iter__(self):
        if False:
            print('Hello World!')
        return ((field.name, getattr(self, field.name)) for field in dataclasses.fields(self))

class SliceGenerator:
    """
    Base class for slice generators.
    """
    _start_date: DateTime = None
    _end_data: DateTime = None

    def __init__(self, start_date: DateTime, end_date: Optional[DateTime]=None):
        if False:
            return 10
        self._start_date = start_date
        self._end_date = end_date or pendulum.now('UTC')

    def __iter__(self):
        if False:
            print('Hello World!')
        return self

class RangeSliceGenerator(SliceGenerator):
    """
    Split slices into event ranges of 90 days (or less for final slice) from
    start_date up to current date.
    """
    RANGE_LENGTH_DAYS: int = 90
    _slices: List[StreamSlice] = []

    def __init__(self, start_date: DateTime, end_date: Optional[DateTime]=None):
        if False:
            return 10
        super().__init__(start_date, end_date)
        self._slices = [StreamSlice(start_date=start, end_date=end) for (start, end) in self.make_datetime_ranges(self._start_date, self._end_date, self.RANGE_LENGTH_DAYS)]

    def __next__(self) -> StreamSlice:
        if False:
            i = 10
            return i + 15
        if not self._slices:
            raise StopIteration()
        return self._slices.pop(0)

    @staticmethod
    def make_datetime_ranges(start: DateTime, end: DateTime, range_days: int) -> Iterable[Tuple[DateTime, DateTime]]:
        if False:
            i = 10
            return i + 15
        '\n        Generates list of ranges starting from start up to end date with duration of ranges_days.\n        Args:\n            start (DateTime): start of the range\n            end (DateTime): end of the range\n            range_days (int): Number in days to split subranges into.\n\n        Returns:\n            List[Tuple[DateTime, DateTime]]: list of tuples with ranges.\n\n            Each tuple contains two daytime variables: first is period start\n            and second is period end.\n        '
        if start > end:
            return []
        next_start = start
        period = pendulum.Duration(days=range_days)
        while next_start < end:
            next_end = min(next_start + period, end)
            yield (next_start, next_end)
            next_start = next_end

class AdjustableSliceGenerator(SliceGenerator):
    """
    Generate slices from start_date up to current date. Every next slice could
    have different range based on was the previous slice processed successfully
    and how much time it took.
    The alghorithm is following:
    1. First slice have INITIAL_RANGE_DAYS (30 days) length.
    2. When slice is processed by stream this class expect "adjust_range"
    method to be called with parameter how much time it took to process
    previous request
    3. Knowing previous slice range we can calculate days per minute processing
    speed. Dividing this speed by REQUEST_PER_MINUTE_LIMIT (4) we can calculate
    next slice range. Next range cannot be greater than MAX_RANGE_DAYS (180 days)

    If processing of previous slice havent been completed "reduce_range" method
    should be called. It would reset next range start date to previous slice
    and reduce next slice range by RANGE_REDUCE_FACTOR (2 times)

    In case if range havent been adjusted before getting next slice (it could
    happend if there were no records for given date range), next slice would
    have MAX_RANGE_DAYS (180) length.
    """
    REQUEST_PER_MINUTE_LIMIT = 4
    INITIAL_RANGE_DAYS: int = 30
    DEFAULT_RANGE_DAYS: int = 90
    MAX_RANGE_DAYS: int = 180
    RANGE_REDUCE_FACTOR = 2
    _current_range: int = INITIAL_RANGE_DAYS
    _prev_start_date: DateTime = None
    _range_adjusted = True

    def adjust_range(self, previous_request_time: Period):
        if False:
            i = 10
            return i + 15
        '\n        Calculate next slice length in days based on previous slice length and\n        processing time.\n        '
        minutes_spent = previous_request_time.total_minutes()
        if minutes_spent == 0:
            self._current_range = self.DEFAULT_RANGE_DAYS
        else:
            days_per_minute = self._current_range / minutes_spent
            next_range = math.floor(days_per_minute / self.REQUEST_PER_MINUTE_LIMIT)
            self._current_range = min(next_range or self.DEFAULT_RANGE_DAYS, self.MAX_RANGE_DAYS)
        self._range_adjusted = True

    def reduce_range(self) -> StreamSlice:
        if False:
            print('Hello World!')
        '\n        This method is supposed to be called when slice processing failed.\n        Reset next slice start date to previous one and reduce slice range by\n        RANGE_REDUCE_FACTOR (2 times).\n        Returns updated slice to try again.\n        '
        self._current_range = int(max(self._current_range / self.RANGE_REDUCE_FACTOR, self.INITIAL_RANGE_DAYS))
        start_date = self._prev_start_date
        end_date = min(self._end_date, start_date + pendulum.Duration(days=self._current_range))
        self._start_date = end_date
        return StreamSlice(start_date=start_date, end_date=end_date)

    def __next__(self) -> StreamSlice:
        if False:
            i = 10
            return i + 15
        '\n        Generates next slice based on prevouis slice processing result. All the\n        next slice range calculations should be done after calling adjust_range\n        and reduce_range methods.\n        '
        if self._start_date >= self._end_date:
            raise StopIteration()
        if not self._range_adjusted:
            self._current_range = self.MAX_RANGE_DAYS
        next_start_date = min(self._end_date, self._start_date + pendulum.Duration(days=self._current_range))
        slice = StreamSlice(start_date=self._start_date, end_date=next_start_date)
        self._prev_start_date = self._start_date
        self._start_date = next_start_date
        self._range_adjusted = False
        return slice