"""`iobase.RestrictionTracker` implementations provided with Apache Beam."""
from typing import Tuple
from apache_beam.io.iobase import RestrictionProgress
from apache_beam.io.iobase import RestrictionTracker
from apache_beam.io.range_trackers import OffsetRangeTracker

class OffsetRange(object):

    def __init__(self, start, stop):
        if False:
            i = 10
            return i + 15
        if start > stop:
            raise ValueError('Start offset must be not be larger than the stop offset. Received %d and %d respectively.' % (start, stop))
        self.start = start
        self.stop = stop

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        if not isinstance(other, OffsetRange):
            return False
        return self.start == other.start and self.stop == other.stop

    def __hash__(self):
        if False:
            i = 10
            return i + 15
        return hash((type(self), self.start, self.stop))

    def __repr__(self):
        if False:
            print('Hello World!')
        return 'OffsetRange(start=%s, stop=%s)' % (self.start, self.stop)

    def split(self, desired_num_offsets_per_split, min_num_offsets_per_split=1):
        if False:
            while True:
                i = 10
        current_split_start = self.start
        max_split_size = max(desired_num_offsets_per_split, min_num_offsets_per_split)
        while current_split_start < self.stop:
            current_split_stop = min(current_split_start + max_split_size, self.stop)
            remaining = self.stop - current_split_stop
            if remaining < desired_num_offsets_per_split // 4 or remaining < min_num_offsets_per_split:
                current_split_stop = self.stop
            yield OffsetRange(current_split_start, current_split_stop)
            current_split_start = current_split_stop

    def split_at(self, split_pos):
        if False:
            while True:
                i = 10
        return (OffsetRange(self.start, split_pos), OffsetRange(split_pos, self.stop))

    def new_tracker(self):
        if False:
            return 10
        return OffsetRangeTracker(self.start, self.stop)

    def size(self):
        if False:
            for i in range(10):
                print('nop')
        return self.stop - self.start

class OffsetRestrictionTracker(RestrictionTracker):
    """An `iobase.RestrictionTracker` implementations for an offset range.

  Offset range is represented as OffsetRange.
  """

    def __init__(self, offset_range):
        if False:
            while True:
                i = 10
        assert isinstance(offset_range, OffsetRange), offset_range
        self._range = offset_range
        self._current_position = None
        self._last_claim_attempt = None
        self._checkpointed = False

    def check_done(self):
        if False:
            print('Hello World!')
        if self._range.start != self._range.stop and (self._last_claim_attempt is None or self._last_claim_attempt < self._range.stop - 1):
            raise ValueError('OffsetRestrictionTracker is not done since work in range [%s, %s) has not been claimed.' % (self._last_claim_attempt if self._last_claim_attempt is not None else self._range.start, self._range.stop))

    def current_restriction(self):
        if False:
            return 10
        return self._range

    def current_progress(self):
        if False:
            while True:
                i = 10
        if self._current_position is None:
            fraction = 0.0
        elif self._range.stop == self._range.start:
            fraction = 1.0
        else:
            fraction = float(self._current_position - self._range.start) / (self._range.stop - self._range.start)
        return RestrictionProgress(fraction=fraction)

    def start_position(self):
        if False:
            print('Hello World!')
        return self._range.start

    def stop_position(self):
        if False:
            while True:
                i = 10
        return self._range.stop

    def try_claim(self, position):
        if False:
            print('Hello World!')
        if self._last_claim_attempt is not None and position <= self._last_claim_attempt:
            raise ValueError('Positions claimed should strictly increase. Trying to claim position %d while last claim attempt was %d.' % (position, self._last_claim_attempt))
        self._last_claim_attempt = position
        if position < self._range.start:
            raise ValueError('Position to be claimed cannot be smaller than the start position of the range. Tried to claim position %r for the range [%r, %r)' % (position, self._range.start, self._range.stop))
        if self._range.start <= position < self._range.stop:
            self._current_position = position
            return True
        return False

    def try_split(self, fraction_of_remainder):
        if False:
            i = 10
            return i + 15
        if not self._checkpointed:
            if self._last_claim_attempt is None:
                cur = self._range.start - 1
            else:
                cur = self._last_claim_attempt
            split_point = cur + int(max(1, (self._range.stop - cur) * fraction_of_remainder))
            if split_point < self._range.stop:
                if fraction_of_remainder == 0:
                    self._checkpointed = True
                (self._range, residual_range) = self._range.split_at(split_point)
                return (self._range, residual_range)

    def is_bounded(self):
        if False:
            for i in range(10):
                print('nop')
        return True

class UnsplittableRestrictionTracker(RestrictionTracker):
    """An `iobase.RestrictionTracker` that wraps another but does not split."""

    def __init__(self, underling_tracker):
        if False:
            return 10
        self._underling_tracker = underling_tracker

    def try_split(self, fraction_of_remainder):
        if False:
            return 10
        return False

    def __getattribute__(self, name):
        if False:
            while True:
                i = 10
        if name.startswith('_') or name in ('try_split',):
            return super().__getattribute__(name)
        else:
            return getattr(self._underling_tracker, name)