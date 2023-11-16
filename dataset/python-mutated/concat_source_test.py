"""Unit tests for the sources framework."""
import logging
import unittest
import apache_beam as beam
from apache_beam.io import iobase
from apache_beam.io import range_trackers
from apache_beam.io import source_test_utils
from apache_beam.io.concat_source import ConcatSource
from apache_beam.testing.test_pipeline import TestPipeline
from apache_beam.testing.util import assert_that
from apache_beam.testing.util import equal_to
__all__ = ['RangeSource']

class RangeSource(iobase.BoundedSource):
    __hash__ = None

    def __init__(self, start, end, split_freq=1):
        if False:
            while True:
                i = 10
        assert start <= end
        self._start = start
        self._end = end
        self._split_freq = split_freq

    def _normalize(self, start_position, end_position):
        if False:
            return 10
        return (self._start if start_position is None else start_position, self._end if end_position is None else end_position)

    def _round_up(self, index):
        if False:
            while True:
                i = 10
        'Rounds up to the nearest mulitple of split_freq.'
        return index - index % -self._split_freq

    def estimate_size(self):
        if False:
            while True:
                i = 10
        return self._end - self._start

    def split(self, desired_bundle_size, start_position=None, end_position=None):
        if False:
            return 10
        (start, end) = self._normalize(start_position, end_position)
        for sub_start in range(start, end, desired_bundle_size):
            sub_end = min(self._end, sub_start + desired_bundle_size)
            yield iobase.SourceBundle(sub_end - sub_start, RangeSource(sub_start, sub_end, self._split_freq), sub_start, sub_end)

    def get_range_tracker(self, start_position, end_position):
        if False:
            print('Hello World!')
        (start, end) = self._normalize(start_position, end_position)
        return range_trackers.OffsetRangeTracker(start, end)

    def read(self, range_tracker):
        if False:
            return 10
        for k in range(self._round_up(range_tracker.start_position()), self._round_up(range_tracker.stop_position())):
            if k % self._split_freq == 0:
                if not range_tracker.try_claim(k):
                    return
            yield k

    def __eq__(self, other):
        if False:
            for i in range(10):
                print('nop')
        return type(self) == type(other) and self._start == other._start and (self._end == other._end) and (self._split_freq == other._split_freq)

class ConcatSourceTest(unittest.TestCase):

    def test_range_source(self):
        if False:
            return 10
        source_test_utils.assert_split_at_fraction_exhaustive(RangeSource(0, 10, 3))

    def test_conact_source(self):
        if False:
            for i in range(10):
                print('nop')
        source = ConcatSource([RangeSource(0, 4), RangeSource(4, 8), RangeSource(8, 12), RangeSource(12, 16)])
        self.assertEqual(list(source.read(source.get_range_tracker())), list(range(16)))
        self.assertEqual(list(source.read(source.get_range_tracker((1, None), (2, 10)))), list(range(4, 10)))
        range_tracker = source.get_range_tracker(None, None)
        self.assertEqual(range_tracker.position_at_fraction(0), (0, 0))
        self.assertEqual(range_tracker.position_at_fraction(0.5), (2, 8))
        self.assertEqual(range_tracker.position_at_fraction(0.625), (2, 10))
        self.assertEqual(range_tracker.try_claim((0, None)), True)
        self.assertEqual(range_tracker.sub_range_tracker(0).try_claim(2), True)
        self.assertEqual(range_tracker.fraction_consumed(), 0.125)
        self.assertEqual(range_tracker.try_claim((1, None)), True)
        self.assertEqual(range_tracker.sub_range_tracker(1).try_claim(6), True)
        self.assertEqual(range_tracker.fraction_consumed(), 0.375)
        self.assertEqual(range_tracker.try_split((0, 1)), None)
        self.assertEqual(range_tracker.try_split((1, 5)), None)
        self.assertEqual(range_tracker.try_split((3, 14)), ((3, None), 0.75))
        self.assertEqual(range_tracker.try_claim((3, None)), False)
        self.assertEqual(range_tracker.sub_range_tracker(1).try_claim(7), True)
        self.assertEqual(range_tracker.try_claim((2, None)), True)
        self.assertEqual(range_tracker.sub_range_tracker(2).try_claim(9), True)
        self.assertEqual(range_tracker.try_split((2, 8)), None)
        self.assertEqual(range_tracker.try_split((2, 11)), ((2, 11), 11.0 / 12))
        self.assertEqual(range_tracker.sub_range_tracker(2).try_claim(10), True)
        self.assertEqual(range_tracker.sub_range_tracker(2).try_claim(11), False)

    def test_fraction_consumed_at_end(self):
        if False:
            print('Hello World!')
        source = ConcatSource([RangeSource(0, 2), RangeSource(2, 4)])
        range_tracker = source.get_range_tracker((2, None), None)
        self.assertEqual(range_tracker.fraction_consumed(), 1.0)

    def test_estimate_size(self):
        if False:
            return 10
        source = ConcatSource([RangeSource(0, 10), RangeSource(10, 100), RangeSource(100, 1000)])
        self.assertEqual(source.estimate_size(), 1000)

    def test_position_at_fration(self):
        if False:
            for i in range(10):
                print('nop')
        ranges = [(0, 4), (4, 16), (16, 24), (24, 32)]
        source = ConcatSource([iobase.SourceBundle((range[1] - range[0]) / 32.0, RangeSource(*range), None, None) for range in ranges])
        range_tracker = source.get_range_tracker()
        self.assertEqual(range_tracker.position_at_fraction(0), (0, 0))
        self.assertEqual(range_tracker.position_at_fraction(0.01), (0, 1))
        self.assertEqual(range_tracker.position_at_fraction(0.1), (0, 4))
        self.assertEqual(range_tracker.position_at_fraction(0.125), (1, 4))
        self.assertEqual(range_tracker.position_at_fraction(0.2), (1, 7))
        self.assertEqual(range_tracker.position_at_fraction(0.7), (2, 23))
        self.assertEqual(range_tracker.position_at_fraction(0.75), (3, 24))
        self.assertEqual(range_tracker.position_at_fraction(0.8), (3, 26))
        self.assertEqual(range_tracker.position_at_fraction(1), (4, None))
        range_tracker = source.get_range_tracker((1, None), (3, None))
        self.assertEqual(range_tracker.position_at_fraction(0), (1, 4))
        self.assertEqual(range_tracker.position_at_fraction(0.01), (1, 5))
        self.assertEqual(range_tracker.position_at_fraction(0.5), (1, 14))
        self.assertEqual(range_tracker.position_at_fraction(0.599), (1, 16))
        self.assertEqual(range_tracker.position_at_fraction(0.601), (2, 17))
        self.assertEqual(range_tracker.position_at_fraction(1), (3, None))

    def test_empty_source(self):
        if False:
            return 10
        read_all = source_test_utils.read_from_source
        empty = RangeSource(0, 0)
        self.assertEqual(read_all(ConcatSource([])), [])
        self.assertEqual(read_all(ConcatSource([empty])), [])
        self.assertEqual(read_all(ConcatSource([empty, empty])), [])
        range10 = RangeSource(0, 10)
        self.assertEqual(read_all(ConcatSource([range10]), (0, None), (0, 0)), [])
        self.assertEqual(read_all(ConcatSource([range10]), (0, 10), (1, None)), [])
        self.assertEqual(read_all(ConcatSource([range10, range10]), (0, 10), (1, 0)), [])

    def test_single_source(self):
        if False:
            print('Hello World!')
        read_all = source_test_utils.read_from_source
        range10 = RangeSource(0, 10)
        self.assertEqual(read_all(ConcatSource([range10])), list(range(10)))
        self.assertEqual(read_all(ConcatSource([range10]), (0, 5)), list(range(5, 10)))
        self.assertEqual(read_all(ConcatSource([range10]), None, (0, 5)), list(range(5)))

    def test_source_with_empty_ranges(self):
        if False:
            for i in range(10):
                print('nop')
        read_all = source_test_utils.read_from_source
        empty = RangeSource(0, 0)
        self.assertEqual(read_all(empty), [])
        range10 = RangeSource(0, 10)
        self.assertEqual(read_all(ConcatSource([empty, empty, range10])), list(range(10)))
        self.assertEqual(read_all(ConcatSource([empty, range10, empty])), list(range(10)))
        self.assertEqual(read_all(ConcatSource([range10, empty, range10, empty])), list(range(10)) + list(range(10)))

    def test_source_with_empty_ranges_exhastive(self):
        if False:
            return 10
        empty = RangeSource(0, 0)
        source = ConcatSource([empty, RangeSource(0, 10), empty, empty, RangeSource(10, 13), RangeSource(13, 17), empty])
        source_test_utils.assert_split_at_fraction_exhaustive(source)

    def test_run_concat_direct(self):
        if False:
            print('Hello World!')
        source = ConcatSource([RangeSource(0, 10), RangeSource(10, 100), RangeSource(100, 1000)])
        with TestPipeline() as pipeline:
            pcoll = pipeline | beam.io.Read(source)
            assert_that(pcoll, equal_to(list(range(1000))))

    def test_conact_source_exhaustive(self):
        if False:
            for i in range(10):
                print('nop')
        source = ConcatSource([RangeSource(0, 10), RangeSource(100, 110), RangeSource(1000, 1010)])
        source_test_utils.assert_split_at_fraction_exhaustive(source)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()