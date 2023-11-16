"""Unit tests for the range_trackers module."""
import copy
import logging
import math
import unittest
from typing import Optional
from typing import Union
from apache_beam.io import range_trackers

class OffsetRangeTrackerTest(unittest.TestCase):

    def test_try_return_record_simple_sparse(self):
        if False:
            while True:
                i = 10
        tracker = range_trackers.OffsetRangeTracker(100, 200)
        self.assertTrue(tracker.try_claim(110))
        self.assertTrue(tracker.try_claim(140))
        self.assertTrue(tracker.try_claim(183))
        self.assertFalse(tracker.try_claim(210))

    def test_try_return_record_simple_dense(self):
        if False:
            for i in range(10):
                print('nop')
        tracker = range_trackers.OffsetRangeTracker(3, 6)
        self.assertTrue(tracker.try_claim(3))
        self.assertTrue(tracker.try_claim(4))
        self.assertTrue(tracker.try_claim(5))
        self.assertFalse(tracker.try_claim(6))

    def test_try_claim_update_last_attempt(self):
        if False:
            while True:
                i = 10
        tracker = range_trackers.OffsetRangeTracker(1, 2)
        self.assertTrue(tracker.try_claim(1))
        self.assertEqual(1, tracker.last_attempted_record_start)
        self.assertFalse(tracker.try_claim(3))
        self.assertEqual(3, tracker.last_attempted_record_start)
        self.assertFalse(tracker.try_claim(6))
        self.assertEqual(6, tracker.last_attempted_record_start)
        with self.assertRaises(Exception):
            tracker.try_claim(6)

    def test_set_current_position(self):
        if False:
            for i in range(10):
                print('nop')
        tracker = range_trackers.OffsetRangeTracker(0, 6)
        self.assertTrue(tracker.try_claim(2))
        with self.assertRaises(Exception):
            tracker.set_current_position(1)
        self.assertFalse(tracker.try_claim(10))
        tracker.set_current_position(11)
        self.assertEqual(10, tracker.last_attempted_record_start)
        self.assertEqual(11, tracker.last_record_start)

    def test_try_return_record_continuous_until_split_point(self):
        if False:
            while True:
                i = 10
        tracker = range_trackers.OffsetRangeTracker(9, 18)
        self.assertTrue(tracker.try_claim(10))
        tracker.set_current_position(12)
        tracker.set_current_position(14)
        self.assertTrue(tracker.try_claim(16))
        tracker.set_current_position(18)
        tracker.set_current_position(20)
        self.assertFalse(tracker.try_claim(22))

    def test_split_at_offset_fails_if_unstarted(self):
        if False:
            print('Hello World!')
        tracker = range_trackers.OffsetRangeTracker(100, 200)
        self.assertFalse(tracker.try_split(150))

    def test_split_at_offset(self):
        if False:
            return 10
        tracker = range_trackers.OffsetRangeTracker(100, 200)
        self.assertTrue(tracker.try_claim(110))
        self.assertFalse(tracker.try_split(109))
        self.assertFalse(tracker.try_split(110))
        self.assertFalse(tracker.try_split(200))
        self.assertFalse(tracker.try_split(210))
        self.assertTrue(copy.copy(tracker).try_split(111))
        self.assertTrue(copy.copy(tracker).try_split(129))
        self.assertTrue(copy.copy(tracker).try_split(130))
        self.assertTrue(copy.copy(tracker).try_split(131))
        self.assertTrue(copy.copy(tracker).try_split(150))
        self.assertTrue(copy.copy(tracker).try_split(199))
        self.assertTrue(tracker.try_split(170))
        self.assertTrue(tracker.try_split(150))
        self.assertTrue(copy.copy(tracker).try_claim(135))
        self.assertTrue(copy.copy(tracker).try_claim(135))
        self.assertTrue(copy.copy(tracker).try_claim(149))
        self.assertFalse(tracker.try_claim(150))
        self.assertFalse(tracker.try_claim(151))
        tracker.set_current_position(152)
        tracker.set_current_position(160)
        tracker.set_current_position(171)

    def test_get_position_for_fraction_dense(self):
        if False:
            while True:
                i = 10
        tracker = range_trackers.OffsetRangeTracker(3, 6)
        self.assertTrue(isinstance(tracker.position_at_fraction(0.0), int))
        self.assertEqual(3, tracker.position_at_fraction(0.0))
        self.assertEqual(4, tracker.position_at_fraction(1.0 / 6))
        self.assertEqual(4, tracker.position_at_fraction(0.333))
        self.assertEqual(5, tracker.position_at_fraction(0.334))
        self.assertEqual(5, tracker.position_at_fraction(0.666))
        self.assertEqual(6, tracker.position_at_fraction(0.667))

    def test_get_fraction_consumed_dense(self):
        if False:
            while True:
                i = 10
        tracker = range_trackers.OffsetRangeTracker(3, 6)
        self.assertEqual(0, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(3))
        self.assertEqual(0.0, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(4))
        self.assertEqual(1.0 / 3, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(5))
        self.assertEqual(2.0 / 3, tracker.fraction_consumed())
        tracker.set_current_position(6)
        self.assertEqual(1.0, tracker.fraction_consumed())
        tracker.set_current_position(7)
        self.assertFalse(tracker.try_claim(7))

    def test_get_fraction_consumed_sparse(self):
        if False:
            return 10
        tracker = range_trackers.OffsetRangeTracker(100, 200)
        self.assertEqual(0, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(110))
        self.assertEqual(0.1, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(150))
        self.assertEqual(0.5, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(195))
        self.assertEqual(0.95, tracker.fraction_consumed())

    def test_everything_with_unbounded_range(self):
        if False:
            print('Hello World!')
        tracker = range_trackers.OffsetRangeTracker(100, range_trackers.OffsetRangeTracker.OFFSET_INFINITY)
        self.assertTrue(tracker.try_claim(150))
        self.assertTrue(tracker.try_claim(250))
        with self.assertRaises(Exception):
            tracker.position_at_fraction(0.5)

    def test_try_return_first_record_not_split_point(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(Exception):
            range_trackers.OffsetRangeTracker(100, 200).set_current_position(120)

    def test_try_return_record_non_monotonic(self):
        if False:
            return 10
        tracker = range_trackers.OffsetRangeTracker(100, 200)
        self.assertTrue(tracker.try_claim(120))
        with self.assertRaises(Exception):
            tracker.try_claim(110)

    def test_try_split_points(self):
        if False:
            i = 10
            return i + 15
        tracker = range_trackers.OffsetRangeTracker(100, 400)

        def dummy_callback(stop_position):
            if False:
                print('Hello World!')
            return int(stop_position // 5)
        tracker.set_split_points_unclaimed_callback(dummy_callback)
        self.assertEqual(tracker.split_points(), (0, 81))
        self.assertTrue(tracker.try_claim(120))
        self.assertEqual(tracker.split_points(), (0, 81))
        self.assertTrue(tracker.try_claim(140))
        self.assertEqual(tracker.split_points(), (1, 81))
        tracker.try_split(200)
        self.assertEqual(tracker.split_points(), (1, 41))
        self.assertTrue(tracker.try_claim(150))
        self.assertEqual(tracker.split_points(), (2, 41))
        self.assertTrue(tracker.try_claim(180))
        self.assertEqual(tracker.split_points(), (3, 41))
        self.assertFalse(tracker.try_claim(210))
        self.assertEqual(tracker.split_points(), (3, 41))

class OrderedPositionRangeTrackerTest(unittest.TestCase):

    class DoubleRangeTracker(range_trackers.OrderedPositionRangeTracker):

        @staticmethod
        def fraction_to_position(fraction, start, end):
            if False:
                print('Hello World!')
            return start + (end - start) * fraction

        @staticmethod
        def position_to_fraction(pos, start, end):
            if False:
                return 10
            return float(pos - start) / (end - start)

    def test_try_claim(self):
        if False:
            i = 10
            return i + 15
        tracker = self.DoubleRangeTracker(10, 20)
        self.assertTrue(tracker.try_claim(10))
        self.assertTrue(tracker.try_claim(15))
        self.assertFalse(tracker.try_claim(20))
        self.assertFalse(tracker.try_claim(25))

    def test_fraction_consumed(self):
        if False:
            while True:
                i = 10
        tracker = self.DoubleRangeTracker(10, 20)
        self.assertEqual(0, tracker.fraction_consumed())
        tracker.try_claim(10)
        self.assertEqual(0, tracker.fraction_consumed())
        tracker.try_claim(15)
        self.assertEqual(0.5, tracker.fraction_consumed())
        tracker.try_claim(17)
        self.assertEqual(0.7, tracker.fraction_consumed())
        tracker.try_claim(25)
        self.assertEqual(0.7, tracker.fraction_consumed())

    def test_try_split(self):
        if False:
            for i in range(10):
                print('nop')
        tracker = self.DoubleRangeTracker(10, 20)
        tracker.try_claim(15)
        self.assertEqual(0.5, tracker.fraction_consumed())
        self.assertEqual((18, 0.8), tracker.try_split(18))
        self.assertEqual(0.625, tracker.fraction_consumed())
        self.assertTrue(tracker.try_claim(17))
        self.assertIsNone(tracker.try_split(16))
        self.assertFalse(tracker.try_claim(18))
        self.assertFalse(tracker.try_claim(19))

    def test_claim_order(self):
        if False:
            for i in range(10):
                print('nop')
        tracker = self.DoubleRangeTracker(10, 20)
        tracker.try_claim(12)
        tracker.try_claim(15)
        with self.assertRaises(ValueError):
            tracker.try_claim(13)

    def test_out_of_range(self):
        if False:
            for i in range(10):
                print('nop')
        tracker = self.DoubleRangeTracker(10, 20)
        with self.assertRaises(ValueError):
            tracker.try_claim(-5)
        self.assertFalse(tracker.try_split(-5))
        self.assertFalse(tracker.try_split(10))
        self.assertFalse(tracker.try_split(25))
        tracker.try_split(15)
        self.assertFalse(tracker.try_split(17))
        self.assertFalse(tracker.try_split(15))
        self.assertTrue(tracker.try_split(14))

class UnsplittableRangeTrackerTest(unittest.TestCase):

    def test_try_claim(self):
        if False:
            for i in range(10):
                print('nop')
        tracker = range_trackers.UnsplittableRangeTracker(range_trackers.OffsetRangeTracker(100, 200))
        self.assertTrue(tracker.try_claim(110))
        self.assertTrue(tracker.try_claim(140))
        self.assertTrue(tracker.try_claim(183))
        self.assertFalse(tracker.try_claim(210))

    def test_try_split_fails(self):
        if False:
            return 10
        tracker = range_trackers.UnsplittableRangeTracker(range_trackers.OffsetRangeTracker(100, 200))
        self.assertTrue(tracker.try_claim(110))
        self.assertFalse(tracker.try_split(109))
        self.assertFalse(tracker.try_split(210))
        self.assertFalse(copy.copy(tracker).try_split(111))
        self.assertFalse(copy.copy(tracker).try_split(130))
        self.assertFalse(copy.copy(tracker).try_split(199))

class LexicographicKeyRangeTrackerTest(unittest.TestCase):
    """Tests of LexicographicKeyRangeTracker."""
    key_to_fraction = range_trackers.LexicographicKeyRangeTracker.position_to_fraction
    fraction_to_key = range_trackers.LexicographicKeyRangeTracker.fraction_to_position

    def _check(self, fraction: Optional[float]=None, key: Union[bytes, str]=None, start: Union[bytes, str]=None, end: Union[bytes, str]=None, delta: float=0.0):
        if False:
            for i in range(10):
                print('nop')
        assert key is not None or fraction is not None
        if fraction is None:
            fraction = self.key_to_fraction(key, start, end)
        elif key is None:
            key = self.fraction_to_key(fraction, start, end)
        if key is None and end is None and (fraction == 1):
            computed_fraction = 1
        else:
            computed_fraction = self.key_to_fraction(key, start, end)
        computed_key = self.fraction_to_key(fraction, start, end)
        if delta:
            self.assertAlmostEqual(computed_fraction, fraction, delta=delta, places=None, msg=str(locals()))
        else:
            self.assertEqual(computed_fraction, fraction, str(locals()))
        self.assertEqual(computed_key, key, str(locals()))

    def test_key_to_fraction_no_endpoints(self):
        if False:
            i = 10
            return i + 15
        self._check(key=b'\x07', fraction=7 / 256.0)
        self._check(key=b'\xff', fraction=255 / 256.0)
        self._check(key=b'\x01\x02\x03', fraction=(2 ** 16 + 2 ** 9 + 3) / 2.0 ** 24)
        self._check(key=b'UUUUUUT', fraction=1 / 3)
        self._check(key=b'3333334', fraction=1 / 5)
        self._check(key=b'$\x92I$\x92I$', fraction=1 / 7, delta=0.001)
        self._check(key=b'\x01\x02\x03', fraction=(2 ** 16 + 2 ** 9 + 3) / 2.0 ** 24)

    def test_key_to_fraction(self):
        if False:
            for i in range(10):
                print('nop')
        self._check(end=b'eeeeee', fraction=0.0)
        self._check(end='eeeeee', fraction=0.0)
        self._check(key=b'bbbbbb', start=b'aaaaaa', end=b'eeeeee')
        self._check(key='bbbbbb', start='aaaaaa', end='eeeeee')
        self._check(key=b'eeeeee', end=b'eeeeee', fraction=1.0)
        self._check(key='eeeeee', end='eeeeee', fraction=1.0)
        self._check(key=b'\x19YYYYY@', end=b'eeeeee', fraction=0.25)
        self._check(key=b'2\xb2\xb2\xb2\xb2\xb2\x80', end='eeeeee', fraction=0.5)
        self._check(key=b'L\x0c\x0c\x0c\x0c\x0b\xc0', end=b'eeeeee', fraction=0.75)
        self._check(key=b'\x87', start=b'\x80', fraction=7 / 128.0)
        self._check(key=b'\x07', end=b'\x10', fraction=7 / 16.0)
        self._check(key=b'G', start=b'@', end=b'\x80', fraction=7 / 64.0)
        self._check(key=b'G\x80', start=b'@', end=b'\x80', fraction=15 / 128.0)
        self._check(key='aaaaaa', start='aaaaaa', end='eeeeee', fraction=0.0)
        self._check(key='bbbbbb', start='aaaaaa', end='eeeeee', fraction=0.25)
        self._check(key='cccccc', start='aaaaaa', end='eeeeee', fraction=0.5)
        self._check(key='dddddd', start='aaaaaa', end='eeeeee', fraction=0.75)
        self._check(key='eeeeee', start='aaaaaa', end='eeeeee', fraction=1.0)

    def test_key_to_fraction_common_prefix(self):
        if False:
            i = 10
            return i + 15
        self._check(key=b'a' * 100 + b'b', start=b'a' * 100 + b'a', end=b'a' * 100 + b'c', fraction=0.5)
        self._check(key=b'a' * 100 + b'b', start=b'a' * 100 + b'a', end=b'a' * 100 + b'e', fraction=0.25)
        self._check(key=b'\xff' * 100 + b'@', start=b'\xff' * 100, end=None, fraction=0.25)
        self._check(key=b'foob', start=b'fooa\xff\xff\xff\xff\xff\xff\xff\xff\xfe', end=b'foob\x00\x00\x00\x00\x00\x00\x00\x00\x02', fraction=0.5)
        self._check(key='a' * 100 + 'a', start='a' * 100 + 'a', end='a' * 100 + 'e', fraction=0.0)
        self._check(key='a' * 100 + 'b', start='a' * 100 + 'a', end='a' * 100 + 'e', fraction=0.25)
        self._check(key='a' * 100 + 'c', start='a' * 100 + 'a', end='a' * 100 + 'e', fraction=0.5)
        self._check(key='a' * 100 + 'd', start='a' * 100 + 'a', end='a' * 100 + 'e', fraction=0.75)
        self._check(key='a' * 100 + 'e', start='a' * 100 + 'a', end='a' * 100 + 'e', fraction=1.0)

    def test_tiny(self):
        if False:
            while True:
                i = 10
        self._check(fraction=0.5 ** 20, key=b'\x00\x00\x10')
        self._check(fraction=0.5 ** 20, start=b'a', end=b'b', key=b'a\x00\x00\x10')
        self._check(fraction=0.5 ** 20, start=b'a', end=b'c', key=b'a\x00\x00 ')
        self._check(fraction=0.5 ** 20, start=b'xy_a', end=b'xy_c', key=b'xy_a\x00\x00 ')
        self._check(fraction=0.5 ** 20, start=b'\xff\xff\x80', key=b'\xff\xff\x80\x00\x08')
        self._check(fraction=0.5 ** 20 / 3, start=b'xy_a', end=b'xy_c', key=b'xy_a\x00\x00\n\xaa\xaa\xaa\xaa\xaa', delta=1e-15)
        self._check(fraction=0.5 ** 100, key=b'\x00' * 12 + b'\x10')
        self._check(fraction=0.5 ** 20, start='a', end='b', key='a\x00\x00\x10')
        self._check(fraction=0.5 ** 20, start='a', end='c', key='a\x00\x00 ')
        self._check(fraction=0.5 ** 20, start='xy_a', end='xy_c', key='xy_a\x00\x00 ')

    def test_lots(self):
        if False:
            return 10
        for fraction in (0, 1, 0.5, 0.75, 7.0 / 512, 1 - 7.0 / 4096):
            self._check(fraction)
            self._check(fraction, start=b'\x01')
            self._check(fraction, end=b'\xf0')
            self._check(fraction, start=b'0x75', end=b'v')
            self._check(fraction, start=b'0x75', end=b'w')
            self._check(fraction, start=b'0x75', end=b'x')
            self._check(fraction, start=b'a' * 100 + b'\x80', end=b'a' * 100 + b'\x81')
            self._check(fraction, start=b'a' * 101 + b'\x80', end=b'a' * 101 + b'\x81')
            self._check(fraction, start=b'a' * 102 + b'\x80', end=b'a' * 102 + b'\x81')
        for fraction in (0.3, 1 / 3.0, 1 / math.e, 0.001, 1e-30, 0.99, 0.999999):
            self._check(fraction, delta=1e-14)
            self._check(fraction, start=b'\x01', delta=1e-14)
            self._check(fraction, end=b'\xf0', delta=1e-14)
            self._check(fraction, start=b'0x75', end=b'v', delta=1e-14)
            self._check(fraction, start=b'0x75', end=b'w', delta=1e-14)
            self._check(fraction, start=b'0x75', end=b'x', delta=1e-14)
            self._check(fraction, start=b'a' * 100 + b'\x80', end=b'a' * 100 + b'\x81', delta=1e-14)

    def test_good_prec(self):
        if False:
            while True:
                i = 10
        self._check(1 / math.e, start='AAAAAAA', end='zzzzzzz', key='VNg/ot\x82', delta=1e-14)
        self._check(1 / math.e, start=b'abc_abc', end=b'abc_xyz', key=b'abc_i\xe0\xf4\x84\x86\x99\x96', delta=1e-15)
        self._check(1 / math.e, start=b'abcd_abc\x00\x00\x00\x00\x00_______________abc', end=b'abcd_xyz\x00\x00\x00\x00\x00\x00_______________abc', key=b'abcd_i\xe0\xf4\x84\x86\x99\x96', delta=1e-15)
        self._check(1e-20 / math.e, start=b'abcd_abc', end=b'abcd_xyz', key=b'abcd_abc\x00\x00\x00\x00\x00\x01\x91#\x172N\xbb', delta=1e-35)
if __name__ == '__main__':
    logging.getLogger().setLevel(logging.INFO)
    unittest.main()