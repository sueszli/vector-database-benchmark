"""Unit tests for the windowed_value."""
import copy
import itertools
import pickle
import unittest
from parameterized import parameterized
from parameterized import parameterized_class
from apache_beam.utils import windowed_value
from apache_beam.utils.timestamp import Timestamp

class WindowedValueTest(unittest.TestCase):

    def test_timestamps(self):
        if False:
            i = 10
            return i + 15
        wv = windowed_value.WindowedValue(None, 3, ())
        self.assertEqual(wv.timestamp, Timestamp.of(3))
        self.assertTrue(wv.timestamp is wv.timestamp)
        self.assertEqual(windowed_value.WindowedValue(None, -2.5, ()).timestamp, Timestamp.of(-2.5))

    def test_with_value(self):
        if False:
            i = 10
            return i + 15
        pane_info = windowed_value.PaneInfo(True, True, windowed_value.PaneInfoTiming.ON_TIME, 0, 0)
        wv = windowed_value.WindowedValue(1, 3, (), pane_info)
        self.assertEqual(wv.with_value(10), windowed_value.WindowedValue(10, 3, (), pane_info))

    def test_equality(self):
        if False:
            while True:
                i = 10
        self.assertEqual(windowed_value.WindowedValue(1, 3, ()), windowed_value.WindowedValue(1, 3, ()))
        self.assertNotEqual(windowed_value.WindowedValue(1, 3, ()), windowed_value.WindowedValue(100, 3, ()))
        self.assertNotEqual(windowed_value.WindowedValue(1, 3, ()), windowed_value.WindowedValue(1, 300, ()))
        self.assertNotEqual(windowed_value.WindowedValue(1, 3, ()), windowed_value.WindowedValue(1, 300, ((),)))
        self.assertNotEqual(windowed_value.WindowedValue(1, 3, ()), object())

    def test_hash(self):
        if False:
            while True:
                i = 10
        wv = windowed_value.WindowedValue(1, 3, ())
        wv_copy = copy.copy(wv)
        self.assertFalse(wv is wv_copy)
        self.assertEqual({wv: 100}.get(wv_copy), 100)

    def test_pickle(self):
        if False:
            i = 10
            return i + 15
        pane_info = windowed_value.PaneInfo(True, True, windowed_value.PaneInfoTiming.ON_TIME, 0, 0)
        wv = windowed_value.WindowedValue(1, 3, (), pane_info)
        self.assertTrue(pickle.loads(pickle.dumps(wv)) == wv)
WINDOWED_BATCH_INSTANCES = [windowed_value.HomogeneousWindowedBatch.of(None, 3, (), windowed_value.PANE_INFO_UNKNOWN), windowed_value.HomogeneousWindowedBatch.of(None, 3, (), windowed_value.PaneInfo(True, False, windowed_value.PaneInfoTiming.ON_TIME, 0, 0))]

class WindowedBatchTest(unittest.TestCase):

    def test_homogeneous_windowed_batch_with_values(self):
        if False:
            print('Hello World!')
        pane_info = windowed_value.PaneInfo(True, True, windowed_value.PaneInfoTiming.ON_TIME, 0, 0)
        wb = windowed_value.HomogeneousWindowedBatch.of(['foo', 'bar'], 6, (), pane_info)
        self.assertEqual(wb.with_values(['baz', 'foo']), windowed_value.HomogeneousWindowedBatch.of(['baz', 'foo'], 6, (), pane_info))

    def test_homogeneous_windowed_batch_as_windowed_values(self):
        if False:
            print('Hello World!')
        pane_info = windowed_value.PaneInfo(True, True, windowed_value.PaneInfoTiming.ON_TIME, 0, 0)
        wb = windowed_value.HomogeneousWindowedBatch.of(['foo', 'bar'], 3, (), pane_info)
        self.assertEqual(list(wb.as_windowed_values(iter)), [windowed_value.WindowedValue('foo', 3, (), pane_info), windowed_value.WindowedValue('bar', 3, (), pane_info)])

    @parameterized.expand(itertools.combinations(WINDOWED_BATCH_INSTANCES, 2))
    def test_inequality(self, left_wb, right_wb):
        if False:
            print('Hello World!')
        self.assertNotEqual(left_wb, right_wb)

    def test_equals_different_type(self):
        if False:
            return 10
        wb = windowed_value.HomogeneousWindowedBatch.of(None, 3, (), windowed_value.PANE_INFO_UNKNOWN)
        self.assertNotEqual(wb, object())

    def test_homogeneous_from_windowed_values(self):
        if False:
            return 10
        pane_info = windowed_value.PaneInfo(True, True, windowed_value.PaneInfoTiming.ON_TIME, 0, 0)
        windowed_values = [windowed_value.WindowedValue('foofoo', 3, (), pane_info), windowed_value.WindowedValue('foobar', 6, (), pane_info), windowed_value.WindowedValue('foobaz', 9, (), pane_info), windowed_value.WindowedValue('barfoo', 3, (), pane_info), windowed_value.WindowedValue('barbar', 6, (), pane_info), windowed_value.WindowedValue('barbaz', 9, (), pane_info), windowed_value.WindowedValue('bazfoo', 3, (), pane_info), windowed_value.WindowedValue('bazbar', 6, (), pane_info), windowed_value.WindowedValue('bazbaz', 9, (), pane_info)]
        self.assertEqual(list(windowed_value.WindowedBatch.from_windowed_values(windowed_values, produce_fn=list)), [windowed_value.HomogeneousWindowedBatch.of(['foofoo', 'barfoo', 'bazfoo'], 3, (), pane_info), windowed_value.HomogeneousWindowedBatch.of(['foobar', 'barbar', 'bazbar'], 6, (), pane_info), windowed_value.HomogeneousWindowedBatch.of(['foobaz', 'barbaz', 'bazbaz'], 9, (), pane_info)])

@parameterized_class(('wb',), [(wb,) for wb in WINDOWED_BATCH_INSTANCES])
class WindowedBatchUtilitiesTest(unittest.TestCase):

    def test_hash(self):
        if False:
            i = 10
            return i + 15
        wb_copy = copy.copy(self.wb)
        self.assertFalse(self.wb is wb_copy)
        self.assertEqual({self.wb: 100}.get(wb_copy), 100)

    def test_pickle(self):
        if False:
            while True:
                i = 10
        self.assertTrue(pickle.loads(pickle.dumps(self.wb)) == self.wb)
if __name__ == '__main__':
    unittest.main()