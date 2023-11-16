"""Unit tests for built-in WatermarkEstimators"""
import unittest
import mock
from apache_beam.io.iobase import WatermarkEstimator
from apache_beam.io.watermark_estimators import ManualWatermarkEstimator
from apache_beam.io.watermark_estimators import MonotonicWatermarkEstimator
from apache_beam.io.watermark_estimators import WalltimeWatermarkEstimator
from apache_beam.utils.timestamp import Duration
from apache_beam.utils.timestamp import Timestamp

class MonotonicWatermarkEstimatorTest(unittest.TestCase):

    def test_initialize_from_state(self):
        if False:
            while True:
                i = 10
        timestamp = Timestamp(10)
        watermark_estimator = MonotonicWatermarkEstimator(timestamp)
        self.assertIsInstance(watermark_estimator, WatermarkEstimator)
        self.assertEqual(watermark_estimator.get_estimator_state(), timestamp)

    def test_observe_timestamp(self):
        if False:
            return 10
        watermark_estimator = MonotonicWatermarkEstimator(Timestamp(10))
        watermark_estimator.observe_timestamp(Timestamp(15))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(15))
        watermark_estimator.observe_timestamp(Timestamp(20))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(20))
        watermark_estimator.observe_timestamp(Timestamp(20))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(20))
        watermark_estimator.observe_timestamp(Timestamp(10))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(20))

    def test_get_estimator_state(self):
        if False:
            while True:
                i = 10
        watermark_estimator = MonotonicWatermarkEstimator(Timestamp(10))
        self.assertEqual(watermark_estimator.get_estimator_state(), Timestamp(10))
        watermark_estimator.observe_timestamp(Timestamp(15))
        self.assertEqual(watermark_estimator.get_estimator_state(), Timestamp(10))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(15))
        self.assertEqual(watermark_estimator.get_estimator_state(), Timestamp(15))

class WalltimeWatermarkEstimatorTest(unittest.TestCase):

    @mock.patch('apache_beam.utils.timestamp.Timestamp.now')
    def test_initialization(self, mock_timestamp):
        if False:
            print('Hello World!')
        now_time = Timestamp.now() - Duration(10)
        mock_timestamp.side_effect = lambda : now_time
        watermark_estimator = WalltimeWatermarkEstimator()
        self.assertIsInstance(watermark_estimator, WatermarkEstimator)
        self.assertEqual(watermark_estimator.get_estimator_state(), now_time)

    def test_observe_timestamp(self):
        if False:
            i = 10
            return i + 15
        now_time = Timestamp.now() + Duration(10)
        watermark_estimator = WalltimeWatermarkEstimator(now_time)
        watermark_estimator.observe_timestamp(Timestamp(10))
        watermark_estimator.observe_timestamp(Timestamp(10))
        self.assertEqual(watermark_estimator.current_watermark(), now_time)

    def test_advance_watermark_with_incorrect_sys_clock(self):
        if False:
            print('Hello World!')
        initial_timestamp = Timestamp.now() + Duration(100)
        watermark_estimator = WalltimeWatermarkEstimator(initial_timestamp)
        self.assertEqual(watermark_estimator.current_watermark(), initial_timestamp)
        self.assertEqual(watermark_estimator.get_estimator_state(), initial_timestamp)

class ManualWatermarkEstimatorTest(unittest.TestCase):

    def test_initialization(self):
        if False:
            while True:
                i = 10
        watermark_estimator = ManualWatermarkEstimator(None)
        self.assertIsNone(watermark_estimator.get_estimator_state())
        self.assertIsNone(watermark_estimator.current_watermark())
        watermark_estimator = ManualWatermarkEstimator(Timestamp(10))
        self.assertEqual(watermark_estimator.get_estimator_state(), Timestamp(10))

    def test_set_watermark(self):
        if False:
            i = 10
            return i + 15
        watermark_estimator = ManualWatermarkEstimator(None)
        self.assertIsNone(watermark_estimator.current_watermark())
        watermark_estimator.observe_timestamp(Timestamp(10))
        self.assertIsNone(watermark_estimator.current_watermark())
        watermark_estimator.set_watermark(Timestamp(20))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(20))
        watermark_estimator.set_watermark(Timestamp(30))
        self.assertEqual(watermark_estimator.current_watermark(), Timestamp(30))
        with self.assertRaises(ValueError):
            watermark_estimator.set_watermark(Timestamp(25))
if __name__ == '__main__':
    unittest.main()