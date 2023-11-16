import datetime
import math
import sys
import unittest
from mock import patch
from apache_beam.io.gcp.datastore.v1new.rampup_throttling_fn import RampupThrottlingFn
DATE_ZERO = datetime.datetime(year=1970, month=1, day=1, tzinfo=datetime.timezone.utc)

class _RampupDelayException(Exception):
    pass

class RampupThrottlerTransformTest(unittest.TestCase):

    @patch('datetime.datetime')
    @patch('time.sleep')
    def test_rampup_throttling(self, mock_sleep, mock_datetime):
        if False:
            return 10
        mock_datetime.now.return_value = DATE_ZERO
        throttling_fn = RampupThrottlingFn(num_workers=1)
        rampup_schedule = [(DATE_ZERO + datetime.timedelta(seconds=0), 500), (DATE_ZERO + datetime.timedelta(milliseconds=1), 0), (DATE_ZERO + datetime.timedelta(seconds=1), 500), (DATE_ZERO + datetime.timedelta(seconds=1, milliseconds=1), 0), (DATE_ZERO + datetime.timedelta(minutes=5), 500), (DATE_ZERO + datetime.timedelta(minutes=10), 750), (DATE_ZERO + datetime.timedelta(minutes=15), 1125), (DATE_ZERO + datetime.timedelta(minutes=30), 3796), (DATE_ZERO + datetime.timedelta(minutes=60), 43248)]
        mock_sleep.side_effect = _RampupDelayException()
        for (date, expected_budget) in rampup_schedule:
            mock_datetime.now.return_value = date
            for _ in range(expected_budget):
                next(throttling_fn.process(None))
            with self.assertRaises(_RampupDelayException):
                next(throttling_fn.process(None))

    def test_budget_overflow(self):
        if False:
            while True:
                i = 10
        throttling_fn = RampupThrottlingFn(num_workers=1)
        normal_date = DATE_ZERO + datetime.timedelta(minutes=2000)
        normal_budget = throttling_fn._calc_max_ops_budget(DATE_ZERO, normal_date)
        self.assertNotEqual(normal_budget, float('inf'))
        overflow_minutes = math.log(sys.float_info.max) / math.log(1.5) * 5
        overflow_date = DATE_ZERO + datetime.timedelta(minutes=overflow_minutes)
        overflow_budget = throttling_fn._calc_max_ops_budget(DATE_ZERO, overflow_date)
        self.assertEqual(overflow_budget, float('inf'))
if __name__ == '__main__':
    unittest.main()