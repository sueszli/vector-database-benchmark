import datetime
from unittest import TestCase
from unittest.mock import patch
from aiortc import clock

class ClockTest(TestCase):

    @patch('aiortc.clock.current_datetime')
    def test_current_ms(self, mock_now):
        if False:
            return 10
        mock_now.return_value = datetime.datetime(2018, 9, 11, tzinfo=datetime.timezone.utc)
        self.assertEqual(clock.current_ms(), 3745612800000)
        mock_now.return_value = datetime.datetime(2018, 9, 11, 0, 0, 1, tzinfo=datetime.timezone.utc)
        self.assertEqual(clock.current_ms(), 3745612801000)

    def test_datetime_from_ntp(self):
        if False:
            i = 10
            return i + 15
        dt = datetime.datetime(2018, 6, 28, 9, 3, 5, 423998, tzinfo=datetime.timezone.utc)
        self.assertEqual(clock.datetime_from_ntp(16059593044731306503), dt)

    def test_datetime_to_ntp(self):
        if False:
            return 10
        dt = datetime.datetime(2018, 6, 28, 9, 3, 5, 423998, tzinfo=datetime.timezone.utc)
        self.assertEqual(clock.datetime_to_ntp(dt), 16059593044731306503)