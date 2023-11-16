import unittest
from unittest.mock import MagicMock
from datetime import datetime, timezone
import pytz
from superagi.helper.calendar_date import CalendarDate

class TestCalendarDate(unittest.TestCase):

    def setUp(self):
        if False:
            return 10
        self.cd = CalendarDate()
        self.service = MagicMock()
        self.service.calendars().get().execute.return_value = {'timeZone': 'Asia/Kolkata'}

    def test_get_time_zone(self):
        if False:
            i = 10
            return i + 15
        time_zone = self.cd._get_time_zone(self.service)
        self.assertEqual(time_zone, 'Asia/Kolkata')

    def test_convert_to_utc(self):
        if False:
            for i in range(10):
                print('nop')
        local_datetime = datetime(2023, 1, 1)
        local_tz = pytz.timezone('US/Pacific')
        utc_datetime = self.cd._convert_to_utc(local_datetime, local_tz)
        expected_utc_datetime = datetime(2023, 1, 1, 8, 0)
        expected_utc_datetime = pytz.timezone('GMT').localize(expected_utc_datetime)
        assert utc_datetime == expected_utc_datetime

    def test_string_to_datetime(self):
        if False:
            while True:
                i = 10
        date_str = '2022-01-01'
        date_format = '%Y-%m-%d'
        date_obj = datetime.strptime(date_str, date_format)
        self.assertEqual(date_obj, self.cd._string_to_datetime(date_str, date_format))

    def test_localize_daterange(self):
        if False:
            print('Hello World!')
        (start_date, end_date) = ('2022-01-01', '2022-01-02')
        (start_time, end_time) = ('10:00:00', '12:00:00')
        local_tz = pytz.timezone('Asia/Kolkata')
        (start_datetime_utc, end_datetime_utc) = self.cd._localize_daterange(start_date, end_date, start_time, end_time, local_tz)
        self.assertEqual(start_datetime_utc, datetime(2022, 1, 1, 4, 30, tzinfo=timezone.utc))
        self.assertEqual(end_datetime_utc, datetime(2022, 1, 2, 6, 30, tzinfo=timezone.utc))

    def test_datetime_to_string(self):
        if False:
            while True:
                i = 10
        date_time = datetime(2022, 1, 1, 0, 0, 0)
        date_format = '%Y-%m-%d'
        date_str = '2022-01-01'
        self.assertEqual(date_str, self.cd._datetime_to_string(date_time, date_format))

    def test_get_date_utc(self):
        if False:
            i = 10
            return i + 15
        (start_date, end_date) = ('2022-01-01', '2022-01-02')
        (start_time, end_time) = ('10:00:00', '12:00:00')
        date_utc = {'start_datetime_utc': '2022-01-01T04:30:00.000000Z', 'end_datetime_utc': '2022-01-02T06:30:00.000000Z'}
        result = self.cd.get_date_utc(start_date, end_date, start_time, end_time, self.service)
        self.assertEqual(date_utc, result)

    def test_create_event_dates(self):
        if False:
            print('Hello World!')
        (start_date, end_date) = ('2022-01-01', '2022-01-02')
        (start_time, end_time) = ('10:00:00', '12:00:00')
        date_utc = {'start_datetime_utc': '2022-01-01T04:30:00.000000Z', 'end_datetime_utc': '2022-01-02T06:30:00.000000Z', 'timeZone': 'Asia/Kolkata'}
        result = self.cd.create_event_dates(self.service, start_date, start_time, end_date, end_time)
        self.assertEqual(date_utc, result)
if __name__ == '__main__':
    unittest.main()