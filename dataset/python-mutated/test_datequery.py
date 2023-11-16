"""Test for dbcore's date-based queries.
"""
import time
import unittest
from datetime import datetime, timedelta
from test import _common
from beets.dbcore.query import DateInterval, DateQuery, InvalidQueryArgumentValueError, _parse_periods

def _date(string):
    if False:
        for i in range(10):
            print('nop')
    return datetime.strptime(string, '%Y-%m-%dT%H:%M:%S')

def _datepattern(datetimedate):
    if False:
        print('Hello World!')
    return datetimedate.strftime('%Y-%m-%dT%H:%M:%S')

class DateIntervalTest(unittest.TestCase):

    def test_year_precision_intervals(self):
        if False:
            return 10
        self.assertContains('2000..2001', '2000-01-01T00:00:00')
        self.assertContains('2000..2001', '2001-06-20T14:15:16')
        self.assertContains('2000..2001', '2001-12-31T23:59:59')
        self.assertExcludes('2000..2001', '1999-12-31T23:59:59')
        self.assertExcludes('2000..2001', '2002-01-01T00:00:00')
        self.assertContains('2000..', '2000-01-01T00:00:00')
        self.assertContains('2000..', '2099-10-11T00:00:00')
        self.assertExcludes('2000..', '1999-12-31T23:59:59')
        self.assertContains('..2001', '2001-12-31T23:59:59')
        self.assertExcludes('..2001', '2002-01-01T00:00:00')
        self.assertContains('-1d..1d', _datepattern(datetime.now()))
        self.assertExcludes('-2d..-1d', _datepattern(datetime.now()))

    def test_day_precision_intervals(self):
        if False:
            i = 10
            return i + 15
        self.assertContains('2000-06-20..2000-06-20', '2000-06-20T00:00:00')
        self.assertContains('2000-06-20..2000-06-20', '2000-06-20T10:20:30')
        self.assertContains('2000-06-20..2000-06-20', '2000-06-20T23:59:59')
        self.assertExcludes('2000-06-20..2000-06-20', '2000-06-19T23:59:59')
        self.assertExcludes('2000-06-20..2000-06-20', '2000-06-21T00:00:00')

    def test_month_precision_intervals(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertContains('1999-12..2000-02', '1999-12-01T00:00:00')
        self.assertContains('1999-12..2000-02', '2000-02-15T05:06:07')
        self.assertContains('1999-12..2000-02', '2000-02-29T23:59:59')
        self.assertExcludes('1999-12..2000-02', '1999-11-30T23:59:59')
        self.assertExcludes('1999-12..2000-02', '2000-03-01T00:00:00')

    def test_hour_precision_intervals(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertExcludes('2000-01-01T12..2000-01-01T13', '2000-01-01T11:59:59')
        self.assertContains('2000-01-01T12..2000-01-01T13', '2000-01-01T12:00:00')
        self.assertContains('2000-01-01T12..2000-01-01T13', '2000-01-01T12:30:00')
        self.assertContains('2000-01-01T12..2000-01-01T13', '2000-01-01T13:30:00')
        self.assertContains('2000-01-01T12..2000-01-01T13', '2000-01-01T13:59:59')
        self.assertExcludes('2000-01-01T12..2000-01-01T13', '2000-01-01T14:00:00')
        self.assertExcludes('2000-01-01T12..2000-01-01T13', '2000-01-01T14:30:00')
        self.assertContains('2008-12-01T22', '2008-12-01T22:30:00')
        self.assertExcludes('2008-12-01T22', '2008-12-01T23:30:00')

    def test_minute_precision_intervals(self):
        if False:
            return 10
        self.assertExcludes('2000-01-01T12:30..2000-01-01T12:31', '2000-01-01T12:29:59')
        self.assertContains('2000-01-01T12:30..2000-01-01T12:31', '2000-01-01T12:30:00')
        self.assertContains('2000-01-01T12:30..2000-01-01T12:31', '2000-01-01T12:30:30')
        self.assertContains('2000-01-01T12:30..2000-01-01T12:31', '2000-01-01T12:31:59')
        self.assertExcludes('2000-01-01T12:30..2000-01-01T12:31', '2000-01-01T12:32:00')

    def test_second_precision_intervals(self):
        if False:
            return 10
        self.assertExcludes('2000-01-01T12:30:50..2000-01-01T12:30:55', '2000-01-01T12:30:49')
        self.assertContains('2000-01-01T12:30:50..2000-01-01T12:30:55', '2000-01-01T12:30:50')
        self.assertContains('2000-01-01T12:30:50..2000-01-01T12:30:55', '2000-01-01T12:30:55')
        self.assertExcludes('2000-01-01T12:30:50..2000-01-01T12:30:55', '2000-01-01T12:30:56')

    def test_unbounded_endpoints(self):
        if False:
            return 10
        self.assertContains('..', date=datetime.max)
        self.assertContains('..', date=datetime.min)
        self.assertContains('..', '1000-01-01T00:00:00')

    def assertContains(self, interval_pattern, date_pattern=None, date=None):
        if False:
            while True:
                i = 10
        if date is None:
            date = _date(date_pattern)
        (start, end) = _parse_periods(interval_pattern)
        interval = DateInterval.from_periods(start, end)
        self.assertTrue(interval.contains(date))

    def assertExcludes(self, interval_pattern, date_pattern):
        if False:
            return 10
        date = _date(date_pattern)
        (start, end) = _parse_periods(interval_pattern)
        interval = DateInterval.from_periods(start, end)
        self.assertFalse(interval.contains(date))

def _parsetime(s):
    if False:
        while True:
            i = 10
    return time.mktime(datetime.strptime(s, '%Y-%m-%d %H:%M').timetuple())

class DateQueryTest(_common.LibTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self.i.added = _parsetime('2013-03-30 22:21')
        self.i.store()

    def test_single_month_match_fast(self):
        if False:
            while True:
                i = 10
        query = DateQuery('added', '2013-03')
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 1)

    def test_single_month_nonmatch_fast(self):
        if False:
            i = 10
            return i + 15
        query = DateQuery('added', '2013-04')
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 0)

    def test_single_month_match_slow(self):
        if False:
            return 10
        query = DateQuery('added', '2013-03')
        self.assertTrue(query.match(self.i))

    def test_single_month_nonmatch_slow(self):
        if False:
            return 10
        query = DateQuery('added', '2013-04')
        self.assertFalse(query.match(self.i))

    def test_single_day_match_fast(self):
        if False:
            print('Hello World!')
        query = DateQuery('added', '2013-03-30')
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 1)

    def test_single_day_nonmatch_fast(self):
        if False:
            while True:
                i = 10
        query = DateQuery('added', '2013-03-31')
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 0)

class DateQueryTestRelative(_common.LibTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self._now = datetime(2017, 12, 31, 22, 55, 4, 101332)
        self.i.added = _parsetime(self._now.strftime('%Y-%m-%d %H:%M'))
        self.i.store()

    def test_single_month_match_fast(self):
        if False:
            return 10
        query = DateQuery('added', self._now.strftime('%Y-%m'))
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 1)

    def test_single_month_nonmatch_fast(self):
        if False:
            while True:
                i = 10
        query = DateQuery('added', (self._now + timedelta(days=30)).strftime('%Y-%m'))
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 0)

    def test_single_month_match_slow(self):
        if False:
            for i in range(10):
                print('nop')
        query = DateQuery('added', self._now.strftime('%Y-%m'))
        self.assertTrue(query.match(self.i))

    def test_single_month_nonmatch_slow(self):
        if False:
            return 10
        query = DateQuery('added', (self._now + timedelta(days=30)).strftime('%Y-%m'))
        self.assertFalse(query.match(self.i))

    def test_single_day_match_fast(self):
        if False:
            for i in range(10):
                print('nop')
        query = DateQuery('added', self._now.strftime('%Y-%m-%d'))
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 1)

    def test_single_day_nonmatch_fast(self):
        if False:
            while True:
                i = 10
        query = DateQuery('added', (self._now + timedelta(days=1)).strftime('%Y-%m-%d'))
        matched = self.lib.items(query)
        self.assertEqual(len(matched), 0)

class DateQueryTestRelativeMore(_common.LibTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.i.added = _parsetime(datetime.now().strftime('%Y-%m-%d %H:%M'))
        self.i.store()

    def test_relative(self):
        if False:
            for i in range(10):
                print('nop')
        for timespan in ['d', 'w', 'm', 'y']:
            query = DateQuery('added', '-4' + timespan + '..+4' + timespan)
            matched = self.lib.items(query)
            self.assertEqual(len(matched), 1)

    def test_relative_fail(self):
        if False:
            while True:
                i = 10
        for timespan in ['d', 'w', 'm', 'y']:
            query = DateQuery('added', '-2' + timespan + '..-1' + timespan)
            matched = self.lib.items(query)
            self.assertEqual(len(matched), 0)

    def test_start_relative(self):
        if False:
            while True:
                i = 10
        for timespan in ['d', 'w', 'm', 'y']:
            query = DateQuery('added', '-4' + timespan + '..')
            matched = self.lib.items(query)
            self.assertEqual(len(matched), 1)

    def test_start_relative_fail(self):
        if False:
            while True:
                i = 10
        for timespan in ['d', 'w', 'm', 'y']:
            query = DateQuery('added', '4' + timespan + '..')
            matched = self.lib.items(query)
            self.assertEqual(len(matched), 0)

    def test_end_relative(self):
        if False:
            return 10
        for timespan in ['d', 'w', 'm', 'y']:
            query = DateQuery('added', '..+4' + timespan)
            matched = self.lib.items(query)
            self.assertEqual(len(matched), 1)

    def test_end_relative_fail(self):
        if False:
            i = 10
            return i + 15
        for timespan in ['d', 'w', 'm', 'y']:
            query = DateQuery('added', '..-4' + timespan)
            matched = self.lib.items(query)
            self.assertEqual(len(matched), 0)

class DateQueryConstructTest(unittest.TestCase):

    def test_long_numbers(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(InvalidQueryArgumentValueError):
            DateQuery('added', '1409830085..1412422089')

    def test_too_many_components(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(InvalidQueryArgumentValueError):
            DateQuery('added', '12-34-56-78')

    def test_invalid_date_query(self):
        if False:
            i = 10
            return i + 15
        q_list = ['2001-01-0a', '2001-0a', '200a', '2001-01-01..2001-01-0a', '2001-0a..2001-01', '200a..2002', '20aa..', '..2aa']
        for q in q_list:
            with self.assertRaises(InvalidQueryArgumentValueError):
                DateQuery('added', q)

    def test_datetime_uppercase_t_separator(self):
        if False:
            while True:
                i = 10
        date_query = DateQuery('added', '2000-01-01T12')
        self.assertEqual(date_query.interval.start, datetime(2000, 1, 1, 12))
        self.assertEqual(date_query.interval.end, datetime(2000, 1, 1, 13))

    def test_datetime_lowercase_t_separator(self):
        if False:
            return 10
        date_query = DateQuery('added', '2000-01-01t12')
        self.assertEqual(date_query.interval.start, datetime(2000, 1, 1, 12))
        self.assertEqual(date_query.interval.end, datetime(2000, 1, 1, 13))

    def test_datetime_space_separator(self):
        if False:
            while True:
                i = 10
        date_query = DateQuery('added', '2000-01-01 12')
        self.assertEqual(date_query.interval.start, datetime(2000, 1, 1, 12))
        self.assertEqual(date_query.interval.end, datetime(2000, 1, 1, 13))

    def test_datetime_invalid_separator(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(InvalidQueryArgumentValueError):
            DateQuery('added', '2000-01-01x12')

def suite():
    if False:
        while True:
            i = 10
    return unittest.TestLoader().loadTestsFromName(__name__)
if __name__ == '__main__':
    unittest.main(defaultTest='suite')