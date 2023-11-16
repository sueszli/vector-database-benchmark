from __future__ import absolute_import
import datetime
import mock
import unittest2
from st2client.utils.date import add_utc_tz
from st2client.utils.date import format_dt
from st2client.utils.date import format_isodate
from st2client.utils.date import format_isodate_for_user_timezone

class DateUtilsTestCase(unittest2.TestCase):

    def test_format_dt(self):
        if False:
            for i in range(10):
                print('nop')
        dt = datetime.datetime(2015, 10, 20, 8, 0, 0)
        dt = add_utc_tz(dt)
        result = format_dt(dt)
        self.assertEqual(result, 'Tue, 20 Oct 2015 08:00:00 UTC')

    def test_format_isodate(self):
        if False:
            i = 10
            return i + 15
        value = 'Tue, 20 Oct 2015 08:00:00 UTC'
        result = format_isodate(value=value)
        self.assertEqual(result, 'Tue, 20 Oct 2015 08:00:00 UTC')
        value = 'Tue, 20 Oct 2015 08:00:00 UTC'
        result = format_isodate(value=value, timezone='Europe/Ljubljana')
        self.assertEqual(result, 'Tue, 20 Oct 2015 10:00:00 CEST')

    @mock.patch('st2client.utils.date.get_config')
    def test_format_isodate_for_user_timezone(self, mock_get_config):
        if False:
            return 10
        mock_get_config.return_value = {}
        value = 'Tue, 20 Oct 2015 08:00:00 UTC'
        result = format_isodate_for_user_timezone(value=value)
        self.assertEqual(result, 'Tue, 20 Oct 2015 08:00:00 UTC')
        mock_get_config.return_value = {'cli': {'timezone': 'Europe/Ljubljana'}}
        value = 'Tue, 20 Oct 2015 08:00:00 UTC'
        result = format_isodate_for_user_timezone(value=value)
        self.assertEqual(result, 'Tue, 20 Oct 2015 10:00:00 CEST')