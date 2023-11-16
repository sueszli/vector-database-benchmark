from __future__ import absolute_import
import datetime
import unittest
from st2common.util import isotime
from st2common.util import date

class IsoTimeUtilsTestCase(unittest.TestCase):

    def test_validate(self):
        if False:
            i = 10
            return i + 15
        self.assertTrue(isotime.validate('2000-01-01 12:00:00Z'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00+00'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00+0000'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00+00:00'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00.000000Z'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00.000000+00'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00.000000+0000'))
        self.assertTrue(isotime.validate('2000-01-01 12:00:00.000000+00:00'))
        self.assertTrue(isotime.validate('2000-01-01T12:00:00Z'))
        self.assertTrue(isotime.validate('2000-01-01T12:00:00.000000Z'))
        self.assertTrue(isotime.validate('2000-01-01T12:00:00+00:00'))
        self.assertTrue(isotime.validate('2000-01-01T12:00:00.000000+00:00'))
        self.assertTrue(isotime.validate('2015-02-10T21:21:53.399Z'))
        self.assertFalse(isotime.validate('2000-01-01', raise_exception=False))
        self.assertFalse(isotime.validate('2000-01-01T12:00:00', raise_exception=False))
        self.assertFalse(isotime.validate('2000-01-01T12:00:00+00:00Z', raise_exception=False))
        self.assertFalse(isotime.validate('2000-01-01T12:00:00.000000', raise_exception=False))
        self.assertFalse(isotime.validate('Epic!', raise_exception=False))
        self.assertFalse(isotime.validate(object(), raise_exception=False))
        self.assertRaises(ValueError, isotime.validate, 'Epic!', True)

    def test_parse(self):
        if False:
            i = 10
            return i + 15
        dt = date.add_utc_tz(datetime.datetime(2000, 1, 1, 12))
        self.assertEqual(isotime.parse('2000-01-01 12:00:00Z'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00+00'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00+0000'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00+00:00'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00.000000Z'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00.000000+00'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00.000000+0000'), dt)
        self.assertEqual(isotime.parse('2000-01-01 12:00:00.000000+00:00'), dt)
        self.assertEqual(isotime.parse('2000-01-01T12:00:00Z'), dt)
        self.assertEqual(isotime.parse('2000-01-01T12:00:00+00:00'), dt)
        self.assertEqual(isotime.parse('2000-01-01T12:00:00.000000Z'), dt)
        self.assertEqual(isotime.parse('2000-01-01T12:00:00.000000+00:00'), dt)
        self.assertEqual(isotime.parse('2000-01-01T12:00:00.000Z'), dt)

    def test_format(self):
        if False:
            return 10
        dt = date.add_utc_tz(datetime.datetime(2000, 1, 1, 12))
        dt_str_usec_offset = '2000-01-01T12:00:00.000000+00:00'
        dt_str_usec = '2000-01-01T12:00:00.000000Z'
        dt_str_offset = '2000-01-01T12:00:00+00:00'
        dt_str = '2000-01-01T12:00:00Z'
        dt_unicode = '2000-01-01T12:00:00Z'
        self.assertEqual(isotime.format(dt, usec=True, offset=True), dt_str_usec_offset)
        self.assertEqual(isotime.format(dt, usec=True, offset=False), dt_str_usec)
        self.assertEqual(isotime.format(dt, usec=False, offset=True), dt_str_offset)
        self.assertEqual(isotime.format(dt, usec=False, offset=False), dt_str)
        self.assertEqual(isotime.format(dt_str, usec=False, offset=False), dt_str)
        self.assertEqual(isotime.format(dt_unicode, usec=False, offset=False), dt_unicode)
        dt = 1557390483
        self.assertEqual(isotime.format(dt, usec=True, offset=True), '2019-05-09T08:28:03.000000+00:00')
        self.assertEqual(isotime.format(dt, usec=False, offset=False), '2019-05-09T08:28:03Z')
        self.assertEqual(isotime.format(dt, usec=False, offset=True), '2019-05-09T08:28:03+00:00')

    def test_format_tz_naive(self):
        if False:
            return 10
        dt1 = datetime.datetime.utcnow()
        dt2 = isotime.parse(isotime.format(dt1, usec=True))
        self.assertEqual(dt2, date.add_utc_tz(dt1))

    def test_format_tz_aware(self):
        if False:
            i = 10
            return i + 15
        dt1 = date.add_utc_tz(datetime.datetime.utcnow())
        dt2 = isotime.parse(isotime.format(dt1, usec=True))
        self.assertEqual(dt2, dt1)

    def test_format_sec_truncated(self):
        if False:
            while True:
                i = 10
        dt1 = date.add_utc_tz(datetime.datetime.utcnow())
        dt2 = isotime.parse(isotime.format(dt1, usec=False))
        dt3 = datetime.datetime(dt1.year, dt1.month, dt1.day, dt1.hour, dt1.minute, dt1.second)
        self.assertLess(dt2, dt1)
        self.assertEqual(dt2, date.add_utc_tz(dt3))