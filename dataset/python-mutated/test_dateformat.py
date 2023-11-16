from datetime import date, datetime, time, timezone, tzinfo
from django.test import SimpleTestCase, override_settings
from django.test.utils import TZ_SUPPORT, requires_tz_support
from django.utils import dateformat, translation
from django.utils.dateformat import format
from django.utils.timezone import get_default_timezone, get_fixed_timezone, make_aware

@override_settings(TIME_ZONE='Europe/Copenhagen')
class DateFormatTests(SimpleTestCase):

    def setUp(self):
        if False:
            return 10
        self._orig_lang = translation.get_language()
        translation.activate('en-us')

    def tearDown(self):
        if False:
            print('Hello World!')
        translation.activate(self._orig_lang)

    def test_date(self):
        if False:
            print('Hello World!')
        d = date(2009, 5, 16)
        self.assertEqual(date.fromtimestamp(int(format(d, 'U'))), d)

    def test_naive_datetime(self):
        if False:
            while True:
                i = 10
        dt = datetime(2009, 5, 16, 5, 30, 30)
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U'))), dt)

    def test_naive_ambiguous_datetime(self):
        if False:
            return 10
        dt = datetime(2015, 10, 25, 2, 30, 0)
        self.assertEqual(format(dt, 'I'), '')
        self.assertEqual(format(dt, 'O'), '')
        self.assertEqual(format(dt, 'T'), '')
        self.assertEqual(format(dt, 'Z'), '')

    @requires_tz_support
    def test_datetime_with_local_tzinfo(self):
        if False:
            for i in range(10):
                print('nop')
        ltz = get_default_timezone()
        dt = make_aware(datetime(2009, 5, 16, 5, 30, 30), ltz)
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U')), ltz), dt)
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U'))), dt.replace(tzinfo=None))

    @requires_tz_support
    def test_datetime_with_tzinfo(self):
        if False:
            print('Hello World!')
        tz = get_fixed_timezone(-510)
        ltz = get_default_timezone()
        dt = make_aware(datetime(2009, 5, 16, 5, 30, 30), ltz)
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U')), tz), dt)
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U')), ltz), dt)
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U'))), dt.astimezone(ltz).replace(tzinfo=None))
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U')), tz).timetuple(), dt.astimezone(tz).timetuple())
        self.assertEqual(datetime.fromtimestamp(int(format(dt, 'U')), ltz).timetuple(), dt.astimezone(ltz).timetuple())

    def test_epoch(self):
        if False:
            i = 10
            return i + 15
        udt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(format(udt, 'U'), '0')

    def test_empty_format(self):
        if False:
            return 10
        my_birthday = datetime(1979, 7, 8, 22, 0)
        self.assertEqual(dateformat.format(my_birthday, ''), '')

    def test_am_pm(self):
        if False:
            print('Hello World!')
        morning = time(7, 0)
        evening = time(19, 0)
        self.assertEqual(dateformat.format(morning, 'a'), 'a.m.')
        self.assertEqual(dateformat.format(evening, 'a'), 'p.m.')
        self.assertEqual(dateformat.format(morning, 'A'), 'AM')
        self.assertEqual(dateformat.format(evening, 'A'), 'PM')

    def test_microsecond(self):
        if False:
            for i in range(10):
                print('nop')
        dt = datetime(2009, 5, 16, microsecond=123)
        self.assertEqual(dateformat.format(dt, 'u'), '000123')

    def test_date_formats(self):
        if False:
            i = 10
            return i + 15
        my_birthday = datetime(1979, 7, 8, 22, 0)
        for (specifier, expected) in [('b', 'jul'), ('d', '08'), ('D', 'Sun'), ('E', 'July'), ('F', 'July'), ('j', '8'), ('l', 'Sunday'), ('L', 'False'), ('m', '07'), ('M', 'Jul'), ('n', '7'), ('N', 'July'), ('o', '1979'), ('S', 'th'), ('t', '31'), ('w', '0'), ('W', '27'), ('y', '79'), ('Y', '1979'), ('z', '189')]:
            with self.subTest(specifier=specifier):
                self.assertEqual(dateformat.format(my_birthday, specifier), expected)

    def test_date_formats_c_format(self):
        if False:
            i = 10
            return i + 15
        timestamp = datetime(2008, 5, 19, 11, 45, 23, 123456)
        self.assertEqual(dateformat.format(timestamp, 'c'), '2008-05-19T11:45:23.123456')

    def test_time_formats(self):
        if False:
            while True:
                i = 10
        my_birthday = datetime(1979, 7, 8, 22, 0)
        for (specifier, expected) in [('a', 'p.m.'), ('A', 'PM'), ('f', '10'), ('g', '10'), ('G', '22'), ('h', '10'), ('H', '22'), ('i', '00'), ('P', '10 p.m.'), ('s', '00'), ('u', '000000')]:
            with self.subTest(specifier=specifier):
                self.assertEqual(dateformat.format(my_birthday, specifier), expected)

    def test_dateformat(self):
        if False:
            return 10
        my_birthday = datetime(1979, 7, 8, 22, 0)
        self.assertEqual(dateformat.format(my_birthday, 'Y z \\C\\E\\T'), '1979 189 CET')
        self.assertEqual(dateformat.format(my_birthday, 'jS \\o\\f F'), '8th of July')

    def test_futuredates(self):
        if False:
            for i in range(10):
                print('nop')
        the_future = datetime(2100, 10, 25, 0, 0)
        self.assertEqual(dateformat.format(the_future, 'Y'), '2100')

    def test_day_of_year_leap(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(dateformat.format(datetime(2000, 12, 31), 'z'), '366')

    def test_timezones(self):
        if False:
            return 10
        my_birthday = datetime(1979, 7, 8, 22, 0)
        summertime = datetime(2005, 10, 30, 1, 0)
        wintertime = datetime(2005, 10, 30, 4, 0)
        noon = time(12, 0, 0)
        tz = get_fixed_timezone(-210)
        aware_dt = datetime(2009, 5, 16, 5, 30, 30, tzinfo=tz)
        if TZ_SUPPORT:
            for (specifier, expected) in [('e', ''), ('O', '+0100'), ('r', 'Sun, 08 Jul 1979 22:00:00 +0100'), ('T', 'CET'), ('U', '300315600'), ('Z', '3600')]:
                with self.subTest(specifier=specifier):
                    self.assertEqual(dateformat.format(my_birthday, specifier), expected)
            self.assertEqual(dateformat.format(aware_dt, 'e'), '-0330')
            self.assertEqual(dateformat.format(aware_dt, 'r'), 'Sat, 16 May 2009 05:30:30 -0330')
            self.assertEqual(dateformat.format(summertime, 'I'), '1')
            self.assertEqual(dateformat.format(summertime, 'O'), '+0200')
            self.assertEqual(dateformat.format(wintertime, 'I'), '0')
            self.assertEqual(dateformat.format(wintertime, 'O'), '+0100')
            for specifier in ['e', 'O', 'T', 'Z']:
                with self.subTest(specifier=specifier):
                    self.assertEqual(dateformat.time_format(noon, specifier), '')
        self.assertEqual(dateformat.format(aware_dt, 'O'), '-0330')

    def test_invalid_time_format_specifiers(self):
        if False:
            i = 10
            return i + 15
        my_birthday = date(1984, 8, 7)
        for specifier in ['a', 'A', 'f', 'g', 'G', 'h', 'H', 'i', 'P', 's', 'u']:
            with self.subTest(specifier=specifier):
                msg = f'The format for date objects may not contain time-related format specifiers (found {specifier!r}).'
                with self.assertRaisesMessage(TypeError, msg):
                    dateformat.format(my_birthday, specifier)

    @requires_tz_support
    def test_e_format_with_named_time_zone(self):
        if False:
            for i in range(10):
                print('nop')
        dt = datetime(1970, 1, 1, tzinfo=timezone.utc)
        self.assertEqual(dateformat.format(dt, 'e'), 'UTC')

    @requires_tz_support
    def test_e_format_with_time_zone_with_unimplemented_tzname(self):
        if False:
            for i in range(10):
                print('nop')

        class NoNameTZ(tzinfo):
            """Time zone without .tzname() defined."""

            def utcoffset(self, dt):
                if False:
                    while True:
                        i = 10
                return None
        dt = datetime(1970, 1, 1, tzinfo=NoNameTZ())
        self.assertEqual(dateformat.format(dt, 'e'), '')

    def test_P_format(self):
        if False:
            while True:
                i = 10
        for (expected, t) in [('midnight', time(0)), ('noon', time(12)), ('4 a.m.', time(4)), ('8:30 a.m.', time(8, 30)), ('4 p.m.', time(16)), ('8:30 p.m.', time(20, 30))]:
            with self.subTest(time=t):
                self.assertEqual(dateformat.time_format(t, 'P'), expected)

    def test_r_format_with_date(self):
        if False:
            for i in range(10):
                print('nop')
        dt = date(2022, 7, 1)
        self.assertEqual(dateformat.format(dt, 'r'), 'Fri, 01 Jul 2022 00:00:00 +0200')

    def test_r_format_with_non_en_locale(self):
        if False:
            i = 10
            return i + 15
        dt = datetime(1979, 7, 8, 22, 0)
        with translation.override('fr'):
            self.assertEqual(dateformat.format(dt, 'r'), 'Sun, 08 Jul 1979 22:00:00 +0100')

    def test_S_format(self):
        if False:
            for i in range(10):
                print('nop')
        for (expected, days) in [('st', [1, 21, 31]), ('nd', [2, 22]), ('rd', [3, 23]), ('th', (n for n in range(4, 31) if n not in [21, 22, 23]))]:
            for day in days:
                dt = date(1970, 1, day)
                with self.subTest(day=day):
                    self.assertEqual(dateformat.format(dt, 'S'), expected)

    def test_y_format_year_before_1000(self):
        if False:
            for i in range(10):
                print('nop')
        tests = [(476, '76'), (42, '42'), (4, '04')]
        for (year, expected_date) in tests:
            with self.subTest(year=year):
                self.assertEqual(dateformat.format(datetime(year, 9, 8, 5, 0), 'y'), expected_date)

    def test_Y_format_year_before_1000(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(dateformat.format(datetime(1, 1, 1), 'Y'), '0001')
        self.assertEqual(dateformat.format(datetime(999, 1, 1), 'Y'), '0999')

    def test_twelve_hour_format(self):
        if False:
            for i in range(10):
                print('nop')
        tests = [(0, '12', '12'), (1, '1', '01'), (11, '11', '11'), (12, '12', '12'), (13, '1', '01'), (23, '11', '11')]
        for (hour, g_expected, h_expected) in tests:
            dt = datetime(2000, 1, 1, hour)
            with self.subTest(hour=hour):
                self.assertEqual(dateformat.format(dt, 'g'), g_expected)
                self.assertEqual(dateformat.format(dt, 'h'), h_expected)