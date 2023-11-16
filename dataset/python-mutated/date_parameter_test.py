import datetime
from helpers import unittest, in_parse
import luigi
import luigi.interface

class DateTask(luigi.Task):
    day = luigi.DateParameter()

class DateHourTask(luigi.Task):
    dh = luigi.DateHourParameter()

class DateMinuteTask(luigi.Task):
    dm = luigi.DateMinuteParameter()

class DateSecondTask(luigi.Task):
    ds = luigi.DateSecondParameter()

class MonthTask(luigi.Task):
    month = luigi.MonthParameter()

class YearTask(luigi.Task):
    year = luigi.YearParameter()

class DateParameterTest(unittest.TestCase):

    def test_parse(self):
        if False:
            i = 10
            return i + 15
        d = luigi.DateParameter().parse('2015-04-03')
        self.assertEqual(d, datetime.date(2015, 4, 3))

    def test_serialize(self):
        if False:
            i = 10
            return i + 15
        d = luigi.DateParameter().serialize(datetime.date(2015, 4, 3))
        self.assertEqual(d, '2015-04-03')

    def test_parse_interface(self):
        if False:
            print('Hello World!')
        in_parse(['DateTask', '--day', '2015-04-03'], lambda task: self.assertEqual(task.day, datetime.date(2015, 4, 3)))

    def test_serialize_task(self):
        if False:
            i = 10
            return i + 15
        t = DateTask(datetime.date(2015, 4, 3))
        self.assertEqual(str(t), 'DateTask(day=2015-04-03)')

class DateHourParameterTest(unittest.TestCase):

    def test_parse(self):
        if False:
            i = 10
            return i + 15
        dh = luigi.DateHourParameter().parse('2013-02-01T18')
        self.assertEqual(dh, datetime.datetime(2013, 2, 1, 18, 0, 0))

    def test_date_to_dh(self):
        if False:
            return 10
        date = luigi.DateHourParameter().normalize(datetime.date(2000, 1, 1))
        self.assertEqual(date, datetime.datetime(2000, 1, 1, 0))

    def test_serialize(self):
        if False:
            i = 10
            return i + 15
        dh = luigi.DateHourParameter().serialize(datetime.datetime(2013, 2, 1, 18, 0, 0))
        self.assertEqual(dh, '2013-02-01T18')

    def test_parse_interface(self):
        if False:
            while True:
                i = 10
        in_parse(['DateHourTask', '--dh', '2013-02-01T18'], lambda task: self.assertEqual(task.dh, datetime.datetime(2013, 2, 1, 18, 0, 0)))

    def test_serialize_task(self):
        if False:
            return 10
        t = DateHourTask(datetime.datetime(2013, 2, 1, 18, 0, 0))
        self.assertEqual(str(t), 'DateHourTask(dh=2013-02-01T18)')

class DateMinuteParameterTest(unittest.TestCase):

    def test_parse(self):
        if False:
            for i in range(10):
                print('nop')
        dm = luigi.DateMinuteParameter().parse('2013-02-01T1842')
        self.assertEqual(dm, datetime.datetime(2013, 2, 1, 18, 42, 0))

    def test_parse_padding_zero(self):
        if False:
            return 10
        dm = luigi.DateMinuteParameter().parse('2013-02-01T1807')
        self.assertEqual(dm, datetime.datetime(2013, 2, 1, 18, 7, 0))

    def test_parse_deprecated(self):
        if False:
            print('Hello World!')
        with self.assertWarnsRegex(DeprecationWarning, 'Using "H" between hours and minutes is deprecated, omit it instead.'):
            dm = luigi.DateMinuteParameter().parse('2013-02-01T18H42')
        self.assertEqual(dm, datetime.datetime(2013, 2, 1, 18, 42, 0))

    def test_serialize(self):
        if False:
            return 10
        dm = luigi.DateMinuteParameter().serialize(datetime.datetime(2013, 2, 1, 18, 42, 0))
        self.assertEqual(dm, '2013-02-01T1842')

    def test_serialize_padding_zero(self):
        if False:
            i = 10
            return i + 15
        dm = luigi.DateMinuteParameter().serialize(datetime.datetime(2013, 2, 1, 18, 7, 0))
        self.assertEqual(dm, '2013-02-01T1807')

    def test_parse_interface(self):
        if False:
            for i in range(10):
                print('nop')
        in_parse(['DateMinuteTask', '--dm', '2013-02-01T1842'], lambda task: self.assertEqual(task.dm, datetime.datetime(2013, 2, 1, 18, 42, 0)))

    def test_serialize_task(self):
        if False:
            return 10
        t = DateMinuteTask(datetime.datetime(2013, 2, 1, 18, 42, 0))
        self.assertEqual(str(t), 'DateMinuteTask(dm=2013-02-01T1842)')

class DateSecondParameterTest(unittest.TestCase):

    def test_parse(self):
        if False:
            for i in range(10):
                print('nop')
        ds = luigi.DateSecondParameter().parse('2013-02-01T184227')
        self.assertEqual(ds, datetime.datetime(2013, 2, 1, 18, 42, 27))

    def test_serialize(self):
        if False:
            for i in range(10):
                print('nop')
        ds = luigi.DateSecondParameter().serialize(datetime.datetime(2013, 2, 1, 18, 42, 27))
        self.assertEqual(ds, '2013-02-01T184227')

    def test_parse_interface(self):
        if False:
            while True:
                i = 10
        in_parse(['DateSecondTask', '--ds', '2013-02-01T184227'], lambda task: self.assertEqual(task.ds, datetime.datetime(2013, 2, 1, 18, 42, 27)))

    def test_serialize_task(self):
        if False:
            while True:
                i = 10
        t = DateSecondTask(datetime.datetime(2013, 2, 1, 18, 42, 27))
        self.assertEqual(str(t), 'DateSecondTask(ds=2013-02-01T184227)')

class MonthParameterTest(unittest.TestCase):

    def test_parse(self):
        if False:
            print('Hello World!')
        m = luigi.MonthParameter().parse('2015-04')
        self.assertEqual(m, datetime.date(2015, 4, 1))

    def test_construct_month_interval(self):
        if False:
            for i in range(10):
                print('nop')
        m = MonthTask(luigi.date_interval.Month(2015, 4))
        self.assertEqual(m.month, datetime.date(2015, 4, 1))

    def test_month_interval_default(self):
        if False:
            return 10

        class MonthDefaultTask(luigi.task.Task):
            month = luigi.MonthParameter(default=luigi.date_interval.Month(2015, 4))
        m = MonthDefaultTask()
        self.assertEqual(m.month, datetime.date(2015, 4, 1))

    def test_serialize(self):
        if False:
            while True:
                i = 10
        m = luigi.MonthParameter().serialize(datetime.date(2015, 4, 3))
        self.assertEqual(m, '2015-04')

    def test_parse_interface(self):
        if False:
            print('Hello World!')
        in_parse(['MonthTask', '--month', '2015-04'], lambda task: self.assertEqual(task.month, datetime.date(2015, 4, 1)))

    def test_serialize_task(self):
        if False:
            i = 10
            return i + 15
        task = MonthTask(datetime.date(2015, 4, 3))
        self.assertEqual(str(task), 'MonthTask(month=2015-04)')

class YearParameterTest(unittest.TestCase):

    def test_parse(self):
        if False:
            print('Hello World!')
        year = luigi.YearParameter().parse('2015')
        self.assertEqual(year, datetime.date(2015, 1, 1))

    def test_construct_year_interval(self):
        if False:
            i = 10
            return i + 15
        y = YearTask(luigi.date_interval.Year(2015))
        self.assertEqual(y.year, datetime.date(2015, 1, 1))

    def test_year_interval_default(self):
        if False:
            i = 10
            return i + 15

        class YearDefaultTask(luigi.task.Task):
            year = luigi.YearParameter(default=luigi.date_interval.Year(2015))
        m = YearDefaultTask()
        self.assertEqual(m.year, datetime.date(2015, 1, 1))

    def test_serialize(self):
        if False:
            return 10
        year = luigi.YearParameter().serialize(datetime.date(2015, 4, 3))
        self.assertEqual(year, '2015')

    def test_parse_interface(self):
        if False:
            for i in range(10):
                print('nop')
        in_parse(['YearTask', '--year', '2015'], lambda task: self.assertEqual(task.year, datetime.date(2015, 1, 1)))

    def test_serialize_task(self):
        if False:
            print('Hello World!')
        task = YearTask(datetime.date(2015, 4, 3))
        self.assertEqual(str(task), 'YearTask(year=2015)')