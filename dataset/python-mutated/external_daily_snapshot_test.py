import unittest
import luigi
from luigi.contrib.external_daily_snapshot import ExternalDailySnapshot
from luigi.mock import MockTarget
import datetime

class DataDump(ExternalDailySnapshot):
    param = luigi.Parameter()
    a = luigi.Parameter(default='zebra')
    aa = luigi.Parameter(default='Congo')

    def output(self):
        if False:
            while True:
                i = 10
        return MockTarget('data-%s-%s-%s-%s' % (self.param, self.a, self.aa, self.date))

class ExternalDailySnapshotTest(unittest.TestCase):

    def test_latest(self):
        if False:
            print('Hello World!')
        MockTarget('data-xyz-zebra-Congo-2012-01-01').open('w').close()
        d = DataDump.latest(date=datetime.date(2012, 1, 10), param='xyz')
        self.assertEquals(d.date, datetime.date(2012, 1, 1))

    def test_latest_not_exists(self):
        if False:
            print('Hello World!')
        MockTarget('data-abc-zebra-Congo-2012-01-01').open('w').close()
        d = DataDump.latest(date=datetime.date(2012, 1, 11), param='abc', lookback=5)
        self.assertEquals(d.date, datetime.date(2012, 1, 7))

    def test_deterministic(self):
        if False:
            while True:
                i = 10
        MockTarget('data-pqr-zebra-Congo-2012-01-01').open('w').close()
        d = DataDump.latest(date=datetime.date(2012, 1, 10), param='pqr', a='zebra', aa='Congo')
        self.assertEquals(d.date, datetime.date(2012, 1, 1))
        MockTarget('data-pqr-zebra-Congo-2012-01-05').open('w').close()
        d = DataDump.latest(date=datetime.date(2012, 1, 10), param='pqr', aa='Congo', a='zebra')
        self.assertEquals(d.date, datetime.date(2012, 1, 1))