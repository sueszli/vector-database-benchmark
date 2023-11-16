import unittest
from datetime import datetime
import pywintypes
import win32com.client
import win32com.server.util
import win32com.test.util
from win32timezone import TimeZoneInfo

class Tester:
    _public_methods_ = ['TestDate']

    def TestDate(self, d):
        if False:
            return 10
        assert isinstance(d, datetime)
        return d

def test_ob():
    if False:
        i = 10
        return i + 15
    return win32com.client.Dispatch(win32com.server.util.wrap(Tester()))

class TestCase(win32com.test.util.TestCase):

    def check(self, d, expected=None):
        if False:
            for i in range(10):
                print('nop')
        if not issubclass(pywintypes.TimeType, datetime):
            self.skipTest('this is testing pywintypes and datetime')
        got = test_ob().TestDate(d)
        self.assertEqual(got, expected or d)

    def testUTC(self):
        if False:
            print('Hello World!')
        self.check(datetime(year=2000, month=12, day=25, microsecond=500000, tzinfo=TimeZoneInfo.utc()))

    def testLocal(self):
        if False:
            print('Hello World!')
        self.check(datetime(year=2000, month=12, day=25, microsecond=500000, tzinfo=TimeZoneInfo.local()))

    def testMSTruncated(self):
        if False:
            i = 10
            return i + 15
        self.check(datetime(year=2000, month=12, day=25, microsecond=500500, tzinfo=TimeZoneInfo.utc()), datetime(year=2000, month=12, day=25, microsecond=500000, tzinfo=TimeZoneInfo.utc()))
if __name__ == '__main__':
    unittest.main()