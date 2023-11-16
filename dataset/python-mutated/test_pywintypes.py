import datetime
import operator
import time
import unittest
import pywintypes

class TestCase(unittest.TestCase):

    def testPyTimeFormat(self):
        if False:
            for i in range(10):
                print('nop')
        struct_current = time.localtime()
        pytime_current = pywintypes.Time(struct_current)
        format_strings = '%a %A %b %B %c %d %H %I %j %m %M %p %S %U %w %W %x %X %y %Y'
        for fmt in format_strings.split():
            v1 = pytime_current.Format(fmt)
            v2 = time.strftime(fmt, struct_current)
            self.assertEqual(v1, v2, f'format {fmt} failed - {v1!r} != {v2!r}')

    def testPyTimePrint(self):
        if False:
            while True:
                i = 10
        try:
            t = pywintypes.Time(-2)
            t.Format()
        except ValueError:
            return

    def testTimeInDict(self):
        if False:
            for i in range(10):
                print('nop')
        d = {}
        d['t1'] = pywintypes.Time(1)
        self.assertEqual(d['t1'], pywintypes.Time(1))

    def testPyTimeCompare(self):
        if False:
            while True:
                i = 10
        t1 = pywintypes.Time(100)
        t1_2 = pywintypes.Time(100)
        t2 = pywintypes.Time(101)
        self.assertEqual(t1, t1_2)
        self.assertTrue(t1 <= t1_2)
        self.assertTrue(t1_2 >= t1)
        self.assertNotEqual(t1, t2)
        self.assertTrue(t1 < t2)
        self.assertTrue(t2 > t1)

    def testPyTimeCompareOther(self):
        if False:
            for i in range(10):
                print('nop')
        t1 = pywintypes.Time(100)
        t2 = None
        self.assertNotEqual(t1, t2)

    def testTimeTuple(self):
        if False:
            i = 10
            return i + 15
        now = datetime.datetime.now()
        pt = pywintypes.Time(now.timetuple())
        if isinstance(pt, datetime.datetime):
            self.assertTrue(pt <= now)

    def testTimeTuplems(self):
        if False:
            print('Hello World!')
        now = datetime.datetime.now()
        tt = now.timetuple() + (now.microsecond // 1000,)
        pt = pywintypes.Time(tt)
        if isinstance(pt, datetime.datetime):
            expectedDelta = datetime.timedelta(milliseconds=1)
            self.assertTrue(-expectedDelta < now - pt < expectedDelta)

    def testPyTimeFromTime(self):
        if False:
            i = 10
            return i + 15
        t1 = pywintypes.Time(time.time())
        self.assertTrue(pywintypes.Time(t1) is t1)

    def testPyTimeTooLarge(self):
        if False:
            for i in range(10):
                print('nop')
        MAX_TIMESTAMP = 9223372036854775807
        ts = pywintypes.TimeStamp(MAX_TIMESTAMP)
        self.assertEqual(ts, datetime.datetime.max)

    def testGUID(self):
        if False:
            print('Hello World!')
        s = '{00020400-0000-0000-C000-000000000046}'
        iid = pywintypes.IID(s)
        iid2 = pywintypes.IID(memoryview(iid), True)
        self.assertEqual(iid, iid2)
        self.assertRaises(ValueError, pywintypes.IID, b'00', True)
        self.assertRaises(TypeError, pywintypes.IID, 0, True)

    def testGUIDRichCmp(self):
        if False:
            while True:
                i = 10
        s = '{00020400-0000-0000-C000-000000000046}'
        iid = pywintypes.IID(s)
        self.assertFalse(s is None)
        self.assertFalse(None == s)
        self.assertTrue(s is not None)
        self.assertTrue(None != s)
        self.assertRaises(TypeError, operator.gt, None, s)
        self.assertRaises(TypeError, operator.gt, s, None)
        self.assertRaises(TypeError, operator.lt, None, s)
        self.assertRaises(TypeError, operator.lt, s, None)

    def testGUIDInDict(self):
        if False:
            for i in range(10):
                print('nop')
        s = '{00020400-0000-0000-C000-000000000046}'
        iid = pywintypes.IID(s)
        d = {'item': iid}
        self.assertEqual(d['item'], iid)
if __name__ == '__main__':
    unittest.main()