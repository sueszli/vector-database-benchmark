"""
Tests for L{twisted.python._tzhelper}.
"""
from __future__ import annotations
from os import environ
try:
    from time import tzset as _tzset
except ImportError:
    tzset = None
else:
    tzset = _tzset
from datetime import datetime, timedelta
from time import mktime as mktime_real
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.trial.unittest import SkipTest, TestCase

def mktime(t9: tuple[int, int, int, int, int, int, int, int, int]) -> float:
    if False:
        for i in range(10):
            print('nop')
    '\n    Call L{mktime_real}, and if it raises L{OverflowError}, catch it and raise\n    SkipTest instead.\n\n    @param t9: A time as a 9-item tuple.\n    @type t9: L{tuple}\n\n    @return: A timestamp.\n    @rtype: L{float}\n    '
    try:
        return mktime_real(t9)
    except OverflowError:
        raise SkipTest(f'Platform cannot construct time zone for {t9!r}')

def setTZ(name: str | None) -> None:
    if False:
        print('Hello World!')
    '\n    Set time zone.\n\n    @param name: a time zone name\n    @type name: L{str}\n    '
    if tzset is None:
        return
    if name is None:
        try:
            del environ['TZ']
        except KeyError:
            pass
    else:
        environ['TZ'] = name
    tzset()

def addTZCleanup(testCase: TestCase) -> None:
    if False:
        while True:
            i = 10
    '\n    Add cleanup hooks to a test case to reset timezone to original value.\n\n    @param testCase: the test case to add the cleanup to.\n    @type testCase: L{unittest.TestCase}\n    '
    tzIn = environ.get('TZ', None)

    @testCase.addCleanup
    def resetTZ() -> None:
        if False:
            print('Hello World!')
        setTZ(tzIn)

class FixedOffsetTimeZoneTests(TestCase):
    """
    Tests for L{FixedOffsetTimeZone}.
    """

    def test_tzinfo(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that timezone attributes respect the timezone as set by the\n        standard C{TZ} environment variable and L{tzset} API.\n        '
        if tzset is None:
            raise SkipTest('Platform cannot change timezone; unable to verify offsets.')

        def testForTimeZone(name: str, expectedOffsetDST: str, expectedOffsetSTD: str) -> None:
            if False:
                return 10
            setTZ(name)
            localDST = mktime((2006, 6, 30, 0, 0, 0, 4, 181, 1))
            localDSTdt = datetime.fromtimestamp(localDST)
            localSTD = mktime((2007, 1, 31, 0, 0, 0, 2, 31, 0))
            localSTDdt = datetime.fromtimestamp(localSTD)
            tzDST = FixedOffsetTimeZone.fromLocalTimeStamp(localDST)
            tzSTD = FixedOffsetTimeZone.fromLocalTimeStamp(localSTD)
            self.assertEqual(tzDST.tzname(localDSTdt), f'UTC{expectedOffsetDST}')
            self.assertEqual(tzSTD.tzname(localSTDdt), f'UTC{expectedOffsetSTD}')
            self.assertEqual(tzDST.dst(localDSTdt), timedelta(0))
            self.assertEqual(tzSTD.dst(localSTDdt), timedelta(0))

            def timeDeltaFromOffset(offset: str) -> timedelta:
                if False:
                    i = 10
                    return i + 15
                assert len(offset) == 5
                sign = offset[0]
                hours = int(offset[1:3])
                minutes = int(offset[3:5])
                if sign == '-':
                    hours = -hours
                    minutes = -minutes
                else:
                    assert sign == '+'
                return timedelta(hours=hours, minutes=minutes)
            self.assertEqual(tzDST.utcoffset(localDSTdt), timeDeltaFromOffset(expectedOffsetDST))
            self.assertEqual(tzSTD.utcoffset(localSTDdt), timeDeltaFromOffset(expectedOffsetSTD))
        addTZCleanup(self)
        testForTimeZone('UTC+00', '+0000', '+0000')
        testForTimeZone('EST+05EDT,M4.1.0,M10.5.0', '-0400', '-0500')
        testForTimeZone('CEST-01CEDT,M4.1.0,M10.5.0', '+0200', '+0100')
        testForTimeZone('CST+06', '-0600', '-0600')