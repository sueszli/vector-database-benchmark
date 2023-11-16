"""
Test cases for L{twisted.names.rfc1982}.
"""
import calendar
from datetime import datetime
from functools import partial
from twisted.names._rfc1982 import SerialNumber
from twisted.trial import unittest

class SerialNumberTests(unittest.TestCase):
    """
    Tests for L{SerialNumber}.
    """

    def test_serialBitsDefault(self):
        if False:
            print('Hello World!')
        '\n        L{SerialNumber.serialBits} has default value 32.\n        '
        self.assertEqual(SerialNumber(1)._serialBits, 32)

    def test_serialBitsOverride(self):
        if False:
            while True:
                i = 10
        '\n        L{SerialNumber.__init__} accepts a C{serialBits} argument whose value is\n        assigned to L{SerialNumber.serialBits}.\n        '
        self.assertEqual(SerialNumber(1, serialBits=8)._serialBits, 8)

    def test_repr(self):
        if False:
            print('Hello World!')
        '\n        L{SerialNumber.__repr__} returns a string containing number and\n        serialBits.\n        '
        self.assertEqual('<SerialNumber number=123 serialBits=32>', repr(SerialNumber(123, serialBits=32)))

    def test_str(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SerialNumber.__str__} returns a string representation of the current\n        value.\n        '
        self.assertEqual(str(SerialNumber(123)), '123')

    def test_int(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SerialNumber.__int__} returns an integer representation of the current\n        value.\n        '
        self.assertEqual(int(SerialNumber(123)), 123)

    def test_hash(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SerialNumber.__hash__} allows L{SerialNumber} instances to be hashed\n        for use as dictionary keys.\n        '
        self.assertEqual(hash(SerialNumber(1)), hash(SerialNumber(1)))
        self.assertNotEqual(hash(SerialNumber(1)), hash(SerialNumber(2)))

    def test_convertOtherSerialBitsMismatch(self):
        if False:
            print('Hello World!')
        '\n        L{SerialNumber._convertOther} raises L{TypeError} if the other\n        SerialNumber instance has a different C{serialBits} value.\n        '
        s1 = SerialNumber(0, serialBits=8)
        s2 = SerialNumber(0, serialBits=16)
        self.assertRaises(TypeError, s1._convertOther, s2)

    def test_eq(self):
        if False:
            return 10
        '\n        L{SerialNumber.__eq__} provides rich equality comparison.\n        '
        self.assertEqual(SerialNumber(1), SerialNumber(1))

    def test_eqForeignType(self):
        if False:
            i = 10
            return i + 15
        '\n        == comparison of L{SerialNumber} with a non-L{SerialNumber} instance\n        returns L{NotImplemented}.\n        '
        self.assertFalse(SerialNumber(1) == object())
        self.assertIs(SerialNumber(1).__eq__(object()), NotImplemented)

    def test_ne(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SerialNumber.__ne__} provides rich equality comparison.\n        '
        self.assertFalse(SerialNumber(1) != SerialNumber(1))
        self.assertNotEqual(SerialNumber(1), SerialNumber(2))

    def test_neForeignType(self):
        if False:
            while True:
                i = 10
        '\n        != comparison of L{SerialNumber} with a non-L{SerialNumber} instance\n        returns L{NotImplemented}.\n        '
        self.assertTrue(SerialNumber(1) != object())
        self.assertIs(SerialNumber(1).__ne__(object()), NotImplemented)

    def test_le(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{SerialNumber.__le__} provides rich <= comparison.\n        '
        self.assertTrue(SerialNumber(1) <= SerialNumber(1))
        self.assertTrue(SerialNumber(1) <= SerialNumber(2))

    def test_leForeignType(self):
        if False:
            return 10
        '\n        <= comparison of L{SerialNumber} with a non-L{SerialNumber} instance\n        raises L{TypeError}.\n        '
        self.assertRaises(TypeError, lambda : SerialNumber(1) <= object())

    def test_ge(self):
        if False:
            print('Hello World!')
        '\n        L{SerialNumber.__ge__} provides rich >= comparison.\n        '
        self.assertTrue(SerialNumber(1) >= SerialNumber(1))
        self.assertTrue(SerialNumber(2) >= SerialNumber(1))

    def test_geForeignType(self):
        if False:
            print('Hello World!')
        '\n        >= comparison of L{SerialNumber} with a non-L{SerialNumber} instance\n        raises L{TypeError}.\n        '
        self.assertRaises(TypeError, lambda : SerialNumber(1) >= object())

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{SerialNumber.__lt__} provides rich < comparison.\n        '
        self.assertTrue(SerialNumber(1) < SerialNumber(2))

    def test_ltForeignType(self):
        if False:
            i = 10
            return i + 15
        '\n        < comparison of L{SerialNumber} with a non-L{SerialNumber} instance\n        raises L{TypeError}.\n        '
        self.assertRaises(TypeError, lambda : SerialNumber(1) < object())

    def test_gt(self):
        if False:
            return 10
        '\n        L{SerialNumber.__gt__} provides rich > comparison.\n        '
        self.assertTrue(SerialNumber(2) > SerialNumber(1))

    def test_gtForeignType(self):
        if False:
            i = 10
            return i + 15
        '\n        > comparison of L{SerialNumber} with a non-L{SerialNumber} instance\n          raises L{TypeError}.\n        '
        self.assertRaises(TypeError, lambda : SerialNumber(2) > object())

    def test_add(self):
        if False:
            while True:
                i = 10
        '\n        L{SerialNumber.__add__} allows L{SerialNumber} instances to be summed.\n        '
        self.assertEqual(SerialNumber(1) + SerialNumber(1), SerialNumber(2))

    def test_addForeignType(self):
        if False:
            i = 10
            return i + 15
        '\n        Addition of L{SerialNumber} with a non-L{SerialNumber} instance raises\n        L{TypeError}.\n        '
        self.assertRaises(TypeError, lambda : SerialNumber(1) + object())

    def test_addOutOfRangeHigh(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SerialNumber} cannot be added with other SerialNumber values larger\n        than C{_maxAdd}.\n        '
        maxAdd = SerialNumber(1)._maxAdd
        self.assertRaises(ArithmeticError, lambda : SerialNumber(1) + SerialNumber(maxAdd + 1))

    def test_maxVal(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        L{SerialNumber.__add__} returns a wrapped value when s1 plus the s2\n        would result in a value greater than the C{maxVal}.\n        '
        s = SerialNumber(1)
        maxVal = s._halfRing + s._halfRing - 1
        maxValPlus1 = maxVal + 1
        self.assertTrue(SerialNumber(maxValPlus1) > SerialNumber(maxVal))
        self.assertEqual(SerialNumber(maxValPlus1), SerialNumber(0))

    def test_fromRFC4034DateString(self):
        if False:
            i = 10
            return i + 15
        "\n        L{SerialNumber.fromRFC4034DateString} accepts a datetime string argument\n        of the form 'YYYYMMDDhhmmss' and returns an L{SerialNumber} instance\n        whose value is the unix timestamp corresponding to that UTC date.\n        "
        self.assertEqual(SerialNumber(1325376000), SerialNumber.fromRFC4034DateString('20120101000000'))

    def test_toRFC4034DateString(self):
        if False:
            print('Hello World!')
        '\n        L{DateSerialNumber.toRFC4034DateString} interprets the current value as\n        a unix timestamp and returns a date string representation of that date.\n        '
        self.assertEqual('20120101000000', SerialNumber(1325376000).toRFC4034DateString())

    def test_unixEpoch(self):
        if False:
            while True:
                i = 10
        '\n        L{SerialNumber.toRFC4034DateString} stores 32bit timestamps relative to\n        the UNIX epoch.\n        '
        self.assertEqual(SerialNumber(0).toRFC4034DateString(), '19700101000000')

    def test_Y2106Problem(self):
        if False:
            return 10
        '\n        L{SerialNumber} wraps unix timestamps in the year 2106.\n        '
        self.assertEqual(SerialNumber(-1).toRFC4034DateString(), '21060207062815')

    def test_Y2038Problem(self):
        if False:
            i = 10
            return i + 15
        '\n        L{SerialNumber} raises ArithmeticError when used to add dates more than\n        68 years in the future.\n        '
        maxAddTime = calendar.timegm(datetime(2038, 1, 19, 3, 14, 7).utctimetuple())
        self.assertEqual(maxAddTime, SerialNumber(0)._maxAdd)
        self.assertRaises(ArithmeticError, lambda : SerialNumber(0) + SerialNumber(maxAddTime + 1))

def assertUndefinedComparison(testCase, s1, s2):
    if False:
        print('Hello World!')
    '\n    A custom assertion for L{SerialNumber} values that cannot be meaningfully\n    compared.\n\n    "Note that there are some pairs of values s1 and s2 for which s1 is not\n    equal to s2, but for which s1 is neither greater than, nor less than, s2.\n    An attempt to use these ordering operators on such pairs of values produces\n    an undefined result."\n\n    @see: U{https://tools.ietf.org/html/rfc1982#section-3.2}\n\n    @param testCase: The L{unittest.TestCase} on which to call assertion\n        methods.\n    @type testCase: L{unittest.TestCase}\n\n    @param s1: The first value to compare.\n    @type s1: L{SerialNumber}\n\n    @param s2: The second value to compare.\n    @type s2: L{SerialNumber}\n    '
    testCase.assertFalse(s1 == s2)
    testCase.assertFalse(s1 <= s2)
    testCase.assertFalse(s1 < s2)
    testCase.assertFalse(s1 > s2)
    testCase.assertFalse(s1 >= s2)
serialNumber2 = partial(SerialNumber, serialBits=2)

class SerialNumber2BitTests(unittest.TestCase):
    """
    Tests for correct answers to example calculations in RFC1982 5.1.

    The simplest meaningful serial number space has SERIAL_BITS == 2.  In this
    space, the integers that make up the serial number space are 0, 1, 2, and 3.
    That is, 3 == 2^SERIAL_BITS - 1.

    https://tools.ietf.org/html/rfc1982#section-5.1
    """

    def test_maxadd(self):
        if False:
            return 10
        '\n        In this space, the largest integer that it is meaningful to add to a\n        sequence number is 2^(SERIAL_BITS - 1) - 1, or 1.\n        '
        self.assertEqual(SerialNumber(0, serialBits=2)._maxAdd, 1)

    def test_add(self):
        if False:
            i = 10
            return i + 15
        '\n        Then, as defined 0+1 == 1, 1+1 == 2, 2+1 == 3, and 3+1 == 0.\n        '
        self.assertEqual(serialNumber2(0) + serialNumber2(1), serialNumber2(1))
        self.assertEqual(serialNumber2(1) + serialNumber2(1), serialNumber2(2))
        self.assertEqual(serialNumber2(2) + serialNumber2(1), serialNumber2(3))
        self.assertEqual(serialNumber2(3) + serialNumber2(1), serialNumber2(0))

    def test_gt(self):
        if False:
            while True:
                i = 10
        '\n        Further, 1 > 0, 2 > 1, 3 > 2, and 0 > 3.\n        '
        self.assertTrue(serialNumber2(1) > serialNumber2(0))
        self.assertTrue(serialNumber2(2) > serialNumber2(1))
        self.assertTrue(serialNumber2(3) > serialNumber2(2))
        self.assertTrue(serialNumber2(0) > serialNumber2(3))

    def test_undefined(self):
        if False:
            print('Hello World!')
        '\n        It is undefined whether 2 > 0 or 0 > 2, and whether 1 > 3 or 3 > 1.\n        '
        assertUndefinedComparison(self, serialNumber2(2), serialNumber2(0))
        assertUndefinedComparison(self, serialNumber2(0), serialNumber2(2))
        assertUndefinedComparison(self, serialNumber2(1), serialNumber2(3))
        assertUndefinedComparison(self, serialNumber2(3), serialNumber2(1))
serialNumber8 = partial(SerialNumber, serialBits=8)

class SerialNumber8BitTests(unittest.TestCase):
    """
    Tests for correct answers to example calculations in RFC1982 5.2.

    Consider the case where SERIAL_BITS == 8.  In this space the integers that
    make up the serial number space are 0, 1, 2, ... 254, 255.  255 ==
    2^SERIAL_BITS - 1.

    https://tools.ietf.org/html/rfc1982#section-5.2
    """

    def test_maxadd(self):
        if False:
            while True:
                i = 10
        '\n        In this space, the largest integer that it is meaningful to add to a\n        sequence number is 2^(SERIAL_BITS - 1) - 1, or 127.\n        '
        self.assertEqual(SerialNumber(0, serialBits=8)._maxAdd, 127)

    def test_add(self):
        if False:
            print('Hello World!')
        '\n        Addition is as expected in this space, for example: 255+1 == 0,\n        100+100 == 200, and 200+100 == 44.\n        '
        self.assertEqual(serialNumber8(255) + serialNumber8(1), serialNumber8(0))
        self.assertEqual(serialNumber8(100) + serialNumber8(100), serialNumber8(200))
        self.assertEqual(serialNumber8(200) + serialNumber8(100), serialNumber8(44))

    def test_gt(self):
        if False:
            return 10
        '\n        Comparison is more interesting, 1 > 0, 44 > 0, 100 > 0, 100 > 44,\n        200 > 100, 255 > 200, 0 > 255, 100 > 255, 0 > 200, and 44 > 200.\n        '
        self.assertTrue(serialNumber8(1) > serialNumber8(0))
        self.assertTrue(serialNumber8(44) > serialNumber8(0))
        self.assertTrue(serialNumber8(100) > serialNumber8(0))
        self.assertTrue(serialNumber8(100) > serialNumber8(44))
        self.assertTrue(serialNumber8(200) > serialNumber8(100))
        self.assertTrue(serialNumber8(255) > serialNumber8(200))
        self.assertTrue(serialNumber8(100) > serialNumber8(255))
        self.assertTrue(serialNumber8(0) > serialNumber8(200))
        self.assertTrue(serialNumber8(44) > serialNumber8(200))

    def test_surprisingAddition(self):
        if False:
            print('Hello World!')
        '\n        Note that 100+100 > 100, but that (100+100)+100 < 100.  Incrementing a\n        serial number can cause it to become "smaller".  Of course, incrementing\n        by a smaller number will allow many more increments to be made before\n        this occurs.  However this is always something to be aware of, it can\n        cause surprising errors, or be useful as it is the only defined way to\n        actually cause a serial number to decrease.\n        '
        self.assertTrue(serialNumber8(100) + serialNumber8(100) > serialNumber8(100))
        self.assertTrue(serialNumber8(100) + serialNumber8(100) + serialNumber8(100) < serialNumber8(100))

    def test_undefined(self):
        if False:
            return 10
        '\n        The pairs of values 0 and 128, 1 and 129, 2 and 130, etc, to 127 and 255\n        are not equal, but in each pair, neither number is defined as being\n        greater than, or less than, the other.\n        '
        assertUndefinedComparison(self, serialNumber8(0), serialNumber8(128))
        assertUndefinedComparison(self, serialNumber8(1), serialNumber8(129))
        assertUndefinedComparison(self, serialNumber8(2), serialNumber8(130))
        assertUndefinedComparison(self, serialNumber8(127), serialNumber8(255))