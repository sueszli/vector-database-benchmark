from __future__ import annotations
import unittest
from datetime import datetime, timedelta
from decimal import Decimal
from fractions import Fraction
from typing_extensions import assert_type
from unittest.mock import MagicMock, Mock, patch
case = unittest.TestCase()
case.assertAlmostEqual(1, 2.4)
case.assertAlmostEqual(2.4, 2.41)
case.assertAlmostEqual(Fraction(49, 50), Fraction(48, 50))
case.assertAlmostEqual(3.14, complex(5, 6))
case.assertAlmostEqual(datetime(1999, 1, 2), datetime(1999, 1, 2, microsecond=1), delta=timedelta(hours=1))
case.assertAlmostEqual(datetime(1999, 1, 2), datetime(1999, 1, 2, microsecond=1), None, 'foo', timedelta(hours=1))
case.assertAlmostEqual(Decimal('1.1'), Decimal('1.11'))
case.assertAlmostEqual(2.4, 2.41, places=8)
case.assertAlmostEqual(2.4, 2.41, delta=0.02)
case.assertAlmostEqual(2.4, 2.41, None, 'foo', 0.02)
case.assertAlmostEqual(2.4, 2.41, places=9, delta=0.02)
case.assertAlmostEqual('foo', 'bar')
case.assertAlmostEqual(datetime(1999, 1, 2), datetime(1999, 1, 2, microsecond=1))
case.assertAlmostEqual(Decimal('0.4'), Fraction(1, 2))
case.assertAlmostEqual(complex(2, 3), Decimal('0.9'))
case.assertAlmostEqual(1, 2.4)
case.assertNotAlmostEqual(Fraction(49, 50), Fraction(48, 50))
case.assertAlmostEqual(3.14, complex(5, 6))
case.assertNotAlmostEqual(datetime(1999, 1, 2), datetime(1999, 1, 2, microsecond=1), delta=timedelta(hours=1))
case.assertNotAlmostEqual(datetime(1999, 1, 2), datetime(1999, 1, 2, microsecond=1), None, 'foo', timedelta(hours=1))
case.assertNotAlmostEqual(2.4, 2.41, places=9, delta=0.02)
case.assertNotAlmostEqual('foo', 'bar')
case.assertNotAlmostEqual(datetime(1999, 1, 2), datetime(1999, 1, 2, microsecond=1))
case.assertNotAlmostEqual(Decimal('0.4'), Fraction(1, 2))
case.assertNotAlmostEqual(complex(2, 3), Decimal('0.9'))

class Spam:

    def __lt__(self, other: object) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return True

class Eggs:

    def __gt__(self, other: object) -> bool:
        if False:
            i = 10
            return i + 15
        return True

class Ham:

    def __lt__(self, other: Ham) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(other, Ham):
            return NotImplemented
        return True

class Bacon:

    def __gt__(self, other: Bacon) -> bool:
        if False:
            print('Hello World!')
        if not isinstance(other, Bacon):
            return NotImplemented
        return True
case.assertGreater(5.8, 3)
case.assertGreater(Decimal('4.5'), Fraction(3, 2))
case.assertGreater(Fraction(3, 2), 0.9)
case.assertGreater(Eggs(), object())
case.assertGreater(object(), Spam())
case.assertGreater(Ham(), Ham())
case.assertGreater(Bacon(), Bacon())
case.assertGreater(object(), object())
case.assertGreater(datetime(1999, 1, 2), 1)
case.assertGreater(Spam(), Eggs())
case.assertGreater(Ham(), Bacon())
case.assertGreater(Bacon(), Ham())

@patch('sys.exit')
def f_default_new(i: int, mock: MagicMock) -> str:
    if False:
        while True:
            i = 10
    return 'asdf'

@patch('sys.exit', new=42)
def f_explicit_new(i: int) -> str:
    if False:
        print('Hello World!')
    return 'asdf'
assert_type(f_default_new(1), str)
f_default_new('a')
assert_type(f_explicit_new(1), str)
f_explicit_new('a')

@patch('sys.exit', new=Mock())
class TestXYZ(unittest.TestCase):
    attr: int = 5

    @staticmethod
    def method() -> int:
        if False:
            print('Hello World!')
        return 123
assert_type(TestXYZ.attr, int)
assert_type(TestXYZ.method(), int)