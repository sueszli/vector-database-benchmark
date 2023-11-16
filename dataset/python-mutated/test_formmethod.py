from __future__ import annotations
'\nTest cases for formmethod module.\n'
from typing import Callable, Iterable
from typing_extensions import Concatenate, ParamSpec
from twisted.python import formmethod
from twisted.trial import unittest
_P = ParamSpec('_P')

class ArgumentTests(unittest.TestCase):

    def argTest(self, argKlass: Callable[Concatenate[str, _P], formmethod.Argument], testPairs: Iterable[tuple[object, object]], badValues: Iterable[object], *args: _P.args, **kwargs: _P.kwargs) -> None:
        if False:
            i = 10
            return i + 15
        arg = argKlass('name', *args, **kwargs)
        for (val, result) in testPairs:
            self.assertEqual(arg.coerce(val), result)
        for val in badValues:
            self.assertRaises(formmethod.InputError, arg.coerce, val)

    def test_argument(self) -> None:
        if False:
            i = 10
            return i + 15
        '\n        Test that corce correctly raises NotImplementedError.\n        '
        arg = formmethod.Argument('name')
        self.assertRaises(NotImplementedError, arg.coerce, '')

    def testString(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.argTest(formmethod.String, [('a', 'a'), (1, '1'), ('', '')], ())
        self.argTest(formmethod.String, [('ab', 'ab'), ('abc', 'abc')], ('2', ''), min=2)
        self.argTest(formmethod.String, [('ab', 'ab'), ('a', 'a')], ('223213', '345x'), max=3)
        self.argTest(formmethod.String, [('ab', 'ab'), ('add', 'add')], ('223213', 'x'), min=2, max=3)

    def testInt(self) -> None:
        if False:
            while True:
                i = 10
        self.argTest(formmethod.Integer, [('3', 3), ('-2', -2), ('', None)], ('q', '2.3'))
        self.argTest(formmethod.Integer, [('3', 3), ('-2', -2)], ('q', '2.3', ''), allowNone=0)

    def testFloat(self) -> None:
        if False:
            return 10
        self.argTest(formmethod.Float, [('3', 3.0), ('-2.3', -2.3), ('', None)], ('q', '2.3z'))
        self.argTest(formmethod.Float, [('3', 3.0), ('-2.3', -2.3)], ('q', '2.3z', ''), allowNone=0)

    def testChoice(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        choices = [('a', 'apple', 'an apple'), ('b', 'banana', 'ook')]
        self.argTest(formmethod.Choice, [('a', 'apple'), ('b', 'banana')], ('c', 1), choices=choices)

    def testFlags(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        flags = [('a', 'apple', 'an apple'), ('b', 'banana', 'ook')]
        self.argTest(formmethod.Flags, [(['a'], ['apple']), (['b', 'a'], ['banana', 'apple'])], (['a', 'c'], ['fdfs']), flags=flags)

    def testBoolean(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        tests = [('yes', 1), ('', 0), ('False', 0), ('no', 0)]
        self.argTest(formmethod.Boolean, tests, ())

    def test_file(self) -> None:
        if False:
            print('Hello World!')
        '\n        Test the correctness of the coerce function.\n        '
        arg = formmethod.File('name', allowNone=0)
        self.assertEqual(arg.coerce('something'), 'something')
        self.assertRaises(formmethod.InputError, arg.coerce, None)
        arg2 = formmethod.File('name')
        self.assertIsNone(arg2.coerce(None))

    def testDate(self) -> None:
        if False:
            return 10
        goodTests = {('2002', '12', '21'): (2002, 12, 21), ('1996', '2', '29'): (1996, 2, 29), ('', '', ''): None}.items()
        badTests = [('2002', '2', '29'), ('xx', '2', '3'), ('2002', '13', '1'), ('1999', '12', '32'), ('2002', '1'), ('2002', '2', '3', '4')]
        self.argTest(formmethod.Date, goodTests, badTests)

    def testRangedInteger(self) -> None:
        if False:
            i = 10
            return i + 15
        goodTests = {'0': 0, '12': 12, '3': 3}.items()
        badTests = ['-1', 'x', '13', '-2000', '3.4']
        self.argTest(formmethod.IntegerRange, goodTests, badTests, 0, 12)

    def testVerifiedPassword(self) -> None:
        if False:
            i = 10
            return i + 15
        goodTests = {('foo', 'foo'): 'foo', ('ab', 'ab'): 'ab'}.items()
        badTests = [('ab', 'a'), ('12345', '12345'), ('', ''), ('a', 'a'), ('a',), ('a', 'a', 'a')]
        self.argTest(formmethod.VerifiedPassword, goodTests, badTests, min=2, max=4)