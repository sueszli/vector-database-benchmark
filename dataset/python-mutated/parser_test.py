"""Tests for the parser module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import parser
from fire import testutils

class ParserTest(testutils.BaseTestCase):

    def testCreateParser(self):
        if False:
            while True:
                i = 10
        self.assertIsNotNone(parser.CreateParser())

    def testSeparateFlagArgs(self):
        if False:
            print('Hello World!')
        self.assertEqual(parser.SeparateFlagArgs([]), ([], []))
        self.assertEqual(parser.SeparateFlagArgs(['a', 'b']), (['a', 'b'], []))
        self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--']), (['a', 'b'], []))
        self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c']), (['a', 'b'], ['c']))
        self.assertEqual(parser.SeparateFlagArgs(['--']), ([], []))
        self.assertEqual(parser.SeparateFlagArgs(['--', 'c', 'd']), ([], ['c', 'd']))
        self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c', 'd']), (['a', 'b'], ['c', 'd']))
        self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c', 'd', '--']), (['a', 'b', '--', 'c', 'd'], []))
        self.assertEqual(parser.SeparateFlagArgs(['a', 'b', '--', 'c', '--', 'd']), (['a', 'b', '--', 'c'], ['d']))

    def testDefaultParseValueStrings(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parser.DefaultParseValue('hello'), 'hello')
        self.assertEqual(parser.DefaultParseValue('path/file.jpg'), 'path/file.jpg')
        self.assertEqual(parser.DefaultParseValue('hello world'), 'hello world')
        self.assertEqual(parser.DefaultParseValue('--flag'), '--flag')

    def testDefaultParseValueQuotedStrings(self):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.DefaultParseValue("'hello'"), 'hello')
        self.assertEqual(parser.DefaultParseValue("'hello world'"), 'hello world')
        self.assertEqual(parser.DefaultParseValue("'--flag'"), '--flag')
        self.assertEqual(parser.DefaultParseValue('"hello"'), 'hello')
        self.assertEqual(parser.DefaultParseValue('"hello world"'), 'hello world')
        self.assertEqual(parser.DefaultParseValue('"--flag"'), '--flag')

    def testDefaultParseValueSpecialStrings(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(parser.DefaultParseValue('-'), '-')
        self.assertEqual(parser.DefaultParseValue('--'), '--')
        self.assertEqual(parser.DefaultParseValue('---'), '---')
        self.assertEqual(parser.DefaultParseValue('----'), '----')
        self.assertEqual(parser.DefaultParseValue('None'), None)
        self.assertEqual(parser.DefaultParseValue("'None'"), 'None')

    def testDefaultParseValueNumbers(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(parser.DefaultParseValue('23'), 23)
        self.assertEqual(parser.DefaultParseValue('-23'), -23)
        self.assertEqual(parser.DefaultParseValue('23.0'), 23.0)
        self.assertIsInstance(parser.DefaultParseValue('23'), int)
        self.assertIsInstance(parser.DefaultParseValue('23.0'), float)
        self.assertEqual(parser.DefaultParseValue('23.5'), 23.5)
        self.assertEqual(parser.DefaultParseValue('-23.5'), -23.5)

    def testDefaultParseValueStringNumbers(self):
        if False:
            return 10
        self.assertEqual(parser.DefaultParseValue("'23'"), '23')
        self.assertEqual(parser.DefaultParseValue("'23.0'"), '23.0')
        self.assertEqual(parser.DefaultParseValue("'23.5'"), '23.5')
        self.assertEqual(parser.DefaultParseValue('"23"'), '23')
        self.assertEqual(parser.DefaultParseValue('"23.0"'), '23.0')
        self.assertEqual(parser.DefaultParseValue('"23.5"'), '23.5')

    def testDefaultParseValueQuotedStringNumbers(self):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.DefaultParseValue('"\'123\'"'), "'123'")

    def testDefaultParseValueOtherNumbers(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parser.DefaultParseValue('1e5'), 100000.0)

    def testDefaultParseValueLists(self):
        if False:
            return 10
        self.assertEqual(parser.DefaultParseValue('[1, 2, 3]'), [1, 2, 3])
        self.assertEqual(parser.DefaultParseValue('[1, "2", 3]'), [1, '2', 3])
        self.assertEqual(parser.DefaultParseValue('[1, \'"2"\', 3]'), [1, '"2"', 3])
        self.assertEqual(parser.DefaultParseValue('[1, "hello", 3]'), [1, 'hello', 3])

    def testDefaultParseValueBareWordsLists(self):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.DefaultParseValue('[one, 2, "3"]'), ['one', 2, '3'])

    def testDefaultParseValueDict(self):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.DefaultParseValue('{"abc": 5, "123": 1}'), {'abc': 5, '123': 1})

    def testDefaultParseValueNone(self):
        if False:
            return 10
        self.assertEqual(parser.DefaultParseValue('None'), None)

    def testDefaultParseValueBool(self):
        if False:
            return 10
        self.assertEqual(parser.DefaultParseValue('True'), True)
        self.assertEqual(parser.DefaultParseValue('False'), False)

    def testDefaultParseValueBareWordsTuple(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parser.DefaultParseValue('(one, 2, "3")'), ('one', 2, '3'))
        self.assertEqual(parser.DefaultParseValue('one, "2", 3'), ('one', '2', 3))

    def testDefaultParseValueNestedContainers(self):
        if False:
            return 10
        self.assertEqual(parser.DefaultParseValue('[(A, 2, "3"), 5, {alpha: 10.2, beta: "cat"}]'), [('A', 2, '3'), 5, {'alpha': 10.2, 'beta': 'cat'}])

    def testDefaultParseValueComments(self):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.DefaultParseValue('"0#comments"'), '0#comments')
        self.assertEqual(parser.DefaultParseValue('0#comments'), 0)

    def testDefaultParseValueBadLiteral(self):
        if False:
            return 10
        self.assertEqual(parser.DefaultParseValue('[(A, 2, "3"), 5'), '[(A, 2, "3"), 5')
        self.assertEqual(parser.DefaultParseValue('x=10'), 'x=10')

    def testDefaultParseValueSyntaxError(self):
        if False:
            while True:
                i = 10
        self.assertEqual(parser.DefaultParseValue('"'), '"')

    def testDefaultParseValueIgnoreBinOp(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(parser.DefaultParseValue('2017-10-10'), '2017-10-10')
        self.assertEqual(parser.DefaultParseValue('1+1'), '1+1')
if __name__ == '__main__':
    testutils.main()