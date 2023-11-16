"""Tests for the fire module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import fire
from fire import test_components as tc
from fire import testutils
import mock
import six

class FireTest(testutils.BaseTestCase):

    def testFire(self):
        if False:
            for i in range(10):
                print('nop')
        with mock.patch.object(sys, 'argv', ['progname']):
            fire.Fire(tc.Empty)
            fire.Fire(tc.OldStyleEmpty)
            fire.Fire(tc.WithInit)
        self.assertEqual(fire.Fire(tc.NoDefaults, command='triple 4'), 12)
        self.assertEqual(fire.Fire(tc.WithDefaults, command=('double', '2')), 4)
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['triple', '4']), 12)
        self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['double', '2']), 4)
        self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['triple', '4']), 12)

    def testFirePositionalCommand(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.NoDefaults, 'double 2'), 4)
        self.assertEqual(fire.Fire(tc.NoDefaults, ['double', '2']), 4)

    def testFireInvalidCommandArg(self):
        if False:
            while True:
                i = 10
        with self.assertRaises(ValueError):
            fire.Fire(tc.WithDefaults, command=10)

    def testFireDefaultName(self):
        if False:
            print('Hello World!')
        with mock.patch.object(sys, 'argv', [os.path.join('python-fire', 'fire', 'base_filename.py')]):
            with self.assertOutputMatches(stdout='SYNOPSIS.*base_filename.py', stderr=None):
                fire.Fire(tc.Empty)

    def testFireNoArgs(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['ten']), 10)

    def testFireExceptions(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.Empty, command=['nomethod'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.NoDefaults, command=['double'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.TypedProperties, command=['delta', 'x'])
        with self.assertRaises(ZeroDivisionError):
            fire.Fire(tc.NumberDefaults, command=['reciprocal', '0.0'])

    def testFireNamedArgs(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['double', '--count', '5']), 10)
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['triple', '--count', '5']), 15)
        self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['double', '--count', '5']), 10)
        self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['triple', '--count', '5']), 15)

    def testFireNamedArgsSingleHyphen(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['double', '-count', '5']), 10)
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['triple', '-count', '5']), 15)
        self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['double', '-count', '5']), 10)
        self.assertEqual(fire.Fire(tc.OldStyleWithDefaults, command=['triple', '-count', '5']), 15)

    def testFireNamedArgsWithEquals(self):
        if False:
            print('Hello World!')
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['double', '--count=5']), 10)
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['triple', '--count=5']), 15)

    def testFireNamedArgsWithEqualsSingleHyphen(self):
        if False:
            while True:
                i = 10
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['double', '-count=5']), 10)
        self.assertEqual(fire.Fire(tc.WithDefaults, command=['triple', '-count=5']), 15)

    def testFireAllNamedArgs(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1', '2']), 5)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '1', '2']), 5)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--beta', '1', '2']), 4)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1', '--alpha', '2']), 4)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1', '--beta', '2']), 5)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '1', '--beta', '2']), 5)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--beta', '1', '--alpha', '2']), 4)

    def testFireAllNamedArgsOneMissing(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum']), 0)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '1']), 1)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '1']), 1)
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--beta', '2']), 4)

    def testFirePartialNamedArgs(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1', '2']), (1, 2))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '1', '2']), (1, 2))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--beta', '1', '2']), (2, 1))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1', '--alpha', '2']), (2, 1))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1', '--beta', '2']), (1, 2))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '1', '--beta', '2']), (1, 2))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--beta', '1', '--alpha', '2']), (2, 1))

    def testFirePartialNamedArgsOneMissing(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.MixedDefaults, command=['identity'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.MixedDefaults, command=['identity', '--beta', '2'])
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '1']), (1, '0'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '1']), (1, '0'))

    def testFireAnnotatedArgs(self):
        if False:
            while True:
                i = 10
        self.assertEqual(fire.Fire(tc.Annotations, command=['double', '5']), 10)
        self.assertEqual(fire.Fire(tc.Annotations, command=['triple', '5']), 15)

    @testutils.skipIf(six.PY2, 'Keyword-only arguments not in Python 2.')
    def testFireKeywordOnlyArgs(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.py3.KeywordOnly, command=['double', '5'])
        self.assertEqual(fire.Fire(tc.py3.KeywordOnly, command=['double', '--count', '5']), 10)
        self.assertEqual(fire.Fire(tc.py3.KeywordOnly, command=['triple', '--count', '5']), 15)

    def testFireProperties(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['alpha']), True)
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['beta']), (1, 2, 3))

    def testFireRecursion(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['charlie', 'double', 'hello']), 'hellohello')
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['charlie', 'triple', 'w']), 'www')

    def testFireVarArgs(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.VarArgs, command=['cumsums', 'a', 'b', 'c', 'd']), ['a', 'ab', 'abc', 'abcd'])
        self.assertEqual(fire.Fire(tc.VarArgs, command=['cumsums', '1', '2', '3', '4']), [1, 3, 6, 10])

    def testFireVarArgsWithNamedArgs(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.VarArgs, command=['varchars', '1', '2', 'c', 'd']), (1, 2, 'cd'))
        self.assertEqual(fire.Fire(tc.VarArgs, command=['varchars', '3', '4', 'c', 'd', 'e']), (3, 4, 'cde'))

    def testFireKeywordArgs(self):
        if False:
            while True:
                i = 10
        self.assertEqual(fire.Fire(tc.Kwargs, command=['props', '--name', 'David', '--age', '24']), {'name': 'David', 'age': 24})
        self.assertEqual(fire.Fire(tc.Kwargs, command=['props', '--message', '"This is a message it has -- in it"']), {'message': 'This is a message it has -- in it'})
        self.assertEqual(fire.Fire(tc.Kwargs, command=['props', '--message', 'This is a message it has -- in it']), {'message': 'This is a message it has -- in it'})
        self.assertEqual(fire.Fire(tc.Kwargs, command='props --message "This is a message it has -- in it"'), {'message': 'This is a message it has -- in it'})
        self.assertEqual(fire.Fire(tc.Kwargs, command=['upper', '--alpha', 'A', '--beta', 'B']), 'ALPHA BETA')
        self.assertEqual(fire.Fire(tc.Kwargs, command=['upper', '--alpha', 'A', '--beta', 'B', '-', 'lower']), 'alpha beta')

    def testFireKeywordArgsWithMissingPositionalArgs(self):
        if False:
            print('Hello World!')
        self.assertEqual(fire.Fire(tc.Kwargs, command=['run', 'Hello', 'World', '--cell', 'is']), ('Hello', 'World', {'cell': 'is'}))
        self.assertEqual(fire.Fire(tc.Kwargs, command=['run', 'Hello', '--cell', 'ok']), ('Hello', None, {'cell': 'ok'}))

    def testFireObject(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(fire.Fire(tc.WithDefaults(), command=['double', '--count', '5']), 10)
        self.assertEqual(fire.Fire(tc.WithDefaults(), command=['triple', '--count', '5']), 15)

    def testFireDict(self):
        if False:
            print('Hello World!')
        component = {'double': lambda x=0: 2 * x, 'cheese': 'swiss'}
        self.assertEqual(fire.Fire(component, command=['double', '5']), 10)
        self.assertEqual(fire.Fire(component, command=['cheese']), 'swiss')

    def testFireObjectWithDict(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['delta', 'echo']), 'E')
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['delta', 'echo', 'lower']), 'e')
        self.assertIsInstance(fire.Fire(tc.TypedProperties, command=['delta', 'nest']), dict)
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['delta', 'nest', '0']), 'a')

    def testFireSet(self):
        if False:
            for i in range(10):
                print('nop')
        component = tc.simple_set()
        result = fire.Fire(component, command=[])
        self.assertEqual(len(result), 3)

    def testFireFrozenset(self):
        if False:
            while True:
                i = 10
        component = tc.simple_frozenset()
        result = fire.Fire(component, command=[])
        self.assertEqual(len(result), 3)

    def testFireList(self):
        if False:
            i = 10
            return i + 15
        component = ['zero', 'one', 'two', 'three']
        self.assertEqual(fire.Fire(component, command=['2']), 'two')
        self.assertEqual(fire.Fire(component, command=['3']), 'three')
        self.assertEqual(fire.Fire(component, command=['-1']), 'three')

    def testFireObjectWithList(self):
        if False:
            while True:
                i = 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['echo', '0']), 'alex')
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['echo', '1']), 'bethany')

    def testFireObjectWithTuple(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['fox', '0']), 'carry')
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['fox', '1']), 'divide')

    def testFireObjectWithListAsObject(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['echo', 'count', 'bethany']), 1)

    def testFireObjectWithTupleAsObject(self):
        if False:
            while True:
                i = 10
        self.assertEqual(fire.Fire(tc.TypedProperties, command=['fox', 'count', 'divide']), 1)

    def testFireNoComponent(self):
        if False:
            print('Hello World!')
        self.assertEqual(fire.Fire(command=['tc', 'WithDefaults', 'double', '10']), 20)
        last_char = lambda text: text[-1]
        self.assertEqual(fire.Fire(command=['last_char', '"Hello"']), 'o')
        self.assertEqual(fire.Fire(command=['last-char', '"World"']), 'd')
        rset = lambda count=0: set(range(count))
        self.assertEqual(fire.Fire(command=['rset', '5']), {0, 1, 2, 3, 4})

    def testFireUnderscores(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.Underscores, command=['underscore-example']), 'fish fingers')
        self.assertEqual(fire.Fire(tc.Underscores, command=['underscore_example']), 'fish fingers')

    def testFireUnderscoresInArg(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.Underscores, command=['underscore-function', 'example']), 'example')
        self.assertEqual(fire.Fire(tc.Underscores, command=['underscore_function', '--underscore-arg=score']), 'score')
        self.assertEqual(fire.Fire(tc.Underscores, command=['underscore_function', '--underscore_arg=score']), 'score')

    def testBoolParsing(self):
        if False:
            while True:
                i = 10
        self.assertEqual(fire.Fire(tc.BoolConverter, command=['as-bool', 'True']), True)
        self.assertEqual(fire.Fire(tc.BoolConverter, command=['as-bool', 'False']), False)
        self.assertEqual(fire.Fire(tc.BoolConverter, command=['as-bool', '--arg=True']), True)
        self.assertEqual(fire.Fire(tc.BoolConverter, command=['as-bool', '--arg=False']), False)
        self.assertEqual(fire.Fire(tc.BoolConverter, command=['as-bool', '--arg']), True)
        self.assertEqual(fire.Fire(tc.BoolConverter, command=['as-bool', '--noarg']), False)

    def testBoolParsingContinued(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', 'True', 'False']), (True, False))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha=False', '10']), (False, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '--beta', '10']), (True, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '--beta=10']), (True, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--noalpha', '--beta']), (False, True))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '10', '--beta']), (10, True))

    def testBoolParsingSingleHyphen(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha=False', '10']), (False, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha', '-beta', '10']), (True, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha', '-beta=10']), (True, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-noalpha', '-beta']), (False, True))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-alpha', '-10', '-beta']), (-10, True))

    def testBoolParsingLessExpectedCases(self):
        if False:
            print('Hello World!')
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '10']), (10, '0'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '--beta=10']), (True, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', 'True', '10']), (True, 10))
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', '--test'])
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', 'True', '"--test"']), (True, '--test'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha=--test']), ('--test', '0'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command='identity --alpha \\"--test\\"'), ('--test', '0'))

    def testSingleCharFlagParsing(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a']), (True, '0'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a', '--beta=10']), (True, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a', '-b']), (True, True))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a', '42', '-b']), (42, True))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a', '42', '-b', '10']), (42, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '--alpha', 'True', '-b', '10']), (True, 10))
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.SimilarArgNames, command=['identity', '-b'])

    def testSingleCharFlagParsingEqualSign(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=True']), (True, '0'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=3', '--beta=10']), (3, 10))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=False', '-b=15']), (False, 15))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a', '42', '-b=12']), (42, 12))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-a=42', '-b', '10']), (42, 10))

    def testSingleCharFlagParsingExactMatch(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-a']), (True, None))
        self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-a=10']), (10, None))
        self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '--a']), (True, None))
        self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-alpha']), (None, True))
        self.assertEqual(fire.Fire(tc.SimilarArgNames, command=['identity2', '-a', '-alpha']), (True, True))

    def testSingleCharFlagParsingCapitalLetter(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.CapitalizedArgNames, command=['sum', '-D', '5', '-G', '10']), 15)

    def testBoolParsingWithNo(self):
        if False:
            for i in range(10):
                print('nop')

        def fn1(thing, nothing):
            if False:
                while True:
                    i = 10
            return (thing, nothing)
        self.assertEqual(fire.Fire(fn1, command=['--thing', '--nothing']), (True, True))
        self.assertEqual(fire.Fire(fn1, command=['--thing', '--nonothing']), (True, False))
        with self.assertRaisesFireExit(2):
            fire.Fire(fn1, command=['--nothing', '--nonothing'])

        def fn2(thing, **kwargs):
            if False:
                return 10
            return (thing, kwargs)
        self.assertEqual(fire.Fire(fn2, command=['--thing']), (True, {}))
        self.assertEqual(fire.Fire(fn2, command=['--nothing']), (False, {}))
        with self.assertRaisesFireExit(2):
            fire.Fire(fn2, command=['--nothing=True'])
        self.assertEqual(fire.Fire(fn2, command=['--nothing', '--nothing=True']), (False, {'nothing': True}))

        def fn3(arg, **kwargs):
            if False:
                return 10
            return (arg, kwargs)
        self.assertEqual(fire.Fire(fn3, command=['--arg=value', '--thing']), ('value', {'thing': True}))
        self.assertEqual(fire.Fire(fn3, command=['--arg=value', '--nothing']), ('value', {'thing': False}))
        self.assertEqual(fire.Fire(fn3, command=['--arg=value', '--nonothing']), ('value', {'nothing': False}))

    def testTraceFlag(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaisesFireExit(0, 'Fire trace:\n'):
            fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '--trace'])
        with self.assertRaisesFireExit(0, 'Fire trace:\n'):
            fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '-t'])
        with self.assertRaisesFireExit(0, 'Fire trace:\n'):
            fire.Fire(tc.BoolConverter, command=['--', '--trace'])

    def testHelpFlag(self):
        if False:
            return 10
        with self.assertRaisesFireExit(0):
            fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '--help'])
        with self.assertRaisesFireExit(0):
            fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '-h'])
        with self.assertRaisesFireExit(0):
            fire.Fire(tc.BoolConverter, command=['--', '--help'])

    def testHelpFlagAndTraceFlag(self):
        if False:
            while True:
                i = 10
        with self.assertRaisesFireExit(0, 'Fire trace:\n.*SYNOPSIS'):
            fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '--help', '--trace'])
        with self.assertRaisesFireExit(0, 'Fire trace:\n.*SYNOPSIS'):
            fire.Fire(tc.BoolConverter, command=['as-bool', 'True', '--', '-h', '-t'])
        with self.assertRaisesFireExit(0, 'Fire trace:\n.*SYNOPSIS'):
            fire.Fire(tc.BoolConverter, command=['--', '-h', '--trace'])

    def testTabCompletionNoName(self):
        if False:
            while True:
                i = 10
        completion_script = fire.Fire(tc.NoDefaults, command=['--', '--completion'])
        self.assertIn('double', completion_script)
        self.assertIn('triple', completion_script)

    def testTabCompletion(self):
        if False:
            i = 10
            return i + 15
        completion_script = fire.Fire(tc.NoDefaults, command=['--', '--completion'], name='c')
        self.assertIn('double', completion_script)
        self.assertIn('triple', completion_script)

    def testTabCompletionWithDict(self):
        if False:
            while True:
                i = 10
        actions = {'multiply': lambda a, b: a * b}
        completion_script = fire.Fire(actions, command=['--', '--completion'], name='actCLI')
        self.assertIn('actCLI', completion_script)
        self.assertIn('multiply', completion_script)

    def testBasicSeparator(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '+', '_']), ('+', '_'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '_', '+', '-']), ('_', '+'))
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['identity', '-', '_', '--', '--separator', '&']), ('-', '_'))
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.MixedDefaults, command=['identity', '-', '_', '+'])

    def testNonComparable(self):
        if False:
            for i in range(10):
                print('nop')
        'Fire should work with classes that disallow comparisons.'
        self.assertIsInstance(fire.Fire(tc.NonComparable, command=''), tc.NonComparable)
        self.assertIsInstance(fire.Fire(tc.NonComparable, command=[]), tc.NonComparable)
        self.assertIsInstance(fire.Fire(tc.NonComparable, command=['-', '-']), tc.NonComparable)

    def testExtraSeparators(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', '-', '-', 'as-bool', 'True']), True)
        self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', '-', '-', '-', 'as-bool', 'True']), True)

    def testSeparatorForChaining(self):
        if False:
            i = 10
            return i + 15
        self.assertIsInstance(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', 'as-bool', 'True']), tc.BoolConverter)
        self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', '-', 'as-bool', 'True']), True)
        self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', 'arg2', '&', 'as-bool', 'True', '--', '--separator', '&']), True)
        self.assertEqual(fire.Fire(tc.ReturnsObj, command=['get-obj', 'arg1', '$$', 'as-bool', 'True', '--', '--separator', '$$']), True)

    def testNegativeNumbers(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '-3', '--beta', '-4']), -11)

    def testFloatForExpectedInt(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.MixedDefaults, command=['sum', '--alpha', '2.2', '--beta', '3.0']), 8.2)
        self.assertEqual(fire.Fire(tc.NumberDefaults, command=['integer_reciprocal', '--divisor', '5.0']), 0.2)
        self.assertEqual(fire.Fire(tc.NumberDefaults, command=['integer_reciprocal', '4.0']), 0.25)

    def testClassInstantiation(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertIsInstance(fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2']), tc.InstanceVars)
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['a1', 'a2'])

    def testTraceErrors(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['a1'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['--arg1=a1'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['a1', 'a2', '-', 'run', 'b1'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2', '-', 'run b1'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['a1', 'a2', '-', 'run', 'b1', 'b2', 'b3'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2', '-', 'run', 'b1', 'b2', 'b3'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['a1', 'a2', '-', 'jog'])
        with self.assertRaisesFireExit(2):
            fire.Fire(tc.InstanceVars, command=['--arg1=a1', '--arg2=a2', '-', 'jog'])

    def testClassWithDefaultMethod(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.DefaultMethod, command=['double', '10']), 20)

    def testClassWithInvalidProperty(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(fire.Fire(tc.InvalidProperty, command=['double', '10']), 20)

    @testutils.skipIf(sys.version_info[0:2] <= (3, 4), 'Cannot inspect wrapped signatures in Python 2 or 3.4.')
    def testHelpKwargsDecorator(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(0):
            fire.Fire(tc.decorated_method, command=['-h'])
        with self.assertRaisesFireExit(0):
            fire.Fire(tc.decorated_method, command=['--help'])

    @testutils.skipIf(six.PY2, 'Asyncio not available in Python 2.')
    def testFireAsyncio(self):
        if False:
            return 10
        self.assertEqual(fire.Fire(tc.py3.WithAsyncio, command=['double', '--count', '10']), 20)
if __name__ == '__main__':
    testutils.main()