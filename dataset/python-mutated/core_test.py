"""Tests for the core module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import test_components as tc
from fire import testutils
from fire import trace
import mock
import six

class CoreTest(testutils.BaseTestCase):

    def testOneLineResult(self):
        if False:
            while True:
                i = 10
        self.assertEqual(core._OneLineResult(1), '1')
        self.assertEqual(core._OneLineResult('hello'), 'hello')
        self.assertEqual(core._OneLineResult({}), '{}')
        self.assertEqual(core._OneLineResult({'x': 'y'}), '{"x": "y"}')

    def testOneLineResultCircularRef(self):
        if False:
            i = 10
            return i + 15
        circular_reference = tc.CircularReference()
        self.assertEqual(core._OneLineResult(circular_reference.create()), "{'y': {...}}")

    @mock.patch('fire.interact.Embed')
    def testInteractiveMode(self, mock_embed):
        if False:
            print('Hello World!')
        core.Fire(tc.TypedProperties, command=['alpha'])
        self.assertFalse(mock_embed.called)
        core.Fire(tc.TypedProperties, command=['alpha', '--', '-i'])
        self.assertTrue(mock_embed.called)

    @mock.patch('fire.interact.Embed')
    def testInteractiveModeFullArgument(self, mock_embed):
        if False:
            for i in range(10):
                print('nop')
        core.Fire(tc.TypedProperties, command=['alpha', '--', '--interactive'])
        self.assertTrue(mock_embed.called)

    @mock.patch('fire.interact.Embed')
    def testInteractiveModeVariables(self, mock_embed):
        if False:
            i = 10
            return i + 15
        core.Fire(tc.WithDefaults, command=['double', '2', '--', '-i'])
        self.assertTrue(mock_embed.called)
        ((variables, verbose), unused_kwargs) = mock_embed.call_args
        self.assertFalse(verbose)
        self.assertEqual(variables['result'], 4)
        self.assertIsInstance(variables['self'], tc.WithDefaults)
        self.assertIsInstance(variables['trace'], trace.FireTrace)

    @mock.patch('fire.interact.Embed')
    def testInteractiveModeVariablesWithName(self, mock_embed):
        if False:
            print('Hello World!')
        core.Fire(tc.WithDefaults, command=['double', '2', '--', '-i', '-v'], name='D')
        self.assertTrue(mock_embed.called)
        ((variables, verbose), unused_kwargs) = mock_embed.call_args
        self.assertTrue(verbose)
        self.assertEqual(variables['result'], 4)
        self.assertIsInstance(variables['self'], tc.WithDefaults)
        self.assertEqual(variables['D'], tc.WithDefaults)
        self.assertIsInstance(variables['trace'], trace.FireTrace)

    def testHelpWithClass(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(0, 'SYNOPSIS.*ARG1'):
            core.Fire(tc.InstanceVars, command=['--', '--help'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*ARG1'):
            core.Fire(tc.InstanceVars, command=['--help'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*ARG1'):
            core.Fire(tc.InstanceVars, command=['-h'])

    def testHelpWithMember(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(0, 'SYNOPSIS.*capitalize'):
            core.Fire(tc.TypedProperties, command=['gamma', '--', '--help'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*capitalize'):
            core.Fire(tc.TypedProperties, command=['gamma', '--help'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*capitalize'):
            core.Fire(tc.TypedProperties, command=['gamma', '-h'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*delta'):
            core.Fire(tc.TypedProperties, command=['delta', '--help'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*echo'):
            core.Fire(tc.TypedProperties, command=['echo', '--help'])

    def testHelpOnErrorInConstructor(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(0, 'SYNOPSIS.*VALUE'):
            core.Fire(tc.ErrorInConstructor, command=['--', '--help'])
        with self.assertRaisesFireExit(0, 'INFO:.*SYNOPSIS.*VALUE'):
            core.Fire(tc.ErrorInConstructor, command=['--help'])

    def testHelpWithNamespaceCollision(self):
        if False:
            i = 10
            return i + 15
        with self.assertOutputMatches(stdout='DESCRIPTION.*', stderr=None):
            core.Fire(tc.WithHelpArg, command=['--help', 'False'])
        with self.assertOutputMatches(stdout='help in a dict', stderr=None):
            core.Fire(tc.WithHelpArg, command=['dictionary', '__help'])
        with self.assertOutputMatches(stdout='{}', stderr=None):
            core.Fire(tc.WithHelpArg, command=['dictionary', '--help'])
        with self.assertOutputMatches(stdout='False', stderr=None):
            core.Fire(tc.function_with_help, command=['False'])

    def testInvalidParameterRaisesFireExit(self):
        if False:
            print('Hello World!')
        with self.assertRaisesFireExit(2, 'runmisspelled'):
            core.Fire(tc.Kwargs, command=['props', '--a=1', '--b=2', 'runmisspelled'])

    def testErrorRaising(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(ValueError):
            core.Fire(tc.ErrorRaiser, command=['fail'])

    def testFireError(self):
        if False:
            i = 10
            return i + 15
        error = core.FireError('Example error')
        self.assertIsNotNone(error)

    def testFireErrorMultipleValues(self):
        if False:
            i = 10
            return i + 15
        error = core.FireError('Example error', 'value')
        self.assertIsNotNone(error)

    def testPrintEmptyDict(self):
        if False:
            while True:
                i = 10
        with self.assertOutputMatches(stdout='{}', stderr=None):
            core.Fire(tc.EmptyDictOutput, command=['totally_empty'])
        with self.assertOutputMatches(stdout='{}', stderr=None):
            core.Fire(tc.EmptyDictOutput, command=['nothing_printable'])

    def testPrintOrderedDict(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertOutputMatches(stdout='A:\\s+A\\s+2:\\s+2\\s+', stderr=None):
            core.Fire(tc.OrderedDictionary, command=['non_empty'])
        with self.assertOutputMatches(stdout='{}'):
            core.Fire(tc.OrderedDictionary, command=['empty'])

    def testPrintNamedTupleField(self):
        if False:
            i = 10
            return i + 15
        with self.assertOutputMatches(stdout='11', stderr=None):
            core.Fire(tc.NamedTuple, command=['point', 'x'])

    def testPrintNamedTupleFieldNameEqualsValue(self):
        if False:
            i = 10
            return i + 15
        with self.assertOutputMatches(stdout='x', stderr=None):
            core.Fire(tc.NamedTuple, command=['matching_names', 'x'])

    def testPrintNamedTupleIndex(self):
        if False:
            print('Hello World!')
        with self.assertOutputMatches(stdout='22', stderr=None):
            core.Fire(tc.NamedTuple, command=['point', '1'])

    def testPrintSet(self):
        if False:
            return 10
        with self.assertOutputMatches(stdout='.*three.*', stderr=None):
            core.Fire(tc.simple_set(), command=[])

    def testPrintFrozenSet(self):
        if False:
            while True:
                i = 10
        with self.assertOutputMatches(stdout='.*three.*', stderr=None):
            core.Fire(tc.simple_frozenset(), command=[])

    def testPrintNamedTupleNegativeIndex(self):
        if False:
            i = 10
            return i + 15
        with self.assertOutputMatches(stdout='11', stderr=None):
            core.Fire(tc.NamedTuple, command=['point', '-2'])

    def testCallable(self):
        if False:
            i = 10
            return i + 15
        with self.assertOutputMatches(stdout='foo:\\s+foo\\s+', stderr=None):
            core.Fire(tc.CallableWithKeywordArgument(), command=['--foo=foo'])
        with self.assertOutputMatches(stdout='foo\\s+', stderr=None):
            core.Fire(tc.CallableWithKeywordArgument(), command=['print_msg', 'foo'])
        with self.assertOutputMatches(stdout='', stderr=None):
            core.Fire(tc.CallableWithKeywordArgument(), command=[])

    def testCallableWithPositionalArgs(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesFireExit(2, ''):
            core.Fire(tc.CallableWithPositionalArgs(), command=['3', '4'])

    def testStaticMethod(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(core.Fire(tc.HasStaticAndClassMethods, command=['static_fn', 'alpha']), 'alpha')

    def testClassMethod(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(core.Fire(tc.HasStaticAndClassMethods, command=['class_fn', '6']), 7)

    def testCustomSerialize(self):
        if False:
            i = 10
            return i + 15

        def serialize(x):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(x, list):
                return ', '.join((str(xi) for xi in x))
            if isinstance(x, dict):
                return ', '.join(('{}={!r}'.format(k, v) for (k, v) in sorted(x.items())))
            if x == 'special':
                return ['SURPRISE!!', "I'm a list!"]
            return x
        ident = lambda x: x
        with self.assertOutputMatches(stdout='a, b', stderr=None):
            _ = core.Fire(ident, command=['[a,b]'], serialize=serialize)
        with self.assertOutputMatches(stdout='a=5, b=6', stderr=None):
            _ = core.Fire(ident, command=['{a:5,b:6}'], serialize=serialize)
        with self.assertOutputMatches(stdout='asdf', stderr=None):
            _ = core.Fire(ident, command=['asdf'], serialize=serialize)
        with self.assertOutputMatches(stdout="SURPRISE!!\nI'm a list!\n", stderr=None):
            _ = core.Fire(ident, command=['special'], serialize=serialize)
        with self.assertRaises(core.FireError):
            core.Fire(ident, command=['asdf'], serialize=55)

    @testutils.skipIf(six.PY2, 'lru_cache is Python 3 only.')
    def testLruCacheDecoratorBoundArg(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(core.Fire(tc.py3.LruCacheDecoratedMethod, command=['lru_cache_in_class', 'foo']), 'foo')

    @testutils.skipIf(six.PY2, 'lru_cache is Python 3 only.')
    def testLruCacheDecorator(self):
        if False:
            while True:
                i = 10
        self.assertEqual(core.Fire(tc.py3.lru_cache_decorated, command=['foo']), 'foo')
if __name__ == '__main__':
    testutils.main()