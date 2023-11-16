"""Tests for the trace module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import testutils
from fire import trace

class FireTraceTest(testutils.BaseTestCase):

    def testFireTraceInitialization(self):
        if False:
            while True:
                i = 10
        t = trace.FireTrace(10)
        self.assertIsNotNone(t)
        self.assertIsNotNone(t.elements)

    def testFireTraceGetResult(self):
        if False:
            for i in range(10):
                print('nop')
        t = trace.FireTrace('start')
        self.assertEqual(t.GetResult(), 'start')
        t.AddAccessedProperty('t', 'final', None, 'example.py', 10)
        self.assertEqual(t.GetResult(), 't')

    def testFireTraceHasError(self):
        if False:
            while True:
                i = 10
        t = trace.FireTrace('start')
        self.assertFalse(t.HasError())
        t.AddAccessedProperty('t', 'final', None, 'example.py', 10)
        self.assertFalse(t.HasError())
        t.AddError(ValueError('example error'), ['arg'])
        self.assertTrue(t.HasError())

    def testAddAccessedProperty(self):
        if False:
            for i in range(10):
                print('nop')
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddAccessedProperty('new component', 'prop', args, 'sample.py', 12)
        self.assertEqual(str(t), '1. Initial component\n2. Accessed property "prop" (sample.py:12)')

    def testAddCalledCallable(self):
        if False:
            for i in range(10):
                print('nop')
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('result', 'cell', args, 'sample.py', 10, False, action=trace.CALLED_CALLABLE)
        self.assertEqual(str(t), '1. Initial component\n2. Called callable "cell" (sample.py:10)')

    def testAddCalledRoutine(self):
        if False:
            i = 10
            return i + 15
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(str(t), '1. Initial component\n2. Called routine "run" (sample.py:12)')

    def testAddInstantiatedClass(self):
        if False:
            for i in range(10):
                print('nop')
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('Classname', 'classname', args, 'sample.py', 12, False, action=trace.INSTANTIATED_CLASS)
        target = '1. Initial component\n2. Instantiated class "classname" (sample.py:12)'
        self.assertEqual(str(t), target)

    def testAddCompletionScript(self):
        if False:
            for i in range(10):
                print('nop')
        t = trace.FireTrace('initial object')
        t.AddCompletionScript('This is the completion script string.')
        self.assertEqual(str(t), '1. Initial component\n2. Generated completion script')

    def testAddInteractiveMode(self):
        if False:
            print('Hello World!')
        t = trace.FireTrace('initial object')
        t.AddInteractiveMode()
        self.assertEqual(str(t), '1. Initial component\n2. Entered interactive mode')

    def testGetCommand(self):
        if False:
            for i in range(10):
                print('nop')
        t = trace.FireTrace('initial object')
        args = ('example', 'args')
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(t.GetCommand(), 'example args')

    def testGetCommandWithQuotes(self):
        if False:
            i = 10
            return i + 15
        t = trace.FireTrace('initial object')
        args = ('example', 'spaced arg')
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(t.GetCommand(), "example 'spaced arg'")

    def testGetCommandWithFlagQuotes(self):
        if False:
            i = 10
            return i + 15
        t = trace.FireTrace('initial object')
        args = ('--example=spaced arg',)
        t.AddCalledComponent('result', 'run', args, 'sample.py', 12, False, action=trace.CALLED_ROUTINE)
        self.assertEqual(t.GetCommand(), "--example='spaced arg'")

class FireTraceElementTest(testutils.BaseTestCase):

    def testFireTraceElementHasError(self):
        if False:
            i = 10
            return i + 15
        el = trace.FireTraceElement()
        self.assertFalse(el.HasError())
        el = trace.FireTraceElement(error=ValueError('example error'))
        self.assertTrue(el.HasError())

    def testFireTraceElementAsStringNoMetadata(self):
        if False:
            return 10
        el = trace.FireTraceElement(component='Example', action='Fake action')
        self.assertEqual(str(el), 'Fake action')

    def testFireTraceElementAsStringWithTarget(self):
        if False:
            i = 10
            return i + 15
        el = trace.FireTraceElement(component='Example', action='Created toy', target='Beaker')
        self.assertEqual(str(el), 'Created toy "Beaker"')

    def testFireTraceElementAsStringWithTargetAndLineNo(self):
        if False:
            while True:
                i = 10
        el = trace.FireTraceElement(component='Example', action='Created toy', target='Beaker', filename='beaker.py', lineno=10)
        self.assertEqual(str(el), 'Created toy "Beaker" (beaker.py:10)')
if __name__ == '__main__':
    testutils.main()