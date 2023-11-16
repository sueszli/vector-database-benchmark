"""Tests for the decorators module."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from fire import core
from fire import decorators
from fire import testutils

class NoDefaults(object):
    """A class for testing decorated functions without default values."""

    @decorators.SetParseFns(count=int)
    def double(self, count):
        if False:
            while True:
                i = 10
        return 2 * count

    @decorators.SetParseFns(count=float)
    def triple(self, count):
        if False:
            print('Hello World!')
        return 3 * count

    @decorators.SetParseFns(int)
    def quadruple(self, count):
        if False:
            print('Hello World!')
        return 4 * count

@decorators.SetParseFns(int)
def double(count):
    if False:
        for i in range(10):
            print('nop')
    return 2 * count

class WithDefaults(object):

    @decorators.SetParseFns(float)
    def example1(self, arg1=10):
        if False:
            print('Hello World!')
        return (arg1, type(arg1))

    @decorators.SetParseFns(arg1=float)
    def example2(self, arg1=10):
        if False:
            while True:
                i = 10
        return (arg1, type(arg1))

class MixedArguments(object):

    @decorators.SetParseFns(float, arg2=str)
    def example3(self, arg1, arg2):
        if False:
            while True:
                i = 10
        return (arg1, arg2)

class PartialParseFn(object):

    @decorators.SetParseFns(arg1=str)
    def example4(self, arg1, arg2):
        if False:
            print('Hello World!')
        return (arg1, arg2)

    @decorators.SetParseFns(arg2=str)
    def example5(self, arg1, arg2):
        if False:
            print('Hello World!')
        return (arg1, arg2)

class WithKwargs(object):

    @decorators.SetParseFns(mode=str, count=int)
    def example6(self, **kwargs):
        if False:
            return 10
        return (kwargs.get('mode', 'default'), kwargs.get('count', 0))

class WithVarArgs(object):

    @decorators.SetParseFn(str)
    def example7(self, arg1, arg2=None, *varargs, **kwargs):
        if False:
            i = 10
            return i + 15
        return (arg1, arg2, varargs, kwargs)

class FireDecoratorsTest(testutils.BaseTestCase):

    def testSetParseFnsNamedArgs(self):
        if False:
            return 10
        self.assertEqual(core.Fire(NoDefaults, command=['double', '2']), 4)
        self.assertEqual(core.Fire(NoDefaults, command=['triple', '4']), 12.0)

    def testSetParseFnsPositionalArgs(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(core.Fire(NoDefaults, command=['quadruple', '5']), 20)

    def testSetParseFnsFnWithPositionalArgs(self):
        if False:
            print('Hello World!')
        self.assertEqual(core.Fire(double, command=['5']), 10)

    def testSetParseFnsDefaultsFromPython(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertTupleEqual(WithDefaults().example1(), (10, int))
        self.assertEqual(WithDefaults().example1(5), (5, int))
        self.assertEqual(WithDefaults().example1(12.0), (12, float))

    def testSetParseFnsDefaultsFromFire(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(core.Fire(WithDefaults, command=['example1']), (10, int))
        self.assertEqual(core.Fire(WithDefaults, command=['example1', '10']), (10, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example1', '13']), (13, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example1', '14.0']), (14, float))

    def testSetParseFnsNamedDefaultsFromPython(self):
        if False:
            print('Hello World!')
        self.assertTupleEqual(WithDefaults().example2(), (10, int))
        self.assertEqual(WithDefaults().example2(5), (5, int))
        self.assertEqual(WithDefaults().example2(12.0), (12, float))

    def testSetParseFnsNamedDefaultsFromFire(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(core.Fire(WithDefaults, command=['example2']), (10, int))
        self.assertEqual(core.Fire(WithDefaults, command=['example2', '10']), (10, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example2', '13']), (13, float))
        self.assertEqual(core.Fire(WithDefaults, command=['example2', '14.0']), (14, float))

    def testSetParseFnsPositionalAndNamed(self):
        if False:
            while True:
                i = 10
        self.assertEqual(core.Fire(MixedArguments, ['example3', '10', '10']), (10, '10'))

    def testSetParseFnsOnlySomeTypes(self):
        if False:
            while True:
                i = 10
        self.assertEqual(core.Fire(PartialParseFn, command=['example4', '10', '10']), ('10', 10))
        self.assertEqual(core.Fire(PartialParseFn, command=['example5', '10', '10']), (10, '10'))

    def testSetParseFnsForKeywordArgs(self):
        if False:
            return 10
        self.assertEqual(core.Fire(WithKwargs, command=['example6']), ('default', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--herring', '"red"']), ('default', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', 'train']), ('train', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', '3']), ('3', 0))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--mode', '-1', '--count', '10']), ('-1', 10))
        self.assertEqual(core.Fire(WithKwargs, command=['example6', '--count', '-2']), ('default', -2))

    def testSetParseFn(self):
        if False:
            print('Hello World!')
        self.assertEqual(core.Fire(WithVarArgs, command=['example7', '1', '--arg2=2', '3', '4', '--kwarg=5']), ('1', '2', ('3', '4'), {'kwarg': '5'}))
if __name__ == '__main__':
    testutils.main()