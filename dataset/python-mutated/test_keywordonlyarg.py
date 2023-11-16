"""Unit tests for the keyword only argument specified in PEP 3102."""
__author__ = 'Jiwon Seo'
__email__ = 'seojiwon at gmail dot com'
import unittest

def posonly_sum(pos_arg1, *arg, **kwarg):
    if False:
        return 10
    return pos_arg1 + sum(arg) + sum(kwarg.values())

def keywordonly_sum(*, k1=0, k2):
    if False:
        i = 10
        return i + 15
    return k1 + k2

def keywordonly_nodefaults_sum(*, k1, k2):
    if False:
        return 10
    return k1 + k2

def keywordonly_and_kwarg_sum(*, k1, k2, **kwarg):
    if False:
        return 10
    return k1 + k2 + sum(kwarg.values())

def mixedargs_sum(a, b=0, *arg, k1, k2=0):
    if False:
        print('Hello World!')
    return a + b + k1 + k2 + sum(arg)

def mixedargs_sum2(a, b=0, *arg, k1, k2=0, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return a + b + k1 + k2 + sum(arg) + sum(kwargs.values())

def sortnum(*nums, reverse=False):
    if False:
        return 10
    return sorted(list(nums), reverse=reverse)

def sortwords(*words, reverse=False, **kwargs):
    if False:
        print('Hello World!')
    return sorted(list(words), reverse=reverse)

class Foo:

    def __init__(self, *, k1, k2=0):
        if False:
            i = 10
            return i + 15
        self.k1 = k1
        self.k2 = k2

    def set(self, p1, *, k1, k2):
        if False:
            return 10
        self.k1 = k1
        self.k2 = k2

    def sum(self):
        if False:
            i = 10
            return i + 15
        return self.k1 + self.k2

class KeywordOnlyArgTestCase(unittest.TestCase):

    def assertRaisesSyntaxError(self, codestr):
        if False:
            return 10

        def shouldRaiseSyntaxError(s):
            if False:
                print('Hello World!')
            compile(s, '<test>', 'single')
        self.assertRaises(SyntaxError, shouldRaiseSyntaxError, codestr)

    def testSyntaxErrorForFunctionDefinition(self):
        if False:
            print('Hello World!')
        self.assertRaisesSyntaxError('def f(p, *):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p1, *, p1=100):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p1, *k1, k1=100):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p1, *, k1, k1=100):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p1, *, **k1):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p1, *, k1, **k1):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p1, *, None, **k1):\n  pass\n')
        self.assertRaisesSyntaxError('def f(p, *, (k1, k2), **kw):\n  pass\n')

    def testSyntaxForManyArguments(self):
        if False:
            for i in range(10):
                print('nop')
        fundef = 'def f(%s):\n  pass\n' % ', '.join(('i%d' % i for i in range(300)))
        compile(fundef, '<test>', 'single')
        fundef = 'def f(*, %s):\n  pass\n' % ', '.join(('i%d' % i for i in range(300)))
        compile(fundef, '<test>', 'single')

    def testTooManyPositionalErrorMessage(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b=None, *, c=None):
            if False:
                while True:
                    i = 10
            pass
        with self.assertRaises(TypeError) as exc:
            f(1, 2, 3)
        expected = f'{f.__qualname__}() takes from 1 to 2 positional arguments but 3 were given'
        self.assertEqual(str(exc.exception), expected)

    def testSyntaxErrorForFunctionCall(self):
        if False:
            return 10
        self.assertRaisesSyntaxError('f(p, k=1, p2)')
        self.assertRaisesSyntaxError('f(p, k1=50, *(1,2), k1=100)')

    def testRaiseErrorFuncallWithUnexpectedKeywordArgument(self):
        if False:
            print('Hello World!')
        self.assertRaises(TypeError, keywordonly_sum, ())
        self.assertRaises(TypeError, keywordonly_nodefaults_sum, ())
        self.assertRaises(TypeError, Foo, ())
        try:
            keywordonly_sum(k2=100, non_existing_arg=200)
            self.fail('should raise TypeError')
        except TypeError:
            pass
        try:
            keywordonly_nodefaults_sum(k2=2)
            self.fail('should raise TypeError')
        except TypeError:
            pass

    def testFunctionCall(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(1, posonly_sum(1))
        self.assertEqual(1 + 2, posonly_sum(1, **{'2': 2}))
        self.assertEqual(1 + 2 + 3, posonly_sum(1, *(2, 3)))
        self.assertEqual(1 + 2 + 3 + 4, posonly_sum(1, *(2, 3), **{'4': 4}))
        self.assertEqual(1, keywordonly_sum(k2=1))
        self.assertEqual(1 + 2, keywordonly_sum(k1=1, k2=2))
        self.assertEqual(1 + 2, keywordonly_and_kwarg_sum(k1=1, k2=2))
        self.assertEqual(1 + 2 + 3, keywordonly_and_kwarg_sum(k1=1, k2=2, k3=3))
        self.assertEqual(1 + 2 + 3 + 4, keywordonly_and_kwarg_sum(k1=1, k2=2, **{'a': 3, 'b': 4}))
        self.assertEqual(1 + 2, mixedargs_sum(1, k1=2))
        self.assertEqual(1 + 2 + 3, mixedargs_sum(1, 2, k1=3))
        self.assertEqual(1 + 2 + 3 + 4, mixedargs_sum(1, 2, k1=3, k2=4))
        self.assertEqual(1 + 2 + 3 + 4 + 5, mixedargs_sum(1, 2, 3, k1=4, k2=5))
        self.assertEqual(1 + 2, mixedargs_sum2(1, k1=2))
        self.assertEqual(1 + 2 + 3, mixedargs_sum2(1, 2, k1=3))
        self.assertEqual(1 + 2 + 3 + 4, mixedargs_sum2(1, 2, k1=3, k2=4))
        self.assertEqual(1 + 2 + 3 + 4 + 5, mixedargs_sum2(1, 2, 3, k1=4, k2=5))
        self.assertEqual(1 + 2 + 3 + 4 + 5 + 6, mixedargs_sum2(1, 2, 3, k1=4, k2=5, k3=6))
        self.assertEqual(1 + 2 + 3 + 4 + 5 + 6, mixedargs_sum2(1, 2, 3, k1=4, **{'k2': 5, 'k3': 6}))
        self.assertEqual(1, Foo(k1=1).sum())
        self.assertEqual(1 + 2, Foo(k1=1, k2=2).sum())
        self.assertEqual([1, 2, 3], sortnum(3, 2, 1))
        self.assertEqual([3, 2, 1], sortnum(1, 2, 3, reverse=True))
        self.assertEqual(['a', 'b', 'c'], sortwords('a', 'c', 'b'))
        self.assertEqual(['c', 'b', 'a'], sortwords('a', 'c', 'b', reverse=True))
        self.assertEqual(['c', 'b', 'a'], sortwords('a', 'c', 'b', reverse=True, ignore='ignore'))

    def testKwDefaults(self):
        if False:
            return 10

        def foo(p1, p2=0, *, k1, k2=0):
            if False:
                print('Hello World!')
            return p1 + p2 + k1 + k2
        self.assertEqual(2, foo.__code__.co_kwonlyargcount)
        self.assertEqual({'k2': 0}, foo.__kwdefaults__)
        foo.__kwdefaults__ = {'k1': 0}
        try:
            foo(1, k1=10)
            self.fail('__kwdefaults__ is not properly changed')
        except TypeError:
            pass

    def test_kwonly_methods(self):
        if False:
            while True:
                i = 10

        class Example:

            def f(self, *, k1=1, k2=2):
                if False:
                    for i in range(10):
                        print('nop')
                return (k1, k2)
        self.assertEqual(Example().f(k1=1, k2=2), (1, 2))
        self.assertEqual(Example.f(Example(), k1=1, k2=2), (1, 2))
        self.assertRaises(TypeError, Example.f, k1=1, k2=2)

    def test_issue13343(self):
        if False:
            print('Hello World!')
        lambda *, k1=unittest: None

    def test_mangling(self):
        if False:
            i = 10
            return i + 15

        class X:

            def f(self, *, __a=42):
                if False:
                    while True:
                        i = 10
                return __a
        self.assertEqual(X().f(), 42)

    def test_default_evaluation_order(self):
        if False:
            for i in range(10):
                print('nop')
        a = 42
        with self.assertRaises(NameError) as err:

            def f(v=a, x=b, *, y=c, z=d):
                if False:
                    for i in range(10):
                        print('nop')
                pass
        self.assertEqual(str(err.exception), "name 'b' is not defined")
        with self.assertRaises(NameError) as err:
            f = lambda v=a, x=b, *, y=c, z=d: None
        self.assertEqual(str(err.exception), "name 'b' is not defined")
if __name__ == '__main__':
    unittest.main()