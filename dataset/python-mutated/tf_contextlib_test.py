"""Unit tests for tf_contextlib."""
from tensorflow.python.platform import test
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect

@tf_contextlib.contextmanager
def test_yield_append_before_and_after_yield(x, before, after):
    if False:
        return 10
    x.append(before)
    yield
    x.append(after)

@tf_contextlib.contextmanager
def test_yield_return_x_plus_1(x):
    if False:
        while True:
            i = 10
    yield (x + 1)

@tf_contextlib.contextmanager
def test_params_and_defaults(a, b=2, c=True, d='hello'):
    if False:
        return 10
    return [a, b, c, d]

class TfContextlibTest(test.TestCase):

    def testRunsCodeBeforeYield(self):
        if False:
            return 10
        x = []
        with test_yield_append_before_and_after_yield(x, 'before', ''):
            self.assertEqual('before', x[-1])

    def testRunsCodeAfterYield(self):
        if False:
            i = 10
            return i + 15
        x = []
        with test_yield_append_before_and_after_yield(x, '', 'after'):
            pass
        self.assertEqual('after', x[-1])

    def testNestedWith(self):
        if False:
            i = 10
            return i + 15
        x = []
        with test_yield_append_before_and_after_yield(x, 'before', 'after'):
            with test_yield_append_before_and_after_yield(x, 'inner', 'outer'):
                with test_yield_return_x_plus_1(1) as var:
                    x.append(var)
        self.assertEqual(['before', 'inner', 2, 'outer', 'after'], x)

    def testMultipleCallsOfSeparateInstances(self):
        if False:
            i = 10
            return i + 15
        x = []
        with test_yield_append_before_and_after_yield(x, 1, 2):
            pass
        with test_yield_append_before_and_after_yield(x, 3, 4):
            pass
        self.assertEqual([1, 2, 3, 4], x)

    def testReturnsResultFromYield(self):
        if False:
            i = 10
            return i + 15
        with test_yield_return_x_plus_1(3) as result:
            self.assertEqual(4, result)

    def testUnwrapContextManager(self):
        if False:
            return 10
        (decorators, target) = tf_decorator.unwrap(test_params_and_defaults)
        self.assertEqual(1, len(decorators))
        self.assertTrue(isinstance(decorators[0], tf_decorator.TFDecorator))
        self.assertEqual('contextmanager', decorators[0].decorator_name)
        self.assertFalse(isinstance(target, tf_decorator.TFDecorator))

    def testGetArgSpecReturnsWrappedArgSpec(self):
        if False:
            return 10
        argspec = tf_inspect.getargspec(test_params_and_defaults)
        self.assertEqual(['a', 'b', 'c', 'd'], argspec.args)
        self.assertEqual((2, True, 'hello'), argspec.defaults)
if __name__ == '__main__':
    test.main()