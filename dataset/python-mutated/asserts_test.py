"""Tests for asserts module."""
from tensorflow.python.autograph.converters import asserts
from tensorflow.python.autograph.converters import functions
from tensorflow.python.autograph.converters import return_statements
from tensorflow.python.autograph.core import converter_testing
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors_impl
from tensorflow.python.platform import test

class AssertsTest(converter_testing.TestCase):

    def test_basic(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                print('Hello World!')
            assert a, 'testmsg'
            return a
        tr = self.transform(f, (functions, asserts, return_statements))
        op = tr(constant_op.constant(False))
        with self.assertRaisesRegex(errors_impl.InvalidArgumentError, 'testmsg'):
            self.evaluate(op)
if __name__ == '__main__':
    test.main()