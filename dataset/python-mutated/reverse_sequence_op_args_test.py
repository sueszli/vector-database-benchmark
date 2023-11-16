"""Tests for tensorflow.ops.reverse_sequence_op."""
from tensorflow.compiler.tests import xla_test
from tensorflow.python.compat import v2_compat
from tensorflow.python.eager import def_function
from tensorflow.python.framework import errors
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ReverseSequenceArgsTest(xla_test.XLATestCase):
    """Tests argument verification of array_ops.reverse_sequence."""

    def testInvalidArguments(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex((errors.InvalidArgumentError, ValueError), 'seq_dim must be >=0'):

            @def_function.function(jit_compile=True)
            def f(x):
                if False:
                    while True:
                        i = 10
                return array_ops.reverse_sequence(x, [2, 2], seq_axis=-1)
            f([[1, 2], [3, 4]])
        with self.assertRaisesRegex(ValueError, 'batch_dim must be >=0'):

            @def_function.function(jit_compile=True)
            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.reverse_sequence(x, [2, 2], seq_axis=1, batch_axis=-1)
            g([[1, 2], [3, 4]])
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()