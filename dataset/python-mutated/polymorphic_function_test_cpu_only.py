from absl.testing import parameterized
from tensorflow.python.eager.polymorphic_function import polymorphic_function
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test

class FunctionCpuOnlyTest(test.TestCase, parameterized.TestCase):
    """Test that jit_compile=True correctly throws an exception if XLA is not available.

  This test should only be run without `--config=cuda`, as that implicitly links
  in XLA JIT.
  """

    def testJitCompileRaisesExceptionWhenXlaIsUnsupported(self):
        if False:
            for i in range(10):
                print('nop')
        if test.is_built_with_rocm() or test_util.is_xla_enabled():
            return
        with self.assertRaisesRegex(errors.UnimplementedError, 'support for that platform linked in'):

            @polymorphic_function.function(jit_compile=True)
            def fn(x):
                if False:
                    while True:
                        i = 10
                return x + x
            fn([1, 1, 2, 3])
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()