from absl.testing import parameterized
from tensorflow.python.eager import def_function
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.platform import test

class ContextCrossPlatformTest(test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters([(f'_{stage}', stage) for stage in ['hlo', 'hlo_serialized']])
    def testGetCompilerIrOnTpuPlatform(self, stage):
        if False:
            return 10

        @def_function.function(jit_compile=True)
        def test_func(x):
            if False:
                i = 10
                return i + 15
            return 2 * x
        a = array_ops.ones((1000, 1000))
        result = test_func.experimental_get_compiler_ir(a)(stage=stage, platform_name='TPU')
        self.assertNotEmpty(result)
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()