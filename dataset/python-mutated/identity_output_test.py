"""This test checks a situation where the same tensor is considered as an output

multiple times because it has been duplicated by 2+ identity ops. Previously,
the tensor would be renamed multiple times, overwriting the output binding name
which resulted in a runtime error when the binding would not be found.
"""
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class IdentityTest(trt_test.TfTrtIntegrationTestBase):
    """Testing engine with the same tensor repeated as output via identity."""

    def GraphFn(self, x):
        if False:
            print('Hello World!')
        x1 = math_ops.exp(x)
        x1 = x1 + x
        out1 = array_ops.identity(x1, name='output_0')
        out2 = array_ops.identity(x1, name='output_1')
        iden1 = array_ops.identity(x1)
        out3 = array_ops.identity(iden1, name='output_2')
        return [out1, out2, out3]

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 32]], [[100, 32]] * 3)

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            i = 10
            return i + 15
        'Return the expected engines to build.'
        return ['TRTEngineOp_000']
if __name__ == '__main__':
    test.main()