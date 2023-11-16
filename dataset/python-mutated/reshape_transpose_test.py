"""Basic tests for TF-TensorRT integration."""
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ReshapeTest(trt_test.TfTrtIntegrationTestBase):

    def GraphFn(self, inp):
        if False:
            return 10
        outputs = []
        orig_shape = constant_op.constant([-1, 24, 24, 2], name='original_shape')
        for shape in [[2, 50, 24, 24, 2], [-1, 50, 24, 24, 2], [2, 50, -1, 24, 2]]:
            incompatible_reshape = array_ops.reshape(inp, shape)
            reshape_back = array_ops.reshape(incompatible_reshape, orig_shape)
            outputs.append(self.trt_incompatible_op(reshape_back))
        compatible_reshape = array_ops.reshape(inp, [-1, 24 * 24, 2], name='reshape-0')
        compatible_reshape = array_ops.reshape(compatible_reshape, [100, 24, -1], name='reshape-1')
        compatible_reshape = array_ops.reshape(compatible_reshape, [100, 24 * 2, 24], name='reshape-2')
        compatible_reshape = array_ops.reshape(compatible_reshape, [-1, 24, 24 * 2], name='reshape-3')
        compatible_reshape = array_ops.reshape(compatible_reshape, [-1, 6, 4, 24, 2], name='reshape-4')
        compatible_reshape = array_ops.reshape(compatible_reshape, [-1, 6, 4, 6, 4, 2, 1], name='reshape-5')
        compatible_reshape = array_ops.reshape(compatible_reshape, [-1, 24, 24, 2], name='reshape-6')
        outputs.append(self.trt_incompatible_op(compatible_reshape))
        return math_ops.add_n(outputs, name='output_0')

    def GetParams(self):
        if False:
            print('Hello World!')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]], [[100, 24, 24, 2]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            print('Hello World!')
        'Return the expected engines to build.'
        return {'TRTEngineOp_000': ['reshape-%d' % i for i in range(7)] + ['reshape-%d/shape' % i for i in range(7)]}

    def ShouldRunTest(self, run_params):
        if False:
            i = 10
            return i + 15
        'Whether to run the test.'
        return (not trt_test.IsQuantizationMode(run_params.precision_mode) and (not run_params.dynamic_engine), 'test static engine and non-INT8')

class TransposeTest(trt_test.TfTrtIntegrationTestBase):

    def GraphFn(self, inp):
        if False:
            i = 10
            return i + 15
        compatible_transpose = array_ops.transpose(inp, [0, 3, 1, 2], name='transpose-1')
        compatible_transpose = array_ops.transpose(compatible_transpose, [0, 2, 3, 1], name='transposeback')
        return array_ops.identity(compatible_transpose, name='output_0')

    def GetParams(self):
        if False:
            for i in range(10):
                print('nop')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]], [[100, 24, 24, 2]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        'Return the expected engines to build.'
        return {'TRTEngineOp_000': ['transpose-1', 'transpose-1/perm', 'transposeback', 'transposeback/perm']}

    def ShouldRunTest(self, run_params):
        if False:
            while True:
                i = 10
        'Whether to run the test.'
        return (not trt_test.IsQuantizationMode(run_params.precision_mode) and (not run_params.dynamic_engine), 'test static engine and non-INT8')

class IncompatibleTransposeTest(TransposeTest):

    def GraphFn(self, inp):
        if False:
            return 10
        incompatible_transpose = array_ops.transpose(inp, [2, 1, 0, 3], name='transpose-2')
        excluded_transpose = array_ops.transpose(incompatible_transpose, [0, 2, 3, 1], name='transpose-3')
        return array_ops.identity(excluded_transpose, name='output_0')

    def GetParams(self):
        if False:
            return 10
        return self.BuildParams(self.GraphFn, dtypes.float32, [[100, 24, 24, 2]], [[24, 100, 2, 24]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            print('Hello World!')
        'Return the expected engines to build.'
        return []
if __name__ == '__main__':
    test.main()