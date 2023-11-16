"""Model script to test TF-TensorRT integration."""
from unittest import SkipTest
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class TrtModeTestBase(trt_test.TfTrtIntegrationTestBase):
    """Test squeeze on batch dim and some unary operations in TF-TRT."""

    def GraphFn(self, x1):
        if False:
            for i in range(10):
                print('nop')
        q = math_ops.abs(x1)
        q = q + 1.0
        q = q * 3.0
        q = array_ops.squeeze(q, 0)
        q = math_ops.abs(q)
        q = q + 5.0
        return array_ops.identity(q, name='output_0')

    def ShouldRunTest(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        return (run_params.dynamic_engine and run_params.is_v2 and (not run_params.use_calibration), 'test v2 dynamic engine and non-calibration')

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        'The input has 1 as a first dimension, which is removed by the squeeze.\n\n    op in the graph.\n\n    In explicit batch mode, TensorRT can convert the whole graph. In this mode\n    it is possible to manipulate the batch dimension using the squeeze op.\n\n    In implicit batch mode TensorRT cannot convert the whole graph. We are not\n    allowed to manipulate (squeeze) the first dimension in implicit batch mode.\n    Therefore the graph will be converted using multiple segments.\n    '
        return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 12, 5]], [[12, 5]])

    def GetMaxBatchSize(self, run_params):
        if False:
            print('Hello World!')
        if run_params.dynamic_engine:
            return None
        return 12

    @classmethod
    def setUpClass(cls):
        if False:
            i = 10
            return i + 15
        if cls is TrtModeTestBase:
            raise SkipTest('TrtModeTestBase defines base class for other test.')
        super(TrtModeTestBase, cls).setUpClass()

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            print('Hello World!')
        "Check that the expected engine is built.\n\n    Args:\n      run_params: the run parameters.\n\n    Returns:\n      the expected engines to build.\n\n    The squeeze op is not converted by TensorRT in implicit batch mode.\n    Because of this we have two TRTEngineOp in the graphs: one for the\n    subgraph before 'squeeze(q,0)', and another one for the rest of the ops\n    after the 'squeeze(q,0)'.\n\n    In explicit batch mode the whole graph is converted using a single engine.\n    "
        if run_params.dynamic_shape:
            return ['TRTEngineOp_000']
        else:
            return ['TRTEngineOp_000', 'TRTEngineOp_001']

class StaticInputTest(TrtModeTestBase):

    def GetParams(self):
        if False:
            for i in range(10):
                print('nop')
        'We specify input/output masks with static (known) shapes.'
        return self.BuildParamsWithMask(self.GraphFn, dtypes.float32, [[1, 12, 5]], [[12, 5]], input_mask=[[True, True, True]], output_mask=[[True, True]], extra_inputs=[], extra_outputs=[])

class DynamicInputTest(TrtModeTestBase):
    """Test with dynamic input shapes.

  The difference to the previous test is that we use input and output masks to
  change the input and output shapes to unknown shapes.
  """

    def GetParams(self):
        if False:
            return 10
        'We specify input/output mask with dynamic (unknown) shapes.\n\n    In dynamic shape mode, single engine with three optimization profiles can\n    handle the three different input shapes.\n    '
        return self.BuildParamsWithMask(self.GraphFn, dtypes.float32, [[1, 12, 5]], [[12, 5]], extra_inputs=[[[1, 2, 3]], [[1, 4, 6]]], extra_outputs=[[[2, 3]], [[4, 6]]], input_mask=[[False, False, False]], output_mask=[[False, False]])
if __name__ == '__main__':
    test.main()