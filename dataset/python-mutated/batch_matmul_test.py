"""Model script to test TF-TensorRT integration."""
import unittest
import numpy as np
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test

class BatchMatMultTestBase(trt_test.TfTrtIntegrationTestBase):
    """Base class for BatchMatMult tests."""

    def BuildParams(self, graph_fn, dtype, input_shapes, output_shapes):
        if False:
            while True:
                i = 10
        return self.BuildParamsWithMask(graph_fn=graph_fn, dtype=dtype, input_shapes=input_shapes, output_shapes=output_shapes, input_mask=[[True] * len(s) for s in input_shapes], output_mask=[[True] * len(s) for s in output_shapes], extra_inputs=[], extra_outputs=[])

    @classmethod
    def setUpClass(cls):
        if False:
            for i in range(10):
                print('nop')
        if cls is BatchMatMultTestBase:
            raise unittest.SkipTest('BatchMatMultTestBase defines base class for other test.')
        super(BatchMatMultTestBase, cls).setUpClass()

class BatchMatMulTwoTensorTest(BatchMatMultTestBase):
    """Testing conversion of BatchMatMul where both inputs are tensors."""

    def GraphFn(self, inp, inp1):
        if False:
            i = 10
            return i + 15
        x1 = math_ops.matmul(inp, inp1, name='matmul')
        x1 = nn.relu(x1, name='relu')
        return array_ops.identity(x1, name='output_0')

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 5, 8, 12], [12, 5, 12, 7]], [[12, 5, 8, 7]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            i = 10
            return i + 15
        'Return the expected engines to build.'
        return {'TRTEngineOp_000': ['matmul', 'relu']}

class BatchMatMulWeightBroadcastTest(BatchMatMultTestBase):
    """Testing BatchMatMulV2: one operand is weight and both have same rank."""

    def ShouldAllowTF32Computation(self):
        if False:
            return 10
        return False

    def GraphFn(self, inp):
        if False:
            print('Hello World!')
        dtype = inp.dtype
        b = constant_op.constant(np.random.randn(1, 5, 7), dtype=dtype, name='kernel')
        x1 = math_ops.matmul(inp, b, name='matmul')
        return array_ops.identity(x1, name='output_0')

    def GetParams(self):
        if False:
            i = 10
            return i + 15
        return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 9, 5]], [[12, 9, 7]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            while True:
                i = 10
        'Return the expected engines to build.'
        return {'TRTEngineOp_000': ['matmul', 'kernel']}

class BatchMatMulWeightBroadcastDims2Test(BatchMatMultTestBase):
    """Testing BatchMatMulV2: weight operand must be broadcasted."""

    def ShouldAllowTF32Computation(self):
        if False:
            while True:
                i = 10
        return False

    def GraphFn(self, inp):
        if False:
            i = 10
            return i + 15
        dtype = inp.dtype
        b = constant_op.constant(np.random.randn(5, 7), dtype=dtype, name='kernel')
        x1 = math_ops.matmul(inp, b, name='matmul')
        return array_ops.identity(x1, name='output_0')

    def GetParams(self):
        if False:
            print('Hello World!')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[12, 9, 5]], [[12, 9, 7]])

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            for i in range(10):
                print('nop')
        'Return the expected engines to build.'
        return {'TRTEngineOp_000': ['matmul', 'kernel']}
if __name__ == '__main__':
    test.main()