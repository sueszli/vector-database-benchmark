"""This script test input and output shapes and dtype of the TRTEngineOp."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from tensorflow.python.compiler.tensorrt.test import tf_trt_integration_test_base as trt_test
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.platform import test
from tensorflow.python.saved_model import load
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import tag_constants

class TRTEngineOpInputOutputShapeTest(trt_test.TfTrtIntegrationTestBase):
    """Testing the output shape of a TRTEngine."""

    def GraphFn(self, inp):
        if False:
            return 10
        b = array_ops.squeeze(inp, axis=[2])
        c = nn.relu(b)
        d1 = c + c
        d2 = math_ops.reduce_sum(d1)
        d1 = array_ops.identity(d1, name='output_0')
        d2 = array_ops.identity(d2, name='output_1')
        return (d1, d2)

    def GetParams(self):
        if False:
            print('Hello World!')
        return self.BuildParams(self.GraphFn, dtypes.float32, [[1, 2, 1, 4]], [[1, 2, 4], []])

    def _GetInferGraph(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        trt_saved_model_dir = super(TRTEngineOpInputOutputShapeTest, self)._GetInferGraph(*args, **kwargs)

        def get_func_from_saved_model(saved_model_dir):
            if False:
                print('Hello World!')
            try:
                saved_model_load_fn = load.load
            except AttributeError:
                saved_model_load_fn = load
            saved_model_loaded = saved_model_load_fn(saved_model_dir, tags=[tag_constants.SERVING])
            graph_func = saved_model_loaded.signatures[signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
            return (graph_func, saved_model_loaded)
        (func, _) = get_func_from_saved_model(trt_saved_model_dir)
        input_shape = func.inputs[0].shape
        if isinstance(input_shape, tensor_shape.TensorShape):
            input_shape = input_shape.as_list()
        output_shapes = [out_shape.shape.as_list() if isinstance(out_shape.shape, tensor_shape.TensorShape) else out_shape.shape for out_shape in func.outputs]
        self.assertEqual(func.inputs[0].dtype, dtypes.float32)
        self.assertEqual(func.outputs[0].dtype, dtypes.float32)
        self.assertEqual(func.outputs[1].dtype, dtypes.float32)
        self.assertEqual(input_shape, [None, 2, 1, 4])
        self.assertEqual(output_shapes[0], [None, 2, 4])
        self.assertEqual(output_shapes[1], [])
        return trt_saved_model_dir

    def ExpectedEnginesToBuild(self, run_params):
        if False:
            while True:
                i = 10
        'Return the expected engines to build.'
        return ['TRTEngineOp_000']
if __name__ == '__main__':
    test.main()