"""Tests for Convolution node name match via the XLA JIT.

The canned results in these tests are created by running each test using the
Tensorflow CPU device and saving the output.
"""
import numpy as np
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import ops
from tensorflow.python.layers import layers
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import googletest

class ConvolutionNodeNameTest(xla_test.XLATestCase):
    """Verify convolution node name match.

  Verify convolution node names on TPU and CPU match with dilation > 1.
  """

    def _verifyNodeNameMatch(self, layer, input_sizes, filter_sizes, strides, dilations):
        if False:
            print('Hello World!')

        def _GetNodeNames(use_xla):
            if False:
                i = 10
                return i + 15
            with self.session():
                input_tensor = array_ops.placeholder(np.float32, shape=input_sizes)
                if use_xla:
                    with self.test_scope():
                        graph = ops.get_default_graph()
                        graph._set_control_flow_context(control_flow_ops.XLAControlFlowContext())
                        conv2d_op = layer(filters=64, kernel_size=filter_sizes, dilation_rate=dilations, padding='same')
                        _ = conv2d_op(input_tensor)
                        return [n.name for n in ops.get_default_graph().as_graph_def().node]
                else:
                    with ops.device('CPU'):
                        conv2d_op = layer(filters=64, kernel_size=filter_sizes, dilation_rate=dilations, padding='same')
                        _ = conv2d_op(input_tensor)
                        names = [n.name for n in ops.get_default_graph().as_graph_def().node]
                        return [name for name in names if 'space' not in name and 'Space' not in name]
        xla_names = _GetNodeNames(use_xla=True)
        no_xla_names = _GetNodeNames(use_xla=False)
        filtered_no_xla_names = []
        for name in no_xla_names:
            if 'dilation_rate' in name or 'filter_shape' in name or 'stack' in name:
                continue
            else:
                filtered_no_xla_names.append(name)
        self.assertListEqual(xla_names, filtered_no_xla_names)

    def testConv1DNodeNameMatch(self):
        if False:
            i = 10
            return i + 15
        input_sizes = [8, 16, 3]
        filter_sizes = [7]
        strides = 1
        dilations = [2]
        layer = layers.Conv1D
        self._verifyNodeNameMatch(layer, input_sizes, filter_sizes, strides, dilations)

    def testConv2DNodeNameMatch(self):
        if False:
            while True:
                i = 10
        input_sizes = [8, 16, 16, 3]
        filter_sizes = [7, 7]
        strides = 1
        dilations = [2, 2]
        layer = layers.Conv2D
        self._verifyNodeNameMatch(layer, input_sizes, filter_sizes, strides, dilations)

    def testConv3DNodeNameMatch(self):
        if False:
            print('Hello World!')
        input_sizes = [8, 16, 16, 16, 3]
        filter_sizes = [7, 7, 7]
        strides = 1
        dilations = [2, 2, 2]
        layer = layers.Conv3D
        self._verifyNodeNameMatch(layer, input_sizes, filter_sizes, strides, dilations)
if __name__ == '__main__':
    googletest.main()