import numpy as np
import time
import unittest
import onnx
import onnx.defs
from onnx.backend.base import namedtupledict
from onnx.helper import make_node, make_graph, make_tensor_value_info, make_model
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace
from caffe2.python.models.download import ModelDownloader
from caffe2.python.onnx.onnxifi import onnxifi_caffe2_net
from caffe2.python.onnx.tests.test_utils import TestCase
ONNXIFI_DATATYPE_FLOAT32 = 1

def _print_net(net):
    if False:
        return 10
    for i in net.external_input:
        print('Input: {}'.format(i))
    for i in net.external_output:
        print('Output: {}'.format(i))
    for op in net.op:
        print('Op {}'.format(op.type))
        for x in op.input:
            print('  input: {}'.format(x))
        for y in op.output:
            print('  output: {}'.format(y))

class OnnxifiTest(TestCase):

    @unittest.skip('Need ONNXIFI backend support')
    def test_relu_graph(self):
        if False:
            i = 10
            return i + 15
        batch_size = 1
        X = np.random.randn(batch_size, 1, 3, 2).astype(np.float32)
        graph_def = make_graph([make_node('Relu', ['X'], ['Y'])], name='test', inputs=[make_tensor_value_info('X', onnx.TensorProto.FLOAT, [batch_size, 1, 3, 2])], outputs=[make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [batch_size, 1, 3, 2])])
        model_def = make_model(graph_def, producer_name='relu-test')
        op = core.CreateOperator('Onnxifi', ['X'], ['Y'], onnx_model=model_def.SerializeToString(), input_names=['X'], output_names=['Y'], output_shape_hint_0=[ONNXIFI_DATATYPE_FLOAT32, batch_size, 1, 3, 2])
        workspace.FeedBlob('X', X)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y')
        np.testing.assert_almost_equal(Y, np.maximum(X, 0))

    @unittest.skip('Need ONNXIFI backend support')
    def test_conv_graph(self):
        if False:
            return 10
        X = np.array([[[[0.0, 1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0, 9.0], [10.0, 11.0, 12.0, 13.0, 14.0], [15.0, 16.0, 17.0, 18.0, 19.0], [20.0, 21.0, 22.0, 23.0, 24.0]]]]).astype(np.float32)
        W = np.array([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]]).astype(np.float32)
        Y_without_padding = np.array([[[[54.0, 63.0, 72.0], [99.0, 108.0, 117.0], [144.0, 153.0, 162.0]]]]).astype(np.float32)
        graph_def = make_graph([make_node('Conv', inputs=['X', 'W'], outputs=['Y'], kernel_shape=[3, 3], pads=[0, 0, 0, 0])], name='test', inputs=[make_tensor_value_info('X', onnx.TensorProto.FLOAT, [1, 1, 5, 5]), make_tensor_value_info('W', onnx.TensorProto.FLOAT, [1, 1, 3, 3])], outputs=[make_tensor_value_info('Y', onnx.TensorProto.FLOAT, [1, 1, 3, 3])])
        model_def = make_model(graph_def, producer_name='conv-test')
        op = core.CreateOperator('Onnxifi', ['X0'], ['Y0'], onnx_model=model_def.SerializeToString(), initializers=['W', 'W0'], input_names=['X'], output_names=['Y'], output_shape_hint_0=[ONNXIFI_DATATYPE_FLOAT32, 1, 1, 3, 3])
        workspace.FeedBlob('X0', X)
        workspace.FeedBlob('W0', W)
        workspace.RunOperatorOnce(op)
        Y = workspace.FetchBlob('Y0')
        np.testing.assert_almost_equal(Y, Y_without_padding)

class OnnxifiTransformTest(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.model_downloader = ModelDownloader()

    def _add_head_tail(self, pred_net, new_head, new_tail):
        if False:
            print('Hello World!')
        orig_head = pred_net.external_input[0]
        orig_tail = pred_net.external_output[0]
        head = caffe2_pb2.OperatorDef()
        head.type = 'Copy'
        head.input.append(new_head)
        head.output.append(orig_head)
        dummy = caffe2_pb2.NetDef()
        dummy.op.extend(pred_net.op)
        del pred_net.op[:]
        pred_net.op.extend([head])
        pred_net.op.extend(dummy.op)
        pred_net.external_input[0] = new_head
        tail = caffe2_pb2.OperatorDef()
        tail.type = 'Copy'
        tail.input.append(orig_tail)
        tail.output.append(new_tail)
        pred_net.op.extend([tail])
        pred_net.external_output[0] = new_tail

    @unittest.skip('Need ONNXIFI backend support')
    def test_resnet50_core(self):
        if False:
            i = 10
            return i + 15
        N = 1
        repeat = 1
        print('Batch size: {}, repeat inference {} times'.format(N, repeat))
        (init_net, pred_net, _) = self.model_downloader.get_c2_model('resnet50')
        self._add_head_tail(pred_net, 'real_data', 'real_softmax')
        input_blob_dims = (N, 3, 224, 224)
        input_name = 'real_data'
        device_option = core.DeviceOption(caffe2_pb2.CPU, 0)
        init_net.device_option.CopyFrom(device_option)
        pred_net.device_option.CopyFrom(device_option)
        for op in pred_net.op:
            op.device_option.CopyFrom(device_option)
        net_outputs = pred_net.external_output
        Y_c2 = None
        data = np.random.randn(*input_blob_dims).astype(np.float32)
        c2_time = 1
        workspace.SwitchWorkspace('onnxifi_test', True)
        with core.DeviceScope(device_option):
            workspace.FeedBlob(input_name, data)
            workspace.RunNetOnce(init_net)
            workspace.CreateNet(pred_net)
            start = time.time()
            for _ in range(repeat):
                workspace.RunNet(pred_net.name)
            end = time.time()
            c2_time = end - start
            output_values = [workspace.FetchBlob(name) for name in net_outputs]
            Y_c2 = namedtupledict('Outputs', net_outputs)(*output_values)
        workspace.ResetWorkspace()
        with core.DeviceScope(device_option):
            workspace.RunNetOnce(init_net)
        start = time.time()
        pred_net_cut = onnxifi_caffe2_net(pred_net, {input_name: input_blob_dims}, infer_shapes=True)
        del init_net, pred_net
        Y_trt = None
        input_name = pred_net_cut.external_input[0]
        print('C2 runtime: {}s'.format(c2_time))
        with core.DeviceScope(device_option):
            workspace.FeedBlob(input_name, data)
            workspace.CreateNet(pred_net_cut)
            end = time.time()
            print('Conversion time: {:.2f}s'.format(end - start))
            start = time.time()
            for _ in range(repeat):
                workspace.RunNet(pred_net_cut.name)
            end = time.time()
            trt_time = end - start
            print('Onnxifi runtime: {}s, improvement: {}%'.format(trt_time, (c2_time - trt_time) / c2_time * 100))
            output_values = [workspace.FetchBlob(name) for name in net_outputs]
            Y_trt = namedtupledict('Outputs', net_outputs)(*output_values)
        np.testing.assert_allclose(Y_c2, Y_trt, rtol=0.001)