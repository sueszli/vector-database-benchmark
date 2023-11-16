import unittest
from caffe2.proto import caffe2_pb2
import caffe2.python.cnn as cnn
import caffe2.python.core as core
import caffe2.contrib.tensorboard.tensorboard_exporter as tb
EXPECTED = '\nnode {\n  name: "conv1/XavierFill"\n  op: "XavierFill"\n  device: "/gpu:0"\n  attr {\n    key: "_output_shapes"\n    value {\n      list {\n        shape {\n          dim {\n            size: 96\n          }\n          dim {\n            size: 3\n          }\n          dim {\n            size: 11\n          }\n          dim {\n            size: 11\n          }\n        }\n      }\n    }\n  }\n}\nnode {\n  name: "conv1/ConstantFill"\n  op: "ConstantFill"\n  device: "/gpu:0"\n  attr {\n    key: "_output_shapes"\n    value {\n      list {\n        shape {\n          dim {\n            size: 96\n          }\n        }\n      }\n    }\n  }\n}\nnode {\n  name: "classifier/XavierFill"\n  op: "XavierFill"\n  device: "/gpu:0"\n  attr {\n    key: "_output_shapes"\n    value {\n      list {\n        shape {\n          dim {\n            size: 1000\n          }\n          dim {\n            size: 4096\n          }\n        }\n      }\n    }\n  }\n}\nnode {\n  name: "classifier/ConstantFill"\n  op: "ConstantFill"\n  device: "/gpu:0"\n  attr {\n    key: "_output_shapes"\n    value {\n      list {\n        shape {\n          dim {\n            size: 1000\n          }\n        }\n      }\n    }\n  }\n}\nnode {\n  name: "ImageInput"\n  op: "ImageInput"\n  input: "db"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "is_test"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "use_cudnn"\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: "NHWC2NCHW"\n  op: "NHWC2NCHW"\n  input: "data_nhwc"\n  device: "/gpu:0"\n}\nnode {\n  name: "conv1/Conv"\n  op: "Conv"\n  input: "data"\n  input: "conv1/conv1_w"\n  input: "conv1/conv1_b"\n  device: "/gpu:0"\n  attr {\n    key: "exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "kernel"\n    value {\n      i: 11\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n  attr {\n    key: "stride"\n    value {\n      i: 4\n    }\n  }\n}\nnode {\n  name: "conv1/Relu"\n  op: "Relu"\n  input: "conv1/conv1"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n}\nnode {\n  name: "conv1/MaxPool"\n  op: "MaxPool"\n  input: "conv1/conv1_1"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "kernel"\n    value {\n      i: 2\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n  attr {\n    key: "stride"\n    value {\n      i: 2\n    }\n  }\n}\nnode {\n  name: "classifier/FC"\n  op: "FC"\n  input: "conv1/pool1"\n  input: "classifier/fc_w"\n  input: "classifier/fc_b"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n  attr {\n    key: "use_cudnn"\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: "classifier/Softmax"\n  op: "Softmax"\n  input: "classifier/fc"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n}\nnode {\n  name: "classifier/LabelCrossEntropy"\n  op: "LabelCrossEntropy"\n  input: "classifier/pred"\n  input: "label"\n  device: "/gpu:0"\n}\nnode {\n  name: "classifier/AveragedLoss"\n  op: "AveragedLoss"\n  input: "classifier/xent"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/ConstantFill"\n  op: "ConstantFill"\n  input: "classifier/loss"\n  device: "/gpu:0"\n  attr {\n    key: "value"\n    value {\n      f: 1.0\n    }\n  }\n}\nnode {\n  name: "GRADIENTS/classifier/AveragedLossGradient"\n  op: "AveragedLossGradient"\n  input: "classifier/xent"\n  input: "GRADIENTS/classifier/loss_autogen_grad"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/LabelCrossEntropyGradient"\n  op: "LabelCrossEntropyGradient"\n  input: "classifier/pred"\n  input: "label"\n  input: "GRADIENTS/classifier/xent_grad"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/SoftmaxGradient"\n  op: "SoftmaxGradient"\n  input: "classifier/pred"\n  input: "GRADIENTS/classifier/pred_grad"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n}\nnode {\n  name: "GRADIENTS/c/FCGradient"\n  op: "FCGradient"\n  input: "conv1/pool1"\n  input: "classifier/fc_w"\n  input: "GRADIENTS/classifier/fc_grad"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n  attr {\n    key: "use_cudnn"\n    value {\n      i: 1\n    }\n  }\n}\nnode {\n  name: "GRADIENTS/conv1/MaxPoolGradient"\n  op: "MaxPoolGradient"\n  input: "conv1/conv1_1"\n  input: "conv1/pool1"\n  input: "GRADIENTS/conv1/pool1_grad"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "kernel"\n    value {\n      i: 2\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n  attr {\n    key: "stride"\n    value {\n      i: 2\n    }\n  }\n}\nnode {\n  name: "GRADIENTS/conv1/ReluGradient"\n  op: "ReluGradient"\n  input: "conv1/conv1_1"\n  input: "GRADIENTS/conv1/conv1_grad"\n  device: "/gpu:0"\n  attr {\n    key: "cudnn_exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n}\nnode {\n  name: "GRADIENTS/ConvGradient"\n  op: "ConvGradient"\n  input: "data"\n  input: "conv1/conv1_w"\n  input: "GRADIENTS/conv1/conv1_grad_1"\n  device: "/gpu:0"\n  attr {\n    key: "exhaustive_search"\n    value {\n      i: 0\n    }\n  }\n  attr {\n    key: "kernel"\n    value {\n      i: 11\n    }\n  }\n  attr {\n    key: "order"\n    value {\n      s: "NCHW"\n    }\n  }\n  attr {\n    key: "stride"\n    value {\n      i: 4\n    }\n  }\n}\nnode {\n  name: "GRADIENTS/NCHW2NHWC"\n  op: "NCHW2NHWC"\n  input: "GRADIENTS/data_grad"\n  device: "/gpu:0"\n}\nnode {\n  name: "conv1/conv1_w"\n  op: "Blob"\n  input: "conv1/XavierFill:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "classifier/fc"\n  op: "Blob"\n  input: "classifier/FC:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "data_nhwc"\n  op: "Blob"\n  input: "ImageInput:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/conv1/conv1_b_grad"\n  op: "Blob"\n  input: "GRADIENTS/ConvGradient:1"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/pred_grad"\n  op: "Blob"\n  input: "GRADIENTS/classifier/LabelCrossEntropyGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/fc_grad"\n  op: "Blob"\n  input: "GRADIENTS/classifier/SoftmaxGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "conv1/conv1_b"\n  op: "Blob"\n  input: "conv1/ConstantFill:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/fc_b_grad"\n  op: "Blob"\n  input: "GRADIENTS/c/FCGradient:1"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/fc_w_grad"\n  op: "Blob"\n  input: "GRADIENTS/c/FCGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "label"\n  op: "Blob"\n  input: "ImageInput:1"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/data_grad"\n  op: "Blob"\n  input: "GRADIENTS/ConvGradient:2"\n  device: "/gpu:0"\n}\nnode {\n  name: "classifier/loss"\n  op: "Blob"\n  input: "classifier/AveragedLoss:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "conv1/conv1"\n  op: "Blob"\n  input: "conv1/Conv:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/conv1/conv1_grad"\n  op: "Blob"\n  input: "GRADIENTS/conv1/MaxPoolGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "classifier/xent"\n  op: "Blob"\n  input: "classifier/LabelCrossEntropy:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/loss_autogen_grad"\n  op: "Blob"\n  input: "GRADIENTS/classifier/ConstantFill:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "classifier/fc_w"\n  op: "Blob"\n  input: "classifier/XavierFill:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "conv1/conv1_1"\n  op: "Blob"\n  input: "conv1/Relu:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "db"\n  op: "Placeholder"\n}\nnode {\n  name: "classifier/pred"\n  op: "Blob"\n  input: "classifier/Softmax:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "classifier/fc_b"\n  op: "Blob"\n  input: "classifier/ConstantFill:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/classifier/xent_grad"\n  op: "Blob"\n  input: "GRADIENTS/classifier/AveragedLossGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "data"\n  op: "Blob"\n  input: "NHWC2NCHW:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/conv1/conv1_w_grad"\n  op: "Blob"\n  input: "GRADIENTS/ConvGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/conv1/conv1_grad_1"\n  op: "Blob"\n  input: "GRADIENTS/conv1/ReluGradient:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/data_nhwc_grad"\n  op: "Blob"\n  input: "GRADIENTS/NCHW2NHWC:0"\n  device: "/gpu:0"\n}\nnode {\n  name: "GRADIENTS/conv1/pool1_grad"\n  op: "Blob"\n  input: "GRADIENTS/c/FCGradient:2"\n  device: "/gpu:0"\n}\nnode {\n  name: "conv1/pool1"\n  op: "Blob"\n  input: "conv1/MaxPool:0"\n  device: "/gpu:0"\n}\n'

class TensorboardExporterTest(unittest.TestCase):

    def test_that_operators_gets_non_colliding_names(self):
        if False:
            while True:
                i = 10
        op = caffe2_pb2.OperatorDef()
        op.type = 'foo'
        op.input.extend(['foo'])
        tb._fill_missing_operator_names([op])
        self.assertEqual(op.input[0], 'foo')
        self.assertEqual(op.name, 'foo_1')

    def test_that_replacing_colons_gives_non_colliding_names(self):
        if False:
            i = 10
            return i + 15
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo:0'
        op.input.extend(['foo:0', 'foo$0'])
        shapes = {'foo:0': [1]}
        track_blob_names = tb._get_blob_names([op])
        tb._replace_colons(shapes, track_blob_names, [op], '$')
        self.assertEqual(op.input[0], 'foo$0')
        self.assertEqual(op.input[1], 'foo$0_1')
        self.assertEqual(op.name, 'foo$0')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['foo$0'], [1])
        self.assertEqual(len(track_blob_names), 2)
        self.assertEqual(track_blob_names['foo$0'], 'foo:0')
        self.assertEqual(track_blob_names['foo$0_1'], 'foo$0')

    def test_that_adding_gradient_scope_does_no_fancy_renaming(self):
        if False:
            i = 10
            return i + 15
        op = caffe2_pb2.OperatorDef()
        op.name = 'foo_grad'
        op.input.extend(['foo_grad', 'foo_grad_1'])
        shapes = {'foo_grad': [1]}
        track_blob_names = tb._get_blob_names([op])
        tb._add_gradient_scope(shapes, track_blob_names, [op])
        self.assertEqual(op.input[0], 'GRADIENTS/foo_grad')
        self.assertEqual(op.input[1], 'GRADIENTS/foo_grad_1')
        self.assertEqual(op.name, 'GRADIENTS/foo_grad')
        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes['GRADIENTS/foo_grad'], [1])
        self.assertEqual(len(track_blob_names), 2)
        self.assertEqual(track_blob_names['GRADIENTS/foo_grad'], 'foo_grad')
        self.assertEqual(track_blob_names['GRADIENTS/foo_grad_1'], 'foo_grad_1')

    def test_that_auto_ssa_gives_non_colliding_names(self):
        if False:
            print('Hello World!')
        op1 = caffe2_pb2.OperatorDef()
        op1.output.extend(['foo'])
        op2 = caffe2_pb2.OperatorDef()
        op2.input.extend(['foo'])
        op2.output.extend(['foo'])
        op2.output.extend(['foo_1'])
        shapes = {'foo': [1], 'foo_1': [2]}
        track_blob_names = tb._get_blob_names([op1, op2])
        tb._convert_to_ssa(shapes, track_blob_names, [op1, op2])
        self.assertEqual(op1.output[0], 'foo')
        self.assertEqual(op2.input[0], 'foo')
        self.assertEqual(op2.output[0], 'foo_1')
        self.assertEqual(op2.output[1], 'foo_1_1')
        self.assertEqual(len(shapes), 3)
        self.assertEqual(shapes['foo'], [1])
        self.assertEqual(shapes['foo_1'], [1])
        self.assertEqual(shapes['foo_1_1'], [2])
        self.assertEqual(len(track_blob_names), 3)
        self.assertEqual(track_blob_names['foo'], 'foo')
        self.assertEqual(track_blob_names['foo_1'], 'foo')
        self.assertEqual(track_blob_names['foo_1_1'], 'foo_1')

    def test_simple_cnnmodel(self):
        if False:
            for i in range(10):
                print('nop')
        model = cnn.CNNModelHelper('NCHW', name='overfeat')
        (data, label) = model.ImageInput(['db'], ['data', 'label'], is_test=0)
        with core.NameScope('conv1'):
            conv1 = model.Conv(data, 'conv1', 3, 96, 11, stride=4)
            relu1 = model.Relu(conv1, conv1)
            pool1 = model.MaxPool(relu1, 'pool1', kernel=2, stride=2)
        with core.NameScope('classifier'):
            fc = model.FC(pool1, 'fc', 4096, 1000)
            pred = model.Softmax(fc, 'pred')
            xent = model.LabelCrossEntropy([pred, label], 'xent')
            loss = model.AveragedLoss(xent, 'loss')
        model.net.RunAllOnGPU()
        model.param_init_net.RunAllOnGPU()
        model.AddGradientOperators([loss], skip=1)
        track_blob_names = {}
        graph = tb.cnn_to_graph_def(model, track_blob_names=track_blob_names, shapes={})
        self.assertEqual(track_blob_names['GRADIENTS/conv1/conv1_b_grad'], 'conv1/conv1_b_grad')
        self.maxDiff = None
        sep = 'node {'
        expected = '\n'.join(sorted((sep + '\n  ' + part.strip() for part in EXPECTED.strip().split(sep) if part.strip())))
        actual = '\n'.join(sorted((sep + '\n  ' + part.strip() for part in str(graph).strip().split(sep) if part.strip())))
        self.assertMultiLineEqual(actual, expected)
if __name__ == '__main__':
    unittest.main()