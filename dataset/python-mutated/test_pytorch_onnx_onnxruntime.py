from __future__ import annotations
import functools
import io
import itertools
import os
import unittest
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple, Type, Union
import numpy as np
import onnx
import onnx_test_common
import parameterized
import torch
import torchvision
from model_defs import lstm_flattening_result, rnn_model_with_packed_sequence, word_language_model
from pytorch_test_common import BATCH_SIZE, RNN_BATCH_SIZE, RNN_HIDDEN_SIZE, RNN_INPUT_SIZE, RNN_SEQUENCE_LENGTH, skipDtypeChecking, skipIfQuantizationBackendQNNPack, skipIfUnsupportedMaxOpsetVersion, skipIfUnsupportedMinOpsetVersion, skipIfUnsupportedOpsetVersion, skipScriptTest, skipShapeChecking, skipTraceTest
from torch import Tensor
from torch.nn.utils import rnn as rnn_utils
from torch.onnx import errors, verification
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import skipIfNoLapack

def _init_test_generalized_rcnn_transform():
    if False:
        i = 10
        return i + 15
    min_size = 100
    max_size = 200
    image_mean = [0.485, 0.456, 0.406]
    image_std = [0.229, 0.224, 0.225]
    transform = torchvision.models.detection.transform.GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    return transform

def _init_test_rpn():
    if False:
        while True:
            i = 10
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    rpn_anchor_generator = torchvision.models.detection.rpn.AnchorGenerator(anchor_sizes, aspect_ratios)
    out_channels = 256
    rpn_head = torchvision.models.detection.rpn.RPNHead(out_channels, rpn_anchor_generator.num_anchors_per_location()[0])
    rpn_fg_iou_thresh = 0.7
    rpn_bg_iou_thresh = 0.3
    rpn_batch_size_per_image = 256
    rpn_positive_fraction = 0.5
    rpn_pre_nms_top_n = dict(training=2000, testing=1000)
    rpn_post_nms_top_n = dict(training=2000, testing=1000)
    rpn_nms_thresh = 0.7
    rpn_score_thresh = 0.0
    rpn = torchvision.models.detection.rpn.RegionProposalNetwork(rpn_anchor_generator, rpn_head, rpn_fg_iou_thresh, rpn_bg_iou_thresh, rpn_batch_size_per_image, rpn_positive_fraction, rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh, score_thresh=rpn_score_thresh)
    return rpn

def _construct_tensor_for_quantization_test(shape: Tuple[int, ...], offset: Optional[Union[int, float]]=None, max_val: Optional[Union[int, float]]=None) -> Tensor:
    if False:
        print('Hello World!')
    'Helper function to generate weights and test inputs in a deterministic way.\n\n    Due to difference in implementation details between PyTorch and ONNXRuntime, randomly generated\n    test data for quantization tests can be flaky. To help stablize the test, this helper function is\n    used to generate weights and test inputs in a deterministic way.\n\n    Args:\n        shape (Tuple[int]): Shape for tensor to construct.\n        offset (Optional[Union[int, float]]): Offset to be added to the generated tensor.\n        max_val (Optional[Union[int, float]]): If any element within tensor has a larger absolute value than\n            max_val, the tensor will be scaled by max_val / tensor.abs().max(). This step is done after\n            applying offset.\n    '
    tensor = torch.arange(np.prod(shape), dtype=torch.float).view(shape)
    if offset is not None:
        tensor = tensor + offset
    if max_val is not None and tensor.abs().max() > max_val:
        tensor = tensor * max_val / tensor.abs().max()
    return tensor

def _parameterized_class_attrs_and_values(min_opset_version: int, max_opset_version: int):
    if False:
        return 10
    attrs = ('opset_version', 'is_script', 'keep_initializers_as_inputs')
    input_values = []
    input_values.extend(itertools.product((7, 8), (True, False), (True,)))
    if min_opset_version < 9:
        raise ValueError('min_opset_version must be >= 9')
    input_values.extend(itertools.product(range(min_opset_version, max_opset_version + 1), (True, False), (True, False)))
    return {'attrs': attrs, 'input_values': input_values}

def _parametrize_rnn_args(arg_name):
    if False:
        i = 10
        return i + 15
    options = {'layers': {1: 'unilayer', 3: 'trilayer'}, 'bidirectional': {True: 'bidirectional', False: 'forward'}, 'initial_state': {True: 'with_initial_state', False: 'no_initial_state'}, 'packed_sequence': {0: 'without_sequence_lengths', 1: 'with_variable_length_sequences', 2: 'with_batch_first_sequence_lengths'}, 'dropout': {0.2: 'with_dropout', 0.0: 'without_dropout'}}
    return {'arg_str': arg_name, 'arg_values': options[arg_name].keys(), 'name_fn': lambda val: options[arg_name][val]}

@parameterized.parameterized_class(**_parameterized_class_attrs_and_values(onnx_test_common.MIN_ONNX_OPSET_VERSION, onnx_test_common.MAX_ONNX_OPSET_VERSION), class_name_func=onnx_test_common.parameterize_class_name)
@common_utils.instantiate_parametrized_tests
class TestONNXRuntime(onnx_test_common._TestONNXRuntime):

    def test_fuse_conv_bn1d(self):
        if False:
            i = 10
            return i + 15

        class Fuse(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.bn = torch.nn.BatchNorm1d(33)

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = self.conv(x)
                return self.bn(out)
        model = Fuse()
        x = torch.randn(20, 16, 50, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn2d(self):
        if False:
            for i in range(10):
                print('nop')

        class Fuse(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 2, kernel_size=1, stride=2, padding=3, bias=False)
                self.bn = torch.nn.BatchNorm2d(2)

            def forward(self, x):
                if False:
                    return 10
                out = self.conv(x)
                return self.bn(out)
        model = Fuse()
        x = torch.randn(2, 3, 2, 2, requires_grad=True)
        self.run_test(model, (x,))

    def test_fuse_conv_bn3d(self):
        if False:
            for i in range(10):
                print('nop')

        class Fuse(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv3d(3, 2, (3, 5, 2), stride=(2, 1, 1), padding=(3, 2, 0), bias=False)
                self.bn = torch.nn.BatchNorm3d(2)

            def forward(self, x):
                if False:
                    return 10
                out = self.conv(x)
                return self.bn(out)
        model = Fuse()
        x = torch.randn(2, 3, 10, 50, 100, requires_grad=True)
        self.run_test(model, (x,), rtol=0.001, atol=1e-06)

    def test_fuse_conv_in_block(self):
        if False:
            while True:
                i = 10

        class Fuse(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv1d(in_channels=5, out_channels=5, kernel_size=3, stride=1, padding=2, dilation=1)
                self.bn = torch.nn.BatchNorm1d(5)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                results_available = True
                if x.sum() > -1:
                    results_available = False
                if results_available:
                    x = self.conv(x)
                    x = self.bn(x)
                return x
        model = Fuse()
        x = torch.randn(2, 5, 9, requires_grad=True)
        self.run_test(torch.jit.script(model), (x,), input_names=['x'], dynamic_axes={'x': [0, 2]}, rtol=0.001, atol=1e-06)

    def test_conv_tbc(self):
        if False:
            for i in range(10):
                print('nop')
        from torch.nn.modules.utils import _single

        class ConvTBC(torch.nn.Module):

            def __init__(self, in_channels, out_channels, kernel_size, padding=0):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.in_channels = in_channels
                self.out_channels = out_channels
                self.kernel_size = _single(kernel_size)
                self.padding = _single(padding)
                self.weight = torch.nn.Parameter(Tensor(self.kernel_size[0], in_channels, out_channels))
                self.bias = torch.nn.Parameter(Tensor(out_channels))
                self.reset_parameters()

            def reset_parameters(self):
                if False:
                    while True:
                        i = 10
                torch.nn.init.xavier_normal_(self.weight)
                torch.nn.init.zeros_(self.bias)

            def conv_tbc(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.conv_tbc(input.contiguous(), self.weight, self.bias, self.padding[0])

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return self.conv_tbc(input)
        in_channels = 3
        out_channels = 5
        kernel_size = 5
        model = ConvTBC(in_channels, out_channels, kernel_size, padding=0)
        x = torch.randn(10, 7, in_channels, requires_grad=True)
        self.run_test(model, (x,), atol=1e-05)

    def test_reshape_constant_fold(self):
        if False:
            while True:
                i = 10

        class Reshape(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('weight', torch.ones(5))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                scale_1 = self.weight.reshape(1, -1, 1, 1)
                return x * scale_1
        x = torch.randn(4, 5)
        self.run_test(Reshape(), (x,), rtol=0.001, atol=1e-05)

    def run_word_language_model(self, model_name):
        if False:
            return 10
        ntokens = 50
        emsize = 5
        nhid = 5
        nlayers = 5
        dropout = 0.2
        tied = False
        batchsize = 5
        if model_name == 'GRU':
            model = word_language_model.RNNModelWithTensorHidden(model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize)
        elif model_name == 'LSTM':
            model = word_language_model.RNNModelWithTupleHidden(model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize)
        else:
            model = word_language_model.RNNModel(model_name, ntokens, emsize, nhid, nlayers, dropout, tied, batchsize)
        x = torch.arange(0, ntokens).long().view(-1, batchsize)
        self.run_test(model, (x, model.hidden))

    def get_image(self, rel_path: str, size: Tuple[int, int]) -> Tensor:
        if False:
            return 10
        from PIL import Image
        from torchvision import transforms
        data_dir = os.path.join(os.path.dirname(__file__), 'assets')
        path = os.path.join(data_dir, *rel_path.split('/'))
        image = Image.open(path).convert('RGB').resize(size, Image.BILINEAR)
        return transforms.ToTensor()(image)

    def get_test_images(self) -> Tuple[List[Tensor], List[Tensor]]:
        if False:
            while True:
                i = 10
        return ([self.get_image('grace_hopper_517x606.jpg', (100, 320))], [self.get_image('rgb_pytorch.png', (250, 380))])

    def test_paste_mask_in_image(self):
        if False:
            while True:
                i = 10
        masks = torch.rand(10, 1, 26, 26)
        boxes = torch.rand(10, 4)
        boxes[:, 2:] += torch.rand(10, 2)
        boxes *= 50
        o_im_s = (100, 100)
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        out = paste_masks_in_image(masks, boxes, o_im_s)
        jit_trace = torch.jit.trace(paste_masks_in_image, (masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])]))
        out_trace = jit_trace(masks, boxes, [torch.tensor(o_im_s[0]), torch.tensor(o_im_s[1])])
        assert torch.all(out.eq(out_trace))
        masks2 = torch.rand(20, 1, 26, 26)
        boxes2 = torch.rand(20, 4)
        boxes2[:, 2:] += torch.rand(20, 2)
        boxes2 *= 100
        o_im_s2 = (200, 200)
        from torchvision.models.detection.roi_heads import paste_masks_in_image
        out2 = paste_masks_in_image(masks2, boxes2, o_im_s2)
        out_trace2 = jit_trace(masks2, boxes2, [torch.tensor(o_im_s2[0]), torch.tensor(o_im_s2[1])])
        assert torch.all(out2.eq(out_trace2))

    def test_heatmaps_to_keypoints(self):
        if False:
            for i in range(10):
                print('nop')
        maps = torch.rand(10, 1, 26, 26)
        rois = torch.rand(10, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints
        out = heatmaps_to_keypoints(maps, rois)
        jit_trace = torch.jit.trace(heatmaps_to_keypoints, (maps, rois))
        out_trace = jit_trace(maps, rois)
        assert torch.all(out[0].eq(out_trace[0]))
        assert torch.all(out[1].eq(out_trace[1]))
        maps2 = torch.rand(20, 2, 21, 21)
        rois2 = torch.rand(20, 4)
        from torchvision.models.detection.roi_heads import heatmaps_to_keypoints
        out2 = heatmaps_to_keypoints(maps2, rois2)
        out_trace2 = jit_trace(maps2, rois2)
        assert torch.all(out2[0].eq(out_trace2[0]))
        assert torch.all(out2[1].eq(out_trace2[1]))

    def test_word_language_model_RNN_TANH(self):
        if False:
            i = 10
            return i + 15
        self.run_word_language_model('RNN_TANH')

    def test_word_language_model_RNN_RELU(self):
        if False:
            print('Hello World!')
        self.run_word_language_model('RNN_RELU')

    @skipScriptTest()
    def test_word_language_model_LSTM(self):
        if False:
            print('Hello World!')
        self.run_word_language_model('LSTM')

    def test_word_language_model_GRU(self):
        if False:
            i = 10
            return i + 15
        self.run_word_language_model('GRU')

    def test_index_1d(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input[0]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_1dimslice(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return input[0:1, :]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_sliceint(self):
        if False:
            while True:
                i = 10

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input[1, :]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    def test_index_2d_neg_slice(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input[0:-1, :]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_mask(self):
        if False:
            print('Hello World!')

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input[torch.tensor([0, 1, 0], dtype=torch.uint8)]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input[torch.tensor([0, 1, 0], dtype=torch.bool)]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_data(self):
        if False:
            i = 10
            return i + 15

        class Data(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.new_zeros(x.data.size())
        x = torch.randn(3, 4)
        self.run_test(Data(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(Data(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_mask_nd(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input[input > 0]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), m1)

    @skipScriptTest()
    def test_dict(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(torch.nn.Module):

            def forward(self, x_in):
                if False:
                    i = 10
                    return i + 15
                x_out = {}
                x_out['test_key_out'] = torch.add(x_in[list(x_in.keys())[0]], list(x_in.keys())[0])
                return x_out
        x = {torch.tensor(1.0): torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()
    def test_dict_str(self):
        if False:
            while True:
                i = 10

        class MyModel(torch.nn.Module):

            def forward(self, x_in):
                if False:
                    while True:
                        i = 10
                x_out = {}
                x_out['test_key_out'] = torch.add(x_in['test_key_in'], 2.0)
                return x_out
        x = {'test_key_in': torch.randn(1, 2, 3)}
        self.run_test(MyModel(), (x,))

    @skipScriptTest()
    def test_dict_output(self):
        if False:
            print('Hello World!')

        class DictModelOutput(OrderedDict):
            tensor_out: Tensor
            tuple_out: Optional[Tuple[Tensor]] = None
            list_out: Optional[List[Tensor]] = None

        class MyModel(torch.nn.Module):

            def forward(self, a, b, c, d):
                if False:
                    return 10
                return DictModelOutput(tensor_out=a, tuple_out=(b, c), list_out=[d])
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_output(self):
        if False:
            i = 10
            return i + 15

        class MyModel(torch.nn.Module):

            def forward(self, a, b, c, d):
                if False:
                    for i in range(10):
                        print('nop')
                return (a, (b, c), d)
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_nested_tuple_output(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(torch.nn.Module):

            def forward(self, a, b, c, d):
                if False:
                    while True:
                        i = 10
                return (a, ((b,), (c, d)))
        a = torch.randn(2, 3)
        b = torch.randn(2, 3)
        c = torch.randn(2, 3)
        d = torch.randn(2, 3)
        self.run_test(MyModel(), (a, b, c, d))

    def test_tuple_input(self):
        if False:
            print('Hello World!')

        class TupleModel(torch.nn.Module):

            def forward(self, a: Tuple[Tensor, Tensor]):
                if False:
                    for i in range(10):
                        print('nop')
                return a
        x = (torch.randn(3, 4), torch.randn(4, 3))
        self.run_test(TupleModel(), input_args=(x,))

    def test_tuple_primitive_input(self):
        if False:
            print('Hello World!')

        class TupleModel(torch.nn.Module):

            def forward(self, a: Tuple[int, Tensor], b):
                if False:
                    return 10
                return (a[0], a[1] + b)
        x = (3, torch.randn(4, 3))
        y = torch.randn(4, 3)
        self.run_test(TupleModel(), input_args=(x, y))

    def test_nested_tuple_input(self):
        if False:
            i = 10
            return i + 15

        class NestedTupleModel(torch.nn.Module):

            def forward(self, a, b: Tuple[Tensor, Tuple[Tensor, Tensor]]):
                if False:
                    return 10
                return a + b[0] + b[1][0] + b[1][1]
        x = torch.randn(4, 5)
        y = (torch.randn(4, 5), (torch.randn(1, 5), torch.randn(4, 1)))
        self.run_test(NestedTupleModel(), input_args=(x, y))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_none(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x, y: Optional[Tensor]=None, z: Optional[Tensor]=None):
                if False:
                    return 10
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))
        self.run_test(model, (x,), {'y': y, 'z': None})
        self.run_test(model, (x,), {'y': None, 'z': z})
        self.run_test(model, (x,), {'z': z})
        self.run_test(model, (x,), {'y': y})

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def forward(self, x, y: Optional[Tensor]=torch.ones(2, 3), z: Optional[Tensor]=torch.zeros(2, 3)):
                if False:
                    while True:
                        i = 10
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, y, None))
        self.run_test(model, (x, None, z))

    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional_default_tensor_script(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def forward(self, x, y: Optional[Tensor]=torch.ones(2, 3), z: Optional[Tensor]=torch.zeros(2, 3)):
                if False:
                    i = 10
                    return i + 15
                if y is not None:
                    return x + y
                if z is not None:
                    return x + z
                return x
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        z = torch.randn(2, 3)
        model = torch.jit.script(Model())
        self.run_test(model, (x, y, z), input_names=('x', 'y', 'z'))
        self.run_test(model, (x,), {'y': y, 'z': z}, input_names=('x', 'y', 'z'))
        self.run_test(model, (x,), {'y': y}, input_names=('x', 'y'))
        for (example_inputs, example_kwargs) in (((x, y, None), {}), ((x, None, z), {}), ((x,), {'y': y, 'z': None}), ((x,), {'y': None, 'z': z})):
            with self.assertRaisesRegex(ValueError, "args contained 1 None's after flattening."):
                self.run_test(model, example_inputs, example_kwargs, input_names=('x', 'y', 'z'))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_none(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def forward(self, x: Optional[Tensor]=None, y: Optional[Tensor]=None):
                if False:
                    for i in range(10):
                        print('nop')
                if x is not None:
                    return x
                if y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)
        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (), {'x': x, 'y': None}, input_names=('x',))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x: Optional[Tensor]=torch.ones(2, 3), y: Optional[Tensor]=torch.zeros(2, 3)):
                if False:
                    while True:
                        i = 10
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (None, y))
        with self.assertRaisesRegex(ValueError, 'got too many positional inputs'):
            self.run_test(model, (x, y))

    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_all_optional_default_tensor_script(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x: Optional[Tensor]=torch.ones(2, 3), y: Optional[Tensor]=torch.zeros(2, 3)):
                if False:
                    print('Hello World!')
                if x is not None:
                    return x
                elif y is not None:
                    return y
                else:
                    return torch.tensor(-1.0)
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        model = torch.jit.script(Model())
        self.run_test(model, (x,))
        with self.assertRaisesRegex(AssertionError, 'Tensor-likes are not close!'):
            self.run_test(model, (), {'y': y}, input_names=['y'])
        self.run_test(model, (x, y))
        self.run_test(model, (), {'x': x, 'y': y}, input_names=('x', 'y'))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logit(self):
        if False:
            return 10

        class Logit(torch.nn.Module):

            def __init__(self, eps):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.eps = eps

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.logit(self.eps)
        model = Logit(eps=1e-06)
        self.run_test(model, torch.randn(1, 3, 640, 640))

    class Atleast1d(torch.nn.Module):

        def forward(self, t, w, x, y, z):
            if False:
                return 10
            return torch.atleast_1d((t, w, x, y, z))

    class Atleast2d(torch.nn.Module):

        def forward(self, t, w, x, y, z):
            if False:
                i = 10
                return i + 15
            return torch.atleast_2d((t, w, x, y, z))

    class Atleast3d(torch.nn.Module):

        def forward(self, t, w, x, y, z):
            if False:
                while True:
                    i = 10
            return torch.atleast_3d((t, w, x, y, z))

    class Atleast1dTensor(torch.nn.Module):

        def forward(self, x):
            if False:
                print('Hello World!')
            return torch.atleast_1d(x)

    class Atleast2dTensor(torch.nn.Module):

        def forward(self, x):
            if False:
                return 10
            return torch.atleast_2d(x)

    class Atleast3dTensor(torch.nn.Module):

        def forward(self, x):
            if False:
                while True:
                    i = 10
            return torch.atleast_3d(x)

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize('module_class', (Atleast1d, Atleast2d, Atleast3d))
    def test_atleast_nd_list_input(self, module_class: torch.nn.Module):
        if False:
            return 10
        inputs = (torch.tensor(1.0), torch.randn(2), torch.randn(2, 3), torch.randn(2, 3, 4), torch.randn(2, 3, 4, 5))
        self.run_test(module_class(), inputs)

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    @common_utils.parametrize('module_class', (Atleast1dTensor, Atleast2dTensor, Atleast3dTensor))
    @common_utils.parametrize('inputs', [torch.tensor(1.0), torch.randn(2), torch.randn(2, 3), torch.randn(2, 3, 4), torch.randn(2, 3, 4, 5)])
    def test_atleast_nd_single_tensor_input(self, module_class: torch.nn.Module, inputs: torch.Tensor):
        if False:
            i = 10
            return i + 15
        self.run_test(module_class(), inputs)

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_mixed_optional(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def forward(self, x, y: Optional[Tensor]):
                if False:
                    print('Hello World!')
                if y is not None:
                    return x + y
                return x
        x = torch.randn(2, 3)
        model = Model()
        self.run_test(model, (x, None))
        self.run_test(model, (x, x))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def forward(self, x, y: Tuple[Optional[Tensor], Optional[Tensor]]):
                if False:
                    return 10
                if y[0] is not None:
                    return x + y[0]
                if y[1] is not None:
                    return x + y[1]
                return x
        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def forward(self, x, y: Tuple[Optional[Tensor], Optional[Tensor]]=(torch.zeros(2, 3), torch.zeros(2, 3))):
                if False:
                    i = 10
                    return i + 15
                (y0, y1) = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x
        x = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        self.run_test(Model(), (x, (None, y1)))

    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(15)
    def test_tuple_of_optional_default_tensor_script(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x, y: Tuple[Optional[Tensor], Optional[Tensor]]=(torch.zeros(2, 3), torch.zeros(2, 3))):
                if False:
                    return 10
                (y0, y1) = y
                if y0 is not None:
                    return x + y0
                if y1 is not None:
                    return x + y1
                return x
        x = torch.randn(2, 3)
        y0 = torch.randn(2, 3)
        y1 = torch.randn(2, 3)
        model = torch.jit.script(Model())
        with self.assertRaisesRegex(ValueError, "args contained 1 None's after flattening."):
            self.run_test(model, (x, (None, y1)))
        self.run_test(model, (x, (y0, y1)))
        torch.onnx.export(model, (x, {'y': (y0, y1)}), io.BytesIO(), opset_version=self.opset_version)

    def test_primitive_input_integer(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x: int, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y
        x = 3
        y = torch.randint(10, (2, 3, 4))
        self.run_test(Model(), (x, y))

    @skipDtypeChecking
    def test_primitive_input_floating(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def forward(self, x: float, y):
                if False:
                    return 10
                return x + y
        x = 3.0
        y = torch.randn(2, 3, 4)
        self.run_test(Model(), (x, y))

    def test_primitive_input_bool(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def forward(self, flag: bool, x, y):
                if False:
                    return 10
                if flag:
                    return x
                else:
                    return y
        flag = True
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(torch.jit.script(Model()), (flag, x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cste_script(self):
        if False:
            return 10

        class MyModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (torch.zeros(x.size(0)), torch.ones((x.size(1), x.size(0)), dtype=torch.int64))
        x = torch.randn(3, 4)
        self.run_test(MyModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(MyModel(), x, remained_onnx_input_idx=[])

    def test_scalar_tensor(self):
        if False:
            while True:
                i = 10

        class test(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return (torch.scalar_tensor(input.size(0)), torch.scalar_tensor(input.size(1), dtype=torch.int64))
        x = torch.randn(2, 3, 4)
        y = torch.randn(7, 8, 9)
        model = test()
        self.run_test(model, x, additional_test_inputs=[y], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2]})

    def test_tensor(self):
        if False:
            print('Hello World!')

        class ScalarInputModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.tensor(input.shape[1])
        x = torch.randn(3, 4)
        self.run_test(ScalarInputModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(ScalarInputModel(), x, remained_onnx_input_idx=[])

        class TensorInputModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.tensor([input.shape[0], input.shape[1]])
        x = torch.randn(3, 4)
        self.run_test(TensorInputModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(TensorInputModel(), x, remained_onnx_input_idx=[])

        class FloatInputModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.tensor([float(input)])
        x = torch.randn(1)
        self.run_test(FloatInputModel(), x)

        class InputWithDtypeModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.tensor(input.shape[1], dtype=torch.long)
        x = torch.randn(3, 4)
        self.run_test(InputWithDtypeModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(InputWithDtypeModel(), x, remained_onnx_input_idx=[])

        class MixedInputModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    return 10
                return torch.tensor([input.shape[0], int(input)])
        x = torch.randn(1)
        self.run_test(MixedInputModel(), x)

    def test_hardtanh(self):
        if False:
            print('Hello World!')
        model = torch.nn.Hardtanh(-1.5, 2.5)
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardtanh_script_with_default_values(self):
        if False:
            while True:
                i = 10

        class MyModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return torch.nn.functional.hardtanh(x)
        x = torch.arange(-5, 5).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardswish(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Hardswish()
        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_hardswish_script(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.nn.functional.hardswish(x)
        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(MyModel(), x)

    def test_hardsigmoid(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.Hardsigmoid()
        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-3).to(dtype=torch.float32)
        self.run_test(model, x)

    def test_tanhshrink(self):
        if False:
            return 10
        model = torch.nn.Tanhshrink()
        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.Hardshrink()
        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hardshrink_dtype(self):
        if False:
            print('Hello World!')
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Hardshrink(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Softshrink()
        x = torch.rand(3, 3).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(0.5).to(dtype=torch.float32)
        self.run_test(model, x)
        x = torch.tensor(-0.5).to(dtype=torch.float32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_softshrink_dtype(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(3, 3).to(dtype=torch.float64)
        self.run_test(torch.nn.Softshrink(), x)

    def test_clamp(self):
        if False:
            return 10

        class ClampModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.clamp(-0.5, 0.5)
        x = torch.randn(3, 4)
        self.run_test(ClampModel(), x)

        class ClampMinModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.clamp(min=-0.5)
        x = torch.randn(3, 4)
        self.run_test(ClampMinModel(), x)

        class ClampMaxModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.clamp(max=0.5)
        x = torch.randn(3, 4)
        self.run_test(ClampMaxModel(), x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_clamp_dyn(self):
        if False:
            for i in range(10):
                print('nop')

        class ClampMaxModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return x.clamp(None, x.size(0))
        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMaxModel(), x)

        class ClampMinModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return x.clamp(x.size(0), None)
        x = torch.arange(16).view(4, 4).float()
        self.run_test(ClampMinModel(), x)

        class ClampMinMaxModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.clamp(x.size(0), x.size(1))
        x = torch.arange(16).view(2, 8).float()
        self.run_test(ClampMinMaxModel(), x)

        class ClampTensorModel(torch.nn.Module):

            def forward(self, x, min, max):
                if False:
                    while True:
                        i = 10
                return x.clamp(min, max)
        x = torch.randn(3, 4)
        y = torch.randn(3, 4)
        z = torch.randn(3, 4)
        self.run_test(ClampTensorModel(), (x, y, z))

        class ClampTensorMinModel(torch.nn.Module):

            def forward(self, x, min):
                if False:
                    return 10
                return x.clamp(min=min)
        self.run_test(ClampTensorMinModel(), (x, y))

        class ClampTensorMaxModel(torch.nn.Module):

            def forward(self, x, max):
                if False:
                    return 10
                return x.clamp(max=max)
        self.run_test(ClampTensorMaxModel(), (x, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_trace(self):
        if False:
            while True:
                i = 10

        class FullModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.full((3, 4), x, dtype=torch.long)
        x = torch.tensor(12)
        self.run_test(FullModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_script(self):
        if False:
            return 10

        class FullModelScripting(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return torch.full((3, 4), x, dtype=torch.long)
        x = torch.tensor(12)
        self.run_test(FullModelScripting(), x)

    def test_fuse_addmm(self):
        if False:
            return 10

        class AddmmModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.mm(x, x) + x
        x = torch.ones(3, 3)
        self.run_test(AddmmModel(), x)

    def test_maxpool(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.MaxPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_conv(self):
        if False:
            return 10

        class TraceModel(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = torch.nn.Conv3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, input1, input2, input3):
                if False:
                    print('Hello World!')
                return (self.conv1(input1), self.conv2(input2), self.conv3(input3))
        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)
        self.run_test(TraceModel(), (x1, x2, x3), atol=0.0001)

    def test_conv_str_padding(self):
        if False:
            return 10

        class TraceModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv1d(16, 33, 3, padding='valid')
                self.conv2 = torch.nn.Conv2d(16, 33, (3, 5), stride=1, padding='valid', dilation=(3, 1))
                self.conv3 = torch.nn.Conv3d(16, 33, (3, 5, 2), stride=1, padding='same')

            def forward(self, input1, input2, input3):
                if False:
                    for i in range(10):
                        print('nop')
                return (self.conv1(input1), self.conv2(input2), self.conv3(input3))
        x1 = torch.randn(20, 16, 50)
        x2 = torch.randn(20, 16, 50, 50)
        x3 = torch.randn(20, 16, 10, 50, 50)
        self.run_test(TraceModel(), (x1, x2, x3), atol=0.0001)

    def test_conv_shape_inference(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv2 = torch.nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return self.conv2(input) + 2
        x = torch.randn(20, 16, 50, 100)
        self.run_test(Model(), x, atol=0.0001, input_names=['x'], dynamic_axes={'x': [0]})

    def test_conv_transpose(self):
        if False:
            print('Hello World!')

        class TraceModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.ConvTranspose1d(16, 33, 3, stride=2)
                self.conv2 = torch.nn.ConvTranspose2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
                self.conv3 = torch.nn.ConvTranspose3d(16, 33, (3, 5, 2), stride=(2, 1, 1), padding=(4, 2, 0))

            def forward(self, input1, input2, input3):
                if False:
                    print('Hello World!')
                return (self.conv1(input1), self.conv2(input2), self.conv3(input3))
        x1 = torch.randn(20, 16, 10)
        x2 = torch.randn(20, 16, 10, 10)
        x3 = torch.randn(20, 16, 10, 10, 10)
        self.run_test(TraceModel(), (x1, x2, x3), atol=0.0001)

    def test_numpy_T(self):
        if False:
            for i in range(10):
                print('nop')

        class NumpyTranspose(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.T
        self.run_test(NumpyTranspose(), torch.randn(4, 7))

    def test_transpose_infer_shape(self):
        if False:
            print('Hello World!')

        class TransposeModule(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                return x.transpose(0, 1)
        x = torch.randn(32, 3, 64, 64)
        y = torch.randn(16, 3, 8, 64)
        self.run_test(TransposeModule(), x, input_names=['x'], dynamic_axes={'x': [0, 2]}, additional_test_inputs=[y])

    def squeeze_model_tests(self, d, x1, x2):
        if False:
            print('Hello World!')

        class Squeeze(torch.nn.Module):

            def __init__(self, d):
                if False:
                    print('Hello World!')
                super().__init__()
                self.d = d

            def forward(self, x):
                if False:
                    return 10
                if self.d is not None:
                    return torch.squeeze(x, dim=self.d)
                else:
                    return torch.squeeze(x)
        x2 = [] if x2 is None else [x2]
        if len(x2) > 0:
            self.run_test(Squeeze(d), x1, input_names=['input'], dynamic_axes={'input': {0: '0', 1: '1', 2: '2'}}, additional_test_inputs=x2)
        else:
            self.run_test(Squeeze(d), x1)

    def test_squeeze_without_no_op(self):
        if False:
            return 10
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(1, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_dynamic(self):
        if False:
            print('Hello World!')
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(1, x_squeeze, x_noop)

    def test_squeeze_neg_without_no_op(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(2, 1, 4)
        self.squeeze_model_tests(-2, x, None)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_neg(self):
        if False:
            print('Hello World!')
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(-2, x_squeeze, x_noop)

    def test_squeeze_all_dims(self):
        if False:
            while True:
                i = 10
        x_squeeze = torch.randn(2, 1, 4)
        x_noop = torch.randn(2, 2, 3)
        self.squeeze_model_tests(None, x_squeeze, x_noop)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_no_op(self):
        if False:
            print('Hello World!')
        x_noop = torch.randn(2, 1, 4)
        x_squeeze = torch.randn(2, 2, 1)
        self.squeeze_model_tests(2, x_noop, x_squeeze)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_squeeze_runtime_dim(self):
        if False:
            i = 10
            return i + 15

        class Squeeze(torch.nn.Module):

            def forward(self, d1, d2):
                if False:
                    while True:
                        i = 10
                t = torch.zeros(d1[0], d2[0])
                return t.squeeze(0)
        d1 = torch.tensor([1])
        d3 = torch.tensor([3])
        d4 = torch.tensor([4])
        self.run_test(Squeeze(), (d1, d4), additional_test_inputs=[(d3, d4)])
        self.run_test(Squeeze(), (d3, d4), additional_test_inputs=[(d1, d3)])

    def test_squeeze(self):
        if False:
            for i in range(10):
                print('nop')

        class Squeeze(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.squeeze(x, dim=-2)
        x = torch.randn(2, 1, 4)
        self.run_test(Squeeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_squeeze_dynamic_dim(self):
        if False:
            print('Hello World!')

        class Squeeze(torch.nn.Module):

            def forward(self, x, dim: int):
                if False:
                    i = 10
                    return i + 15
                return torch.squeeze(x, dim)
        x = torch.randn(2, 1, 4)
        dim = 1
        self.run_test(Squeeze(), (x, dim))

    def test_unsqueeze(self):
        if False:
            return 10

        class Unsqueeze(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.unsqueeze(x, dim=-2)
        x = torch.randn(2, 3, 4)
        self.run_test(Unsqueeze(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_unsqueeze_dynamic_dim(self):
        if False:
            print('Hello World!')

        class Unsqueeze(torch.nn.Module):

            def forward(self, x, dim: int):
                if False:
                    i = 10
                    return i + 15
                return torch.unsqueeze(x, dim)
        x = torch.randn(2, 1, 4)
        dim = -1
        self.run_test(Unsqueeze(), (x, dim))

    def test_maxpool_default_stride(self):
        if False:
            for i in range(10):
                print('nop')

        class MaxPoolModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.max_pool2d(x, 2)
        model = MaxPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_adaptive(self):
        if False:
            return 10
        model = torch.nn.AdaptiveMaxPool1d(5, return_indices=False)
        x = torch.randn(20, 16, 50, requires_grad=True)
        y = torch.randn(32, 16, 50, requires_grad=True)
        self.run_test(model, x, input_names=['x'], dynamic_axes={'x': [0]}, additional_test_inputs=[y])

    def test_maxpool_2d(self):
        if False:
            return 10
        model = torch.nn.MaxPool2d(5, padding=(1, 2))
        x = torch.randn(1, 20, 16, 50, requires_grad=True)
        self.run_test(model, x)

    def test_maxpool_1d_ceil(self):
        if False:
            return 10
        model = torch.nn.MaxPool1d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_maxpool_2d_ceil(self):
        if False:
            print('Hello World!')
        model = torch.nn.MaxPool2d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    def test_maxpool_3d_ceil(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.MaxPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dynamic(self):
        if False:
            i = 10
            return i + 15

        class test(torch.nn.Module):

            def __init__(self, in_channels, out_channels):
                if False:
                    while True:
                        i = 10
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.MaxPool2d((2, 2), stride=2, ceil_mode=True)
                self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.norm(self.conv(self.avgpool(x)))
        model = test(8, 16)
        inputs = torch.randn(2, 8, 64, 64)
        self.run_test(model, inputs, input_names=['input_0'], dynamic_axes={'input_0': {3: 'x', 2: 'y'}, 'output_0': {3: 'x', 2: 'y'}}, output_names=['output_0'])

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_1d_ceil_corner(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.MaxPool1d(kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=False)
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_2d_ceil_corner(self):
        if False:
            while True:
                i = 10
        model = torch.nn.MaxPool2d(kernel_size=[1, 1], dilation=[1, 1], stride=[2, 2], ceil_mode=True, return_indices=False)
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    def test_maxpool_3d_ceil_corner(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.MaxPool3d(kernel_size=[7, 8, 4], dilation=[1, 1, 1], stride=[10, 11, 3], padding=[2, 2, 2], ceil_mode=True, return_indices=False)
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_1d_ceil_corner_with_indices(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.MaxPool1d(kernel_size=1, dilation=1, stride=2, ceil_mode=True, return_indices=True)
        x = torch.randn(1, 3, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_2d_ceil_corner_with_indices(self):
        if False:
            return 10
        model = torch.nn.MaxPool2d(kernel_size=[1, 1], dilation=[1, 1], stride=[2, 2], ceil_mode=True, return_indices=True)
        x = torch.randn(1, 3, 32, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMaxOpsetVersion(9)
    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_3d_ceil_corner_with_indices(self):
        if False:
            while True:
                i = 10
        model = torch.nn.MaxPool3d(kernel_size=[7, 8, 4], dilation=[1, 1, 1], stride=[10, 11, 3], padding=[2, 2, 2], ceil_mode=True, return_indices=True)
        x = torch.randn(1, 3, 51, 52, 45)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_maxpool_with_indices(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.MaxPool1d(2, stride=1, return_indices=True)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_maxpool_dilation(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.MaxPool1d(2, stride=1, dilation=2)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_default_stride(self):
        if False:
            while True:
                i = 10

        class AvgPoolModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.avg_pool2d(x, 2)
        model = AvgPoolModel()
        x = torch.randn(10, 20, 16, 50)
        self.run_test(model, x)

    def test_avgpool(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.AvgPool1d(2, stride=1)
        x = torch.randn(20, 16, 50)
        self.run_test(model, x)

    def test_avgpool_1d_ceil(self):
        if False:
            return 10
        model = torch.nn.AvgPool1d(3, 2, ceil_mode=True)
        x = torch.randn(1, 1, 7)
        self.run_test(model, x)

    @common_utils.parametrize('padding', (0, 1))
    @common_utils.parametrize('count_include_pad', (True, False))
    def test_avgpool_2d(self, padding, count_include_pad):
        if False:
            print('Hello World!')
        model = torch.nn.AvgPool2d(3, 3, padding=padding, count_include_pad=count_include_pad)
        x = torch.randn(20, 16, 50, 32)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(19)
    def test_avgpool_3d_ceil(self):
        if False:
            print('Hello World!')
        model = torch.nn.AvgPool3d(3, 2, ceil_mode=True)
        x = torch.randn(20, 16, 50, 44, 31)
        y = torch.randn(32, 8, 50, 44, 31)
        self.run_test(model, x, input_names=['x'], dynamic_axes={'x': [0, 1]}, additional_test_inputs=[y])

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_avgpool_dynamic(self):
        if False:
            while True:
                i = 10

        class test(torch.nn.Module):

            def __init__(self, in_channels, out_channels):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                norm_layer = functools.partial(torch.nn.BatchNorm2d, eps=0.0009)
                self.avgpool = torch.nn.AvgPool2d((2, 2), stride=2, ceil_mode=True, count_include_pad=False)
                self.conv = torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False)
                self.norm = norm_layer(out_channels)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.norm(self.conv(self.avgpool(x)))
        model = test(8, 16)
        inputs = torch.randn(2, 8, 64, 64)
        self.run_test(model, inputs, input_names=['input_0'], dynamic_axes={'input_0': {3: 'x', 2: 'y'}, 'output_0': {3: 'x', 2: 'y'}}, output_names=['output_0'])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floating_point(self):
        if False:
            for i in range(10):
                print('nop')

        class FloatingPoint(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if x.is_floating_point():
                    return x.new_zeros(x.shape)
                return x.new_zeros(x.shape)
        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x + 1
                return x
        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_floating_point_infer_dtype(self):
        if False:
            return 10

        class FloatingPoint(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x.new_zeros(x.shape[1:])
                    return x.new_zeros(x.shape)
                return x
        x = torch.randn(2, 3, 4)
        self.run_test(FloatingPoint(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(FloatingPoint(), x, remained_onnx_input_idx=[])

        class FloatingPoint(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                if x.size(0) > 1:
                    a = x + 2
                    if a.is_floating_point():
                        return x + 1
                    return x
                return x
        x = torch.randn(2, 3, 4).to(torch.int32)
        self.run_test(FloatingPoint(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_prim_min(self):
        if False:
            print('Hello World!')

        @torch.jit.script
        def list_append(boxes: List[Tensor]):
            if False:
                i = 10
                return i + 15
            temp = []
            for (i, b) in enumerate(boxes):
                temp.append(torch.full_like(b[:, 1], i))
            return temp[0]

        class Min(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                boxes = [x for _ in range(3)]
                return list_append(boxes)
        x = torch.rand(5, 5)
        self.run_test(Min(), (x,))

        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                i = 3
                return min(x[i], i)
        x = torch.arange(6, dtype=torch.int64)
        self.run_test(M(), (x,))

    def test_arithmetic(self):
        if False:
            while True:
                i = 10

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x
        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x)

    def test_arithmetic_prim_long(self):
        if False:
            return 10

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x, y: int):
                if False:
                    i = 10
                    return i + 15
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x
        x = torch.randn(2, 3, 4)
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = x + 2
                x = x - 3
                return x.shape[0]
        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_float(self):
        if False:
            i = 10
            return i + 15

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x, y: float):
                if False:
                    return 10
                x = x + y
                x = x - y
                x = x * (y * 3)
                x = x / (y * 4)
                return x
        x = torch.randn(2, 3, 4)
        y = 2.5
        self.run_test(ArithmeticModule(), (x, y))

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = x + 2
                x = x - 3
                return x.shape[1] / 2
        x = torch.randn(2, 3, 4)
        self.run_test(ArithmeticModule(), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_arithmetic_prim_bool(self):
        if False:
            print('Hello World!')

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x, y: int, z: bool, t: float):
                if False:
                    print('Hello World!')
                x = x + y
                x = x - y
                if z:
                    x = x * (y * 3)
                    x = x / (y * 4)
                return (x / t, z)
        x = torch.randn(2, 3, 4)
        y = 2
        z = False
        t = 2.5
        self.run_test(ArithmeticModule(), (x, y, z, t))

        class ArithmeticModule(torch.nn.Module):

            def forward(self, x: int, y: int):
                if False:
                    print('Hello World!')
                return x == y
        x = 3
        y = 2
        self.run_test(ArithmeticModule(), (x, y))

    @skipScriptTest(15, reason='In trace: Outputs that are always None are removed.                 In script: Outputs that are always None are removed before opset 15.                 After opset 15, we replace the None in output with Optional node.')
    def test_tuple_with_none_outputs(self):
        if False:
            for i in range(10):
                print('nop')

        class TupleModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (x, (x, None, (x, None)))
        x = torch.randn(3, 4)
        self.run_test(TupleModel(), (x,))

    def test_arithmetic_infer_dtype(self):
        if False:
            for i in range(10):
                print('nop')

        class ArithmeticModule(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                x = x.t()
                x = x + 2
                x = x - 4
                x = x * 6
                x = x / 8
                return x
        x = torch.randn(2, 3)
        self.run_test(ArithmeticModule(), x)

    @unittest.skip('Floor division on ONNX is inconsistent with eager (see #78411)')
    def test_floor_div(self):
        if False:
            while True:
                i = 10

        class FloorDivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return (x // 3, x // 2.0, x.to(dtype=torch.float64) // 3, x.to(dtype=torch.float64) // 2.0, x.to(dtype=torch.int64) // 3, x.to(dtype=torch.int64) // 2.0, x // (y + 1.0).to(dtype=torch.int64), x // y, x.to(dtype=torch.float64) // y.to(dtype=torch.int64), x.to(dtype=torch.float64) // y.to(dtype=torch.float64), x.to(dtype=torch.int64) // y.to(dtype=torch.int64), x.to(dtype=torch.int64) // y)
        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip('Floor division on ONNX is inconsistent with eager (see #78411)')
    def test_floor_div_script(self):
        if False:
            for i in range(10):
                print('nop')

        class FloorDivModule(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    return 10
                return (x // 3, x // 2.0, x // y)
        x = torch.arange(-2, 4).reshape(2, 3, 1)
        y = torch.randn(2, 3, 4)
        self.run_test(FloorDivModule(), (x, y))

    @unittest.skip('Floor division on ONNX is inconsistent with eager (see #78411)')
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_floordiv(self):
        if False:
            i = 10
            return i + 15

        class FloordivModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.new_zeros(x.size(2) // x.size(1))
        x = torch.randn(2, 3, 4)
        self.run_test(FloordivModule(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(FloordivModule(), (x,), remained_onnx_input_idx=[])

    def test_div(self):
        if False:
            for i in range(10):
                print('nop')

        class DivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (x / y, torch.true_divide(x, y))
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        self.run_test(DivModule(), (x, y))
        self.run_test(DivModule(), (x.float(), y.float()))

    def test_div_promotion_trace(self):
        if False:
            for i in range(10):
                print('nop')

        class DivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return (x / y, torch.true_divide(x, y))
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))
        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.trace(DivModule(), (x, y)), (x, y))

    def test_div_promotion_script(self):
        if False:
            i = 10
            return i + 15

        class DivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                x = x.transpose(1, 2)
                y = y.transpose(1, 2)
                return (x / y, torch.true_divide(x, y))
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        with common_utils.set_default_dtype(torch.float):
            self.run_test(torch.jit.script(DivModule()), (x, y))
        with common_utils.set_default_dtype(torch.double):
            self.run_test(torch.jit.script(DivModule()), (x, y))
        x = torch.randn(2, 3, 4).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.double)
        self.run_test(torch.jit.script(DivModule()), (x, y))

    @skipDtypeChecking
    def test_div_rounding_mode(self):
        if False:
            i = 10
            return i + 15

        class TrueDivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return (x.div(y, rounding_mode=None), torch.div(x, y, rounding_mode=None))

        class TruncDivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (x.div(y, rounding_mode='trunc'), torch.div(x, y, rounding_mode='trunc'))

        class FloorDivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return (x.div(y, rounding_mode='floor'), torch.div(x, y, rounding_mode='floor'))
        modules = [TrueDivModule(), TruncDivModule(), FloorDivModule()]
        x = (torch.randn(2, 3, 4) * 100).to(torch.int)
        y = torch.arange(1, 2 * 3 * 4 + 1).reshape(2, 3, 4).to(torch.int)
        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))
        x = torch.randn(2, 3, 4)
        y = torch.rand(2, 3, 4) * 10.0 + 0.1
        for module in modules:
            self.run_test(module, (x, y))
            self.run_test(torch.jit.trace(module, (x, y)), (x, y))
            self.run_test(torch.jit.script(module), (x, y))

    def test_slice_trace(self):
        if False:
            return 10

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x[0:1]
        x = torch.randn(3)
        self.run_test(MyModule(), x)

    def test_slice_neg(self):
        if False:
            while True:
                i = 10

        class NegSlice(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x[-1:]
        x = torch.randn(3, 4, 5)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large(self):
        if False:
            return 10

        class NegSlice(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x[:, :, -3:-1, :, -1]
        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    def test_slice_neg_large_negone(self):
        if False:
            while True:
                i = 10

        class NegSlice(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x[:, :, :, :, -1]
        x = torch.randn(3, 4, 5, 6, 7)
        self.run_test(NegSlice(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_index(self):
        if False:
            i = 10
            return i + 15

        class InputIndexSlice(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x[:y.size(0), 0, :] = y
                return x
        x = torch.zeros((56, 6, 256))
        y = torch.rand((22, 256))
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_slice_with_1d_input_index(self):
        if False:
            while True:
                i = 10

        class InputIndexSlice(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x[:y, 0, :] = y
                return x
        x = torch.zeros((56, 6, 256))
        y = torch.tensor([5], dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_slice_with_input_step_size(self):
        if False:
            while True:
                i = 10

        class InputIndexSlice(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    i = 10
                    return i + 15
                x[:y:z, 0::z, :] = 1
                return x
        x = torch.zeros((56, 6, 256))
        y = torch.tensor(5, dtype=torch.int64)
        z = torch.tensor(2, dtype=torch.int64)
        self.run_test(InputIndexSlice(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()
    def test_slice_dynamic(self):
        if False:
            i = 10
            return i + 15

        class DynamicSliceExportMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                results = []
                for i in range(4):
                    results.append(x[:x.size(0) - i, i:x.size(2), i:3])
                return tuple(results)
        x = torch.rand(5, 5, 5)
        y = torch.randn(6, 7, 8)
        self.run_test(DynamicSliceExportMod(), x, additional_test_inputs=[y], input_names=['input_1'], output_names=['output_1'], dynamic_axes={'input_1': [0, 1, 2], 'output_1': [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_script(self):
        if False:
            print('Hello World!')

        class DynamicSliceModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x[1:x.size(1)]
        x = torch.rand(1, 2)
        self.run_test(DynamicSliceModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_slice_dynamic_shape_script(self):
        if False:
            i = 10
            return i + 15

        class DynamicSliceModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.new_zeros(x.shape[1:x.size(2)])
        x = torch.rand(1, 2, 3, 4)
        self.run_test(DynamicSliceModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2, 3]})
        self.run_test(DynamicSliceModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()
    def test_slice_dynamic_to_end(self):
        if False:
            return 10

        class DynamicSliceExportMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                results = []
                for i in range(4):
                    results.append(x[:, i:, x.size(2) - 5])
                return tuple(results)
        x = torch.rand(5, 5, 5)
        self.run_test(DynamicSliceExportMod(), x, dynamic_axes={'input_1': [0, 1, 2], 'output_1': [0, 1, 2]})

    def test_square(self):
        if False:
            print('Hello World!')

        class Square(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.square(x)
        x = torch.randn(2, 3, 4)
        self.run_test(Square(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_dynamic(self):
        if False:
            while True:
                i = 10

        class ArangeModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return (torch.arange(input.shape[0]), torch.arange(12), torch.arange(start=input.shape[0], end=input.shape[0] + 5))
        x = torch.randn(5, 3, 2)
        y = torch.randn(8, 3, 2)
        self.run_test(ArangeModel(), x, additional_test_inputs=[y], input_names=['input_1'], output_names=['output_1', 'output_2', 'output_3'], dynamic_axes={'input_1': [0], 'output_1': [0]})
        self.run_test(torch.jit.script(ArangeModel()), x, additional_test_inputs=[y], input_names=['input_1'], output_names=['output_1', 'output_2', 'output_3'], dynamic_axes={'input_1': [0], 'output_1': [0]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_out(self):
        if False:
            i = 10
            return i + 15

        class ArangeOutModel(torch.nn.Module):

            def forward(self, end):
                if False:
                    for i in range(10):
                        print('nop')
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(end, out=out_t)
        x = torch.tensor(8)
        self.run_test(ArangeOutModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dynamic_arange_start_out(self):
        if False:
            i = 10
            return i + 15

        class ArangeStartOutModel(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    while True:
                        i = 10
                out_t = torch.tensor([1], dtype=torch.int64)
                return torch.arange(start.size(0), end, out=out_t)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8)
        self.run_test(ArangeStartOutModel(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace(self):
        if False:
            return 10

        class LinspaceModel(torch.nn.Module):

            def forward(self, start, end, steps):
                if False:
                    print('Hello World!')
                return torch.linspace(start, end, steps)
        x = torch.tensor(3, dtype=torch.float)
        y = torch.tensor(10, dtype=torch.float)
        z = torch.tensor(5, dtype=torch.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linspace_negative_start(self):
        if False:
            return 10

        class LinspaceModel(torch.nn.Module):

            def forward(self, start, end, steps):
                if False:
                    print('Hello World!')
                return torch.linspace(start, end, steps)
        x = torch.tensor(-1, dtype=torch.float)
        y = torch.tensor(1, dtype=torch.float)
        z = torch.tensor(6, dtype=torch.int)
        self.run_test(LinspaceModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_out(self):
        if False:
            i = 10
            return i + 15

        class ArangeModelEnd(torch.nn.Module):

            def forward(self, end):
                if False:
                    i = 10
                    return i + 15
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), y)

        class ArangeModelStep(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    return 10
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, 1.5, out=out_t)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelStep(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats(self):
        if False:
            for i in range(10):
                print('nop')

        class ArangeModelEnd(torch.nn.Module):

            def forward(self, end):
                if False:
                    return 10
                return torch.arange(end)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), y)

        class ArangeModelStep(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    return 10
                return torch.arange(start.size(0), end, 1.5)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelStep(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStepNeg(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.arange(end, start.size(0), -1.5)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelStepNeg(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModelStepNeg(), (x, y), remained_onnx_input_idx=[1])

        class ArangeModelStart(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    print('Hello World!')
                return torch.arange(start.size(0), end)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelStart(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModelStart(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_with_floats_override(self):
        if False:
            while True:
                i = 10

        class ArangeModelEnd(torch.nn.Module):

            def forward(self, end):
                if False:
                    i = 10
                    return i + 15
                return torch.arange(end, dtype=torch.int64)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelEnd(), y)

        class ArangeModelStep(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    i = 10
                    return i + 15
                return torch.arange(start.size(0), end, 1.5, dtype=torch.int64)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeModelStep(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModelStep(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_out(self):
        if False:
            return 10

        class ArangeOutModel(torch.nn.Module):

            def forward(self, end):
                if False:
                    i = 10
                    return i + 15
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(end, out=out_t)
        x = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeOutModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_out(self):
        if False:
            print('Hello World!')

        class ArangeStartOutModel(torch.nn.Module):

            def forward(self, start, end):
                if False:
                    print('Hello World!')
                out_t = torch.tensor([1], dtype=torch.float)
                return torch.arange(start.size(0), end, out=out_t)
        x = torch.randn(2, 3, 4)
        y = torch.tensor(8.5, dtype=torch.float)
        self.run_test(ArangeStartOutModel(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeStartOutModel(), (x, y), remained_onnx_input_idx=[1])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_no_type(self):
        if False:
            return 10

        class ArangeModel(torch.nn.Module):

            def forward(self, end):
                if False:
                    i = 10
                    return i + 15
                return (torch.arange(end), torch.arange(0, end))
        x = torch.tensor(6.2, dtype=torch.float)
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_size(self):
        if False:
            return 10

        class SizeModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return (torch.arange(input.size(0)), torch.arange(input.size(-1)), torch.ones(input.shape))
        x = torch.randn(5, 3, 2)
        self.run_test(SizeModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(SizeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_as_strided(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                chunk_size = list(x.size())
                chunk_size[1] = chunk_size[1] * 2 - 1
                chunk_stride = list(x.stride())
                chunk_stride[1] = chunk_stride[1] // 2
                return (x.as_strided((3, 3, 3), (1, 4, 2), storage_offset=2), x.as_strided(chunk_size, chunk_stride))
        x = torch.randn(5, 8, 7)
        self.run_test(Model(), x)

    @skipScriptTest()
    def test_tensor_index_advanced_indexing_ellipsis(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input[..., torch.tensor([2, 1]), torch.tensor([0, 3])]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing(self):
        if False:
            print('Hello World!')

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return input[:, torch.tensor([[0, 2], [1, 1]]), :, torch.tensor([2, 1]), torch.tensor([0, 3])]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input[:, torch.tensor([0, 2]), None, 2:4, torch.tensor([[1, 3], [4, 0]])]
        self.run_test(MyModel(), (m1,))

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input[:, torch.tensor([0, 2]), torch.tensor([1]), 2:4, torch.tensor([[1], [4]])]
        self.run_test(MyModel(), (m1,))

    def test_tensor_index_advanced_indexing_consecutive(self):
        if False:
            print('Hello World!')

        class MyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input[:, torch.tensor([0, 2]), torch.tensor([[1, 3], [4, 0]]), None]
        m1 = torch.randn(3, 4, 5, 6, 7)
        self.run_test(MyModel(), (m1,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put(self):
        if False:
            for i in range(10):
                print('nop')

        class IndexPutModel(torch.nn.Module):

            def forward(self, x, ind, update):
                if False:
                    return 10
                x[ind] = update
                return x
        x = torch.randn(3, 4)
        ind = torch.tensor([1], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_singular(self):
        if False:
            while True:
                i = 10

        class IndexPutBoolModel(torch.nn.Module):

            def forward(self, mask, indices):
                if False:
                    print('Hello World!')
                mask[indices] = True
                return mask
        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutBoolModel(), (mask, indices))

        class IndexPutFloatModel(torch.nn.Module):

            def forward(self, mask, indices):
                if False:
                    for i in range(10):
                        print('nop')
                mask[indices] = torch.tensor(5.5)
                return mask
        mask = torch.rand(100, dtype=torch.float)
        indices = (torch.rand(50) * mask.shape[0]).to(torch.int64)
        self.run_test(IndexPutFloatModel(), (mask, indices))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_accumulate(self):
        if False:
            while True:
                i = 10

        class IndexPutModel(torch.nn.Module):

            def forward(self, x, ind, update):
                if False:
                    return 10
                return x.index_put((ind,), update, accumulate=True)
        x = torch.randn(3, 4)
        ind = torch.tensor([2], dtype=torch.long)
        update = torch.ones(4)
        self.run_test(IndexPutModel(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_slice_index(self):
        if False:
            return 10

        class IndexPutModel(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    return 10
                x[1:2, 1:3, torch.tensor([1])] += update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(1, 2, 1)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    return 10
                x[torch.tensor([0, 2]), torch.tensor([1, 2])] += update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.randn(2, 5)
        self.run_test(IndexPutModel2(), (x, update))

        class IndexPutModel3(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    i = 10
                    return i + 15
                x[torch.tensor([0, 2]), 1:2] += update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1, 1)
        self.run_test(IndexPutModel3(), (x, update))

        class IndexPutModel4(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    while True:
                        i = 10
                x[torch.tensor([0, 2]), 2] += update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel4(), (x, update))

        class IndexPutModel5(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    for i in range(10):
                        print('nop')
                x[1:3, torch.tensor([0, 2]), 2] += update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.tensor([10, 15]).view(2, 1)
        self.run_test(IndexPutModel5(), (x, update))

        class IndexPutModel6(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    while True:
                        i = 10
                x[1:3, 0] = update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel6(), (x, update))

        class IndexPutModel7(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    for i in range(10):
                        print('nop')
                x[1:, 0] = update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.arange(2 * 5).to(torch.float).view(2, 5)
        self.run_test(IndexPutModel7(), (x, update))

        class IndexPutModel8(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    while True:
                        i = 10
                x[:3, 0] = update
                return x
        x = torch.randn(3, 4, 5)
        update = torch.arange(3 * 5).to(torch.float).view(3, 5)
        self.run_test(IndexPutModel8(), (x, update))

        class IndexPutModel9(torch.nn.Module):

            def forward(self, poses):
                if False:
                    for i in range(10):
                        print('nop')
                w = 32
                x = poses[:, :, 0] - (w - 1) // 2
                boxes = torch.zeros([poses.shape[0], 17, 4])
                boxes[:, :, 0] = x
                return boxes
        x = torch.zeros([2, 17, 3], dtype=torch.int64)
        self.run_test(IndexPutModel9(), (x,))

        class IndexPutModel10(torch.nn.Module):

            def forward(self, x, ind, update):
                if False:
                    while True:
                        i = 10
                x[ind, 1:3] = update.view(1, 1, 1, 5).expand(2, 2, 2, 5)
                return x
        x = torch.randn(3, 4, 5)
        ind = torch.tensor([[0, 2], [1, 1]])
        update = torch.randn(5)
        self.run_test(IndexPutModel10(), (x, ind, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_index_put_ellipsis(self):
        if False:
            while True:
                i = 10

        class IndexPutModel(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    for i in range(10):
                        print('nop')
                x[..., torch.tensor([2, 1, 3]), 2:4] += update
                return x
        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(3, 1, 1, 3, 2)
        self.run_test(IndexPutModel(), (x, update))

        class IndexPutModel2(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    print('Hello World!')
                x[2, ..., torch.tensor([2, 1, 3]), 2:4] += update
                return x
        x = torch.randn(3, 4, 5, 6, 7)
        update = torch.randn(4, 1, 3, 2)
        self.run_test(IndexPutModel2(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_loop(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def ngram_attention_bias(sequence_length: int, ngram: int, device: torch.device, dtype: torch.dtype):
            if False:
                return 10
            bias = torch.ones((ngram, sequence_length), device=device, dtype=dtype) * float('-inf')
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias = bias * 2
                    bias[stream_idx, i] = 5
                    bias = bias * 5
                    bias[0, 0] = 5
            for stream_idx in range(ngram):
                for i in range(sequence_length):
                    bias[stream_idx, i] = 5
                    bias[0, i] = 5
            return bias

        class ScriptModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.ngram = 2
                self.max_target_positions = 512

            def forward(self, hidden_states):
                if False:
                    i = 10
                    return i + 15
                (seq_length, batch_size) = hidden_states.shape[:2]
                predict_causal_mask = ngram_attention_bias(self.max_target_positions, self.ngram, hidden_states.device, hidden_states.dtype)
                predict_causal_mask = predict_causal_mask[:, :seq_length]
                return predict_causal_mask
        x = torch.randn(6, 2)
        y = torch.randn(4, 1)
        self.run_test(ScriptModel(), x, input_names=['x'], dynamic_axes={'x': {0: 'seq_length', 1: 'batch_size'}}, additional_test_inputs=[y])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_(self):
        if False:
            print('Hello World!')

        class CopyModel(torch.nn.Module):

            def forward(self, x, data):
                if False:
                    while True:
                        i = 10
                x[1:3] = data
                return x
        x = torch.randn(3, 4)
        update = torch.randn(2, 4)
        self.run_test(CopyModel(), (x, update))

        class CopyModel2(torch.nn.Module):

            def forward(self, x, data):
                if False:
                    i = 10
                    return i + 15
                x[1:3, 0] = data
                return x
        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))
        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel2(), (x, update))
        update = torch.randn(2)
        self.run_test(CopyModel2(), (x, update))

        class CopyModel3(torch.nn.Module):

            def forward(self, x, data):
                if False:
                    return 10
                x[1, 1:3] = data
                return x
        x = torch.randn(3, 4)
        update = torch.tensor([0], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))
        update = torch.tensor([2, 3], dtype=torch.float32)
        self.run_test(CopyModel3(), (x, update))
        update = torch.randn(2)
        self.run_test(CopyModel3(), (x, update))

        class CopyModel4(torch.nn.Module):

            def forward(self, x, ind, data):
                if False:
                    while True:
                        i = 10
                x[ind] = data
                return x
        x = torch.randn(3, 4)
        ind = torch.tensor(2)
        data = torch.randn(4)
        self.run_test(CopyModel4(), (x, ind, data))

        class CopyModel5(torch.nn.Module):

            def forward(self, x, mask):
                if False:
                    i = 10
                    return i + 15
                if mask is not None:
                    x.copy_(mask)
                    return x
        x = torch.randn(3, 4)
        mask = torch.randn(3, 1)
        self.run_test(CopyModel5(), (x, mask))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_copy_tracing(self):
        if False:
            print('Hello World!')

        class CopyModel(torch.nn.Module):

            def forward(self, x, data):
                if False:
                    return 10
                x[1, 1:3] = data
                return x
        x = torch.randn(3, 4)
        update = torch.randn(1, 2)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis(self):
        if False:
            print('Hello World!')

        class CopyModel(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    for i in range(10):
                        print('nop')
                x[..., 1] = update
                return x
        x = torch.randn(2, 3, 4)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))
        x = torch.randn(2, 3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_copy_ellipsis_script(self):
        if False:
            i = 10
            return i + 15

        class CopyModel(torch.nn.Module):

            def forward(self, x, update):
                if False:
                    for i in range(10):
                        print('nop')
                x = x.reshape(4, 3, 5, 6)
                x[2, ..., 1:3] = update
                return x
        x = torch.randn(3, 4, 5, 6)
        update = torch.ones(1)
        self.run_test(CopyModel(), (x, update))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_flip(self):
        if False:
            print('Hello World!')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.flip(x, dims=[0])
        x = torch.tensor(np.arange(6.0).reshape(2, 3))
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint(self):
        if False:
            return 10

        class RandInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                randint = torch.randint(1, 10, x.shape)
                x = 0 * randint + x
                return x
        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_value(self):
        if False:
            print('Hello World!')

        class RandInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.randint(3, 4, x.shape) + x
        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randint_like(self):
        if False:
            for i in range(10):
                print('nop')

        class RandInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.randint_like(x, 3, 4) + x
        x = torch.randn(2, 3, 4)
        self.run_test(RandInt(), x)

    def test_randn(self):
        if False:
            return 10

        class RandN(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.mul(x, (torch.randn(2, 3, 4) + x).size(0))
        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand(self):
        if False:
            while True:
                i = 10

        class Rand(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.mul(x, (torch.rand(2, 3, 4) + x).size(0))
        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_dtype(self):
        if False:
            while True:
                i = 10

        class RandN(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.to(torch.float32) * torch.randn(2, 3, 4, dtype=torch.double) * torch.tensor(0, dtype=torch.float32)
        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    def test_rand_dtype(self):
        if False:
            while True:
                i = 10

        class Rand(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x.to(torch.float32) * torch.rand(2, 3, 4, dtype=torch.double) * torch.tensor(0, dtype=torch.float32)
        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_randn_dynamic_size(self):
        if False:
            for i in range(10):
                print('nop')

        class RandN(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.mul(x, torch.randn(x.size()).size(1))
        x = torch.randn(2, 3, 4)
        self.run_test(RandN(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_rand_dynamic_size(self):
        if False:
            while True:
                i = 10

        class Rand(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mul(x, torch.rand(x.size()).size(1))
        x = torch.randn(2, 3, 4)
        self.run_test(Rand(), x)

    def test_randn_like(self):
        if False:
            while True:
                i = 10

        class RandNLike(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mul(x, torch.randn_like(x).size(0))
        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)
        self.run_test(torch.jit.script(RandNLike()), x)

    def test_rand_like(self):
        if False:
            for i in range(10):
                print('nop')

        class RandLike(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.mul(x, torch.rand_like(x).size(0))
        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)
        self.run_test(torch.jit.script(RandLike()), x)

    def test_randn_like_dtype(self):
        if False:
            for i in range(10):
                print('nop')

        class RandNLike(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.to(torch.float32) * torch.randn_like(x, dtype=torch.double) * torch.tensor(0, dtype=torch.float32)
        x = torch.randn(2, 3, 4)
        self.run_test(RandNLike(), x)

    def test_rand_like_dtype(self):
        if False:
            i = 10
            return i + 15

        class RandLike(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.to(torch.float32) * torch.rand_like(x, dtype=torch.double) * torch.tensor(0, dtype=torch.float32)
        x = torch.randn(2, 3, 4)
        self.run_test(RandLike(), x)

    def test_bernoulli(self):
        if False:
            return 10

        class Bernoulli(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.mul(x, torch.bernoulli(x).size(0))
        x = torch.empty(3, 3).uniform_(0, 1)
        self.run_test(Bernoulli(), x)
        x = torch.empty(2, 3, 3, dtype=torch.double).uniform_(0, 1)
        self.run_test(Bernoulli(), x)

    def test_bernoulli_p(self):
        if False:
            for i in range(10):
                print('nop')

        class Bernoulli_float(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.mul(x, torch.bernoulli(x, 0.2).size(0))

        class Bernoulli_tensor(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.mul(x, torch.rand_like(x).bernoulli_(x).size(0))
        x = torch.rand(3, 3)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)
        x = torch.rand(2, 3, 3, dtype=torch.double)
        self.run_test(Bernoulli_float(), x)
        self.run_test(Bernoulli_tensor(), x)

    @unittest.skip('Bug in ORT, skip test until rel-1.11.')
    @skipIfUnsupportedMinOpsetVersion(14)
    def test_reshape_allowzero(self):
        if False:
            return 10

        class ReshapeModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = x.reshape(3, 4, 0)
                return x
        x = torch.randn(0, 3, 4)
        self.run_test(ReshapeModel(), x)

    def test_reshape_different_rank(self):
        if False:
            for i in range(10):
                print('nop')

        class ReshapeModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = x.reshape(-1, 2, 4, 4, 5, 5)
                return x
        x = torch.randn(1, 32, 5, 5)
        self.run_test(ReshapeModel(), x)

    def _interpolate(self, x, mode, use_size, is_upsample, align_corners=False):
        if False:
            return 10

        class MyModel(torch.nn.Module):
            __constants__ = ['mode', 'use_size', 'is_upsample', 'size', 'scale', 'size_array', 'scale_array', 'align_corners']

            def __init__(self, mode, use_size, is_upsample, align_corners):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mode = mode
                self.use_size = use_size
                self.is_upsample = is_upsample
                self.align_corners = align_corners
                self.scale = 2.0 if self.is_upsample else 0.5
                self.size = 24 if self.is_upsample else 2
                if x.dim() == 3:
                    self.scale_array = [2.3]
                    self.size_array = [16]
                elif x.dim() == 4:
                    self.scale_array = [2.3, 3.1]
                    self.size_array = [16, 32]
                else:
                    self.scale_array = [2.3, 3.1, 4.6]
                    self.size_array = [16, 32, 64]

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if self.use_size:
                    if self.align_corners:
                        return (torch.nn.functional.interpolate(x, mode=self.mode, size=self.size, align_corners=True), torch.nn.functional.interpolate(x, mode=self.mode, size=self.size_array, align_corners=True))
                    return (torch.nn.functional.interpolate(x, mode=self.mode, size=self.size), torch.nn.functional.interpolate(x, mode=self.mode, size=self.size_array))
                if self.align_corners:
                    return (torch.nn.functional.interpolate(x, mode=self.mode, scale_factor=self.scale, recompute_scale_factor=False), torch.nn.functional.interpolate(x, mode=self.mode, scale_factor=self.scale_array, recompute_scale_factor=False))
                return (torch.nn.functional.interpolate(x, mode=self.mode, scale_factor=self.scale, recompute_scale_factor=False), torch.nn.functional.interpolate(x, mode=self.mode, scale_factor=self.scale_array, recompute_scale_factor=False))
        model = MyModel(mode, use_size, is_upsample, align_corners)
        self.run_test(model, x, atol=1e-06)

    def _interpolate_tests(self, is_upsample):
        if False:
            print('Hello World!')
        modes = ['nearest', 'linear', 'bicubic']
        if self.opset_version < 11:
            modes = ['nearest']
        x = [torch.randn(1, 2, 6, requires_grad=True), torch.randn(1, 2, 4, 6, requires_grad=True), torch.randn(1, 2, 4, 4, 6, requires_grad=True)]
        for mode in modes:
            for xi in x:
                mode_i = mode
                if mode == 'bicubic' and xi.dim() != 4:
                    continue
                elif mode == 'linear':
                    if xi.dim() == 3:
                        continue
                    elif xi.dim() == 4:
                        mode_i = 'bilinear'
                    elif xi.dim() == 5:
                        mode_i = 'trilinear'
                        continue
                self._interpolate(xi, mode_i, True, is_upsample)
                if mode != 'nearest':
                    self._interpolate(xi, mode_i, True, is_upsample, True)
                if self.opset_version >= 9:
                    self._interpolate(xi, mode_i, True, is_upsample)
                    if mode != 'nearest':
                        self._interpolate(xi, mode_i, False, is_upsample, True)
                    self._interpolate(xi, mode_i, False, is_upsample)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_upsample(self):
        if False:
            for i in range(10):
                print('nop')
        self._interpolate_tests(True)

    @skipIfUnsupportedMaxOpsetVersion(8)
    @skipScriptTest()
    def test_interpolate_upsample_trace(self):
        if False:
            i = 10
            return i + 15
        self._interpolate_tests(True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_interpolate_function_substitution(self):
        if False:
            print('Hello World!')

        class ScriptModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2.0)

        class ScriptModule(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.submodule = ScriptModel()

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return self.submodule(input)
        x = torch.randn(1, 2, 4, 4, 6)
        self.run_test(ScriptModule(), (x,))

        @torch.jit.script
        def script_method(x):
            if False:
                print('Hello World!')
            return torch.nn.functional.interpolate(x, mode='nearest', scale_factor=2.0)

        class TracingModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return script_method(x)
        self.run_test(TracingModule(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_interpolate_downsample(self):
        if False:
            print('Hello World!')
        self._interpolate_tests(False)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_half_pixel(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(torch.nn.Module):

            def __init__(self, mode, size):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.mode = mode
                self.size = size

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.nn.functional.interpolate(x, mode=self.mode, size=self.size)
        modes = ['linear', 'bicubic']
        x = [torch.randn(1, 2, 6, requires_grad=True), torch.randn(1, 2, 4, 6, requires_grad=True), torch.randn(1, 2, 4, 4, 6, requires_grad=True)]
        for mode in modes:
            for xi in x:
                mode_i = mode
                if mode == 'bicubic' and xi.dim() != 4:
                    continue
                elif mode == 'linear':
                    if xi.dim() == 4:
                        mode_i = 'bilinear'
                    elif xi.dim() == 5:
                        mode_i = 'trilinear'
                for i in range(xi.dim() - 2):
                    size = list(xi.shape[2:])
                    size[i] = 1
                    self.run_test(MyModel(mode_i, size), xi)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_interpolate_no_shape(self):
        if False:
            while True:
                i = 10

        class MyModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = torch.add(x, x)
                out1 = torch.nn.functional.interpolate(x, mode='bilinear', size=(16, 16), align_corners=False)
                out2 = torch.nn.functional.interpolate(x, mode='nearest', size=(int(y.size(0)), int(y.size(1))))
                return (out1, out2)
        x = torch.randn(1, 2, 4, 4, requires_grad=True)
        y = torch.randn(16, 16, requires_grad=True)
        self.run_test(MyModel(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2, 3], 'y': [0, 1]})
        self.run_test(MyModel(), (x, y), remained_onnx_input_idx=[0])

    @skipScriptTest()
    def test_interpolate_adaptive_pooling_error(self):
        if False:
            while True:
                i = 10
        x = torch.randn(1, 2, 6, requires_grad=True)
        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, 'area', True, True)
        with self.assertRaises(RuntimeError) as cm:
            self._interpolate(x, 'area', False, True)

    def test_groupnorm(self):
        if False:
            return 10
        model = torch.nn.GroupNorm(3, 6, 0.002)
        x = torch.randn(4, 6, 36, 36, 18)
        self.run_test(model, x)
        model = torch.nn.GroupNorm(1, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)
        model = torch.nn.GroupNorm(6, 6, 0.002)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    def test_groupnorm_noaffine(self):
        if False:
            return 10
        model = torch.nn.GroupNorm(4, 8, 0.002, affine=False)
        x = torch.randn(3, 8, 224, 224)
        self.run_test(model, x)
        model = torch.nn.GroupNorm(1, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)
        model = torch.nn.GroupNorm(6, 6, 0.002, affine=False)
        x = torch.randn(4, 6, 180, 180)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_scripted(self):
        if False:
            print('Hello World!')

        class ListUnpack(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                (a, b) = x.shape
                return x.new_zeros((a, b))
        x = torch.randn(2, 3)
        self.run_test(torch.jit.script(ListUnpack()), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(torch.jit.script(ListUnpack()), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_scripted_runs_without_error_with_constructed_list_as_input(self):
        if False:
            print('Hello World!')

        class PackUnpack(torch.nn.Module):
            """Create and unpack a list of tensors.

            When scripted, it should produce a graph similar to

            ```
            graph(%self : __torch__.PackUnpack,
                %a.1 : Tensor,
                %b.1 : Tensor):
            %packed.1 : Tensor[] = prim::ListConstruct(%a.1, %b.1)
            %c.1 : Tensor, %8 : Tensor = prim::ListUnpack(%packed.1)
            return (%c.1)
            ```
            """

            def forward(self, a, b):
                if False:
                    i = 10
                    return i + 15
                packed = [a, b]
                (c, _) = packed
                return c
        self.run_test(torch.jit.script(PackUnpack()), (torch.tensor(0), torch.tensor([42])), remained_onnx_input_idx=[0])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_unpack_slice_scripted(self):
        if False:
            return 10

        class ListUnpackSlice(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                (a, b) = x.shape[2:]
                return x.new_zeros((a, b))
        x = torch.randn(2, 3, 4, 5)
        self.run_test(torch.jit.script(ListUnpackSlice()), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2, 3]})
        self.run_test(torch.jit.script(ListUnpackSlice()), x, remained_onnx_input_idx=[])

    @skipDtypeChecking
    def test_pow(self):
        if False:
            while True:
                i = 10

        class PowModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x.pow(y)
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(PowModule(), (x, y))
        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4)).to(dtype=torch.int32)
        self.run_test(PowModule(), (x, y))
        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))
        x = torch.randn(2, 3, 4).to(dtype=torch.float64)
        y = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule(), (x, y))

        class PowModule2(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.pow(2, x)
        x = torch.randn(1, 10)
        self.run_test(PowModule2(), (x,))
        x = torch.randint(10, (2, 3, 4))
        self.run_test(PowModule2(), (x,))
        x = torch.randn(1, 10).to(dtype=torch.float64)
        self.run_test(PowModule2(), (x,))

        class PowModule3(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return y[torch.pow(2, x)]
        x = torch.randint(5, (2, 3, 4))
        y = torch.rand(100)
        self.run_test(PowModule3(), (x, y))

    @skipIfUnsupportedMaxOpsetVersion(13)
    @skipDtypeChecking
    def test_arithmeticOps_with_low_precision(self):
        if False:
            for i in range(10):
                print('nop')

        class AddModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x + y

        class SubModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x - y

        class MulModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return x * y

        class DivModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x / y

        class PowModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x.pow(y)
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        z = torch.tensor([1], dtype=torch.uint8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))
        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        z = torch.tensor([1], dtype=torch.int8)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))
        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        z = torch.tensor([1], dtype=torch.int16)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float32)
        z = torch.tensor([1], dtype=torch.float64)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int64)
        z = torch.tensor([1], dtype=torch.int32)
        self.run_test(AddModule(), (x, y))
        self.run_test(SubModule(), (x, y))
        self.run_test(MulModule(), (x, y))
        self.run_test(DivModule(), (x, y))
        self.run_test(PowModule(), (x, z))

    def test_mul_bool(self):
        if False:
            return 10

        class MyModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.mul(x, y)
        x_t = torch.tensor([True, False, True, False])
        y_t = torch.tensor([True, True, False, False])
        z_t = torch.tensor([1.0, 2.0, 3.0, 0.0])
        self.run_test(MyModel(), (x_t, y_t))
        self.run_test(MyModel(), (x_t, z_t))
        self.run_test(MyModel(), (z_t, y_t))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfUnsupportedMaxOpsetVersion(13)
    def test_mod_with_low_precision(self):
        if False:
            print('Hello World!')

        class ModModule(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.fmod(x, y).to(dtype=torch.long)
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.uint8)
        self.run_test(ModModule(), (x, y))
        x = torch.tensor([2, 3, 5], dtype=torch.int8)
        y = torch.tensor([2, 3, 5], dtype=torch.int8)
        self.run_test(ModModule(), (x, y))
        x = torch.tensor([2, 3, 5], dtype=torch.int16)
        y = torch.tensor([2, 3, 5], dtype=torch.int16)
        self.run_test(ModModule(), (x, y))
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.int32)
        self.run_test(ModModule(), (x, y))
        x = torch.tensor([2, 3, 5], dtype=torch.uint8)
        y = torch.tensor([2, 3, 5], dtype=torch.float64)
        self.run_test(ModModule(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_empty_constant_shape(self):
        if False:
            while True:
                i = 10

        class Zeros(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = torch.zeros(())
                y += x
                return y
        x = torch.tensor(42.0)
        self.run_test(Zeros(), x)

        class Ones(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = torch.ones(())
                y += x
                return y
        x = torch.tensor(42.0)
        self.run_test(Ones(), x)

        class Full(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = torch.full((), 1.0)
                y += x
                return y
        x = torch.tensor(42.0)
        self.run_test(Full(), x)

        class Empty(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = torch.empty(()).fill_(0)
                y += x
                return y
        x = torch.tensor(42.0)
        self.run_test(Empty(), x)

    def test_std(self):
        if False:
            i = 10
            return i + 15

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.std(input, unbiased=False)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.std(input, unbiased=True)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_along_dims(self):
        if False:
            while True:
                i = 10

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.std(input, dim=(0, 1), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.std(input, dim=(0, 1), unbiased=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_keepdim(self):
        if False:
            for i in range(10):
                print('nop')

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.std(input, dim=(0, 1), unbiased=False, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.std(input, dim=(0, 1), unbiased=True, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_correction(self):
        if False:
            while True:
                i = 10

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.std(input, dim=(0, 1), correction=3, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_var(self):
        if False:
            for i in range(10):
                print('nop')

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.var(input, unbiased=False)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.var(input, unbiased=True)
        model = VarianceUnbiased()
        self.run_test(model, x)

        class VarianceSqrt(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                y = torch.var(input, 1)
                return torch.sqrt(y + 1e-08)
        x = torch.randn(1, 2, 3, 300, 300)
        model = VarianceSqrt()
        self.run_test(model, x)

    def test_var_along_dims(self):
        if False:
            while True:
                i = 10

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.var(input, dim=(0, 1), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.var(input, dim=(0, 1), unbiased=True)
        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_keepdim(self):
        if False:
            while True:
                i = 10

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.var(input, dim=(0, 1), unbiased=False, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.var(input, dim=(0, 1), unbiased=True, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_correction(self):
        if False:
            for i in range(10):
                print('nop')

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.var(input, dim=(0, 1), correction=3, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_var_mean(self):
        if False:
            while True:
                i = 10

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.var_mean(input, unbiased=False)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.var_mean(input, unbiased=True)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_along_dims(self):
        if False:
            while True:
                i = 10

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.var_mean(input, dim=(0, 1), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.var_mean(input, dim=(0, 1), unbiased=True)
        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_mixed_dims(self):
        if False:
            i = 10
            return i + 15

        class ReverseDims(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.var_mean(input, dim=(2, 1), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = ReverseDims()
        self.run_test(model, x)

        class SkipDims(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.var_mean(input, dim=(0, 2), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = SkipDims()
        self.run_test(model, x)

        class NonZeroDims(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.var_mean(input, dim=(1, 2), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = NonZeroDims()
        self.run_test(model, x)

    def test_var_mean_keepdim(self):
        if False:
            while True:
                i = 10

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.var_mean(input, dim=(0, 1), unbiased=False, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.var_mean(input, dim=(0, 1), unbiased=True, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_var_mean_correction(self):
        if False:
            print('Hello World!')

        class Variance(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = Variance()
        self.run_test(model, x)

    def test_std_mean(self):
        if False:
            for i in range(10):
                print('nop')

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.std_mean(input, unbiased=False)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.std_mean(input, unbiased=True)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_along_dims(self):
        if False:
            i = 10
            return i + 15

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.std_mean(input, dim=(0, 1), unbiased=False)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class VarianceUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.std_mean(input, dim=(0, 1), unbiased=True)
        x = torch.randn(2, 3, 4)
        model = VarianceUnbiased()
        self.run_test(model, x)

    def test_std_mean_keepdim(self):
        if False:
            print('Hello World!')

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.std_mean(input, dim=(0, 1), unbiased=False, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

        class StandardDeviationUnbiased(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.std_mean(input, dim=(0, 1), unbiased=True, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviationUnbiased()
        self.run_test(model, x)

    def test_std_mean_correction(self):
        if False:
            i = 10
            return i + 15

        class StandardDeviation(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.var_mean(input, dim=(0, 1), correction=3, keepdim=True)
        x = torch.randn(2, 3, 4)
        model = StandardDeviation()
        self.run_test(model, x)

    def test_bitshift(self):
        if False:
            print('Hello World!')

        class BitshiftModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return (input >> 1, input << 3, input >> torch.tensor([1, 2]), input << 4)
        input = torch.arange(24, dtype=torch.int64).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_bitshift_uint8(self):
        if False:
            while True:
                i = 10

        class BitshiftModel(torch.nn.Module):

            def forward(self, input, input2):
                if False:
                    print('Hello World!')
                return (input >> 1, input << 3, input2 >> torch.tensor([1, 2], dtype=torch.uint8), input2 << 4)
        input = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        input2 = torch.arange(24, dtype=torch.uint8).reshape(3, 4, 2)
        self.run_test(BitshiftModel(), (input, input2))

    def test_narrow(self):
        if False:
            i = 10
            return i + 15

        class NarrowModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.narrow(input, 0, 0, 2)
        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_narrow_dynamic(self):
        if False:
            return 10

        class NarrowModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.narrow(input, 0, 0, input.shape[0] - 1)
        x = torch.randn(3, 3, requires_grad=True)
        self.run_test(NarrowModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_fill(self):
        if False:
            i = 10
            return i + 15

        class IndexFillModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                index = torch.tensor([2, 0])
                return input.index_fill(2, index, -1)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexFillModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_copy(self):
        if False:
            return 10

        class IndexCopyModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                index = torch.tensor([2, 0])
                source = torch.ones(3, 2, 5)
                return input.index_copy(1, index, source)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(IndexCopyModel(), x)

    def test_select(self):
        if False:
            for i in range(10):
                print('nop')

        class Select(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x[:, 1]
        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_select_negative_index(self):
        if False:
            while True:
                i = 10

        class Select(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x[:, -1]
        x = torch.randn(3, 4)
        self.run_test(Select(), x)

    def test_index_select_constant_scaler_index(self):
        if False:
            while True:
                i = 10

        class IndexSelectScalerIndexModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                index = 2
                return torch.index_select(x, 1, torch.tensor(index))
        x = torch.randn(3, 4)
        self.run_test(IndexSelectScalerIndexModel(), x)

    def test_index_select_scaler_index(self):
        if False:
            i = 10
            return i + 15

        class IndexSelectScalerIndexModel(torch.nn.Module):

            def __init__(self, index_base):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.index_base = torch.tensor(index_base)

            def forward(self, x, index_offset):
                if False:
                    while True:
                        i = 10
                index = self.index_base + index_offset
                return torch.index_select(x, 1, index)
        x = torch.randn(3, 4)
        offset = 2
        index_offset = torch.tensor(offset)
        base = 1
        self.run_test(IndexSelectScalerIndexModel(base), (x, index_offset))

    def test_take(self):
        if False:
            print('Hello World!')

        class TakeModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.take(x, y)
        x = torch.randn(6, 4, 3, 3)
        y = torch.tensor([4, 1, 7, 15, 63])
        self.run_test(TakeModel(), (x, y))

    def test_topk(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.topk(x, 3)
        x = torch.arange(1.0, 6.0, requires_grad=True)
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_int32_k(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x, k):
                if False:
                    while True:
                        i = 10
                return torch.topk(x, k)
        x = torch.arange(1.0, 6.0)
        k = torch.tensor(3, dtype=torch.int32)
        self.run_test(Model(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_topk_smallest_unsorted(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def forward(self, x, k):
                if False:
                    i = 10
                    return i + 15
                topk_unsorted = torch.topk(x, k, largest=False, sorted=False)
                topk_sorted = torch.topk(x, k, largest=False, sorted=True)
                return (topk_sorted, torch.sort(topk_unsorted.values).values)
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModule(), (x, k))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_topk_script(self):
        if False:
            return 10

        class MyModuleDynamic(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, k):
                if False:
                    print('Hello World!')
                return torch.topk(x, k)
        x = torch.arange(1.0, 6.0, requires_grad=True)
        k = torch.tensor(3)
        self.run_test(MyModuleDynamic(), (x, k))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_auto_grad(self):
        if False:
            return 10

        class MyClip(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input, scalar):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.save_for_backward(input)
                return input.clamp(min=scalar)

        class MyRelu(torch.autograd.Function):

            @staticmethod
            def forward(ctx, input):
                if False:
                    for i in range(10):
                        print('nop')
                ctx.save_for_backward(input)
                return input.clamp(min=0)

        def symbolic_python_op(ctx: torch.onnx.SymbolicContext, g: torch._C.Graph, *args, **kwargs):
            if False:
                i = 10
                return i + 15
            n = ctx.cur_node
            name = kwargs['name']
            if name == 'MyClip':
                return g.op('Clip', args[0], args[1], outputs=n.outputsSize())
            elif name == 'MyRelu':
                return g.op('Relu', args[0], outputs=n.outputsSize())
            else:
                return torch.onnx.symbolic_helper._unimplemented('prim::PythonOp', 'unknown node kind: ' + name)
        torch.onnx.register_custom_op_symbolic('prim::PythonOp', symbolic_python_op, 1)
        self.addCleanup(torch.onnx.unregister_custom_op_symbolic, 'prim::PythonOp', 1)

        class MyClipModule(torch.nn.Module):

            def forward(self, x, min):
                if False:
                    print('Hello World!')
                return MyClip.apply(x, min)
        x = torch.randn(3, 3)
        min = torch.tensor([0.0])
        self.run_test(MyClipModule(), (x, min))

        class MyReluModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return MyRelu.apply(x)
        x = torch.randn(3, 3)
        self.run_test(MyReluModule(), x)

    def test_clip_int(self):
        if False:
            for i in range(10):
                print('nop')

        class MyClipInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.clamp(x, 0, 1)
        self.run_test(MyClipInt(), torch.randn(3, 3).to(torch.int64))

    def test_relu_int(self):
        if False:
            for i in range(10):
                print('nop')
        self.run_test(torch.nn.ReLU(), torch.randn(3, 3).to(torch.int32))

    def test_pad_int(self):
        if False:
            print('Hello World!')

        class MyPadInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.pad(x, (1, 1))
        self.run_test(MyPadInt(), torch.randn(3, 3).to(torch.int32))

    def test_min_int(self):
        if False:
            print('Hello World!')

        class MyMinInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.min(x, x + 1)
        self.run_test(MyMinInt(), torch.randn(3, 3).to(torch.int32))

    def test_max_int(self):
        if False:
            return 10

        class MyMaxnInt(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.max(x, x + 1)
        self.run_test(MyMaxnInt(), torch.randn(3, 3).to(torch.int32))

    @skipIfUnsupportedOpsetVersion([7])
    def test_normalize(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.normalize(x)
        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    def test_norm_with_dtype(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.ops.aten.norm(x, p=2, dim=[1], keepdim=True, dtype=torch.float64)
        x = torch.randn(3, 3)
        self.run_test(Model(), x)

    def test_layer_norm(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.LayerNorm([10, 10, 10])
        x = torch.randn(20, 5, 10, 10, 10)
        self.run_test(model, x)

    def test_batchnorm1d(self):
        if False:
            while True:
                i = 10
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=True)
        self.run_test(model, x)
        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_noaffine(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, affine=False)
        self.run_test(model, x)
        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm1d_norunningstats(self):
        if False:
            return 10
        x = torch.randn(10, 10)
        model = torch.nn.BatchNorm1d(10, track_running_stats=False)
        self.run_test(model, x)
        x = torch.randn(10, 10, 128)
        self.run_test(model, x)

    def test_batchnorm2d(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm2d_noaffine(self):
        if False:
            return 10
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, affine=False)
        self.run_test(model, x)

    def test_batchnorm2d_norunningstats(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.BatchNorm2d(3, track_running_stats=False)
        self.run_test(model, x)

    def test_batchnorm3d(self):
        if False:
            return 10
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.BatchNorm3d(3, affine=True)
        self.run_test(model, x)

    def test_batchnorm3d_noaffine(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.BatchNorm3d(3, affine=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_instancenorm1d_runningstats(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(10, 5, 128)
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=True)
        self.run_test(model, x)
        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm1d_norunningstats(self):
        if False:
            while True:
                i = 10
        x = torch.randn(10, 5, 128)
        model = torch.nn.InstanceNorm1d(5, affine=True, track_running_stats=False)
        self.run_test(model, x)
        model = torch.nn.InstanceNorm1d(5, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_instancenorm2d_runningstats(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)
        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm2d_norunningstats(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 3, 128, 128)
        model = torch.nn.InstanceNorm2d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)
        model = torch.nn.InstanceNorm2d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_instancenorm3d_runningstats(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=True)
        self.run_test(model, x)
        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=True)
        self.run_test(model, x)

    def test_instancenorm3d_norunningstats(self):
        if False:
            print('Hello World!')
        x = torch.randn(10, 3, 64, 64, 64)
        model = torch.nn.InstanceNorm3d(3, affine=True, track_running_stats=False)
        self.run_test(model, x)
        model = torch.nn.InstanceNorm3d(3, affine=False, track_running_stats=False)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar(self):
        if False:
            while True:
                i = 10

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices):
                if False:
                    return 10
                values = 1.0
                return input.scatter(1, indices, values)
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float64)
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_with_scalar_different_types(self):
        if False:
            print('Hello World!')

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices):
                if False:
                    return 10
                values = 1.0
                return input.scatter(1, indices, values)
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=torch.float32)
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), input_args=(input, indices))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter(self):
        if False:
            return 10

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices, values):
                if False:
                    while True:
                        i = 10
                return input.scatter(1, indices, values)
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), (input, indices, values))
        input = torch.zeros(3, 4, 5, 6)
        indices = torch.tensor([[1, 0], [0, 2], [0, 1]], dtype=torch.int64)
        indices = indices.view(3, 2, 1, 1).expand(3, 2, 5, 6)
        values = torch.arange(3 * 2 * 5 * 6, dtype=torch.float32).view(3, 2, 5, 6)
        self.run_test(ScatterModel(), (input, indices, values))
        input = torch.zeros(3, 4, 2)
        indices = torch.tensor([[[1, 0], [0, 2]], [[1, 1], [0, 1]], [[2, 1], [2, 2]]])
        values = torch.arange(3 * 2 * 2, dtype=torch.float32).view(3, 2, 2)
        self.run_test(ScatterModel(), (input, indices, values))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scatter_add(self):
        if False:
            i = 10
            return i + 15

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices, values):
                if False:
                    i = 10
                    return i + 15
                return input.scatter_add(1, indices, values)
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        @torch.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            if False:
                for i in range(10):
                    print('nop')
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):

            def forward(self, src, index):
                if False:
                    print('Hello World!')
                return scatter_sum(src, index)
        src = torch.rand(3, 2)
        index = torch.tensor([[0, 1], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_index_not_unique(self):
        if False:
            i = 10
            return i + 15

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices, values):
                if False:
                    print('Hello World!')
                return input.scatter_add(1, indices, values)
        input = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        indices = torch.tensor([[0, 0], [1, 1], [2, 2]], dtype=torch.int64)
        values = torch.tensor([[1.0, 1.1], [2.0, 2.1], [3.0, 3.1]])
        self.run_test(ScatterModel(), input_args=(input, indices, values))

        @torch.jit.script
        def scatter_sum(src: Tensor, index: Tensor):
            if False:
                i = 10
                return i + 15
            size = src.size()
            out = torch.zeros(size, dtype=src.dtype)
            return out.scatter_add_(1, index, src)

        class ScatterModel(torch.nn.Module):

            def forward(self, src, index):
                if False:
                    print('Hello World!')
                return scatter_sum(src, index)
        src = torch.rand(3, 2)
        index = torch.tensor([[0, 0], [1, 1], [0, 1]], dtype=torch.int64)
        self.run_test(ScatterModel(), (src, index))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_different_size_index_src(self):
        if False:
            i = 10
            return i + 15

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices, src):
                if False:
                    i = 10
                    return i + 15
                return input.scatter_add(0, indices, src)
        src = torch.ones((2, 5))
        input = torch.zeros(3, 5, dtype=src.dtype)
        indices = torch.tensor([[0, 1, 2, 0, 0]])
        self.run_test(ScatterModel(), input_args=(input, indices, src))

    @common_utils.parametrize('src, indices', [common_utils.subtest([torch.ones((1, 5)), torch.tensor([[0, 1, 2, 0, 0]])], name='src_indices_dynamic_combination1'), common_utils.subtest([torch.ones((2, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])], name='src_indices_dynamic_combination2'), common_utils.subtest([torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0, 0], [1, 0, 2, 1, 2]])], name='src_indices_dynamic_combination3'), common_utils.subtest([torch.ones((3, 5)), torch.tensor([[0, 1, 2, 0], [1, 0, 2, 1]])], name='src_indices_dynamic_combination4')])
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_add_dynamic_index(self, src, indices):
        if False:
            return 10

        class ScatterModel(torch.nn.Module):

            def forward(self, input, indices, src):
                if False:
                    for i in range(10):
                        print('nop')
                return input.scatter_add(0, indices, src)
        input = torch.zeros(3, 5, dtype=src.dtype)
        self.run_test(ScatterModel(), input_args=(input, indices, src), input_names=['input', 'indices', 'src'], dynamic_axes={'indices': {0: 'a', 1: 'b'}, 'src': {0: 'c', 1: 'd'}})

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()

            def forward(self, x, index, input):
                if False:
                    print('Hello World!')
                y_max = input.scatter_reduce(0, index, x, reduce='amax')
                y_sum = input.scatter_reduce(0, index, x, reduce='sum')
                y_min = input.scatter_reduce(0, index, x, reduce='amin')
                y_mul = input.scatter_reduce(0, index, x, reduce='prod')
                return (y_max, y_sum, y_min, y_mul)
        model = Model()
        model.eval()
        src = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        index = torch.tensor([0, 1, 0, 1, 2, 1])
        input = torch.tensor([1.0, 2.0, 3.0, 8.0])
        self.run_test(model, (src, index, input))

    @skipIfUnsupportedMinOpsetVersion(16)
    def test_scatter_reduce_self_rank_zero(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()

            def forward(self, x, index, input):
                if False:
                    for i in range(10):
                        print('nop')
                y_max = input.scatter_reduce(0, index, x, reduce='amax')
                y_sum = input.scatter_reduce(0, index, x, reduce='sum')
                y_min = input.scatter_reduce(0, index, x, reduce='amin')
                y_mul = input.scatter_reduce(0, index, x, reduce='prod')
                return (y_max, y_sum, y_min, y_mul)
        model = Model()
        model.eval()
        empty_tensor = torch.tensor([])
        empty_idx = torch.tensor([], dtype=torch.int64)
        self.run_test(model, (empty_tensor, empty_idx, empty_tensor))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_bucketize(self):
        if False:
            while True:
                i = 10

        class BucketModel(torch.nn.Module):

            def forward(self, input, boundaries):
                if False:
                    for i in range(10):
                        print('nop')
                return (torch.bucketize(input, boundaries), torch.bucketize(input, boundaries, right=True))
        input = torch.tensor([[2, 5, 10], [6, 8, 3]])
        boundaries = torch.tensor([1, 5, 7, 8, 10])
        self.run_test(BucketModel(), (input, boundaries))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_one_hot(self):
        if False:
            return 10

        class OneHot(torch.nn.Module):

            def __init__(self, num_classes):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.num_classes = num_classes

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.one_hot(x, self.num_classes)
        x = torch.arange(10)
        self.run_test(OneHot(15), x)

        class OneHot(torch.nn.Module):

            def forward(self, x, num_classes):
                if False:
                    i = 10
                    return i + 15
                num_classes = num_classes.to(torch.int32)
                return torch.nn.functional.one_hot(x, num_classes[0])
        x = torch.arange(10)
        num_classes = 15 * torch.ones(1)
        self.run_test(OneHot(), (x, num_classes))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gather(self):
        if False:
            for i in range(10):
                print('nop')

        class GatherModel(torch.nn.Module):

            def forward(self, input, indices):
                if False:
                    return 10
                return input.gather(1, indices)
        input = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
        indices = torch.tensor([[1, 0], [0, 1], [0, 1]], dtype=torch.int64)
        self.run_test(GatherModel(), input_args=(input, indices))

    @skipScriptTest()
    def test_gather_constant_fold(self):
        if False:
            for i in range(10):
                print('nop')

        class GatherModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.register_buffer('weight', torch.ones(5))
                self.embed = torch.nn.Embedding(8, 3)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                shape = self.weight.shape[0]
                m = 5 - shape
                y = torch.ones(1, 4, dtype=torch.long)
                return (x.clamp(min=m), self.embed(y))
        x = torch.randn(1)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('weight', torch.ones(2))

            def forward(self, x):
                if False:
                    return 10
                shape = self.weight.shape[0]
                pad = [1, shape, shape, shape]
                zero_pad = torch.nn.ZeroPad2d(pad)
                return zero_pad(x)
        x = torch.randn(1, 3, 2)
        self.run_test(GatherModule(), (x,))

        class GatherModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.register_buffer('rb', torch.randn(1, 1, 3, 1, 1))

            def forward(self, x):
                if False:
                    return 10
                x += self.rb[0]
                return x
        x = torch.randn(1, 3, 224, 224)
        self.run_test(GatherModule(), (x,), dynamic_axes={'input': {0: 'batch', 2: 'height', 3: 'width'}, 'output': {0: 'batch', 1: 'class', 2: 'height', 3: 'width'}}, input_names=['input'], output_names=['output'])

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_expand(self):
        if False:
            while True:
                i = 10

        class ExpandModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input.expand(2, 3, -1)
        input = torch.randn(2, 1, 4)
        self.run_test(ExpandModel(), input_args=input)

        class ExpandInferDimModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return input.expand(-1, input.size(0))
        input = torch.randn(3, 1)
        self.run_test(ExpandInferDimModel(), input_args=input)

        class ExpandTensorSizeModel(torch.nn.Module):

            def forward(self, input, size):
                if False:
                    return 10
                return input.expand(size)
        input = torch.randn(3)
        size = torch.tensor(-1)
        self.run_test(ExpandTensorSizeModel(), input_args=(input, size))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_expand_as(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x[:, x.size(0):] = 0
                return x
        x = torch.ones(2, 5)
        x2 = torch.randn(3, 4)
        self.run_test(Model(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1]}, additional_test_inputs=[x2])

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                x[:, x.size(0):] = torch.tensor([1, 2, 3])
                return x
        x = torch.ones(2, 5, 3)
        x2 = torch.randn(3, 4, 3)
        self.run_test(Model(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1, 2]}, additional_test_inputs=[x2])

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                aa = torch.tensor([[0], [1], [2]])
                return aa.expand_as(x)
        x = torch.ones(3, 2)
        x2 = torch.randn(3, 5)
        self.run_test(Model(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1]}, additional_test_inputs=[x2])

    def test_multinomial(self):
        if False:
            for i in range(10):
                print('nop')

        class Multinomial(torch.nn.Module):

            def forward(self, weight):
                if False:
                    i = 10
                    return i + 15
                return torch.multinomial(weight, 3, replacement=True)

        class MultinomialNoReplacement(torch.nn.Module):

            def forward(self, weight):
                if False:
                    print('Hello World!')
                return torch.multinomial(weight, 1)
        weight = torch.tensor([[0, 10, 0, 0], [0, 0, 100, 0]], dtype=torch.float)
        self.run_test(Multinomial(), (weight,))
        self.run_test(MultinomialNoReplacement(), (weight,))

    def _test_reduced_ops(self, op):
        if False:
            while True:
                i = 10

        class ReducedOpModule(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return op(input, dim=-1)
        if op != torch.mean:
            x = torch.randint(10, (4, 4), dtype=torch.uint8)
            self.run_test(ReducedOpModule(), x)
            x = torch.randint(10, (4, 4), dtype=torch.int8)
            self.run_test(ReducedOpModule(), x)
            x = torch.randint(10, (4, 4), dtype=torch.int16)
            self.run_test(ReducedOpModule(), x)
            x = torch.randint(10, (4, 4), dtype=torch.int32)
            self.run_test(ReducedOpModule(), x)
            x = torch.randint(10, (4, 4), dtype=torch.int64)
            self.run_test(ReducedOpModule(), x)
        if op != torch.prod and op != torch.mean:
            x = torch.randn(4, 5, dtype=torch.double)
            self.run_test(ReducedOpModule(), x)
        if op != torch.prod:
            x = torch.randn(4, 4, dtype=torch.half)
            self.run_test(ReducedOpModule(), x)
        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedOpModule(), x)

    def test_reduced_sum(self):
        if False:
            while True:
                i = 10
        return self._test_reduced_ops(op=torch.sum)

    def test_reduced_mean(self):
        if False:
            return 10
        return self._test_reduced_ops(op=torch.mean)

    def test_reduced_prod(self):
        if False:
            return 10
        return self._test_reduced_ops(op=torch.prod)

    def test_reduced_sum_dtypes(self):
        if False:
            print('Hello World!')

        class NoDimModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return input.sum(dtype=torch.float)

        class DimModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input.sum(dim=-1, dtype=torch.float)
        input = torch.randn((4, 4), dtype=torch.half)
        self.run_test(NoDimModel(), input)
        self.run_test(DimModel(), input)

    def test_reduced_min_max(self):
        if False:
            i = 10
            return i + 15

        class ReducedMinMaxModule(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return (torch.min(input, dim=-1)[0], torch.max(input, dim=0)[0])
        x = torch.randint(10, (4, 4), dtype=torch.int32)
        self.run_test(ReducedMinMaxModule(), x)
        x = torch.randint(10, (4, 4), dtype=torch.int64)
        self.run_test(ReducedMinMaxModule(), x)
        x = torch.randn(4, 5, dtype=torch.float)
        self.run_test(ReducedMinMaxModule(), x)

    def test_reduce_log_sum_exp(self):
        if False:
            i = 10
            return i + 15

        class ReduceLogSumExpModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                a = torch.logsumexp(input, dim=0)
                b = torch.logsumexp(input, dim=(0, 1))
                return a + b
        x = torch.randn(4, 4, requires_grad=True)
        self.run_test(ReduceLogSumExpModel(), x)

    def test_softmax(self):
        if False:
            i = 10
            return i + 15
        for i in range(-4, 3):
            model = torch.nn.Softmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):

                def __init__(self, i):
                    if False:
                        i = 10
                        return i + 15
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    if False:
                        return 10
                    return self.softmax(x.reshape(3, 4, 5, 6))
            model = torch.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_softmax_large_values(self):
        if False:
            for i in range(10):
                print('nop')
        input = torch.tensor([[-1000000000000.0, -1000000000000.0, -1000000000000.0], [1000000000000.0, 0.0, -5.0], [3.0, 4.0, 5.0]])
        for i in range(-2, 1):
            model = torch.nn.Softmax(dim=i)
            self.run_test(model, input)

            class SoftmaxUnknownRank(torch.nn.Module):

                def __init__(self, i):
                    if False:
                        return 10
                    super().__init__()
                    self.softmax = torch.nn.Softmax(dim=i)

                def forward(self, x):
                    if False:
                        return 10
                    return self.softmax(x.reshape(3, 3))
            model = torch.jit.script(SoftmaxUnknownRank(i))
            self.run_test(model, input)

    def test_logsoftmax(self):
        if False:
            print('Hello World!')
        for i in range(7)[2:]:
            model = torch.nn.LogSoftmax(dim=i - 1)
            dims = [2] * (i - 2) + [3, 4]
            input = torch.ones(*dims, requires_grad=True)
            self.run_test(model, input)

    def test_logsoftmax_dim(self):
        if False:
            return 10
        for i in range(-4, 3):
            model = torch.nn.LogSoftmax(dim=i)
            input = torch.randn(3, 4, 5, 6)
            self.run_test(model, input)

    def test_logsoftmax_dtype(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.nn.functional.log_softmax(x, dim=1, dtype=torch.float64)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(Model(), x)

    def test_softplus(self):
        if False:
            for i in range(10):
                print('nop')

        class BetaOneModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.softplus(x)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaOneModel(), x)

        class BetaModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.nn.functional.softplus(x, beta=2)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaModel(), x)

        class BetaFloatModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.nn.functional.softplus(x, beta=1.7)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(BetaFloatModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_hidden(self):
        if False:
            return 10

        class LSTMModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.rnn(x)
        input = torch.randn((10, 16, 16))
        self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_proj_no_hidden(self):
        if False:
            while True:
                i = 10

        class LSTMModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.rnn = torch.nn.LSTM(input_size=16, hidden_size=16, proj_size=8)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.rnn(x)
        input = torch.randn((10, 16, 16))
        with self.assertRaises(RuntimeError):
            self.run_test(LSTMModel(), (input,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm(self):
        if False:
            return 10

        class LSTMModel(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.rnn = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)

            def forward(self, x, h0, c0):
                if False:
                    while True:
                        i = 10
                return self.rnn(x, (h0, c0))
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        self.run_test(LSTMModel(), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_cell(self):
        if False:
            for i in range(10):
                print('nop')

        class LSTMCellModel(torch.nn.Module):

            def __init__(self, bias):
                if False:
                    print('Hello World!')
                super().__init__()
                self.lstm_cell = torch.nn.LSTMCell(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, bias=bias)

            def forward(self, x, h0, c0):
                if False:
                    for i in range(10):
                        print('nop')
                return self.lstm_cell(x, (h0, c0))
        input = torch.randn(BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(BATCH_SIZE, RNN_HIDDEN_SIZE)
        for bias in [True, False]:
            self.run_test(LSTMCellModel(bias), (input, h0, c0))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_default_init_state(self):
        if False:
            i = 10
            return i + 15

        class LSTMModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.rnn = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.rnn(x)
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(LSTMModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_fixed_batch_size(self):
        if False:
            print('Hello World!')

        class LSTMModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                if False:
                    print('Hello World!')
                batch_size = input.size()[1]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(LSTMModel(), input, fixed_batch_size=True, additional_test_inputs=[input2])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_post_fix_init_state(self):
        if False:
            print('Hello World!')

        class LSTMModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
                self.RNN_HIDDEN_SIZE = RNN_HIDDEN_SIZE

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                batch_size = input.size()[1]
                h0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                c0 = torch.ones([1, batch_size, self.RNN_HIDDEN_SIZE])
                return self.lstm(input, (h0, c0))
        model = LSTMModel()
        input = torch.randn(RNN_SEQUENCE_LENGTH, 1, RNN_INPUT_SIZE)
        input2 = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        self.run_test(model, input, input_names=['input.1'], dynamic_axes={'input.1': {0: 'seq', 1: 'batch'}}, additional_test_inputs=[input2])

    def test_lstm_constant_folding(self):
        if False:
            for i in range(10):
                print('nop')

        class LstmNet(torch.nn.Module):

            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.lstm = torch.nn.LSTM(input_size, hidden_size, num_layers, bidirectional=bidirectional)

            def forward(self, input, initial_state: Tuple[Tensor, Tensor]):
                if False:
                    while True:
                        i = 10
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional):
            if False:
                while True:
                    i = 10
            num_directions = 2 if bidirectional else 1
            model = LstmNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            c0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return (model, (input, (h0, c0)))
        batch_size1 = 3
        (model1, input1) = get_LstmNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)
        batch_size2 = 4
        (model2, input2) = get_LstmNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_no_bias(self):
        if False:
            return 10

        class LstmNet(torch.nn.Module):

            def __init__(self, num_layers, bidirectional):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers, bias=False, bidirectional=bidirectional)

            def forward(self, input, initial_state: Tuple[Tensor, Tensor]):
                if False:
                    print('Hello World!')
                return self.lstm(input, initial_state)

        def get_LstmNet_model_and_inputs(num_layers, bidirectional):
            if False:
                for i in range(10):
                    print('nop')
            input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
            num_directions = 2 if bidirectional else 1
            model = LstmNet(num_layers, bidirectional)
            h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            c0 = torch.randn(num_layers * num_directions, BATCH_SIZE, RNN_HIDDEN_SIZE)
            return (model, (input, (h0, c0)))
        num_layers = [1, 1, 2, 3]
        bidirectional = [True, False, True, False]
        models_and_inputs = [get_LstmNet_model_and_inputs(n, b) for (n, b) in zip(num_layers, bidirectional)]
        for (model, input) in models_and_inputs:
            self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lstm_sequence(self):
        if False:
            print('Hello World!')

        class LstmNet(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.rnn1 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear1 = torch.nn.Linear(8 * 2, 8)
                self.rnn2 = torch.nn.LSTM(8, 8, bidirectional=True, batch_first=True)
                self.linear2 = torch.nn.Linear(8 * 2, 8)

            def forward(self, input):
                if False:
                    return 10
                (rnn_output1, _) = self.rnn1(input)
                linear_output1 = self.linear1(rnn_output1)
                (rnn_output2, _) = self.rnn2(linear_output1)
                linear_output2 = self.linear2(rnn_output2)
                return linear_output2
        input = torch.zeros((1, 100, 8), dtype=torch.float32)
        self.run_test(LstmNet(), input, input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size', 1: 'w', 2: 'h'}, 'output': {0: 'batch_size', 1: 'w', 2: 'h'}})

    @skipScriptTest()
    def test_rnn_no_bias(self):
        if False:
            for i in range(10):
                print('nop')

        def make_model(layers, packed_sequence):
            if False:
                return 10
            batch_first = True if packed_sequence == 2 else False
            model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, bidirectional=False, batch_first=batch_first, bias=False)
            if packed_sequence == 1:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(model, False)
            if packed_sequence == 2:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequence(model, True)
            return model

        def make_input(batch_size, layers, packed_sequence):
            if False:
                print('Hello World!')
            batch_first = True if packed_sequence == 2 else False
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            h0 = torch.randn(layers, batch_size, RNN_HIDDEN_SIZE)
            inputs.append(h0)
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return input
        layers = [1, 3, 1, 3, 1, 3]
        packed_sequence = [0, 0, 1, 1, 2, 2]
        models = [make_model(l, p) for (l, p) in zip(layers, packed_sequence)]
        inputs = [make_input(RNN_BATCH_SIZE, l, p) for (l, p) in zip(layers, packed_sequence)]
        for (model, input) in zip(models, inputs):
            self.run_test(model, input)

    def test_gru_no_bias(self):
        if False:
            for i in range(10):
                print('nop')

        class GruNet(torch.nn.Module):

            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                if False:
                    print('Hello World!')
                super().__init__()
                self.mygru = torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, bias=False)

            def forward(self, input, initial_state):
                if False:
                    i = 10
                    return i + 15
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional):
            if False:
                for i in range(10):
                    print('nop')
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return (model, (input, h0))
        input_size = [7, 5]
        hidden_size = [3, 4]
        num_layers = [2, 3]
        batch_size = [3, 4]
        seq_len = [5, 7]
        bidirectional = [True, False]
        models_and_inputs = [get_GruNet_model_and_inputs(i, h, n, b, s, bi) for (i, h, n, b, s, bi) in zip(input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional)]
        for (model, input) in models_and_inputs:
            self.run_test(model, input, do_constant_folding=True)

    def test_gru_constant_folding(self):
        if False:
            i = 10
            return i + 15

        class GruNet(torch.nn.Module):

            def __init__(self, input_size, hidden_size, num_layers, bidirectional):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mygru = torch.nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional)

            def forward(self, input, initial_state):
                if False:
                    i = 10
                    return i + 15
                out = self.mygru(input, initial_state)
                return out

        def get_GruNet_model_and_inputs(input_size, hidden_size, num_layers, batch_size, seq_len, bidirectional):
            if False:
                while True:
                    i = 10
            num_directions = 2 if bidirectional else 1
            model = GruNet(input_size, hidden_size, num_layers, bidirectional)
            input = torch.randn(seq_len, batch_size, input_size)
            h0 = torch.randn(num_layers * num_directions, batch_size, hidden_size)
            return (model, (input, h0))
        batch_size1 = 3
        (model1, input1) = get_GruNet_model_and_inputs(7, 3, 2, batch_size1, 5, True)
        self.run_test(model1, input1, do_constant_folding=True)
        batch_size2 = 4
        (model2, input2) = get_GruNet_model_and_inputs(5, 4, 3, batch_size2, 7, False)
        self.run_test(model2, input2, do_constant_folding=True)

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_max_tensors(self):
        if False:
            i = 10
            return i + 15

        class MaxModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    i = 10
                    return i + 15
                return torch.max(input, other)
        model = MaxModel()
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 1, requires_grad=True)
        self.run_test(model, (x, y))

    def test_amax_amin(self):
        if False:
            for i in range(10):
                print('nop')

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (torch.amax(x, dim=0, keepdim=True), torch.amin(x, dim=[0, 1], keepdim=False))
        model = Model()
        x = torch.randn(4, 4)
        self.run_test(model, x)

    def test_aminmax(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (torch.aminmax(x, dim=1, keepdim=True), torch.aminmax(x, keepdim=False))
        model = Model()
        x = torch.randn(3, 4)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_end(self):
        if False:
            return 10

        class ArangeScript(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    return 10
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a
        x = torch.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):

            def forward(self, a):
                if False:
                    i = 10
                    return i + 15
                return torch.arange(a.size(0), dtype=torch.float).view(-1, 1) + a
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_end_notype(self):
        if False:
            i = 10
            return i + 15

        class ArangeScript(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    return 10
                return torch.arange(a.size(0))
        x = torch.randn(3, 4, requires_grad=True)
        outputs = ArangeScript()(x)
        self.run_test(ArangeScript(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(ArangeScript(), x, remained_onnx_input_idx=[])

        class ArangeModel(torch.nn.Module):

            def forward(self, a):
                if False:
                    return 10
                return torch.arange(a.size(0))
        self.run_test(ArangeModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(ArangeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end(self):
        if False:
            for i in range(10):
                print('nop')

        class ArangeScript(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    return 10
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a
        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):

            def forward(self, a):
                if False:
                    return 10
                return torch.arange(2, a.size(0) + 2, dtype=torch.float).view(-1, 1) + a
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_notype(self):
        if False:
            print('Hello World!')

        class ArangeScript(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    i = 10
                    return i + 15
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a
        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):

            def forward(self, a):
                if False:
                    while True:
                        i = 10
                return torch.arange(2.7, a.size(0) + 2).view(-1, 1) + a
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_arange_start_end_step(self):
        if False:
            while True:
                i = 10

        class ArangeScript(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    return 10
                return torch.arange(2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float).view(-1, 1) + a
        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):

            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.arange(2, a.size(0) * a.size(1) + 2, a.size(1), dtype=torch.float).view(-1, 1) + a
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_arange_start_end_step_notype(self):
        if False:
            print('Hello World!')

        class ArangeScript(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, a):
                if False:
                    i = 10
                    return i + 15
                return torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1) + a
        x = torch.randn(3, 4, requires_grad=True)
        self.run_test(ArangeScript(), x)

        class ArangeModel(torch.nn.Module):

            def forward(self, a):
                if False:
                    print('Hello World!')
                return torch.arange(2.7, a.size(0) * a.size(1) + 2, a.size(1)).view(-1, 1) + a
        self.run_test(ArangeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test__dim_arange(self):
        if False:
            i = 10
            return i + 15

        class DimArange(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch._dim_arange(input, 1)
        x = torch.ones(5, 6)
        self.run_test(DimArange(), x, input_names=['x'], dynamic_axes={'x': [0, 1]})
        remained_onnx_input_idx = None if self.opset_version < 11 else []
        self.run_test(DimArange(), x, remained_onnx_input_idx=remained_onnx_input_idx)

    def _test_compare_ops(self, model, num_inputs):
        if False:
            return 10
        x_float = torch.randn(1, 2, 3, 4, requires_grad=True)
        x_int = torch.randint(10, (3, 4), dtype=torch.int32)
        if num_inputs > 1:
            y_float = torch.randn(1, 2, 3, 4, requires_grad=True)
            y_int = torch.randint(10, (3, 4), dtype=torch.int32)
            self.run_test(model, (x_float, y_float))
            self.run_test(model, (x_float, y_int))
            self.run_test(model, (x_int, y_float))
            self.run_test(model, (x_int, y_int))
        else:
            self.run_test(model, x_float)
            self.run_test(model, x_int)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_and_or_xor(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return (x ^ y, x | y, x & y, ~x)
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(MyModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_and(self):
        if False:
            return 10

        class AndModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.logical_and(x, y)
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(AndModel(), input_args=(x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(AndModel(), input_args=(x, y))
        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(AndModel(), input_args=(x, y))
        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(AndModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_or(self):
        if False:
            i = 10
            return i + 15

        class OrModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.logical_or(x, y)
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(OrModel(), input_args=(x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(OrModel(), input_args=(x, y))
        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(OrModel(), input_args=(x, y))
        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(OrModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_xor(self):
        if False:
            for i in range(10):
                print('nop')

        class XorModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return torch.logical_xor(x, y)
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        y = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(XorModel(), input_args=(x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(XorModel(), input_args=(x, y))
        x = torch.randint(10, (5, 5), dtype=torch.double)
        y = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(XorModel(), input_args=(x, y))
        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        y = torch.randint(10, (2, 3, 5), dtype=torch.long)
        self.run_test(XorModel(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_logical_not(self):
        if False:
            i = 10
            return i + 15

        class NotModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.logical_not(x)
        x = torch.randint(0, 2, (5, 5), dtype=torch.bool)
        self.run_test(NotModel(), input_args=(x,))
        x = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(NotModel(), input_args=(x,))
        x = torch.randint(10, (5, 5), dtype=torch.double)
        self.run_test(NotModel(), input_args=(x,))
        x = torch.randint(10, (2, 3, 5), dtype=torch.float32)
        self.run_test(NotModel(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_eq(self):
        if False:
            i = 10
            return i + 15

        class EqualModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    while True:
                        i = 10
                return input == other
        self._test_compare_ops(EqualModel(), 2)

    def test_gt(self):
        if False:
            print('Hello World!')

        class GreaterModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    while True:
                        i = 10
                return input > other
        self._test_compare_ops(GreaterModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge(self):
        if False:
            return 10

        class GreaterOrEqualModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    i = 10
                    return i + 15
                return input >= other
        self._test_compare_ops(GreaterOrEqualModel(), 2)

    def test_gt_scalar(self):
        if False:
            for i in range(10):
                print('nop')

        class GreaterModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input > 1
        self._test_compare_ops(GreaterModel(), 1)

    def test_gt_primitive(self):
        if False:
            while True:
                i = 10

        class GreaterModel(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.y: int = 2

            def forward(self, x: int):
                if False:
                    while True:
                        i = 10
                return self.y > x
        x = 3
        self.run_test(GreaterModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ge_scalar(self):
        if False:
            for i in range(10):
                print('nop')

        class GreaterOrEqualModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return input >= 1
        self._test_compare_ops(GreaterOrEqualModel(), 1)

    def test_lt(self):
        if False:
            for i in range(10):
                print('nop')

        class LessModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    return 10
                return input > other
        self._test_compare_ops(LessModel(), 2)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le(self):
        if False:
            return 10

        class LessOrEqualModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    return 10
                return input <= other
        self._test_compare_ops(LessOrEqualModel(), 2)

    def test_lt_scalar(self):
        if False:
            return 10

        class LessModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return input < 1
        self._test_compare_ops(LessModel(), 1)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_le_scalar(self):
        if False:
            for i in range(10):
                print('nop')

        class LessOrEqualModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input <= 1
        self._test_compare_ops(LessOrEqualModel(), 1)

    def test_matmul(self):
        if False:
            print('Hello World!')

        class MatmulModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    i = 10
                    return i + 15
                return torch.matmul(input, other)
        x = torch.randn(3, 4, requires_grad=True)
        y = torch.randn(4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))
        x = torch.randint(10, (3, 4))
        y = torch.randint(10, (4, 5))
        self.run_test(MatmulModel(), (x, y))

    def test_matmul_batch(self):
        if False:
            print('Hello World!')

        class MatmulModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.matmul(input, other)
        x = torch.randn(2, 3, 4, requires_grad=True)
        y = torch.randn(2, 4, 5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))
        x = torch.randint(10, (2, 3, 4))
        y = torch.randint(10, (2, 4, 5))
        self.run_test(MatmulModel(), (x, y))

    def _argmin_argmax_model(self, input):
        if False:
            for i in range(10):
                print('nop')

        class ArgminArgmaxModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return (torch.argmin(input), torch.argmax(input), torch.argmin(input, keepdim=True), torch.argmax(input, keepdim=True), torch.argmin(input, dim=0, keepdim=True), torch.argmax(input, dim=1, keepdim=True))
        self.run_test(ArgminArgmaxModel(), input)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_argmin_argmax(self):
        if False:
            for i in range(10):
                print('nop')
        input = torch.randn(7, 3, 5)
        self._argmin_argmax_model(input)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_argmin_argmax_select_last_index(self):
        if False:
            for i in range(10):
                print('nop')
        input = torch.tensor([[1.0, 2.0, 3.0], [1.0, 1.0, 2.0]])
        self._argmin_argmax_model(input)
        input = torch.ones(7, 3, 5)
        self._argmin_argmax_model(input)

    def test_repeat(self):
        if False:
            i = 10
            return i + 15

        class RepeatModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x2 = x.repeat(y.shape[0], 1)
                y1 = y.view(-1, 1)
                return x2 + y1
        x = torch.tensor([1, 2, 3])
        y = torch.tensor([4, 5, 8, 9])
        self.run_test(RepeatModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave(self):
        if False:
            i = 10
            return i + 15

        class FlattenModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.repeat_interleave(2)
        x = torch.tensor([1, 2, 3])
        self.run_test(FlattenModel(), (x,))

        class DimsModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.repeat_interleave(4, dim=1)
        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel(), (x,))

        class DimsModel2(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=1)
        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(DimsModel2(), (x,))

        class RepeatsDimsModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=0)
        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel(), (x,))

        class RepeatsDimsModel2(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                repeats = torch.tensor([1, 2])
                return torch.repeat_interleave(x, repeats, dim=1)
        x = torch.tensor([[1, 2], [3, 4]])
        self.run_test(RepeatsDimsModel2(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_repeat_interleave_noop(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.repeat_interleave(1, dim=1)
        x = torch.randn(4, 1, 8)
        self.run_test(Model(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_dynamic_repeat_interleave(self):
        if False:
            return 10

        class SingleDynamicModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=1)
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        self.run_test(SingleDynamicModel(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': {1: 'w'}})

        class NegDynamicModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                repeats = torch.tensor(4)
                return torch.repeat_interleave(x, repeats, dim=-1)
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        self.run_test(NegDynamicModel(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': {1: 'w'}})

        class SingleDynamicModelFloat(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                repeats = torch.tensor([4])
                return torch.repeat_interleave(x, repeats, dim=0)
        x = torch.tensor([[1.1, 2.1], [3.1, 4.1]])
        another_x = torch.tensor([[7.1, 8.1], [5.1, 6.1]])
        self.run_test(SingleDynamicModelFloat(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': {0: 'h'}})

        class DynamicRepeatsModel(torch.nn.Module):

            def forward(self, x, repeats):
                if False:
                    while True:
                        i = 10
                return torch.repeat_interleave(x, repeats, dim=1)
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        another_x = torch.tensor([[7, 8], [5, 6]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(DynamicRepeatsModel(), (x, repeats), additional_test_inputs=[(another_x, another_repeats)], input_names=['input_1', 'repeats_1'], dynamic_axes={'input_1': {1: 'w'}, 'repeats_1': {0: 'r'}})

        class DynamicRepeatsModel2(torch.nn.Module):

            def forward(self, x, repeats):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.repeat_interleave(x, repeats, dim=1)
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2])
        another_repeats = torch.tensor([4])
        self.run_test(DynamicRepeatsModel2(), (x, repeats), additional_test_inputs=[(x, another_repeats)], input_names=['input_1', 'repeats_1'], dynamic_axes={'repeats_1': {0: 'r'}})

        class DynamicFlattenModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.repeat_interleave(2)
        x = torch.tensor([1, 2, 3])
        self.run_test(DynamicFlattenModel(), x, input_names=['input_1'], dynamic_axes={'input_1': {0: 'w'}})

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_multiple_dynamic_repeat_interleave(self):
        if False:
            i = 10
            return i + 15

        class DynamicRepeatsModel(torch.nn.Module):

            def forward(self, x, repeats):
                if False:
                    while True:
                        i = 10
                return torch.repeat_interleave(x, repeats, dim=1)
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3, 4])
        another_repeats = torch.tensor([4, 3, 2])
        self.run_test(DynamicRepeatsModel(), (x, repeats), additional_test_inputs=[(x, another_repeats)], input_names=['input_1', 'repeats_1'], dynamic_axes={'repeats_1': {0: 'r'}})

        class DynamicRepeatsModel2(torch.nn.Module):

            def forward(self, x, repeats):
                if False:
                    print('Hello World!')
                return torch.repeat_interleave(x, repeats, dim=0)
        x = torch.tensor([[1, 2, 4], [3, 4, 7]])
        repeats = torch.tensor([2, 3])
        another_repeats = torch.tensor([4, 3])
        self.run_test(DynamicRepeatsModel2(), (x, repeats), additional_test_inputs=[(x, another_repeats)], input_names=['input_1', 'repeats_1'], dynamic_axes={'repeats_1': {0: 'r'}})

    def test_view(self):
        if False:
            i = 10
            return i + 15

        class ViewModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return input.view(4, 24)
        x = torch.randint(10, (4, 2, 3, 4), dtype=torch.int32)
        self.run_test(ViewModel(), x)

    def test_view_dynamic(self):
        if False:
            return 10

        class ViewModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    for i in range(10):
                        print('nop')
                return input.view(other.shape)
        x = torch.randn(2, 3, 4)
        shape = torch.randn(6, 4)
        self.run_test(ViewModel(), (x, shape), input_names=['x', 'shape'], dynamic_axes={'x': [0, 1, 2], 'shape': [0, 1]})
        self.run_test(ViewModel(), (x, shape), remained_onnx_input_idx=[0])

    def test_view_dynamic_zero_dim(self):
        if False:
            print('Hello World!')

        class ViewModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                input = input.view(-1, 2)
                return input.view(1, -1)
        x = torch.ones(2)
        another_x = torch.empty((0,))
        self.run_test(ViewModel(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0]})

    def test_view_as(self):
        if False:
            print('Hello World!')

        class ViewModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    return 10
                return input.view_as(other)
        x = torch.randn(2, 3, 4)
        y = torch.randn(6, 4)
        self.run_test(ViewModel(), (x, y))

    def test_linear(self):
        if False:
            return 10

        class LinearModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc = torch.nn.Linear(16, 16)

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = self.fc(x)
                out = self.fc(out)
                return out
        x = torch.randn(3, 16)
        self.run_test(LinearModel(), (x,))

        class LinearModel(torch.nn.Module):

            def forward(self, input, weight, bias):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.linear(input, weight, bias)
        x = torch.randn(2, 2)
        y = torch.randn(2, 2)
        z = torch.randn(1)
        self.run_test(LinearModel(), (x, y, z))
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        z = torch.randn(1)
        self.run_test(LinearModel(), (x, y, z))

    @skipScriptTest()
    def test_weight_norm(self):
        if False:
            while True:
                i = 10
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=1)
        x = torch.randn(4, 5, requires_grad=True)
        self.run_test(model, x)
        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3))
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)
        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(1, 1, 3), dim=-2)
        x = torch.randn(1, 1, 5, requires_grad=True)
        self.run_test(model, x)
        model = torch.nn.utils.weight_norm(torch.nn.Conv1d(3, 6, 3), name='weight')
        x = torch.randn(3, 3, 5, requires_grad=True)
        self.run_test(model, x)

    @skipScriptTest()
    def test_weight_norm_nodim(self):
        if False:
            return 10
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(model, x)
        model = torch.nn.utils.weight_norm(torch.nn.Linear(5, 10), dim=None)
        x = torch.randn(4, 5, requires_grad=True)
        self.run_test(model, x)

    def test_flatten(self):
        if False:
            return 10

        class FlattenModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.flatten(input)
        model = FlattenModel()
        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(model, x)
        x = torch.randn([])
        self.run_test(model, x)
        x = torch.randn(4)
        self.run_test(model, x)

    def test_flatten2d(self):
        if False:
            return 10

        class FlattenModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.flatten(input, 1)
        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    def test_flatten2d_neg(self):
        if False:
            return 10

        class FlattenModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return (torch.flatten(x, 1, -1), torch.flatten(x, 0, -2), torch.flatten(x, 1, -2))
        x = torch.randint(10, (1, 2, 3, 4))
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_flatten_dynamic_axes(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.flatten(x, start_dim=2, end_dim=3)
        batch_size = 3
        x = torch.randn(batch_size, 5, 4, 5)
        y = torch.randn(5, 5, 4, 5)
        model = MyModule()
        self.run_test(model, x, additional_test_inputs=[y], input_names=['input'], output_names=['output'], dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}})

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_getitem(self):
        if False:
            while True:
                i = 10

        class GetItemModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y, z, ind):
                if False:
                    print('Hello World!')
                arr = [x, y, z]
                return arr[ind]
        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        ind = torch.tensor(1, dtype=torch.long)
        self.run_test(GetItemModel(), (x, y, z, ind))
        ind = torch.tensor(-2, dtype=torch.long)
        self.run_test(GetItemModel(), (x, y, z, ind))

    @skipDtypeChecking
    def test_item(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x, y, i: int):
                if False:
                    while True:
                        i = 10
                return int(x[y[i]].item())
        x = torch.arange(6, dtype=torch.float)
        y = torch.tensor([0, 1, 2, 3, 4], dtype=torch.long)
        i = 3
        self.run_test(torch.jit.script(M()), (x, y, i))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_nonzero(self):
        if False:
            while True:
                i = 10

        class NonzeroModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (x.nonzero(), x.nonzero(as_tuple=True))
        x = torch.randn(60).index_fill_(0, torch.randint(0, 60, (20,)), 0).view(3, 4, 5)
        self.run_test(NonzeroModel(), (x,))

    def test_unbind(self):
        if False:
            for i in range(10):
                print('nop')

        class UnbindModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                (_, out, _) = input.unbind()
                return out
        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                (_, out, _, _) = input.unbind(1)
                return out
        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

        class UnbindModel3(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                (_, out, _, _) = input.unbind(-2)
                return out
        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_len(self):
        if False:
            i = 10
            return i + 15

        class LenModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return len(input.unbind()) + input
        x = torch.randn(4, 5)
        self.run_test(LenModel(), x, input_names=['input'], dynamic_axes={'input': {0: 'seq'}}, additional_test_inputs=(torch.randn(5, 5),))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_len_list(self):
        if False:
            while True:
                i = 10

        class LenListModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.ones(len(input.shape))
        x = torch.randn(4, 5)
        self.run_test(LenListModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unbind_dynamic(self):
        if False:
            while True:
                i = 10

        class UnbindModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    return 10
                return input.unbind()[1]
        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel(), x)

        class UnbindModel2(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    return 10
                return input.unbind(-1)[1]
        x = torch.randn(3, 4, 5)
        self.run_test(UnbindModel2(), x)

    @skipScriptTest()
    def test_split(self):
        if False:
            print('Hello World!')

        class SplitModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return (input.split([2, 1, 2]), input.split([3, 2])[0])
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return (input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1])
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return input.split([2, 1, 2])
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_script(self):
        if False:
            while True:
                i = 10

        class SplitModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return (input.split([2, 1, 2]), input.split([3, 2])[0])
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return (input.split([2, 1, 1], -2), input.split([2, 2], -2)[-1])
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

        class SplitModel3(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return input.split([2, 1, 2])
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel3(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_split_size_as_list(self):
        if False:
            while True:
                i = 10

        class SplitModel(torch.nn.Module):

            def forward(self, input, split_sizes: List[int]):
                if False:
                    i = 10
                    return i + 15
                out = []
                split_list: List[Tensor] = input.split(split_sizes)
                for ob in split_list:
                    out.append(ob)
                return torch.cat(out, dim=0)
        x = torch.randn(6, 4, 3)
        split_sizes = [torch.tensor(2), torch.tensor(4)]
        self.run_test(SplitModel(), (x, split_sizes))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_size_with_slice(self):
        if False:
            i = 10
            return i + 15

        class SplitModule(torch.nn.Module):

            def forward(self, x, y, t):
                if False:
                    while True:
                        i = 10
                splits = (x.size(1), y.size(1))
                (out, out2) = torch.split(t, splits, dim=1)
                return (out, out2)
        x = torch.randn(2, 3)
        y = torch.randn(2, 4)
        t = torch.randn(2, 7)
        self.run_test(SplitModule(), (x, y, t), input_names=['x', 'y', 't'], dynamic_axes={'x': [0, 1], 'y': [0, 1], 't': [0, 1]})
        self.run_test(SplitModule(), (x, y, t), remained_onnx_input_idx=[2])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic(self):
        if False:
            i = 10
            return i + 15

        class SplitModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return input.split(2)[1]
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel(), x)

        class SplitModel2(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    print('Hello World!')
                return input.split(2, -3)[1]
        x = torch.randn(5, 4, 3)
        self.run_test(SplitModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_split_dynamic_axes(self):
        if False:
            while True:
                i = 10

        class Split(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.split(1, dim=-1)
        x = torch.randn(4, 384, 2)
        input_names = ['logits']
        self.run_test(Split(), x, input_names=input_names, dynamic_axes={input_names[0]: {0: 'batch'}})

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_chunk(self):
        if False:
            return 10

        class ChunkModel(torch.nn.Module):

            def __init__(self, dim=1):
                if False:
                    return 10
                super().__init__()
                self.dim = dim

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.chunk(x, 3, dim=self.dim)
        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = torch.randn(1, 18)
        for dim_size_ in range(13, 16):
            y = torch.randn(1, dim_size_)
            self.run_test(model, x, additional_test_inputs=[y], input_names=['x'], dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})
            self.run_test(model_neg_dim, x, additional_test_inputs=[y], input_names=['x'], dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dynamic_chunk(self):
        if False:
            while True:
                i = 10

        class ChunkModel(torch.nn.Module):

            def __init__(self, dim=1):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.dim = dim

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.chunk(x, x.size(0), dim=self.dim)
        model = ChunkModel()
        model.eval()
        model_neg_dim = ChunkModel(-1)
        model_neg_dim.eval()
        x = torch.randn(3, 18)
        for dim_size_ in range(13, 16):
            y = torch.randn(3, dim_size_)
            self.run_test(model, x, additional_test_inputs=[y], input_names=['x'], dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})
            self.run_test(model_neg_dim, x, additional_test_inputs=[y], input_names=['x'], dynamic_axes={'x': {0: 'batch_size', 1: 'dims'}})

    def test_concat(self):
        if False:
            return 10

        class ConcatModel(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    print('Hello World!')
                return torch.cat((x, y, z))
        x = torch.randn(3, 4, 5)
        y = torch.randn(1, 4, 5)
        z = torch.randn(2, 4, 5)
        self.run_test(ConcatModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_concat_dynamic(self):
        if False:
            while True:
                i = 10

        class ConcatDynamicModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    return 10
                return torch.cat(x.unbind())
        x = torch.randn(4, 5, 6)
        self.run_test(ConcatDynamicModel(), x)

    def test_stack(self):
        if False:
            for i in range(10):
                print('nop')

        class StackModel(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    return 10
                return torch.stack((x, y, z), 1)
        x = torch.randn(3, 4, 5)
        y = torch.randn(3, 4, 5)
        z = torch.randn(3, 4, 5)
        self.run_test(StackModel(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_stack_dynamic(self):
        if False:
            print('Hello World!')

        class StackDynamicModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.stack(x.unbind(), 1)
        x = torch.randn(4, 5, 6)
        self.run_test(StackDynamicModel(), x)

    def test_loop_dynamic(self):
        if False:
            return 10

        class LoopModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    print('Hello World!')
                for i in range(x.size(2)):
                    x = x + i
                return x
        model = LoopModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_loop_nested(self):
        if False:
            for i in range(10):
                print('nop')

        class NestedLoopsModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                for i in range(5):
                    a = 0
                    while a < 4:
                        a += 1
                    x = x + a
                return x
        model = NestedLoopsModel()
        inputs = torch.zeros(1, 2, 3, dtype=torch.long)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_with_list(self):
        if False:
            print('Hello World!')

        class ListLoopModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                res = []
                res1 = []
                arr = x.split([3, 4, 1, 1, 2, 3, 2], 0)
                res2 = torch.zeros(3, 4, dtype=torch.long)
                res3 = []
                res4 = []
                for i in range(len(arr)):
                    res.append(arr[i].sum(0, False))
                    res1.append(arr[-1 - i].sum(0, False))
                    res2 += 1
                    res3 = res3 + [arr[i].sum(0, False)]
                    res4 += [arr[-1 - i].sum(0, False)]
                return (res, res1, res2, torch.stack(res3), torch.stack(res4))
        model = ListLoopModel()
        inputs = torch.randn(16)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_transpose(self):
        if False:
            i = 10
            return i + 15

        class LoopModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                res = torch.zeros_like(x[0])
                for i in range(x.size(0)):
                    res += x[0].transpose(0, 1)
                return res
        model = torch.jit.script(LoopModel())
        x = torch.randn(5, 3, 3)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_loop_multi_dim(self):
        if False:
            i = 10
            return i + 15

        class LoopMultiDimModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                for x_ in torch.flip(x.narrow(0, 0, 7), [0]):
                    y = x_[0][y]
                return y
        model = LoopMultiDimModel()
        x = torch.randint(0, 5, (8, 1, 17), dtype=torch.long)
        y = torch.ones(1, dtype=torch.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list(self):
        if False:
            return 10

        class ListModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                tensors = x.unbind()
                res = []
                res.append(tensors[0])
                res.append(tensors[1])
                res.pop(1)
                res.insert(0, tensors[1])
                res.append(tensors[2])
                res += [tensors[3], tensors[4]]
                res = res + [tensors[5]]
                return torch.ones(len(res))
        model = ListModel()
        inputs = torch.randn(16, 1)
        self.run_test(model, inputs)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append(self):
        if False:
            print('Hello World!')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested(self):
        if False:
            while True:
                i = 10

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_list_append_nested_2(self):
        if False:
            i = 10
            return i + 15

        class ListModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                res = []
                res_replicate = []
                for i in range(x.size(0)):
                    if len(res) > 2:
                        for j in range(x.size(1)):
                            res.append(x[i][j])
                        res_replicate.append(res[-1])
                        res.append(res_replicate[-1])
                return (res, res_replicate)
        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        self.run_test(model, (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_nested_mixed_dtype(self):
        if False:
            return 10

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        if i == j:
                            res.append(x == y)
                        else:
                            res.append(x != y)
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(3, 4)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_pop(self):
        if False:
            return 10

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                res.pop()
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_nested(self):
        if False:
            print('Hello World!')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        res.pop()
                    res += [torch.matmul(x[i][0], y)]
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_del(self):
        if False:
            i = 10
            return i + 15

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                res = []
                for i in range(x.size(0)):
                    res += [torch.matmul(x[i], y)]
                del res[2]
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_nested(self):
        if False:
            print('Hello World!')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res += [torch.matmul(x[i][j], y)]
                        del res[i]
                    res += [torch.matmul(x[i][0], y)]
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_set(self):
        if False:
            print('Hello World!')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                res[y] = x[y]
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(12, 4)
        y = torch.tensor(2, dtype=torch.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_idx_sum(self):
        if False:
            for i in range(10):
                print('nop')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                indices = torch.arange(x.size(0))
                res = []
                for i in range(x.size(0)):
                    res.append(x[i])
                return res[torch.sum(indices[:y])]
        model = torch.jit.script(ListModel())
        x = torch.randn(12, 4)
        y = torch.tensor(2, dtype=torch.long)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories(self):
        if False:
            print('Hello World!')

        class TensorFactory(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.zeros(x.size()) + torch.ones(x.size())
        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_factories_script(self):
        if False:
            return 10

        class TensorFactory(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.zeros(x.shape, dtype=torch.float) + torch.ones(x.shape, dtype=torch.float)
        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tensor_like_factories_script(self):
        if False:
            while True:
                i = 10

        class TensorFactory(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                zeros = torch.zeros_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                ones = torch.ones_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return zeros + ones
        x = torch.randn(2, 3, 4)
        self.run_test(TensorFactory(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(TensorFactory(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split(self):
        if False:
            return 10

        class TensorSplitModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return (input.tensor_split([1, 3]), input.tensor_split([2, 4])[0], input.tensor_split([1, 3, 4], dim=-2), input.tensor_split([0, 2], dim=-2)[-1], input.tensor_split([2, 3, 5]))
        self.run_test(TensorSplitModel(), torch.randn(5, 4, 3))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_scalar(self):
        if False:
            return 10

        class TensorSplitModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.tensor_split(x, x.size(1))
        self.run_test(TensorSplitModel(), torch.randn(1, 2, 3))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_tensor_split_dynamic_axes(self):
        if False:
            print('Hello World!')

        class TensorSplitModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.tensor_split(1, dim=-1)
        x = torch.randn(4, 384, 2)
        input_names = ['logits']
        self.run_test(TensorSplitModel(), x, input_names=input_names, dynamic_axes={input_names[0]: {0: 'batch'}})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_eye(self):
        if False:
            i = 10
            return i + 15

        class TensorFactory(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (torch.eye(x.size()[1], 3), torch.eye(4, 4, dtype=torch.long), torch.eye(x.size()[1], 2, dtype=torch.long), torch.eye(x.shape[0]), torch.eye(x.shape[0], dtype=torch.float64))
        x = torch.randn(2, 3, 4)
        another_x = torch.randn(5, 6, 7)
        self.run_test(TensorFactory(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_diagonal(self):
        if False:
            for i in range(10):
                print('nop')

        class DiagonalModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.diagonal(x)
        x = torch.randn(2, 4, 5, 2)
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(DiagonalModel(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

        class DiagonalModelNegOffset(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.diagonal(x, offset=-1)
        x = torch.randn(2, 4, 5, 2)
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(DiagonalModelNegOffset(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

        class DiagonalModelPosOffset(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.diagonal(x, offset=1)
        x = torch.randn(2, 4, 5, 2)
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(DiagonalModelPosOffset(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

        class DiagonalModelWithDims(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.diagonal(x, offset=-1, dim1=1, dim2=2)
        x = torch.randn(2, 4, 5, 2)
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(DiagonalModelWithDims(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

        class DiagonalModelWithNegativeDims(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.diagonal(x, offset=0, dim1=-2, dim2=-1)
        x = torch.randn(2, 4, 5, 2)
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(DiagonalModelWithNegativeDims(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

        class DiagonalModelOffsetOverrun(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return (torch.diagonal(x, offset=-2), torch.diagonal(x, offset=5))
        x = torch.randn(2, 4, 5, 2)
        another_x = torch.randn(5, 6, 7, 8)
        self.run_test(DiagonalModelOffsetOverrun(), x, additional_test_inputs=[another_x], input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_zero(self):
        if False:
            i = 10
            return i + 15

        class Zero_(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (x.zero_(), x)
        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_zero_qkv(self):
        if False:
            for i in range(10):
                print('nop')

        class Zero_(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x[2:4].zero_()
        x = torch.randn(24, 3, 4)
        self.run_test(Zero_(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros(self):
        if False:
            print('Hello World!')

        class Zero_(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return (x.new_zeros(x.shape[1:2]), x.new_zeros(x.shape[2:], dtype=torch.long))
        x = torch.randn(2, 3, 4)
        self.run_test(Zero_(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(Zero_(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_zeros_with_dtype(self):
        if False:
            while True:
                i = 10

        class MyModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.emb = torch.nn.Embedding(50, 64)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                inp = x.new_zeros(x.shape)
                return self.emb(inp)
        model = MyModel()
        x = torch.Tensor([[2, 5, 6], [3, 2, 5]]).to(torch.int64)
        self.run_test(model, x, input_names=['x'], dynamic_axes={'x': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_ones(self):
        if False:
            return 10

        class OnesModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (x.new_ones(x.shape[1:2]), x.new_ones(x.shape[2:], dtype=torch.long))
        x = torch.randn(2, 3, 4)
        self.run_test(OnesModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(OnesModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_zeros_ones_with_tensor_input(self):
        if False:
            return 10

        class ZeroAndOnes(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return (torch.zeros(x, 1), torch.ones(x, 1))
        x = torch.tensor([2])
        self.run_test(ZeroAndOnes(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipShapeChecking
    def test_tolist(self):
        if False:
            print('Hello World!')

        class List(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    while True:
                        i = 10
                res: List[int] = input.tolist()
                return res
        self.run_test(List(), (torch.randint(100, (1,)),))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_list_pass(self):
        if False:
            return 10

        class Slice(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x.new_zeros(x.shape[2:] + y.shape[1:])
        x = torch.randn(2, 3, 4, 5)
        y = torch.randn(1, 2, 3, 4)
        self.run_test(Slice(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3]})
        self.run_test(Slice(), (x, y), remained_onnx_input_idx=[])

        class Size(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x.new_zeros(x.shape + y.shape)
        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(Size(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2], 'y': [0, 1, 2]})
        self.run_test(Size(), (x, y), remained_onnx_input_idx=[])

        class Array(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                arr1 = [x.shape[0], x.shape[1], 2]
                arr2 = [y.shape[0], y.shape[1]]
                return x.new_zeros(arr1 + arr2)
        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(Array(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2], 'y': [0, 1, 2]})
        self.run_test(Array(), (x, y), remained_onnx_input_idx=[])

        class List(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                l1 = list(x.shape)
                l2 = list(y.shape)
                return x.new_zeros(l1 + l2)
        x = torch.randn(2, 3, 4)
        y = torch.randn(1, 2, 3)
        self.run_test(List(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2], 'y': [0, 1, 2]})
        self.run_test(List(), (x, y), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_empty(self):
        if False:
            for i in range(10):
                print('nop')

        class Emtpy(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return (x.new_empty(x.shape[0]).fill_(0), x.new_empty(x.shape[0], dtype=torch.long) * 0)
        x = torch.randn(2, 3, 4)
        self.run_test(Emtpy(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(Emtpy(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_new_full(self):
        if False:
            for i in range(10):
                print('nop')

        class Full(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return (x.new_full(x.shape[1:2], 5), x.new_full(x.shape[0:1], 1.3, dtype=torch.long))
        x = torch.randn(2, 3, 4)
        self.run_test(Full(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(Full(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_list(self):
        if False:
            return 10

        class Arithmetic(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.cat([x.add_(3), y.fill_(0)])
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(Arithmetic(), (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1], 'y': [0, 1]})
        self.run_test(Arithmetic(), (x, y), remained_onnx_input_idx=[0])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_fill(self):
        if False:
            for i in range(10):
                print('nop')

        class Fill_(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (x.fill_(3), x)
        x = torch.randn(2, 3, 4)
        self.run_test(Fill_(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(Fill_(), x, remained_onnx_input_idx=[])

    def test_inplace_arithmetic(self):
        if False:
            for i in range(10):
                print('nop')

        class Arithmetic(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                x.add_(3)
                y.mul_(x)
                return (x, y)
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4)
        self.run_test(Arithmetic(), (x, y))

    def test_inplace_arithmetic_half(self):
        if False:
            while True:
                i = 10

        class InplaceAddModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return x.add_(y)

        class InplaceMulModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x.mul_(y)
        x = torch.randn(2, 2, dtype=torch.half)
        y = torch.randn(2, 2, dtype=torch.float)
        self.run_test(InplaceAddModel(), (x, y), rtol=0.01, atol=0.01)
        self.run_test(InplaceMulModel(), (x, y), rtol=0.01, atol=0.01)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                a = torch.ones(12)
                for i in range(10):
                    a.add_(torch.ones(12))
                return a + x
        m = M()
        x = torch.randn(12)
        self.run_test(torch.jit.script(M()), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_inplace_with_loop_2(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                _bias = torch.ones(12)
                a = torch.ones(12)
                a_ref = a
                b = x.clone()
                b_ref = b
                for i in range(10):
                    if i == 3:
                        for j in range(5):
                            a += _bias
                            _bias.add_(torch.ones(12))
                            b = b + torch.ones(12)
                    _bias.add_(torch.ones(12))
                    a += _bias
                b_ref += torch.ones(12)
                return (_bias + x, a, b, b_ref)
        m = M()
        x = torch.zeros(12)
        self.run_test(torch.jit.script(M()), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_with_loop(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self._bias = torch.arange(12)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                self._bias = torch.arange(12)
                for i in range(10):
                    if i == 3:
                        for j in range(5):
                            self._bias += torch.arange(12)
                return self._bias + x
        m = M()
        x = torch.zeros(12)
        self.run_test(torch.jit.script(M()), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_inplace_attr_copy_with_loop(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self._bias = torch.arange(12)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                self._bias = torch.arange(12)
                for i in range(10):
                    if i == 3:
                        for j in range(5):
                            self._bias.copy_(torch.arange(12))
                        self._bias.copy_(self._bias + torch.arange(12))
                    self._bias.copy_(self._bias + torch.arange(12))
                return self._bias + x
        m = M()
        x = torch.zeros(12)
        self.run_test(torch.jit.script(M()), x)

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_inplace_sequence_with_loop(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def process(self, beam_hyps: List[Tensor], done: Tensor, x):
                if False:
                    print('Hello World!')
                batch_size = x.shape[0]
                for i in range(batch_size):
                    if done[i]:
                        continue
                    beam_idx = 0
                    for (_, token) in enumerate(x[i]):
                        beam_hyps.append(token)
                        beam_idx += 1
                        if beam_idx == 6:
                            break
                    done[i] = len(beam_hyps) > 4
                return (beam_hyps, done)

            def forward(self, x):
                if False:
                    print('Hello World!')
                beam_hyps: List[Tensor] = []
                batch_size = x.shape[0]
                cur_len = 0
                max_len = x.shape[1]
                done = torch.zeros(batch_size, dtype=torch.bool)
                while cur_len < max_len:
                    (beam_hyps, done) = self.process(beam_hyps, done, x[:, 0, :])
                    cur_len = cur_len + 1
                return beam_hyps
        m = torch.jit.script(M())
        x = torch.randn(8, 4, 3)
        self.run_test(torch.jit.script(M()), x)

    @skipScriptTest()
    def test_sort(self):
        if False:
            print('Hello World!')

        class SortModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=True))
                return out
        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_sort_ascending(self):
        if False:
            return 10

        class SortModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = []
                for i in range(-2, 2):
                    out.append(torch.sort(x, dim=i, descending=False))
                return out
        x = torch.randn(3, 4)
        self.run_test(SortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_argsort(self):
        if False:
            while True:
                i = 10

        class ArgSortModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.argsort(x, dim=1, descending=False)
        x = torch.randn(3, 4)
        self.run_test(ArgSortModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill(self):
        if False:
            for i in range(10):
                print('nop')

        class MaskedFillModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                return x.masked_fill(mask, 2)
        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.masked_fill(x > 3, -1)
        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_masked_fill_inplace(self):
        if False:
            while True:
                i = 10

        class MaskedFillModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                mask = torch.tensor([[0, 0, 1], [1, 1, 0]], dtype=torch.bool)
                x.masked_fill_(mask, 2)
                return x
        x = torch.zeros(4, 2, 3, requires_grad=True)
        self.run_test(MaskedFillModel(), x)

        class MaskedFillModel2(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x.masked_fill_(x > 3, -1)
                return x
        x = torch.arange(16).view(2, 2, 4).to(torch.float32)
        self.run_test(MaskedFillModel2(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_scatter(self):
        if False:
            while True:
                i = 10

        class MaskedScatterModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.masked_scatter(x, x.ge(0.5), torch.ones(100, 100) * 5)
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedScatterModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_masked_select(self):
        if False:
            return 10

        class MaskedSelectModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.masked_select(x, x.ge(0.5))
        x = torch.randn(3, 4, 5, requires_grad=True)
        self.run_test(MaskedSelectModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_fill(self):
        if False:
            print('Hello World!')

        class MaskedFillModel(torch.nn.Module):

            def forward(self, input_mask, some_const):
                if False:
                    while True:
                        i = 10
                mask = input_mask.clone()
                mask[mask != some_const] = 1
                mask[mask == some_const] = 0
                return mask
        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedFillModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_to_masked_scatter(self):
        if False:
            i = 10
            return i + 15

        class MaskedScatterModel(torch.nn.Module):

            def forward(self, input_mask, some_const):
                if False:
                    i = 10
                    return i + 15
                mask = input_mask.clone()
                mask[mask != some_const] = torch.ones(8)
                return mask
        mask = torch.randn(2, 2, 2, requires_grad=True)
        constant = torch.tensor(5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (mask, constant))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_with_1d_mask_to_masked_scatter(self):
        if False:
            i = 10
            return i + 15

        class MaskedScatterModel(torch.nn.Module):

            def forward(self, tensor, mask, some_const):
                if False:
                    return 10
                tensor[mask] = some_const
                return tensor
        mask = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1], dtype=torch.bool)
        tensor = torch.randn(8, 4, 5, requires_grad=True)
        some_const = torch.randn(4, 4, 5, dtype=torch.float)
        self.run_test(MaskedScatterModel(), (tensor, mask, some_const))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_shuffle(self):
        if False:
            print('Hello World!')

        class PixelShuffle(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.pixel_shuffle(x, upscale_factor=2)
        x = torch.randn(2, 16, 4, 3, requires_grad=True)
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelShuffle(), x)
        self.run_test(PixelShuffle(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2, 3]}, additional_test_inputs=[y])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pixel_unshuffle(self):
        if False:
            print('Hello World!')

        class PixelUnshuffle(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.pixel_unshuffle(x, downscale_factor=2)
        x = torch.randn(2, 16, 4, 6, requires_grad=True)
        y = torch.randn(4, 32, 8, 4, requires_grad=True)
        self.run_test(PixelUnshuffle(), x)
        self.run_test(PixelUnshuffle(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2, 3]}, additional_test_inputs=[y])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_reciprocal(self):
        if False:
            while True:
                i = 10

        class ReciprocalModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.reciprocal(x)
        model = ReciprocalModel()
        x = torch.tensor([2, 4])
        self.run_test(model, x.to(torch.long))
        self.run_test(model, x.to(torch.float))
        self.run_test(model, x.to(torch.double))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type(self):
        if False:
            i = 10
            return i + 15

        class ArithmeticModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (x.size(0) * 2 * x, 2 - x)
        x = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ArithmeticModel(), x)

        class ComparisonModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                a = torch.tensor([12.0])
                return (x.lt(1.5) & y.le(2) & x.le(1), x.gt(y), x.lt(y), a.ge(x.size(0)))
        x = torch.ones(2, 3, dtype=torch.int32)
        y = torch.ones(2, 3, dtype=torch.float32)
        self.run_test(ComparisonModel(), (x, y))

        class MatMulModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.mm(x, x) + x + torch.mm(x, x) + x
        x = torch.ones(3, 3)
        self.run_test(MatMulModel(), x)

        class AddMMModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.mm(x, x) + x
        x = torch.ones(3, 3)
        self.run_test(AddMMModel(), x)

        class FullModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.full((3, 4), x)
        x = torch.tensor(12.0)
        self.run_test(FullModel(), x)

        class CatModel(torch.nn.Module):

            def forward(self, fp16, fp32):
                if False:
                    print('Hello World!')
                return torch.cat([fp16, fp32])
        fp16 = Tensor([0.5])
        fp16 = fp16.half()
        fp32 = Tensor([1.5])
        self.run_test(CatModel(), (fp16, fp32))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_scalar_type_does_not_trigger_upcast_type_promotion(self):
        if False:
            i = 10
            return i + 15

        class DoNotUpcastModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                scale = x.size()[-1] ** (-0.5)
                return x * scale
        x = torch.ones(2, 3, dtype=torch.float16)
        self.run_test(DoNotUpcastModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_full_like(self):
        if False:
            for i in range(10):
                print('nop')

        class FullLikeModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.full_like(x, 1.3, dtype=torch.int)
        x = torch.tensor(12)
        self.run_test(FullLikeModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipDtypeChecking
    def test_full_like_value(self):
        if False:
            print('Hello World!')

        class FullLikeModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                out = y + 2
                return torch.full_like(x, out)
        x = torch.tensor(12)
        y = torch.tensor(2)
        self.run_test(FullLikeModel(), (x, y))

    def test_l1_norm(self):
        if False:
            while True:
                i = 10

        class NormModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.norm(x, p=1, dim=-1, keepdim=False)
        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_l2_norm(self):
        if False:
            i = 10
            return i + 15

        class NormModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.norm(x, p=2, dim=-2, keepdim=False)
        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm(self):
        if False:
            return 10

        class NormModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.norm(x, p='fro', dim=0, keepdim=False)
        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_frobenius_norm_keepdim(self):
        if False:
            for i in range(10):
                print('nop')

        class NormModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.norm(x, p='fro', dim=(0, 1), keepdim=True)
        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(NormModel(), x)

    def test_unfold(self):
        if False:
            return 10

        class UnfoldModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.unfold(dimension=2, size=2, step=2)
        x = torch.randn(4, 2, 3, requires_grad=True)
        y = torch.randn(2, 1, 3, requires_grad=True)
        self.run_test(UnfoldModel(), x, dynamic_axes={'x': [0, 1]}, input_names=['x'], additional_test_inputs=[y])

    def test_unfold_infer_shape(self):
        if False:
            for i in range(10):
                print('nop')

        class UnfoldModule(torch.jit.ScriptModule):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 1, 3, stride=2)

            @torch.jit.script_method
            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                return x.unfold(dimension=2, size=2, step=2)
        x = torch.randn(32, 3, 64)
        self.run_test(UnfoldModule(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_unfold_dynamic_inputs(self):
        if False:
            print('Hello World!')

        class UnfoldModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.unfold(dimension=2, size=x.shape[1], step=x.shape[1] - 1)
        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

        class UnfoldModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.unfold(dimension=2, size=x.shape[1], step=1)
        x = torch.randn(4, 2, 4, requires_grad=True)
        self.run_test(UnfoldModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_mv(self):
        if False:
            i = 10
            return i + 15

        class MatmulModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.mv(input, other)
        x = torch.randn(4, 5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))
        x = torch.randint(10, (4, 5))
        y = torch.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_dot(self):
        if False:
            while True:
                i = 10

        class MatmulModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    i = 10
                    return i + 15
                return torch.dot(input, other)
        x = torch.randn(5, requires_grad=True)
        y = torch.randn(5, requires_grad=True)
        self.run_test(MatmulModel(), (x, y))
        x = torch.randint(10, (5,))
        y = torch.randint(10, (5,))
        self.run_test(MatmulModel(), (x, y))

    @skipScriptTest()
    def test_spectral_norm(self):
        if False:
            print('Hello World!')
        m = torch.nn.utils.spectral_norm(torch.nn.Linear(2, 4))
        x = torch.randn(6, 2)
        self.run_test(m, (x,))

    def test_prelu(self):
        if False:
            print('Hello World!')

        class PReluModel(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.prelu = torch.nn.PReLU()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.prelu(x)
        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 4, 5)
        self.run_test(PReluModel(), x, input_names=['x'], dynamic_axes={'x': [1, 2]}, additional_test_inputs=[y])

    def test_prelu_scalar(self):
        if False:
            while True:
                i = 10
        x = torch.scalar_tensor(1.0)
        self.run_test(torch.nn.PReLU(), x, input_names=['x'])

    def test_relu6(self):
        if False:
            print('Hello World!')

        class Relu6Model(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.relu6 = torch.nn.ReLU6()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.relu6(x)
        x = torch.randn(2, 3, 4) * 100.0
        y = torch.randn(2, 4, 5) * 100.0
        self.run_test(Relu6Model(), x, input_names=['x'], dynamic_axes={'x': [1, 2]}, additional_test_inputs=[y])

    def test_silu(self):
        if False:
            for i in range(10):
                print('nop')

        class SiLUModel(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.silu = torch.nn.SiLU()

            def forward(self, x):
                if False:
                    return 10
                return self.silu(x)
        x = torch.randn(2, 3, 4)
        self.run_test(SiLUModel(), x)

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_tril(self):
        if False:
            for i in range(10):
                print('nop')

        class trilModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.tril(x)
        x = torch.randn(2, 3, 4)
        self.run_test(trilModel(), x)

        class trilModelwithDiagonal(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.tril(x, diagonal=1)
        x = torch.randn(2, 3, 4)
        self.run_test(trilModelwithDiagonal(), x)

        class trilModelwithNegDiagonal(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.tril(x, diagonal=-1)
        x = torch.randn(2, 3, 4)
        self.run_test(trilModelwithNegDiagonal(), x)

        class trilModelWithDiagonalInput(torch.nn.Module):

            def forward(self, x, diagnonal: int):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.tril(x, diagonal=diagnonal)
        x = torch.randn(2, 3, 4)
        self.run_test(trilModelWithDiagonalInput(), (x, 5))

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_triu(self):
        if False:
            return 10

        class triuModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.triu(x)
        x = torch.randn(2, 3, 4)
        self.run_test(triuModel(), x)

        class triuModelwithDiagonal(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.triu(x, diagonal=1)
        x = torch.randn(2, 3, 4)
        self.run_test(triuModelwithDiagonal(), x)

        class triuModelwithNegDiagonal(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.triu(x, diagonal=-1)
        x = torch.randn(2, 3, 4)
        self.run_test(triuModelwithNegDiagonal(), x)

        class triuModelWithDiagonalInput(torch.nn.Module):

            def forward(self, x, diagnonal: int):
                if False:
                    return 10
                return torch.triu(x, diagonal=diagnonal)
        x = torch.randn(2, 3, 4)
        self.run_test(triuModelWithDiagonalInput(), (x, 5))

    def test_mish(self):
        if False:
            return 10

        class MishModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mish = torch.nn.Mish()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.mish(x)
        x = torch.randn(2, 3, 4)
        self.run_test(MishModel(), x)

    def test_remainder(self):
        if False:
            for i in range(10):
                print('nop')

        class RemainderModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    i = 10
                    return i + 15
                return torch.remainder(input, other)
        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(RemainderModel(), (x, y))
        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        y = torch.tensor([2], dtype=torch.long)
        self.run_test(RemainderModel(), (x, y))
        x = x.to(torch.float)
        self.run_test(RemainderModel(), (x, y))
        y = y.to(torch.float)
        self.run_test(RemainderModel(), (x, y))
        x = x.to(torch.int32)
        self.run_test(RemainderModel(), (x, y))

    def test_remainder_scalar(self):
        if False:
            return 10

        class RemainderModel(torch.nn.Module):

            def __init__(self, scalar=2.55):
                if False:
                    print('Hello World!')
                super().__init__()
                self.scalar = scalar

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.remainder(input, self.scalar)
        x = torch.randint(10, (2, 3))
        self.run_test(RemainderModel(), x)
        x = torch.tensor([7, 6, -7, -6], dtype=torch.long)
        self.run_test(RemainderModel(2), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod(self):
        if False:
            i = 10
            return i + 15

        class FModModel(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    while True:
                        i = 10
                return torch.fmod(input, other)
        x = torch.randn(4, 2, 3)
        y = torch.randn(1, 2, 1)
        self.run_test(FModModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fmod_scalar(self):
        if False:
            print('Hello World!')

        class FModModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.fmod(input, 2.55)
        x = torch.randint(10, (2, 3))
        self.run_test(FModModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_glu(self):
        if False:
            print('Hello World!')

        class GluModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.glu(x)
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_gelu(self):
        if False:
            while True:
                i = 10

        class GeluModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.gelu(x, approximate='none')
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_tanh_gelu(self):
        if False:
            while True:
                i = 10

        class GeluModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.nn.functional.gelu(x, approximate='tanh')
        x = torch.randn(2, 4, 5, 6, requires_grad=True)
        self.run_test(GeluModel(), x)

    def test_add_inplace(self):
        if False:
            for i in range(10):
                print('nop')

        class InplaceAddModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x += 12
                return x
        x = torch.randn(4, 2, 3, requires_grad=True)
        self.run_test(InplaceAddModel(), x)

    def test_addcmul(self):
        if False:
            i = 10
            return i + 15

        class AddcmulModel(torch.nn.Module):

            def forward(self, x, t1, t2):
                if False:
                    return 10
                return (torch.addcmul(x, t1, t2), torch.addcmul(x, t1, t2, value=2.2))
        x = torch.randn(1, 3)
        t1 = torch.randn(3, 1)
        t2 = torch.randn(1, 3)
        self.run_test(AddcmulModel(), (x, t1, t2))

    def test_rsqrt(self):
        if False:
            print('Hello World!')

        class RsqrtModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.rsqrt()
        x = torch.randn(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    def test_rsqrt_zeros(self):
        if False:
            i = 10
            return i + 15

        class RsqrtModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x.rsqrt()
        x = torch.zeros(4, 2, 3, requires_grad=True, dtype=torch.float64)
        self.run_test(RsqrtModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique(self):
        if False:
            for i in range(10):
                print('nop')

        class UniqueModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.unique(x, sorted=True, return_inverse=False, return_counts=True)
        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_unique_along_dim(self):
        if False:
            return 10

        class UniqueModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.unique(x, dim=0, sorted=True, return_inverse=True, return_counts=False)
        x = torch.tensor([1, 3, 2, 3], dtype=torch.long)
        self.run_test(UniqueModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum(self):
        if False:
            print('Hello World!')

        class CumSum(torch.nn.Module):

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return torch.cumsum(input, dim=0)
        x = torch.randn(2, 3, 4)
        model = CumSum()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_cumsum_with_cast(self):
        if False:
            print('Hello World!')

        class CumSum(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.cumsum(input, dim=0, dtype=torch.float32)
        model = CumSum()
        x = torch.tensor([2, 3, 4], dtype=torch.int32)
        self.run_test(model, x)
        x = torch.tensor([False, True, True])
        self.run_test(model, x)

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_embedding_bag(self):
        if False:
            while True:
                i = 10
        model = torch.nn.EmbeddingBag(10, 5, mode='sum', scale_grad_by_freq=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))
        model = torch.nn.EmbeddingBag(10, 5, mode='sum', include_last_offset=True)
        input = torch.randint(10, (7,))
        offset = torch.tensor([0, 2, 5, 6])
        self.run_test(model, (input, offset))
        model = torch.nn.EmbeddingBag(10, 5, mode='max')
        input = torch.randint(10, (7, 5))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_1d_per_sample_weights(self):
        if False:
            i = 10
            return i + 15

        class EmbeddingModel(torch.nn.Module):

            def forward(self, embedding_matrix, input, offset, weights):
                if False:
                    print('Hello World!')
                return torch.nn.functional.embedding_bag(input, embedding_matrix, offsets=offset, mode='sum', per_sample_weights=weights)
        model = EmbeddingModel()
        x = torch.randint(7, (6,))
        w = torch.randn(6)
        offset = torch.tensor([0, 2, 5])
        embedding_matrix = torch.rand(10, 15)
        self.run_test(model, (embedding_matrix, x, offset, w))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_bag_2d_per_sample_weights(self):
        if False:
            for i in range(10):
                print('nop')

        class EmbeddingModel(torch.nn.Module):

            def forward(self, embedding_matrix, input, weights):
                if False:
                    return 10
                return torch.nn.functional.embedding_bag(input, embedding_matrix, mode='sum', per_sample_weights=weights)
        embedding_matrix = torch.rand(10, 15)
        model = EmbeddingModel()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)
        x2 = torch.randint(7, (4, 3))
        w2 = torch.randn(4, 3)
        self.run_test(model, (embedding_matrix, x, w), input_names=['embed', 'x', 'w'], dynamic_axes={'x': [0], 'w': [0]}, additional_test_inputs=[(embedding_matrix, x2, w2)])

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    @unittest.skip('Due to ONNX Loop shape inference issue. https://msdata.visualstudio.com/Vienna/_workitems/edit/1352001')
    def test_embedding_bag_dynamic_input(self):
        if False:
            while True:
                i = 10

        class EmbeddingModel1D(torch.nn.Module):

            def forward(self, embedding_matrix, input, weights, offsets):
                if False:
                    print('Hello World!')
                return torch.nn.functional.embedding_bag(input, embedding_matrix, offsets=offsets, mode='sum', per_sample_weights=weights)
        model = EmbeddingModel1D()
        x = torch.randint(7, (6,))
        w = torch.randn(6)
        offsets = torch.tensor([0, 2, 5], dtype=torch.long)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (2,))
        w2 = torch.randn(2)
        embedding_matrix2 = torch.rand(12, 25)
        offsets2 = torch.tensor([0], dtype=torch.long)
        self.run_test(model, (embedding_matrix, x, w, offsets), additional_test_inputs=[(embedding_matrix2, x2, w2, offsets2)], input_names=['embedding_matrix', 'x', 'offsets', 'w'], dynamic_axes={'embedding_matrix': [0, 1], 'x': [0], 'offsets': [0], 'w': [0]})

        class EmbeddingModel2D(torch.nn.Module):

            def forward(self, embedding_matrix, input, weights):
                if False:
                    return 10
                return torch.nn.functional.embedding_bag(input, embedding_matrix, mode='sum', per_sample_weights=weights)
        model = EmbeddingModel2D()
        x = torch.randint(7, (2, 3))
        w = torch.randn(2, 3)
        embedding_matrix = torch.rand(10, 15)
        x2 = torch.randint(7, (3, 5))
        w2 = torch.randn(3, 5)
        embedding_matrix2 = torch.rand(12, 25)
        self.run_test(model, (embedding_matrix, x, w), additional_test_inputs=[(embedding_matrix2, x2, w2)], input_names=['embedding_matrix', 'x', 'w'], dynamic_axes={'embedding_matrix': [0, 1], 'x': [0, 1], 'w': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid(self):
        if False:
            i = 10
            return i + 15

        class Meshgrid(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    for i in range(10):
                        print('nop')
                (output1, output2, output3) = torch.meshgrid(x, y, z)
                return (output1, output2, output3)
        x = torch.randn(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.randn(5, requires_grad=True)
        self.run_test(Meshgrid(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_indexing(self):
        if False:
            while True:
                i = 10

        class Meshgrid(torch.nn.Module):

            def __init__(self, indexing):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.indexing = indexing

            def forward(self, x, y, z):
                if False:
                    while True:
                        i = 10
                (output1, output2, output3) = torch.meshgrid(x, y, z, indexing=self.indexing)
                return (output1, output2, output3)
        x = torch.randn(5, requires_grad=True)
        y = torch.zeros(6, requires_grad=True)
        z = torch.randn(7, requires_grad=True)
        for indexing in ('xy', 'ij'):
            self.run_test(Meshgrid(indexing), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(8)
    def test_meshgrid_scalar(self):
        if False:
            while True:
                i = 10

        class Meshgrid(torch.nn.Module):

            def forward(self, x, y, z):
                if False:
                    print('Hello World!')
                (output1, output2, output3) = torch.meshgrid(x, y, z)
                return (output1, output2, output3)
        x = torch.ones(3, requires_grad=True)
        y = torch.zeros(4, requires_grad=True)
        z = torch.tensor(2.0)
        self.run_test(Meshgrid(), (x, y, z))

    def test_baddbmm(self):
        if False:
            print('Hello World!')

        class MyModule(torch.nn.Module):

            def forward(self, input, batch1, batch2):
                if False:
                    i = 10
                    return i + 15
                return torch.baddbmm(input, batch1, batch2, alpha=torch.tensor(5), beta=3.5)
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2))

    def test_baddbmm_dynamic(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def forward(self, input, batch1, batch2, alpha, beta):
                if False:
                    while True:
                        i = 10
                return torch.baddbmm(input, batch1, batch2, alpha=alpha, beta=beta)
        x = torch.randn(10, 3, 5)
        batch1 = torch.randn(10, 3, 4)
        batch2 = torch.randn(10, 4, 5)
        alpha = torch.tensor(5)
        beta = torch.tensor(3.5)
        model = MyModule()
        self.run_test(model, (x, batch1, batch2, alpha, beta))

    def test_numel(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input.numel() * input
        x = torch.randn(2, 3, 5)
        x2 = torch.randn(4, 5, 6)
        model = MyModule()
        self.run_test(model, (x,), input_names=['x'], dynamic_axes={'x': [0, 1, 2]}, additional_test_inputs=[(x2,)])

    def test_numel_empty(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return input.numel() * input
        x = torch.randn(0)
        x2 = torch.randn(4)
        model = MyModule()
        self.run_test(model, (x,), input_names=['x'], dynamic_axes={'x': [0]}, additional_test_inputs=[(x2,)])

    def test_dtype(self):
        if False:
            while True:
                i = 10

        class MyModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input, other):
                if False:
                    return 10
                return input.to(dtype=other.dtype) + other
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_dtype_eq(self):
        if False:
            i = 10
            return i + 15

        class MyModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input, other):
                if False:
                    print('Hello World!')
                if input.dtype == other.dtype:
                    return input + other
                return input
        x = torch.randn(2, 3)
        y = torch.randn(2, 3)
        self.run_test(MyModel(), (x, y))

    def test_cast_to(self):
        if False:
            return 10

        class MyModule(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input, other):
                if False:
                    return 10
                return input.to(other) + other
        x = torch.randn(2, 3, 4)
        y = torch.tensor([1], dtype=torch.int64)
        model = MyModule()
        self.run_test(model, (x, y))

    def test_cast_to_bool(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    while True:
                        i = 10
                return torch.cat((input.to(other), other), 0)
        x = torch.randn(2, 3, 4)
        y = torch.zeros([2, 3, 4], dtype=torch.bool)
        model = MyModule()
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_cast_type_as_with_bfloat16(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = torch.ones((3, 4), dtype=torch.bfloat16)
                x = x.type_as(y)
                return x.to(dtype=torch.float16)
        x = torch.ones(3, 4, dtype=torch.float16)
        model = MyModule()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_type_as(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = torch.tensor([1.0])
                return x.type_as(y)
        a = torch.tensor([True, False], dtype=torch.bool)
        b = torch.randn(3, 4, dtype=torch.double)
        c = torch.ones((2, 2), dtype=torch.int64)
        model = MyModule()
        self.run_test(model, a)
        self.run_test(model, b)
        self.run_test(model, c)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_ones_bool(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                true = torch.ones(input.shape, dtype=torch.bool)
                return input.to(true) & true
        x = torch.randn(2, 3, 4)
        model = MyModule()
        self.run_test(model, x)

    def test_log(self):
        if False:
            while True:
                i = 10

        class Log(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.log(input)
        x = torch.rand(2, 3, 4)
        model = Log()
        self.run_test(model, x)

    def test_log1p(self):
        if False:
            return 10

        class Log1p(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.log1p(input)
        x = torch.rand(2, 3, 4)
        model = Log1p()
        self.run_test(model, x)

    def test_log10(self):
        if False:
            return 10

        class Log10(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.log10(input)
        x = torch.rand(2, 3, 4)
        model = Log10()
        self.run_test(model, x)

    def test_log2(self):
        if False:
            for i in range(10):
                print('nop')

        class Log2(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.log2(input)
        x = torch.tensor(1.0)
        model = Log2()
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round(self):
        if False:
            return 10

        class Round(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.round(x)
        x = torch.tensor([0.992, -1.0362, -1.5, 3.5], requires_grad=True)
        self.run_test(Round(), x)
        int_x = torch.tensor([9920, 1036, -1500, 35], dtype=torch.int32)
        self.run_test(Round(), int_x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_round_with_decimals(self):
        if False:
            return 10

        class Round(torch.nn.Module):

            def __init__(self, decimals):
                if False:
                    print('Hello World!')
                super().__init__()
                self.decimals = decimals

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.round(x, decimals=self.decimals)
        x = torch.tensor([0.992, -1234.0362, -1.5896, 3.5])
        for decimals in (0, -2, 3):
            self.run_test(Round(decimals), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_default(self):
        if False:
            for i in range(10):
                print('nop')

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_hop_length(self):
        if False:
            print('Hello World!')

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                n_fft = 16
                hop_length = 4
                return torch.stft(x, n_fft=n_fft, center=False, hop_length=hop_length, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_non_divisible_hop_length(self):
        if False:
            while True:
                i = 10

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                n_fft = 16
                hop_length = 5
                return torch.stft(x, n_fft=n_fft, center=False, hop_length=hop_length, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_int_same_size(self):
        if False:
            i = 10
            return i + 15

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                n_fft = 16
                win_length = 16
                return torch.stft(x, n_fft=n_fft, center=False, win_length=win_length, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_int_different_size(self):
        if False:
            return 10

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                n_fft = 16
                win_length = 9
                return torch.stft(x, n_fft=n_fft, center=False, win_length=win_length, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_custom(self):
        if False:
            while True:
                i = 10

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                n_fft = 16
                window = torch.hann_window(16)
                return torch.stft(x, n_fft=n_fft, center=False, window=window, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_custom_window_size(self):
        if False:
            for i in range(10):
                print('nop')

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                n_fft = 16
                window = torch.hann_window(10)
                return torch.stft(x, n_fft=n_fft, window=window, center=False, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises((AssertionError, RuntimeError)):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_window_length(self):
        if False:
            print('Hello World!')

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                n_fft = 16
                win_len = 17
                return torch.stft(x, n_fft=n_fft, win_length=win_len, center=False, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_window_size_with_win_len(self):
        if False:
            for i in range(10):
                print('nop')

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                n_fft = 16
                window = torch.hann_window(10)
                win_len = 10
                return torch.stft(x, n_fft=n_fft, window=window, win_length=win_len, center=False, return_complex=False)
        x = torch.randn((1, 32), requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_one_dimension(self):
        if False:
            while True:
                i = 10

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)
        x = torch.randn(32, requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_input_size(self):
        if False:
            return 10

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=False)
        x = torch.randn((1, 1, 32), requires_grad=True)
        with self.assertRaises(RuntimeError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_wrong_return_complex(self):
        if False:
            for i in range(10):
                print('nop')

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, return_complex=True)
        x = torch.randn((1, 32), requires_grad=True)
        with self.assertRaises(errors.SymbolicValueError):
            self.run_test(STFT(), x)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_normalize(self):
        if False:
            i = 10
            return i + 15

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, normalized=True, return_complex=False)
        x = torch.randn(32, requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(17)
    def test_stft_not_onesided(self):
        if False:
            i = 10
            return i + 15

        class STFT(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                n_fft = 16
                return torch.stft(x, n_fft=n_fft, center=False, onesided=False, return_complex=False)
        x = torch.randn(32, requires_grad=True)
        self.run_test(STFT(), x, atol=1e-06)

    def test_constant_pad(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.ConstantPad1d(2, 3.5)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)
        model = torch.nn.ConstantPad2d((3, 0, 2, 1), 3.5)
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @common_utils.parametrize('pad', [common_utils.subtest([2, 4], name='scalar_list'), common_utils.subtest([torch.tensor(2, dtype=torch.int64), torch.tensor(4, dtype=torch.int64)], name='scalar_tensor_list')])
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_types(self, pad):
        if False:
            i = 10
            return i + 15

        class Pad(torch.nn.Module):

            def forward(self, x, pad: List[int]):
                if False:
                    i = 10
                    return i + 15
                return torch.nn.functional.pad(x, pad)
        x = torch.randn(2, 2, 4, 4)
        self.run_test(Pad(), (x, pad))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular(self):
        if False:
            for i in range(10):
                print('nop')

        class PadModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.nn.functional.pad(x, (1, 2, 1, 2), mode='circular')
                return out
        x = torch.randn(2, 3, 3, 4)
        self.run_test(PadModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular_negative(self):
        if False:
            i = 10
            return i + 15

        class PadModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.nn.functional.pad(x, (-1, -2), mode='circular')
                return out
        x = torch.randn(2, 3, 6)
        self.run_test(PadModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_pad_circular_dynamic_axes(self):
        if False:
            print('Hello World!')

        class PadModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = torch.nn.functional.pad(x, (2, 1, 2, 1), mode='circular')
                return out
        x = torch.randn(4, 3, 5, 6)
        self.run_test(PadModel(), x, input_names=['input_1'], dynamic_axes={'input_1': [0, 1, 2, 3]})

    @skipIfUnsupportedMaxOpsetVersion(10)
    @skipScriptTest()
    def test_unsupported_pad(self):
        if False:
            i = 10
            return i + 15

        class Pad(torch.nn.Module):

            def forward(self, x, pad: List[int]):
                if False:
                    return 10
                return torch.nn.functional.pad(x, pad)
        x = torch.randn(2, 2, 4, 4)
        y = [2, 4]
        with self.assertRaisesRegex(RuntimeError, 'Unsupported: ONNX export of Pad.*' + 'The sizes of the padding must be constant'):
            self.run_test(Pad(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_if_fold(self):
        if False:
            while True:
                i = 10

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    return 10
                if y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    y = y - 1
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    while True:
                        i = 10
                if y.numel() > 1:
                    y = y + 4
                else:
                    y = y + 2
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    for i in range(10):
                        print('nop')
                if y.dim() != 3:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    return 10
                if y.dim() >= 1:
                    y = y + 4
                else:
                    y = y - 1
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    i = 10
                    return i + 15
                if y.dim() <= 1:
                    y = y + 4
                else:
                    y = y + 2
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    print('Hello World!')
                if y.dim() < 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    while True:
                        i = 10
                if y.dim() == 3 and y.dtype == torch.int:
                    y = y + 4
                    y = y + 2
                else:
                    y = y + 1
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    for i in range(10):
                        print('nop')
                if y.numel() != 0 and y.dim() == 2:
                    y = y + 4
                    y = y + 2
                else:
                    return y
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), x)

        class IfFoldModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                if x.numel() == y.numel():
                    y = x + y
                else:
                    y = y - x
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

        class IfFoldModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                if x.numel() != y.numel():
                    y = x + y
                else:
                    y = y - x
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((3, 4), dtype=torch.int)
        self.run_test(IfFoldModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_uninitialized(self):
        if False:
            for i in range(10):
                print('nop')

        class UninitializedModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    i = 10
                    return i + 15
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(UninitializedModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_uninitialized_dynamic(self):
        if False:
            print('Hello World!')

        class UninitializedModel(torch.nn.Module):

            def forward(self, y):
                if False:
                    while True:
                        i = 10
                if y.shape[1] < 5:
                    if y.size(0) == 1:
                        y = y + 4
                    else:
                        return y
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((6, 7), dtype=torch.int)
        self.run_test(UninitializedModel(), x, additional_test_inputs=[y], input_names=['input_1'], dynamic_axes={'input_1': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList(self):
        if False:
            while True:
                i = 10

        class UninitializedTensorListModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        return [x]
                return [x]
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(torch.jit.script(UninitializedTensorListModel()), x)

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_dynamic(self):
        if False:
            for i in range(10):
                print('nop')

        class UninitializedTensorListModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                if x[0].shape[0] < 5:
                    if x.size(0) == 1:
                        x += x
                    else:
                        return list(x)
                return list(x)
        x = torch.ones((3, 4), dtype=torch.double)
        self.run_test(torch.jit.script(UninitializedTensorListModel()), x, input_names=['input_1'], dynamic_axes={'input_1': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_intList(self):
        if False:
            while True:
                i = 10

        class UninitializedListModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                y = list(range(x.size(0)))
                if y[0] < 5:
                    if x.size(0) == 3:
                        y.append(10)
                    else:
                        return y
                return y
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(torch.jit.script(UninitializedListModel()), x, input_names=['input_1'], dynamic_axes={'input_1': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(14)
    def test_uninitialized_tensorList_shape(self):
        if False:
            while True:
                i = 10

        class UninitializedModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                if x.shape[1] < 5:
                    if x.size(0) == 1:
                        x = x + 4
                    else:
                        x_list = list(x)
                        x_list.append(x)
                        return x_list
                return [x, x]
        x = torch.ones((3, 4), dtype=torch.int)
        y = torch.ones((4, 6), dtype=torch.int)
        self.run_test(torch.jit.script(UninitializedModel()), x, additional_test_inputs=[y], input_names=['input_1'], dynamic_axes={'input_1': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequance_loopcarried(self):
        if False:
            while True:
                i = 10

        class SequanceLoopModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                outputs = []
                for i in range(3):
                    outputs += [x]
                return torch.stack(outputs).transpose(0, 1)
        x = torch.ones((3, 4), dtype=torch.int)
        self.run_test(torch.jit.script(SequanceLoopModel()), x)

    def test_reflection_pad(self):
        if False:
            while True:
                i = 10
        model = torch.nn.ReflectionPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)
        model = torch.nn.ReflectionPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    def test_replication_pad(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.ReplicationPad1d(2)
        x = torch.randn(2, 4, 4)
        self.run_test(model, x)
        model = torch.nn.ReplicationPad2d((3, 0, 2, 1))
        x = torch.randn(2, 2, 4, 4)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_im2col(self):
        if False:
            print('Hello World!')

        class Unfold(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return (torch.nn.functional.unfold(input, kernel_size=(10, 15), dilation=2, padding=5, stride=3), torch.nn.functional.unfold(input, kernel_size=(2, 2), dilation=1, padding=0, stride=3), torch.nn.functional.unfold(input, kernel_size=(1, 1), dilation=5, padding=2, stride=3))
        x = torch.rand(1, 1, 200, 100)
        self.run_test(Unfold(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_det(self):
        if False:
            i = 10
            return i + 15

        class Det(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.linalg.det(x)
        x = torch.randn(2, 3, 5, 5)
        self.run_test(Det(), x)

    def test_linalg_norm(self):
        if False:
            while True:
                i = 10

        class LinalgSingleDimModel(torch.nn.Module):

            def __init__(self, ord_val):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.linalg.norm(x, ord=self.ord, dim=1)
        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgSingleDimModel(None), x)
        self.run_test(LinalgSingleDimModel(2), x)
        self.run_test(LinalgSingleDimModel(float('inf')), x)
        self.run_test(LinalgSingleDimModel(-float('inf')), x)
        self.run_test(LinalgSingleDimModel(-4), x)
        self.run_test(LinalgSingleDimModel(1.5), x)

        class LinalgMultiDimModel(torch.nn.Module):

            def __init__(self, ord_val):
                if False:
                    print('Hello World!')
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.linalg.norm(x, ord=self.ord, dim=(0, 2))
        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgMultiDimModel('fro'), x)
        self.run_test(LinalgMultiDimModel(float('inf')), x)
        self.run_test(LinalgMultiDimModel(-float('inf')), x)
        self.run_test(LinalgMultiDimModel(1), x)
        self.run_test(LinalgMultiDimModel(-1), x)

        class LinalgNoDimNoOrdModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.linalg.norm(x)
        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgNoDimNoOrdModel(), x)
        y = torch.randn(2, 3)
        self.run_test(LinalgNoDimNoOrdModel(), y)
        z = torch.randn(2)
        self.run_test(LinalgNoDimNoOrdModel(), z)

        class LinalgNoDim1DModel(torch.nn.Module):

            def __init__(self, ord_val):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.linalg.norm(x, ord=self.ord)
        x = torch.randn(2)
        self.run_test(LinalgNoDim1DModel(None), x)
        self.run_test(LinalgNoDim1DModel(2), x)
        self.run_test(LinalgNoDim1DModel(float('inf')), x)
        self.run_test(LinalgNoDim1DModel(-float('inf')), x)
        self.run_test(LinalgNoDim1DModel(-4), x)
        self.run_test(LinalgNoDim1DModel(1.5), x)

        class LinalgNoDim2DModel(torch.nn.Module):

            def __init__(self, ord_val):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.linalg.norm(x, ord=self.ord)
        x = torch.randn(2, 3)
        self.run_test(LinalgNoDim2DModel('fro'), x)
        self.run_test(LinalgNoDim2DModel(float('inf')), x)
        self.run_test(LinalgNoDim2DModel(-float('inf')), x)
        self.run_test(LinalgNoDim2DModel(1), x)
        self.run_test(LinalgNoDim2DModel(-1), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_linalg_vector_norm_zero(self):
        if False:
            i = 10
            return i + 15

        class LinalgVectorNormModel(torch.nn.Module):

            def __init__(self, ord_val):
                if False:
                    print('Hello World!')
                super().__init__()
                self.ord = ord_val

            def forward(self, x):
                if False:
                    return 10
                return torch.linalg.vector_norm(x, ord=self.ord)
        x = torch.randn(2, 3, 5, 5)
        self.run_test(LinalgVectorNormModel(0), x)

    def test_linalg_vector_norm(self):
        if False:
            while True:
                i = 10

        class LinalgVectorNormModel(torch.nn.Module):

            def __init__(self, ord_val, dim_info):
                if False:
                    print('Hello World!')
                super().__init__()
                self.ord = ord_val
                (self.dim, self.keepdim) = dim_info

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.linalg.vector_norm(x, ord=self.ord, dim=self.dim, keepdim=self.keepdim)
        x = torch.randn(2, 3, 5, 5)
        ord_options = [2, float('inf'), -float('inf'), -4, 1.5]
        dim_options = [(None, False), (1, False), ((1, 2), False), ((1, 2), True)]
        for ord_val in ord_options:
            for dim_info in dim_options:
                self.run_test(LinalgVectorNormModel(ord_val, dim_info), x)

    def test_linalg_matrix_norm(self):
        if False:
            while True:
                i = 10

        class LinalgMatrixNormModel(torch.nn.Module):

            def __init__(self, ord_val, dim_val=(-2, -1), keepdim_val=False):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.ord = ord_val
                self.dim = dim_val
                self.keepdim = keepdim_val

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.linalg.matrix_norm(x, ord=self.ord, dim=self.dim, keepdim=self.keepdim)
        x = torch.randn(2, 3, 5, 5)
        ord_options = ['fro', float('inf'), -float('inf'), 1, -1]
        for ord_val in ord_options:
            self.run_test(LinalgMatrixNormModel(ord_val), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2)), x)
            self.run_test(LinalgMatrixNormModel(ord_val, (0, 2), True), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_linalg_cross(self):
        if False:
            i = 10
            return i + 15

        class Cross(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return (torch.linalg.cross(x, y, dim=1), torch.linalg.cross(x, y))
        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(1, 3, 1, 3)
        self.run_test(Cross(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_trace_script(self):
        if False:
            return 10

        @torch.jit.script
        def center_slice_helper(input, h_offset):
            if False:
                return 10
            return input[:, h_offset:]

        class CenterCrop(torch.nn.Module):

            def forward(self, input):
                if False:
                    for i in range(10):
                        print('nop')
                return center_slice_helper(input, torch.tensor(input.shape[1] - 1))
        x = torch.randn(3, 4)
        self.run_test(CenterCrop(), x)

    @skipIfNoLapack
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_logdet(self):
        if False:
            i = 10
            return i + 15

        class LogDet(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.logdet(x)
        x = torch.randn(2, 3, 5, 5)
        self.run_test(LogDet(), x)

    def test_dim(self):
        if False:
            for i in range(10):
                print('nop')

        class DimModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    while True:
                        i = 10
                out = input * 2
                out *= out.dim()
                return out
        empty_input = torch.randn(0, requires_grad=True)
        multi_dim_input = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(DimModel(), empty_input)
        self.run_test(DimModel(), multi_dim_input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dim_1(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, poses):
                if False:
                    for i in range(10):
                        print('nop')
                boxes = torch.zeros([poses.shape[0], 2, 4])
                batch_boxes = []
                for kp_boxes in boxes:
                    kp_boxes = torchvision.ops.clip_boxes_to_image(kp_boxes, (2, 3))
                    batch_boxes.append(kp_boxes)
                return batch_boxes
        dummy_inputs = torch.rand(2, 2, 3)
        self.run_test(M(), (dummy_inputs,), input_names=['x'], dynamic_axes={'x': [0]})

    @skipIfUnsupportedMinOpsetVersion(12)
    @skipDtypeChecking
    def test_outer(self):
        if False:
            return 10

        class Outer(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return torch.outer(x, y)
        x = torch.arange(1, 5)
        y = torch.arange(1, 4)
        self.run_test(Outer(), input_args=(x, y))
        x = torch.arange(1, 6).to(dtype=torch.float32)
        y = torch.arange(1, 4).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))
        x = torch.arange(2, 5).to(dtype=torch.float32)
        y = torch.arange(2, 4).to(dtype=torch.float64)
        self.run_test(Outer(), input_args=(x, y))
        x = torch.arange(3, 6).to(dtype=torch.int32)
        y = torch.arange(4, 7).to(dtype=torch.long)
        self.run_test(Outer(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_movedim(self):
        if False:
            i = 10
            return i + 15

        class MovedimModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (x.movedim(1, 3), x.movedim(2, 0), x.movedim(1, 1), x.movedim((1, 2, 3), (3, 0, 1)), x.movedim((0, 1, 2), (1, 2, 3)), x.movedim((1, 3, 2), (1, 3, 2)))
        x = torch.randn(5, 3, 4, 2)
        self.run_test(MovedimModel(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_moveaxis(self):
        if False:
            i = 10
            return i + 15

        class MoveaxisModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return (x.moveaxis(1, 3), x.moveaxis(2, 0), x.moveaxis(1, 1), x.moveaxis((1, 2, 3), (3, 0, 1)), x.moveaxis((0, 1, 2), (1, 2, 3)), x.moveaxis((1, 3, 2), (1, 3, 2)))
        x = torch.randn(5, 3, 4, 2)
        self.run_test(MoveaxisModel(), x)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_einsum(self):
        if False:
            for i in range(10):
                print('nop')

        class EinsumModelBatchDiagonal(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                eqn = '...ii ->...i'
                return torch.einsum(eqn, x)
        for x in [torch.randn(3, 5, 5), torch.randn(3, 5, 5).to(dtype=torch.bool)]:
            self.run_test(EinsumModelBatchDiagonal(), input_args=(x,))

        class EinsumModelBatchMatmul(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                eqn = 'bij, bjk -> bik'
                return torch.einsum(eqn, x, y)
        x = torch.randn(5, 2, 3)
        y = torch.randn(5, 3, 4)
        self.run_test(EinsumModelBatchMatmul(), input_args=(x, y))

        class EinsumModelInnerProd(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                eqn = 'i,i'
                return torch.einsum(eqn, x, y)
        x = torch.randn(5)
        y = torch.randn(5)
        self.run_test(EinsumModelInnerProd(), input_args=(x, y))

        class EinsumModelTranspose(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                eqn = 'ij->ji'
                return torch.einsum(eqn, x)
        for x in [torch.randn(3, 4), torch.randn(3, 4).to(dtype=torch.bool)]:
            self.run_test(EinsumModelTranspose(), input_args=(x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cosine_similarity(self):
        if False:
            print('Hello World!')
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        self.run_test(torch.nn.CosineSimilarity(dim=2), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_pairwise_distance(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(5, 3, 2)
        y = torch.randn(5, 3, 2)
        self.run_test(torch.nn.PairwiseDistance(p=2.0), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cross(self):
        if False:
            for i in range(10):
                print('nop')

        class Cross(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return (torch.cross(x, y, dim=3), torch.cross(x, y))
        x = torch.randn(5, 3, 2, 3)
        y = torch.randn(5, 3, 2, 3)
        self.run_test(Cross(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_cdist(self):
        if False:
            i = 10
            return i + 15

        class Cdist(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.cdist(x, y)
        x = torch.randn(5, 3, 3)
        y = torch.randn(5, 2, 3)
        self.run_test(Cdist(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_crossentropyloss(self):
        if False:
            while True:
                i = 10
        for ignore_index in [-100, 1]:
            x = torch.randn(3, 5)
            y = torch.empty(3, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)
            x = torch.randn(3, 5, 2)
            y = torch.empty(3, 2, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)
            x = torch.randn(3, 5, 2, 7)
            y = torch.empty(3, 2, 7, dtype=torch.long).random_(5)
            y[y == 1] = ignore_index
            self._crossentropyloss(x, y, ignore_index)

    def _crossentropyloss(self, x, y, ignore_index):
        if False:
            return 10

        class CrossEntropyLossNone(torch.nn.Module):

            def __init__(self, ignore_index):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none')
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=ignore_index)

            def forward(self, input, target):
                if False:
                    i = 10
                    return i + 15
                return self.loss(input, target)
        self.run_test(CrossEntropyLossNone(ignore_index), input_args=(x, y))

        class CrossEntropyLossNoneWeight(torch.nn.Module):

            def __init__(self, ignore_index):
                if False:
                    return 10
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='none', weight=torch.randn(5), ignore_index=ignore_index)

            def forward(self, input, target):
                if False:
                    i = 10
                    return i + 15
                return self.loss(input, target)
        self.run_test(CrossEntropyLossNoneWeight(ignore_index), input_args=(x, y))

        class CrossEntropyLossSum(torch.nn.Module):

            def __init__(self, ignore_index):
                if False:
                    print('Hello World!')
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum')
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=ignore_index)

            def forward(self, input, target):
                if False:
                    i = 10
                    return i + 15
                return self.loss(input, target)
        self.run_test(CrossEntropyLossSum(ignore_index), input_args=(x, y))

        class CrossEntropyLossSumWeight(torch.nn.Module):

            def __init__(self, ignore_index):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum', weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(reduction='sum', weight=torch.randn(5), ignore_index=ignore_index)

            def forward(self, input, target):
                if False:
                    for i in range(10):
                        print('nop')
                return self.loss(input, target)
        self.run_test(CrossEntropyLossSumWeight(ignore_index), input_args=(x, y))

        class CrossEntropyLossMean(torch.nn.Module):

            def __init__(self, ignore_index):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss()
                else:
                    self.loss = torch.nn.CrossEntropyLoss(ignore_index=ignore_index)

            def forward(self, input, target):
                if False:
                    return 10
                return self.loss(input, target)
        self.run_test(CrossEntropyLossMean(ignore_index), input_args=(x, y))

        class CrossEntropyLossMeanWeight(torch.nn.Module):

            def __init__(self, ignore_index):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                if ignore_index == -100:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.randn(5))
                else:
                    self.loss = torch.nn.CrossEntropyLoss(weight=torch.randn(5), ignore_index=ignore_index)

            def forward(self, input, target):
                if False:
                    i = 10
                    return i + 15
                return self.loss(input, target)
        self.run_test(CrossEntropyLossMeanWeight(ignore_index), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_MSELoss(self):
        if False:
            return 10

        class MSELoss(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.loss1 = torch.nn.MSELoss(reduction='none')
                self.loss2 = torch.nn.MSELoss(reduction='sum')
                self.loss3 = torch.nn.MSELoss(reduction='mean')

            def forward(self, input, target):
                if False:
                    print('Hello World!')
                return (self.loss1(input, target), self.loss2(input, target), self.loss3(input, target))
        x = torch.randn(2, 3, 5)
        y = torch.randn(2, 3, 5)
        self.run_test(MSELoss(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_kldiv_loss(self):
        if False:
            return 10
        x = torch.rand(5).log()
        y = torch.rand(5)
        self._kldiv_loss(x, y)
        x = torch.rand(2, 3, 5).log()
        y = torch.rand(2, 3, 5)
        self._kldiv_loss(x, y)
        x = torch.rand(2, 3, 5, 7).log()
        y = torch.rand(2, 3, 5, 7)
        self._kldiv_loss(x, y)

    def _kldiv_loss(self, x, y):
        if False:
            print('Hello World!')

        class KLDivLossNone(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction='none', log_target=True)

            def forward(self, input, target):
                if False:
                    return 10
                return self.loss(input, target.log())
        self.run_test(KLDivLossNone(), input_args=(x, y))

        class KLDivLossMean(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction='mean', log_target=False)

            def forward(self, input, target):
                if False:
                    for i in range(10):
                        print('nop')
                return self.loss(input, target)
        self.run_test(KLDivLossMean(), input_args=(x, y))

        class KLDivLossSum(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction='sum', log_target=True)

            def forward(self, input, target):
                if False:
                    return 10
                return self.loss(input, target.log())
        self.run_test(KLDivLossSum(), input_args=(x, y))

        class KLDivLossBatchMean(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction='batchmean', log_target=False)

            def forward(self, input, target):
                if False:
                    for i in range(10):
                        print('nop')
                return self.loss(input, target)
        self.run_test(KLDivLossBatchMean(), input_args=(x, y))

        class KLDivLossMiniBatchMean(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.loss = torch.nn.KLDivLoss(reduction='batchmean', size_average=False, log_target=True)

            def forward(self, input, target):
                if False:
                    while True:
                        i = 10
                return self.loss(input, target.log())
        self.run_test(KLDivLossMiniBatchMean(), input_args=(x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss(self):
        if False:
            while True:
                i = 10

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='none')
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    i = 10
                    return i + 15
                output = self.loss(self.m(2 * input), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16)
        target = torch.empty(N, dtype=torch.long).random_(0, C)
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_none(self):
        if False:
            print('Hello World!')

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='none')
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    while True:
                        i = 10
                output = self.loss(self.m(self.conv(input)), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean(self):
        if False:
            print('Hello World!')

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean')
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    return 10
                output = self.loss(self.m(self.conv(input)), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_sum(self):
        if False:
            for i in range(10):
                print('nop')

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='sum')
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    while True:
                        i = 10
                output = self.loss(self.m(self.conv(input)), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_weights(self):
        if False:
            i = 10
            return i + 15

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean', weight=torch.randn(C))
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    print('Hello World!')
                output = self.loss(self.m(self.conv(input)), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        target[target == 1] = -100
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index(self):
        if False:
            return 10

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean', ignore_index=1)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    for i in range(10):
                        print('nop')
                output = self.loss(self.m(self.conv(input)), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_dynamic_ignore_index(self):
        if False:
            i = 10
            return i + 15
        import torch.nn.functional as F

        def linear_combination(x, y, epsilon):
            if False:
                return 10
            return epsilon * x + (1 - epsilon) * y

        def reduce_loss(loss, reduction='mean'):
            if False:
                while True:
                    i = 10
            return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

        class LabelSmoothingCrossEntropy(torch.nn.Module):

            def __init__(self, epsilon: float=0.1, reduction='mean'):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.epsilon = epsilon
                self.reduction = reduction

            def forward(self, preds, target, start_position):
                if False:
                    while True:
                        i = 10
                n = preds.size()[-1]
                log_preds = F.log_softmax(preds, dim=-1)
                ignore_index = start_position.size(1)
                nll = F.nll_loss(log_preds, target, reduction=self.reduction, ignore_index=ignore_index)
                return nll + start_position.float()
        N = 5
        preds = torch.randn(N, 16)
        target = torch.randint(5, (N,))
        start_position = torch.randint(10, (N, N))
        self.run_test(LabelSmoothingCrossEntropy(), (preds, target, start_position))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_nllloss_2d_mean_ignore_index_weights(self):
        if False:
            i = 10
            return i + 15

        class NLLModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.loss = torch.nn.NLLLoss(reduction='mean', weight=torch.randn(C), ignore_index=1)
                self.conv = torch.nn.Conv2d(16, C, (3, 3))
                self.m = torch.nn.LogSoftmax(dim=1)

            def forward(self, input, target):
                if False:
                    for i in range(10):
                        print('nop')
                output = self.loss(self.m(self.conv(input)), target)
                return output
        (N, C) = (5, 4)
        input = torch.randn(N, 16, 10, 10)
        target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)
        self.run_test(NLLModel(), (input, target))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_binary_cross_entropy_with_logits(self):
        if False:
            i = 10
            return i + 15
        x = torch.randn(5)
        y = torch.empty(5).random_(2)
        self._bce_logits(x, y)
        x = torch.randn(3, 4)
        y = torch.empty(3, 4).random_(2)
        weight = torch.tensor([3])
        self._bce_logits_wegiht(x, y, weight)
        x = torch.randn(3, 2, 4)
        y = torch.empty(3, 2, 4).random_(2)
        pos_weight = torch.empty([2, 4]).random_(2)
        self._bce_logits_posweight(x, y, pos_weight)
        x = torch.randn(3, 3, 4)
        y = torch.empty(3, 3, 4).random_(2)
        weight = torch.tensor([3])
        pos_weight = torch.empty([3, 4]).random_(2)
        self._bce_logits_loss_weight_posweight(x, y, weight, pos_weight)

    def _bce_logits(self, x, y):
        if False:
            for i in range(10):
                print('nop')

        class BCEWithLogitsLossNone(torch.nn.Module):

            def forward(self, input, target):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction='none')
        self.run_test(BCEWithLogitsLossNone(), input_args=(x, y))

        class BCEWithLogitsLossMean(torch.nn.Module):

            def forward(self, input, target):
                if False:
                    return 10
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction='mean')
        self.run_test(BCEWithLogitsLossMean(), input_args=(x, y))

        class BCEWithLogitsLossSum(torch.nn.Module):

            def forward(self, input, target):
                if False:
                    i = 10
                    return i + 15
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, reduction='sum')
        self.run_test(BCEWithLogitsLossSum(), input_args=(x, y))

    def _bce_logits_wegiht(self, x, y, weight):
        if False:
            return 10

        class BCEWithLogitsLossWegihtNone(torch.nn.Module):

            def forward(self, input, target, weight):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='none')
        self.run_test(BCEWithLogitsLossWegihtNone(), input_args=(x, y, weight))

        class BCEWithLogitsLossWegihtMean(torch.nn.Module):

            def forward(self, input, target, weight):
                if False:
                    i = 10
                    return i + 15
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='mean')
        self.run_test(BCEWithLogitsLossWegihtMean(), input_args=(x, y, weight))

        class BCEWithLogitsLossWegihtSum(torch.nn.Module):

            def forward(self, input, target, weight):
                if False:
                    i = 10
                    return i + 15
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, reduction='sum')
        self.run_test(BCEWithLogitsLossWegihtSum(), input_args=(x, y, weight))

    def _bce_logits_posweight(self, x, y, pos_weight):
        if False:
            for i in range(10):
                print('nop')

        class BCEWithLogitsLossPosWegihtNone(torch.nn.Module):

            def forward(self, input, target, pos_weight):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='none')
        self.run_test(BCEWithLogitsLossPosWegihtNone(), input_args=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtMean(torch.nn.Module):

            def forward(self, input, target, pos_weight):
                if False:
                    print('Hello World!')
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='mean')
        self.run_test(BCEWithLogitsLossPosWegihtMean(), input_args=(x, y, pos_weight))

        class BCEWithLogitsLossPosWegihtSum(torch.nn.Module):

            def forward(self, input, target, pos_weight):
                if False:
                    print('Hello World!')
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, pos_weight=pos_weight, reduction='sum')
        self.run_test(BCEWithLogitsLossPosWegihtSum(), input_args=(x, y, pos_weight))

    def _bce_logits_loss_weight_posweight(self, x, y, weight, pos_weight):
        if False:
            while True:
                i = 10

        class BCEWithLogitsLossWeightPosweightNone(torch.nn.Module):

            def forward(self, input, target, weight, pos_weight):
                if False:
                    return 10
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, pos_weight=pos_weight, reduction='none')
        self.run_test(BCEWithLogitsLossWeightPosweightNone(), input_args=(x, y, weight, pos_weight))

        class BCEWithLogitsLossWeightPosweightMean(torch.nn.Module):

            def forward(self, input, target, weight, pos_weight):
                if False:
                    print('Hello World!')
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, pos_weight=pos_weight, reduction='mean')
        self.run_test(BCEWithLogitsLossWeightPosweightMean(), input_args=(x, y, weight, pos_weight))

        class BCEWithLogitsLossWeightPosweightSum(torch.nn.Module):

            def forward(self, input, target, weight, pos_weight):
                if False:
                    return 10
                return torch.nn.functional.binary_cross_entropy_with_logits(input, target, weight=weight, pos_weight=pos_weight, reduction='sum')
        self.run_test(BCEWithLogitsLossWeightPosweightSum(), input_args=(x, y, weight, pos_weight))

    def test_torch_mm(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, mat1, mat2):
                if False:
                    while True:
                        i = 10
                mm = torch.mm(mat1, mat2)
                return mm
        mat1 = torch.randn(2, 3)
        mat2 = torch.randn(3, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where_with_bool_tensor(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, mat1, mat2):
                if False:
                    while True:
                        i = 10
                out = torch.where(mat1 > 0, mat1, mat2)
                return out
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input_args=(mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where_with_byte_tensor(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, cond, mat1, mat2):
                if False:
                    for i in range(10):
                        print('nop')
                out = torch.where(cond, mat1, mat2)
                return out
        cond = torch.ones(2, 3, dtype=torch.uint8)
        cond[1, 2] = 0
        mat1 = torch.randn(2, 3)
        mat2 = torch.ones(2, 3)
        self.run_test(M(), input_args=(cond, mat1, mat2))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_isinf(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.isinf()
        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), float('inf')]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_isfinite(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.isfinite()
        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), -float('inf')]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_isnan(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.isnan()
        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), float('inf')]])
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_nan_to_num(self):
        if False:
            for i in range(10):
                print('nop')

        class NoParams(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.nan_to_num()
        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), -float('inf')]])
        xint = torch.ones((2, 4), dtype=torch.int)
        xhalf = torch.ones((2, 4), dtype=torch.half)
        self.run_test(NoParams(), (x,))
        self.run_test(NoParams(), (xint,))
        self.run_test(NoParams(), (xhalf,))

        class WithParams(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.nan_to_num(nan=2.3, posinf=4.5, neginf=6.7)
        x = torch.tensor([[1, 2, float('inf')], [2, float('nan'), -float('inf')]])
        self.run_test(WithParams(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_maximum_minimum(self):
        if False:
            print('Hello World!')

        class ModelWithNan(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return (torch.maximum(x, y), torch.minimum(x, y))
        x = torch.tensor([-2, -2, float('nan')])
        y = torch.rand(1, 3)
        self.run_test(ModelWithNan(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_minimum_dtypes(self):
        if False:
            return 10

        class MinimumModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.minimum(x, y)
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MinimumModel(), (x, y))
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MinimumModel(), (x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MinimumModel(), (x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MinimumModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_maximum_dtypes(self):
        if False:
            for i in range(10):
                print('nop')

        class MaximumModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.maximum(x, y)
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randn((5, 5), dtype=torch.float)
        self.run_test(MaximumModel(), (x, y))
        x = torch.randn((5, 5), dtype=torch.float16)
        y = torch.randint(10, (5, 5), dtype=torch.int16)
        self.run_test(MaximumModel(), (x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int16)
        y = torch.randint(10, (5, 5), dtype=torch.int32)
        self.run_test(MaximumModel(), (x, y))
        x = torch.randint(10, (5, 5), dtype=torch.int)
        y = torch.full_like(x, True)
        self.run_test(MaximumModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_any(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.any()
        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.any(dim=1)
        x = torch.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.any(dim=1, keepdim=True)
        x = torch.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_all(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x.all()
        x = torch.tensor([[True, False], [False, False]])
        self.run_test(M(), (x,))

        class MDim(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x.all(dim=1)
        x = torch.rand(3, 4).bool()
        self.run_test(MDim(), (x,))

        class MKeepdim(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.all(dim=1, keepdim=True)
        x = torch.rand(3, 4).bool()
        self.run_test(MKeepdim(), (x,))

    def test_dropout(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.dropout = torch.nn.Dropout(0.3)

            def forward(self, x):
                if False:
                    print('Hello World!')
                dropout = self.dropout(x)
                return dropout
        x = torch.randn(10, 3, 53)
        self.run_test(M(), x)

    def test_rrelu_eval(self):
        if False:
            while True:
                i = 10
        x = torch.tensor([0.5, -0.5])
        self.run_test(torch.nn.RReLU(0.1, 0.3).eval(), x)

    def test_shape_constant_fold(self):
        if False:
            i = 10
            return i + 15

        class ShapeModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.register_buffer('weight', torch.ones(5))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                shape = self.weight.shape[0]
                return x + shape
        x = torch.randn(2, 5)
        self.run_test(ShapeModule(), (x,), rtol=0.001, atol=1e-05)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu(self):
        if False:
            print('Hello World!')

        class Celu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.celu = torch.nn.CELU(alpha=1.0)

            def forward(self, input):
                if False:
                    return 10
                return self.celu(input)
        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_default(self):
        if False:
            return 10

        class Celu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.celu = torch.nn.CELU()

            def forward(self, input):
                if False:
                    return 10
                return self.celu(input)
        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_alpha(self):
        if False:
            i = 10
            return i + 15

        class Celu(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.celu = torch.nn.CELU(alpha=2.0)

            def forward(self, input):
                if False:
                    return 10
                return self.celu(input)
        input = torch.randn(2)
        self.run_test(Celu(), (input,))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_celu_cast(self):
        if False:
            i = 10
            return i + 15

        class Celu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.celu = torch.nn.CELU()

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return self.celu(input)
        input = torch.randn(2, 5, 7, dtype=torch.float64)
        self.run_test(Celu(), (input,))

    def test_lower_tuple(self):
        if False:
            return 10

        class TupleModule(torch.nn.Module):

            def forward(self, input1: Tensor, input2: Tensor, input3: Tensor) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                a = (input1, input2)
                b = a
                c = (input1, input2, input3)
                for i in range(5):
                    d = a[0]
                    for j in range(2):
                        (e, f) = a
                        a = (d, f)
                        f = c[2]
                        if f.size(0) != input1.size(-1):
                            g = b[1]
                            b = (g, f)
                        else:
                            k = c[1:]
                            b = (f, k[0])
                    (m, n) = b
                    c = (input1, n, m)
                (p, q, r) = c
                return p + q + r
        input1 = torch.randn(2)
        input2 = torch.randn(2)
        input3 = torch.randn(2)
        self.run_test(TupleModule(), (input1, input2, input3))

    def test_lower_tuple_2(self):
        if False:
            for i in range(10):
                print('nop')

        class TupleModule(torch.nn.Module):

            def forward(self, input1: Tensor, input2: Tensor) -> Tuple[Tensor, Tensor]:
                if False:
                    i = 10
                    return i + 15
                a = (input1, input2)
                for x in range(5):
                    (c, d) = a
                    a = (c, d)
                return a
        input1 = torch.randn(2)
        input2 = torch.randn(2)
        self.run_test(TupleModule(), (input1, input2))

    def test_lower_tuple_3(self):
        if False:
            for i in range(10):
                print('nop')

        class TupleModule(torch.nn.Module):

            def forward(self, input1: Tuple[Tensor, Tensor], input2: Tuple[Tensor, Tensor]) -> Tuple[Tuple[Tensor, Tensor], Tuple[Tensor, Tensor]]:
                if False:
                    print('Hello World!')
                a = input1
                b = input2
                for x in range(5):
                    (c, d) = a
                    (e, f) = b
                    if c.shape[0] == e.shape[0]:
                        e = e + c
                    else:
                        f = f + d
                    a = (e, f)
                    b = (c, d)
                return (a, b)
        input1 = (torch.randn(2), torch.randn(2))
        input2 = (torch.randn(2), torch.randn(2))
        self.run_test(TupleModule(), (input1, input2))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_where(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def forward(self, cond, input, other):
                if False:
                    i = 10
                    return i + 15
                return torch.where(cond, input, other)
        x = torch.randint(0, 1, (2, 3, 4), dtype=torch.bool)
        y = torch.randn(2, 1, 4)
        z = torch.ones(2, 3, 1)
        self.run_test(Model(), (x, y, z))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_where_condition(self):
        if False:
            while True:
                i = 10

        class Model1(torch.nn.Module):

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.stack(torch.where(input > 0.5), dim=1)
        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), x)

        class Model2(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    return 10
                return torch.stack(torch.where(input > other), dim=1)
        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    @skipIfUnsupportedOpsetVersion([13])
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_where_condition_script(self):
        if False:
            while True:
                i = 10

        class Model1(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return torch.stack(torch.where(input > 0.5), dim=1)
        x = torch.randint(0, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model1(), x)

        class Model2(torch.nn.Module):

            def forward(self, input, other):
                if False:
                    while True:
                        i = 10
                return torch.stack(torch.where(input > other), dim=1)
        x = torch.randint(0, 1, (2, 3, 4), dtype=bool)
        y = torch.randint(1, 2, (2, 3, 4), dtype=bool)
        self.run_test(Model2(), (x, y))

    def test_empty_branch(self):
        if False:
            i = 10
            return i + 15

        class EmptyBranchModel(torch.jit.ScriptModule):

            @torch.jit.script_method
            def forward(self, input):
                if False:
                    print('Hello World!')
                out = input + 1
                if out.dim() > 2:
                    if out.dim() > 3:
                        out += 3
                    else:
                        pass
                else:
                    pass
                return out
        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(EmptyBranchModel(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_derive_index_scripting(self):
        if False:
            print('Hello World!')

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    print('Hello World!')
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j
        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j
        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    i = 10
                    return i + 15
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j
        self.run_test(MyModule(), x)

    @skipScriptTest()
    def test_derive_index(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    i = 10
                    return i + 15
                j = []
                for idx in range(len(x) - 1, -len(x), -2):
                    y = x[idx]
                    j += [x * y]
                return j
        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    i = 10
                    return i + 15
                j = []
                for idx in range(-len(x), len(x) - 1, 2):
                    y = x[idx]
                    j += [x * y]
                return j
        x = torch.randn(5, 13)
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    print('Hello World!')
                j = []
                for idx in range(len(x) - 1, -len(x), -3):
                    y = x[idx]
                    j += [x * y]
                return j
        self.run_test(MyModule(), x)

        class MyModule(torch.nn.Module):

            def forward(self, x: Tensor):
                if False:
                    i = 10
                    return i + 15
                j = []
                for idx in range(-len(x), len(x) - 1, 3):
                    y = x[idx]
                    j += [x * y]
                return j
        self.run_test(MyModule(), x)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_if_transpose(self):
        if False:
            i = 10
            return i + 15

        class IfModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x.transpose(0, 1)
                if x.size(0) == 2:
                    return x.transpose(0, 1)
                else:
                    return x
        x = torch.randn(2, 3)
        self.run_test(torch.jit.script(IfModel()), x, output_names=['output_1'], dynamic_axes={'output_1': [0, 1]})

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_list(self):
        if False:
            i = 10
            return i + 15

        class IfModel(torch.nn.Module):

            def forward(self, x, y, cond):
                if False:
                    while True:
                        i = 10
                res = []
                if cond:
                    res = res + [x]
                else:
                    res = res + [y]
                return res
        x = torch.randn(2, 3)
        y = torch.randn(3, 3)
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(torch.jit.script(IfModel()), (x, y, cond))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_if_view(self):
        if False:
            while True:
                i = 10

        class IfModel(torch.nn.Module):

            def forward(self, x, y, cond):
                if False:
                    print('Hello World!')
                (bs, seq) = y.shape[:2]
                if cond:
                    res = x.view(bs, seq, -1)
                else:
                    res = y
                return res.transpose(1, 2)
        x = torch.randn(2, 16, 2, 2)
        y = torch.randn(2, 16, 8)
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(torch.jit.script(IfModel()), (x, y, cond), output_names=['output_1'], dynamic_axes={'output_1': [1]})

    @skipScriptTest(skip_before_opset_version=11, reason='dynamic split support added in 11')
    def test_split_tensor_scalar(self):
        if False:
            print('Hello World!')

        class SplitModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.split(x, x.size(1))
        x = torch.randn(1, 2, 3, requires_grad=True)
        self.run_test(SplitModel(), x)

    def test_split_tensor_multi(self):
        if False:
            while True:
                i = 10

        class SplitModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.split(x, torch.ones(3))
        x = torch.randn(1, 2, 3, requires_grad=True)

        def run_model():
            if False:
                print('Hello World!')
            SplitModel(x)
        self.assertRaises(TypeError, run_model)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding(self):
        if False:
            print('Hello World!')

        class EmbedModel(torch.nn.Module):

            def forward(self, input, emb):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.embedding(input, emb, padding_idx=1)
        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        embedding_matrix = torch.rand(10, 3)
        self.run_test(model, (x, embedding_matrix))
        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x, embedding_matrix))
        self.run_test(model, (x, embedding_matrix), training=torch.onnx.TrainingMode.TRAINING)

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):

            def forward(self, input, emb):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.embedding(input, emb)
        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x, embedding_matrix))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_embedding_module(self):
        if False:
            for i in range(10):
                print('nop')

        class EmbedModel(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3, padding_idx=1)
                self.emb2 = torch.nn.Embedding(4, 3, padding_idx=1)
                with torch.no_grad():
                    self.emb2.weight[1] = torch.ones(3)

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return (self.emb(input), self.emb2(input))
        model = EmbedModel()
        x = torch.randint(4, (4,))
        x[2] = x[0] = 1
        self.run_test(model, (x,))
        x = torch.randint(4, (4, 3, 2))
        x[2] = 1
        x[0][1] = 1
        self.run_test(model, (x,))

        class EmbedModelWithoutPaddingIdx(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.emb = torch.nn.Embedding(4, 3)

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                return self.emb(input)
        model = EmbedModelWithoutPaddingIdx()
        x = torch.randint(4, (4, 3, 2))
        self.run_test(model, (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_embedding_renorm(self):
        if False:
            print('Hello World!')
        (n, d) = (7, 5)
        embedding = torch.nn.Embedding(n, d, max_norm=0.2)
        idx = torch.tensor([2, 1])
        self.run_test(embedding, idx)
        embedding = torch.nn.Embedding(n, d, max_norm=0.5, norm_type=1.0)
        idx = torch.tensor([4, 3, 4, 2])
        self.run_test(embedding, idx)

    def _dispatch_rnn_test(self, name, *args, **kwargs):
        if False:
            while True:
                i = 10
        if name == 'elman':
            self._elman_rnn_test(*args, **kwargs)
        if name == 'lstm':
            self._lstm_test(*args, **kwargs)
        if name == 'gru':
            self._gru_test(*args, **kwargs)

    def _elman_rnn_test(self, layers, nonlinearity, bidirectional, initial_state, packed_sequence, dropout, **extra_kwargs):
        if False:
            i = 10
            return i + 15

        class ElmanWithStateModel(torch.nn.Module):

            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, nonlinearity=nonlinearity, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input: rnn_utils.PackedSequence, hx=None):
                if False:
                    i = 10
                    return i + 15
                return self.inner_model(input, hx)

        class ElmanWithoutStateModel(torch.nn.Module):

            def __init__(self, layers, nonlinearity, bidirect, dropout, batch_first):
                if False:
                    return 10
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, nonlinearity=nonlinearity, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input: rnn_utils.PackedSequence):
                if False:
                    while True:
                        i = 10
                return self.inner_model(input)
        batch_first = packed_sequence == 2
        if initial_state:
            model = ElmanWithStateModel(layers=layers, bidirect=bidirectional, nonlinearity=nonlinearity, dropout=dropout, batch_first=batch_first)
            if packed_sequence:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(model, batch_first)
        else:
            model = ElmanWithoutStateModel(layers=layers, bidirect=bidirectional, nonlinearity=nonlinearity, dropout=dropout, batch_first=batch_first)
            if packed_sequence:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(model, batch_first)

        def make_input(batch_size):
            if False:
                print('Hello World!')
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ['input']
            directions = 2 if bidirectional else 1
            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
                input_names.append('h0')
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append('seq_lengths')
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return (input, input_names)
        (input, input_names) = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {'input': [0, 1], 'seq_lengths': [0]}
        if initial_state:
            dynamic_axes.update({'h0': [1]})
        export_options = {'input_names': input_names, 'dynamic_axes': dynamic_axes}
        (other_input, _) = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(model, input, additional_test_inputs=[other_input], **export_options)

    def _lstm_test(self, layers, bidirectional, initial_state, packed_sequence, dropout, **extra_kwargs):
        if False:
            for i in range(10):
                print('nop')
        batch_first = packed_sequence == 2
        if packed_sequence:
            model = lstm_flattening_result.LstmFlatteningResultWithSeqLength(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, bidirectional, dropout, batch_first)
            if initial_state:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(model, batch_first)
            else:
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(model, batch_first)
        else:
            model = lstm_flattening_result.LstmFlatteningResultWithoutSeqLength(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, layers, bidirectional, dropout, batch_first)

        def make_input(batch_size):
            if False:
                return 10
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ['input']
            directions = 2 if bidirectional else 1
            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                c0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append((h0, c0))
                input_names.append('h0')
                input_names.append('c0')
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append('seq_lengths')
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return (input, input_names)
        (input, input_names) = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {'input': [0, 1], 'seq_lengths': [0]}
        if initial_state:
            dynamic_axes.update({'h0': [1], 'c0': [1]})
        export_options = {'input_names': input_names, 'dynamic_axes': dynamic_axes}
        (other_input, _) = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(model, input, additional_test_inputs=[other_input], **export_options)

    def _gru_test(self, layers, bidirectional, initial_state, packed_sequence, dropout, **extra_kwargs):
        if False:
            while True:
                i = 10

        class GRUWithStateModel(torch.nn.Module):

            def __init__(self, layers, bidirect, dropout, batch_first):
                if False:
                    print('Hello World!')
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input: rnn_utils.PackedSequence, hx):
                if False:
                    print('Hello World!')
                return self.inner_model(input, hx)

        class GRUWithoutStateModel(torch.nn.Module):

            def __init__(self, layers, bidirect, dropout, batch_first):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input: rnn_utils.PackedSequence):
                if False:
                    i = 10
                    return i + 15
                return self.inner_model(input)

        class GRUNoSeqLengthWithoutStateModel(torch.nn.Module):

            def __init__(self, layers, bidirect, dropout, batch_first):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input):
                if False:
                    i = 10
                    return i + 15
                return self.inner_model(input)

        class GRUNoSeqLengthWithStateModel(torch.nn.Module):

            def __init__(self, layers, bidirect, dropout, batch_first):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.batch_first = batch_first
                self.inner_model = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, num_layers=layers, bidirectional=bidirectional, dropout=dropout, batch_first=batch_first)

            def forward(self, input, hx):
                if False:
                    while True:
                        i = 10
                return self.inner_model(input, hx)
        batch_first = packed_sequence == 2
        if packed_sequence:
            if initial_state:
                model = GRUWithStateModel(layers=layers, bidirect=bidirectional, dropout=dropout, batch_first=batch_first)
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithState(model, batch_first)
            else:
                model = GRUWithoutStateModel(layers=layers, bidirect=bidirectional, dropout=dropout, batch_first=batch_first)
                model = rnn_model_with_packed_sequence.RnnModelWithPackedSequenceWithoutState(model, batch_first)
        elif initial_state:
            model = GRUNoSeqLengthWithStateModel(layers=layers, bidirect=bidirectional, dropout=dropout, batch_first=batch_first)
        else:
            model = GRUNoSeqLengthWithoutStateModel(layers=layers, bidirect=bidirectional, dropout=dropout, batch_first=batch_first)

        def make_input(batch_size):
            if False:
                for i in range(10):
                    print('nop')
            seq_lengths = np.random.randint(1, RNN_SEQUENCE_LENGTH + 1, size=batch_size)
            seq_lengths = sorted(map(int, seq_lengths), reverse=True)
            inputs = [torch.randn(l, RNN_INPUT_SIZE) for l in seq_lengths]
            inputs = rnn_utils.pad_sequence(inputs, batch_first=batch_first)
            inputs = [inputs]
            input_names = ['input']
            directions = 2 if bidirectional else 1
            if initial_state:
                h0 = torch.randn(directions * layers, batch_size, RNN_HIDDEN_SIZE)
                inputs.append(h0)
                input_names.append('h0')
            if packed_sequence != 0:
                inputs.append(torch.IntTensor(seq_lengths))
                input_names.append('seq_lengths')
            if len(inputs) == 1:
                input = inputs[0]
            else:
                input = tuple(inputs)
            return (input, input_names)
        (input, input_names) = make_input(RNN_BATCH_SIZE)
        dynamic_axes = {'input': [0, 1], 'seq_lengths': [0]}
        if initial_state:
            dynamic_axes.update({'h0': [1]})
        export_options = {'input_names': input_names, 'dynamic_axes': dynamic_axes}
        (other_input, _) = make_input(RNN_BATCH_SIZE + 1)
        self.run_test(model, input, additional_test_inputs=[other_input], **export_options)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_fake_quantize_per_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        class FakeQuantizePerTensorModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                scale = 1.0 / 127
                zero_point = 0
                quant_min = -128
                quant_max = 127
                return torch.fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max)
        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerTensorModel(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_tensor_dynamic_scale_zeropoint(self):
        if False:
            for i in range(10):
                print('nop')

        class FakeQuantizePerTensorModel(torch.nn.Module):

            def forward(self, input, scale, zero_point):
                if False:
                    i = 10
                    return i + 15
                quant_min = -128
                quant_max = 127
                return torch.fake_quantize_per_tensor_affine(input, scale, zero_point, quant_min, quant_max)
        x = torch.randn(6, 4, 3, 3)
        scale = torch.tensor(1.0 / 127)
        zero_point = torch.tensor(0)
        self.run_test(FakeQuantizePerTensorModel(), (x, scale, zero_point))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_fake_quantize_per_channel(self):
        if False:
            while True:
                i = 10

        class FakeQuantizePerChannelModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    while True:
                        i = 10
                amax = torch.ones(4)
                scale = amax / 127.0
                zero_point = torch.zeros_like(amax, dtype=torch.int)
                y = torch.fake_quantize_per_channel_affine(input, scale, zero_point, 1, 0, 255)
                return torch.fake_quantize_per_channel_affine(y, scale, zero_point, 1, -128, 127)
        x = torch.randn(6, 4, 3, 3)
        self.run_test(FakeQuantizePerChannelModel(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    @skipScriptTest()
    def test_fake_quantize_activation(self):
        if False:
            while True:
                i = 10
        from torch.ao import quantization
        m = torch.nn.Linear(1, 1)
        m.qconfig = quantization.QConfig(activation=quantization.default_fake_quant, weight=quantization.default_per_channel_weight_fake_quant)
        quantization.prepare_qat(m.train(), inplace=True)
        m.apply(quantization.enable_observer)
        m.apply(quantization.enable_fake_quant)
        for module in m.modules():
            if isinstance(module, quantization.FakeQuantize):
                module.calculate_qparams()
        m.apply(quantization.disable_observer)
        m.eval()
        m.weight = torch.nn.Parameter(torch.tensor([[1.0], [1.0], [1.0]]))
        m.bias = torch.nn.Parameter(torch.tensor([0.0]))
        x = torch.tensor([[150.0], [127.0], [-5.0]])
        self.run_test(m, x)

    def test_batchnorm_training(self):
        if False:
            print('Hello World!')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=False)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x
        x = torch.randn(10, 3, 20, 20) * 2
        model_export = MyModule()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.TRAINING, rtol=0.001, atol=1e-05)
        model_export.train()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.PRESERVE, rtol=0.001, atol=1e-05)

    def test_batchnorm_training_mode_fix_layer(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                self.bn3.eval()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x
        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.TRAINING, rtol=0.001, atol=1e-05)
        model_export.train()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.PRESERVE, rtol=0.001, atol=1e-05)

    def test_batchnorm_eval_mode_train_layer(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.bn1 = torch.nn.BatchNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.bn2 = torch.nn.BatchNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.bn3 = torch.nn.BatchNorm2d(3, affine=True)
                self.bn3.train()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.bn1(x)
                x = self.cv1(x)
                x = self.bn2(x)
                x = self.cv2(x)
                x = self.bn3(x)
                return x
        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL, rtol=0.001, atol=1e-05)
        model_export.eval()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.PRESERVE, rtol=0.001, atol=1e-05)

    def test_instancenorm_training(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x
        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.TRAINING, rtol=0.001, atol=1e-05)
        model_export.train()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.PRESERVE, rtol=0.001, atol=1e-05)

    def test_instancenorm_training_mode_fix_layer(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(3, affine=True)
                self.cv1 = torch.nn.Conv2d(3, 3, 10)
                self.in2 = torch.nn.InstanceNorm2d(3, affine=False)
                self.cv2 = torch.nn.Conv2d(3, 3, 10)
                self.in3 = torch.nn.InstanceNorm2d(3, affine=True)
                self.in3.eval()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x
        x = torch.randn(10, 3, 128, 128)
        model_export = MyModule()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.TRAINING, rtol=0.001, atol=1e-05)
        model_export.train()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.PRESERVE, rtol=0.001, atol=1e-05)

    def test_instancenorm_eval_mode_train_layer(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.in1 = torch.nn.InstanceNorm2d(8, affine=True)
                self.cv1 = torch.nn.Conv2d(8, 8, 10)
                self.in2 = torch.nn.InstanceNorm2d(8, affine=False)
                self.cv2 = torch.nn.Conv2d(8, 8, 10)
                self.in3 = torch.nn.InstanceNorm2d(8, affine=True)
                self.in3.train()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.in1(x)
                x = self.cv1(x)
                x = self.in2(x)
                x = self.cv2(x)
                x = self.in3(x)
                return x
        x = torch.randn(10, 8, 128, 128)
        model_export = MyModule()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL, rtol=0.001, atol=1e-05)
        model_export.eval()
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.PRESERVE, rtol=0.001, atol=1e-05)

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.dropout = torch.nn.Dropout(0.4)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                dropout = self.dropout(x)
                return dropout
        model = MyModule()
        x = torch.randn(10)
        model.train()
        model_onnx = io.BytesIO()
        torch.onnx.export(model, x, model_onnx, opset_version=self.opset_version, do_constant_folding=False, training=torch.onnx.TrainingMode.TRAINING)
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        assert not torch.all(torch.eq(x, torch.from_numpy(ort_outs[0])))
        script_model = torch.jit.script(model)
        output = model(x)
        model_onnx = io.BytesIO()
        torch.onnx.export(model, x, model_onnx, opset_version=self.opset_version, do_constant_folding=False, training=torch.onnx.TrainingMode.TRAINING)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        assert not torch.all(torch.eq(x, torch.from_numpy(ort_outs[0])))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_dropout_training_zero(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.dropout = torch.nn.Dropout(0.5)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                dropout = self.dropout(x)
                return dropout
        model = MyModule()
        x = torch.randn(10, 3, 128, 128)
        y = x.numpy()
        y_mask = np.where(y == 0, 1, y)
        input = torch.from_numpy(y_mask)
        nb_elements = torch.numel(input)
        model.train()
        model_onnx = io.BytesIO()
        torch.onnx.export(model, x, model_onnx, opset_version=self.opset_version, do_constant_folding=False, training=torch.onnx.TrainingMode.TRAINING)
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        y = model(input)
        output = y.cpu().numpy()
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)
        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements
        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)
        script_model = torch.jit.script(model)
        y = model(input)
        output = y.cpu().numpy()
        model_onnx = io.BytesIO()
        torch.onnx.export(model, x, model_onnx, opset_version=self.opset_version, do_constant_folding=False, training=torch.onnx.TrainingMode.TRAINING)
        ort_sess = verification._ort_session(model_onnx)
        ort_outs = verification._run_onnx(ort_sess, (x,))
        ort_mask = np.where(ort_outs[0] != 0, 1, 0)
        pyt_mask = np.where(output != 0, 1, 0)
        ratio_pytorch = np.sum(pyt_mask) / nb_elements
        ratio_ort = np.sum(ort_mask) / nb_elements
        np.testing.assert_allclose(ratio_pytorch, ratio_ort, rtol=0.01, atol=0.01)

    def test_conv_bn(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=1, stride=2, padding=3, bias=True)
                self.bn = torch.nn.BatchNorm2d(16, affine=True)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                bn = self.bn(x)
                return bn
        model_export = MyModule()
        x = torch.randn(10, 3, 128, 128)
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL)
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.TRAINING, rtol=0.001, atol=1e-05)

    def test_multiple_conv_bn(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
                self.conv2 = torch.nn.Conv2d(64, 2, kernel_size=1, stride=1, padding=0, bias=False)
                self.conv3 = torch.nn.Conv2d(2, 2, kernel_size=3, stride=1, padding=1, bias=False)
                self.bn = torch.nn.BatchNorm2d(64)
                self.bn2 = torch.nn.BatchNorm2d(2)
                self.relu = torch.nn.ReLU(inplace=True)
                self.maxpool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                x = self.bn(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                x = self.conv3(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x
        model_export = MyModule()
        x = torch.randn(2, 3, 224, 224)
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.TRAINING, rtol=0.001, atol=1e-05)
        self.run_test(model_export, (x,), training=torch.onnx.TrainingMode.EVAL)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nms(self):
        if False:
            return 10
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)

        class Module(torch.nn.Module):

            def forward(self, boxes, scores):
                if False:
                    i = 10
                    return i + 15
                return torchvision.ops.nms(boxes, scores, 0.5)
        self.run_test(Module(), (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_batched_nms(self):
        if False:
            i = 10
            return i + 15
        num_boxes = 100
        boxes = torch.rand(num_boxes, 4)
        boxes[:, 2:] += boxes[:, :2]
        scores = torch.randn(num_boxes)
        idxs = torch.randint(0, 5, size=(num_boxes,))

        class Module(torch.nn.Module):

            def forward(self, boxes, scores, idxs):
                if False:
                    return 10
                return torchvision.ops.batched_nms(boxes, scores, idxs, 0.5)
        self.run_test(Module(), (boxes, scores, idxs))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_clip_boxes_to_image(self):
        if False:
            return 10
        boxes = torch.randn(5, 4) * 500
        boxes[:, 2:] += boxes[:, :2]
        size = torch.randn(200, 300)
        size_2 = torch.randn(300, 400)

        class Module(torch.nn.Module):

            def forward(self, boxes, size):
                if False:
                    i = 10
                    return i + 15
                shape = (size.shape[0], size.shape[1])
                return torchvision.ops.boxes.clip_boxes_to_image(boxes, shape)
        self.run_test(Module(), (boxes, size), input_names=['boxes', 'size'], dynamic_axes={'size': [0, 1]}, additional_test_inputs=[(boxes, size), (boxes, size_2)])

    @skipScriptTest(reason='Conditioning on input type via prim::isinstance unsupported in ONNX')
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_align(self):
        if False:
            return 10
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        model = torchvision.ops.RoIAlign((5, 5), 1.0, 2)
        self.run_test(model, (x, single_roi))

    @skipScriptTest(reason='Conditioning on input type via prim::isinstance unsupported in ONNX')
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_roi_align_aligned(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 1.5, 1.5, 3, 3]], dtype=torch.float32)
        model1 = torchvision.ops.RoIAlign((5, 5), 1.0, 2, aligned=True)
        self.run_test(model1, (x, single_roi))
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model2 = torchvision.ops.RoIAlign((5, 5), 0.5, 3, aligned=True)
        self.run_test(model2, (x, single_roi))
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model3 = torchvision.ops.RoIAlign((5, 5), 1.8, 2, aligned=True)
        self.run_test(model3, (x, single_roi))
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        single_roi = torch.tensor([[0, 0.2, 0.3, 4.5, 3.5]], dtype=torch.float32)
        model4 = torchvision.ops.RoIAlign((2, 2), 2.5, 0, aligned=True)
        self.run_test(model4, (x, single_roi))

    @skipScriptTest(reason='Conditioning on input type via prim::isinstance unsupported in ONNX')
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_roi_pool(self):
        if False:
            while True:
                i = 10
        x = torch.rand(1, 1, 10, 10, dtype=torch.float32)
        rois = torch.tensor([[0, 0, 0, 4, 4]], dtype=torch.float32)
        pool_h = 5
        pool_w = 5
        model = torchvision.ops.RoIPool((pool_h, pool_w), 2.0)
        self.run_test(model, (x, rois))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_resize_images(self):
        if False:
            return 10

        class TransformModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images):
                if False:
                    while True:
                        i = 10
                return self.transform.resize(images, None)[0]
        input = torch.rand(3, 10, 20)
        input_test = torch.rand(3, 100, 150)
        self.run_test(TransformModule(), (input,), input_names=['input1'], dynamic_axes={'input1': [0, 1, 2]}, additional_test_inputs=[(input,), (input_test,)])

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_transform_images(self):
        if False:
            while True:
                i = 10

        class TransformModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.transform = _init_test_generalized_rcnn_transform()

            def forward(self, images: List[Tensor]):
                if False:
                    for i in range(10):
                        print('nop')
                return self.transform(images)[0].tensors
        input = (torch.rand(3, 100, 200), torch.rand(3, 200, 200))
        input_test = (torch.rand(3, 100, 200), torch.rand(3, 200, 200))
        self.run_test(TransformModule(), (input,), additional_test_inputs=[(input,), (input_test,)])

    def get_features(self, images):
        if False:
            while True:
                i = 10
        (s0, s1) = images.shape[-2:]
        features = [('0', torch.rand(2, 256, s0 // 4, s1 // 4)), ('1', torch.rand(2, 256, s0 // 8, s1 // 8)), ('2', torch.rand(2, 256, s0 // 16, s1 // 16)), ('3', torch.rand(2, 256, s0 // 32, s1 // 32)), ('4', torch.rand(2, 256, s0 // 64, s1 // 64))]
        features = OrderedDict(features)
        return features

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_rpn(self):
        if False:
            return 10

        class RPNModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.rpn = _init_test_rpn()

            def forward(self, images, features: Dict[str, Tensor]):
                if False:
                    print('Hello World!')
                images_m = torchvision.models.detection.image_list.ImageList(images, [(i.shape[-1], i.shape[-2]) for i in images])
                return self.rpn(images_m, features)
        images = torch.rand(2, 3, 150, 150)
        features = self.get_features(images)
        images2 = torch.rand(2, 3, 80, 80)
        test_features = self.get_features(images2)
        model = RPNModule()
        model.eval()
        model(images, features)
        self.run_test(model, (images, features), input_names=['input1', 'input2', 'input3', 'input4', 'input5', 'input6'], dynamic_axes={'input1': [0, 1, 2, 3], 'input2': [0, 1, 2, 3], 'input3': [0, 1, 2, 3], 'input4': [0, 1, 2, 3], 'input5': [0, 1, 2, 3], 'input6': [0, 1, 2, 3]}, additional_test_inputs=[(images, features), (images2, test_features)])

    @skipIfUnsupportedMaxOpsetVersion(15)
    @skipIfUnsupportedMinOpsetVersion(11)
    @skipScriptTest()
    def test_multi_scale_roi_align(self):
        if False:
            print('Hello World!')

        class TransformModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.model = torchvision.ops.MultiScaleRoIAlign(['feat1', 'feat2'], 3, 2)
                self.image_sizes = [(512, 512)]

            def forward(self, input: Dict[str, Tensor], boxes: List[Tensor]) -> Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return self.model(input, boxes, self.image_sizes)
        i = OrderedDict()
        i['feat1'] = torch.rand(1, 5, 64, 64)
        i['feat2'] = torch.rand(1, 5, 16, 16)
        boxes = torch.rand(6, 4) * 256
        boxes[:, 2:] += boxes[:, :2]
        i1 = OrderedDict()
        i1['feat1'] = torch.rand(1, 5, 64, 64)
        i1['feat2'] = torch.rand(1, 5, 16, 16)
        boxes1 = torch.rand(6, 4) * 256
        boxes1[:, 2:] += boxes1[:, :2]
        self.run_test(TransformModule(), (i, [boxes]), additional_test_inputs=[(i, [boxes]), (i1, [boxes1])])

    def test_set_(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x.set_(y)
                return x
        x = torch.ones(2, 3)
        y = torch.randn(4, 6)
        self.run_test(M(), (x, y), remained_onnx_input_idx=[1])
        y2 = torch.randn(5, 2)
        self.run_test(M(), (x, y), remained_onnx_input_idx=[1], input_names=['x', 'y'], dynamic_axes={'x': [0, 1], 'y': [0, 1]}, additional_test_inputs=[(y, y2)])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules(self):
        if False:
            return 10

        class InnerModule2(torch.nn.Module):

            def __init__(self, embedding_dim):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weights = InnerModule2.get_embedding(embedding_dim)
                self.register_buffer('_float_tensor', torch.FloatTensor(1))
                self.const = 2

            @staticmethod
            def get_embedding(embedding_dim: int):
                if False:
                    print('Hello World!')
                emb = 4 / (embedding_dim // 2 - 1)
                emb = torch.exp(torch.arange(embedding_dim // 2, dtype=torch.float) * -emb)
                return emb

            def forward(self, input, incremental_state: Optional[Tensor]=None):
                if False:
                    i = 10
                    return i + 15
                (bsz, seq_len) = (input.shape[0], input.shape[1])
                self.const = 3
                if self.weights is None:
                    self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.weights = self.weights.to(self._float_tensor)
                self.weights = self.weights * self.const
                if incremental_state is not None:
                    pos = seq_len
                    return self.weights[1 + pos, :].expand(bsz, 1, -1)
                return self.weights.index_select(0, torch.ones(bsz * seq_len, dtype=torch.int64)).view(bsz, seq_len, -1)

        class InnerModule(torch.nn.Module):

            def __init__(self, embedding_dim):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weights = InnerModule.get_embedding(embedding_dim)
                self.module = InnerModule2(embedding_dim=8)

            @staticmethod
            def get_embedding(embedding_dim: int):
                if False:
                    for i in range(10):
                        print('nop')
                emb = 4 / (embedding_dim // 2 - 1)
                emb = torch.exp(torch.arange(embedding_dim // 2, dtype=torch.float) * -emb)
                return emb

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.module(x) + self.weights

        class Module(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.module(x)
        x = torch.randn(3, 256)
        self.run_test(Module(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_set_attr_modules_2(self):
        if False:
            return 10

        class InnerModule(torch.nn.Module):

            def __init__(self, embedding_dim):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.embedding_dim = embedding_dim
                self.const = 2.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                self.register_buffer('_float_tensor', torch.FloatTensor(1))

            @staticmethod
            def get_embedding(embedding_dim: int):
                if False:
                    while True:
                        i = 10
                emb = 4 / (embedding_dim // 2 - 1)
                emb = torch.exp(torch.arange(embedding_dim // 2, dtype=torch.float) * -emb)
                return emb

            def forward(self, input, incremental_state: Optional[Tensor]=None):
                if False:
                    while True:
                        i = 10
                (bsz, seq_len) = (input.shape[0], input.shape[1])
                self.const = 1.5
                self.weights = InnerModule.get_embedding(self.embedding_dim)
                return self.weights.index_select(0, torch.ones(bsz * seq_len, dtype=torch.int64)).view(bsz, seq_len, -1) * self.const

        class Module(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.module = InnerModule(embedding_dim=8)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.module(x)
        x = torch.randn(3, 256)
        self.run_test(Module(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(Module(), (x,), remained_onnx_input_idx=[])

    def test_set_attr(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv1d(3, 10, 2)
                self.b = False

            def forward(self, box_regression, weight):
                if False:
                    i = 10
                    return i + 15
                self.b = True
                self.conv.weight = weight
                w = torch.softmax(self.conv.weight, dim=0)
                self.conv.weight = w + w
                if self.b:
                    return box_regression + self.conv.weight
                else:
                    return box_regression - self.conv.weight
        model = torch.jit.script(MyModule())
        weight = torch.ones(3, 2)
        box_regression = torch.randn(3, 2)
        self.run_test(model, (box_regression, weight))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_2(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                if False:
                    for i in range(10):
                        print('nop')
                if self.conv.bias is not None:
                    b = self.conv.bias
                    assert b is not None
                    self.conv.bias = anchors + b
                elif self.conv.weight is not None:
                    self.conv.weight = torch.randn(3, 10)
                    self.conv.bias = self.conv.weight[:]

            def forward(self, anchors) -> Optional[Tensor]:
                if False:
                    print('Hello World!')
                self.set_cell_anchors(anchors)
                return self.conv.bias
        model = torch.jit.script(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_3(self):
        if False:
            return 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                if False:
                    return 10
                self.conv.weight = torch.ones(3, 10)
                if self.conv.bias is not None:
                    self.conv.bias = torch.randn(3, 10, 3)
                    self.conv.weight = anchors + self.conv.weight
                    boxes[:] = torch.zeros(2, 3)

            def forward(self, anchors) -> Tuple[Tensor, Tensor]:
                if False:
                    for i in range(10):
                        print('nop')
                boxes = torch.ones(2, 2, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return (self.conv.weight, boxes)
                return (anchors, boxes)
        model = torch.jit.script(MyModule())
        anchors = torch.rand(3, 10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_4(self):
        if False:
            return 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                if False:
                    for i in range(10):
                        print('nop')
                self.conv.weight = torch.zeros(10, 3)
                if self.conv.bias is not None:
                    w = self.conv.bias
                    assert w is not None
                    self.conv.bias = anchors + w
                else:
                    self.conv.bias = torch.ones(3, 10, 3)

            def forward(self, feature_maps, anchors) -> Tuple[Tensor, Tensor]:
                if False:
                    i = 10
                    return i + 15
                self.set_cell_anchors(anchors)
                result = []
                if self.conv.bias is not None:
                    a = self.conv.bias
                    assert a is not None
                    result += [a]
                result += [feature_maps]
                return (result[0], result[1])
        model = torch.jit.script(MyModule())
        x = torch.rand(5, 11, 30)
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, (x, anchors))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_5(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors):
                if False:
                    return 10
                self.conv.weight = torch.arange(10)
                for i in range(10):
                    if i == 3:
                        for j in range(10):
                            w = self.conv.weight
                            self.conv.weight = torch.arange(10) + w
                    self.conv.weight = self.conv.weight + torch.arange(10)
                    if self.conv.bias is not None:
                        a = self.conv.bias
                        assert a is not None
                        self.conv.bias = anchors + a

            def forward(self, anchors):
                if False:
                    i = 10
                    return i + 15
                self.set_cell_anchors(anchors)
                return (self.conv.weight, self.conv.bias)
        model = torch.jit.script(MyModule())
        anchors = torch.ones(3, 10, 3)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_set_attr_in_loop(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))

            def set_cell_anchors(self, anchors, boxes):
                if False:
                    return 10
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for j in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        boxes[j] += torch.ones(3, 3)

            def forward(self, anchors) -> Tuple[Tensor, Tensor]:
                if False:
                    i = 10
                    return i + 15
                boxes = torch.ones(10, 3, 3)
                self.set_cell_anchors(anchors, boxes)
                if self.conv.bias is not None:
                    return (self.conv.weight, boxes)
                return (anchors, boxes)
        model = torch.jit.script(MyModule())
        anchors = torch.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_set_attr_in_loop_with_list(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv1d(10, 3, 3)
                self.conv.weight = torch.nn.Parameter(torch.zeros(3, 10))
                self.conv.bias = torch.nn.Parameter(torch.zeros(3, 10, 3))
                self.boxes: List[Tensor] = [torch.ones(1)]

            def set_cell_anchors(self, anchors):
                if False:
                    i = 10
                    return i + 15
                self.conv.weight = torch.randn(3, 10)
                for i in range(self.conv.weight.size(0)):
                    for j in range(10):
                        self.conv.bias = torch.randn(3, 10, 3)
                        self.conv.weight = anchors * i
                        self.boxes.append(torch.ones(3, 3))

            def forward(self, anchors) -> Tuple[Tensor, List[Tensor]]:
                if False:
                    i = 10
                    return i + 15
                self.boxes = []
                self.set_cell_anchors(anchors)
                if self.conv.bias is not None:
                    return (self.conv.weight, self.boxes)
                return (anchors, self.boxes)
        model = torch.jit.script(MyModule())
        anchors = torch.rand(10)
        self.run_test(model, anchors)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if(self):
        if False:
            return 10

        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int, prev_state: Tensor) -> Tuple[Tensor, Tensor]:
            if False:
                print('Hello World!')
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state[:] = torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1) + state[:]
                state_copy[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 2
                state_copy[:] = torch.zeros(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 2
            else:
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            return (state, state_copy)

        class Example(torch.nn.Module):

            def __init__(self, hidden_size):
                if False:
                    print('Hello World!')
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                if False:
                    i = 10
                    return i + 15
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return (prev_state[0], prev_state[1])
        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor), input_names=['random_data', 'empty_tensor'], dynamic_axes={'random_data': [0, 1, 2, 3], 'empty_tensor': [0, 1, 2, 3, 4]})
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_2(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int, prev_state: Tensor) -> Tuple[Tensor, Tensor]:
            if False:
                i = 10
                return i + 15
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_copy = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                for i in range(2):
                    state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * i
                    state_copy[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * i
            elif prev_state.size(0) == 1:
                s = state[:]
                state[:] = prev_state + s
            elif prev_state.size(0) == 2:
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            return (state, state_copy)

        class Example(torch.nn.Module):

            def __init__(self, hidden_size):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                if False:
                    return 10
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return (prev_state[0], prev_state[1])
        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        random_state = torch.rand((1, 1, 10, 30, 30))
        self.run_test(model, (random_data, empty_tensor), input_names=['data', 'state'], dynamic_axes={'data': [0, 1, 2], 'state': [0, 1, 2, 3, 4]}, additional_test_inputs=[(random_data, random_state)])
        self.run_test(model, (random_data, empty_tensor), input_names=['data', 'state'], dynamic_axes={'state': [0, 1, 2, 3, 4]}, additional_test_inputs=[(random_data, random_state)], remained_onnx_input_idx=[1])
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_3(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int, prev_state: Tensor) -> Tensor:
            if False:
                return 10
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) < 2:
                state = state * 3
                if prev_state.size(0) == 0:
                    state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
                else:
                    state = state + 2
            return state

        class Example(torch.nn.Module):

            def __init__(self, hidden_size):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                if False:
                    for i in range(10):
                        print('nop')
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state
        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor), input_names=['random_data', 'empty_tensor'], dynamic_axes={'random_data': [0, 1, 2, 3], 'empty_tensor': [0, 1, 2, 3, 4]})
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_4(self):
        if False:
            return 10

        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int, prev_state: Tensor) -> Tensor:
            if False:
                while True:
                    i = 10
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if prev_state.size(0) == 0:
                state = state + 3
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
                state = state + 3
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            else:
                state = state + 2
            return state

        class Example(torch.nn.Module):

            def __init__(self, hidden_size):
                if False:
                    print('Hello World!')
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                if False:
                    print('Hello World!')
                prev_state = check_init(input_data, self.hidden_size, prev_state)
                return prev_state
        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor), input_names=['random_data', 'empty_tensor'], dynamic_axes={'random_data': [0, 1, 2, 3], 'empty_tensor': [0, 1, 2, 3, 4]})
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_if_5(self):
        if False:
            while True:
                i = 10

        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int, prev_state: Tensor) -> Tuple[Tensor, Tensor]:
            if False:
                for i in range(10):
                    print('nop')
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            state_ref = state
            if prev_state.size(0) == 0:
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
                state = state + 3
                state[:] = torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 4
            else:
                state = state + 2
            return (state, state_ref)

        class Example(torch.nn.Module):

            def __init__(self, hidden_size):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data, prev_state):
                if False:
                    for i in range(10):
                        print('nop')
                (prev_state, state_ref) = check_init(input_data, self.hidden_size, prev_state)
                return (prev_state, state_ref)
        model = Example(4)
        random_data = torch.rand((1, 5, 4, 4))
        empty_tensor = torch.tensor([], dtype=torch.float).view(0, 0, 0, 0, 0)
        self.run_test(model, (random_data, empty_tensor), input_names=['random_data', 'empty_tensor'], dynamic_axes={'random_data': [0, 1, 2, 3], 'empty_tensor': [0, 1, 2, 3, 4]})
        self.run_test(model, (random_data, empty_tensor), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_append_in_block(self):
        if False:
            print('Hello World!')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                res = []
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_append_in_nested_block(self):
        if False:
            return 10

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                res = []
                for i in range(x.size(0)):
                    for j in range(x.size(1)):
                        res.append(torch.matmul(x[i][j], y))
                return res
        model = torch.jit.script(ListModel())
        x = torch.randn(4, 4, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_pop_in_block(self):
        if False:
            return 10

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for i in range(x.size(0)):
                    elem = res.pop()
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    elem = res.pop()
                return res.append(elem)
        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_list_del_in_block(self):
        if False:
            for i in range(10):
                print('nop')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                for i in range(x.size(0)):
                    del res[0]
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                    del res[0]
                return res.append(elem)
        model = torch.jit.script(ListModel())
        x = torch.randn(16, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_list_unpack(self):
        if False:
            for i in range(10):
                print('nop')

        class ListModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                res = []
                elem = torch.matmul(x[0], y)
                for i in range(x.size(0)):
                    res.append(torch.matmul(x[i], y))
                (a, b, c) = res
                return (a, b)
        model = torch.jit.script(ListModel())
        x = torch.randn(3, 3, 4)
        y = torch.randn(4, 5)
        self.run_test(model, (x, y))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_index_put_inplace_ops(self):
        if False:
            for i in range(10):
                print('nop')

        @torch.jit.script
        def check_init(input_data: Tensor, hidden_size: int) -> Tensor:
            if False:
                i = 10
                return i + 15
            batch_size = input_data.size(0)
            spatial_size_0 = input_data.size(2)
            spatial_size_1 = input_data.size(3)
            state_size = (2, batch_size, hidden_size, spatial_size_0, spatial_size_1)
            state = torch.zeros(state_size, device=input_data.device)
            if input_data.size(0) == 1:
                state[1] += torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 2
                state[1] /= torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * 3
            for i in range(input_data.size(0)):
                state[1] += torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1)
                state[1] /= torch.ones(batch_size, hidden_size, spatial_size_0, spatial_size_1) * i
            return state

        class Example(torch.nn.Module):

            def __init__(self, hidden_size):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.hidden_size = hidden_size

            def forward(self, input_data):
                if False:
                    print('Hello World!')
                state = check_init(input_data, self.hidden_size)
                return state
        model = Example(10)
        random_data = torch.rand((1, 5, 30, 30))
        self.run_test(model, random_data, input_names=['random_data'], dynamic_axes={'random_data': [0, 1, 2, 3]})
        self.run_test(model, random_data, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_input_mask_model(self):
        if False:
            i = 10
            return i + 15

        class InputMaskModel(torch.nn.Module):

            def __init__(self, output_size):
                if False:
                    return 10
                super().__init__()
                self.bias = torch.nn.Parameter(torch.empty(output_size, dtype=torch.float))
                with torch.no_grad():
                    self.bias.zero_()

            def forward(self, model_input, y):
                if False:
                    for i in range(10):
                        print('nop')
                input_mask = (model_input <= 0) | (model_input > 25)
                y[input_mask, :] = 0.0
                output = y + self.bias
                return output
        output_size = 4
        m = InputMaskModel(output_size)
        x = torch.tensor([0, 4, 24, 25], dtype=torch.int64)
        y = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float)
        self.run_test(m, (x, y))

        class InputMaskModel(torch.nn.Module):

            def __init__(self, output_size):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()

            def forward(self, model_input_1, model_input_2, y):
                if False:
                    print('Hello World!')
                input_mask_1 = (model_input_1 <= 0) | (model_input_1 > 25)
                input_mask_2 = (model_input_2 < 1) | (model_input_2 >= 12)
                y[input_mask_1, input_mask_2] = 0.0
                return y
        output_size = 4
        m = InputMaskModel(output_size)
        x1 = torch.tensor([0, 4, 24, 25], dtype=torch.int64)
        x2 = torch.tensor([0, 3, 12, 15], dtype=torch.int64)
        y = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4], [0.1, 0.2, 0.3, 0.4]], dtype=torch.float)
        self.run_test(m, (x1, x2, y))

    @skipScriptTest()
    def test_unsafe_chunk(self):
        if False:
            print('Hello World!')

        class ChunkModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.unsafe_chunk(x, 3, dim=1)
        model = ChunkModel()
        model.eval()
        x = torch.randn(1, 18)
        self.run_test(model, x, input_names=['x'])

    def test_symbolic_shape_inference(self):
        if False:
            while True:
                i = 10

        class ShapeModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                shape = x.size()[:3] + (-1,)
                y = y.reshape(shape)
                return y.transpose(1, 2)
        model = ShapeModel()
        model.eval()
        x = torch.ones(2, 3, 4, 5)
        y = torch.ones(3, 4, 5, 2)
        self.run_test(model, (x, y), input_names=['x', 'y'], dynamic_axes={'x': [0, 1, 2, 3], 'y': [0, 1, 2, 3]})
        self.run_test(model, (x, y), remained_onnx_input_idx=[1])

        class ViewModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x.view(-1)
        model = ViewModel()
        model.eval()
        x = torch.tensor(2.0)
        self.run_test(model, (x,))

        class ViewModel_2(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                (N, C, H, W) = (x.shape[0], x.shape[2], x.shape[3], x.shape[4])
                x1 = x.view(N, -1, C, H, W)
                x2 = x1.permute(0, 3, 4, 1, 2)
                return x2.reshape(N, -1, C)
        model = ViewModel_2()
        model.eval()
        x = torch.ones(2, 3, 4, 5, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_arange(self):
        if False:
            return 10

        class ArangeModel(torch.nn.Module):

            def forward(self, signal):
                if False:
                    i = 10
                    return i + 15
                frame_step = 2
                outer_dimensions = signal.size()[:-2]
                (frames, frame_length) = signal.size()[-2:]
                subframe_length = signal.size()[0]
                subframe_step = frame_step // subframe_length
                subframes_per_frame = frame_length // subframe_length
                output_size = frame_step * (frames - 1) + frame_length
                output_subframes = output_size // subframe_length
                frame = torch.arange(0, output_subframes)
                return frame
        model = ArangeModel()
        model.eval()
        (M, C, K, N) = (1, 2, 3, 4)
        x = torch.randint(5, (M, C, K, N))
        y = torch.randint(5, (M, C + 1, K + 1, N + 1))
        self.run_test(model, x, input_names=['x'], dynamic_axes={'x': [0, 1, 2, 3]})
        self.run_test(model, x, remained_onnx_input_idx=[])
        self.run_test(model, x, input_names=['x'], dynamic_axes={'x': [0, 1, 2, 3]}, additional_test_inputs=[(x,), (y,)])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box(self):
        if False:
            print('Hello World!')

        class BoxModel(torch.nn.Module):

            def forward(self, boxes):
                if False:
                    while True:
                        i = 10
                min_size = 0.01
                (ws, hs) = (boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1])
                keep = (ws >= min_size) & (hs >= min_size)
                keep = torch.where(keep)[0]
                return keep
        model = BoxModel()
        model.eval()
        x = torch.ones(2, 4)
        y = torch.ones(3, 5)
        self.run_test(model, x)
        self.run_test(model, x, input_names=['x'], dynamic_axes={'x': [0, 1]}, additional_test_inputs=[(x,), (y,)])

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_symbolic_shape_inference_box_if(self):
        if False:
            print('Hello World!')

        class BoxIfModel(torch.nn.Module):

            def forward(self, boxes, scores):
                if False:
                    while True:
                        i = 10
                score_thresh = 0.0
                inds = torch.where(scores > score_thresh)[0]
                boxes_1 = boxes[inds]
                if boxes_1.numel() > 3:
                    return boxes_1
                else:
                    return boxes_1 * 2
        model = BoxIfModel()
        model.eval()
        boxes = torch.ones(2, 4)
        scores = torch.ones(1, 4)
        self.run_test(model, (boxes, scores))

    @skipIfUnsupportedMinOpsetVersion(11)
    @skipDtypeChecking
    def test_symbolic_shape_inference_arange_2(self):
        if False:
            return 10

        class ArangeModel(torch.nn.Module):

            def forward(self, start):
                if False:
                    while True:
                        i = 10
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.int64)
        x = torch.randn(2, 3, 4)
        self.run_test(ArangeModel(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModel(), (x,), remained_onnx_input_idx=[])

        class ArangeModel2(torch.nn.Module):

            def forward(self, start):
                if False:
                    print('Hello World!')
                return torch.arange(start.size(0), 8.5, 1.5, dtype=torch.double)
        x = torch.randn(2, 3, 4)
        self.run_test(ArangeModel2(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ArangeModel2(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_nonzero(self):
        if False:
            return 10

        class OneLikeModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                ones = torch.ones_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return torch.nonzero(ones)
        x = torch.randn(2)
        self.run_test(OneLikeModel(), x, input_names=['x'], dynamic_axes={'x': [0]})
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])
        x = torch.randn(2, 3, 4)
        self.run_test(OneLikeModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(OneLikeModel(), x, remained_onnx_input_idx=[])

        class ZeroLikeModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                zeros = torch.zeros_like(x, dtype=torch.float, layout=torch.strided, device=torch.device('cpu'))
                return torch.nonzero(zeros)
        x = torch.randn(2)
        self.run_test(ZeroLikeModel(), x, input_names=['x'], dynamic_axes={'x': [0]})
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])
        x = torch.randn(2, 3, 4)
        self.run_test(ZeroLikeModel(), x, input_names=['x'], dynamic_axes={'x': [0, 1, 2]})
        self.run_test(ZeroLikeModel(), x, remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_1(self):
        if False:
            while True:
                i = 10

        class ExpandModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x.expand(4, 6, 2)
        x = torch.randn(6, 1, requires_grad=True)
        self.run_test(ExpandModel(), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_symbolic_shape_inference_expand_2(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                input_shape = x.size()
                (batch_size, seq_length) = input_shape
                seq_ids = torch.arange(seq_length)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                return causal_mask.transpose(0, 1)
        x = torch.randn(3, 16)
        self.run_test(M(), (x,), input_names=['x'], dynamic_axes={'x': [0, 1]})
        self.run_test(M(), (x,), remained_onnx_input_idx=[])

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_symbolic_shape_inference_slice(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x, position_bias):
                if False:
                    for i in range(10):
                        print('nop')
                input_shape = x.size()
                (batch_size, seq_length) = input_shape
                position_bias = position_bias[:, :, -seq_length:, :]
                return position_bias.transpose(0, 1)
        x = torch.randn(3, 16)
        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(M(), (x, position_bias), input_names=['x', 'position_bias'], dynamic_axes={'x': [0, 1], 'position_bias': [0, 1, 2, 3]})
        self.run_test(M(), (x, position_bias), remained_onnx_input_idx=[1])

    def test_symbolic_shape_inference_slice_2(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, position_bias):
                if False:
                    while True:
                        i = 10
                position_bias = position_bias[:, :, -2:, :]
                return position_bias.transpose(0, 1)
        position_bias = torch.randn(1, 3, 20, 8)
        self.run_test(M(), (position_bias,))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_symbolic_shape_inference_time(self):
        if False:
            print('Hello World!')
        input = torch.randn(RNN_SEQUENCE_LENGTH, BATCH_SIZE, RNN_INPUT_SIZE)
        h0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        c0 = torch.randn(1, BATCH_SIZE, RNN_HIDDEN_SIZE)
        model_lstm = torch.nn.LSTM(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False)
        self.run_test(model_lstm, (input, (h0, c0)), input_names=['x', 'y'], dynamic_axes={'x': [0, 1]})
        model_gru = torch.nn.GRU(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False)
        self.run_test(model_gru, (input, h0), input_names=['x', 'y'], dynamic_axes={'x': [0, 1]})
        model_rnn = torch.nn.RNN(RNN_INPUT_SIZE, RNN_HIDDEN_SIZE, 1, bidirectional=False, bias=False)
        self.run_test(model_rnn, (input, h0), input_names=['x', 'y'], dynamic_axes={'x': [0, 1]})

    def test_symbolic_shape_inference_dynamic_axes(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, input_ids):
                if False:
                    while True:
                        i = 10
                input_shape = input_ids.size()
                input_ids = input_ids.view(-1, input_shape[-1])
                return input_ids.transpose(0, 1)
        x = torch.randn(3, 16)
        self.run_test(M(), (x,), input_names=['input_ids'], dynamic_axes={'input_ids': {0: 'batch', 1: 'sequence'}})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_periodic(self):
        if False:
            return 10

        class HannWindowModule_Periodic(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                if False:
                    i = 10
                    return i + 15
                self.window_length = window_length
                return torch.add(x, torch.hann_window(self.window_length, periodic=True, dtype=torch.float))
        win_length = 100
        x = torch.randn(win_length)
        module = HannWindowModule_Periodic()
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_hann_window_not_periodic(self):
        if False:
            return 10

        class HannWindowModule_NotPeriodic(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                if False:
                    print('Hello World!')
                self.window_length = window_length
                return torch.add(x, torch.hann_window(self.window_length, periodic=False, dtype=torch.float))
        win_length = 100
        x = torch.randn(win_length)
        module = HannWindowModule_NotPeriodic()
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(9)
    @skipScriptTest()
    def test_hann_window_default_values(self):
        if False:
            print('Hello World!')

        class HannWindowModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.window_length = 0

            def forward(self, x, window_length: int):
                if False:
                    while True:
                        i = 10
                import torch.nn.functional as F
                self.window_length = window_length
                return torch.add(x, F.relu(torch.hann_window(self.window_length)))
        win_length = 100
        x = torch.randn(win_length, dtype=torch.float)
        module = HannWindowModule()
        output = module(x, win_length)
        self.run_test(module, (x, win_length))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_count(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                output = torch.tensordot(x, y, 2)
                return output
        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))
        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dim_list(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                output = torch.tensordot(x, y, ([1, -2, -1], [1, 0, 3]))
                return output
        x = torch.randint(6, (7, 4, 3, 5, 2))
        y = torch.randint(6, (5, 4, 4, 2, 6))
        self.run_test(M(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(12)
    def test_tensordot_dynamic_dim(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                output = torch.tensordot(x, y, 2)
                return output
        x = torch.randint(6, (7, 5, 3, 4))
        y = torch.randint(6, (3, 4, 9, 2))
        new_x = torch.randint(6, (8, 6, 2, 5))
        new_y = torch.randint(6, (2, 5, 3, 4))
        self.run_test(M(), (x, y), additional_test_inputs=[(new_x, new_y)], input_names=['input_x', 'input_y'], dynamic_axes={'input_x': [0, 1, 2, 3], 'input_y': [0, 1, 2, 3]})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_to_device(self):
        if False:
            while True:
                i = 10

        class M_ToDevice(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return (x.to(y.device), y)

        class M_ToDeviceDtype(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return (x.to(y.device, dtype=torch.long), y)
        x = torch.randn(6)
        y = torch.randn(6)
        self.run_test(M_ToDevice(), (x, y))
        self.run_test(M_ToDeviceDtype(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_fill(self):
        if False:
            print('Hello World!')

        class FillModule(torch.nn.Module):

            def forward(self, x, filled_value: int):
                if False:
                    return 10
                return x.fill_(filled_value)
        x = torch.randn((4, 5, 6))
        filled_value = 7
        self.run_test(FillModule(), (x, filled_value))

        class FillFloatModule(torch.nn.Module):

            def forward(self, x, filled_value: float):
                if False:
                    i = 10
                    return i + 15
                return x.fill_(filled_value)
        x = torch.randn((4, 5, 6))
        filled_value = 7.5
        self.run_test(FillFloatModule(), (x, filled_value))

        class FillScalarModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                res = x + 2
                res.fill_(2.5)
                return (res, x)
        x = torch.ones(2, 3, 4, dtype=torch.long)
        self.run_test(FillScalarModule(), x)

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_normal(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self, dim, index, updates):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x.index_add_(self.dim, self.index, self.updates)
                return x
        x = torch.ones(5, 1)
        updates = torch.tensor([[1], [4], [7], [3], [2]], dtype=torch.float)
        index = torch.tensor([0, 2, 3, 1, 4])
        self.run_test(M(0, index, updates), (x,))
        x = torch.ones(1, 4, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float)
        index = torch.tensor([0, 2, 3, 1])
        self.run_test(M(1, index, updates), (x,))
        updates = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9], [2, 3, 4]]], dtype=torch.float)
        index = torch.tensor([0, 2, 1])
        self.run_test(M(2, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dim_size_differ(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self, dim, index, updates):
                if False:
                    return 10
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x.index_add_(self.dim, self.index, self.updates)
                return x
        x = torch.ones(1, 4, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6]]], dtype=torch.float)
        index = torch.tensor([0, 2, 1])
        self.run_test(M(1, index, updates), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_in_loop(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self, dim, index, updates, loop_count):
                if False:
                    return 10
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates
                self.loop_count = loop_count

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                for i in range(self.loop_count):
                    x.index_add_(self.dim, self.index, self.updates)
                return x
        x = torch.ones(1, 4, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float)
        index = torch.tensor([0, 2, 3, 1])
        loop_count = torch.randint(20, (1,))[0].item()
        self.run_test(M(1, index, updates, loop_count), (x,))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_if(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, dim, updates, index_true, index_false):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.dim = dim
                self.updates = updates
                self.index_true = index_true
                self.index_false = index_false

            def forward(self, x, cond):
                if False:
                    print('Hello World!')
                if cond:
                    x.index_add_(self.dim, self.index_true, self.updates)
                else:
                    x.index_add_(self.dim, self.index_false, self.updates)
                return x
        x = torch.ones(1, 4, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float)
        index_true = torch.tensor([0, 2, 3, 1])
        index_false = torch.tensor([1, 0, 2, 3])
        cond = torch.tensor(1, dtype=torch.bool)
        self.run_test(torch.jit.script(M(1, updates, index_true, index_false)), (x, cond))

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_index_add_dynamic_axes(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self, dim, index, updates):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.dim = dim
                self.index = index
                self.updates = updates

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x.index_add_(self.dim, self.index, self.updates)
                return x
        x = torch.ones(1, 4, 3)
        updates = torch.tensor([[[1, 5, 7], [2, 4, 5], [5, 5, 6], [2, 3, 4]]], dtype=torch.float)
        index = torch.tensor([0, 2, 3, 1])
        self.run_test(M(1, index, updates), (x,), input_names=['input_1'], dynamic_axes={'input_1': [0, 1]})

    def test_roll(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, shifts, dims):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.shifts = shifts
                self.dims = dims

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.roll(x, self.shifts, self.dims)
        x = torch.randn(2, 3, 4)
        self.run_test(M([1, 1], [1, 0]), (x,))
        self.run_test(M([0, 1, 2], [1, 0, 2]), (x,))
        self.run_test(M(2, 1), (x,))
        self.run_test(M([-1, 3], [-2, -1]), (x,))

    def test_sum(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.sum(x)
        x = torch.ones(12, 3)
        self.run_test(M(), (x,), input_names=['x'], dynamic_axes={'x': [0]})

    @skipShapeChecking
    def test_sum_empty_tensor(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (x[0:0].sum(), x.sum())
        x = torch.ones(12)
        self.run_test(M(), (x,))
        x = torch.ones(2, 0, 3)
        self.run_test(M(), (x,))
        x = torch.ones(0)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_broad_cast_tensors(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                m = torch.broadcast_tensors(x, y)
                return m
        x = torch.randint(5, (1,))
        y = torch.randint(5, (5,))
        self.run_test(M(), (x, y))
        x = torch.randint(5, (4, 2, 1, 4))
        y = torch.randint(5, (2, 3, 1))
        self.run_test(M(), (x, y))
        x = torch.randn(2, 1, 4)
        y = torch.randn(5, 2, 3, 1)
        self.run_test(M(), (x, y))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return (torch.distributions.Normal(x, y).sample().size(0), x, y)
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([[1.0], [2.0]])))
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([1.0])))
        self.run_test(M(), (torch.tensor([[[0.0], [10.0]], [[2.0], [8.0]], [[2.0], [8.0]]]), torch.tensor([[1.0], [3.0]])))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_normal_correctness(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return torch.distributions.Normal(x, y).sample([20000])
        expected_mean = 5.0
        expected_std = 10.0
        model_export = M()
        dummy_input = (torch.tensor([expected_mean]), torch.tensor([expected_std]))
        model_onnx = io.BytesIO()
        torch.onnx.export(model_export, dummy_input, model_onnx, opset_version=self.opset_version)
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)
        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)
        assert abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1, 'the gap of mean between ort outputs and expected one is unacceptable.'
        assert abs(abs(actual_std) - expected_std) <= expected_std * 0.1, 'the gap of variance between ort outputs and expected one is unacceptable.'

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_nn_init_normal_correctness(self):
        if False:
            return 10
        expected_mean = 5.0
        expected_std = 10.0

        class M(torch.nn.Module):

            def forward(self):
                if False:
                    i = 10
                    return i + 15
                x = torch.ones([]).new_empty(1, 400, 50)
                torch.nn.init.normal_(x, expected_mean, expected_std)
                return x
        model_export = M()
        model_onnx = io.BytesIO()
        test_inputs = tuple()
        torch.onnx.export(model_export, test_inputs, model_onnx, opset_version=self.opset_version)
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=test_inputs)
        actual_std = np.std(ort_out)
        actual_mean = np.mean(ort_out)
        assert abs(abs(actual_mean) - expected_mean) <= expected_mean * 0.1, 'the gap of mean between ort outputs and expected one is unacceptable.'
        assert abs(abs(actual_std) - expected_std) <= expected_std * 0.1, 'the gap of variance between ort outputs and expected one is unacceptable.'

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return (torch.distributions.Uniform(x, y).sample().size(0), x, y)
        self.run_test(M(), (torch.tensor([0.0]), torch.tensor([10.0])))
        self.run_test(M(), (torch.tensor([[0.0], [6.0]]), torch.tensor([[1.0], [7.0]])))
        self.run_test(M(), (torch.tensor([1.0]), torch.tensor([[10.0], [7.0], [9.0], [20.0]])))

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(11)
    def test_dist_uniform_correctness(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return torch.distributions.Uniform(x, y).sample([10000])
        expected_min = 5.0
        expected_max = 10.0
        expected_mean = (expected_min + expected_max) / 2
        model_export = M()
        dummy_input = (torch.tensor([expected_min]), torch.tensor([expected_max]))
        model_onnx = io.BytesIO()
        torch.onnx.export(model_export, dummy_input, model_onnx, opset_version=self.opset_version)
        ort_sess = verification._ort_session(model_onnx)
        ort_out = verification._run_onnx(ort_sess, inputs=dummy_input)
        actual_min = np.min(ort_out)
        actual_max = np.max(ort_out)
        actual_mean = np.mean(ort_out)
        assert actual_min >= expected_min, 'the minimum value of ort outputs is out of scope.'
        assert actual_max <= expected_max, 'the maximum value of ort outputs is out of scope.'
        assert abs(actual_mean - expected_mean) <= expected_mean * 0.05, 'the mean value of ort outputs is out of scope.'

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_int(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                result = torch.tensor([2 for i in range(x.size()[0])], dtype=torch.int)
                return (x, result)
        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_float(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                result = torch.tensor([1.1 for i in range(x.size()[0])], dtype=torch.float)
                return (x, result)
        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_sequence_to_bool(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                result = torch.tensor([False for i in range(x.size()[0])], dtype=torch.bool)
                return (x, result)
        x = torch.randn(10, 5)
        self.run_test(M(), (x,))

    def test_tuple_output_from_if_with_raised_exception(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, t: Tensor) -> Tuple[Tensor, Tensor]:
                if False:
                    while True:
                        i = 10
                if float(t) < 0:
                    raise Exception('Negative input')
                else:
                    return (torch.zeros(5), torch.zeros(5))
        x = torch.zeros(1)
        self.run_test(torch.jit.script(M()), (x,))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_linear(self):
        if False:
            print('Hello World!')
        model = torch.ao.nn.quantized.Linear(4, 8)
        weight = torch.quantize_per_tensor(torch.arange(32, dtype=torch.float).view(8, 4), 0.5, 0, torch.qint8)
        bias = torch.arange(8, dtype=torch.float)
        model.set_weight_bias(weight, bias)
        input = torch.randn(4, 4)
        input = torch.arange(16, dtype=torch.float).view(4, 4) - 8
        input_tensor = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, input_tensor)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d(self):
        if False:
            while True:
                i = 10
        model = torch.ao.nn.quantized.Conv1d(16, 33, 3, stride=2)
        q_weight = torch.quantize_per_tensor(torch.randn(33, 16, 3), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d(self):
        if False:
            return 10
        model = torch.ao.nn.quantized.Conv2d(16, 33, 3, stride=2)
        q_weight = torch.quantize_per_tensor(torch.randn(33, 16, 3, 3), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d(self):
        if False:
            i = 10
            return i + 15
        model = torch.ao.nn.quantized.Conv3d(16, 33, [2, 3, 4], stride=[3, 1, 2])
        q_weight = torch.quantize_per_tensor(torch.randn(33, 16, 2, 3, 4), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 8, 8, 8)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_adaptive_avg_pool2d(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.AdaptiveAvgPool2d((5, 7))
        input = torch.randn(4, 3, 10, 14)
        q_input = torch.quantize_per_tensor(input, 0.2, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv1d_relu(self):
        if False:
            i = 10
            return i + 15
        model = torch.ao.nn.intrinsic.quantized.ConvReLU1d(16, 33, 3, stride=2)
        q_weight = torch.quantize_per_tensor(torch.randn(33, 16, 3), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv2d_relu(self):
        if False:
            while True:
                i = 10
        model = torch.ao.nn.intrinsic.quantized.ConvReLU2d(16, 33, 3, stride=2)
        q_weight = torch.quantize_per_tensor(torch.randn(33, 16, 3, 3), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv3d_relu(self):
        if False:
            while True:
                i = 10
        model = torch.ao.nn.intrinsic.quantized.ConvReLU3d(16, 33, [2, 3, 4], stride=[3, 1, 2])
        q_weight = torch.quantize_per_tensor(torch.randn(33, 16, 2, 3, 4), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 8, 8, 8)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose1d(self):
        if False:
            print('Hello World!')
        model = torch.ao.nn.quantized.ConvTranspose1d(16, 33, 3, output_padding=1, stride=2)
        q_weight = torch.quantize_per_tensor(torch.randn(16, 33, 3), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_conv_transpose2d(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.ao.nn.quantized.ConvTranspose2d(16, 33, 3, output_padding=(0, 1), stride=2)
        q_weight = torch.quantize_per_tensor(torch.randn(16, 33, 3, 3), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 32, 32)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipIfQuantizationBackendQNNPack
    def test_quantized_conv_transpose3d(self):
        if False:
            print('Hello World!')
        model = torch.ao.nn.quantized.ConvTranspose3d(16, 33, [2, 3, 4], output_padding=(0, 1, 2), stride=[3, 1, 2])
        q_weight = torch.quantize_per_tensor(torch.randn(16, 33, 2, 3, 4), 0.5, 0, torch.qint8)
        bias = torch.arange(33).to(torch.float) - 16
        model.set_weight_bias(q_weight, bias)
        input = torch.randn(3, 16, 8, 8, 8)
        q_input = torch.quantize_per_tensor(input, 0.5, 128, torch.quint8)
        self.run_test(model, q_input)

    @common_utils.parametrize('function_or_module', [common_utils.subtest(torch.nn.ReLU(), name='relu'), common_utils.subtest(torch.nn.LeakyReLU(), name='leaky_relu'), common_utils.subtest(torch.ao.nn.quantized.LeakyReLU(2.0, 1), name='quantized_leaky_relu'), common_utils.subtest(torch.ao.nn.quantized.Hardswish(2.0, 1), name='quantized_hardswish'), common_utils.subtest(torch.nn.Sigmoid(), name='sigmoid'), common_utils.subtest(torch.ao.nn.quantized.Sigmoid(2.0, 1), name='quantized_sigmoid'), common_utils.subtest(torch.nn.Hardsigmoid(), name='hardsigmoid'), common_utils.subtest(torch.nn.Tanh(), name='tanh'), common_utils.subtest(torch.nn.Hardtanh(), name='hardtanh'), common_utils.subtest(lambda x: torch.transpose(x, 0, 1), name='transpose'), common_utils.subtest(lambda x: x.expand(2, 4, 2, 3), name='expand'), common_utils.subtest(lambda x: x.view(1, 4, 6), name='view'), common_utils.subtest(lambda x: x.select(1, 1), name='select'), common_utils.subtest(torch.ao.nn.quantized.LayerNorm([4, 2, 3], torch.nn.Parameter(torch.ones([4, 2, 3])), torch.nn.Parameter(torch.zeros([4, 2, 3])), 2.0, 1), name='layer_norm'), common_utils.subtest(torch.ao.nn.quantized.InstanceNorm1d(2, torch.nn.Parameter(torch.ones(4)), torch.nn.Parameter(torch.zeros(4)), 2.0, 1), name='instance_norm'), common_utils.subtest(torch.ao.nn.quantized.GroupNorm(2, 4, torch.nn.Parameter(torch.zeros(4)), torch.nn.Parameter(torch.zeros(4)), 2.0, 1), name='group_norm'), common_utils.subtest(lambda x: torch.as_strided(x, (2, 2), (1, 2)), name='as_strided')])
    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_unary_ops(self, function_or_module):
        if False:
            print('Hello World!')
        input = torch.randn(1, 4, 2, 3)
        q_input = torch.quantize_per_tensor(input, 0.26, 128, torch.quint8)

        class Model(torch.nn.Module):

            def __init__(self, function_or_module):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.function_or_module = function_or_module

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.function_or_module(x)
        self.run_test(Model(function_or_module), q_input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_flatten(self):
        if False:
            return 10

        class FlattenModel(torch.nn.Module):

            def forward(self, input):
                if False:
                    return 10
                return torch.flatten(input)
        x = torch.quantize_per_tensor(torch.randn(1, 2, 3, 4), 1, 0, torch.quint8)
        self.run_test(FlattenModel(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()
    def test_quantized_cat_when_concatinating_the_same_tensor(self):
        if False:
            i = 10
            return i + 15

        class QuantizedSelfConcatenationModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.ao.nn.quantized.QFunctional().cat((x, x), dim=1)
        q_input = torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8)
        self.run_test(QuantizedSelfConcatenationModel(), q_input)

    @common_utils.parametrize('x, y', [common_utils.subtest([torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8), torch.quantize_per_tensor(torch.zeros(1, 3), 0.26, 128, torch.quint8)], name='different_shape'), common_utils.subtest([torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8), torch.quantize_per_tensor(torch.ones(2, 3), 42, 1, torch.quint8)], name='different_scale'), common_utils.subtest([torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8), torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 63, torch.quint8)], name='different_zero_point'), common_utils.subtest([torch.quantize_per_tensor(torch.ones(2, 3), 0.26, 128, torch.quint8), torch.quantize_per_tensor(torch.ones(2, 3), 0.1, 63, torch.quint8)], name='different_zero_point_and_scale')])
    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()
    def test_quantized_cat(self, x: torch.Tensor, y: torch.Tensor):
        if False:
            return 10

        class QuantizedConcatenationModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return torch.ao.nn.quantized.QFunctional().cat((x, y), dim=0)
        self.run_test(QuantizedConcatenationModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()
    def test_quantized_arithmetic_qfunctional(self):
        if False:
            i = 10
            return i + 15
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
        y = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)

        class ArithmeticModel(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                o = torch.ao.nn.quantized.QFunctional().add(x, y)
                o = torch.ao.nn.quantized.QFunctional().mul(o, x)
                return o
        self.run_test(ArithmeticModel(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantized_arithmetic(self):
        if False:
            i = 10
            return i + 15
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)
        y = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 128, torch.quint8)

        class ArithmeticModel2(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                o = torch.ops.quantized.add(x, y, 0.4, 100)
                o = torch.ops.quantized.mul(o, x, 0.4, 100)
                return o
        self.run_test(ArithmeticModel2(), (x, y))

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_quantize_per_tensor(self):
        if False:
            i = 10
            return i + 15

        class Module(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return (torch.quantize_per_tensor(x, 0.2, 0, torch.qint8), torch.quantize_per_tensor(x, 0.2, 128, torch.quint8))
        x = torch.randn(4, 6)
        self.run_test(Module(), x)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_dequantize(self):
        if False:
            i = 10
            return i + 15

        class Module(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.dequantize(x)
        x = torch.quantize_per_tensor(torch.randn(3, 4), 0.2, 0, torch.qint8)
        self.run_test(Module(), x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_per_channel(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.linear = torch.nn.Linear(4, 3)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    return 10
                x = self.quant(x)
                x = self.linear(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model)
        model.linear.weight = torch.nn.Parameter(_construct_tensor_for_quantization_test((3, 4)))
        model.linear.bias = torch.nn.Parameter(torch.arange(3, dtype=torch.float))
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 4), offset=-8)
        self.run_test(model, input)

    @unittest.skip('ORT fails with Validating no unexpected access using an invalid node_index on torch converted model')
    @skipIfUnsupportedMinOpsetVersion(13)
    def test_quantized_list_of_inputs_with_cat(self):
        if False:
            print('Hello World!')

        class TestModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.quant(x)
                x = torch.cat([x, x], 1)
                x = self.dequant(x)
                return x
        model = TestModel()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model)
        model = torch.ao.quantization.convert(model)
        x = torch.randn(2, 4, 6)
        self.run_test(model, x)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_relu(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.quant(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model)
        model = torch.ao.quantization.convert(model)
        input = torch.randn(8, 4)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.quant(x)
                x = self.conv(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model)
        model.conv.weight = torch.nn.Parameter(_construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2))
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((3, 4, 8, 8), offset=-384, max_val=12)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d_relu(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    return 10
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model)
        model.conv.weight = torch.nn.Parameter(_construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2))
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((3, 4, 8, 8), offset=-384, max_val=12)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_conv2d_relu_fused(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.conv = torch.nn.Conv2d(4, 2, 3, stride=2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.quant(x)
                x = self.conv(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.fuse_modules(model.eval(), [['conv', 'relu']])
        model = torch.ao.quantization.prepare_qat(model.train())
        model.conv.weight = torch.nn.Parameter(_construct_tensor_for_quantization_test((2, 4, 3, 3), max_val=2))
        model.conv.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((3, 4, 8, 8), offset=-384, max_val=12)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(13)
    def test_qat_linear_relu_fused(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.linear = torch.nn.Linear(4, 2)
                self.relu = torch.nn.ReLU()
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.quant(x)
                x = self.linear(x)
                x = self.relu(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.fuse_modules(model.eval(), [['linear', 'relu']])
        model = torch.ao.quantization.prepare_qat(model.train())
        model.linear.weight = torch.nn.Parameter(_construct_tensor_for_quantization_test((2, 4), max_val=2))
        model.linear.bias = torch.nn.Parameter(torch.tensor([0.0, 1.0]))
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((3, 4), offset=-384, max_val=12)
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    def test_qat_maxpool2d(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.quant = torch.ao.quantization.QuantStub()
                self.pool = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.dequant = torch.ao.quantization.DeQuantStub()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.quant(x)
                x = self.pool(x)
                x = self.dequant(x)
                return x
        model = M()
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model.train())
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(10)
    @skipScriptTest()
    def test_qat_avg_pool2d(self):
        if False:
            print('Hello World!')
        model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), torch.nn.AvgPool2d(kernel_size=3, stride=2, padding=1), torch.ao.quantization.DeQuantStub())
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model.train())
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 4, 3, 2))
        self.run_test(model, input)

    @skipIfUnsupportedMinOpsetVersion(11)
    def test_qat_upsample_nearest2d(self):
        if False:
            print('Hello World!')
        model = torch.nn.Sequential(torch.ao.quantization.QuantStub(), torch.nn.UpsamplingNearest2d(scale_factor=1.5), torch.ao.quantization.DeQuantStub())
        model.qconfig = torch.ao.quantization.get_default_qconfig('fbgemm')
        model = torch.ao.quantization.prepare_qat(model.train())
        model = torch.ao.quantization.convert(model)
        input = _construct_tensor_for_quantization_test((4, 3, 2, 2))
        self.run_test(model, input)

    def test_0d_tensor_broadcast(self):
        if False:
            i = 10
            return i + 15

        class fn(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                a = torch.add(x, y)
                b = torch.mul(y, y)
                return a + b
        x = torch.ones(0)
        y = torch.ones(1)
        self.run_test(fn(), (x, y), input_names=['x', 'y'], output_names=['output'])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_convolution_allow_tf32(self):
        if False:
            while True:
                i = 10

        class Module(torch.nn.Module):

            def __init__(self, allow_tf32):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.allow_tf32 = allow_tf32
                weight = torch.rand(32, 3, 3, 3)
                self.weight = torch.nn.Parameter(weight)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                if self.allow_tf32:
                    return torch._convolution(x, self.weight, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1, False, False, True, True)
                else:
                    return torch._convolution(x, self.weight, None, [2, 2], [0, 0], [1, 1], False, [0, 0], 1, False, False, True)
        x = torch.randn(1, 3, 224, 224)
        self.run_test(Module(False), x, rtol=0.001, atol=1e-06)
        self.run_test(Module(True), x, rtol=0.001, atol=1e-06)

    @skipIfUnsupportedMinOpsetVersion(16)
    @common_utils.parametrize('mode', ('bilinear', 'nearest', 'bicubic'))
    @common_utils.parametrize('padding_mode', ('zeros', 'border', 'reflection'))
    @common_utils.parametrize('align_corners', (True, False), name_fn=lambda align_corners: str(align_corners))
    def test_grid_sample(self, mode, padding_mode, align_corners):
        if False:
            for i in range(10):
                print('nop')
        (n, c, h_in, w_in, h_out, w_out) = (1, 1, 3, 2, 2, 4)

        class GridSampleModule(torch.nn.Module):

            def __init__(self, mode, padding_mode, align_corners) -> None:
                if False:
                    return 10
                super().__init__()
                (self.mode, self.padding_mode, self.align_corners) = (mode, padding_mode, align_corners)

            def forward(self, input, grid):
                if False:
                    print('Hello World!')
                return torch.nn.functional.grid_sample(input, grid, self.mode, self.padding_mode, self.align_corners)
        atol_rtol = {}
        if (mode, padding_mode) == ('bicubic', 'border'):
            if align_corners:
                atol_rtol.update({'atol': 0.3, 'rtol': 0.4})
            else:
                atol_rtol.update({'atol': 0.02, 'rtol': 0.02})
        (input, grid) = (torch.randn(n, c, h_in, w_in), torch.randn(n, h_out, w_out, 2))
        self.run_test(GridSampleModule(mode, padding_mode, align_corners), (input, grid), **atol_rtol)
        d_in = 2
        d_out = 3
        volumetric_input_tensor = torch.randn(n, c, d_in, h_in, w_in)
        volumetric_grid_tensor = torch.randn(n, d_out, h_out, w_out, 3)
        for (mode, padding_mode, align_corners) in itertools.product(('bilinear', 'nearest'), ('zeros', 'border', 'reflection'), (True, False)):
            with self.assertRaises(torch.onnx.errors.OnnxExporterError):
                self.run_test(GridSampleModule(mode, padding_mode, align_corners), (volumetric_input_tensor, volumetric_grid_tensor), **atol_rtol)

    class IfNoneInput(torch.nn.Module):

        def forward(self, x) -> Optional[Tensor]:
            if False:
                print('Hello World!')
            y: Optional[Tensor] = None
            if x.size(0) > 1:
                y = x
            return y

    class IfNoneOutput(torch.nn.Module):

        def forward(self, x) -> Optional[Tensor]:
            if False:
                i = 10
                return i + 15
            y: Optional[Tensor] = x
            if x.size(0) > 1:
                y = None
            return y

    class LoopNoneInput(torch.nn.Module):

        def forward(self, x) -> Optional[Tensor]:
            if False:
                return 10
            y: Optional[Tensor] = None
            for _ in range(x.size(0)):
                y = x
            return y

    class LoopNoneOutput(torch.nn.Module):

        def forward(self, x) -> Optional[Tensor]:
            if False:
                while True:
                    i = 10
            y: Optional[Tensor] = x
            for _ in range(x.size(0)):
                y = None
            return y

    @common_utils.parametrize('module_class', (IfNoneOutput, IfNoneInput, LoopNoneOutput, LoopNoneInput), name_fn=lambda module_class: module_class.__name__)
    @common_utils.parametrize('x_size', (0, 1), name_fn=lambda x_size: str(x_size))
    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_optional_output(self, module_class: Type[torch.nn.Module], x_size: int):
        if False:
            print('Hello World!')
        model = torch.jit.script(module_class())
        f = io.BytesIO()
        x = torch.ones(x_size)
        dynamic_axis_name = 'condition'
        torch.onnx.export(model, x, f, opset_version=self.opset_version, dynamic_axes={'x': {0: dynamic_axis_name}}, input_names=['x'])
        exported = onnx.load_from_string(f.getvalue())
        expected_elem_type = torch.onnx.JitScalarType.from_value(x).onnx_type()
        expected_output_type = onnx.helper.make_optional_type_proto(onnx.helper.make_tensor_type_proto(expected_elem_type, (dynamic_axis_name,)))
        self.assertEqual(expected_output_type, exported.graph.output[0].type)
        for node in exported.graph.node:
            if node.op_type == 'If':
                for attr in node.attribute:
                    if attr.name in ('then_branch', 'else_branch'):
                        self.assertEqual(expected_output_type, attr.g.output[0].type)
        self.run_test(module_class(), x, dynamic_axes={'x': {0: dynamic_axis_name}}, input_names=['x'])

    @skipTraceTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    def test_uninitialized_optional(self):
        if False:
            for i in range(10):
                print('nop')

        class Module(torch.nn.Module):

            def forward(self, y: Optional[Tensor]) -> Optional[Tensor]:
                if False:
                    i = 10
                    return i + 15
                if y is not None:
                    if y.shape[1] < 5:
                        if y.size(0) == 1:
                            y = y + 4
                        else:
                            return y
                return y
        self.run_test(Module(), torch.ones((3, 4), dtype=torch.int), dynamic_axes={'y': {0: 'y0', 1: 'y1'}}, input_names=['y'])

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_device_eq(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                if a.device != torch.device('cpu'):
                    return a
                return torch.zeros_like(a)
        mod = torch.jit.script(M())
        self.run_test(mod, torch.randn(3, 3, device='cpu'), input_names=['a'], dynamic_axes={'a': {0: 'a0'}})

    @skipIfUnsupportedMinOpsetVersion(9)
    def test_lerp(self):
        if False:
            for i in range(10):
                print('nop')

        class LerpModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return (x.lerp(torch.full_like(x, 10), 0.4), x.lerp(torch.full_like(x, 20), 0.7), x.lerp(torch.full_like(x, 30), torch.tensor(0.4)), x.lerp(torch.full_like(x, 40), x / 10.0), x.lerp(torch.tensor(10.0), x / 10.0), x.lerp(torch.tensor(10.0), 0.4), x.lerp(torch.tensor(10.0), torch.tensor(0.4)))
        self.run_test(LerpModel(), torch.rand(5, 4, 3))

    @common_utils.parametrize('input_dtype', [torch.cfloat, torch.float])
    @skipIfUnsupportedMinOpsetVersion(9)
    def test_print_tensor_within_torch_nn_module(self, input_dtype: torch.dtype):
        if False:
            return 10

        class PrintTensorOnMyModel(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                x_firsts = x[:, 0]
                print(f'x_firsts: {x_firsts}')
                _: List[float] = x.tolist()
                return x_firsts
        m = PrintTensorOnMyModel()
        x = torch.randn(10, 5, dtype=input_dtype)
        if input_dtype == torch.cfloat:
            with self.assertRaises(RuntimeError):
                self.run_test(m, x)
        else:
            self.run_test(m, x)

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(16)
    @unittest.skipIf(not torch.hub._check_module_exists('torch_geometric'), 'torch_geometric not installed.')
    def test_sage_conv(self):
        if False:
            while True:
                i = 10
        from torch_geometric import nn as torch_geometric_nn
        coords0 = torch.randn(1, 6)
        coords1 = torch.randn(1, 6)
        coords = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)
        adj = torch_geometric_nn.knn_graph(coords, k=2, batch=None, loop=True)
        edge_from = adj[0:1, :]
        edge_to = adj[1:, :]
        inputs = (coords0, coords1, edge_from, edge_to)

        class MySAGEConv(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.SAGEConvBlock1 = torch_geometric_nn.SAGEConv(2, 512, normalize=True)
                self.bano1 = torch_geometric_nn.BatchNorm(512)
                self.relu = torch.nn.ReLU()
                self.dense1 = torch.nn.Seq(Lin(512, 1))
                self.sigmoid = torch.nn.Sigmoid()

            def forward(self, coords0, coords1, edge_from, edge_to):
                if False:
                    while True:
                        i = 10
                adj = torch.cat((edge_from, edge_to), dim=0)
                gra = torch.transpose(torch.cat((coords0, coords1), dim=0), 0, 1)
                x1 = self.SAGEConvBlock1(gra, edge_index=adj)
                x = torch.unsqueeze(torch.sum(x1), dim=0)
                return x
        input_names = ['coords0', 'coords1', 'edge_from', 'edge_to']
        output_names = ['outputs']
        dynamic_axes = {'coords0': {0: 'batch_size', 1: 'features'}, 'coords1': {0: 'batch_size', 1: 'features'}, 'edge_from': {0: 'batch_size', 1: 'features'}, 'edge_to': {0: 'batch_size', 1: 'features'}, 'outputs': {0: 'batch_size'}}
        self.run_test(MySAGEConv(), inputs, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes)

    @skipScriptTest()
    @skipIfUnsupportedMinOpsetVersion(9)
    @common_utils.parametrize('name, nonlinearity', [('elman', 'relu'), ('elman', 'tanh'), ('lstm', None), ('gru', None)])
    @common_utils.parametrize(**_parametrize_rnn_args('layers'))
    @common_utils.parametrize(**_parametrize_rnn_args('bidirectional'))
    @common_utils.parametrize(**_parametrize_rnn_args('initial_state'))
    @common_utils.parametrize(**_parametrize_rnn_args('packed_sequence'))
    @common_utils.parametrize(**_parametrize_rnn_args('dropout'))
    def test_rnn(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self._dispatch_rnn_test(*args, **kwargs)
if __name__ == '__main__':
    common_utils.TestCase._default_dtype_check_enabled = True
    common_utils.run_tests()