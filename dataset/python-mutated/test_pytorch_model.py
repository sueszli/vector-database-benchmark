from __future__ import absolute_import as _
from __future__ import division as _
from __future__ import print_function as _
from __future__ import unicode_literals as _
import unittest
from coremltools._deps import _HAS_ONNX, MSG_ONNX_NOT_FOUND, _IS_MACOS
if _HAS_ONNX:
    import onnx
    from coremltools.converters.onnx import convert
    from coremltools.converters.onnx._converter import SupportedVersion
    from ._test_utils import _assert_outputs
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import shutil
import tempfile
import os
import pytest
from coremltools.models.utils import _macos_version
np.random.seed(10)
torch.manual_seed(10)
MIN_MACOS_VERSION_10_15 = (10, 15)
DEBUG = False

def _test_torch_model_single_io(torch_model, torch_input_shape, coreml_input_shape, minimum_ios_deployment_target='12', decimal=4, opset_version=9):
    if False:
        while True:
            i = 10
    torch_input = torch.rand(*torch_input_shape)
    torch_out_raw = torch_model(torch_input)
    if isinstance(torch_out_raw, tuple):
        torch_out = torch_out_raw[0].detach().numpy()
    else:
        torch_out = torch_out_raw.detach().numpy()
    model_dir = tempfile.mkdtemp()
    if DEBUG:
        model_dir = '/tmp'
    onnx_file = os.path.join(model_dir, 'torch_model.onnx')
    torch.onnx.export(torch_model, torch_input, onnx_file, opset_version=opset_version)
    onnx_model = onnx.load(onnx_file)
    coreml_model = convert(onnx_model, minimum_ios_deployment_target=minimum_ios_deployment_target)
    output_name = [o.name for o in onnx_model.graph.output][0]
    initializer_names = {t.name for t in onnx_model.graph.initializer}
    input_name = [i.name for i in onnx_model.graph.input if i.name not in initializer_names][0]
    input_numpy = torch_input.detach().numpy()
    if SupportedVersion.is_nd_array_supported(minimum_ios_deployment_target):
        input_dict = {input_name: input_numpy}
    else:
        input_dict = {input_name: np.reshape(input_numpy, coreml_input_shape)}
    if _IS_MACOS:
        coreml_out = coreml_model.predict(input_dict, useCPUOnly=True)[output_name]
        if DEBUG:
            coreml_model.save(model_dir + '/torch_model.mlmodel')
            print('coreml_out')
            print(np.squeeze(coreml_out))
            print('torch_out')
            print(np.squeeze(torch_out))
            print('coreml out shape ', coreml_out.shape)
            print('torch out shape: ', torch_out.shape)
        _assert_outputs([torch_out], [coreml_out], decimal=decimal)
        if not DEBUG:
            if os.path.exists(model_dir):
                shutil.rmtree(model_dir)

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class OnnxModelTest(unittest.TestCase):

    def test_functional_average_pool(self, minimum_ios_deployment_target='12'):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(Net, self).__init__()

            def forward(self, x):
                if False:
                    return 10
                y = F.avg_pool2d(x, [15, 18], [15, 18])
                return y
        torch_model = Net()
        torch_model.train(False)
        if minimum_ios_deployment_target == '12':
            coreml_shape = (1, 64, 64)
        else:
            coreml_shape = (1, 1, 64, 64)
        _test_torch_model_single_io(torch_model, (1, 1, 64, 64), coreml_shape, minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_functional_average_pool_disable_rank5_mapping(self):
        if False:
            return 10
        self.test_functional_average_pool(minimum_ios_deployment_target='13')

    def test_linear_no_bias(self, minimum_ios_deployment_target='12'):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(Net, self).__init__()
                self.simple_nn = nn.Sequential(nn.Linear(256, 128, bias=False), nn.ReLU())

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.simple_nn(x)
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 256), 256, minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_linear_no_bias_disable_rank5_mapping(self):
        if False:
            i = 10
            return i + 15
        self.test_linear_no_bias(minimum_ios_deployment_target='13')

    def test_linear_bias(self):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.simple_nn = nn.Sequential(nn.Linear(256, 128, bias=True), nn.ReLU())

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.simple_nn(x)
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 256), 256)

    def test_dynamic_reshape(self):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super(Net, self).__init__()
                self.conv = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), stride=1, padding=0, bias=True)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                x = x.view(x.size()[0], -1)
                return x
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3, 100, 100), (3, 100, 100), '13')

    def test_const_initializer1(self):
        if False:
            for i in range(10):
                print('nop')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.ones = torch.nn.Parameter(torch.ones(1))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                y = x + self.ones
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3), (3,))

    def test_const_initializer2(self):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(Net, self).__init__()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = x + torch.nn.Parameter(torch.ones(2, 3), requires_grad=False)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 2, 3), (1, 2, 3))

    def test_conv2D_transpose(self):
        if False:
            for i in range(10):
                print('nop')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, output_padding=0, padding=3, groups=1)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                y = self.convT(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 1, 64, 64), (1, 64, 64))

    def test_conv2D_transpose_output_padding(self):
        if False:
            for i in range(10):
                print('nop')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, output_padding=1, padding=3, groups=1)

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = self.convT(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 1, 64, 64), (1, 64, 64))

    def test_conv2D_transpose_groups(self):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(4, 4, kernel_size=3, stride=2, output_padding=1, padding=1, groups=2)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = self.convT(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 4, 8, 8), (4, 8, 8))

    def test_conv2D_transpose_2(self):
        if False:
            for i in range(10):
                print('nop')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.convT = torch.nn.ConvTranspose2d(1, 1, kernel_size=3, stride=3, output_padding=2, padding=1, groups=1)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                y = self.convT(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 1, 3, 3), (1, 3, 3))

    def test_pow(self):
        if False:
            return 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(Net, self).__init__()

            def forward(self, x):
                if False:
                    return 10
                y = x.pow(3)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (3, 2, 3), (3, 2, 3))

    @pytest.mark.skip(reason='rdar://64224329')
    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_lstm(self):
        if False:
            print('Hello World!')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super(Net, self).__init__()
                self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=1)

            def forward(self, x):
                if False:
                    return 10
                y = self.lstm(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (3, 1, 256), (3, 1, 256), minimum_ios_deployment_target='13')

    @pytest.mark.skip(reason='rdar://64224329')
    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_bidirlstm(self):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.lstm = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, bidirectional=True)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                y = self.lstm(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (3, 1, 256), (3, 1, 256), minimum_ios_deployment_target='13')

    @pytest.mark.skip(reason='rdar://64224329')
    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_gru(self):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.gru = nn.GRU(input_size=256, hidden_size=64, num_layers=1)

            def forward(self, x):
                if False:
                    return 10
                y = self.gru(x)
                return y
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (3, 1, 256), (3, 1, 256), minimum_ios_deployment_target='13', decimal=1)

    def test_1d_conv(self):
        if False:
            print('Hello World!')

        class Net(nn.Module):

            def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
                if False:
                    return 10
                super(Net, self).__init__()
                self.conv = torch.nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=0, dilation=dilation, groups=groups, bias=bias)
                self.__padding = (kernel_size - 1) * dilation

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                result = self.conv(x)
                if self.__padding != 0:
                    return result[:, :, :-self.__padding]
                return result
        B = 1
        Cin = 5
        Cout = 11
        k = 3
        Win = 15
        torch_model = Net(Cin, Cout, k)
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, Cin, Win), (Cin, 1, Win))

    def test_conv1d_after_reshape(self):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(Net, self).__init__()
                self.conv = torch.nn.Conv1d(in_channels=300, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = x.view(1, 300, 100)
                x = self.conv(x)
                return x
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3, 100, 100), (3, 100, 100))

    def test_conv2d_stride(self):
        if False:
            for i in range(10):
                print('nop')

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                in_channels = 1
                out_channels = 1
                bsz = 1
                super(TestModule, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 4), stride=1)
                self.conv2 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 5), stride=(2, 1), padding=(1, 2))

            def forward(self, x):
                if False:
                    return 10
                return (self.conv2(x),)
        torch_model = TestModule()
        torch_model.train(False)
        (H, W) = (6, 3)
        _test_torch_model_single_io(torch_model, (1, 1, H, W), (1, H, W))

    def test_conv2d_dilation(self):
        if False:
            for i in range(10):
                print('nop')

        class TestModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                in_channels = 1
                out_channels = 3
                bsz = 1
                super(TestModule, self).__init__()
                self.conv1 = torch.nn.Conv2d(in_channels, out_channels, kernel_size=(3, 4), stride=2, dilation=2)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.conv1(x)
        torch_model = TestModule()
        torch_model.train(False)
        (H, W) = (64, 64)
        _test_torch_model_single_io(torch_model, (1, 1, H, W), (1, H, W))

    def test_bachnorm_after_reshape(self):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Net, self).__init__()
                self.conv = torch.nn.Conv1d(in_channels=300, out_channels=32, kernel_size=3, stride=1, padding=0, bias=True)
                self.bachnorm = nn.BatchNorm1d(32)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = x.view(1, 300, 100)
                x = self.conv(x)
                x = self.bachnorm(x)
                return x
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3, 100, 100), (3, 100, 100))

    def test_res_connect_downsampling_after_reshape(self):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super(Net, self).__init__()
                self.conv = torch.nn.Conv1d(in_channels=300, out_channels=32, kernel_size=3, stride=1, padding=1, bias=True)
                self.downsample = torch.nn.Conv1d(in_channels=300, out_channels=32, kernel_size=1, stride=1, padding=0, bias=True)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = x.view(1, 300, 100)
                y = self.conv(x)
                res = self.downsample(x)
                return y + res
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 3, 100, 100), (3, 100, 100))

    def test_fc_plus_convenet(self):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def __init__(self, channel_size=1, output_h=16, output_w=16, filter_num=32, latent_size=16):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.channel_size = channel_size
                self.output_h = output_h
                self.output_w = output_w
                self.filter_num = filter_num
                self.latent_size = latent_size
                self.fc3 = nn.Linear(latent_size, 128)
                self.fc4 = nn.Linear(128, 256)
                self.relu = nn.ReLU()
                self.convt = nn.Sequential(nn.ConvTranspose2d(256, self.filter_num * 4, 4, 1), nn.BatchNorm2d(self.filter_num * 4), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.filter_num * 4, self.filter_num * 2, 4, 1), nn.BatchNorm2d(self.filter_num * 2), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.filter_num * 2, self.filter_num, 4, 1), nn.BatchNorm2d(self.filter_num), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.filter_num, self.filter_num, 4, 1), nn.BatchNorm2d(self.filter_num), nn.ReLU(inplace=True), nn.ConvTranspose2d(self.filter_num, 1, 4, 1), nn.Sigmoid())

            def forward(self, z):
                if False:
                    while True:
                        i = 10
                x = self.relu(self.fc3(z))
                deconv_input = self.fc4(x)
                deconv_input = deconv_input.view(-1, 256, 1, 1)
                x = self.convt(deconv_input)
                return x
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 16), (1, 1, 16))

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_conv1d_pool1d(self, minimum_ios_deployment_target='13'):
        if False:
            print('Hello World!')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.conv1 = nn.Conv1d(in_channels=4, out_channels=32, kernel_size=3, stride=1, padding=1)
                self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x.permute(0, 2, 1)
                x = self.conv1(x)
                x = F.relu(x)
                x = F.max_pool1d(x, 2)
                x = self.conv2(x)
                x = F.relu(x)
                return x
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (2, 10, 4), (2, 10, 4), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_slice(self, minimum_ios_deployment_target='13'):
        if False:
            for i in range(10):
                print('nop')

        class Net(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super(Net, self).__init__()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = x[:, :5] + x[:, 5:]
                return x
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (10, 10), (10, 10), minimum_ios_deployment_target=minimum_ios_deployment_target)
        _test_torch_model_single_io(torch_model, (10, 10), (10, 10), opset_version=10, minimum_ios_deployment_target=minimum_ios_deployment_target)

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class ReshapeTransposeTests(unittest.TestCase):
    """
    tests for models that have patterns like:
    rank(4) ---> reshape (rank 6) ----> transpose (rank 6) ----> reshape(4)
    """

    @pytest.mark.xfail
    def test_pixel_shuffle_not_working(self):
        if False:
            i = 10
            return i + 15
        '\n        (1, c, h, w) --> reshape ---> (1, sh, sw, c/(sh*sw), h, w)\n        --> transpose [0,1,4,2,5,3] ---> (1, sh, h, sw, w, c/(sh*sw))\n        --> reshape ---> (1, c/(s1*s2), sh*h, sw*w)\n        '

        class Net(nn.Module):

            def __init__(self, upscale_factor=3):
                if False:
                    while True:
                        i = 10
                super(Net, self).__init__()
                self.upscale_factor = upscale_factor
                self.ps = nn.PixelShuffle(self.upscale_factor)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.ps(x)
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 18, 4, 5), (18, 4, 5))

    def test_pixel_shuffle_working(self):
        if False:
            while True:
                i = 10
        '\n        (1, c, h, w) --> reshape ---> (1, c/(sh*sw), sh, sw, h, w)\n        --> transpose [0,1,4,2,5,3] ---> (1, sh, h, sw, w, c/(sh*sw))\n        --> reshape ---> (1, c/(sh*sw), sh*h, sw*w)\n        '

        class Net(nn.Module):

            def __init__(self, C=12, H=4, W=6, sh=3, sw=2):
                if False:
                    while True:
                        i = 10
                super(Net, self).__init__()
                self.C = C
                self.H = H
                self.W = W
                self.sh = sh
                self.sw = sw

            def forward(self, x):
                if False:
                    return 10
                y1 = x.view(1, self.C // (self.sh * self.sw), self.sh, self.sw, self.H, self.W).contiguous()
                y2 = y1.permute(0, 1, 4, 2, 5, 3).contiguous()
                y3 = y2.view(1, self.C // (self.sh * self.sw), self.sh * self.H, self.sw * self.W).contiguous()
                return y3
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 12, 4, 6), (12, 4, 6))

    def test_reorganize_1(self):
        if False:
            print('Hello World!')
        '\n        (1, c, h, w) --> reshape ---> (1, c/(sh*sw), h, sh, w, sw)\n        --> transpose [0,3,5,1,2,4] ---> (1, sh, sw, c/(sh*sw), h, w)\n        --> reshape ---> (1, c*sh*sw, h/sh, w/sw)\n        '

        class Net(nn.Module):

            def __init__(self, C=12, H=4, W=6, sh=2, sw=3):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.C = C
                self.H = H
                self.W = W
                self.sh = sh
                self.sw = sw

            def forward(self, x):
                if False:
                    return 10
                y1 = x.view(1, self.C // (self.sh * self.sw), self.H, self.sh, self.W, self.sw).contiguous()
                y2 = y1.permute(0, 3, 5, 1, 2, 4).contiguous()
                y3 = y2.view(1, self.C * (self.sh * self.sw), self.H // self.sh, self.W // self.sw).contiguous()
                return y3
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 12, 4, 6), (12, 4, 6))

    def test_reorganize_2(self):
        if False:
            print('Hello World!')
        '\n        (1, c, h, w) --> reshape ---> (1, c, h/sh, sh, w/sw, sw)\n        --> transpose [0,1,2,4,3,5] ---> (1, c, h/sh, w/sw, sh, sw)\n        --> reshape ---> (1, c*sh*sw, h/sh, w/sw)\n        '

        class Net(nn.Module):

            def __init__(self, C=12, H=4, W=6, sh=2, sw=3):
                if False:
                    print('Hello World!')
                super(Net, self).__init__()
                self.C = C
                self.H = H
                self.W = W
                self.sh = sh
                self.sw = sw

            def forward(self, x):
                if False:
                    print('Hello World!')
                y1 = x.view(1, self.C, self.H // self.sh, self.sh, self.W // self.sw, self.sw).contiguous()
                y2 = y1.transpose(4, 3).contiguous()
                y3 = y2.view(1, self.C * (self.sh * self.sw), self.H // self.sh, self.W // self.sw).contiguous()
                return y3
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 12, 4, 6), (12, 4, 6))

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class UnaryOperationTests(unittest.TestCase):
    """
    Unary Operation Test cases
    """

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_sqrt_tensor(self, minimum_ios_deployment_target='13'):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.sqrt(x)
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class OperatorTests(unittest.TestCase):
    """
    Operator test for Operator
    """

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_repeat(self, minimum_ios_deployment_target='13'):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.repeat([2, 3, 1])
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class BinaryOperationTests(unittest.TestCase):
    """
    Binary Operation Test cases
    """

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_add_same_shape(self, minimum_ios_deployment_target='13'):
        if False:
            print('Hello World!')

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.add(x, y)
        y = torch.rand((18, 4, 5))
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_add_same_shape_multiple(self, minimum_ios_deployment_target='13'):
        if False:
            return 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y + y1 + y2 + y3
        y = torch.rand((18, 4, 5))
        y1 = torch.rand((4, 5))
        y2 = torch.rand((18, 4, 5))
        y3 = 7.234
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_add_tensor_scalar(self, minimum_ios_deployment_target='13'):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.add(x, y)
        y = 5
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_add_diff_shape(self, minimum_ios_deployment_target='13'):
        if False:
            return 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.add(x, y)
        y = torch.rand((4, 5))
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_sub_same_shape(self, minimum_ios_deployment_target='13'):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.sub(x, y)
        y = torch.rand((18, 4, 5))
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_sub_same_shape_multiple(self, minimum_ios_deployment_target='13'):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x - y - y1 - y2 - y3
        y = torch.rand((18, 4, 5))
        y1 = torch.rand((4, 5))
        y2 = torch.rand((18, 4, 5))
        y3 = 7.234
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_sub_tensor_scalar(self, minimum_ios_deployment_target='13'):
        if False:
            i = 10
            return i + 15

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.sub(x, y)
        y = 5
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_sub_diff_shape(self, minimum_ios_deployment_target='13'):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.sub(x, y)
        y = torch.rand((4, 5))
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_bianry_ops_mix_test(self, minimum_ios_deployment_target='13'):
        if False:
            print('Hello World!')

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return (x * g + a - d * (c + b) + (a * e - g) / e) / f
        a = torch.rand((18, 4, 5))
        b = torch.rand((4, 5))
        c = torch.rand((18, 4, 5))
        d = 7.234
        e = torch.rand(5)
        f = 8.234
        g = 5
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class ReduceOperationTests(unittest.TestCase):
    """
    Reduction Operation Test cases
    """

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_reducesum(self, minimum_ios_deployment_target='13'):
        if False:
            while True:
                i = 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return x.sum(dim=0)
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (4, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    def test_reducemean(self, minimum_ios_deployment_target='13'):
        if False:
            return 10

        class Net(nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return x.mean(dim=1)
        torch_model = Net()
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (18, 4, 5), (18, 5), minimum_ios_deployment_target=minimum_ios_deployment_target)

@unittest.skipUnless(_HAS_ONNX, MSG_ONNX_NOT_FOUND)
class TransformationTests(unittest.TestCase):
    """
    Test cases for validating transformations
    """

    @unittest.skipIf(_macos_version() < MIN_MACOS_VERSION_10_15, 'macOS 10.15+ required. Skipping test.')
    @pytest.mark.skip(reason='test failure: <rdar://63138211>')
    def test_cast_removal_transformation(self, minimum_ios_deployment_target='13'):
        if False:
            return 10
        torch_model = nn.Upsample(scale_factor=2)
        torch_model.train(False)
        _test_torch_model_single_io(torch_model, (1, 18, 4, 5), (1, 18, 8, 10), minimum_ios_deployment_target=minimum_ios_deployment_target)