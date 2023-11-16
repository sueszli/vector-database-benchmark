import sys
from coremltools.converters.mil import testing_reqs
from coremltools.converters.mil.testing_reqs import *
from .testing_utils import *
backends = testing_reqs.backends
torch = pytest.importorskip('torch')
pytestmark = pytest.mark.skipif(sys.version_info >= (3, 8), reason='Segfault with Python 3.8+')

class TestBatchNorm:

    @pytest.mark.parametrize('num_features, eps, backend', itertools.product([5, 3, 2, 1], [0.1, 1e-05, 1e-09], backends))
    def test_batchnorm(self, num_features, eps, backend):
        if False:
            i = 10
            return i + 15
        model = nn.BatchNorm2d(num_features, eps)
        run_compare_torch((1, num_features, 5, 5), model, backend=backend)

class TestLinear:

    @pytest.mark.parametrize('in_features, out_features, backend', itertools.product([10, 25, 100], [3, 6], backends))
    def test_addmm(self, in_features, out_features, backend):
        if False:
            while True:
                i = 10
        model = nn.Linear(in_features, out_features)
        run_compare_torch((1, in_features), model, backend=backend)

class TestConv:

    @pytest.mark.parametrize('height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend', itertools.product([5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], [1, 3], backends))
    def test_convolution2d(self, height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend, groups=1):
        if False:
            for i in range(10):
                print('nop')
        model = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        run_compare_torch((1, in_channels, height, width), model, backend=backend)

    @pytest.mark.parametrize('height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend', itertools.product([5, 6], [5, 7], [1, 3], [1, 3], [1, 3], [2, 3], [0, 1], [1, 3], backends))
    def test_convolution_transpose2d(self, height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, backend, groups=1):
        if False:
            return 10
        model = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation)
        run_compare_torch((1, in_channels, height, width), model, backend=backend)

    @pytest.mark.parametrize('height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, backend', list(itertools.product([10], [10], [1, 3], [1, 3], [1, 3], [1, 2, 3], [1, 3], [1, 2], [1, 2, (1, 2)], backends)) + [pytest.param(5, 5, 1, 1, 3, 4, 1, 1, 2, 'nn_proto', marks=pytest.mark.xfail), pytest.param(5, 5, 1, 1, 3, 2, 1, 3, 2, 'nn_proto', marks=pytest.mark.xfail)])
    def test_convolution_transpose2d_output_padding(self, height, width, in_channels, out_channels, kernel_size, stride, padding, dilation, output_padding, backend, groups=1):
        if False:
            while True:
                i = 10
        if isinstance(output_padding, int):
            if output_padding >= stride and output_padding >= dilation:
                return
        elif isinstance(output_padding, tuple):
            for _output_padding in output_padding:
                if _output_padding >= stride and _output_padding >= dilation:
                    return
        model = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, output_padding=output_padding)
        run_compare_torch((1, in_channels, height, width), model, backend=backend)

class TestLoop:

    @pytest.mark.parametrize('backend', backends)
    def test_for_loop(self, backend):
        if False:
            i = 10
            return i + 15

        class TestLayer(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super(TestLayer, self).__init__()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = 2.0 * x
                return x

        class TestNet(nn.Module):
            input_size = (64,)

            def __init__(self):
                if False:
                    print('Hello World!')
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                for _ in range(7):
                    x = self.layer(x)
                return x
        model = TestNet().eval()
        torch_model = torch.jit.script(model)
        run_compare_torch(model.input_size, torch_model, backend=backend)

    @pytest.mark.parametrize('backend', backends)
    def test_while_loop(self, backend):
        if False:
            i = 10
            return i + 15

        class TestLayer(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super(TestLayer, self).__init__()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = 0.5 * x
                return x

        class TestNet(nn.Module):
            input_size = (1,)

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                while x > 0.01:
                    x = self.layer(x)
                return x
        model = TestNet().eval()
        torch_model = torch.jit.script(model)
        run_compare_torch(model.input_size, torch_model, backend=backend)

class TestUpsample:

    @pytest.mark.parametrize('output_size, align_corners, backend', [x for x in itertools.product([(10, 10), (1, 1), (20, 20), (2, 3), (190, 170)], [True, False], backends)])
    def test_upsample_bilinear2d_with_output_size(self, output_size, align_corners, backend):
        if False:
            i = 10
            return i + 15
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(nn.functional.interpolate, {'size': output_size, 'mode': 'bilinear', 'align_corners': align_corners})
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('scales_h, scales_w, align_corners, backend', [x for x in itertools.product([2, 3, 4.5], [4, 5, 5.5], [True, False], backends)])
    def test_upsample_bilinear2d_with_scales(self, scales_h, scales_w, align_corners, backend):
        if False:
            i = 10
            return i + 15
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(nn.functional.interpolate, {'scale_factor': (scales_h, scales_w), 'mode': 'bilinear', 'align_corners': align_corners})
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('output_size, backend', [x for x in itertools.product([(10, 10), (30, 20), (20, 20), (20, 30), (190, 170)], backends)])
    def test_upsample_nearest2d_with_output_size(self, output_size, backend):
        if False:
            print('Hello World!')
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(nn.functional.interpolate, {'size': output_size, 'mode': 'nearest'})
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('scales_h, scales_w, backend', [x for x in itertools.product([2, 3, 5], [4, 5, 2], backends)])
    def test_upsample_nearest2d_with_scales(self, scales_h, scales_w, backend):
        if False:
            for i in range(10):
                print('nop')
        input_shape = (1, 3, 10, 10)
        model = ModuleWrapper(nn.functional.interpolate, {'scale_factor': (scales_h, scales_w), 'mode': 'nearest'})
        run_compare_torch(input_shape, model, backend=backend)

class TestBranch:

    @pytest.mark.parametrize('backend', backends)
    def test_if(self, backend):
        if False:
            i = 10
            return i + 15

        class TestLayer(nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super(TestLayer, self).__init__()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = torch.mean(x)
                return x

        class TestNet(nn.Module):
            input_size = (64,)

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super(TestNet, self).__init__()
                layer = TestLayer()
                self.layer = torch.jit.trace(layer, torch.rand(self.input_size))

            def forward(self, x):
                if False:
                    return 10
                m = self.layer(x)
                if m < 0:
                    scale = -2.0
                else:
                    scale = 2.0
                x = scale * x
                return x
        model = TestNet().eval()
        torch_model = torch.jit.script(model)
        run_compare_torch(model.input_size, torch_model, backend=backend)

class TestAvgPool:

    @pytest.mark.xfail(reason='rdar://problem/61064173')
    @pytest.mark.parametrize('input_shape, kernel_size, stride, pad, include_pad, backend', itertools.product([(1, 3, 15), (1, 1, 7), (1, 3, 10)], [1, 2, 3], [1, 2], [0, 1], [True, False], backends))
    def test_avg_pool1d(self, input_shape, kernel_size, stride, pad, include_pad, backend):
        if False:
            print('Hello World!')
        if pad > kernel_size / 2:
            raise ValueError('pad must be less than half the kernel size')
        model = nn.AvgPool1d(kernel_size, stride, pad, False, include_pad)
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('input_shape, kernel_size, stride, pad, include_pad, backend', itertools.product([(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)], [1, 2, 3], [1, 2], [0, 1], [True, False], backends))
    def test_avg_pool2d(self, input_shape, kernel_size, stride, pad, include_pad, backend):
        if False:
            return 10
        if pad > kernel_size / 2:
            return
        model = nn.AvgPool2d(kernel_size, stride, pad, False, include_pad)
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('input_shape, kernel_size, stride, pad, include_pad, backend', itertools.product([(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)], [3], [1, 2], [0, 1], [True, False], backends))
    def test_avg_pool2d_ceil_mode(self, input_shape, kernel_size, stride, pad, include_pad, backend):
        if False:
            print('Hello World!')
        if pad > kernel_size / 2:
            return
        model = nn.AvgPool2d(kernel_size, stride, pad, True, include_pad)
        run_compare_torch(input_shape, model, backend=backend)

class TestMaxPool:

    @pytest.mark.xfail(reason='PyTorch convert function for op max_pool1d not implemented, we will also likely run into rdar://problem/61064173')
    @pytest.mark.parametrize('input_shape, kernel_size, stride, pad, backend', itertools.product([(1, 3, 15), (1, 1, 7), (1, 3, 10)], [1, 2, 3], [1, 2], [0, 1], backends))
    def test_max_pool1d(self, input_shape, kernel_size, stride, pad, backend):
        if False:
            return 10
        if pad > kernel_size / 2:
            raise ValueError('pad must be less than half the kernel size')
        model = nn.MaxPool1d(kernel_size, stride, pad, ceil_mode=False)
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('input_shape, kernel_size, stride, pad, backend', itertools.product([(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)], [1, 2, 3], [1, 2], [0, 1], backends))
    def test_max_pool2d(self, input_shape, kernel_size, stride, pad, backend):
        if False:
            print('Hello World!')
        if pad > kernel_size / 2:
            return
        model = nn.MaxPool2d(kernel_size, stride, pad, ceil_mode=False)
        run_compare_torch(input_shape, model, backend=backend)

    @pytest.mark.parametrize('input_shape, kernel_size, stride, pad, backend', itertools.product([(1, 3, 15, 15), (1, 1, 7, 7), (1, 3, 10, 10)], [3], [1, 2], [0, 1], backends))
    def test_max_pool2d_ceil_mode(self, input_shape, kernel_size, stride, pad, backend):
        if False:
            return 10
        if pad > kernel_size / 2:
            return
        model = nn.MaxPool2d(kernel_size, stride, pad, ceil_mode=True)
        run_compare_torch(input_shape, model, backend=backend)

class TestLSTM:

    def _pytorch_hidden_to_coreml(self, x):
        if False:
            for i in range(10):
                print('nop')
        (f, b) = torch.split(x, [1] * x.shape[0], dim=0)
        x = torch.cat((f, b), dim=2)
        return x

    @pytest.mark.parametrize('input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend', itertools.product([7], [5], [1], [True, False], [False], [0.3], [True, False], backends))
    def test_lstm(self, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend):
        if False:
            i = 10
            return i + 15
        model = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        SEQUENCE_LENGTH = 3
        BATCH_SIZE = 2
        num_directions = int(bidirectional) + 1
        if batch_first:
            _input = torch.rand(BATCH_SIZE, SEQUENCE_LENGTH, input_size)
        else:
            _input = torch.randn(SEQUENCE_LENGTH, BATCH_SIZE, input_size)
        h0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)
        c0 = torch.randn(num_layers * num_directions, BATCH_SIZE, hidden_size)
        inputs = (_input, (h0, c0))
        expected_results = model(*inputs)
        if bidirectional:
            ex_hn = self._pytorch_hidden_to_coreml(expected_results[1][0])
            ex_cn = self._pytorch_hidden_to_coreml(expected_results[1][1])
            expected_results = (expected_results[0], (ex_hn, ex_cn))
        run_compare_torch(inputs, model, expected_results, input_as_shape=False, backend=backend)

    @pytest.mark.parametrize('input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend', [(7, 3, 2, True, True, 0.3, True, list(backends)[-1]), (7, 3, 2, False, False, 0.3, False, list(backends)[0])])
    def test_lstm_xexception(self, input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            self.test_lstm(input_size, hidden_size, num_layers, bias, batch_first, dropout, bidirectional, backend=backend)

class TestConcat:

    @pytest.mark.parametrize('backend', backends)
    def test_cat(self, backend):
        if False:
            i = 10
            return i + 15

        class TestNet(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super(TestNet, self).__init__()

            def forward(self, x):
                if False:
                    return 10
                x = torch.cat((x,), axis=1)
                return x
        model = TestNet()
        run_compare_torch((1, 3, 16, 16), model, backend=backend)

class TestReduction:

    @pytest.mark.parametrize('input_shape, dim, keepdim, backend', itertools.product([(2, 2), (1, 1)], [0, 1], [True, False], backends))
    def test_max(self, input_shape, dim, keepdim, backend):
        if False:
            return 10

        class TestMax(nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super(TestMax, self).__init__()

            def forward(self, x):
                if False:
                    return 10
                return torch.max(x, dim=dim, keepdim=keepdim)
        input_data = torch.rand(input_shape)
        model = TestMax()
        expected_results = model(input_data)[::-1]
        run_compare_torch(input_data, model, expected_results=expected_results, input_as_shape=False, backend=backend)

class TestLayerNorm:

    @pytest.mark.parametrize('input_shape, eps, backend', itertools.product([(1, 3, 15, 15), (1, 1, 1, 1)], [1e-05, 1e-09], backends))
    def test_layer_norm(self, input_shape, eps, backend):
        if False:
            i = 10
            return i + 15
        model = nn.LayerNorm(input_shape, eps=eps)
        run_compare_torch(input_shape, model, backend=backend)

class TestPixelShuffle:

    @pytest.mark.parametrize('batch_size, CHW, r, backend', itertools.product([1, 3], [(1, 4, 4), (3, 2, 3)], [2, 4], backends))
    def test_pixel_shuffle(self, batch_size, CHW, r, backend):
        if False:
            while True:
                i = 10
        (C, H, W) = CHW
        input_shape = (batch_size, C * r * r, H, W)
        model = nn.PixelShuffle(upscale_factor=r)
        run_compare_torch(input_shape, model, backend=backend)

class TestElementWiseUnary:

    @pytest.mark.parametrize('use_cpu_only, backend, rank, mode', itertools.product([True, False], backends, [rank for rank in range(1, 6)], ['sinh']))
    def test_unary(self, use_cpu_only, backend, rank, mode):
        if False:
            for i in range(10):
                print('nop')
        input_shape = np.random.randint(low=2, high=6, size=rank)
        input_shape = tuple(input_shape)
        if mode == 'sinh':
            operation = torch.sinh
        model = ModuleWrapper(function=operation)
        run_compare_torch(input_shape, model, backend=backend)