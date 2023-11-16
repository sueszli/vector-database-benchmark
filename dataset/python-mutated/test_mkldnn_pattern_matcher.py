import contextlib
import itertools
import torch
import torch.ao.quantization.quantizer.x86_inductor_quantizer as xiq
from torch._dynamo import config as dynamo_config
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._export import capture_pre_autograd_graph
from torch._inductor import config
from torch._inductor.utils import run_and_get_code
from torch.ao.quantization.quantize_pt2e import convert_pt2e, prepare_pt2e, prepare_qat_pt2e
from torch.ao.quantization.quantizer.x86_inductor_quantizer import X86InductorQuantizer
from torch.nn import functional as F
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport, skipIfNoONEDNN, skipIfNoONEDNNBF16
from torch.testing._internal.common_utils import IS_LINUX, skipIfRocm
from torch.testing._internal.inductor_utils import _check_has_dynamic_shape, HAS_CPU
unary_list = {torch.nn.ReLU(): 2, torch.nn.Sigmoid(): 2, torch.nn.Tanh(): 2, torch.nn.Hardswish(): 6, torch.nn.LeakyReLU(0.1, inplace=False): 4, torch.nn.Hardtanh(min_val=-0.5, max_val=4, inplace=False): 3, torch.nn.Hardtanh(min_val=-0.5, max_val=float('inf'), inplace=False): 3, torch.nn.GELU(approximate='none'): 6, torch.nn.GELU(approximate='tanh'): 10, torch.nn.ReLU6(): 3, torch.nn.SiLU(): 3, torch.nn.Hardsigmoid(): 5}
non_decomposed_unary_list = [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]
binary_list = {lambda x, y: torch.add(x, y): (1, 2, False), lambda x, y: torch.add(y, x): (1, 2, False), lambda x, y: x.add(y): (1, 2, False), lambda x, y: x.add_(y): (1, 2, True), lambda x, y: torch.sub(x, y): (1, 2, False), lambda x, y: x.sub(y): (1, 2, False), lambda x, y: x.sub_(y): (1, 2, True)}
quantization_add_fn_list = [lambda x, y: torch.add(x, y), lambda x, y: x.add(y)]
quantization_inplace_add_fn_list = [lambda x, y: x.add_(y)]

@config.patch({'freezing': True})
class TestPatternMatcherBase(TestCase):

    def _check_unary_is_decomposed(self, unary_fn):
        if False:
            print('Hello World!')
        return not any((isinstance(unary_fn, fn) for fn in [torch.nn.ReLU, torch.nn.Sigmoid, torch.nn.Tanh]))

    def _clone_inputs(self, inputs):
        if False:
            for i in range(10):
                print('nop')

        def clone(x):
            if False:
                for i in range(10):
                    print('nop')
            if not isinstance(x, torch.Tensor):
                return x
            return x.clone()
        return tuple((clone(x) for x in inputs))

    def _generate_qdq_quantized_model(self, mod, inputs, is_qat=False):
        if False:
            i = 10
            return i + 15
        maybe_no_grad = contextlib.nullcontext() if is_qat else torch.no_grad()
        with maybe_no_grad:
            export_model = capture_pre_autograd_graph(mod, inputs)
            quantizer = X86InductorQuantizer()
            quantizer.set_global(xiq.get_default_x86_inductor_quantization_config(is_qat=is_qat))
            prepare_model = prepare_qat_pt2e(export_model, quantizer) if is_qat else prepare_pt2e(export_model, quantizer)
            prepare_model(*inputs)
            convert_model = convert_pt2e(prepare_model, fold_quantize=True)
            torch.ao.quantization.move_exported_model_to_eval(convert_model)
            return convert_model

    def _test_common(self, mod, inputs, matcher_count=None, matcher_nodes=None, atol=1e-05, rtol=1.3e-06, check_autocast=False, check_quantization=False, is_qat=False, matcher_check_fn=None):
        if False:
            return 10
        counters.clear()
        torch._dynamo.reset()
        maybe_autocast = contextlib.nullcontext()
        assert matcher_check_fn is not None or (matcher_count is not None and matcher_nodes is not None)
        if check_autocast and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            maybe_autocast = torch.cpu.amp.autocast()
            (atol, rtol) = (0.01, 0.01)
        if check_quantization:
            convert_model = self._generate_qdq_quantized_model(mod, inputs, is_qat)
            with torch.no_grad(), maybe_autocast:
                _ = torch.compile(convert_model)(*inputs)
                if matcher_count is not None:
                    self.assertEqual(counters['inductor']['pattern_matcher_count'], matcher_count)
                if matcher_nodes is not None:
                    self.assertEqual(counters['inductor']['pattern_matcher_nodes'], matcher_nodes)
                if matcher_check_fn is not None:
                    matcher_check_fn()
        else:
            with torch.no_grad(), maybe_autocast:
                clone_inputs = self._clone_inputs(inputs)
                expected = mod(*inputs)
                actual = torch.compile(mod)(*clone_inputs)
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
                self.assertEqual(counters['inductor']['pattern_matcher_count'], matcher_count)
                self.assertEqual(counters['inductor']['pattern_matcher_nodes'], matcher_nodes)

    def _test_code_common(self, mod, inputs, include_ops, exclude_ops, atol=1e-05, rtol=1.3e-06, check_quantization=False, check_dynamic=None):
        if False:
            while True:
                i = 10
        with torch.no_grad():
            clone_inputs = self._clone_inputs(inputs)
            if check_quantization:
                mod = self._generate_qdq_quantized_model(mod, inputs)
            expected = mod(*inputs)
            (actual, (source_code,)) = run_and_get_code(torch.compile(mod, fullgraph=True, dynamic=check_dynamic), *clone_inputs)
            for op in include_ops:
                self.assertIn(op, source_code)
            for op in exclude_ops:
                self.assertNotIn(op, source_code)
            if check_dynamic is not None:
                _check_has_dynamic_shape(self, source_code)
            if not check_quantization:
                torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)

class TestPatternMatcher(TestPatternMatcherBase):

    def test_conv2d_unary_cpu(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self, unary_fn, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.unary_fn = unary_fn

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                return self.unary_fn(x)
        options = itertools.product(unary_list.keys(), [torch.contiguous_format, torch.channels_last], [True, False] if torch.ops.mkldnn._is_mkldnn_bf16_supported() else [False])
        for (unary_fn, memory_format, check_autocast) in options:
            x_shape = (1, 3, 56, 56)
            mod = M(unary_fn).to(memory_format=memory_format).eval()
            v = torch.randn(x_shape, dtype=torch.float32).add(1).to(memory_format=memory_format)
            match_nodes = unary_list[unary_fn] + 1
            if check_autocast and self._check_unary_is_decomposed(unary_fn):
                match_nodes += 2
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=check_autocast)

    def test_linear_unary(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, unary_fn, in_features, out_features, bias, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = torch.nn.Linear(in_features, out_features, bias, **kwargs)
                self.unary_fn = unary_fn

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.linear(x)
                return self.unary_fn(x)
        options = itertools.product(unary_list, [True, False])
        dtype = torch.bfloat16
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            for (unary_fn, bias) in options:
                mod = M(unary_fn, 10, 30, bias=bias).eval()
                mod = mod.to(dtype)
                v = torch.randn(2, 10).to(dtype)
                matcher_count = 2
                matcher_nodes = unary_list[unary_fn] + 1
                if self._check_unary_is_decomposed(unary_fn):
                    matcher_nodes += 2
                self._test_common(mod, (v,), matcher_count, matcher_nodes, check_autocast=True)

    def test_linear_fp32(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self, bias):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(10, 30, bias)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.linear(x)
        for bias in [True, False]:
            mod = M(bias=bias).eval()
            v = torch.randn(2, 10)
            matcher_count = 1
            matcher_nodes = 1
            self._test_common(mod, (v,), matcher_count, matcher_nodes)

    def test_conv_transpose2d_unary(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, unary_fn, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1)
                self.unary_fn = unary_fn

            def forward(self, x):
                if False:
                    return 10
                x = self.conv_transpose2d(x)
                return self.unary_fn(x)
        options = itertools.product(unary_list, [torch.contiguous_format, torch.channels_last], [True, False] if torch.ops.mkldnn._is_mkldnn_bf16_supported() else [False])
        for (unary_fn, memory_format, check_autocast) in options:
            x_shape = (1, 3, 28, 28)
            mod = M(unary_fn).eval()
            v = torch.randn(x_shape, dtype=torch.float32).to(memory_format=memory_format)
            match_nodes = unary_list[unary_fn] + 1
            if check_autocast and self._check_unary_is_decomposed(unary_fn):
                match_nodes += 2
            self._test_common(mod, (v,), 2, match_nodes, check_autocast=check_autocast)

    def test_conv2d_binary(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self, binary_fn, has_relu, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1)
                self.binary_fn = binary_fn
                self.has_relu = has_relu

            def forward(self, x):
                if False:
                    print('Hello World!')
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                if has_relu:
                    return self.binary_fn(x1, x2).relu()
                else:
                    return self.binary_fn(x1, x2)
        test_memory_format = [torch.contiguous_format, torch.channels_last]
        options = itertools.product(binary_list, [True, False], test_memory_format)
        for (binary_fn, has_relu, memory_format) in options:
            x_shape = (1, 3, 56, 56)
            mod = M(binary_fn, has_relu).eval()
            v = torch.randn(x_shape, dtype=torch.float32, requires_grad=True).add(1).to(memory_format=memory_format)
            match_count = binary_list[binary_fn][0] + 2
            match_nodes = binary_list[binary_fn][1]
            if has_relu:
                match_nodes += 1
            self._test_common(mod, (v,), match_count, match_nodes + 2)

    def test_linear_binary(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self, binary_fn, in_channels, out_channels, bias, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = torch.nn.Linear(in_channels, out_channels, bias=bias, **kwargs)
                self.binary_fn = binary_fn

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                x = self.linear(x)
                x = self.binary_fn(x, y.clone())
                return x
        options = itertools.product(binary_list, [[2, 3, 10], [2, 10]], [True, False])
        dtype = torch.bfloat16
        out_feature = 30
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            for (binary_fn, input_shape, bias) in options:
                torch._dynamo.reset()
                match_count = 2
                match_nodes = 3
                if len(input_shape) == 3:
                    is_inplace = binary_list[binary_fn][2]
                    match_count = match_count + 5 if is_inplace else match_count + 3
                    match_nodes = match_nodes + 7 if is_inplace else match_nodes + 5
                mod = M(binary_fn, input_shape[-1], out_feature, bias).to(dtype).eval()
                v = torch.randn(input_shape).to(dtype)
                other = torch.randn(input_shape[:-1] + [out_feature]).to(dtype)
                mod_c = torch.compile(mod)
                (out, code) = run_and_get_code(mod_c, v, other)
                self.assertEqual(out, mod(v, other), rtol=0.01, atol=0.01)

    def test_multi_linear_share_same_input(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return F.silu(self.w1(x)) * F.relu(self.w2(x))
        mod = M().to(torch.bfloat16).eval()
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            v = torch.randn(2, 4, 16).to(torch.bfloat16)
            match_count = 10
            match_nodes = 19
            self._test_common(mod, (v,), match_count, match_nodes, rtol=0.01, atol=0.01)

    def _qconv2d_cpu_test_helper(self, int8_mixed_bf16=False):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv2(self.conv(x))
        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            if False:
                i = 10
                return i + 15
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 2)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_nodes'], 16 if int8_mixed_bf16 else 12)
        self._test_common(mod, (v,), check_quantization=True, check_autocast=int8_mixed_bf16, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_cpu(self):
        if False:
            print('Hello World!')
        '\n        This testcase will quantize a single Conv2d module.\n        '
        self._qconv2d_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_int8_mixed_bf16(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This testcase will quantize a single Conv2d module with int8_mixed_bf16 quantization.\n        '
        self._qconv2d_cpu_test_helper(int8_mixed_bf16=True)

    def _qconv2d_unary_cpu_test_helper(self, int8_mixed_bf16=False):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = torch.nn.ReLU()
                self.conv2 = torch.nn.Conv2d(128, 128, kernel_size=3, stride=1)
                self.unary_fn2 = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                tmp = self.unary_fn(self.conv(x))
                return self.unary_fn2(self.conv2(tmp))
        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 2)
            self.assertEqual(counters['inductor']['qconv2d_unary_matcher_count'], 2)
        self._test_common(mod, (v,), check_quantization=True, check_autocast=int8_mixed_bf16, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_relu_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This testcase will quantize Conv2d->ReLU pattern.\n        '
        self._qconv2d_unary_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_relu_int8_mixed_bf16(self):
        if False:
            print('Hello World!')
        '\n        This testcase will quantize Conv2d->ReLU pattern with int8_mixed_bf16 quantization.\n        '
        self._qconv2d_unary_cpu_test_helper(int8_mixed_bf16=True)

    def _qconv2d_add_cpu_test_helper(self, use_relu=False, int8_mixed_bf16=False):
        if False:
            i = 10
            return i + 15
        '\n        This testcase will quantize a Conv2d->Add pattern as:\n                 X\n               /   \\\n        Conv1(X)   Conv2(X)\n               \\   /\n                Add\n                 |\n           Optional(relu)\n                 |\n                 Y\n        '

        class M(torch.nn.Module):

            def __init__(self, add_fn, use_relu, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.add_fn = add_fn
                self.relu = torch.nn.ReLU()
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv4 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.add_fn2 = add_fn
                self.relu2 = torch.nn.ReLU()
                self.use_relu = use_relu

            def forward(self, x):
                if False:
                    print('Hello World!')
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                tmp = self.add_fn(x1, x2)
                if self.use_relu:
                    tmp = self.relu(tmp)
                tmp1 = self.conv3(tmp)
                tmp2 = self.conv4(tmp)
                res = self.add_fn2(tmp1, tmp2)
                if self.use_relu:
                    res = self.relu2(res)
                return res
        for add_fn in quantization_add_fn_list + quantization_inplace_add_fn_list:
            mod = M(add_fn, use_relu).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

            def matcher_check_fn():
                if False:
                    while True:
                        i = 10
                self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 4)
                self.assertEqual(counters['inductor']['qconv2d_binary_matcher_count'], 2)
            self._test_common(mod, (v,), check_quantization=True, check_autocast=int8_mixed_bf16, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_cpu(self):
        if False:
            i = 10
            return i + 15
        self._qconv2d_add_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_int8_mixed_bf16(self):
        if False:
            print('Hello World!')
        self._qconv2d_add_cpu_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_relu_cpu(self):
        if False:
            while True:
                i = 10
        self._qconv2d_add_cpu_test_helper(use_relu=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_add_relu_int8_mixed_bf16(self):
        if False:
            for i in range(10):
                print('nop')
        self._qconv2d_add_cpu_test_helper(use_relu=True, int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d(self):
        if False:
            print('Hello World!')
        '\n        This testcase will quantize a single Conv2d module with qat flow.\n        '

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.bn = torch.nn.BatchNorm2d(128)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.bn(self.conv(x))
        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            if False:
                while True:
                    i = 10
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_nodes'], 6)
            self.assertEqual(counters['inductor']['qconv2d_unary_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_unary_matcher_nodes'], 7)
        self._test_common(mod, (v,), check_quantization=True, is_qat=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_relu(self):
        if False:
            i = 10
            return i + 15
        '\n        This testcase will quantize Conv2d->ReLU pattern with qat flow.\n        '

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 128, kernel_size=3, stride=1)
                self.unary_fn = torch.nn.ReLU()
                self.bn = torch.nn.BatchNorm2d(128)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.unary_fn(self.bn(self.conv(x)))
        mod = M()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            if False:
                i = 10
                return i + 15
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_nodes'], 6)
            self.assertEqual(counters['inductor']['qconv2d_unary_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_unary_matcher_nodes'], 8)
        self._test_common(mod, (v,), check_quantization=True, is_qat=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add(self):
        if False:
            return 10
        '\n        This testcase will quantize a Conv2d->Add pattern as:\n                 X\n               /   \\\n        Conv1(X)   Conv2(X)\n               \\   /\n                Add\n                 |\n                 Y\n        '

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return x1 + x2
        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            if False:
                while True:
                    i = 10
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 2)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_nodes'], 12)
            self.assertEqual(counters['inductor']['qconv2d_binary_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_binary_matcher_nodes'], 11)
        self._test_common(mod, (v,), check_quantization=True, is_qat=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qat_qconv2d_add_relu(self):
        if False:
            print('Hello World!')
        '\n        This testcase will quantize a Conv2d->Add->ReLU pattern as:\n                 X\n               /   \\\n        Conv1(X)   Conv2(X)\n               \\   /\n                Add\n                 |\n                ReLU\n                 |\n                 Y\n        '

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn1 = torch.nn.BatchNorm2d(6)
                self.conv2 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.bn2 = torch.nn.BatchNorm2d(6)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x1 = self.bn1(self.conv1(x))
                x2 = self.bn2(self.conv2(x))
                return self.relu(x1 + x2)
        mod = M().train()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=True).add(1)

        def matcher_check_fn():
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 2)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_nodes'], 12)
            self.assertEqual(counters['inductor']['qconv2d_binary_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_binary_matcher_nodes'], 12)
        self._test_common(mod, (v,), check_quantization=True, is_qat=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qconv2d_dequant_promotion_cpu(self):
        if False:
            return 10
        '\n        This testcase tests if dequant node before conv2d is promoted correctly:\n                 X\n                 |\n              Conv1(X)\n               /   \\\n        Conv2(X)   Conv3(X)\n               \\   /\n                Add\n                 |\n                 Y\n        '

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 6, kernel_size=3, stride=1)
                self.conv2 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)
                self.conv3 = torch.nn.Conv2d(6, 6, kernel_size=3, stride=1)

            def forward(self, x):
                if False:
                    return 10
                temp = self.conv1(x)
                temp = self.conv2(temp) + self.conv3(temp)
                return temp
        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)

        def matcher_check_fn():
            if False:
                print('Hello World!')
            self.assertEqual(counters['inductor']['dequant_promotion_matcher_count'], 1)
            self.assertEqual(counters['inductor']['dequant_promotion_matcher_nodes'], 3)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_count'], 3)
            self.assertEqual(counters['inductor']['qconv2d_weight_prepack_matcher_nodes'], 18)
            self.assertEqual(counters['inductor']['qconv2d_binary_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qconv2d_binary_matcher_nodes'], 2)
        self._test_common(mod, (v,), check_quantization=True, matcher_check_fn=matcher_check_fn)

    def _qlinear_cpu_test_helper(self, int8_mixed_bf16=False):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self, use_bias):
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.linear2 = torch.nn.Linear(4, 4, use_bias)

            def forward(self, x):
                if False:
                    return 10
                return self.linear2(self.linear(x))
        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            v = torch.randn((2, 4))

            def matcher_check_fn():
                if False:
                    while True:
                        i = 10
                self.assertEqual(counters['inductor']['qlinear_weight_prepack_matcher_count'], 2)
                self.assertEqual(counters['inductor']['qlinear_weight_prepack_matcher_nodes'], 16 if int8_mixed_bf16 else 12)
            self._test_common(mod, (v,), check_autocast=int8_mixed_bf16, check_quantization=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_cpu(self):
        if False:
            while True:
                i = 10
        '\n        This testcase will quantize a single Linear Moduel.\n        '
        self._qlinear_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_int8_mixed_bf16(self):
        if False:
            while True:
                i = 10
        '\n        This testcase will quantize a single Linear Moduel with int8_mixed_bf16 quantization.\n        '
        self._qlinear_cpu_test_helper(int8_mixed_bf16=True)

    def _qlinear_unary_cpu_test_helper(self, int8_mixed_bf16=False):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, use_bias):
                if False:
                    return 10
                super().__init__()
                self.linear = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn = torch.nn.ReLU()
                self.linear2 = torch.nn.Linear(4, 4, use_bias)
                self.unary_fn2 = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                tmp = self.unary_fn(self.linear(x))
                return self.unary_fn2(self.linear2(tmp))
        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            v = torch.randn((2, 4))

            def matcher_check_fn():
                if False:
                    return 10
                self.assertEqual(counters['inductor']['qlinear_weight_prepack_matcher_count'], 2)
                self.assertEqual(counters['inductor']['qlinear_unary_matcher_count'], 2)
            self._test_common(mod, (v,), check_autocast=int8_mixed_bf16, check_quantization=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_relu_cpu(self):
        if False:
            while True:
                i = 10
        '\n        This testcase will quantize a Linear->ReLU pattern.\n        '
        self._qlinear_unary_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_relu_int8_mixed_bf16(self):
        if False:
            i = 10
            return i + 15
        '\n        This testcase will quantize a Linear->ReLU pattern with int8_mixed_bf16 quantization.\n        '
        self._qlinear_unary_cpu_test_helper(int8_mixed_bf16=True)

    def _qlinear_dequant_promotion_cpu_test_helper(self, int8_mixed_bf16=False):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    print('Hello World!')
                super().__init__()
                self.linear1 = torch.nn.Linear(4, 4)
                self.linear2 = torch.nn.Linear(4, 4)
                self.linear3 = torch.nn.Linear(4, 4)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                temp = self.linear1(x)
                temp = self.linear2(temp) + self.linear3(temp)
                return temp
        mod = M().eval()
        v = torch.rand((2, 4))

        def matcher_check_fn():
            if False:
                while True:
                    i = 10
            self.assertEqual(counters['inductor']['dequant_promotion_matcher_count'], 1)
            self.assertEqual(counters['inductor']['qlinear_weight_prepack_matcher_count'], 3)
            self.assertEqual(counters['inductor']['qlinear_unary_matcher_count'], 1)
        self._test_common(mod, (v,), check_autocast=int8_mixed_bf16, check_quantization=True, matcher_check_fn=matcher_check_fn)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion_cpu(self):
        if False:
            return 10
        '\n        This testcase test if dequant node before linear is promoted correctly:\n                  X\n                  |\n               Linear1(X)\n                /   \\\n        Linear2(X)   Linear3(X)\n                \\   /\n                 Add\n                  |\n                  Y\n        '
        self._qlinear_dequant_promotion_cpu_test_helper()

    @skipIfNoDynamoSupport
    @skipIfNoONEDNNBF16
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_dequant_promotion_int8_mixed_bf16(self):
        if False:
            while True:
                i = 10
        '\n        Test with int8_mixed_bf16 quantization.\n        This testcase test if dequant node before linear is promoted correctly:\n                  X\n                  |\n               Linear1(X)\n                /   \\\n        Linear2(X)   Linear3(X)\n                \\   /\n                 Add\n                  |\n                  Y\n        '
        self._qlinear_dequant_promotion_cpu_test_helper(int8_mixed_bf16=True)

    @skipIfNoDynamoSupport
    @skipIfNoONEDNN
    @skipIfRocm
    def test_qlinear_mul_cpu(self):
        if False:
            return 10
        '\n        This testcase will quantize a Linear->Mul pattern.\n        '

        class M(torch.nn.Module):

            def __init__(self, use_bias):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = torch.nn.Linear(4, 5, use_bias)

            def forward(self, x1, x2):
                if False:
                    return 10
                return torch.mul(self.linear(x1), x2)
        bias_list = [True, False]
        for bias in bias_list:
            mod = M(bias).eval()
            x1 = torch.randn((2, 4))
            x2 = torch.randn((2, 5))
            self._test_common(mod, (x1, x2), 2, 8, check_quantization=True)

    @skipIfNoDynamoSupport
    @skipIfRocm
    def test_qmaxpool2d(self):
        if False:
            while True:
                i = 10
        '\n        This testcase will quantize Conv2d->ReLU->MaxPool2d pattern.\n        '

        class M(torch.nn.Module):

            def __init__(self, kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 7, bias=True, stride=2, padding=3, dilation=1)
                self.relu = torch.nn.ReLU()
                self.maxpool = torch.nn.MaxPool2d(3, **kwargs)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.maxpool(self.relu(self.conv(x)))
        kwargs_list = [{'stride': 2}, {'stride': 2, 'padding': 1}, {'stride': 2, 'padding': 1, 'dilation': 1}, {'stride': 2, 'padding': 1, 'dilation': 1, 'ceil_mode': False}]
        for kwargs in kwargs_list:
            mod = M(kwargs).eval()
            v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)
            self._test_common(mod, (v,), 6, 31, check_quantization=True)

    @skipIfNoDynamoSupport
    @skipIfRocm
    def test_qcat(self):
        if False:
            print('Hello World!')
        '\n        This testcase will quantize cat based pattern:\n                X\n             /     \\\n        Conv1(X)  Pow(x)\n            \\        \\\n             \\     Conv2(X)\n              \\    /\n               Cat\n                |\n                Y\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, 7, bias=True, stride=2, padding=3, dilation=1)
                self.conv2 = torch.nn.Conv2d(3, 64, 7, bias=True, stride=2, padding=3, dilation=1)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                temp1 = self.conv(x)
                temp2 = self.conv2(torch.pow(x, 2))
                return torch.cat((temp1, temp2), 1)
        mod = M().eval()
        v = torch.randn((1, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)
        self._test_common(mod, (v,), 10, 49, check_quantization=True)

    def test_hardtanh_pattern_fallback(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv_transpose = torch.nn.ConvTranspose2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x, min_value, max_value):
                if False:
                    return 10
                conv_transpose_output = self.conv_transpose(x)
                clamp_min_output = torch.clamp_min(conv_transpose_output, min_value)
                clamp_max_output = torch.clamp_max(clamp_min_output, max_value)
                return clamp_max_output
        min_values = [3, torch.randn(1, 32, 28, 28)]
        max_values = [0, torch.randn(1, 32, 28, 28)]
        v = torch.randn(1, 3, 28, 28)
        for (min_value, max_value) in zip(min_values, max_values):
            mod = Model().eval()
            self._test_common(mod, (v, min_value, max_value), 2, 4)

    def test_leaky_relu_pattern_fallback(self):
        if False:
            while True:
                i = 10

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x, negative_slope):
                if False:
                    while True:
                        i = 10
                conv_out = self.conv(x)
                return torch.where(conv_out > 0, conv_out, conv_out * negative_slope)
        negative_slopes = [0.1, torch.randn(1, 32, 28, 28)]
        with torch.no_grad():
            v = torch.randn(1, 3, 28, 28)
            for negative_slope in negative_slopes:
                mod = Model().eval()
                self._test_common(mod, (v, negative_slope), 2, 5)

    def test_conv2d_add_scalar(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                if False:
                    return 10
                out_conv = self.conv(x)
                out = torch.add(out_conv, 1.0)
                return out
        with torch.no_grad():
            mod = Model().eval()
            v = torch.randn(1, 3, 28, 28)
            self._test_common(mod, (v,), 1, 1)

    def test_conv2d_binary_inplace_fusion_pass_cpu(self, include_ops=None, exclude_ops=None):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x, other):
                if False:
                    while True:
                        i = 10
                conv_out = self.conv(x)
                return torch.add(conv_out, other.relu())
        inputs = [torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last), torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last)]
        mod = Model().to(memory_format=torch.channels_last).eval()
        if include_ops is None:
            include_ops = ['mkldnn._convolution_pointwise_.binary']
        if exclude_ops is None:
            exclude_ops = ['mkldnn._convolution_pointwise.binary']
        self._test_code_common(mod, inputs, include_ops, exclude_ops)

    def test_conv2d_binary_inplace_fusion_failed_cpu(self, include_ops=None, exclude_ops=None):
        if False:
            return 10

        class Model_v1(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x, other):
                if False:
                    i = 10
                    return i + 15
                conv_out = self.conv(x)
                return torch.add(conv_out, other)

        class Model_v2(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x, other):
                if False:
                    return 10
                conv_out = self.conv(x)
                return (torch.add(conv_out, other[1:2, :, :, :]), other)
        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last), torch.randn(2, 32, 28, 28).to(memory_format=torch.channels_last)]
        mod_v1 = Model_v1().to(memory_format=torch.channels_last).eval()
        mod_v2 = Model_v2().to(memory_format=torch.channels_last).eval()
        if include_ops is None:
            include_ops = ['mkldnn._convolution_pointwise.binary']
        if exclude_ops is None:
            exclude_ops = ['mkldnn._convolution_pointwise_.binary']
        for (other, mod) in zip(others, [mod_v1, mod_v2]):
            self._test_code_common(mod, (input, other), include_ops, exclude_ops)

    def test_conv2d_binary_fusion_failed(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1)

            def forward(self, x, other, alpha):
                if False:
                    while True:
                        i = 10
                conv_out = self.conv(x)
                return torch.add(conv_out, other, alpha=alpha)

        class Model2(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                if False:
                    print('Hello World!')
                out = self.conv(x)
                out = torch.add(out, out)
                return out

        class Model3(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                temp = self.conv(x)
                other = torch.ones(temp.shape, dtype=torch.double)
                out = torch.add(temp, other)
                return out
        input = torch.randn(1, 3, 28, 28).to(memory_format=torch.channels_last)
        others = [torch.randn(1, 32, 28, 28).to(memory_format=torch.channels_last), torch.randn(32, 28, 28)]
        include_ops = ['mkldnn._convolution_pointwise']
        exclude_ops = ['mkldnn._convolution_pointwise.binary', 'mkldnn._convolution_pointwise_.binary']
        for (other, alpha) in zip(others, [0.1, 1.0]):
            mod = Model().to(memory_format=torch.channels_last).eval()
            self._test_code_common(mod, (input, other, alpha), include_ops, exclude_ops)
        mod = Model2().to(memory_format=torch.channels_last).eval()
        self._test_code_common(mod, (input,), include_ops, exclude_ops)
        mod = Model3().to(memory_format=torch.channels_last).eval()
        self._test_code_common(mod, (input,), include_ops, exclude_ops)

    def test_reproduce_99842_issue(self):
        if False:
            print('Hello World!')

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)

            def forward(self, input_tensor):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(input_tensor)
                x = F.relu(x + torch.ones(x.size()))
                return x
        input = torch.randn(1, 3, 14, 14)
        mod = Model().eval()
        include_ops = ['mkldnn._convolution_pointwise_.binary']
        self._test_code_common(mod, (input,), include_ops, [])

@dynamo_config.patch({'dynamic_shapes': True, 'assume_static_by_default': False})
class TestDynamicPatternMatcher(TestPatternMatcherBase):
    test_conv2d_unary_dynamic_shapes = TestPatternMatcher.test_conv2d_unary_cpu
    test_conv2d_binary_dynamic_shapes = TestPatternMatcher.test_conv2d_binary
    test_linear_unary_dynamic_shapes = TestPatternMatcher.test_linear_unary

    def test_conv_transpose2d_dynamic_shapes(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv_transpose2d = torch.nn.ConvTranspose2d(3, 16, 3, stride=2, padding=1)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.conv_transpose2d(x)
        x_shape = (1, 3, 28, 28)
        mod = M().eval()
        v = torch.randn(x_shape, dtype=torch.float32)
        self._test_common(mod, (v,), 0, 0)

    def test_multi_linear_share_same_input_dynamic(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.w1 = torch.nn.Linear(16, 16, bias=False)
                self.w2 = torch.nn.Linear(16, 16, bias=False)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return F.silu(self.w1(x)) * F.relu(self.w2(x))
        mod = M().to(torch.bfloat16).eval()
        if torch.ops.mkldnn._is_mkldnn_bf16_supported():
            v = torch.randn(2, 4, 16).to(torch.bfloat16)
            match_count = 8
            match_nodes = 12
            self._test_common(mod, (v,), match_count, match_nodes, rtol=0.01, atol=0.01)

    def test_qconv2d_maxpool2d_linear_dynamic_cpu(self, include_ops=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        This testcase will quantize a single Conv2d->Maxpool2d->Linear module\n        with dynamic batch size input.\n        '

        class M(torch.nn.Module):

            def __init__(self, **kwargs):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 16, (2, 2), stride=(1, 1), padding=(1, 1))
                self.relu = torch.nn.ReLU()
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                temp = self.relu(self.conv(x))
                temp = self.maxpool2d(temp)
                temp = self.avgpool(temp)
                temp = torch.flatten(temp, 1)
                return self.linear(temp)
        mod = M().eval()
        v = torch.randn((2, 3, 8, 8), dtype=torch.float32, requires_grad=False).add(1)
        if include_ops is None:
            include_ops = ['torch.ops.onednn.qconv2d_pointwise', 'torch.ops.quantized.max_pool2d', 'torch.ops.onednn.qlinear_pointwise']
        exclude_ops = []
        self._test_code_common(mod, (v,), include_ops, exclude_ops, check_quantization=True, check_dynamic=True)
if __name__ == '__main__':
    if IS_LINUX and HAS_CPU and torch.backends.mkldnn.is_available():
        run_tests()