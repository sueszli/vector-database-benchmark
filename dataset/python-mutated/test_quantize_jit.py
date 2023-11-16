import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import torch.jit.quantized
from torch.ao.quantization import QConfig, default_dynamic_qconfig, float16_dynamic_qconfig, default_observer, per_channel_dynamic_qconfig, default_per_channel_weight_observer, default_qconfig, get_default_qconfig, quantize, quantize_dynamic, default_weight_observer, default_histogram_observer, fuse_modules, quantize_jit, quantize_dynamic_jit, PlaceholderObserver
from torch.ao.quantization.quantize_jit import convert_jit, convert_dynamic_jit, fuse_conv_bn_jit, prepare_jit, prepare_dynamic_jit, script_qconfig
from torch.testing._internal.common_quantized import override_qengines, qengine_is_fbgemm, qengine_is_qnnpack
from torch.testing._internal.common_quantization import QuantizationTestCase, skipIfNoFBGEMM, get_script_module, SingleLayerLinearModel, SkipQuantModel, NestedModel, ConvModel, ConvTransposeModel, default_per_channel_qconfig, test_only_eval_fn, ConvBnModel
from torch.testing._internal.common_quantization import AnnotatedSingleLayerLinearModel, AnnotatedSkipQuantModel, AnnotatedNestedModel, AnnotatedConvModel, AnnotatedConvTransposeModel, AnnotatedConvBnModel
from torch.testing import FileCheck
from torch.testing._internal.jit_utils import attrs_with_prefix
from torch.testing._internal.jit_utils import get_forward
from torch.testing._internal.jit_utils import get_forward_graph
from torch.testing._internal.common_utils import set_default_dtype
from torch.jit._recursive import wrap_cpp_module
from typing import List, Tuple
import io
import itertools
import unittest

class TestQuantizeJitPasses(QuantizationTestCase):
    """Test graph mode quantization passes used by quantize_jit"""

    def test_skip_dequant_constant_prop(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv(x)
        m = torch.jit.script(M())
        observer = default_per_channel_weight_observer.with_args(ch_axis=1)
        qconfig_dict = {'': QConfig(activation=default_observer, weight=observer)}
        m = prepare_jit(m, qconfig_dict)
        data = torch.randn(1, 3, 10, 10, dtype=torch.float)
        m(data)
        m = convert_jit(m, debug=True)
        freezed = torch.jit.freeze(m)
        freezed(data)
        FileCheck().check_count('aten::quantize_per_tensor', 2, exactly=True).run(freezed.graph)
        FileCheck().check_count('aten::quantize_per_channel', 0, exactly=True).run(freezed.graph)
        FileCheck().check_count('aten::dequantize', 3, exactly=True).run(freezed.graph)
        FileCheck().check('aten::quantize_per_tensor').check_next('aten::dequantize').check_not('aten::quantize_per_channel').check('aten::dequantize').check_next('aten::conv2d').check_next('aten::quantize_per_tensor').check_next('aten::dequantize').run(freezed.graph)

    def test_foldbn_trivial(self):
        if False:
            for i in range(10):
                print('nop')
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class TestModule(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    return 10
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)
                self.bn.eps = 0.0023

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                x = self.bn(x)
                return x
        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        for (tracing, dim) in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            FileCheck().check_count('prim::CallMethod[name="forward"]', 2, exactly=True).run(str(get_forward(scripted_or_traced._c).graph))
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)
            FileCheck().check_count('prim::CallMethod[name="forward"]', 1, exactly=True).run(str(get_forward_graph(scripted_or_traced._c)))
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_trivial_nobias(self):
        if False:
            i = 10
            return i + 15
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class TestModule(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1, bias=False)
                self.bn = bn_module[dim](num_features=20)
                self.bn.eps = 0.0027
                self.bn.bias = torch.nn.Parameter(torch.rand([20]))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                x = self.bn(x)
                return x
        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 6, 6), 3: torch.rand(1, 1, 6, 6, 6)}
        for (tracing, dim) in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            FileCheck().check_count('prim::CallMethod[name="forward"]', 2, exactly=True).run(str(get_forward_graph(scripted_or_traced._c)))
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)
            FileCheck().check_count('prim::CallMethod[name="forward"]', 1, exactly=True).run(str(get_forward_graph(scripted_or_traced._c)))
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_in_submodule(self):
        if False:
            i = 10
            return i + 15
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class SubModule(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = conv_module[dim](1, 20, 5, 1)
                self.bn = bn_module[dim](num_features=20)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                x = self.bn(x)
                return x

        class TestModule(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.sub = SubModule(dim)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.sub(x)
                return x
        options = itertools.product([True, False], [2, 3])
        data = {2: torch.rand(1, 1, 10, 10), 3: torch.rand(1, 1, 10, 10, 10)}
        for (tracing, dim) in options:
            eager = TestModule(dim).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            FileCheck().check_count('prim::CallMethod[name="forward"]', 2, exactly=True).run(str(get_forward_graph(scripted_or_traced.sub._c)))
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)
            FileCheck().check_count('prim::CallMethod[name="forward"]', 1, exactly=True).run(str(get_forward_graph(scripted_or_traced.sub._c)))
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_shared_classtype(self):
        if False:
            for i in range(10):
                print('nop')
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class TestModule(torch.nn.Module):

            def __init__(self, dim, bias=False):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = conv_module[dim](5, 5, 3, bias=bias)
                self.bn1 = bn_module[dim](num_features=5)
                self.bn1.running_mean.fill_(-0.2)
                self.bn1.bias = torch.nn.Parameter(torch.rand([5]))
                self.bn1.eps = 0.0023
                self.conv2 = conv_module[dim](5, 5, 3, bias=bias)
                self.bn2 = bn_module[dim](num_features=5)
                self.bn2.eps = 0.0029
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.conv2(x)
                x = self.bn2(x)
                x = self.relu(x)
                return x
        options = itertools.product([True, False], [2, 2], [True, False])
        data = {2: torch.rand(1, 5, 6, 6), 3: torch.rand(1, 5, 6, 6, 6)}
        for (tracing, dim, bias) in options:
            eager = TestModule(dim, bias).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x)
            folded = fuse_conv_bn_jit(scripted_or_traced)
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_foldbn_no_fusion(self):
        if False:
            i = 10
            return i + 15
        "Test that we don't fuse the cases when module type does not match"

        class CustomConv(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class CustomBn(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = CustomConv()
                self.bn = CustomBn()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.bn(self.conv(x))
        m = torch.jit.script(M())
        m = fuse_conv_bn_jit(m)
        FileCheck().check_count('prim::CallMethod', 2, exactly=True).run(m.graph)

    @set_default_dtype(torch.double)
    def test_foldbn_complex_cases(self):
        if False:
            print('Hello World!')
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}
        conv_module = {2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class SubModule(torch.nn.Module):

            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                if False:
                    print('Hello World!')
                super().__init__()
                layers = []
                for i in range(num_blocks):
                    layers.append(conv_module[dim](20, 20, 5, 1, bias=enable_bias))
                    bn_obj = bn_module[dim](num_features=20, affine=enable_affine)
                    if enable_affine:
                        bn_obj.weight = torch.nn.Parameter(torch.rand_like(bn_obj.weight))
                        bn_obj.bias = torch.nn.Parameter(torch.rand_like(bn_obj.bias))
                    bn_obj.running_mean = torch.rand_like(bn_obj.running_mean)
                    bn_obj.running_var = torch.rand_like(bn_obj.running_var)
                    layers.append(bn_obj)
                self.layers = nn.Sequential(*layers)

            def forward(self, x):
                if False:
                    return 10
                return self.layers(x)

        class TestModule(torch.nn.Module):

            def __init__(self, dim, num_blocks, enable_bias, enable_affine):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.sub = SubModule(dim, num_blocks, enable_bias, enable_affine)

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.sub(x)
                return x
        options = itertools.product([True, False], [2, 3], [True, False], [True, False], [1, 2])
        data = {2: torch.rand(1, 20, 10, 10), 3: torch.rand(1, 20, 10, 10, 10)}
        for (tracing, dim, enable_bias, enable_bn_affine, num_layers) in options:
            eager = TestModule(dim, num_layers, enable_bias, enable_bn_affine).eval()
            x = data[dim]
            scripted_or_traced = get_script_module(eager, tracing, x).eval()
            FileCheck().check_count('prim::CallMethod[name="forward"]', num_layers * 2, exactly=True).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))
            scripted_or_traced = fuse_conv_bn_jit(scripted_or_traced)
            FileCheck().check_count('prim::CallMethod[name="forward"]', num_layers, exactly=True).run(str(get_forward_graph(scripted_or_traced.sub.layers._c)))
            self.assertEqual(eager(x), scripted_or_traced(x))

    def test_fuse_linear(self):
        if False:
            print('Hello World!')

        class FunctionalLinear(torch.nn.Module):

            def __init__(self, weight, bias):
                if False:
                    print('Hello World!')
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                res = torch.matmul(x, self.weight.t())
                if self.bias is not None:
                    res.add_(self.bias)
                return res
        x1 = torch.rand(3)
        w1 = torch.rand(5, 3)
        b1 = torch.rand(5)
        x2 = torch.rand(5, 5)
        w2 = torch.rand(5, 5)
        b2 = torch.rand(5)
        x3 = torch.rand(5, 5, 5)
        w3 = torch.rand(5, 5)
        b3 = torch.rand(5)
        for (has_bias, (x, weight, b)) in itertools.product([True, False], [(x1, w1, b1), (x2, w2, b2), (x3, w3, b3)]):
            bias = b if has_bias else None
            model = torch.jit.trace(FunctionalLinear(weight, bias), [x])
            for node in model.graph.nodes():
                if node.kind() == 'aten::matmul':
                    source_range_1 = node.sourceRange()
            torch._C._jit_pass_fuse_linear(model.graph)
            for node in model.graph.nodes():
                if node.kind() == 'aten::linear':
                    source_range_2 = node.sourceRange()
            FileCheck().check('aten::linear').run(model.graph)
            check_not = ['aten::matmul', 'aten::addmm', 'aten::add_', 'aten::t(']
            for cn in check_not:
                FileCheck().check_not(cn).run(model.graph)
            self.assertTrue(source_range_1 == source_range_2)
            model(x)

        class Matmul(torch.nn.Module):

            def __init__(self, weight):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weight = weight

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.matmul(x, self.weight)
        x = torch.rand(5, 6, 5)
        w = torch.rand(5, 5, 100)
        model = torch.jit.trace(Matmul(w), [x])
        torch._C._jit_pass_fuse_linear(model.graph)
        FileCheck().check('aten::matmul').run(model.graph)
        FileCheck().check_not('aten::linear').run(model.graph)
        model(x)

    def test_insert_observers(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.conv(x)
        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1

    def test_insert_observers_interface(self):
        if False:
            i = 10
            return i + 15

        @torch.jit.interface
        class SubInterface(torch.nn.Module):

            def addOne(self, inp) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                pass

        class Sub(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def addOne(self, inp):
                if False:
                    return 10
                return self.fc(inp) + 1

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.addOne(x)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                if False:
                    return 10
                return self.sub(self.conv(x))
        m = torch.jit.script(M())
        qconfig_dict = {'sub.conv': default_qconfig}
        m = prepare_jit(m, qconfig_dict)

    def test_insert_observers_interface_unshare_type(self):
        if False:
            print('Hello World!')

        @torch.jit.interface
        class OperatorIf(nn.Module):

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                pass

        class Operator(nn.Module):

            def __init__(self, a):
                if False:
                    return 10
                super().__init__()
                self.a = a

            def forward(self, inp: torch.Tensor) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return self.a * (inp + self.a)

        class Inner(nn.Module):
            op: OperatorIf

            def __init__(self, op):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.op = op

            def forward(self, inp):
                if False:
                    i = 10
                    return i + 15
                return self.op(inp)

        class Outer(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.inner_a = Inner(Operator(1))
                self.inner_b = Inner(Operator(3.0))

            def forward(self, inp):
                if False:
                    for i in range(10):
                        print('nop')
                return self.inner_a(inp) + self.inner_b(inp)
        qconfig_dict = {'inner_a': default_qconfig, 'inner_b': default_qconfig}
        eager_model = Outer()
        for tracing in [True, False]:
            x = torch.rand(3)
            script_model = get_script_module(eager_model, tracing, x)
            prepare_jit(script_model, qconfig_dict)

    def test_insert_observers_child_qconfig(self):
        if False:
            i = 10
            return i + 15

        class Sub(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.fc(x)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                if False:
                    return 10
                return self.sub(self.conv(x))
        m = torch.jit.script(M())
        qconfig_dict = {'sub.fc': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines, ' Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support avx2 or newer.')
    def test_insert_observers_skip_values(self):
        if False:
            while True:
                i = 10

        class ConvFunctionalReLU(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return F.relu(self.conv(x))

        class ConvReLUModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.relu(self.conv(x))

        class AddReLUModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.relu = torch.nn.ReLU()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                if False:
                    return 10
                out = self.conv(x)
                out += x
                return self.relu(out)

        class AddFunctionalReLU(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                out = self.conv(x)
                out += x
                return F.relu(out)

        def attrs_with_prefix(module, prefix):
            if False:
                print('Hello World!')
            return [x for (x, _) in module._modules._c.items() if x.startswith(prefix)]
        qconfig_dict = {'': default_qconfig}
        m = torch.jit.script(ConvFunctionalReLU())
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        m = torch.jit.script(ConvReLUModule())
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 1
        assert len(attrs_with_prefix(m.relu, '_observer_')) == 0
        m = torch.jit.script(AddReLUModule())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer')) == 3
        assert len(attrs_with_prefix(m.relu, '_observer')) == 0
        FileCheck().check('aten::add_').check_not('Observer = prim::GetAttr[name="_observer_').check('ReLU = prim::GetAttr').run(str(get_forward_graph(m._c)))
        m = torch.jit.script(AddFunctionalReLU())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer')) == 3
        FileCheck().check('aten::add_').check_not('Observer = prim::GetAttr[name="_observer_').check('CallFunction').check('Observer = prim::GetAttr[name="_observer_').run(str(get_forward_graph(m._c)))

    def test_insert_observers_weight_dtype(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return F.relu(self.conv(x))
        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        activation_dtypes = {obs.getattr('dtype') for (x, obs) in m._modules._c.items() if x.startswith('_observer_')}
        weight_dtypes = {obs.getattr('dtype') for (x, obs) in m.conv._modules._c.items() if x.startswith('_observer_')}
        assert len(activation_dtypes) == 1, 'Expected to have 1 activation dtype'
        assert len(weight_dtypes) == 1, 'Expected to have 1 weight dtype'
        assert list(activation_dtypes)[0] != list(weight_dtypes)[0], 'Expected activation dtype to '
        ' be different from wegiht dtype'

    def test_insert_observers_for_reused_weight(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, x, y, weight):
                if False:
                    for i in range(10):
                        print('nop')
                x = F.conv2d(x, weight)
                y = F.conv2d(y, weight)
                return x + y
        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {'': default_qconfig})
        assert len(attrs_with_prefix(m, '_observer')) == 6

    def test_insert_observers_shared_class_type(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 5, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.conv2(self.conv1(x))
        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        conv1_observers = attrs_with_prefix(m.conv1, '_observer_')
        conv2_observers = attrs_with_prefix(m.conv2, '_observer_')
        assert len(conv1_observers) == 1, 'Expected to have 1 observer submodules'
        assert len(conv2_observers) == 1, 'Expected to have 1 observer submodules'
        assert conv1_observers == conv2_observers, 'Expect conv1 and conv2 to have same observers since the class type is shared'

    def test_insert_observers_for_general_ops(self):
        if False:
            i = 10
            return i + 15
        "Make sure we skip observers for ops that doesn't require\n        observation, e.g. flatten\n        "

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                x = torch.flatten(x)
                return x
        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer_')) == 2
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check('prim::GetAttr[name="conv"]').check('prim::CallMethod').check('Observer = prim::GetAttr[name="_observer_').check('aten::flatten').check_not('Observer = prim::GetAttr[name="_observer_').run(m.graph)

    def test_insert_observers_propagate_observed(self):
        if False:
            print('Hello World!')
        'Make sure we propagate observed property through general ops'

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                x = torch.flatten(x)
                x = self.conv2(x)
                return x
        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer_')) == 3
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check('prim::GetAttr[name="conv1"]').check('prim::CallMethod').check('Observer = prim::GetAttr[name="_observer_').check('aten::flatten').check_not('Observer = prim::GetAttr[name="_observer_').check('prim::GetAttr[name="conv2"]').check('Observer = prim::GetAttr[name="_observer_').run(m.graph)

    def test_insert_observers_propagate_observed_in_submodule(self):
        if False:
            return 10
        'Make sure we propagate observed property through general ops'

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AdaptiveAvgPool2d((1, 1))

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                x = self.avgpool(x)
                x = self.conv2(x)
                return x
        m = torch.jit.script(M())
        qconfig_dict = {'': default_qconfig}
        m = prepare_jit(m, qconfig_dict)
        assert len(attrs_with_prefix(m, '_observer_')) == 3
        FileCheck().check('Observer = prim::GetAttr[name="_observer_').check('prim::GetAttr[name="conv1"]').check('prim::CallMethod').check('Observer = prim::GetAttr[name="_observer_').check('prim::CallMethod').check_not('Observer = prim::GetAttr[name="_observer_').check('prim::GetAttr[name="conv2"]').check('Observer = prim::GetAttr[name="_observer_').run(m.graph)

    def test_insert_observers_propagate_observed_for_function(self):
        if False:
            for i in range(10):
                print('nop')

        def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
            if False:
                for i in range(10):
                    print('nop')
            (batchsize, num_channels, height, width) = x.data.size()
            channels_per_group = num_channels // groups
            x = x.view(batchsize, groups, channels_per_group, height, width)
            x = torch.transpose(x, 1, 2).contiguous()
            x = x.view(batchsize, -1, height, width)
            return x

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                x = channel_shuffle(x, 1)
                x = self.conv2(x)
                return x
        data = [(torch.rand((1, 3, 10, 10), dtype=torch.float), torch.randint(0, 1, (1,), dtype=torch.long)) for _ in range(2)]
        m = torch.jit.script(M()).eval()
        m = prepare_jit(m, {'': default_qconfig})
        assert len(attrs_with_prefix(m, '_observer_')) == 3

    def test_insert_observers_for_if(self):
        if False:
            while True:
                i = 10

        class QuantProp(torch.nn.Module):

            def __init__(self, use_skip):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                if False:
                    return 10
                if self.use_skip:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)
                else:
                    x = self.conv(x)
                    return torch.reshape(x, x.shape)

        class Res(torch.nn.Module):

            def __init__(self, use_skip):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = use_skip

            def forward(self, x):
                if False:
                    print('Hello World!')
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.quant_prop = QuantProp(True)
                self.res = Res(False)

            def forward(self, x):
                if False:
                    return 10
                x = self.quant_prop(x)
                x = self.res(x)
                return x
        data = [torch.rand(1, 3, 10, 10, dtype=torch.float)]
        result = {False: [1, 2, 2], True: [2, 1, 0]}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_jit(m, {'': default_qconfig})
            assert len(attrs_with_prefix(m, '_observer_')) == result[tracing][0]
            assert len(attrs_with_prefix(m.quant_prop, '_observer_')) == result[tracing][1]
            assert len(attrs_with_prefix(m.res, '_observer_')) == result[tracing][2]

    def test_insert_observers_for_nested_if(self):
        if False:
            for i in range(10):
                print('nop')

        class Res(torch.nn.Module):

            def __init__(self, use_skip):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.cond = use_skip
                self.use_skip = use_skip

            def forward(self, x):
                if False:
                    return 10
                if self.use_skip:
                    if self.cond:
                        return self.conv(x)
                    else:
                        return self.conv(x)
                else:
                    return self.conv(x)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.res1 = Res(True)
                self.res2 = Res(False)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.res1(x)
                x = self.res2(x)
                return x
        data = torch.rand((1, 3, 10, 10), dtype=torch.float)
        result = {True: 3, False: 1}
        for tracing in [True, False]:
            if tracing:
                m = torch.jit.trace(M(), data).eval()
            else:
                m = torch.jit.script(M()).eval()
            m = prepare_jit(m, {'': default_qconfig})
            assert len(attrs_with_prefix(m, '_observer_')) == result[tracing]

    def test_insert_observers_for_if_consistent_observation(self):
        if False:
            print('Hello World!')
        'check quantization for if works as long as\n        output of all branches are quantized/observed consistently\n        '

        class M(torch.nn.Module):

            def __init__(self, cond):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                if self.cond:
                    x = torch.flatten(x)
                return x

        class M2(torch.nn.Module):

            def __init__(self, cond):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()
                self.cond = cond

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                if self.cond:
                    x = self.conv2(x)
                else:
                    x = torch.flatten(x)
                return x
        data = torch.rand((1, 3, 5, 5), dtype=torch.float)
        options = list(itertools.product([True, False], [True, False]))
        for (cond, tracing) in options:
            if tracing:
                m = torch.jit.trace(M(cond), data)
            else:
                m = torch.jit.script(M(cond))
            m = prepare_jit(m, {'': default_qconfig})
            assert len(attrs_with_prefix(m, '_observer_')) == 2
        for (cond, tracing) in options:
            if tracing:
                m = torch.jit.trace(M2(cond), data)
            else:
                m = torch.jit.script(M2(cond))
            m = prepare_jit(m, {'': default_qconfig})
            num_observers = 2 if tracing and (not cond) else 3
            assert len(attrs_with_prefix(m, '_observer_')) == num_observers

    def test_insert_quant_dequant(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.conv(x)
        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = default_per_channel_weight_observer.with_args(ch_axis=1) if is_per_channel else default_observer
            qconfig_dict = {'': QConfig(activation=observer, weight=observer)}
            m = prepare_jit(m, qconfig_dict)
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)
            m(data)
            m = convert_jit(m, debug=True)
            assert len(m._modules._c.items()) == 1, 'Expected to have single submodule of conv'
            m(data)
            quant_func = 'aten::quantize_per_channel' if is_per_channel else 'aten::quantize_per_tensor'
            FileCheck().check_count(quant_func, 3, exactly=True).run(m.graph)

    def test_insert_quant_dequant_shared_class_type(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                if False:
                    return 10
                return self.conv2(self.conv1(x))
        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            observer = default_per_channel_weight_observer.with_args(ch_axis=1) if is_per_channel else default_observer
            qconfig = QConfig(activation=observer, weight=observer)
            qconfig_dict = {'': qconfig}
            m = prepare_jit(m, qconfig_dict)
            assert len(attrs_with_prefix(m, '_observer_')) == 3, 'Expected to have 3 obervers'
            assert len(attrs_with_prefix(m.conv1, '_observer_')) == 1, 'Expected to have 1 obervers'
            assert len(attrs_with_prefix(m.conv2, '_observer_')) == 1, 'Expected to have 1 obervers'
            data = torch.randn(1, 3, 10, 10, dtype=torch.float)
            m(data)
            m = convert_jit(m, debug=True)
            m(data)
            assert m.conv1._c._type() == m.conv2._c._type()
            assert len(attrs_with_prefix(m, '_observer_')) == 0, 'Expected to have 0 obervers'
            assert len(attrs_with_prefix(m.conv1, '_observer_')) == 0, 'Expected to have 0 obervers'
            assert len(attrs_with_prefix(m.conv2, '_observer_')) == 0, 'Expected to have 0 obervers'
            quant_func = 'aten::quantize_per_channel' if is_per_channel else 'aten::quantize_per_tensor'
            for module in ['conv1', 'conv2']:
                conv = m._c.getattr(module)
                FileCheck().check(quant_func).check_next('aten::dequantize').check('prim::CallMethod[name="_conv_forward"]').check('return').run(get_forward_graph(conv))
                FileCheck().check_not(quant_func).check('aten::conv2d').check_not(quant_func).check('return').run(conv._get_method('_conv_forward').graph)

    def test_dedup_module_uses(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.relu(x)
                x -= 0.5
                return self.relu(x)
        data = torch.randn((2, 2))
        m = torch.jit.script(M())
        ref_res = m(data)
        assert len([x for (x, _) in m._modules._c.items() if x.startswith('relu')]) == 1, 'Expected to have 1 relu modules after dedup module uses'
        torch._C._jit_pass_dedup_module_uses(m._c)
        m = torch.jit._recursive.wrap_cpp_module(m._c)
        res = m(data)
        assert len([x for (x, _) in m._modules._c.items() if x.startswith('relu')]) == 2, 'Expected to have 2 relu modules after dedup module uses'
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = torch.dequantize(x)
                r = self.conv(x)
                r += x
                return r
        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M())
        ref_res = m(x)
        FileCheck().check_count('aten::dequantize', 1, exactly=True).run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count('aten::dequantize', 2, exactly=True).run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_replicate_dequantize_in_block(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self, cond):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.cond = cond

            def forward(self, x):
                if False:
                    return 10
                x = torch.dequantize(x)
                if self.cond:
                    x = self.conv(x)
                else:
                    x = x + 3
                return x
        x = torch.randn([1, 3, 10, 10], dtype=torch.float)
        x = torch.quantize_per_tensor(x, 0.5, 1, torch.quint8)
        m = torch.jit.script(M(True))
        ref_res = m(x)
        FileCheck().check_count('aten::dequantize', 1, exactly=True).run(m.graph)
        torch._C._jit_pass_replicate_dequantize(m.graph)
        FileCheck().check_count('aten::dequantize', 2, exactly=True).run(m.graph)
        FileCheck().check('aten::dequantize').check_next('CallMethod').run(m.graph)
        FileCheck().check('aten::dequantize').check('aten::dequantize').check_next('aten::add').run(m.graph)
        res = get_forward(m._c)(x)
        self.assertEqual(res, ref_res)

    def test_swap_functional_linear(self):
        if False:
            return 10

        def linear(input, weight, bias):
            if False:
                for i in range(10):
                    print('nop')
            return torch.nn.functional.linear(input, weight, bias)

        class M(torch.nn.Module):

            def forward(self, x, weight, bias):
                if False:
                    for i in range(10):
                        print('nop')
                x = torch.dequantize(x)
                weight = torch.dequantize(weight)
                x = linear(x, weight, bias)
                x = torch.quantize_per_tensor(x, scale=1.0, zero_point=0, dtype=torch.quint8)
                return x
        x = torch.rand((10, 5), dtype=torch.float)
        x = torch.quantize_per_tensor(x, scale=0.5, zero_point=1, dtype=torch.quint8)
        weight = torch.rand((5, 5), dtype=torch.float)
        weight = torch.quantize_per_tensor(weight, scale=0.5, zero_point=1, dtype=torch.qint8)
        bias = torch.rand(5, dtype=torch.float)
        m = torch.jit.script(M())
        ref_res = m(x, weight, bias)
        FileCheck().check('CallFunction').run(m.graph)
        torch._C._jit_pass_swap_functional_linear(m.graph)
        FileCheck().check('aten::linear').check_not('CallFunction').run(m.graph)
        res = m(x, weight, bias)
        self.assertEqual(res, ref_res)

    def test_replicate_quantize_for_if(self):
        if False:
            i = 10
            return i + 15
        'We want to move quantize nodes for output of prim::If\n        inside the prim::If blocks so that we can match quantization\n        patterns.\n        '

        class Res(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 1).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 1).float()
                self.use_skip = True

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                self.use_skip = cond
                if self.use_skip:
                    return self.conv(x)
                else:
                    return self.conv2(x)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x
        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {'': default_qconfig}
        m = torch.jit.script(M()).eval()
        m = quantize_jit(m, qconfig_dict, test_only_eval_fn, [data])
        FileCheck().check_count('quantized::conv2d(', 4, exactly=True).run(m.graph)

    def test_finalize_for_linear(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.fc(x)
        data = [[torch.rand((1, 5), dtype=torch.float)]]
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        FileCheck().check_count('aten::quantize_per_tensor', 1, exactly=True).check_not('quantized::linear_prepack').check('quantized::linear').run(model.graph)

    def test_inplace_option(self):
        if False:
            return 10
        for tracing in [True, False]:
            model = get_script_module(torch.nn.Conv2d(3, 3, 3).float(), tracing, self.img_data_2d[0][0])
            qconfig_dict = {'': default_qconfig}
            quantize_jit(model, qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=True)
            FileCheck().check('quantized::conv2d').run(model.graph)
            FileCheck().check_not('aten::conv2d').run(model.graph)

    def test_finalize_debug(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()
                self.avgpool = torch.nn.AvgPool2d(3)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                x = self.avgpool(x)
                return x
        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(M()).eval()
        model = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data], debug=True)
        FileCheck().check_not('quantized::conv2d').check('aten::conv2d').check('aten::avg_pool2d').check('aten::q_scale').check_next('aten::q_zero_point').check_next('prim::dtype').check_next('aten::quantize_per_tensor').check('aten::dequantize').run(model.graph)

    def test_module_list(self):
        if False:
            for i in range(10):
                print('nop')

        class SimpleLinearLayer(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.fc(x)

        class ComplexModel(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.layers = torch.nn.ModuleList([SimpleLinearLayer() for i in range(2)])

            def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
                if False:
                    while True:
                        i = 10
                states = []
                for layer in self.layers:
                    val = layer(x)
                    states.append(val)
                return states
        data = torch.rand((1, 5), dtype=torch.float)
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(ComplexModel()).eval()
        model = prepare_jit(model, qconfig_dict)
        assert len(attrs_with_prefix(model, '_observer')) == 3
        model(data)
        model = convert_jit(model, debug=False)
        FileCheck().check('quantized::linear').check('quantized::linear').run(model.graph)

    def test_conv_trace(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1d = torch.nn.Conv1d(3, 3, 3).float()
                self.conv2d = torch.nn.Conv2d(3, 3, 3).float()
                self.conv3d = torch.nn.Conv3d(3, 3, 3).float()

            def forward(self, x, y, z):
                if False:
                    return 10
                a = self.conv1d(x)
                b = self.conv2d(y)
                c = self.conv3d(z)
                return (a, b, c)
        qconfig_dict = {'': default_qconfig}
        inputs = (torch.rand((1, 3, 10), dtype=torch.float), torch.rand((1, 3, 10, 10), dtype=torch.float), torch.rand((1, 3, 10, 10, 10), dtype=torch.float))
        model = torch.jit.trace(M(), inputs).eval()
        m = prepare_jit(model, qconfig_dict)
        FileCheck().check('aten::conv1d').check_not('aten::_convolution').run(str(get_forward_graph(m.conv1d._c)))
        FileCheck().check('aten::conv2d').check_not('aten::_convolution').run(str(get_forward_graph(m.conv2d._c)))
        FileCheck().check('aten::conv3d').check_not('aten::_convolution').run(str(get_forward_graph(m.conv3d._c)))

    def test_convtranspose_trace(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.convtranspose1d = torch.nn.ConvTranspose1d(3, 3, 3).float()
                self.convtranspose2d = torch.nn.ConvTranspose2d(3, 3, 3).float()
                self.convtranspose3d = torch.nn.ConvTranspose3d(3, 3, 3).float()

            def forward(self, x, y, z):
                if False:
                    print('Hello World!')
                a = self.convtranspose1d(x)
                b = self.convtranspose2d(y)
                c = self.convtranspose3d(z)
                return (a, b, c)
        qconfig_dict = {'': default_qconfig}
        inputs = (torch.rand((1, 3, 10), dtype=torch.float), torch.rand((1, 3, 10, 10), dtype=torch.float), torch.rand((1, 3, 10, 10, 10), dtype=torch.float))
        model = torch.jit.trace(M(), inputs).eval()
        m = prepare_jit(model, qconfig_dict)
        FileCheck().check('aten::conv_transpose1d').check_not('aten::_convolution').run(str(get_forward_graph(m.convtranspose1d._c)))
        FileCheck().check('aten::conv_transpose2d').check_not('aten::_convolution').run(str(get_forward_graph(m.convtranspose2d._c)))
        FileCheck().check('aten::conv_transpose3d').check_not('aten::_convolution').run(str(get_forward_graph(m.convtranspose3d._c)))

    @unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines, ' Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs with instruction set support avx2 or newer.')
    def test_replicate_dequant_same_value(self):
        if False:
            for i in range(10):
                print('nop')

        class Mul(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                return x * x
        data = [[torch.rand((1, 3, 10, 10), dtype=torch.float)]]
        qconfig_dict = {'': default_qconfig}
        model = torch.jit.script(Mul()).eval()
        m = quantize_jit(model, qconfig_dict, test_only_eval_fn, [data])
        FileCheck().check('quantized::mul(').check_not('aten::mul').run(m.graph)

    def test_interface_with_fork(self):
        if False:
            return 10

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.embedding1 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, sparse=False, mode='sum')

            def forward(self, x, y):
                if False:
                    return 10
                return self.embedding1(x, y)

        class OrigMod(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.embedding1 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, sparse=False, mode='sum')

            def forward(self, x, y):
                if False:
                    return 10
                return self.embedding1(x, y)

        @torch.jit.interface
        class ModInterface(torch.nn.Module):

            def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                pass

        class TestModule(torch.nn.Module):
            proxy_mod: ModInterface

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                a = self.proxy_mod(x, y)
                b = self.sub(x, y)
                return b

        class MainModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.test = TestModule()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                fut = torch.jit._fork(self.test.forward, x, y)
                z = torch.jit._wait(fut)
                return z
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        m = torch.jit.trace(MainModule(), (indices, offsets))
        m.eval()
        int8_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, custom_op_name='embedding_bag_byte'), weight=PlaceholderObserver.with_args(custom_op_name='embedding_bag_byte'))
        m = prepare_jit(m, {'': int8_qconfig})
        m = convert_jit(m)
        FileCheck().check('quantized::embedding_bag_byte_rowwise_offsets').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantize_fork_wait(self):
        if False:
            i = 10
            return i + 15
        'Tests the case where fork and wait calls are in different subgraphs\n        Calling inline fork-wait only removes the fork call and leaves aten::wait\n        calls in the graph, with Tensor as input (instead of Future[Tensor])\n        '

        class MainModule(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.fork_ops = ForkModule()

            def init_values(self, x):
                if False:
                    i = 10
                    return i + 15
                shared_module = self.fork_ops(x)
                self.fork_dict = shared_module

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                val = torch.jit._wait(self.fork_ops(x))
                return val

        class TestModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                w = torch.ones(5, 5)
                b = torch.zeros(5)
                return torch.nn.functional.linear(x, w, b)

        class ForkModule(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.test = TestModule()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                fut = torch.jit._fork(self.test.forward, x)
                return fut
        model = MainModule().eval()
        traced = torch.jit.trace(model, (torch.randn(5, 5),))
        model = prepare_dynamic_jit(traced, {'': default_qconfig})
        model = convert_dynamic_jit(model)
        FileCheck().check('quantized::linear_dynamic').run(model.graph)
        b = io.BytesIO()
        torch.jit.save(model, b)

class TestQuantizeJitOps(QuantizationTestCase):
    """Test graph mode post training static quantization works
    for individual ops end to end.
    """

    @skipIfNoFBGEMM
    def test_linear(self):
        if False:
            while True:
                i = 10

        class ModuleLinear(torch.nn.Module):

            def __init__(self, has_relu=False, f_relu=False):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = torch.nn.Linear(30, 4).float()
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.relu(self.linear(x))

        class FuncLinear(torch.nn.Module):

            def __init__(self, has_relu=False, f_relu=False):
                if False:
                    print('Hello World!')
                super().__init__()
                self.w = torch.randn(4, 30)
                self.b = torch.randn(4)
                if has_relu:
                    if f_relu:
                        self.relu = F.relu
                    else:
                        self.relu = torch.nn.ReLU()
                else:
                    self.relu = torch.nn.Identity()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.relu(F.linear(x, self.w, self.b))
        data = [[torch.rand((1, 30), dtype=torch.float)]]
        for (model, tracing) in itertools.product([ModuleLinear(has_relu=False), FuncLinear(has_relu=False)], [True, False]):
            model = self.checkGraphModeOp(model, data, 'quantized::linear', tracing)
            FileCheck().check_count('aten::quantize_per_tensor', 1, exactly=True).run(model.graph)
            FileCheck().check_not('quantized::linear_prepack').run(model.graph)
        for (f_relu, tracing) in itertools.product([True, False], [True, False]):
            for model in [ModuleLinear(has_relu=True, f_relu=f_relu), FuncLinear(has_relu=True, f_relu=f_relu)]:
                model = self.checkGraphModeOp(model, data, 'quantized::linear_relu', tracing)
                checker = FileCheck().check_not('aten::linear').check_not('aten::relu').check_not('quantized::linear(').check_not('quantized::relu(').run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_conv(self):
        if False:
            for i in range(10):
                print('nop')
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class Conv(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.conv(x)
        options = itertools.product([1, 2, 3], [True, False])
        for (dim, tracing) in options:
            model = self.checkGraphModeOp(Conv(dim), self.img_data_dict[dim], f'quantized::conv{dim}d', tracing)
            FileCheck().check_count('aten::quantize_per_tensor', 1, exactly=True).run(model.graph)
            FileCheck().check_not(f'quantized::conv{dim}d_prepack').run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_conv_relu(self):
        if False:
            while True:
                i = 10
        'tests for conv1d_relu/conv2d_relu/conv3d_relu'
        conv_module = {1: torch.nn.Conv1d, 2: torch.nn.Conv2d, 3: torch.nn.Conv3d}

        class ConvNdRelu(torch.nn.Module):

            def __init__(self, dim, inplace):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.relu(self.conv(x))

        class ConvNdFunctionalRelu(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                if False:
                    return 10
                return F.relu(self.conv(x))

        class ConvNdInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = conv_module[dim](3, 3, 3).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return F.relu(self.conv(x), True)
        options = itertools.product([1, 2, 3], [True, False])
        for (dim, tracing) in options:
            for orig_m in [ConvNdRelu(dim, True), ConvNdRelu(dim, False), ConvNdFunctionalRelu(dim), ConvNdInplaceFunctionalRelu(dim)]:
                conv_name = f'conv{dim}d'
                m = self.checkGraphModeOp(orig_m, self.img_data_dict[dim], f'quantized::conv{dim}d_relu(', tracing=tracing)
                FileCheck().check_not(f'aten::conv{dim}d(').check_not('aten::relu').check_not(f'quantized::conv{dim}d(').check_not('quantized::relu(').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_alpha(self):
        if False:
            for i in range(10):
                print('nop')
        'Test quant fusion for multiple aten::add using same\n        constant alpha as the third argument\n        '

        class QuantizedAdd(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                z = x + y
                w = y + z
                return z + w
        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float), torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        for tracing in [True, False]:
            m = self.checkGraphModeOp(QuantizedAdd(), data, 'quantized::add', tracing)
            FileCheck().check_count('quantized::add', 3, exactly=True).run(m.graph)
            FileCheck().check_not('aten::add').check_not('aten::add_').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_relu_alpha(self):
        if False:
            return 10
        'Test quant fusion for multiple aten::add using same\n        constant alpha as the third argument in add_relu pattern\n        '

        class AddRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = self.relu(x)
                x = x + y
                return self.relu(x)

        class InplaceAddRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = self.relu(x)
                x += y
                return self.relu(x)

        class AddFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    return 10
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = F.relu(x)
                x = x + y
                return F.relu(x)

        class InplaceAddFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = F.relu(x)
                x += y
                return F.relu(x)

        class AddInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    return 10
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                x = F.relu(x, True)
                x = x + y
                return F.relu(x, True)

        class InplaceAddInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                x = F.relu(x, True)
                x += y
                return F.relu(x, True)
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float), torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        for m_orig in [AddRelu(True), AddRelu(False), InplaceAddRelu(True), InplaceAddRelu(False), AddFunctionalRelu(), InplaceAddFunctionalRelu(), AddInplaceFunctionalRelu(), InplaceAddInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m_orig, data, 'quantized::add_relu(', tracing=tracing)
                FileCheck().check_count('quantized::add_relu(', 2, exactly=True).run(m.graph)
                FileCheck().check_not('aten::add(').check_not('aten::add_(').check_not('aten::relu(').check_not('aten::relu_(').check_not('quantized::add(').check_not('quantized::relu(').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add(self):
        if False:
            while True:
                i = 10

        class QuantizedAdd(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    return 10
                x = self.conv1(x)
                y = self.conv2(y)
                return x + y

        class QuantizedInplaceAdd(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return x

        class NonQuantizedAdd(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return x + y

        class NonQuantizedInplaceAdd(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x += y
                return x
        data = [[torch.randn(1, 2, 3, 3, dtype=torch.float), torch.randn(1, 2, 3, 3, dtype=torch.float)]]
        for (m, quantized) in [(QuantizedAdd(), True), (QuantizedInplaceAdd(), True), (NonQuantizedAdd(), False), (NonQuantizedInplaceAdd(), False)]:
            for tracing in [True, False]:
                op = 'quantized::add' if quantized else 'aten::add'
                m = self.checkGraphModeOp(m, data, op, tracing)
                if quantized:
                    FileCheck().check_not('aten::add').check_not('aten::add_').run(m.graph)
                else:
                    FileCheck().check_not('quantized::add').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_scalar(self):
        if False:
            while True:
                i = 10

        class QuantizedAddScalar(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                return x + 3

        class QuantizedInplaceAddScalar(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                x += 3
                return x

        class NonQuantizedAddScalar(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x + 3

        class NonQuantizedInplaceAddScalar(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x += 3
                return x
        data = [[torch.randn(1, 2, 3, 3, dtype=torch.float)]]
        for (m, quantized) in [(QuantizedAddScalar(), True), (QuantizedInplaceAddScalar(), True), (NonQuantizedAddScalar(), False), (NonQuantizedInplaceAddScalar(), False)]:
            for tracing in [True, False]:
                op = 'quantized::add_scalar' if quantized else 'aten::add'
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                if quantized:
                    FileCheck().check_not('aten::add').check_not('aten::add_').run(m.graph)
                else:
                    FileCheck().check_not('quantized::add_scalar').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_relu(self):
        if False:
            for i in range(10):
                print('nop')

        class AddRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return self.relu(x)

        class InplaceAddRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return self.relu(x)

        class AddFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return F.relu(x)

        class InplaceAddFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return F.relu(x)

        class AddInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x = x + y
                return F.relu(x, True)

        class InplaceAddInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x += y
                return F.relu(x, True)
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float), torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        for m in [AddRelu(True), AddRelu(False), InplaceAddRelu(True), InplaceAddRelu(False), AddFunctionalRelu(), InplaceAddFunctionalRelu(), AddInplaceFunctionalRelu(), InplaceAddInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, 'quantized::add_relu(', tracing)
                FileCheck().check_not('aten::add(').check_not('aten::add_(').check_not('aten::relu(').check_not('aten::relu_(').check_not('quantized::add(').check_not('quantized::relu(').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_add_scalar_relu(self):
        if False:
            return 10

        class AddScalarRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                return self.relu(x + 3)

        class InplaceAddScalarRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                x += 3
                return self.relu(x)

        class AddScalarFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                return F.relu(x + 3)

        class InplaceAddScalarFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    return 10
                x = self.conv(x)
                x += 3
                return F.relu(x)

        class AddScalarInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                return F.relu(x + 3, True)

        class InplaceAddScalarInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    return 10
                x = self.conv(x)
                x += 3
                return F.relu(x, True)
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        for m in [AddScalarRelu(True), AddScalarRelu(False), InplaceAddScalarRelu(True), InplaceAddScalarRelu(False), AddScalarFunctionalRelu(), InplaceAddScalarFunctionalRelu(), AddScalarInplaceFunctionalRelu(), InplaceAddScalarInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, 'quantized::add_scalar_relu', tracing, check=False)
                FileCheck().check_not('aten::add(').check_not('aten::add_(').check_not('aten::relu(').check_not('aten::relu_(').check_not('quantized::add_scalar(').check_not('quantized::relu(').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_cat(self):
        if False:
            i = 10
            return i + 15
        'quantization of the output of cat will be depend on the\n        input of cat. we only quantize the output of cat when its inputs are quantized.\n        '

        class QuantizedCat(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    return 10
                x = self.conv1(x)
                y = self.conv2(y)
                return torch.cat([x, y], 1)

        class NonQuantizedCat(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                return torch.cat([x, y], 1)
        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float), torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        for tracing in [True, False]:
            m = self.checkGraphModeOp(QuantizedCat(), data, 'quantized::cat', tracing)
            FileCheck().check_not('aten::cat').run(m.graph)
            m = self.checkGraphModeOp(NonQuantizedCat(), data, 'aten::cat', tracing)
            FileCheck().check_not('quantized::cat').run(m.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm(self):
        if False:
            print('Hello World!')
        bn_module = {1: torch.nn.BatchNorm1d, 2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class M(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.bn(x)
        options = itertools.product([True, False], [1, 2, 3])
        for (tracing, dim) in options:
            model = self.checkGraphModeOp(M(dim), self.img_data_dict[dim], 'quantized::batch_norm', tracing)
            FileCheck().check_not('aten::batch_norm').run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNRelu(self):
        if False:
            return 10
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNRelu(torch.nn.Module):

            def __init__(self, dim, inplace):
                if False:
                    print('Hello World!')
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)
                self.relu = torch.nn.ReLU(inplace=inplace)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.relu(self.bn(x))
        options = itertools.product([True, False], [2, 3])
        for (tracing, dim) in options:
            for instance in [BNRelu(dim, True), BNRelu(dim, False)]:
                model = self.checkGraphModeOp(instance, self.img_data_dict[dim], 'quantized::batch_norm_relu', tracing)
                FileCheck().check_not('aten::batch_norm').check_not('aten::relu').check_not('aten::relu_').run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNFuncRelu(self):
        if False:
            print('Hello World!')
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNFuncRelu(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return F.relu(self.bn(x), False)
        options = itertools.product([True, False], [2, 3])
        for (tracing, dim) in options:
            instance = BNFuncRelu(dim)
            model = self.checkGraphModeOp(instance, self.img_data_dict[dim], 'quantized::batch_norm_relu', tracing)
            FileCheck().check_not('aten::batch_norm').check_not('aten::relu').check_not('aten::relu_').run(model.graph)

    @skipIfNoFBGEMM
    def test_qbatch_norm_relu_BNFuncInplaceRelu(self):
        if False:
            print('Hello World!')
        bn_module = {2: torch.nn.BatchNorm2d, 3: torch.nn.BatchNorm3d}

        class BNFuncInplaceRelu(torch.nn.Module):

            def __init__(self, dim):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.bn = bn_module[dim](3).to(torch.float)

            def forward(self, x):
                if False:
                    return 10
                return F.relu(self.bn(x), True)
        options = itertools.product([True, False], [2, 3])
        for (tracing, dim) in options:
            instance = BNFuncInplaceRelu(dim)
            model = self.checkGraphModeOp(instance, self.img_data_dict[dim], 'quantized::batch_norm_relu', tracing)
            FileCheck().check_not('aten::batch_norm').check_not('aten::relu').check_not('aten::relu_').run(model.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul(self):
        if False:
            for i in range(10):
                print('nop')

        class QuantizedMul(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                return x * y

        class QuantizedInplaceMul(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return x

        class NonQuantizedMul(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x * y

        class NonQuantizedInplaceMul(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                x *= y
                return x
        data = [[torch.randn(1, 2, 10, 10, dtype=torch.float), torch.randn(1, 2, 10, 10, dtype=torch.float)]]
        for (m, quantized) in [(QuantizedMul(), True), (QuantizedInplaceMul(), True), (NonQuantizedMul(), False), (NonQuantizedInplaceMul(), False)]:
            for tracing in [True, False]:
                op = 'quantized::mul' if quantized else 'aten::mul'
                m = self.checkGraphModeOp(m, data, op, tracing)
                if quantized:
                    FileCheck().check_not('aten::mul').check_not('aten::mul_').run(m.graph)
                else:
                    FileCheck().check_not('quantized::mul').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar(self):
        if False:
            while True:
                i = 10

        class QuantizedMulScalar(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv(x)
                return x * 3

        class QuantizedInplaceMulScalar(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                x *= 3
                return x

        class NonQuantizedMulScalar(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x * 3

        class NonQuantizedInplaceMulScalar(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                x *= 3
                return x
        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        for (m, quantized) in [(QuantizedMulScalar(), True), (QuantizedInplaceMulScalar(), True), (NonQuantizedMulScalar(), False), (NonQuantizedInplaceMulScalar(), False)]:
            for tracing in [True, False]:
                op = 'quantized::mul_scalar' if quantized else 'aten::mul'
                m = self.checkGraphModeOp(m, data, op, tracing, check=False)
                if quantized:
                    FileCheck().check_not('aten::mul').check_not('aten::mul_').run(m.graph)
                else:
                    FileCheck().check_not('quantized::mul_scalar').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_relu(self):
        if False:
            return 10

        class MulRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return self.relu(x)

        class InplaceMulRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    return 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return self.relu(x)

        class MulFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return F.relu(x)

        class InplaceMulFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return F.relu(x)

        class MulInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                x = self.conv1(x)
                y = self.conv2(y)
                x = x * y
                return F.relu(x, True)

        class InplaceMulInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv1 = torch.nn.Conv2d(2, 2, 2).float()
                self.conv2 = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                x = self.conv1(x)
                y = self.conv2(y)
                x *= y
                return F.relu(x, True)
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float), torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        for m in [MulRelu(True), MulRelu(False), InplaceMulRelu(True), InplaceMulRelu(False), MulFunctionalRelu(), InplaceMulFunctionalRelu(), MulInplaceFunctionalRelu(), InplaceMulInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, 'quantized::mul_relu(', tracing)
                FileCheck().check_not('aten::mul(').check_not('aten::mul_(').check_not('aten::relu(').check_not('aten::relu_(').check_not('quantized::mul(').check_not('quantized::relu(').run(m.graph)

    @skipIfNoFBGEMM
    def test_quantized_mul_scalar_relu(self):
        if False:
            print('Hello World!')

        class MulScalarRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                return self.relu(x * 3)

        class InplaceMulScalarRelu(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu = torch.nn.ReLU(inplace)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                x *= 3
                return self.relu(x)

        class MulScalarFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                return F.relu(x * 3)

        class InplaceMulScalarFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                x *= 3
                return F.relu(x)

        class MulScalarInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.conv(x)
                return F.relu(x * 3, True)

        class InplaceMulScalarInplaceFunctionalRelu(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                x *= 3
                return F.relu(x, True)
        data = [[torch.randn(1, 2, 5, 5, dtype=torch.float)]]
        for m in [MulScalarRelu(True), MulScalarRelu(False), InplaceMulScalarRelu(True), InplaceMulScalarRelu(False), MulScalarFunctionalRelu(), InplaceMulScalarFunctionalRelu(), MulScalarInplaceFunctionalRelu(), InplaceMulScalarInplaceFunctionalRelu()]:
            for tracing in [True, False]:
                m = self.checkGraphModeOp(m, data, 'quantized::mul_scalar_relu', tracing, check=False)
                FileCheck().check_not('aten::mul(').check_not('aten::mul_(').check_not('aten::relu(').check_not('aten::relu_(').check_not('quantized::mul_scalar(').check_not('quantized::relu(').run(m.graph)

    @override_qengines
    def test_hardswish(self):
        if False:
            return 10

        class FunctionalHardswish(torch.nn.Module):

            def __init__(self, inplace):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.inplace = inplace

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.nn.functional.hardswish(input, inplace=self.inplace)
        modules = [torch.nn.Hardswish(), FunctionalHardswish(True), FunctionalHardswish(False)]
        for test_case in itertools.product([True, False], modules):
            (tracing, m) = test_case
            m = self.checkGraphModeOp(m, self.img_data_2d, 'quantized::hardswish', tracing)
            FileCheck().check_not('aten::hardswish').check_not('aten::hardswish_').run(m.graph)

    @override_qengines
    def test_elu(self):
        if False:
            print('Hello World!')

        class FunctionalELU(torch.nn.Module):

            def __init__(self, inplace=False):
                if False:
                    print('Hello World!')
                super().__init__()
                self.inplace = inplace

            def forward(self, input):
                if False:
                    print('Hello World!')
                return torch.nn.functional.elu(input, inplace=self.inplace)
        modules = [torch.nn.ELU, FunctionalELU]
        for test_case in itertools.product([True, False], [True, False], modules):
            (tracing, inplace, mod_class) = test_case
            m = mod_class(inplace=inplace)
            m = self.checkGraphModeOp(m, self.img_data_2d, 'quantized::elu', tracing)
            FileCheck().check_not('aten::elu').check_not('aten::elu_').run(m.graph)

    @override_qengines
    def test_layer_norm(self):
        if False:
            while True:
                i = 10
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)] for _ in range(2)]
        layer_norm = torch.nn.LayerNorm([2, 5, 5])
        for tracing in [True, False]:
            m = self.checkGraphModeOp(layer_norm, data, 'quantized::layer_norm', tracing)
            FileCheck().check_not('aten::layer_norm').run(m.graph)

    @override_qengines
    def test_group_norm(self):
        if False:
            print('Hello World!')
        data = [[torch.rand((1, 4, 5, 5), dtype=torch.float)] for _ in range(2)]
        group_norm = torch.nn.GroupNorm(2, 4)
        for tracing in [True, False]:
            m = self.checkGraphModeOp(group_norm, data, 'quantized::group_norm', tracing)
            FileCheck().check_not('aten::group_norm').run(m.graph)

    @override_qengines
    def test_instance_norm(self):
        if False:
            for i in range(10):
                print('nop')
        data_1d = [[torch.rand((1, 4, 5), dtype=torch.float)] for _ in range(2)]
        data_2d = [[torch.rand((1, 4, 5, 1), dtype=torch.float)] for _ in range(2)]
        data_3d = [[torch.rand((1, 4, 5, 1, 1), dtype=torch.float)] for _ in range(2)]
        data = {1: data_1d, 2: data_2d, 3: data_3d}
        instance_norm_modules = {1: torch.nn.InstanceNorm1d, 2: torch.nn.InstanceNorm2d, 3: torch.nn.InstanceNorm3d}
        options = itertools.product([1, 2, 3], [True, False])
        for (dim, tracing) in options:
            instance_norm = instance_norm_modules[dim](4)
            m = self.checkGraphModeOp(instance_norm, data[dim], 'quantized::instance_norm', tracing)
            FileCheck().check_not('aten::instance_norm').run(m.graph)

    @skipIfNoFBGEMM
    def test_dequantize_tuple(self):
        if False:
            while True:
                i = 10
        'Make sure dequantize can support Tuple of tensor'

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv1 = torch.nn.Conv2d(3, 3, 3).float()
                self.conv2 = torch.nn.Conv2d(3, 3, 3).float()

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if False:
                    for i in range(10):
                        print('nop')
                x1 = self.conv1(x)
                x2 = self.conv2(x)
                return (x1, x2)
        for tracing in [True, False]:
            self.checkGraphModeOp(M(), self.img_data_2d, 'quantized::conv2d', tracing)

    @skipIfNoFBGEMM
    def test_clamp(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.conv = torch.nn.Conv2d(2, 2, 2).float()
                self.relu6 = torch.nn.ReLU6()
                self.relu6_ = torch.nn.ReLU6(True)
                self.hardtanh = torch.nn.Hardtanh()
                self.hardtanh_ = torch.nn.Hardtanh(inplace=True)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.conv(x)
                x = self.relu6(x)
                self.relu6_(x)
                x = F.relu6(x)
                x = torch.clamp(x, -3, 3)
                x = x.clamp(-2.5, 2.5)
                x = self.hardtanh(x)
                self.hardtanh_(x)
                x = F.hardtanh(x)
                F.hardtanh_(x)
                return x
        data = [[torch.rand((1, 2, 5, 5), dtype=torch.float)]]
        options = itertools.product(['aten::clamp', 'aten::hardtanh', 'aten::hardtanh_'], [True, False])
        for (op, tracing) in options:
            m = self.checkGraphModeOp(M(), data, op, tracing)
            FileCheck().check_count('aten::quantize_per_tensor', 1, exactly=True).run(m.graph)
            FileCheck().check_count('aten::dequantize', 1, exactly=True).run(m.graph)

    def test_general_shape_ops(self):
        if False:
            return 10
        'A test that checks dequantize will be swapped for\n        all supported general shape ops like aten::flatten\n        without actually checking for execution of these ops\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.maxpool1d = torch.nn.MaxPool1d(kernel_size=3)
                self.maxpool2d = torch.nn.MaxPool2d(kernel_size=3)
                self.maxpool3d = torch.nn.MaxPool3d(kernel_size=3)
                self.dropout = torch.nn.Dropout()
                self.conv1 = torch.nn.Conv2d(3, 3, 3)
                self.conv2 = torch.nn.Conv2d(3, 3, 3)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.conv1(x)
                x = x + 3
                x = x * 3
                x += 3
                x *= 3
                x = x + 3
                x = F.relu(x)
                x += 3
                x = F.relu(x)
                x = x * 3
                x = F.relu(x)
                x *= 3
                x = F.relu(x)
                x = self.maxpool1d(x)
                x = self.maxpool2d(x)
                x = self.maxpool3d(x)
                x = torch.flatten(x)
                x = torch.max(x)
                x = torch.min(x)
                x = x.reshape([-1])
                x = x.resize_(1, 1, x.numel())
                x = x.view(-1)
                xs = [x, x]
                (x, y) = xs
                xs = (x, x)
                (x, y) = xs
                x = x.transpose(1, 2)
                x = x.contiguous()
                (x, y) = torch.chunk(x, 2)
                x = F.dropout(x)
                x = self.dropout(x)
                (x, _) = torch.sort(x)
                x = x.permute(0, 2, 3, 1)
                x = torch.repeat_interleave(x, 3, 1)
                x = self.relu(x)
                x = F.relu(x)
                x.relu_()
                x = x.squeeze(0)
                x.squeeze_(0)
                x = torch.squeeze(x, 0)
                x = x.unsqueeze(0)
                x.unsqueeze_(0)
                x = torch.unsqueeze(x, 0)
                x = x.detach()
                x.detach_()
                x = x.repeat(4, 2)
                y = []
                y.append(x)
                z = torch.stack(y, 0)
                z = [z, z]
                (x, _) = z
                x = self.conv2(x)
                return x
        data = torch.rand(1, 3, 10, 10)
        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, 'forward', {'': qconfig}, inplace=False))
        m = convert_jit(m)
        FileCheck().check_count('aten::quantize_per_tensor', 1, exactly=True).run(m.graph)
        FileCheck().check_count('quantized::conv2d(', 2, exactly=True).run(m.graph)
        FileCheck().check_count('aten::dequantize', 1, exactly=True).run(m.graph)
        FileCheck().check('quantized::add_scalar').check('quantized::mul_scalar').run(m.graph)

    def test_general_value_ops(self):
        if False:
            for i in range(10):
                print('nop')
        ' A test that checks correct patterns are produced for\n        all supported general value ops like aten::avg_pool2d         without actually checking for execution of these ops\n        '

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 3, 3)
                self.avg_pool1d = torch.nn.AvgPool1d(3)
                self.avg_pool2d = torch.nn.AvgPool2d(3)
                self.avg_pool3d = torch.nn.AvgPool3d(3)
                self.adaptive_avg_pool1d = torch.nn.AdaptiveAvgPool1d(1)
                self.adaptive_avg_pool2d = torch.nn.AdaptiveAvgPool2d((1, 1))
                self.adaptive_avg_pool3d = torch.nn.AdaptiveAvgPool3d((1, 1, 1))
                self.leaky_relu = torch.nn.LeakyReLU()
                self.hardsigmoid = torch.nn.Hardsigmoid()
                self.sigmoid = torch.nn.Sigmoid()
                self.tanh = torch.nn.Tanh()

            def forward(self, x):
                if False:
                    print('Hello World!')
                x = self.conv(x)
                x = self.avg_pool1d(x)
                x = self.avg_pool2d(x)
                x = self.avg_pool3d(x)
                x = self.adaptive_avg_pool1d(x)
                x = self.adaptive_avg_pool2d(x)
                x = self.adaptive_avg_pool3d(x)
                x = F.avg_pool1d(x, 3)
                x = F.avg_pool2d(x, 3)
                x = F.avg_pool3d(x, 3)
                x = F.adaptive_avg_pool1d(x, 1)
                x = F.adaptive_avg_pool2d(x, (1, 1))
                x = F.adaptive_avg_pool3d(x, (1, 1, 1))
                x = torch.mean(x)
                x = torch.mean(x, [2, 3], False)
                x = x.mean()
                x = x.mean([2, 3], True)
                x = F.interpolate(x, 4, mode='nearest')
                x = F.upsample(x, (32, 32))
                x = F.upsample_nearest(x, (32, 32))
                x = F.interpolate(x, 4, mode='linear')
                x = F.upsample_bilinear(x, (32, 32))
                x = self.leaky_relu(x)
                x = F.leaky_relu(x)
                x.leaky_relu_()
                x = self.hardsigmoid(x)
                x = F.hardsigmoid(x)
                x.hardsigmoid_()
                x = self.sigmoid(x)
                x = torch.sigmoid(x)
                x = x.sigmoid()
                x.sigmoid_()
                x = self.tanh(x)
                x = torch.tanh(x)
                x = x.tanh()
                x.tanh_()
                x = self.conv(x)
                return x
        m = torch.jit.script(M())
        qconfig = script_qconfig(default_qconfig)
        data = torch.rand(1, 3, 10, 10)
        get_forward(qconfig.activation)(data)
        get_forward(qconfig.weight)(data)
        m = wrap_cpp_module(torch._C._jit_pass_insert_observers(m._c, 'forward', {'': qconfig}, inplace=False))
        m1 = convert_jit(m, debug=True)
        num_op_by_num_quant = {1: 32, 2: 2, 3: 3}
        num_quantize_per_tensor = 1
        for (num_quant, num_op) in num_op_by_num_quant.items():
            num_quantize_per_tensor += num_op * num_quant
        num_quantize_per_tensor -= 4
        FileCheck().check_count('aten::quantize_per_tensor(', num_quantize_per_tensor, exactly=True).run(m1.graph)
        m2 = convert_jit(m, debug=False)
        FileCheck().check_count('aten::quantize_per_tensor(', 1, exactly=True).run(m2.graph)
        FileCheck().check_count('quantized::conv2d(', 2, exactly=True).check('aten::dequantize(').run(m2.graph)

    @override_qengines
    def test_conv_with_benchmark_flag(self):
        if False:
            i = 10
            return i + 15
        'Verifies that convolutions get quantized when\n        torch.backends.cudnn.benchmark is enabled\n        '
        if not qengine_is_qnnpack():
            return
        with torch.backends.cudnn.flags(enabled=True):
            m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
            m.eval()
            m = torch.jit.trace(m, torch.rand(4, 1, 4, 4))
            qconfig = torch.ao.quantization.get_default_qconfig('qnnpack')
            prepared_model = torch.ao.quantization.prepare_jit(m, {'': qconfig})
            prepared_model(torch.rand(4, 1, 4, 4))
            converted_model = torch.ao.quantization.convert_jit(prepared_model)
            FileCheck().check('quantized::conv2d').run(converted_model.graph)

    @skipIfNoFBGEMM
    def test_cat_linear(self):
        if False:
            i = 10
            return i + 15

        class LinearModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weight = torch.randn(5, 5)

            def forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                a = torch.cat([x, y])
                b = F.linear(a, self.weight)
                c = F.linear(b, self.weight)
                return (b, c)
        model = LinearModel().eval()
        qconfig = {'': default_qconfig}
        float_model = torch.jit.script(model)
        prepared_model = prepare_jit(float_model, qconfig)
        prepared_model(torch.rand(5, 5), torch.rand(5, 5))
        converted_model = convert_jit(prepared_model)
        FileCheck().check('quantized::linear').check('quantized::linear').run(converted_model.graph)

class TestQuantizeDynamicJitPasses(QuantizationTestCase):

    def test_prepare_dynamic(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.fc(x)
        model = torch.jit.script(M())
        for qconfig in [float16_dynamic_qconfig, default_dynamic_qconfig]:
            m = prepare_dynamic_jit(model, {'': qconfig})
            assert len(attrs_with_prefix(m.fc, '_observer_')) == 1
            if qconfig == float16_dynamic_qconfig:
                observer_name = 'PlaceholderObserver = prim::GetAttr[name="_observer_'
                FileCheck().check(observer_name).run(m.fc.graph)
            else:
                assert len(attrs_with_prefix(m, '_observer_')) == 1
                observer_name = 'Observer = prim::GetAttr[name="_observer_'
                FileCheck().check(observer_name).check('prim::GetAttr[name="fc"]').check('prim::CallMethod').check_not(observer_name).run(m.graph)

    def test_prepare_dynamic_child_qconfig(self):
        if False:
            print('Hello World!')

        class Sub(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                if False:
                    return 10
                return self.fc(x)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.conv = torch.nn.Conv2d(3, 5, 3)
                self.sub = Sub()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.sub(self.conv(x))
        m = torch.jit.script(M())
        m = prepare_dynamic_jit(m, {'sub.fc': default_dynamic_qconfig})
        assert len(attrs_with_prefix(m, '_observer_')) == 1
        assert len(attrs_with_prefix(m.conv, '_observer_')) == 0
        assert len(attrs_with_prefix(m.sub, '_observer_')) == 0
        assert len(attrs_with_prefix(m.sub.fc, '_observer_')) == 1
        FileCheck().check('prim::GetAttr[name="sub').check('prim::CallMethod').check('Observer = prim::GetAttr[name="_observer_').check('prim::CallMethod').check_not('Observer = prim::GetAttr[name="_observer_').run(m.graph)

    def test_insert_quant_dequant_linear_dynamic(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.fc1(x)
                return self.fc2(x)
        for is_per_channel in [True, False]:
            m = torch.jit.script(M())
            qconfig = per_channel_dynamic_qconfig if is_per_channel is True else default_dynamic_qconfig
            m = quantize_dynamic_jit(m, {'': qconfig}, debug=True)
            assert len(m._modules._c.items()) == 2, 'Expected to have two submodule of linear'
            wt_quant_func = 'aten::quantize_per_channel' if is_per_channel else 'aten::quantize_per_tensor'
            act_quant_func = 'aten::quantize_per_tensor'
            FileCheck().check('aten::_choose_qparams_per_tensor').check_next(act_quant_func).check_next('aten::dequantize').check('aten::_choose_qparams_per_tensor').check_next(act_quant_func).check_next('aten::dequantize').check(wt_quant_func).check_next('aten::dequantize').check_not(wt_quant_func).check('return').run(m.graph)

    @override_qengines
    def test_dynamic_multi_op(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fc1 = torch.nn.Linear(5, 5).to(dtype=torch.float)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = x + 5
                return self.fc1(x)
        x = torch.randn(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(M(), x, 'quantized::linear_dynamic', tracing=tracing, dynamic=True)
            FileCheck().check('aten::add').run(model.graph)

    @override_qengines
    def test_dynamic_quant_multi_uses(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                if False:
                    return 10
                size1 = x.size()
                size2 = x.size()
                return (self.fc(x), size1, size2)
        x = torch.randn(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(M(), x, 'quantized::linear_dynamic', tracing=tracing, dynamic=True)
            FileCheck().check_not('aten::_choose_qparams_per_tensor').run(model.graph)

    @override_qengines
    def test_dynamic_shared_weights(self):
        if False:
            while True:
                i = 10

        class myMod(torch.nn.Module):

            def __init__(self, weight):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = nn.Linear(5, 5)
                self.linear.weight = weight

            def forward(self, x):
                if False:
                    return 10
                return self.linear(x)

        class DynamicModel(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))
                self.mod1 = myMod(self.weight)

            def forward(self, x):
                if False:
                    print('Hello World!')
                y = self.mod1(x)
                z = torch.nn.functional.linear(y, self.weight)
                return z
        model = torch.jit.script(DynamicModel()).eval()
        data = torch.randn(5, 5, dtype=torch.float)
        quant_ops = ['mod1', '']
        counts = [1, 2]
        for (op, count) in zip(quant_ops, counts):
            qconfig_dict = {op: default_dynamic_qconfig}
            m1 = quantize_dynamic_jit(model, qconfig_dict)
            out_graph = m1(data)
            FileCheck().check_count('quantized::linear_dynamic(', count, exactly=True).check_not('aten::_choose_qparams_per_tensor').run(m1.graph)
            m2 = prepare_dynamic_jit(model, qconfig_dict)
            m2(data)
            m2 = convert_dynamic_jit(m2, debug=False)
            out_ref = m2(data)
            self.assertEqual(out_graph, out_ref)

    @override_qengines
    def test_dynamic_with_if(self):
        if False:
            for i in range(10):
                print('nop')

        class Res(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(5, 5))

            def forward(self, x: torch.Tensor, cond: bool) -> torch.Tensor:
                if False:
                    return 10
                if cond:
                    return torch.nn.functional.linear(x, self.weight)
                else:
                    return torch.nn.functional.linear(x, self.weight)

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.res1 = Res()
                self.res2 = Res()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                x = self.res1(x, True)
                x = self.res2(x, False)
                return x
        model = torch.jit.script(M()).eval()
        data = torch.randn(5, 5, dtype=torch.float)
        qconfig_dict = {'': default_dynamic_qconfig}
        for tracing in [True, False]:
            m1 = self.checkGraphModeOp(M(), data, 'quantized::linear_dynamic', tracing=tracing, dynamic=True)
            FileCheck().check_count('quantized::linear_dynamic(', 2, exactly=True).check_not('aten::_choose_qparams_per_tensor').run(m1.graph)
        ref_qparams = []
        qconfig = script_qconfig(default_dynamic_qconfig)
        wt_module = wrap_cpp_module(qconfig.weight)
        for wt in [model.res1.weight, model.res2.weight]:
            wt_module(wt)
            qparams = wt_module.calculate_qparams()
            ref_qparams.append((qparams[0].item(), qparams[1].item()))
        m2 = quantize_dynamic_jit(model, qconfig_dict, debug=True)
        graph_params = []
        for (x, obs) in m2._modules._c.items():
            if x == 'res1':
                graph_params.append((obs.getattr('weight.2_scale_0'), obs.getattr('weight.2_zero_point_0')))
            elif x == 'res2':
                graph_params.append((obs.getattr('weight.4_scale_0'), obs.getattr('weight.4_zero_point_0')))
        self.assertEqual(ref_qparams, graph_params)

    def test_dynamic_weight_observer(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5).float()
                self.fc2 = torch.nn.Linear(5, 5).float()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = self.fc(x)
                return self.fc2(x)
        qconfig_dict = {'': default_dynamic_qconfig}
        eager_model = M().eval()
        for tracing in [True, False]:
            x = torch.rand(5, 5)
            model = get_script_module(eager_model, tracing, x)
            ref_qparams = []
            for wt in [model.fc.weight, model.fc2.weight]:
                wt_module = default_dynamic_qconfig.weight()
                wt_module(wt)
                qparams = wt_module.calculate_qparams()
                ref_qparams.append((qparams[0].item(), qparams[1].item()))
            model = quantize_dynamic_jit(model, qconfig_dict, debug=True)
            graph_qparams = []
            for (x, obs) in model._modules._c.items():
                n = 2 if x == 'fc' and tracing else 1
                graph_qparams.append((obs.getattr(f'weight.{n}_scale_0'), obs.getattr(f'weight.{n}_zero_point_0')))
            self.assertEqual(ref_qparams, graph_qparams)

    def test_convert_dynamic_fp16(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.fc(x)
        m = torch.jit.script(M())
        m = quantize_dynamic_jit(m, {'': float16_dynamic_qconfig}, debug=True)
        FileCheck().check('aten::_saturate_weight_to_fp16').check('aten::linear').check_not('aten::dequantize').check_not('aten::quantize').run(m.graph)

    def test_quantize_dynamic_fp16(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.fc = torch.nn.Linear(5, 5)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.fc(x)
        m = torch.jit.script(M())
        m = quantize_dynamic_jit(m, {'': float16_dynamic_qconfig})
        FileCheck().check('quantized::linear_dynamic_fp16').check_not('aten::linear').check_not('aten::dequantize').check_not('aten::quantize').run(m.graph)

class TestQuantizeDynamicJitOps(QuantizationTestCase):
    """Test graph mode post training dynamic quantization works
    for individual ops end to end.
    """

    @override_qengines
    def test_linear(self):
        if False:
            while True:
                i = 10

        class FunctionalLinear(torch.nn.Module):

            def __init__(self, weight, bias):
                if False:
                    return 10
                super().__init__()
                self.weight = weight
                self.bias = bias

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return F.linear(x, self.weight, self.bias)
        x = torch.rand(5, 5)
        for tracing in [True, False]:
            model = self.checkGraphModeOp(torch.nn.Linear(5, 5), x, 'quantized::linear_dynamic', tracing=tracing, dynamic=True)
        weight = torch.rand(5, 5)
        b = torch.rand(5)
        for (tracing, has_bias) in itertools.product([True, False], [True, False]):
            bias = b if has_bias else None
            model = self.checkGraphModeOp(FunctionalLinear(weight, bias), x, 'quantized::linear_dynamic', tracing=tracing, dynamic=True)

    @skipIfNoFBGEMM
    def test_embedding_bag(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self, weights):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.embedding1 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, sparse=True, _weight=weights, mode='sum')
                self.embedding2 = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, sparse=True, _weight=weights, mode='sum')

            def forward(self, indices1, offsets1, indices2, offsets2):
                if False:
                    while True:
                        i = 10
                e1 = self.embedding1(indices1, offsets1)
                e2 = self.embedding2(indices2, offsets2)
                return (e1, e2)
        weights = torch.randn(10, 12, dtype=torch.float32)
        module = M(weights)
        indices = torch.tensor([9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8, 3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3])
        offsets = torch.tensor([0, 19, 20, 28, 28, 32])
        dummy_inputs = (indices, offsets, indices, offsets)
        for trace in [True, False]:
            if trace:
                m = torch.jit.trace(module, dummy_inputs)
            else:
                m = torch.jit.script(module)
            int4_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, custom_op_name='embedding_bag_4bit'), weight=PlaceholderObserver.with_args(custom_op_name='embedding_bag_4bit'))
            int8_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, custom_op_name='embedding_bag_byte'), weight=PlaceholderObserver.with_args(custom_op_name='embedding_bag_byte'))
            m = prepare_jit(m, {'embedding1': int4_qconfig, 'embedding2': int8_qconfig})
            m = convert_jit(m)
            FileCheck().check('quantized::embedding_bag_4bit_rowwise_offsets').check('quantized::embedding_bag_byte_rowwise_offsets').run(m.graph)
            m(*dummy_inputs)

    @skipIfNoFBGEMM
    def test_embedding_bag_padding_idx_error(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self, weights):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.embedding = torch.nn.EmbeddingBag(num_embeddings=10, embedding_dim=12, include_last_offset=True, sparse=True, _weight=weights, mode='sum', padding_idx=0)

            def forward(self, indices, offsets):
                if False:
                    for i in range(10):
                        print('nop')
                e = self.embedding(indices, offsets)
                return e
        weights = torch.randn(10, 12, dtype=torch.float32)
        module = M(weights)
        indices = torch.tensor([0, 1, 2, 3, 4])
        offsets = torch.tensor([0, 2, 5])
        dummy_inputs = (indices, offsets)
        int4_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, custom_op_name='embedding_bag_4bit'), weight=PlaceholderObserver.with_args(custom_op_name='embedding_bag_4bit'))
        int8_qconfig = QConfig(activation=PlaceholderObserver.with_args(dtype=torch.float, custom_op_name='embedding_bag_byte'), weight=PlaceholderObserver.with_args(custom_op_name='embedding_bag_byte'))
        error_msg = 'Expected aten::embedding_bag padding_idx input to be None'
        for (trace, qconfig) in itertools.product([True, False], [int4_qconfig, int8_qconfig]):
            if trace:
                m = torch.jit.trace(module, dummy_inputs)
            else:
                m = torch.jit.script(module)
            m = prepare_jit(m, {'embedding': qconfig})
            with self.assertRaisesRegex(RuntimeError, error_msg):
                m = convert_jit(m)

class TestQuantizeJit(QuantizationTestCase):

    @override_qengines
    def test_single_linear(self):
        if False:
            return 10
        'Compare the result of quantizing single linear layer in\n        eager mode and graph mode\n        '
        annotated_linear_model = AnnotatedSingleLayerLinearModel(torch.backends.quantized.engine).eval()
        linear_model = SingleLayerLinearModel().eval()
        linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
        linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
        model_eager = quantize(annotated_linear_model, test_only_eval_fn, [self.calib_data])
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}
        model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
        model_script = torch.jit.script(linear_model)
        result_eager = model_eager(self.calib_data[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(model_under_test, qconfig_dict, test_only_eval_fn, [self.calib_data], inplace=False)
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @skipIfNoFBGEMM
    def test_observer_with_ignored_function(self):
        if False:
            print('Hello World!')
        'Test observers with ignored function and make sure it works in\n        graph mode\n        '
        annotated_linear_model = AnnotatedSingleLayerLinearModel('fbgemm').eval()
        for qconfig in [QConfig(activation=default_observer, weight=default_weight_observer), QConfig(activation=default_histogram_observer, weight=default_weight_observer), QConfig(activation=default_observer, weight=default_per_channel_weight_observer)]:
            annotated_linear_model.qconfig = qconfig
            linear_model = SingleLayerLinearModel().eval()
            linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
            linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
            model_eager = quantize(annotated_linear_model, test_only_eval_fn, [self.calib_data])
            qconfig_dict = {'': qconfig}
            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            model_script = torch.jit.script(linear_model)
            result_eager = model_eager(self.calib_data[0][0])
            for model_under_test in [model_traced, model_script]:
                model_quantized = quantize_jit(model_under_test, qconfig_dict, test_only_eval_fn, [self.calib_data], inplace=False)
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_conv(self):
        if False:
            i = 10
            return i + 15
        'Compare the result of quantizing conv layer in\n        eager mode and graph mode\n        '
        annotated_conv_model = AnnotatedConvModel(torch.backends.quantized.engine).eval()
        conv_model = ConvModel().eval()
        conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())
        model_eager = quantize(annotated_conv_model, test_only_eval_fn, [self.img_data_2d])
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}
        model_traced = torch.jit.trace(conv_model, self.img_data_2d[0][0])
        model_script = torch.jit.script(conv_model)
        result_eager = model_eager(self.img_data_2d[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(model_under_test, qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=False)
            self.assertEqual(model_quantized(self.img_data_2d[0][0]), result_eager)

    @override_qengines
    def test_conv_transpose(self):
        if False:
            while True:
                i = 10
        'Compare the result of quantizing conv_transpose layer in\n        eager mode and graph mode\n        '
        if not qengine_is_qnnpack():
            return
        annotated_conv_model = AnnotatedConvTransposeModel(torch.backends.quantized.engine).eval()
        conv_model = ConvTransposeModel().eval()
        conv_model.conv.weight = torch.nn.Parameter(annotated_conv_model.conv.weight.detach())
        model_eager = quantize(annotated_conv_model, test_only_eval_fn, [self.img_data_2d])
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine)}
        model_traced = torch.jit.trace(conv_model, self.img_data_2d[0][0])
        model_script = torch.jit.script(conv_model)
        result_eager = model_eager(self.img_data_2d[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(model_under_test, qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=False)
            self.assertEqual(model_quantized(self.img_data_2d[0][0]), result_eager)

    @override_qengines
    def test_conv_bn(self):
        if False:
            i = 10
            return i + 15
        'Compare the result of quantizing conv + bn layer in\n        eager mode and graph mode\n        '
        conv_model = AnnotatedConvBnModel().eval()
        conv_model_to_script = ConvBnModel().eval()
        conv_model_to_script.conv.weight = torch.nn.Parameter(conv_model.conv.weight.detach())
        fuse_modules(conv_model, ['conv', 'bn'], inplace=True)
        model_eager = quantize(conv_model, test_only_eval_fn, [self.img_data_2d])
        qconfig_dict = {'': default_qconfig}
        model_script = quantize_jit(torch.jit.script(conv_model_to_script), qconfig_dict, test_only_eval_fn, [self.img_data_2d], inplace=False)
        result_eager = model_eager(self.img_data_2d[0][0])
        result_script = model_script(self.img_data_2d[0][0])
        self.assertEqual(result_eager, result_script)

    @override_qengines
    def test_nested(self):
        if False:
            return 10
        eager_model = AnnotatedNestedModel(torch.backends.quantized.engine).eval()
        script_model = NestedModel().eval()
        script_model.sub1.fc.weight = torch.nn.Parameter(eager_model.sub1.fc.weight.detach())
        script_model.sub1.fc.bias = torch.nn.Parameter(eager_model.sub1.fc.bias.detach())
        script_model.sub2.fc1.weight = torch.nn.Parameter(eager_model.sub2.fc1.module.weight.detach())
        script_model.sub2.fc1.bias = torch.nn.Parameter(eager_model.sub2.fc1.module.bias.detach())
        script_model.sub2.fc2.weight = torch.nn.Parameter(eager_model.sub2.fc2.weight.detach())
        script_model.sub2.fc2.bias = torch.nn.Parameter(eager_model.sub2.fc2.bias.detach())
        script_model.fc3.weight = torch.nn.Parameter(eager_model.fc3.module.weight.detach())
        script_model.fc3.bias = torch.nn.Parameter(eager_model.fc3.module.bias.detach())
        model_eager = quantize(eager_model, test_only_eval_fn, [self.calib_data])
        qconfig_dict = {'sub2.fc1': default_per_channel_qconfig if qengine_is_fbgemm() else default_qconfig, 'fc3': default_qconfig}
        model_traced = torch.jit.trace(script_model, self.calib_data[0][0])
        model_script = torch.jit.script(script_model)
        result_eager = model_eager(self.calib_data[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(model_under_test, qconfig_dict, test_only_eval_fn, [self.calib_data], inplace=False)
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_skip_quant(self):
        if False:
            i = 10
            return i + 15
        'Test None qconfig'
        eager_model = AnnotatedSkipQuantModel(torch.backends.quantized.engine).eval()
        script_model = SkipQuantModel().eval()
        script_model.sub.fc1.weight = torch.nn.Parameter(eager_model.sub.module.fc1.weight.detach())
        script_model.sub.fc1.bias = torch.nn.Parameter(eager_model.sub.module.fc1.bias.detach())
        script_model.sub.fc2.weight = torch.nn.Parameter(eager_model.sub.module.fc2.weight.detach())
        script_model.sub.fc2.bias = torch.nn.Parameter(eager_model.sub.module.fc2.bias.detach())
        script_model.fc.weight = torch.nn.Parameter(eager_model.fc.weight.detach())
        script_model.fc.bias = torch.nn.Parameter(eager_model.fc.bias.detach())
        eager_model.fuse_modules()
        model_eager = quantize(eager_model, test_only_eval_fn, [self.calib_data])
        qconfig_dict = {'': get_default_qconfig(torch.backends.quantized.engine), 'fc': None}
        model_traced = torch.jit.trace(script_model, self.calib_data[0][0])
        model_script = torch.jit.script(script_model)
        result_eager = model_eager(self.calib_data[0][0])
        for model_under_test in [model_traced, model_script]:
            model_quantized = quantize_jit(model_under_test, qconfig_dict, test_only_eval_fn, [self.calib_data], inplace=False)
            self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)

    @override_qengines
    def test_single_linear_dynamic(self):
        if False:
            i = 10
            return i + 15
        'Compare the result of dynamic quantization of single linear layer in\n        eager mode and graph mode.\n        '
        if qengine_is_qnnpack():
            annotated_linear_model = AnnotatedSingleLayerLinearModel('qnnpack').eval()
            linear_model = SingleLayerLinearModel().eval()
            linear_model.fc1.weight = torch.nn.Parameter(annotated_linear_model.fc1.module.weight.detach())
            linear_model.fc1.bias = torch.nn.Parameter(annotated_linear_model.fc1.module.bias.detach())
            qconfig_dict = {'': default_dynamic_qconfig}
            model_eager = quantize_dynamic(annotated_linear_model, qconfig_dict)
            model_traced = torch.jit.trace(linear_model, self.calib_data[0][0])
            model_script = torch.jit.script(linear_model)
            result_eager = model_eager(self.calib_data[0][0])
            for model_under_test in [model_traced, model_script]:
                model_quantized = quantize_dynamic_jit(model_under_test, qconfig_dict)
                self.assertEqual(model_quantized(self.calib_data[0][0]), result_eager)
                model_fake_quantized = quantize_dynamic_jit(model_under_test, qconfig_dict, debug=True)
                self.assertEqual(model_fake_quantized(self.calib_data[0][0]), result_eager)

    @skipIfNoFBGEMM
    def test_linear_dynamic_fp16(self):
        if False:
            while True:
                i = 10
        linear_model = SingleLayerLinearModel().eval()
        x = torch.ones(5, 5) * 65532
        linear_model.fc1.weight = torch.nn.Parameter(x)
        import warnings
        model_eager = quantize_dynamic(linear_model, dtype=torch.float16)
        result_eager = model_eager(self.calib_data[0][0])
        for trace in [True]:
            with warnings.catch_warnings(record=True) as w:
                quantized_model = self.checkGraphModeOp(linear_model, self.calib_data[0][0], 'quantized::linear_dynamic_fp16', tracing=trace, dynamic=True, qconfig=float16_dynamic_qconfig)
            self.assertEqual(quantized_model(self.calib_data[0][0]), result_eager)