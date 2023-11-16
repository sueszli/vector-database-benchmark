import contextlib
import importlib
import math
import os
import sys
import unittest
from functools import partial
import torch
import torch._custom_ops as custom_ops
import torch.library
from torch._dynamo.testing import make_test_cls_with_patches
from torch.testing._internal.common_device_type import instantiate_device_type_tests, onlyCPU, onlyCUDA
from torch.testing._internal.common_utils import IS_CI, IS_WINDOWS, TEST_WITH_ASAN, TEST_WITH_ROCM, TestCase
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA
if IS_WINDOWS and IS_CI:
    sys.stderr.write('Windows CI does not have necessary dependencies for test_torchinductor_dynamic_shapes yet\n')
    if __name__ == '__main__':
        sys.exit(0)
    raise unittest.SkipTest('requires sympy/functorch/filelock')
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from inductor.test_torchinductor import check_model, check_model_cuda, CommonTemplate, copy_tests, TestFailure
importlib.import_module('filelock')
test_failures = {'test_kwargs_dynamic_shapes': TestFailure(('cpu',)), 'test_AllenaiLongformerBase_repro_dynamic_shapes': TestFailure(('cpu', 'cuda'))}
if TEST_WITH_ROCM:
    test_failures['test_convolution1_dynamic_shapes'] = TestFailure(('cpu', 'cuda'), is_skip=True)
    test_failures['test_convolution3_dynamic_shapes'] = TestFailure('cuda', is_skip=True)
    test_failures['test_expanded_reduction_dynamic_shapes'] = TestFailure('cuda', is_skip=True)

def make_dynamic_cls(cls, xfail_prop='_expected_failure_dynamic'):
    if False:
        return 10
    return make_test_cls_with_patches(cls, 'DynamicShapes', '_dynamic_shapes', (torch._dynamo.config, 'assume_static_by_default', False), xfail_prop=xfail_prop)
DynamicShapesCommonTemplate = make_dynamic_cls(CommonTemplate)
if HAS_CPU:

    class DynamicShapesCpuTests(TestCase):
        common = check_model
        device = 'cpu'
    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCpuTests, 'cpu', test_failures)
if HAS_CUDA and (not TEST_WITH_ASAN):

    class DynamicShapesCudaTests(TestCase):
        common = check_model_cuda
        device = 'cuda'
    copy_tests(DynamicShapesCommonTemplate, DynamicShapesCudaTests, 'cuda', test_failures)

class TestInductorDynamic(TestCase):
    compile_fn = partial(torch.compile, dynamic=True)

    def setUp(self):
        if False:
            while True:
                i = 10
        if self.device_type == 'cuda' and (not HAS_CUDA):
            self.skipTest('Triton not available')
        torch._dynamo.reset()
        super(TestCase, self).setUp()
        self._stack = contextlib.ExitStack()
        self._stack.enter_context(torch._inductor.config.patch({'debug': False, 'cpp.min_chunk_size': 1, 'triton.autotune_pointwise': False, 'implicit_fallbacks': False}))

    def tearDown(self):
        if False:
            return 10
        self._stack.close()
        super(TestCase, self).tearDown()
        torch._dynamo.reset()

    def test_arange_dynamic(self, device):
        if False:
            return 10

        def fn(a):
            if False:
                print('Hello World!')
            batch_size = a.numel()
            max_len = a.max()
            return ~torch.arange(0, max_len, device=a.device).type_as(a).repeat(batch_size, 1).lt(a.unsqueeze(1))
        a = torch.randint(10, 30, (10,), device=device)
        a[0] = 29
        opt = self.compile_fn(fn)
        res = opt(a)
        ref = fn(a)
        self.assertEqual(res, ref)

    def test_shape_as_constant_reciprocal_float_exp(self, device):
        if False:
            i = 10
            return i + 15

        def fn(x, a):
            if False:
                print('Hello World!')
            return (x, -1 / a ** 1.0)
        x = torch.rand(10, 20, device=device)
        opt = self.compile_fn(fn)
        res = opt(x, x.size(0))
        ref = fn(x, x.size(0))
        self.assertEqual(res, ref)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_bool_mask_nobreak(self, device):
        if False:
            return 10

        def f(x, b):
            if False:
                i = 10
                return i + 15
            return (x[b] * 2).sum()
        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    def test_adaptive_max_pool3d_with_indices(self, device):
        if False:
            i = 10
            return i + 15
        x = 5
        y = torch.rand([9, 10, 9, 8, 6], dtype=torch.float32, device=device)

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.adaptive_max_pool3d_with_indices(output_size=x, input=y, return_indices=True)
        opt_f = self.compile_fn(fn)
        r = fn(x, y)
        opt_r = opt_f(x, y)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True)
    def test_nonzero_size_factory_nobreak(self, device):
        if False:
            for i in range(10):
                print('nop')

        def f(x, b):
            if False:
                while True:
                    i = 10
            y = torch.nonzero(b)
            return x.new_zeros(y.size(0))
        opt_f = torch.compile(f, fullgraph=True)
        x = torch.randn(5, device=device)
        b = torch.tensor([True, True, False, False, True], device=device)
        r = f(x, b)
        opt_r = opt_f(x, b)
        self.assertEqual(r, opt_r)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_nobreak(self, device):
        if False:
            return 10

        @torch.compile(fullgraph=True)
        def f(x):
            if False:
                return 10
            y = x.item()
            return torch.empty(y)
        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_zeros_nobreak(self, device):
        if False:
            print('Hello World!')

        @torch.compile(fullgraph=True)
        def f(x):
            if False:
                i = 10
                return i + 15
            y = x.item()
            torch.empty(y)
            return x.new_zeros(y)
        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_return(self, device):
        if False:
            print('Hello World!')

        @torch.compile(fullgraph=True)
        def f(x):
            if False:
                return 10
            y = x.item()
            z = x.item()
            return y + z
        f(torch.tensor([3], device=device))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_item_to_inputs_kernel_nobreak(self, device):
        if False:
            print('Hello World!')
        lib = torch.library.Library('test', 'DEF')
        try:

            @custom_ops.custom_op('test::foo')
            def foo(x: torch.Tensor, y: int) -> torch.Tensor:
                if False:
                    return 10
                raise NotImplementedError()

            @custom_ops.impl('test::foo')
            def foo_impl(x: torch.Tensor, y: int) -> torch.Tensor:
                if False:
                    for i in range(10):
                        print('nop')
                return x.clone()

            @torch.library.impl_abstract('test::foo', lib=lib)
            def foo_meta(x: torch.Tensor, y: int) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                return x.clone()

            @torch.compile(fullgraph=True)
            def f(x, r):
                if False:
                    return 10
                y = x.item()
                return torch.ops.test.foo(r, y)
            f(torch.tensor([3], device=device), torch.randn(10, device=device))
        finally:
            custom_ops._destroy('test::foo')

    @torch._dynamo.config.patch(capture_scalar_outputs=True, capture_dynamic_output_shape_ops=True)
    @torch._inductor.config.patch(implicit_fallbacks=True)
    def test_dynamic_stride_nobreak(self, device):
        if False:
            return 10
        lib = torch.library.Library('test', 'DEF')
        try:

            @custom_ops.custom_op('test::foo')
            def foo(x: torch.Tensor) -> torch.Tensor:
                if False:
                    print('Hello World!')
                raise NotImplementedError()

            @custom_ops.impl('test::foo')
            def foo_impl(x: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                stride = x.item()
                return torch.empty_strided((1,), (stride,), device=x.device)

            @torch.library.impl_abstract('test::foo', lib=lib)
            def foo_meta(x: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                ctx = torch.library.get_ctx()
                stride = ctx.new_dynamic_size()
                return torch.empty_strided((1,), (stride,), device=x.device)

            @torch.compile(fullgraph=True)
            def f(x):
                if False:
                    while True:
                        i = 10
                r = torch.ops.test.foo(x)
                y = r.stride(0)
                return torch.empty(y, device=x.device)
            f(torch.tensor([3], device=device))
        finally:
            custom_ops._destroy('test::foo')

    @torch._inductor.config.patch(disable_cpp_codegen=True)
    def test_floor(self):
        if False:
            for i in range(10):
                print('nop')

        def fn(x):
            if False:
                while True:
                    i = 10
            n = x.size(-1)
            y = x + int(n * 0.2) + 1
            return y
        opt = self.compile_fn(fn)
        x0 = torch.rand(5)
        ref0 = fn(x0)
        res0 = opt(x0)
        self.assertEqual(ref0, res0)
        x1 = torch.rand(8)
        ref1 = fn(x1)
        res1 = opt(x1)
        self.assertEqual(ref1, res1)

    @onlyCUDA
    def test_pad_dynamic(self, device):
        if False:
            while True:
                i = 10

        def get_same_padding(x: int, k: int, s: int, d: int):
            if False:
                while True:
                    i = 10
            return max((math.ceil(x / s) - 1) * s + (k - 1) * d + 1 - x, 0)

        def pad_same(x, k, s, d=(1, 1), value=0):
            if False:
                return 10
            (ih, iw) = x.size()[-2:]
            (pad_h, pad_w) = (get_same_padding(ih, k[0], s[0], d[0]), get_same_padding(iw, k[1], s[1], d[1]))
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2], value=value)
            return x
        x = torch.randn(2, 24, 110, 110, device=device)
        opt = self.compile_fn(pad_same)
        res = opt(x, (5, 5), (2, 2))
        ref = pad_same(x, (5, 5), (2, 2))
        self.assertEqual(res, ref, atol=0, rtol=0)

    def test_slice_scatter(self, device):
        if False:
            i = 10
            return i + 15

        def fn(i):
            if False:
                print('Hello World!')
            s3 = i.size(0)
            x = torch.ones(64, s3, device=device)
            y = torch.ones(64, s3 // 2, device=device)
            return torch.slice_scatter(x, y, 1, s3 // 2, 2 * (s3 // 2))
        a = torch.randn(16, device=device)
        cfn = self.compile_fn(fn)
        expect = fn(a)
        actual = cfn(a)
        self.assertEqual(expect, actual)

    def test_slice_index_changing_sign(self, device):
        if False:
            while True:
                i = 10

        def fn(x, y):
            if False:
                while True:
                    i = 10
            (y0, y1) = y.shape
            return x[:y0 - y1].clone()
        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    def test_sym_stride_lowering(self, device):
        if False:
            i = 10
            return i + 15

        def fn(x):
            if False:
                while True:
                    i = 10
            s0 = (x + 1).stride(0)
            return x * s0
        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        self.assertEqual(fn(a), cfn(a))

    def test_abs(self, device):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            (y0, y1) = y.shape
            return x[:abs(y0 - y1)] * abs(y0 - y1)
        a = torch.randn(32, 32, device=device)
        cfn = self.compile_fn(fn)
        b = torch.randn(16, 2, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)
        b = torch.randn(2, 16, device=device)
        expect = fn(a, b)
        actual = cfn(a, b)
        self.assertEqual(expect, actual)

    @onlyCPU
    def test_arithmetic_constant_folding(self, device):
        if False:
            return 10

        def test(fn):
            if False:
                return 10
            cfn = self.compile_fn(fn)
            expect = fn(3)
            actual = cfn(3)
            self.assertEqual(expect, actual)

        def add(x):
            if False:
                while True:
                    i = 10
            return x + torch.zeros(3)
        test(add)

        def mul(x):
            if False:
                return 10
            return x * torch.ones(3)
        test(mul)

        def div(x):
            if False:
                i = 10
                return i + 15
            return x / torch.ones(3)
        test(div)

    @onlyCPU
    def test_sub_constant_folding(self, device):
        if False:
            for i in range(10):
                print('nop')

        def sub(x):
            if False:
                print('Hello World!')
            return x - torch.zeros(3)
        cfn = self.compile_fn(sub)
        expect = sub(3)
        actual = cfn(3)
        self.assertEqual(expect, actual)

    def test_full(self, device):
        if False:
            return 10

        def fn(a):
            if False:
                while True:
                    i = 10
            return (torch.full((3,), a), torch.full((3,), torch.sym_float(a)))
        cfn = self.compile_fn(fn)
        expect = fn(5)
        actual = cfn(5)
        self.assertEqual(expect, actual)
instantiate_device_type_tests(TestInductorDynamic, globals())
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if (HAS_CPU or HAS_CUDA) and (not TEST_WITH_ASAN):
        run_tests(needs='filelock')