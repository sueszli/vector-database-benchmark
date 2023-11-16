from torch.testing._internal.common_utils import TestCase, TEST_WITH_TORCHDYNAMO, run_tests, skipIfCrossRef, skipIfRocm, skipIfTorchDynamo, parametrize, instantiate_parametrized_tests
import torch
import torch._dynamo
import itertools
import numpy as np
from torch.testing._internal.jit_utils import RUN_CUDA
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode, FakeTensorConverter, DynamicOutputShapeException, UnsupportedOperatorException
from torch.fx.experimental.symbolic_shapes import ShapeEnv, DimDynamic, free_symbols
from torch.testing._internal.custom_op_db import custom_op_db
from torch.testing._internal.common_device_type import ops
from torch.testing._internal.common_device_type import instantiate_device_type_tests, OpDTypes
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_FLASH_ATTENTION
from torch.fx.passes.fake_tensor_prop import FakeTensorProp
from torch._dynamo.testing import rand_strided
from torch.testing import FileCheck
import unittest
import torch._prims as prims
import contextlib
import weakref
import copy
import torch._functorch.config
import torch.testing._internal.optests as optests
from unittest.mock import patch
from torch import distributed as dist
from torch.utils._mode_utils import no_dispatch
from torch.utils._python_dispatch import TorchDispatchMode
import torch.utils._pytree as pytree

class FakeTensorTest(TestCase):

    def checkType(self, t, device_str, size):
        if False:
            while True:
                i = 10
        self.assertTrue(isinstance(t, FakeTensor))
        self.assertEqual(t.device.type, device_str)
        self.assertEqual(list(t.size()), size)

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_cuda_initialized(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            p = torch.randn(4, 2, requires_grad=True, device='cuda')
            x = torch.randn(8, 4, device='cuda')
            y = torch.mm(x, p).square().sum()
            y.backward()

    def test_basic(self):
        if False:
            while True:
                i = 10
        x = torch.empty(2, 2, device='cpu')
        y = torch.empty(4, 2, 2, device='cpu')
        with FakeTensorMode() as mode:
            x = mode.from_tensor(x)
            y = mode.from_tensor(y)
            z = x + y
            self.assertEqual(z.shape, (4, 2, 2))
            self.assertEqual(z.device, torch.device('cpu'))
            self.assertTrue(isinstance(z, FakeTensor))

    def test_basic_forced_memo_only(self):
        if False:
            return 10
        x = torch.empty(2, 2, device='cpu')
        y = torch.empty(4, 2, 2, device='cpu')
        with FakeTensorMode() as mode:
            x_fake = mode.from_tensor(x)
            x2 = mode.from_tensor(x, memoized_only=True)
            self.assertTrue(x2 is not None)
            y = mode.from_tensor(y, memoized_only=True)
            self.assertIs(y, None)

    def test_custom_op_fallback(self):
        if False:
            return 10
        from torch.library import Library, impl
        test_lib = Library('my_test_op', 'DEF')
        test_lib.define('foo(Tensor self) -> Tensor')

        @impl(test_lib, 'foo', 'CPU')
        def foo_impl(self):
            if False:
                i = 10
                return i + 15
            return self.cos()
        x = torch.empty(2, 2, device='cpu')
        with self.assertRaisesRegex(UnsupportedOperatorException, 'my_test_op.foo.default'):
            with FakeTensorMode(allow_fallback_kernels=True) as mode:
                x = mode.from_tensor(x)
                torch.ops.my_test_op.foo(x)

    def test_parameter_instantiation(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.nn.parameter.Parameter(x)
            self.assertTrue(isinstance(y, torch.nn.Parameter))

    @unittest.skipIf(not dist.is_available(), 'requires distributed')
    def test_fsdp_flat_param(self):
        if False:
            i = 10
            return i + 15
        from torch.distributed.fsdp._flat_param import FlatParameter
        with FakeTensorMode() as m:
            data = torch.randn(2, 2)
            param = FlatParameter(data, requires_grad=True)
        self.assertIsInstance(param, FlatParameter)
        self.assertIsInstance(param, torch.nn.Parameter)
        self.assertIsInstance(param, FakeTensor)

    def test_non_parameter_grad(self):
        if False:
            return 10
        mode = FakeTensorMode()
        t = torch.rand([4], requires_grad=True)
        fake_t = mode.from_tensor(t)
        self.assertEqual(fake_t.requires_grad, t.requires_grad)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_index_cuda_with_cpu(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x = torch.rand([2048], device='cuda')
            out = x[torch.zeros([36], dtype=torch.int64)]
            self.checkType(out, 'cuda', [36])

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_shape_take_not_device(self):
        if False:
            return 10
        with FakeTensorMode():
            x = torch.empty(1, device='cpu')
            y = torch.empty(8, 8, device='cuda')
            out = x.resize_as_(y)
            self.assertEqual(out.shape, (8, 8))
            self.assertEqual(out.device.type, 'cpu')
            self.assertTrue(isinstance(out, FakeTensor))

    def test_repr(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            x = torch.empty(2, 2, device='cpu')
            self.assertEqual(repr(x), 'FakeTensor(..., size=(2, 2))')
            x = torch.empty(2, 2, device='meta')
            self.assertEqual(repr(x), "FakeTensor(..., device='meta', size=(2, 2))")

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_zero_dim(self):
        if False:
            print('Hello World!')
        with FakeTensorMode() as mode:
            x = torch.tensor(0.0)
            y = torch.rand([4, 4], device='cuda')
            out = x + y
            self.assertEqual(out.shape, (4, 4))
            self.assertEqual(out.device, y.device)
            self.assertTrue(isinstance(out, FakeTensor))

    def test_nan_to_num(self):
        if False:
            return 10
        with FakeTensorMode():
            for dtype in [torch.float16, torch.float32]:
                x = torch.rand([4], dtype=dtype)
                y = torch.nan_to_num(x, nan=None)
                z = torch.nan_to_num(x, 0.0)
                self.assertEqual(dtype, y.dtype)
                self.assertEqual(dtype, z.dtype)

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_throw(self):
        if False:
            return 10
        x = torch.tensor(0.0)
        with FakeTensorMode() as mode:
            x_conv = mode.from_tensor(x)
            y = torch.rand([4, 4], device='cuda')
            z = torch.rand([4, 4], device='cpu')
            self.assertRaises(Exception, lambda : torch.lerp(x_conv, y, z))

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_type_as(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            x = torch.rand([16, 1], device='cpu')
            y = torch.rand([4, 4], device='cuda')
            out = x.type_as(y)
            self.assertEqual(out.device.type, 'cuda')
            self.assertTrue(isinstance(out, FakeTensor))

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_setitem(self):
        if False:
            while True:
                i = 10
        for device in ['cpu', 'cuda']:
            with FakeTensorMode():
                x = torch.rand([16, 1], device=device)
                x[..., 0] = 0

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_device_inplace_copy(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x = torch.rand([8, 8], device='cpu')
            y = torch.rand([8, 8], device='cuda')
            assert x.copy_(y).device.type == 'cpu'
            assert y.copy_(x).device.type == 'cuda'

    def test_fake_dispatch_keys(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            x = torch.rand([4])
            f = FileCheck().check('CPU').check('ADInplaceOrView').check('AutogradCPU').check('AutocastCPU')
            f.run(torch._C._dispatch_key_set(x))
            with torch.inference_mode():
                x = torch.rand([4])
                y = x + x
                FileCheck().check('CPU').check('AutocastCPU').run(torch._C._dispatch_key_set(y))
                FileCheck().check_not('ADInplaceOrView').check_not('Autograd').run(torch._C._dispatch_key_set(y))

    def test_constructor(self):
        if False:
            return 10
        with FakeTensorMode():
            x = torch.rand([4, 4], device='cpu')
        self.assertTrue(isinstance(x, FakeTensor))
        self.assertTrue(x.device.type == 'cpu')

    def test_mode(self):
        if False:
            return 10
        with FakeTensorMode():
            y = torch.rand([4], device='cpu')
            out = y + y
        self.assertTrue(isinstance(out, FakeTensor))

    def test_full(self):
        if False:
            for i in range(10):
                print('nop')
        with torch._subclasses.CrossRefFakeMode():
            y = torch.full((4, 4), 1)

    def check_function_with_fake(self, fn):
        if False:
            i = 10
            return i + 15
        out = fn()
        with torch._subclasses.FakeTensorMode():
            out_fake = fn()
        for (a, b) in zip(pytree.tree_leaves(out), pytree.tree_leaves(out_fake)):
            if not isinstance(a, torch.Tensor):
                self.assertTrue(not isinstance(b, torch.Tensor))
                continue
            prims.utils.compare_tensor_meta(a, b, check_strides=True)

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_non_kwarg_device(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            x = torch.rand([16, 1], device='cpu')
            y = x.to(torch.device('cpu'))
            self.assertIs(x, y)
            z = x.to(torch.device('cuda'))
            self.assertEqual(z.device.type, 'cuda')

    def test_non_overlapping_stride_zero(self):
        if False:
            while True:
                i = 10

        def foo():
            if False:
                while True:
                    i = 10
            x = torch.empty_strided([1, 3, 427, 640], (0, 1, 1920, 3))
            return x.half()
        self.check_function_with_fake(foo)

    def test_fake_mode_error(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand([4, 4])
        with self.assertRaisesRegex(Exception, 'Please convert all Tensors'):
            with FakeTensorMode():
                y = x[0]

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    def test_fake_grad_copy(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand([4, 4], requires_grad=True)
        x.grad = torch.rand([4, 4])
        mode = FakeTensorMode()
        fake_x = mode.from_tensor(x)
        prims.utils.compare_tensor_meta(fake_x, x)
        prims.utils.compare_tensor_meta(fake_x.grad, x.grad)
        self.assertTrue(isinstance(fake_x.grad, FakeTensor))

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_index_put_error(self):
        if False:
            while True:
                i = 10
        mode = FakeTensorMode()
        for context in [contextlib.nullcontext, lambda : mode]:
            with context():
                y = torch.randn(2, 2, 3)
                x = torch.randn(2, 2, 3).to('cuda')
                with self.assertRaises(RuntimeError):
                    x[[1, 1]] = y
                with self.assertRaises(RuntimeError):
                    torch.ops.aten.index_put(x, torch.tensor([1, 1], device='cuda'), y)
                torch.ops.aten.index_put(x, torch.tensor([1, 1], device='cuda'), torch.tensor(5.0))
                torch.ops.aten.index_put_(x, torch.tensor([1, 1], device='cuda'), torch.tensor(5.0))

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_like_constructor(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            x = torch.rand([4, 4])
            y = torch.ones_like(x)
            self.assertTrue(isinstance(y, FakeTensor))
            self.assertEqual(y.device.type, 'cpu')
            z = torch.ones_like(x, device='cuda')
            self.assertTrue(isinstance(z, FakeTensor))
            self.assertEqual(z.device.type, 'cuda')

    def test_binary_op_type_promotion(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            x = torch.empty([2, 2], dtype=torch.float)
            y = torch.empty([2, 2], dtype=torch.int64)
            out = x / y
            self.assertEqual(out.dtype, torch.float)
            self.assertEqual(out.device.type, 'cpu')

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    def test_from_numpy(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            x = torch.tensor(np.zeros([4, 4]))
            self.checkType(x, 'cpu', [4, 4])

    def test_randperm(self):
        if False:
            print('Hello World!')
        x = torch.randperm(10)
        y = torch.randperm(5, device='cpu')
        with FakeTensorMode():
            x1 = torch.randperm(10)
            prims.utils.compare_tensor_meta(x, x1)
            y1 = torch.randperm(5, device='cpu')
            prims.utils.compare_tensor_meta(y, y1)

    def test_print_in_fake_mode(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.zeros(2)
        with FakeTensorMode():
            out = str(x)
        assert 'FakeTensor' not in out

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_upsample_bilinear_small_channels(self):
        if False:
            for i in range(10):
                print('nop')
        out = []
        mode = FakeTensorMode()
        for (i, context) in enumerate([contextlib.nullcontext, lambda : mode]):
            with context():
                arg0_1 = torch.empty_strided((3, 427, 640), (1, 1920, 3), dtype=torch.float32, device='cuda')
                unsqueeze = torch.ops.aten.unsqueeze.default(arg0_1, 0)
                out.append(torch.ops.aten.upsample_bilinear2d.default(unsqueeze, [800, 1199], False))
        self.assertTrue(out[1].is_contiguous())
        self.checkMetaProps(out[0], out[1])

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_cpu_fallback(self):
        if False:
            print('Hello World!')
        with FakeTensorMode(allow_fallback_kernels=False):
            filters = torch.randn(8, 4, 3, 3).cuda()
            inputs = torch.randn(1, 4, 5, 5).cuda()
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, 'cuda')
            self.assertEqual(list(out.size()), [1, 8, 5, 5])
        with FakeTensorMode(allow_fallback_kernels=True):
            filters = torch.randn(8, 20, 3, 3).cuda()
            inputs = torch.randn(1, 7, 10, 5).cuda()
            with self.assertRaises(RuntimeError):
                torch.nn.functional.conv2d(inputs, filters, padding=1)
        with FakeTensorMode(allow_fallback_kernels=True):
            filters = torch.randn(8, 4, 3, 3).cuda()
            inputs = torch.randn(1, 4, 5, 5).cuda()
            out = torch.nn.functional.conv2d(inputs, filters, padding=1)
            self.assertEqual(out.device.type, 'cuda')
            self.assertEqual(list(out.size()), [1, 8, 5, 5])

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_out_multi_device(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x = torch.rand([4])
            y = torch.rand([4], device='cuda')
            with self.assertRaisesRegex(Exception, 'found two different devices'):
                torch.sin(x, out=y)
            with self.assertRaisesRegex(Exception, 'found two different devices'):
                x.add_(y)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_normalize_device(self):
        if False:
            return 10
        with FakeTensorMode():
            x = torch.empty(1, device='cuda')
            y = torch.empty(1, device=f'cuda:{torch.cuda.current_device()}')
            out = x + y
        self.checkType(out, 'cuda', [1])

    def test_recursive_invocation(self):
        if False:
            i = 10
            return i + 15
        mode = FakeTensorMode()
        with mode:
            x = torch.tensor(2)
            mode.in_kernel_invocation = True
            y = x + x
            self.assertTrue(mode.in_kernel_invocation)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @skipIfRocm
    @parametrize('allow_fallback_kernels', [False, True], lambda a: 'with_fallback' if a else 'without_fallback')
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_cudnn_rnn(self, allow_fallback_kernels):
        if False:
            i = 10
            return i + 15

        def fn(a0, b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15, a3, a4, a5):
            if False:
                while True:
                    i = 10
            a1 = [b0, b1, b2, b3, b4, b5, b6, b7, b8, b9, b10, b11, b12, b13, b14, b15]
            return torch.ops.aten._cudnn_rnn(a0, a1, 4, a3, a4, a5, 2, 2048, 0, 2, False, 0.0, False, True, [], None)
        mode = FakeTensorMode(allow_fallback_kernels=allow_fallback_kernels)
        for (i, context) in enumerate([contextlib.nullcontext, lambda : mode]):
            with context():
                inps1 = [torch.randn([92, 8, 2048]).cuda(), torch.randn([8192, 2048]).cuda(), torch.randn([8192, 2048]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192, 2048]).cuda(), torch.randn([8192, 2048]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192, 4096]).cuda(), torch.randn([8192, 2048]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192, 4096]).cuda(), torch.randn([8192, 2048]).cuda(), torch.randn([8192]).cuda(), torch.randn([8192]).cuda(), torch.randn([167837696]).cuda(), torch.randn([4, 8, 2048]).cuda(), torch.randn([4, 8, 2048]).cuda()]
                inps2 = inps1
                inps2[len(inps2) - 1] = None
                for inps in [inps1, inps2]:
                    out = fn(*inps)
                    self.assertIs(out[4], inps[-3])
                    for ten in out:
                        if i == 1:
                            self.assertTrue(isinstance(ten, FakeTensor))
                        self.assertEqual(ten.device.type, 'cuda')

    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_cuda_lstm(self):
        if False:
            i = 10
            return i + 15
        with torch.backends.cudnn.flags(enabled=False):
            fake_tensor_mode = FakeTensorMode(allow_fallback_kernels=False)
            with fake_tensor_mode:
                N = 5
                L = 4
                H_in = 2
                hidden_size = 3
                proj_size = 2
                num_layers = 2
                bidir = False
                D = 2 if bidir else 1
                H_out = proj_size if proj_size > 0 else hidden_size
                lstm = torch.nn.LSTM(input_size=H_in, hidden_size=hidden_size, num_layers=num_layers, proj_size=proj_size, batch_first=False, bias=True, bidirectional=bidir, device='cuda')
                h_0 = torch.randn((num_layers * D, N, H_out), device='cuda')
                c_0 = torch.randn((num_layers * D, N, hidden_size), device='cuda')
                inp = torch.randn((L, N, H_in), device='cuda')
                (output, (h_n, c_n)) = lstm(inp, (h_0, c_0))
                output.sum().backward()
                self.assertEqual(output.shape, (L, N, D * H_out))
                self.assertEqual(h_n.shape, (D * num_layers, N, H_out))
                self.assertEqual(c_n.shape, (D * num_layers, N, hidden_size))

    def test_data_dependent_operator(self):
        if False:
            print('Hello World!')
        with FakeTensorMode(allow_fallback_kernels=False):
            x = torch.rand([10, 10])
            self.assertRaises(DynamicOutputShapeException, lambda : torch.nonzero(x))

    def test_tolist(self):
        if False:
            i = 10
            return i + 15
        shape_env = ShapeEnv()
        with FakeTensorMode(allow_fallback_kernels=False, shape_env=shape_env):
            x = torch.rand([10])
            x.tolist()

    def test_same_shape_env_preserved(self):
        if False:
            while True:
                i = 10
        shape_env = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=shape_env)
        t1 = mode1.from_tensor(torch.randn(10), dynamic_dims=[DimDynamic.DYNAMIC])
        mode2 = FakeTensorMode(shape_env=shape_env)
        t2 = mode2.from_tensor(t1)
        self.assertIsNot(t2, t1)
        self.assertIs(t1.fake_mode, mode1)
        self.assertIs(t2.fake_mode, mode2)
        self.assertIs(t2.size(0).node.shape_env, t1.size(0).node.shape_env)
        self.assertEqual(str(t2.size(0)), str(t1.size(0)))

    def test_jagged_fake_to_fake_preserved(self):
        if False:
            i = 10
            return i + 15
        from torch.nested._internal.nested_tensor import jagged_from_list
        (S0, S1, S2) = (3, 4, 5)
        D = 4
        a = torch.randn(S0, D, requires_grad=True, dtype=torch.float64)
        b = torch.randn(S1, D, requires_grad=True, dtype=torch.float64)
        c = torch.randn(S2, D, requires_grad=True, dtype=torch.float64)
        offsets = None
        (jt, _) = jagged_from_list([a, b, c], offsets)
        shape_env = ShapeEnv()
        mode1 = FakeTensorMode(shape_env=shape_env)
        t1 = mode1.from_tensor(jt)
        mode2 = FakeTensorMode(shape_env=shape_env)
        t2 = mode2.from_tensor(t1)
        self.assertTrue(free_symbols(t1.size()))
        self.assertIsNot(t2, t1)
        self.assertIs(t1.offsets().fake_mode, mode1)
        self.assertIs(t2.offsets().fake_mode, mode2)
        self.assertIs(t2.size(1).node.shape_env, t1.size(1).node.shape_env)
        self.assertEqual(str(t2.size(1)), str(t1.size(1)))

    def checkMetaProps(self, t1, t2):
        if False:
            for i in range(10):
                print('nop')
        prims.utils.compare_tensor_meta(t1, t2, check_strides=True)

    @skipIfCrossRef
    def test_deepcopy(self):
        if False:
            return 10
        with FakeTensorMode() as mode:
            pass
        mod = torch.nn.BatchNorm2d(10)
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)

        def check_copy(mod, mod_copied):
            if False:
                i = 10
                return i + 15
            for (name, param) in itertools.chain(mod.named_parameters(), mod.named_buffers()):
                param_copied = getattr(mod_copied, name)
                self.checkMetaProps(param, param_copied)
                self.assertTrue(isinstance(param_copied, FakeTensor))
                self.assertEqual(isinstance(param, torch.nn.Parameter), isinstance(param_copied, torch.nn.Parameter))
                self.assertEqual(param.requires_grad, param_copied.requires_grad)
        check_copy(mod, mod_copied)

        class ModuleNew(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = torch.rand([10, 2])
                self.b = self.a
                self.c = self.a[0]
        mod = ModuleNew()
        with torch._subclasses.fake_tensor.FakeCopyMode(mode):
            mod_copied = copy.deepcopy(mod)
        self.assertIs(mod_copied.a, mod_copied.b)
        self.assertEqual(mod_copied.b.storage()._cdata, mod_copied.a.storage()._cdata)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_new(self):
        if False:
            print('Hello World!')
        with FakeTensorMode():
            a = torch.rand([16, 1])
            self.checkType(a.new(10, 10), 'cpu', [10, 10])
            self.checkType(a.new([1, 2, 3, 4]), 'cpu', [4])
            b = torch.rand([4, 4], device='cuda')
            self.checkType(b.new(device='cuda'), 'cuda', [0])
            self.checkType(a.new(torch.rand([1])), 'cpu', [1])

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    def test_scalar_inputs(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            self.checkType(torch.div(3, 2), 'cpu', [])
            ten = torch.zeros(2, dtype=torch.int32) * 2.0
            self.assertEqual(ten.dtype, torch.float)
            self.checkType(ten, 'cpu', [2])

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    def test_allow_meta(self):
        if False:
            return 10

        def run_meta():
            if False:
                print('Hello World!')
            with FakeTensorMode():
                x = torch.rand([4], device='meta')
                return x + x
        self.checkType(run_meta(), 'meta', [4])
        with patch.object(torch._functorch.config, 'fake_tensor_allow_meta', False):
            self.assertRaises(Exception, run_meta)

    def test_embedding_bag_meta(self):
        if False:
            for i in range(10):
                print('nop')

        def f():
            if False:
                i = 10
                return i + 15
            embedding = torch.nn.EmbeddingBag(10, 3, mode='sum', device='meta')
            input = torch.tensor([1, 2, 4, 5, 4, 3, 2, 9], dtype=torch.long)
            offsets = torch.tensor([0, 4], dtype=torch.long)
            return embedding(input, offsets)
        real_out = f()
        with FakeTensorMode():
            fake_out = f()
        for (r, f) in zip(real_out, fake_out):
            self.assertEqual(r.size(), f.size())
            self.assertEqual(r.device, f.device)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    def test_mixed_real_and_fake_inputs(self):
        if False:
            for i in range(10):
                print('nop')

        class _TestPattern(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.conv = torch.nn.Conv2d(1, 1, 1)
                self.bn = torch.nn.BatchNorm2d(1)

            def forward(self, input):
                if False:
                    print('Hello World!')
                running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
                scale_factor = self.bn.weight / running_std
                weight_shape = [1] * len(self.conv.weight.shape)
                weight_shape[0] = -1
                bias_shape = [1] * len(self.conv.weight.shape)
                bias_shape[1] = -1
                scaled_weight = self.conv.weight * scale_factor.reshape(weight_shape)
                zero_bias = torch.zeros_like(self.conv.bias, dtype=input.dtype)
                conv = self.conv._conv_forward(input, scaled_weight, zero_bias)
                conv_orig = conv / scale_factor.reshape(bias_shape)
                conv_orig = conv_orig + self.conv.bias.reshape(bias_shape)
                conv = self.bn(conv_orig)
                return conv
        example_inputs = (torch.randn(1, 1, 3, 3),)
        mod = _TestPattern()
        with FakeTensorMode(allow_non_fake_inputs=True):
            out = mod(torch.randn(1, 1, 3, 3))
        self.checkType(out, 'cpu', (1, 1, 3, 3))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_aten_copy_multi_device(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x1 = torch.rand(4, device='cpu')
            x2 = torch.rand(4, device='cuda')
            copy1 = torch.ops.aten.copy.default(x1, x2)
            copy2 = torch.ops.aten.copy.default(x2, x1)
            out = torch.empty(4, device='cpu')
            torch.ops.aten.copy.out(x1, x2, out=out)
        self.checkType(copy1, 'cpu', (4,))
        self.checkType(copy2, 'cuda', (4,))
        self.checkType(out, 'cpu', (4,))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_aten_index_multi_device(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x1 = torch.rand(4, 4, device='cpu')
            x2 = torch.rand(4, 4, device='cuda')
            i1 = torch.tensor([0, 1], device='cuda')
            i2 = torch.tensor([0, 1], device='cpu')
            r1 = torch.ops.aten.index(x1, i1)
            r2 = torch.ops.aten.index(x2, i2)
            y1 = torch.rand(4, device='cpu')
            y2 = torch.rand(4, device='cuda')
            j1 = torch.tensor([2], device='cuda')
            j2 = torch.tensor([2], device='cpu')
            r3 = torch.ops.aten.index_put.default(x1, j1, y1)
            r4 = torch.ops.aten.index_put.default(x2, j2, y2)
        self.checkType(r1, 'cpu', ())
        self.checkType(r2, 'cuda', ())
        self.checkType(r3, 'cpu', (4, 4))
        self.checkType(r4, 'cuda', (4, 4))

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "isinstance check for FakeTensor won't work with compile")
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_aten_slice_scatter_multi_device(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            x1 = torch.rand(4, 4, device='cpu')
            y1 = torch.rand(2, 4, device='cuda')
            x2 = torch.rand(4, 4, device='cuda')
            y2 = torch.rand(2, 4, device='cpu')
            out = torch.empty(4, 4, device='cpu')
            r1 = torch.ops.aten.slice_scatter.default(x1, y1, start=2)
            r2 = torch.ops.aten.slice_scatter.default(x2, y2, start=2)
            r3 = torch.ops.aten.slice_scatter.out(x1, y1, out=out, start=2)
        self.checkType(r1, 'cpu', (4, 4))
        self.checkType(r2, 'cuda', (4, 4))
        self.checkType(r3, 'cpu', (4, 4))
        self.checkType(out, 'cpu', (4, 4))

    def test__adaptive_avg_pool2d_backward(self):
        if False:
            print('Hello World!')
        with FakeTensorMode():
            grad_out = torch.rand(2, 3, 4, 4)
            inp = torch.rand(2, 3, 4, 4).to(memory_format=torch.channels_last)
            grad_in = torch.ops.aten._adaptive_avg_pool2d_backward(grad_out, inp)
            self.assertTrue(torch._prims_common.suggest_memory_format(grad_in) == torch.channels_last)

class FakeTensorConstHandling(TestCase):

    def assertConst(self, *args):
        if False:
            return 10
        for arg in args:
            self.assertTrue(arg.constant is not None)

    def assertNotConst(self, *args):
        if False:
            for i in range(10):
                print('nop')
        for arg in args:
            self.assertTrue(arg.constant is None)

    def test_simple(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x = torch.tensor(4.0)
            self.assertEqual(x.item(), 4.0)

    def test_inplace_add(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            x = torch.tensor(4.0)
            y = x.add_(1)
            self.assertEqual(x.item(), 5.0)
            self.assertEqual(y.item(), 5.0)
            self.assertConst(x, y)

    def test_shared_storages(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            x = torch.tensor([4.0])
            y = x[:]
            self.assertEqual(x.storage()._cdata, y.storage()._cdata)
            self.assertEqual(x.constant.storage()._cdata, y.constant.storage()._cdata)

    def test_constant_invalidation(self):
        if False:
            print('Hello World!')
        with FakeTensorMode():
            x = torch.tensor([1.0])
            self.assertConst(x)
            y = torch.rand([1])
            x.add_(y)
            self.assertNotConst(x)

    def test_inplace_view_invalidation(self):
        if False:
            for i in range(10):
                print('nop')
        with FakeTensorMode():
            x = torch.tensor([1])
            self.assertConst(x)
            x.resize_([2])
            self.assertEqual(x.size(0), 2)
            self.assertNotConst(x)

    def test_fake_tensor_in_intlist_repro(self):
        if False:
            while True:
                i = 10

        def fn(tensors):
            if False:
                return 10
            max_size = torch.tensor([800, 1216], dtype=torch.int64)
            batch_shape = [len(tensors)] + list(tensors[0].shape[:-2]) + list(max_size)
            return tensors[0].new_full(batch_shape, 0.0)
        with self.assertRaises(torch._subclasses.fake_tensor.DataDependentOutputException):
            with torch._subclasses.fake_tensor.FakeTensorMode():
                a = torch.randn(3, 800, 1199)
                b = torch.randn(3, 800, 800)
                inputs = [a, b]
                ref = fn(inputs)

    def test_fake_tensor_batch_norm_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        with torch._subclasses.CrossRefFakeMode():
            m = torch.nn.Sequential(torch.nn.BatchNorm2d(10), torch.nn.ReLU())
            m.eval()
            out = m(torch.randn([2, 10, 8, 8]))

    def test_shared_storage_invalidation(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x = torch.tensor([1.0])
            y = x[:]
            self.assertConst(x, y)
            y.add_(torch.rand([1]))
            self.assertNotConst(x, y)

    def test_aliased_const_write(self):
        if False:
            return 10
        with FakeTensorMode():
            x = torch.tensor([1])
            y = x.expand([4])
            self.assertNotConst(y)
            y[0] = 1
            self.assertNotConst(x)

    def test_constant_propagate_through_functions(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            y = torch.div(4, 4, rounding_mode='trunc')
            self.assertConst(y)

def contains_type(type: torch._C.Type, maybe_contained_type: torch._C.Type):
    if False:
        return 10
    return maybe_contained_type.isSubtypeOf(type) or any((contains_type(e, maybe_contained_type) for e in type.containedTypes()))

class FakeTensorOpInfoTest(TestCase):

    @ops(custom_op_db, dtypes=OpDTypes.any_one)
    def test_fake(self, device, dtype, op):
        if False:
            i = 10
            return i + 15
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        for sample_input in sample_inputs_itr:
            args = (sample_input.input,) + sample_input.args
            kwargs = sample_input.kwargs
            optests.fake_check(op, args, kwargs)

class FakeTensorConverterTest(TestCase):

    def test_memoized_conversion_to_meta(self):
        if False:
            i = 10
            return i + 15
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        self.assertTrue(mode.from_tensor(x) is mode.from_tensor(x))

    def test_memoized_conversion_from_meta(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(2, 2).to(device='meta')
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        self.assertTrue(converter.from_meta_and_device(mode, x, 'cpu') is converter.from_meta_and_device(mode, x, 'cpu'))

    def test_separate_tensor_storages_view(self):
        if False:
            print('Hello World!')
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter(mode, x)
        y_conv = converter(mode, y)
        self.assertEqual(torch._C._storage_id(x_conv), torch._C._storage_id(y_conv))

    @skipIfTorchDynamo('https://github.com/pytorch/torchdynamo/issues/1991')
    def test_separate_tensor_storages_non_view(self):
        if False:
            print('Hello World!')
        x = torch.rand(2, 2, 2)
        y = torch.rand(4, 2)
        y.set_(x.storage())
        mode = FakeTensorMode()
        converter = mode.fake_tensor_converter
        x_conv = converter(mode, x)
        y_conv = converter(mode, y)
        stor_id = torch._C._storage_id(x_conv)
        self.assertEqual(stor_id, torch._C._storage_id(y_conv))
        del x
        self.assertEqual(len(converter.tensor_memo), 1)
        converter.meta_converter.check_for_expired_weak_storages()
        self.assertEqual(len(converter.meta_converter.storage_memo), 1)
        del y
        self.assertEqual(len(converter.tensor_memo), 0)
        converter.meta_converter.check_for_expired_weak_storages()
        self.assertEqual(len(converter.meta_converter.storage_memo), 0)

    @skipIfTorchDynamo('https://github.com/pytorch/torchdynamo/issues/1991')
    def test_dead_weak_ref(self):
        if False:
            return 10
        x = torch.rand(2, 2, 2)
        y = x[0]
        mode = FakeTensorMode()
        converter = FakeTensorConverter()
        x_conv = converter(mode, x)
        x_conv_storage = torch._C._storage_id(x_conv)
        del x_conv
        self.assertFalse(x in converter.tensor_memo)
        y_conv = converter(mode, y)
        self.assertEqual(x_conv_storage, torch._C._storage_id(y_conv))

    @skipIfTorchDynamo('https://github.com/pytorch/torchdynamo/issues/1991')
    def test_dead_key(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.rand(2, 2, 2)
        mode = FakeTensorMode()
        converter = FakeTensorConverter()
        x_conv = converter(mode, x)
        self.assertEqual(len(converter.tensor_memo), 1)
        x_conv2 = converter(mode, x)
        assert x_conv2 is x_conv
        del x
        self.assertEqual(len(converter.tensor_memo), 0)

    def test_no_active_mode(self):
        if False:
            return 10
        with FakeTensorMode() as mode:
            x = torch.empty(2, 2, device='cpu')
            y = torch.empty(2, 2, device='cpu')
        out = x + y
        self.assertEqual(mode, out.fake_mode)
        self.assertTrue(isinstance(out, FakeTensor))
        self.assertEqual(out.device.type, 'cpu')

    def test_multiple_modes(self):
        if False:
            for i in range(10):
                print('nop')
        t = torch.rand([4])
        t2 = torch.rand([4])
        with FakeTensorMode() as m:
            with FakeTensorMode() as m2:
                t_fake = m.from_tensor(t)
                t2_fake = m2.from_tensor(t2)
                with self.assertRaisesRegex(Exception, 'Mixing fake modes'):
                    t_fake + t2_fake

    def test_separate_mode_error(self):
        if False:
            return 10
        with FakeTensorMode():
            x = torch.empty(2, 2, device='cpu')
        with FakeTensorMode():
            y = torch.empty(2, 2, device='cpu')
        self.assertRaises(Exception, lambda : x, y)

    @skipIfTorchDynamo('https://github.com/pytorch/torchdynamo/issues/1991')
    def test_no_ref_cycle(self):
        if False:
            while True:
                i = 10
        x = torch.rand([4])
        mode = FakeTensorMode()
        y = mode.from_tensor(x)
        self.assertEqual(len(mode.fake_tensor_converter.tensor_memo), 1)
        mode_weak = weakref.ref(mode)
        y_weak = weakref.ref(mode)
        del mode
        del y
        assert mode_weak() is None
        assert y_weak() is None

class FakeTensorOperatorInvariants(TestCase):

    @staticmethod
    def get_aten_op(schema):
        if False:
            i = 10
            return i + 15
        (namespace, name) = schema.name.split('::')
        overload = schema.overload_name if schema.overload_name else 'default'
        assert namespace == 'aten'
        return getattr(getattr(torch.ops.aten, name), overload)

    @staticmethod
    def get_all_aten_schemas():
        if False:
            return 10
        for schema in torch._C._jit_get_all_schemas():
            namespace = schema.name.split('::')[0]
            if namespace != 'aten':
                continue
            yield schema

    def test_non_kwarg_only_device(self):
        if False:
            return 10
        for schema in self.get_all_aten_schemas():
            ten_type = torch._C.TensorType.get()
            if not any((contains_type(arg.type, ten_type) for arg in itertools.chain(schema.arguments, schema.returns))):
                continue
            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_non_kwarg_device = any((not arg.kwarg_only and arg.type.isSubtypeOf(opt_device) for arg in schema.arguments))
            if has_non_kwarg_device:
                self.assertTrue(self.get_aten_op(schema) in torch._subclasses.fake_tensor._device_not_kwarg_ops)

    def test_tensor_constructors_all_have_kwarg_device(self):
        if False:
            while True:
                i = 10
        for schema in self.get_all_aten_schemas():
            op = self.get_aten_op(schema)
            if not torch._subclasses.fake_tensor._is_tensor_constructor(op):
                continue
            opt_device = torch._C.OptionalType(torch._C.DeviceObjType.get())
            has_kwarg_device = any((arg.kwarg_only and arg.type.isSubtypeOf(opt_device) for arg in schema.arguments))
            self.assertTrue(has_kwarg_device or op == torch.ops.aten._list_to_tensor.default)

    @unittest.expectedFailure
    def test_sparse_new(self):
        if False:
            while True:
                i = 10
        with FakeTensorMode():
            indices = torch.randn(1, 1, dtype=torch.int64)
            values = torch.randn(1)
            extra = (2,)
            sparse = torch.randn(1).to_sparse()
            sparse2 = sparse.new(indices, values, extra)

    def test_tensor_new(self):
        if False:
            i = 10
            return i + 15
        with FakeTensorMode():
            x = torch.Tensor([1, 2, 3])
        self.assertIsInstance(x, FakeTensor)

    def test_like_ops(self):
        if False:
            i = 10
            return i + 15
        for schema in self.get_all_aten_schemas():
            if '_like' == schema.name[-5:]:
                op = self.get_aten_op(schema)
                self.assertIn(op, torch._subclasses.fake_tensor._like_tensor_constructors)

    def test_embedding_bag_private(self):
        if False:
            return 10
        args = [torch.ones(6, 1), torch.ones(6, dtype=torch.int64), torch.arange(2, dtype=torch.int64), False, 2]
        ref_out = torch.ops.aten._embedding_bag(*args)
        with FakeTensorMode() as m:
            meta_args = [m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
            meta_out = torch.ops.aten._embedding_bag(*meta_args)
        self.assertEqual(len(ref_out), len(meta_out))
        for (ref_o, meta_o) in zip(ref_out, meta_out):
            self.assertEqual(ref_o.size(), meta_o.size())

    def test_cross_entropy_loss(self):
        if False:
            while True:
                i = 10
        inp = torch.randn(3, 5)
        target = torch.randint(5, (3,), dtype=torch.long)
        weight = torch.rand(5)
        fn = torch.nn.functional.cross_entropy
        for w in (weight, None):
            args = (inp, target, w)
            ref = fn(*args)
            with FakeTensorMode() as m:
                meta_args = [m.from_tensor(a) if isinstance(a, torch.Tensor) else a for a in args]
                meta_out = torch.nn.functional.cross_entropy(*meta_args, label_smoothing=0.5)
            self.assertEqual(ref.size(), meta_out.size())

    @skipIfRocm
    @unittest.skipIf(not PLATFORM_SUPPORTS_FLASH_ATTENTION, 'Does not support SDPA or pre-SM80 hardware')
    def test_flash_attention(self):
        if False:
            return 10

        class Repro(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()

            def forward(self, arg1, arg2, arg3):
                if False:
                    i = 10
                    return i + 15
                torch.ops.aten._scaled_dot_product_flash_attention(arg1, arg2, arg3, scale=0.17677669529663687)
        args_new = [[((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, 'cuda'), ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, 'cuda'), ((1, 48, 64, 64), (0, 4096, 64, 1), torch.float16, 'cuda')], [((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, 'cuda'), ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, 'cuda'), ((4, 2, 16, 32), (1024, 512, 32, 1), torch.float16, 'cuda')]]
        for args_list in args_new:
            args = [rand_strided(bsz, num_heads, seq_len, head_dim) for (bsz, num_heads, seq_len, head_dim) in args_list]
            try:
                with torch._subclasses.CrossRefFakeMode():
                    Repro()(*args)
            except RuntimeError as e:
                self.assertTrue('output[0]' not in str(e))
                self.assertTrue('found mismatched tensor metadata for output[6]: Devices cpu and cuda:0 are not equal!' in str(e))

    @skipIfRocm
    @unittest.skipIf(not RUN_CUDA, 'requires cuda')
    def test_conv_c1_backward(self):
        if False:
            for i in range(10):
                print('nop')

        class Repro(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()

            def forward(self, arg1, arg2, arg3):
                if False:
                    i = 10
                    return i + 15
                torch.ops.aten.convolution_backward.default(arg1, arg2, arg3, [1], [1, 1], [1, 1], [1, 1], False, [0, 0], 1, [True, True, False])
        args_new = [((16, 1, 128, 128), (16384, 16384, 128, 1), torch.float16, 'cuda'), ((16, 64, 128, 128), (1048576, 1, 8192, 64), torch.float16, 'cuda'), ((1, 64, 3, 3), (576, 9, 3, 1), torch.float16, 'cuda')]
        args = [rand_strided(sh, st, dt, dev) for (sh, st, dt, dev) in args_new]
        with torch._subclasses.CrossRefFakeMode():
            Repro()(*args)

    def test_no_dispatch_with_like_function(self):
        if False:
            print('Hello World!')

        class CountingMode(TorchDispatchMode):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                self.count = 0

            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if False:
                    for i in range(10):
                        print('nop')
                self.count += 1
                return func(*args, **kwargs)
        with FakeTensorMode():
            x = torch.randn(2)
            with CountingMode() as mode:
                with no_dispatch():
                    torch.zeros_like(x)
        self.assertEqual(mode.count, 0)

class FakeTensorPropTest(TestCase):

    def test_fake_tensor_prop_on_nn_module(self):
        if False:
            i = 10
            return i + 15

        class ToyNnModuleWithParameters(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value):
                if False:
                    while True:
                        i = 10
                value = self.layer1(value)
                value = torch.relu(value)
                value = self.layer2(value)
                return value
        model = ToyNnModuleWithParameters()
        value = torch.randn(5, 4)
        graph_model = torch.fx.symbolic_trace(model, (value,))
        with FakeTensorMode() as fake_tensor_mode:

            def to_fake_tensor(x):
                if False:
                    print('Hello World!')
                if isinstance(x, torch.Tensor) and (not isinstance(x, FakeTensor)):
                    return fake_tensor_mode.from_tensor(x)
                return x
            fake_parameters_and_buffers = {k: to_fake_tensor(v) for (k, v) in itertools.chain(graph_model.named_parameters(), graph_model.named_buffers())}
            with torch.nn.utils.stateless._reparametrize_module(graph_model, fake_parameters_and_buffers):
                result = FakeTensorProp(graph_model, fake_tensor_mode).propagate(value)
                self.assertTrue(isinstance(result, FakeTensor))
                self.assertEqual(result.shape, (5, 2))
                failed = False
                try:
                    FakeTensorProp(graph_model).propagate(value)
                except AssertionError:
                    failed = True
                self.assertTrue(failed)

    def test_fake_tensor_prop_on_nn_module_with_optional_args(self):
        if False:
            print('Hello World!')

        class OptionalArgumentInBetween(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.layer1 = torch.nn.Linear(4, 3)
                self.layer2 = torch.nn.Linear(3, 2)

            def forward(self, value, another_value=None, another_optional_value=None):
                if False:
                    for i in range(10):
                        print('nop')
                if another_value is None:
                    another_value = torch.rand_like(value)
                if another_optional_value is None:
                    another_optional_value = torch.rand_like(value)
                value = value + another_value + another_optional_value
                return value * value
        fake_mode = FakeTensorMode(allow_non_fake_inputs=True, allow_fallback_kernels=False)
        with fake_mode:
            model = OptionalArgumentInBetween()
            value = torch.randn(5, 4)
            another_optional_value = torch.randn(5, 4)
            graph_model = torch.fx.symbolic_trace(model, (value, None, another_optional_value))
            FakeTensorProp(graph_model, fake_mode).propagate(value, None, another_optional_value)
instantiate_parametrized_tests(FakeTensorTest)
only_for = ('cpu', 'cuda')
instantiate_device_type_tests(FakeTensorOpInfoTest, globals(), only_for=only_for)
if __name__ == '__main__':
    run_tests()