import collections
import unittest
import torch
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS
from torch.testing._internal.autocast_test_lists import AutocastCPUTestLists
from torch.utils._python_dispatch import TorchDispatchMode

class TestAutocastCPU(TestCase):

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        self.autocast_lists = AutocastCPUTestLists(torch.device('cpu'))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        del self.autocast_lists
        super().tearDown()

    def _run_autocast_outofplace(self, op, args, run_as_type, out_type=None, module=torch, add_kwargs=None):
        if False:
            for i in range(10):
                print('nop')

        def cast(val, to_type):
            if False:
                while True:
                    i = 10
            if isinstance(val, torch.Tensor):
                return val.to(to_type) if val.is_floating_point() else val
            elif isinstance(val, collections.abc.Iterable):
                return type(val)((cast(v, to_type) for v in val))
            else:
                return val
        if add_kwargs is None:
            add_kwargs = {}
        self.assertFalse(torch.is_autocast_cpu_enabled())
        with torch.cpu.amp.autocast():
            self.assertTrue(torch.is_autocast_cpu_enabled())
            out_type = out_type if out_type is not None else run_as_type
            output = output_method = None
            if module is not None and hasattr(module, op):
                output = getattr(module, op)(*args, **add_kwargs)
                if isinstance(output, torch.Tensor):
                    self.assertTrue(out_type == output.dtype, f'autocast for torch.{op} produced {output.dtype}, should produce {out_type}')
            if hasattr(torch.Tensor, op):
                output_method = getattr(args[0], op)(*args[1:], **add_kwargs)
                if isinstance(output_method, torch.Tensor):
                    self.assertTrue(out_type == output_method.dtype, 'autocast for torch.{} produced {}, should produce torch.{}'.format(op, output_method.dtype, out_type))
            self.assertTrue(output is not None or output_method is not None, f'{op} not found as an attribute on either Tensor or the requested module {module}')

            def compare(first, second):
                if False:
                    for i in range(10):
                        print('nop')
                if isinstance(first, torch.Tensor):
                    return torch.equal(first, second)
                elif isinstance(first, collections.abc.Iterable):
                    return all((compare(f, s) for (f, s) in zip(first, second)))
                else:
                    return first == second
            if output is not None and output_method is not None:
                self.assertTrue(type(output) == type(output_method))
                comparison = compare(output, output_method)
                self.assertTrue(comparison, f'torch.{op} result did not match Tensor.{op} result')
            output_to_compare = output if output is not None else output_method
            with torch.cpu.amp.autocast(enabled=False):
                self.assertFalse(torch.is_autocast_cpu_enabled())
                if module is not None and hasattr(module, op):
                    control = getattr(module, op)(*cast(args, run_as_type), **add_kwargs)
                else:
                    control = getattr(args[0].to(run_as_type), op)(*cast(args[1:], run_as_type), **add_kwargs)
                self.assertTrue(type(output_to_compare) == type(control))
                comparison = compare(output_to_compare, control)
                self.assertTrue(comparison, f'torch.{op} result did not match control')
            self.assertTrue(torch.is_autocast_cpu_enabled())
        self.assertFalse(torch.is_autocast_cpu_enabled())

    def args_maybe_kwargs(self, op_with_args):
        if False:
            while True:
                i = 10
        if len(op_with_args) == 2:
            return (op_with_args[0], op_with_args[1], {})
        else:
            return (op_with_args[0], op_with_args[1], op_with_args[2])

    def test_autocast_torch_expect_builtin_promote(self):
        if False:
            for i in range(10):
                print('nop')
        for (op, args, out_type) in self.autocast_lists.torch_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, out_type=out_type)

    def test_autocast_methods_expect_builtin_promote(self):
        if False:
            i = 10
            return i + 15
        for (op, args, out_type) in self.autocast_lists.methods_expect_builtin_promote:
            self._run_autocast_outofplace(op, args, torch.float32, module=None, out_type=out_type)

    def test_autocast_torch_bf16(self):
        if False:
            print('Hello World!')
        for op_with_args in self.autocast_lists.torch_bf16:
            (op, args, maybe_kwargs) = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.bfloat16, add_kwargs=maybe_kwargs)

    def test_autocast_nn_bf16(self):
        if False:
            print('Hello World!')
        for op_with_args in self.autocast_lists.nn_bf16:
            (op, args, maybe_kwargs) = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.bfloat16, module=torch._C._nn, add_kwargs=maybe_kwargs)

    def test_autocast_torch_fp32(self):
        if False:
            for i in range(10):
                print('nop')
        for op_with_args in self.autocast_lists.torch_fp32:
            (op, args, maybe_kwargs) = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, add_kwargs=maybe_kwargs)

    def test_autocast_nn_fp32(self):
        if False:
            for i in range(10):
                print('nop')
        for op_with_args in self.autocast_lists.nn_fp32:
            (op, args, maybe_kwargs) = self.args_maybe_kwargs(op_with_args)
            self._run_autocast_outofplace(op, args, torch.float32, module=torch._C._nn, add_kwargs=maybe_kwargs)

    def test_autocast_torch_need_autocast_promote(self):
        if False:
            for i in range(10):
                print('nop')
        for (op, args) in self.autocast_lists.torch_need_autocast_promote:
            self._run_autocast_outofplace(op, args, torch.float32)

    @unittest.skipIf(IS_WINDOWS, 'Limit support for bf16 path')
    def test_autocast_rnn(self):
        if False:
            for i in range(10):
                print('nop')
        if torch.backends.mkldnn.is_available() and torch.ops.mkldnn._is_mkldnn_bf16_supported():
            x = torch.randn(1, 2, 1)
            hx = torch.randn(2, 2, 1)
            cx = torch.randn(2, 2, 1)
            m = torch.nn.LSTM(1, 1, 2).to(torch.bfloat16)
            with self.assertRaisesRegex(ValueError, 'input must have the type'):
                m(x, (hx, cx))
            with torch.cpu.amp.autocast():
                m(x, (hx, cx))

    def test_autocast_disabled_with_fp32_dtype(self):
        if False:
            return 10
        with torch.autocast(device_type='cpu', dtype=torch.float32, enabled=False):
            _ = torch.ones(10)

class CustomLinear(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, w_t):
        if False:
            return 10
        ctx.save_for_backward(x, w_t)
        return torch.nn.functional.linear(x, w_t)

    @staticmethod
    def backward(ctx, grad_output):
        if False:
            print('Hello World!')
        (x, w_t) = ctx.saved_tensors
        with torch.autocast(device_type='cuda'):
            dL_dX = torch.matmul(grad_output, w_t)
            dL_dW = torch.matmul(x.transpose(0, 1), grad_output).transpose(0, 1)
        return (dL_dX, dL_dW)

class WeightDTypeCastCounterMode(TorchDispatchMode):

    def __init__(self, weight):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dtype_cast_counter = 0
        self.weight = weight

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        if func is torch.ops.aten._to_copy.default and args[0] is self.weight and (kwargs['dtype'] is torch.float16):
            self.dtype_cast_counter += 1
        return func(*args, **kwargs)

    def __enter__(self):
        if False:
            return 10
        self.old_clear_cache = torch.clear_autocast_cache
        torch.clear_autocast_cache = lambda : None
        return super().__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            for i in range(10):
                print('nop')
        torch.clear_autocast_cache = self.old_clear_cache
        return super().__exit__(exc_type, exc_val, exc_tb)

@unittest.skipIf(not torch.cuda.is_available(), 'requires cuda')
class TestAutocastGPU(TestCase):

    def test_cast_cache_is_global(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verifies that the autocast cache is global. This is done by\n        mocking out cache clearing at the end of the forward pass,\n        running forward+backward with an explicit call to autocast in the\n        backward, and verifying that the weight only get cast to float16 once.\n        '
        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())
        with WeightDTypeCastCounterMode(weight) as mode:
            with torch.autocast(device_type='cuda'):
                output = CustomLinear.apply(data, weight)
                s = output.sum()
            s.backward()
        self.assertEqual(mode.dtype_cast_counter, 1)

    def test_cache_disabled(self):
        if False:
            i = 10
            return i + 15
        data = torch.randn(2, 3).cuda()
        weight = torch.nn.Parameter(torch.randn(4, 3).cuda())
        try:
            torch._C._set_cached_tensors_enabled(True)
            torch._C._add_cached_tensor(weight)
            with WeightDTypeCastCounterMode(weight) as mode:
                with torch.autocast(device_type='cuda'):
                    output = CustomLinear.apply(data, weight)
                    s = output.sum()
                s.backward()
            self.assertEqual(mode.dtype_cast_counter, 2)
        finally:
            torch._C._set_cached_tensors_enabled(False)

class TestTorchAutocast(TestCase):

    def test_autocast_fast_dtype(self):
        if False:
            return 10
        gpu_fast_dtype = torch.get_autocast_gpu_dtype()
        cpu_fast_dtype = torch.get_autocast_cpu_dtype()
        self.assertEqual(gpu_fast_dtype, torch.half)
        self.assertEqual(cpu_fast_dtype, torch.bfloat16)

    def test_invalid_device(self):
        if False:
            while True:
                i = 10
        dev = 'not a real device'
        msg = f"unsupported autocast device_type '{dev}'"
        with self.assertRaisesRegex(RuntimeError, msg):
            with torch.autocast(device_type=dev):
                _ = torch.tensor(1)
if __name__ == '__main__':
    run_tests()