from typing import Sequence
import torch
import functools
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import ops, instantiate_device_type_tests
import torch._lazy
import torch._lazy.config
import torch._lazy.metrics
import torch._lazy.ir_cache
import torch._lazy.ts_backend
import itertools
import yaml
import os
import pathlib
from unittest import skip
torch._lazy.ts_backend.init()

def get_test_device():
    if False:
        while True:
            i = 10
    return 'cuda' if 'LTC_TS_CUDA' in os.environ else 'cpu'

def remove_suffixes(l):
    if False:
        while True:
            i = 10
    return [x.split('.')[0] for x in l]

def init_lists():
    if False:
        i = 10
        return i + 15
    path_to_script = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
    TS_NATIVE_FUNCTIONS_PATH = path_to_script.parent.parent / 'aten/src/ATen/native/ts_native_functions.yaml'
    with open(TS_NATIVE_FUNCTIONS_PATH) as f:
        yaml_ts = yaml.load(f, yaml.SafeLoader)
    LAZY_OPS_LIST = set(remove_suffixes(itertools.chain(yaml_ts['full_codegen'], yaml_ts['supported'], yaml_ts['autograd'])))
    HAS_SYMINT_SUFFIX = yaml_ts['symint']
    FALLBACK_LIST = {'clamp'}
    SKIP_RUNTIME_ERROR_LIST = {'index_select', 'clone', 'nonzero', 'all', 'any', 'logdet'}
    SKIP_INCORRECT_RESULTS_LIST = {'squeeze', 't', 'transpose', 'bernoulli', 'pow', 'addcdiv'}
    FUNCTIONAL_DECOMPOSE_LIST = {'diag_embed', 'block_diag', 'new_empty_strided', 'narrow_copy', 'pixel_shuffle', 'pixel_unshuffle', 'select_backward', '_trilinear', 'linalg_inv_ex', 'linalg_pinv.atol_rtol_tensor', 'logsumexp'}
    SKIP_VARIANT_LIST = {'norm_nuc', 'min_reduction_with_dim'}
    return (LAZY_OPS_LIST, FALLBACK_LIST, SKIP_RUNTIME_ERROR_LIST, SKIP_INCORRECT_RESULTS_LIST, FUNCTIONAL_DECOMPOSE_LIST, HAS_SYMINT_SUFFIX, SKIP_VARIANT_LIST)
(LAZY_OPS_LIST, FALLBACK_LIST, SKIP_RUNTIME_ERROR_LIST, SKIP_INCORRECT_RESULTS_LIST, FUNCTIONAL_DECOMPOSE_LIST, HAS_SYMINT_SUFFIX, SKIP_VARIANT_LIST) = init_lists()
torch.manual_seed(42)

def clone_move(t):
    if False:
        for i in range(10):
            print('nop')
    dev = 'lazy'
    copy_t = t.detach().clone().requires_grad_(True).to(device=dev)
    return copy_t

class TestLazyTensor(JitTestCase):

    @skip('Disable until autograd supports symints')
    def testConvolutionBackward(self):
        if False:
            print('Hello World!')
        test_device = get_test_device()
        inp = torch.rand(1, 3, 128, 128, device=test_device, requires_grad=True)
        inp_copy = clone_move(inp)
        grad = torch.rand(1, 32, 121, 121, device=test_device)
        grad_copy = clone_move(grad)
        weight = torch.rand(32, 3, 8, 8, device=test_device, requires_grad=True)
        weight_copy = clone_move(weight)
        bias = torch.rand(32, device=test_device, requires_grad=True)
        bias_copy = clone_move(bias)
        conv_out = torch.nn.functional.conv2d(inp, weight, bias)
        (inp_grad, weight_grad, bias_grad) = torch.autograd.grad([conv_out], [inp, weight, bias], [grad])
        conv_copy_out = torch.nn.functional.conv2d(inp_copy, weight_copy, bias_copy)
        (inp_copy_grad, weight_copy_grad, bias_copy_grad) = torch.autograd.grad([conv_copy_out], [inp_copy, weight_copy, bias_copy], [grad_copy])
        torch.testing.assert_close(bias_copy_grad.cpu(), bias_grad.cpu())
        torch.testing.assert_close(weight_copy_grad.cpu(), weight_grad.cpu())
        torch.testing.assert_close(inp_copy_grad.cpu(), inp_grad.cpu())

    def test_view_mark_step_preserved(self):
        if False:
            print('Hello World!')
        test_device = get_test_device()
        inp = torch.rand(4, device=test_device)
        inp_lazy = clone_move(inp)

        def foo(x, *, mark_step):
            if False:
                return 10
            y = x.view(2, 2)
            y.add_(1)
            z = x + x
            if mark_step:
                torch._lazy.mark_step()
            y.add_(1)
            return x
        out_ref = foo(inp, mark_step=False)
        out = foo(inp_lazy, mark_step=True)
        torch.testing.assert_close(out_ref.cpu(), out.cpu())

    def test_tensor_ctr(self):
        if False:
            for i in range(10):
                print('nop')
        test_device = get_test_device()
        inp = torch.tensor([[1, 2, 3, 4, 5]], device=test_device)
        inp_lazy = torch.tensor([[1, 2, 3, 4, 5]], device='lazy')

        def foo(x):
            if False:
                i = 10
                return i + 15
            return x.view(-1)
        out_ref = foo(inp)
        out = foo(inp_lazy)
        torch.testing.assert_close(out_ref.cpu(), out.cpu())

class TestLazyOpInfo(TestCase):

    @ops([op for op in op_db if op.name in LAZY_OPS_LIST and op.name not in SKIP_RUNTIME_ERROR_LIST and (op.name not in FUNCTIONAL_DECOMPOSE_LIST) and (op.formatted_name not in SKIP_VARIANT_LIST)], allowed_dtypes=(torch.float,))
    def test_dispatched_to_lazy(self, device, dtype, op):
        if False:
            i = 10
            return i + 15

        def get_name(op):
            if False:
                for i in range(10):
                    print('nop')
            l = [op.name]
            if op.variant_test_name != '':
                l.append(op.variant_test_name)
            return '.'.join(l)
        global HAS_SYMINT_SUFFIX, FALLBACK_LIST
        samples = op.sample_inputs('lazy', dtype, requires_grad=False)
        sample = list(samples)[0]
        args = [sample.input] + list(sample.args)
        kwargs = sample.kwargs
        torch._lazy.mark_step()
        torch._lazy.wait_device_ops()
        torch._lazy.metrics.reset()
        r = op(*args, **kwargs)
        torch._lazy.mark_step()
        torch._lazy.wait_device_ops()
        prefix = 'aten' if op.name in FALLBACK_LIST else 'lazy'
        symint_suffix = '_symint' if op.name in HAS_SYMINT_SUFFIX else ''
        found = f'{prefix}::{op.name}{symint_suffix}' in remove_suffixes(torch._lazy.metrics.counter_names())
        if not found:
            for alias in op.aliases:
                alias_found = f'{prefix}::{alias.name}{symint_suffix}' in remove_suffixes(torch._lazy.metrics.counter_names())
                found = found or alias_found
                if found:
                    break
        self.assertTrue(found)

    @ops([op for op in op_db if op.name in LAZY_OPS_LIST and op.name not in SKIP_RUNTIME_ERROR_LIST | SKIP_INCORRECT_RESULTS_LIST], allowed_dtypes=(torch.float,))
    def test_correctness(self, device, dtype, op):
        if False:
            while True:
                i = 10
        test_device = get_test_device()

        def clone_to_device(input, dev):
            if False:
                while True:
                    i = 10
            if isinstance(input, torch.Tensor):
                return input.detach().clone().to(device=dev)
            if isinstance(input, Sequence) and (not isinstance(input, str)):
                return tuple(map(functools.partial(clone_to_device, dev=dev), input))
            return input

        def assert_allclose_rec(t):
            if False:
                while True:
                    i = 10
            (a, b) = t
            self.assertEqual(type(a), type(b))
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.allclose(clone_to_device(a, test_device), b, atol=0.0001))
            if isinstance(a, Sequence):
                map(assert_allclose_rec, zip(a, b))
        samples = op.sample_inputs('lazy', dtype, requires_grad=False)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            copy_args = clone_to_device(args, test_device)
            r_exp = op(*copy_args, **kwargs)
            r_actual = op(*args, **kwargs)
            assert_allclose_rec((r_actual, r_exp))

    @ops([op for op in op_db if op.name in LAZY_OPS_LIST and op.name not in SKIP_RUNTIME_ERROR_LIST | SKIP_INCORRECT_RESULTS_LIST], allowed_dtypes=(torch.float,))
    def test_correctness_with_reusing_ir(self, device, dtype, op):
        if False:
            return 10
        torch._lazy.config.set_reuse_ir(True)
        test_device = get_test_device()

        def clone_to_device(input, dev):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(input, torch.Tensor):
                return input.detach().clone().to(device=dev)
            if isinstance(input, Sequence) and (not isinstance(input, str)):
                return tuple(map(functools.partial(clone_to_device, dev=dev), input))
            return input

        def assert_allclose_rec(t):
            if False:
                i = 10
                return i + 15
            (a, b) = t
            self.assertEqual(type(a), type(b))
            if isinstance(a, torch.Tensor):
                self.assertTrue(torch.allclose(clone_to_device(a, test_device), b, atol=0.0001))
            if isinstance(a, Sequence):
                map(assert_allclose_rec, zip(a, b))
        samples = op.sample_inputs('lazy', dtype, requires_grad=False)
        for sample in samples:
            args = [sample.input] + list(sample.args)
            kwargs = sample.kwargs
            copy_args = clone_to_device(args, test_device)
            r_exp = op(*copy_args, **kwargs)
            r_actual = op(*args, **kwargs)
            torch._lazy.mark_step()
            assert_allclose_rec((r_actual, r_exp))
        torch._lazy.ir_cache.reset()
        torch._lazy.config.set_reuse_ir(False)
instantiate_device_type_tests(TestLazyOpInfo, globals(), only_for='cpu')

class TestLazyDynamicOps(TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        if False:
            while True:
                i = 10
        cls.old_ssa_mode = torch._C._lazy._get_symbolic_shape_mode()
        torch._C._lazy._set_symbolic_shape_mode(True)
        return super().setUpClass()

    @classmethod
    def tearDownClass(cls) -> None:
        if False:
            print('Hello World!')
        torch._C._lazy._set_symbolic_shape_mode(cls.old_ssa_mode)
        return super().tearDownClass()

    def test_nonzero_dynamic(self):
        if False:
            while True:
                i = 10
        test_device = get_test_device()
        x1 = torch.tensor([[0, 1.0, 2.0], [3.0, 0, 0]], device=test_device, requires_grad=True)
        x1_lazy = clone_move(x1)
        x2_lazy = torch.nonzero(x1_lazy)
        x2_eager = x2_lazy.cpu()
        self.assertEqual(tuple(x2_eager.size()), (3, 2))

    def test_adaptiveavgpool3d_dynamic(self):
        if False:
            print('Hello World!')
        img_cpu = torch.zeros([2, 3, 4, 5, 6], device='cpu')
        out_cpu = torch.nn.AdaptiveAvgPool3d(2).to(device='cpu')(img_cpu)
        test_device = get_test_device()
        img_lazy = torch.zeros([2, 3, 4, 5, 6], device=test_device)
        out_lazy = torch.nn.AdaptiveAvgPool3d(2).to(test_device)(img_lazy)
        self.assertEqual(out_cpu.shape, out_lazy.shape)
if __name__ == '__main__':
    run_tests()