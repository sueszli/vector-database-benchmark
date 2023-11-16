import functools
import re
import sys
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from unittest import mock
import torch
import torch.nn as nn
from torch import _inductor as inductor
from torch._dynamo import compiled_autograd
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch.testing._internal.inductor_utils import HAS_CPU, HAS_CUDA

def compiler_fn(gm):
    if False:
        while True:
            i = 10
    'Same as torch.compile() but counts number of compiles'

    def inner_compiler(gm_, example_inputs_):
        if False:
            i = 10
            return i + 15
        counters['compiled_autograd']['compiles'] += 1
        return inductor.compile(gm_, example_inputs_)
    return torch.compile(gm, backend=inner_compiler, fullgraph=True, dynamic=True)

def hook1(grad):
    if False:
        for i in range(10):
            print('nop')
    return grad * 2

def hook2(grads):
    if False:
        while True:
            i = 10
    return (grads[0] + 1,)

def hook3(gI, gO):
    if False:
        print('Hello World!')
    return (torch.sin(gI[0]) + gO[0],)

class TestCompiledAutograd(TestCase):

    def check_output_and_recompiles(self, fn, count=1):
        if False:
            print('Hello World!')
        with torch.autograd.set_multithreading_enabled(False):
            torch._dynamo.reset()
            counters['compiled_autograd'].clear()
            torch.manual_seed(123)
            expected = list(fn())
            torch.manual_seed(123)
            with compiled_autograd.enable(compiler_fn):
                actual = list(fn())
            self.assertEqual(expected, actual)
            self.assertEqual(counters['compiled_autograd']['captures'], count)
            self.assertEqual(counters['compiled_autograd']['compiles'], count)

    def test_basic(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU())
            x = torch.randn([2, 4])
            result = model(x).sum()
            result.backward()
            yield model[0].weight.grad
            yield model[0].bias.grad
            yield model[2].weight.grad
            yield model[2].bias.grad
        self.check_output_and_recompiles(fn)

    def test_cache_hit(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                print('Hello World!')
            for _ in range(3):
                model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU())
                x = torch.randn([2, 4])
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook1(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                i = 10
                return i + 15
            for _ in range(3):
                model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
                x = torch.randn([2, 4])
                model[0].weight.register_hook(hook1)
                result = model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook2(self):
        if False:
            return 10

        def fn():
            if False:
                while True:
                    i = 10
            for _ in range(3):
                model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
                x = torch.randn([1, 4])
                result = model(x).sum()
                result.grad_fn.register_prehook(hook2)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
        self.check_output_and_recompiles(fn)

    def test_tensor_grad_hook3(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                return 10
            for _ in range(3):
                model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
                x = torch.randn([1, 4])
                result = model(x).sum()
                result.grad_fn.register_hook(hook3)
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
        self.check_output_and_recompiles(fn)

    def test_torch_compile(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                while True:
                    i = 10
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.Sigmoid())
            opt_model = torch.compile(model, fullgraph=True)
            for _ in range(3):
                x = torch.randn([1, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                model.zero_grad()
        self.check_output_and_recompiles(fn)

    def test_implicit_add(self):
        if False:
            return 10

        def fn():
            if False:
                while True:
                    i = 10
            y = torch.randn(1, 4, requires_grad=True)

            def model(x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.sigmoid(x * y + torch.sin(y) + torch.cos(y))
            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                result.backward()
                yield result
                yield y.grad
                y.grad = None
        self.check_output_and_recompiles(fn)

    def test_output_nodes(self):
        if False:
            while True:
                i = 10

        def fn():
            if False:
                print('Hello World!')
            y = torch.randn(1, 4, requires_grad=True)
            z = torch.randn(1, 4, requires_grad=True)

            def model(x):
                if False:
                    while True:
                        i = 10
                return torch.sigmoid(x * z + torch.sin(y) + torch.cos(y))
            for _ in range(3):
                x = torch.randn([1, 4])
                result = model(x).sum()
                (gy, gz) = torch.autograd.grad(result, [y, z])
                assert y.grad is None
                assert z.grad is None
                yield gy
                yield gz
        self.check_output_and_recompiles(fn)

    def test_dynamic_shapes(self):
        if False:
            for i in range(10):
                print('nop')

        def fn():
            if False:
                return 10
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU())
            opt_model = torch.compile(model, dynamic=True)
            for b in range(10, 100, 10):
                x = torch.randn([b, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad
                yield model[0].bias.grad
                yield model[2].weight.grad
                yield model[2].bias.grad
                model.zero_grad()
        self.check_output_and_recompiles(fn, count=2)

    def test_accumulate_without_zero(self):
        if False:
            i = 10
            return i + 15

        def fn():
            if False:
                while True:
                    i = 10
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU(), torch.nn.Linear(4, 4), torch.nn.ReLU())
            opt_model = torch.compile(model, dynamic=True)
            for _ in range(10):
                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                yield model[0].weight.grad.clone()
                yield model[0].bias.grad.clone()
                yield model[2].weight.grad.clone()
                yield model[2].bias.grad.clone()
        self.check_output_and_recompiles(fn, count=2)

    def test_inplace_grad_update(self):
        if False:
            print('Hello World!')

        def fn():
            if False:
                while True:
                    i = 10
            model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.ReLU())
            opt_model = torch.compile(model, dynamic=True)
            for _ in range(10):
                w_grad = torch.rand_like(model[0].weight)
                b_grad = torch.rand_like(model[0].bias)
                model[0].weight.grad = w_grad
                model[0].bias.grad = b_grad
                x = torch.randn([10, 4])
                result = opt_model(x).sum()
                result.backward()
                assert model[0].weight.grad is w_grad
                assert model[0].bias.grad is b_grad
                yield w_grad.clone()
                yield b_grad.clone()
        self.check_output_and_recompiles(fn, count=1)

    @unittest.skipIf(not HAS_CUDA, 'requires cuda')
    def test_issue106555(self):
        if False:
            for i in range(10):
                print('nop')
        DEVICE = torch.device('cuda:0')
        NUM_FEATURES = 256

        def bias_sigmoid_mul(x1, x2, bias):
            if False:
                i = 10
                return i + 15
            x2 = torch.sigmoid(x2 + bias)
            y = x1 * x2
            return y
        bias_sigmoid_mul_jit = torch.compile(bias_sigmoid_mul)

        class ModuleWithJit(nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.linear_1 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=True)
                self.linear_2 = nn.Linear(NUM_FEATURES, NUM_FEATURES, bias=False)
                self.linear_2_bias = nn.Parameter(torch.zeros(NUM_FEATURES))

            def forward(self, input_tensor):
                if False:
                    while True:
                        i = 10
                x1 = self.linear_1(input_tensor)
                x2 = self.linear_2(input_tensor)
                output = bias_sigmoid_mul_jit(x1, x2, self.linear_2_bias)
                return output

        class Model(nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.module_with_jit_1 = ModuleWithJit()
                self.module_with_jit_2 = ModuleWithJit()

            def forward(self, x, gradient_checkpointing: bool):
                if False:
                    print('Hello World!')
                if gradient_checkpointing:
                    y = torch.utils.checkpoint.checkpoint(self._forward, x, use_reentrant=True)
                else:
                    y = self._forward(x)
                return y

            def _forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                x = x + self.module_with_jit_1(x)
                x = x + self.module_with_jit_2(x.transpose(-2, -3)).transpose(-2, -3)
                return x
        torch.cuda.set_device(device=DEVICE)
        torch.manual_seed(1234567890)
        model = Model()
        model.train()
        model.to(device=DEVICE)
        model_parameters = list(model.parameters())
        torch.manual_seed(1234567890)
        input_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(device=DEVICE)
        input_tensor.requires_grad = True
        target_tensor = torch.randn(1, 128, 256, NUM_FEATURES).to(dtype=input_tensor.dtype, device=DEVICE)
        for iteration in range(10):
            for param in model_parameters:
                param.grad = None
            output_tensor = model(x=input_tensor.clone(), gradient_checkpointing=True)
            loss = torch.mean(torch.abs(target_tensor - output_tensor))
            loss.backward()

    def test_keep_graph_simple(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2
        y.backward(retain_graph=True)
        self.assertEqual(x.grad, torch.Tensor([4]))

        def fn():
            if False:
                while True:
                    i = 10
            x.grad = torch.tensor([0.0])
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))
            return x.grad
        self.check_output_and_recompiles(fn, count=1)

    def test_keep_graph_usage_after_compiled(self):
        if False:
            return 10
        x = torch.tensor([2.0], requires_grad=True)
        y = x ** 2

        def eager_check():
            if False:
                while True:
                    i = 10
            y.backward(retain_graph=True)
            self.assertEqual(x.grad, torch.Tensor([4]))
            x.grad = torch.tensor([0.0])
        eager_check()
        for i in range(0, 5):
            with compiled_autograd.enable(compiler_fn):
                eager_check()
            eager_check()

def load_test_module(name):
    if False:
        while True:
            i = 10
    testdir = Path(__file__).absolute().parent.parent
    with mock.patch('sys.path', [*sys.path, str(testdir)]):
        return SourceFileLoader(name, str(testdir / f"{name.replace('.', '/')}.py")).load_module()
test_autograd = load_test_module('test_autograd')

class EagerAutogradTests(TestCase):

    @classmethod
    def add_test(cls, name, fn):
        if False:
            while True:
                i = 10

        @functools.wraps(fn)
        def wrapped(self: EagerAutogradTests):
            if False:
                return 10
            torch._dynamo.reset()
            with compiled_autograd.enable(compiler_fn):
                return fn(self)
        if not callable(fn):
            return
        elif known_failures_re.match(name) or name in known_failing_tests:
            setattr(cls, name, unittest.expectedFailure)
        elif name.startswith('test'):
            setattr(cls, name, wrapped)
        else:
            setattr(cls, name, fn)
known_failures_re = re.compile('^test_(sparse|profiler|gradcheck|checkpoint|named_tensor)')
known_failing_tests = {'test_current_graph_task_execution_order', 'test_input_buffer_accum', 'test_graph_save_on_cpu_cuda', 'test_graph_save_on_cpu', 'test_reentrant_with_leaf_variable_hook', 'test_reentrant_with_non_leaf_variable_hook', 'test_saved_variable_saved_original_inplace_detach', 'test_saving_variable_to_disk', 'test_setitem_mask', 'test_tensor_hooks_inplace_over_view', 'test_tensor_hooks_inplace', 'test_wrapped_number_saved_variable_hooks', 'test_accumulate_grad_posthooks_can_observe_tensor_prehook', 'test_accumulate_grad_tensor_reference', 'test_anomaly_detect_nan', 'test_anomaly_grad_warnings', 'test_autograd_inplace_views_cross_dtype', 'test_autograd_multiple_views_python', 'test_autograd_node_isinstance', 'test_autograd_python_custom_function_inplace', 'test_backward_with_inputs', 'test_callback_adds_callback', 'test_current_node', 'test_custom_function_cycle', 'test_custom_function_error', 'test_custom_function_exception', 'test_custom_function_non_tensor_inputs_outputs', 'test_custom_function_save_for_forward', 'test_custom_function_saved_tensors', 'test_custom_function_setup_context_multi_input', 'test_custom_function_setup_context_multi_output', 'test_custom_function_setup_context_simple', 'test_deep_reentrant', 'test_dep_nograd', 'test_dont_materialize_grads', 'test_function_returns_input', 'test_function_returns_undefined_tensor', 'test_grad_batched_grad', 'test_grad_fn_prehooks', 'test_grad_fn_prehooks_multiple_outputs', 'test_grad_fn_prehooks_remove_hooks', 'test_grad_mode_restored_reentrant', 'test_grad_unreachable_discovery', 'test_hook_none', 'test_index_backward_does_not_save_tensor', 'test_invalid_gradients', 'test_mark_non_differentiable_mixed', 'test_materialize_grads', 'test_naughty_anomaly_access', 'test_no_grad_copy', 'test_post_accumulate_grad_hook_e2e', 'test_post_accumulate_grad_hook_gets_cleaned_up', 'test_post_accumulate_grad_hook_multiple_hooks', 'test_post_accumulate_grad_hook_multiple_tensors', 'test_post_accumulate_grad_hook_ordering', 'test_post_accumulate_grad_hook_returns_not_None', 'test_reentrant_child_error', 'test_reentrant_priority', 'test_reentrant_with_callbacks_both_depths', 'test_reentrant_with_callbacks_depth_0', 'test_reentrant_with_callbacks_depth_1', 'test_retain_grad_cycle', 'test_retain_grad_inplace', 'test_retain_grad_inplace_over_view', 'test_retains_grad_can_always_observe_tensor_prehook', 'test_retains_grad_inplace_multiple_outputs', 'test_return_leaf', 'test_return_leaf_inplace', 'test_save_none_for_backward', 'test_save_output_nr', 'test_saved_tensor_hooks_custom_function_intermediates', 'test_saved_variables_deprecated', 'test_set_materialize_non_diff_grads', 'test_setup_context_when_forward_has_default_args', 'test_simple_reentrant', 'test_tensor_hooks_inplace_multiple_outputs', 'test_to_sparse_backward', 'test_too_many_grads', 'test_accumulate_grad', 'test_anomaly_assign_parent_cleanup', 'test_anomaly_mode_no_check_nan', 'test_autograd_simple_views_python', 'test_backward_create_graph_warns', 'test_backward_with_nonleaf_inputs', 'test_create_graph_and_full_backward_hook_cycle', 'test_current_graph_task_id', 'test_custom_autograd_no_early_free', 'test_custom_autograd_repeated_grad_grad', 'test_custom_function_forward_mode_forward_is_no_op', 'test_custom_function_forward_mode_inplace_checks', 'test_custom_function_forward_mode_view_checks', 'test_custom_function_forward_mode_wrong_formula', 'test_default_saved_variable_hooks_double_backward', 'test_full_backward_hook_double_backward', 'test_function', 'test_grad', 'test_grad_materialize_grads', 'test_grad_nonleaf', 'test_grad_nonleaf_many_outputs', 'test_hessian_vector', 'test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_False', 'test_hook_closure_cycle_use_custom_function_True_use_tensor_hook_True', 'test_hook_edge_case_when_called_with_grad', 'test_hooks', 'test_inplace_on_view_backward', 'test_lobpcg', 'test_multi_grad_hooks', 'test_naughty_autograd_function_stashing_ctx', 'test_nested_anomaly_detect_nan', 'test_nested_anomaly_printstack_cleanup', 'test_no_grad_copy_sparse', 'test_once_differentiable', 'test_prehook_ordering', 'test_retain_grad', 'test_return_duplicate', 'test_return_duplicate_inplace', 'test_saved_variable_packing_unpacking_saved_original_with_hooks', 'test_select_sum', 'test_unrelated_inputs', 'test_will_engine_execute_node', 'test_backward_to_node'}
if not HAS_CUDA:
    known_failing_tests.add('test_type_conversions')
for (name, fn) in test_autograd.TestAutograd.__dict__.items():
    EagerAutogradTests.add_test(name, fn)
if __name__ == '__main__':
    if HAS_CPU:
        run_tests(needs='filelock')