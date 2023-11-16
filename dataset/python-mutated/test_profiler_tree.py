import functools
import os
import re
import textwrap
import traceback
import unittest
import expecttest
import torch
from torch._C._profiler import _ExtraFields_PyCall, _ExtraFields_PyCCall
from torch.testing._internal.common_utils import TestCase, run_tests, IS_WINDOWS, TEST_WITH_CROSSREF, IS_ARM64
from torch.utils._pytree import tree_map
PRUNE_ALL = 1
KEEP_ELLIPSES = 2
KEEP_NAME_AND_ELLIPSES = 3
PRUNE_FUNCTIONS = {'torch/utils/_pytree.py(...): tree_map': KEEP_NAME_AND_ELLIPSES, 'torch/profiler/profiler.py(...): start': KEEP_ELLIPSES, 'torch/profiler/profiler.py(...): stop_trace': KEEP_ELLIPSES, 'torch/profiler/profiler.py(...): _transit_action': KEEP_ELLIPSES, '<built-in method __exit__ of torch._C.DisableTorchFunctionSubclass object at 0xXXXXXXXXXXXX>': PRUNE_ALL, 'cudaStreamIsCapturing': PRUNE_ALL, 'cudaGetDeviceCount': PRUNE_ALL, 'cudaGetDeviceProperties_v2': PRUNE_ALL}
ALLOW_CUDA_FAILURE = torch.version.hip is not None or IS_WINDOWS

class TorchFunctionTensor(torch.Tensor):

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10
        return super().__torch_function__(func, types, args, kwargs)

class TorchDispatchTensor(torch.Tensor):

    @staticmethod
    def __new__(cls, elem):
        if False:
            print('Hello World!')
        t = torch.Tensor._make_subclass(cls, elem, elem.requires_grad)
        t.elem = elem
        return t
    __torch_function__ = torch._C._disabled_torch_function_impl

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        if False:
            while True:
                i = 10

        def unwrap(x):
            if False:
                print('Hello World!')
            return x.elem if isinstance(x, TorchDispatchTensor) else x

        def wrap(x):
            if False:
                while True:
                    i = 10
            return TorchDispatchTensor(x) if isinstance(x, torch.Tensor) else x
        args = tree_map(unwrap, args)
        kwargs = tree_map(unwrap, kwargs or {})
        return tree_map(wrap, func(*args, **kwargs))

class ProfilerTree:

    @staticmethod
    def test(f):
        if False:
            for i in range(10):
                print('nop')
        'Mark unit test that will be using ProfilerTree to test traces.\n\n        This decorator serves two purposes. First, it provides a method name\n        that `format` can use to tell where the test runner (which is\n        environment specific) ends and the unit test begins. Second, it runs\n        the test with replicates and allows `assertTreesMatch` to adjust\n        based on which replicate is running.\n        '

        @functools.wraps(f)
        def begin_unit_test_marker(self, replicates=3):
            if False:
                print('Hello World!')
            try:
                for i in range(replicates):
                    self.tree_replicate = i
                    out = f(self)
                    if self.tree_replicate is None:
                        break
                return out
            finally:
                delattr(self, 'tree_replicate')
        return begin_unit_test_marker

    @classmethod
    def format(cls, profiler, indent: int=0):
        if False:
            print('Hello World!')

        def flatten(nodes, depth=0, out=None):
            if False:
                for i in range(10):
                    print('nop')
            if out is None:
                out = []
            for node in nodes:
                cls.validate_node(node)
                name = cls.fmt_name(node.name)
                prune_level = PRUNE_FUNCTIONS.get(name.strip(), None)
                if prune_level is None:
                    out.append((depth, name))
                    flatten(node.children, depth + 1, out)
                elif prune_level == KEEP_NAME_AND_ELLIPSES:
                    out.append((depth, name))
                    if node.children:
                        out.append((depth + 1, '...'))
                elif prune_level == KEEP_ELLIPSES:
                    out.append((depth, '...'))
                else:
                    assert prune_level == PRUNE_ALL
            return out
        flat_nodes = flatten(profiler.kineto_results.experimental_event_tree())
        if flat_nodes and flat_nodes[-2][1] == 'cudaDeviceSynchronize':
            flat_nodes = flat_nodes[:-2]
        if flat_nodes and flat_nodes[-1][1] == 'cudaDeviceSynchronize':
            flat_nodes = flat_nodes[:-1]
        if flat_nodes and flat_nodes[-1][1] == 'hipDeviceSynchronize':
            flat_nodes = flat_nodes[:-1]
        min_depth = min([d + 1 for (d, name) in flat_nodes if 'begin_unit_test_marker' in name] or [0])
        return textwrap.indent('\n'.join([f"{'  ' * (d - min_depth)}{name.rstrip()}" for (d, name) in flat_nodes if d >= min_depth]), ' ' * indent)

    @staticmethod
    def fmt_name(name: str) -> str:
        if False:
            return 10
        match = re.match('^(.*)\\.py\\(([0-9]+)\\): (.*)$', name)
        if match:
            (filename, _, fn) = match.groups()
            test_file = os.path.splitext(os.path.split(__file__)[1])[0]
            if filename.endswith(test_file):
                filename = test_file
            filename = filename.replace(os.sep, '/')
            lineno = '...'
            return f'{filename}.py({lineno}): {fn}'
        for kernel_pattern in ('void at::native::elementwise_kernel', 'void at::native::reduce_kernel', 'void at::native::vectorized_elementwise_kernel', 'void at::native::unrolled_elementwise_kernel', 'void [a-zA-Z0-9]+_kernel'):
            name = re.sub(f'{kernel_pattern}<.+>\\(.+\\)$', f"{kernel_pattern.replace('[a-zA-Z0-9]+', '...')}<...>(...)", name)
        return re.sub('object at 0x[0-9a-fA-F]+>', 'object at 0xXXXXXXXXXXXX>', name)

    @classmethod
    def validate_node(cls, node):
        if False:
            while True:
                i = 10
        extra_fields = node.extra_fields
        if isinstance(extra_fields, (_ExtraFields_PyCall, _ExtraFields_PyCCall)):
            parent = node.parent
            while parent is not None:
                if isinstance(parent.extra_fields, _ExtraFields_PyCall):
                    break
                parent = parent.parent

            def to_string(frame_state):
                if False:
                    return 10
                return f'{frame_state.file_name}(...): {frame_state.function_name}'
            if parent:
                parent_name = to_string(parent.extra_fields.callsite)
                caller_name = to_string(extra_fields.caller)
                assert parent_name == caller_name, f'{parent_name} vs. {caller_name}'

@unittest.skipIf(IS_ARM64, 'Not working on ARM')
class TestProfilerTree(TestCase):

    def assertTreesMatch(self, actual: str, expected: str, allow_failure: bool=False):
        if False:
            i = 10
            return i + 15
        if not expecttest.ACCEPT:
            actual = actual.ljust(len(expected))
        self.maxDiff = None
        replicate = getattr(self, 'tree_replicate', None)
        self.assertIsNotNone(replicate, 'Please annotate test with `@ProfilerTree.test`')
        if replicate:
            self.assertEqual(actual, expected)
        else:
            try:
                self.assertExpectedInline(actual, expected, skip=1)
            except AssertionError as e:
                if allow_failure:
                    self.tree_replicate = None
                    msg = traceback.format_exception_only(type(e), e)[0]
                    print(msg.split('AssertionError:')[-1])
                else:
                    raise

    @ProfilerTree.test
    def test_profiler_experimental_tree(self):
        if False:
            for i in range(10):
                print('nop')
        (t1, t2) = (torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True))
        with torch.profiler.profile() as p:
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = (y - z) ** 2
            loss.backward()
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            aten::add\n            aten::ones\n              aten::empty\n              aten::fill_\n            aten::sub\n            aten::pow\n              aten::result_type\n              aten::to\n            aten::ones_like\n              aten::empty_like\n                aten::empty_strided\n              aten::fill_\n            autograd::engine::evaluate_function: PowBackward0\n              PowBackward0\n                aten::pow\n                  aten::result_type\n                  aten::to\n                  aten::copy_\n                aten::mul\n                  aten::mul\n                    aten::to\n                      aten::_to_copy\n                        aten::empty_strided\n                        aten::copy_\n                aten::mul\n            autograd::engine::evaluate_function: SubBackward0\n              SubBackward0\n                aten::neg\n            autograd::engine::evaluate_function: AddBackward0\n              AddBackward0\n            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n              torch::autograd::AccumulateGrad\n                aten::new_empty_strided\n                  aten::empty_strided\n                aten::copy_\n            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n              torch::autograd::AccumulateGrad\n                aten::detach\n                  detach')

    @ProfilerTree.test
    def test_profiler_experimental_tree_with_record_function(self):
        if False:
            for i in range(10):
                print('nop')
        with torch.profiler.profile() as p:
            with torch.autograd.profiler.record_function('Top level Annotation'):
                with torch.autograd.profiler.record_function('First Annotation'):
                    x = torch.ones((1,), requires_grad=True)
                _ = torch.autograd.profiler.record_function('Second Annotation').__enter__()
                y = x + 1
                with torch.autograd.profiler.record_function('Third Annotation'):
                    y.backward()
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            Top level Annotation\n              First Annotation\n                aten::ones\n                  aten::empty\n                  aten::fill_\n              Second Annotation\n                aten::add\n                  aten::to\n                    aten::_to_copy\n                      aten::empty_strided\n                      aten::copy_\n                Third Annotation\n                  aten::ones_like\n                    aten::empty_like\n                      aten::empty_strided\n                    aten::fill_\n                  autograd::engine::evaluate_function: AddBackward0\n                    AddBackward0\n                  autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n                    torch::autograd::AccumulateGrad\n                      aten::new_empty_strided\n                        aten::empty_strided\n                      aten::copy_')

    @ProfilerTree.test
    def test_profiler_experimental_tree_with_memory(self):
        if False:
            while True:
                i = 10
        (t1, t2) = (torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True))
        with torch.profiler.profile(profile_memory=True) as p:
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = (y - z) ** 2
            loss.backward()
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            aten::add\n              [memory]\n            aten::ones\n              aten::empty\n                [memory]\n              aten::fill_\n            aten::sub\n              [memory]\n            aten::pow\n              aten::result_type\n              aten::to\n              [memory]\n            aten::ones_like\n              aten::empty_like\n                aten::empty_strided\n                  [memory]\n              aten::fill_\n            autograd::engine::evaluate_function: PowBackward0\n              PowBackward0\n                aten::pow\n                  aten::result_type\n                  aten::to\n                  [memory]\n                  aten::copy_\n                aten::mul\n                  [memory]\n                  aten::mul\n                    aten::to\n                      aten::_to_copy\n                        aten::empty_strided\n                          [memory]\n                        aten::copy_\n                    [memory]\n                    [memory]\n                  [memory]\n                aten::mul\n                  [memory]\n                [memory]\n                [memory]\n              [memory]\n            autograd::engine::evaluate_function: SubBackward0\n              SubBackward0\n                aten::neg\n                  [memory]\n              [memory]\n            autograd::engine::evaluate_function: AddBackward0\n              AddBackward0\n            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n              torch::autograd::AccumulateGrad\n                aten::new_empty_strided\n                  aten::empty_strided\n                    [memory]\n                aten::copy_\n            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n              torch::autograd::AccumulateGrad\n                aten::detach\n                  detach\n            [memory]')

    @unittest.skipIf(TEST_WITH_CROSSREF, 'crossref intercepts calls and changes the callsite.')
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_memory_and_stack(self):
        if False:
            for i in range(10):
                print('nop')
        (t1, t2) = (torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True))
        with torch.profiler.profile(with_stack=True, profile_memory=True) as p:
            z = torch.add(t1, t2)
            y = torch.ones(1)
            loss = torch.pow(y - z, 2)
            loss.backward()
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            test_profiler_tree.py(...): test_profiler_experimental_tree_with_memory_and_stack\n              torch/profiler/profiler.py(...): __enter__\n                ...\n              <built-in method add of type object at 0xXXXXXXXXXXXX>\n                aten::add\n                  [memory]\n              <built-in method ones of type object at 0xXXXXXXXXXXXX>\n                aten::ones\n                  aten::empty\n                    [memory]\n                  aten::fill_\n              aten::sub\n                [memory]\n              <built-in method pow of type object at 0xXXXXXXXXXXXX>\n                aten::pow\n                  aten::result_type\n                  aten::to\n                  [memory]\n              torch/_tensor.py(...): backward\n                <built-in function _has_torch_function_unary>\n                torch/autograd/__init__.py(...): backward\n                  <built-in method _are_functorch_transforms_active of PyCapsule object at 0xXXXXXXXXXXXX>\n                  <built-in function isinstance>\n                  <built-in function isinstance>\n                  <built-in function len>\n                  torch/autograd/__init__.py(...): _tensor_or_tensors_to_tuple\n                  torch/autograd/__init__.py(...): _make_grads\n                    <built-in function isinstance>\n                    <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>\n                    <built-in method ones_like of type object at 0xXXXXXXXXXXXX>\n                      aten::ones_like\n                        aten::empty_like\n                          aten::empty_strided\n                            [memory]\n                        aten::fill_\n                    <built-in method append of list object at 0xXXXXXXXXXXXX>\n                  <built-in method run_backward of torch._C._EngineBase object at 0xXXXXXXXXXXXX>\n                    autograd::engine::evaluate_function: PowBackward0\n                      PowBackward0\n                        aten::pow\n                          aten::result_type\n                          aten::to\n                          [memory]\n                          aten::copy_\n                        aten::mul\n                          [memory]\n                          aten::mul\n                            aten::to\n                              aten::_to_copy\n                                aten::empty_strided\n                                  [memory]\n                                aten::copy_\n                            [memory]\n                            [memory]\n                          [memory]\n                        aten::mul\n                          [memory]\n                        [memory]\n                        [memory]\n                      [memory]\n                    autograd::engine::evaluate_function: SubBackward0\n                      SubBackward0\n                        aten::neg\n                          [memory]\n                      [memory]\n                    autograd::engine::evaluate_function: AddBackward0\n                      AddBackward0\n                    autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n                      torch::autograd::AccumulateGrad\n                        aten::new_empty_strided\n                          aten::empty_strided\n                            [memory]\n                        aten::copy_\n                    autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n                      torch::autograd::AccumulateGrad\n                        aten::detach\n                          detach\n                [memory]\n              torch/profiler/profiler.py(...): __exit__\n                torch/profiler/profiler.py(...): stop\n                  ...')

    @unittest.skipIf(TEST_WITH_CROSSREF, 'crossref intercepts calls and changes the callsite.')
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_modules(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.layers = [torch.nn.ReLU(), torch.nn.Linear(1, 1), torch.nn.ReLU()]

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                if False:
                    i = 10
                    return i + 15
                for l in self.layers:
                    x = l(x)
                return x
        model = MyModule()
        with torch.profiler.profile(with_stack=True) as p:
            for _ in range(2):
                model(torch.ones((1,)))
        self.maxDiff = None
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_modules\n              torch/profiler/profiler.py(...): __enter__\n                ...\n              <built-in method ones of type object at 0xXXXXXXXXXXXX>\n                aten::ones\n                  aten::empty\n                  aten::fill_\n              nn.Module: MyModule_0\n                torch/nn/modules/module.py(...): _call_impl\n                  <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                  test_profiler_tree.py(...): forward\n                    nn.Module: ReLU_0\n                      torch/nn/modules/module.py(...): _call_impl\n                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                        torch/nn/modules/activation.py(...): forward\n                          torch/nn/functional.py(...): relu\n                            <built-in function _has_torch_function_unary>\n                            <built-in method relu of type object at 0xXXXXXXXXXXXX>\n                              aten::relu\n                                aten::clamp_min\n                    nn.Module: Linear_0\n                      torch/nn/modules/module.py(...): _call_impl\n                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                        torch/nn/modules/linear.py(...): forward\n                          torch/nn/modules/module.py(...): __getattr__\n                          torch/nn/modules/module.py(...): __getattr__\n                          <built-in function linear>\n                            aten::linear\n                              aten::reshape\n                                aten::view\n                              aten::t\n                                aten::transpose\n                                  aten::as_strided\n                              aten::addmm\n                                aten::expand\n                                  aten::as_strided\n                                aten::copy_\n                                aten::resolve_conj\n                                aten::resolve_conj\n                                aten::resolve_conj\n                              aten::view\n                    nn.Module: ReLU_1\n                      torch/nn/modules/module.py(...): _call_impl\n                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                        torch/nn/modules/activation.py(...): forward\n                          torch/nn/functional.py(...): relu\n                            <built-in function _has_torch_function_unary>\n                            <built-in method relu of type object at 0xXXXXXXXXXXXX>\n                              aten::relu\n                                aten::clamp_min\n              <built-in method ones of type object at 0xXXXXXXXXXXXX>\n                aten::ones\n                  aten::empty\n                  aten::fill_\n              nn.Module: MyModule_0\n                torch/nn/modules/module.py(...): _call_impl\n                  <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                  test_profiler_tree.py(...): forward\n                    nn.Module: ReLU_0\n                      torch/nn/modules/module.py(...): _call_impl\n                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                        torch/nn/modules/activation.py(...): forward\n                          torch/nn/functional.py(...): relu\n                            <built-in function _has_torch_function_unary>\n                            <built-in method relu of type object at 0xXXXXXXXXXXXX>\n                              aten::relu\n                                aten::clamp_min\n                    nn.Module: Linear_0\n                      torch/nn/modules/module.py(...): _call_impl\n                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                        torch/nn/modules/linear.py(...): forward\n                          torch/nn/modules/module.py(...): __getattr__\n                          torch/nn/modules/module.py(...): __getattr__\n                          <built-in function linear>\n                            aten::linear\n                              aten::reshape\n                                aten::view\n                              aten::t\n                                aten::transpose\n                                  aten::as_strided\n                              aten::addmm\n                                aten::expand\n                                  aten::as_strided\n                                aten::copy_\n                                aten::resolve_conj\n                                aten::resolve_conj\n                                aten::resolve_conj\n                              aten::view\n                    nn.Module: ReLU_1\n                      torch/nn/modules/module.py(...): _call_impl\n                        <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                        torch/nn/modules/activation.py(...): forward\n                          torch/nn/functional.py(...): relu\n                            <built-in function _has_torch_function_unary>\n                            <built-in method relu of type object at 0xXXXXXXXXXXXX>\n                              aten::relu\n                                aten::clamp_min\n              torch/profiler/profiler.py(...): __exit__\n                torch/profiler/profiler.py(...): stop\n                  ...')

    @unittest.skipIf(TEST_WITH_CROSSREF, 'crossref intercepts calls and changes the callsite.')
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_torch_function(self):
        if False:
            while True:
                i = 10
        x = TorchFunctionTensor(torch.ones((1,)))
        y = torch.ones((1,))
        torch.add(x, y)
        with torch.profiler.profile(with_stack=True) as p:
            torch.add(x, y)
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_torch_function\n              torch/profiler/profiler.py(...): __enter__\n                ...\n              <built-in method add of type object at 0xXXXXXXXXXXXX>\n                test_profiler_tree.py(...): __torch_function__\n                  torch/_tensor.py(...): __torch_function__\n                    <built-in function all>\n                      torch/_tensor.py(...): <genexpr>\n                        <built-in function issubclass>\n                      torch/_tensor.py(...): <genexpr>\n                    <built-in method add of type object at 0xXXXXXXXXXXXX>\n                      aten::add\n                    torch/_tensor.py(...): _convert\n                      <built-in function isinstance>\n                      <built-in function isinstance>\n                      <built-in method as_subclass of Tensor object at 0xXXXXXXXXXXXX>\n                        aten::alias\n                      <built-in function isinstance>\n              torch/profiler/profiler.py(...): __exit__\n                torch/profiler/profiler.py(...): stop\n                  ...')

    @unittest.skipIf(TEST_WITH_CROSSREF, 'crossref intercepts calls and changes the callsite.')
    @ProfilerTree.test
    def test_profiler_experimental_tree_with_stack_and_torch_dispatch(self):
        if False:
            i = 10
            return i + 15
        x = TorchDispatchTensor(torch.ones((1,)))
        y = torch.ones((1,))
        with torch.profiler.profile(with_stack=True) as p:
            x + y
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            test_profiler_tree.py(...): test_profiler_experimental_tree_with_stack_and_torch_dispatch\n              torch/profiler/profiler.py(...): __enter__\n                ...\n              aten::add\n                test_profiler_tree.py(...): __torch_dispatch__\n                  torch/utils/_pytree.py(...): tree_map\n                    ...\n                  torch/utils/_pytree.py(...): tree_map\n                    ...\n                  torch/_ops.py(...): __call__\n                    <built-in method  of PyCapsule object at 0xXXXXXXXXXXXX>\n                      aten::add\n                  torch/utils/_pytree.py(...): tree_map\n                    ...\n              torch/profiler/profiler.py(...): __exit__\n                torch/profiler/profiler.py(...): stop\n                  ...')

    @unittest.skip('https://github.com/pytorch/pytorch/issues/83606')
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is required')
    @ProfilerTree.test
    def test_profiler_experimental_tree_cuda(self):
        if False:
            i = 10
            return i + 15
        with torch.profiler.profile(profile_memory=True) as p:
            weight = torch.ones(1, device='cuda', requires_grad=True)
            x = torch.ones(1, device='cuda')
            y = torch.add(weight, x)
            loss = torch.pow(y, 2)
            loss.backward()
            torch.optim.SGD([weight], lr=0.01, momentum=0.9).step()
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            aten::ones\n              aten::empty\n                [memory]\n              aten::fill_\n                cudaLaunchKernel\n                  void at::native::vectorized_elementwise_kernel<...>(...)\n            aten::ones\n              aten::empty\n                [memory]\n              aten::fill_\n                cudaLaunchKernel\n                  void at::native::vectorized_elementwise_kernel<...>(...)\n            aten::add\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            aten::pow\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              aten::result_type\n              aten::to\n              [memory]\n            aten::ones_like\n              aten::empty_like\n                aten::empty_strided\n                  [memory]\n              aten::fill_\n                cudaLaunchKernel\n                  void at::native::vectorized_elementwise_kernel<...>(...)\n            autograd::engine::evaluate_function: PowBackward0\n              PowBackward0\n                aten::pow\n                  aten::result_type\n                  aten::to\n                  [memory]\n                  aten::copy_\n                    cudaMemcpyAsync\n                      Memcpy DtoD (Device -> Device)\n                aten::mul\n                  [memory]\n                  aten::mul\n                    cudaLaunchKernel\n                      void at::native::vectorized_elementwise_kernel<...>(...)\n                    [memory]\n                  [memory]\n                aten::mul\n                  cudaLaunchKernel\n                    void at::native::vectorized_elementwise_kernel<...>(...)\n                  [memory]\n                [memory]\n                [memory]\n            autograd::engine::evaluate_function: AddBackward0\n              AddBackward0\n            autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n              torch::autograd::AccumulateGrad\n                aten::detach\n                  detach\n            [memory]\n            aten::zeros\n              aten::zeros\n                aten::empty\n                  [memory]\n                aten::zero_\n            Optimizer.step#SGD.step\n              aten::empty\n                [memory]\n              [memory]\n              [memory]\n              aten::clone\n                aten::empty_strided\n                  [memory]\n                aten::copy_\n                  cudaMemcpyAsync\n                    Memcpy DtoD (Device -> Device)\n              aten::detach\n                detach\n              aten::add_\n                cudaLaunchKernel\n                  void at::native::vectorized_elementwise_kernel<...>(...)\n            [memory]', allow_failure=ALLOW_CUDA_FAILURE)

    @unittest.skip('https://github.com/pytorch/pytorch/issues/83606')
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is required')
    @ProfilerTree.test
    def test_profiler_experimental_tree_cuda_with_stream(self):
        if False:
            while True:
                i = 10
        streams = [torch.cuda.Stream() for _ in range(3)]
        results = []
        with torch.profiler.profile(profile_memory=True) as p:
            x = torch.ones((4, 4), device='cuda')
            for stream in streams:
                with torch.cuda.stream(stream):
                    results.append(torch.tanh(x) - x)
        del results
        for s in streams:
            torch.cuda.current_stream().wait_stream(s)
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            aten::ones\n              aten::empty\n                [memory]\n              aten::fill_\n                cudaLaunchKernel\n                  void at::native::vectorized_elementwise_kernel<...>(...)\n            aten::tanh\n              cudaMalloc\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            aten::sub\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            [memory]\n            aten::tanh\n              cudaMalloc\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            aten::sub\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            [memory]\n            aten::tanh\n              cudaMalloc\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            aten::sub\n              cudaLaunchKernel\n                void at::native::vectorized_elementwise_kernel<...>(...)\n              [memory]\n            [memory]', allow_failure=ALLOW_CUDA_FAILURE)

    @unittest.skip('https://github.com/pytorch/pytorch/issues/83606')
    @unittest.skipIf(TEST_WITH_CROSSREF, 'crossref intercepts calls and changes the callsite.')
    @unittest.skipIf(not torch.cuda.is_available(), 'CUDA is required')
    @ProfilerTree.test
    def test_profiler_experimental_tree_cuda_detailed(self):
        if False:
            i = 10
            return i + 15
        model = torch.nn.modules.Linear(1, 1, device='cuda')
        model.train()
        opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

        def step():
            if False:
                for i in range(10):
                    print('nop')
            x = torch.ones((1, 1), device='cuda')
            loss = model(x)
            loss.backward()
            opt.step()
        for _ in range(3):
            step()
        with torch.profiler.profile(profile_memory=True, with_stack=True) as p:
            step()
        self.assertTreesMatch(ProfilerTree.format(p.profiler, 12), '            test_profiler_tree.py(...): test_profiler_experimental_tree_cuda_detailed\n              torch/profiler/profiler.py(...): __enter__\n                ...\n              test_profiler_tree.py(...): step\n                <built-in method ones of type object at 0xXXXXXXXXXXXX>\n                  aten::ones\n                    aten::empty\n                      [memory]\n                    aten::fill_\n                      cudaLaunchKernel\n                        void at::native::vectorized_elementwise_kernel<...>(...)\n                nn.Module: Linear_0\n                  <built-in method _get_tracing_state of PyCapsule object at 0xXXXXXXXXXXXX>\n                  torch/nn/modules/linear.py(...): forward\n                    torch/nn/modules/module.py(...): __getattr__\n                    torch/nn/modules/module.py(...): __getattr__\n                    <built-in function linear>\n                      aten::linear\n                        aten::t\n                          aten::transpose\n                            aten::as_strided\n                        aten::addmm\n                          cudaMemcpyAsync\n                            Memcpy DtoD (Device -> Device)\n                          cudaLaunchKernel\n                            void ..._kernel<...>(...)\n                          [memory]\n                          aten::expand\n                            aten::as_strided\n                torch/_tensor.py(...): backward\n                  <built-in function _has_torch_function_unary>\n                  torch/autograd/__init__.py(...): backward\n                    <built-in function isinstance>\n                    <built-in function isinstance>\n                    <built-in function len>\n                    torch/autograd/__init__.py(...): _tensor_or_tensors_to_tuple\n                    torch/autograd/__init__.py(...): _make_grads\n                      <built-in function isinstance>\n                      <built-in method numel of Tensor object at 0xXXXXXXXXXXXX>\n                      <built-in method ones_like of type object at 0xXXXXXXXXXXXX>\n                        aten::ones_like\n                          aten::empty_like\n                            aten::empty_strided\n                              [memory]\n                          aten::fill_\n                            cudaLaunchKernel\n                              void at::native::vectorized_elementwise_kernel<...>(...)\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                    <built-in method run_backward of torch._C._EngineBase object at 0xXXXXXXXXXXXX>\n                      autograd::engine::evaluate_function: AddmmBackward0\n                        AddmmBackward0\n                          aten::t\n                            aten::transpose\n                              aten::as_strided\n                          aten::mm\n                            cudaLaunchKernel\n                              void ..._kernel<...>(...)\n                            [memory]\n                          aten::t\n                            aten::transpose\n                              aten::as_strided\n                        aten::sum\n                          aten::sum\n                            cudaLaunchKernel\n                              void at::native::reduce_kernel<...>(...)\n                            [memory]\n                        aten::view\n                          aten::view\n                      autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n                        torch::autograd::AccumulateGrad\n                          aten::add_\n                            cudaLaunchKernel\n                              void at::native::vectorized_elementwise_kernel<...>(...)\n                          [memory]\n                      autograd::engine::evaluate_function: TBackward0\n                        TBackward0\n                          aten::t\n                            aten::transpose\n                              aten::as_strided\n                      autograd::engine::evaluate_function: torch::autograd::AccumulateGrad\n                        torch::autograd::AccumulateGrad\n                          aten::add_\n                            cudaLaunchKernel\n                              void at::native::vectorized_elementwise_kernel<...>(...)\n                          [memory]\n                  [memory]\n                torch/optim/optimizer.py(...): wrapper\n                  <built-in method format of str object at 0xXXXXXXXXXXXX>\n                  torch/autograd/profiler.py(...): __init__\n                    <built-in method zeros of type object at 0xXXXXXXXXXXXX>\n                      aten::zeros\n                        aten::zeros\n                          aten::empty\n                            [memory]\n                          aten::zero_\n                  torch/autograd/profiler.py(...): __enter__\n                    torch/_ops.py(...): __call__\n                      <built-in method _record_function_enter of PyCapsule object at 0xXXXXXXXXXXXX>\n                        Optimizer.step#SGD.step\n                          aten::empty\n                            [memory]\n                          [memory]\n                    [memory]\n                  torch/optim/optimizer.py(...): _use_grad\n                    <built-in function is_grad_enabled>\n                    torch/autograd/grad_mode.py(...): __init__\n                      <built-in function is_grad_enabled>\n                      <built-in function _set_grad_enabled>\n                    torch/optim/sgd.py(...): step\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                      torch/_tensor.py(...): __hash__\n                        <built-in function id>\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                      torch/_tensor.py(...): __hash__\n                        <built-in function id>\n                      <built-in method append of list object at 0xXXXXXXXXXXXX>\n                      torch/optim/sgd.py(...): sgd\n                        torch/optim/sgd.py(...): _single_tensor_sgd\n                          <built-in method mul_ of Tensor object at 0xXXXXXXXXXXXX>\n                            [memory]\n                            aten::mul_\n                              cudaLaunchKernel\n                                void at::native::vectorized_elementwise_kernel<...>(...)\n                            [memory]\n                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>\n                            aten::add_\n                              cudaLaunchKernel\n                                void at::native::vectorized_elementwise_kernel<...>(...)\n                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>\n                            aten::add_\n                              cudaLaunchKernel\n                                void at::native::vectorized_elementwise_kernel<...>(...)\n                          <built-in method mul_ of Tensor object at 0xXXXXXXXXXXXX>\n                            [memory]\n                            aten::mul_\n                              cudaLaunchKernel\n                                void at::native::vectorized_elementwise_kernel<...>(...)\n                            [memory]\n                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>\n                            aten::add_\n                              cudaLaunchKernel\n                                void at::native::vectorized_elementwise_kernel<...>(...)\n                          <built-in method add_ of Tensor object at 0xXXXXXXXXXXXX>\n                            aten::add_\n                              cudaLaunchKernel\n                                void at::native::vectorized_elementwise_kernel<...>(...)\n                      torch/_tensor.py(...): __hash__\n                        <built-in function id>\n                      torch/_tensor.py(...): __hash__\n                        <built-in function id>\n                    torch/autograd/grad_mode.py(...): __init__\n                      <built-in function is_grad_enabled>\n                      <built-in function _set_grad_enabled>\n                  torch/autograd/profiler.py(...): __exit__\n                    torch/_ops.py(...): __call__\n                      <built-in method _record_function_exit of PyCapsule object at 0xXXXXXXXXXXXX>\n              [memory]\n              [memory]\n              torch/profiler/profiler.py(...): __exit__\n                torch/profiler/profiler.py(...): stop\n                  torch/profiler/profiler.py(...): _transit_action\n                    <built-in method get of dict object at 0xXXXXXXXXXXXX>\n                      enum.py(...): __hash__\n                        <built-in function hash>\n                    ...', allow_failure=ALLOW_CUDA_FAILURE)
if __name__ == '__main__':
    run_tests()