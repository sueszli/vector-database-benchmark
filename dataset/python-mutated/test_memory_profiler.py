import functools
import gc
import itertools as it
import textwrap
from typing import Callable, Dict, Iterator, List, Optional, Tuple
import torch
from torch._C._profiler import _EventType, _TensorMetadata
from torch.profiler import _memory_profiler, _utils
from torch.testing._internal.common_utils import run_tests, skipIfTorchDynamo, TestCase
from torch.utils import _pytree as pytree
profile = functools.partial(torch.profiler.profile, record_shapes=True, profile_memory=True, with_stack=True)

@skipIfTorchDynamo('TorchDynamo removes profiler altogether.')
class TestMemoryProfiler(TestCase):

    def test_config_check(self) -> None:
        if False:
            return 10
        with torch.profiler.profile() as prof:
            pass
        pattern = 'record_shapes=True, profile_memory=True, with_stack=True'
        with self.assertRaisesRegex(ValueError, pattern):
            prof._memory_profile()
        with torch.profiler.profile(record_shapes=True, with_stack=True) as prof:
            pass
        pattern = '^profile_memory=True required for memory profiling\\.$'
        with self.assertRaisesRegex(ValueError, pattern):
            prof._memory_profile()
        with profile() as prof:
            pass
        self.assertIsInstance(prof._memory_profile(), _memory_profiler.MemoryProfile)

class ScaleLayer(torch.nn.Module):

    def __init__(self) -> None:
        if False:
            print('Hello World!')
        super().__init__()
        self.scale = torch.nn.Parameter(torch.rand(()), requires_grad=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if False:
            i = 10
            return i + 15
        return x * self.scale

class LazyLinear(torch.nn.Module):

    def __init__(self, in_features: int, out_features: int):
        if False:
            while True:
                i = 10
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

    def forward(self, x) -> torch.Tensor:
        if False:
            for i in range(10):
                print('nop')
        if getattr(self, 'weight', None) is None:
            self.weight = torch.nn.Parameter(torch.empty((self.out_features, self.in_features)))
            self.bias = torch.nn.Parameter(torch.empty(self.out_features))
        return torch.nn.functional.linear(x, self.weight, self.bias)

class RecordInputOutputDispatchMode(torch.utils._python_dispatch.TorchDispatchMode):

    def __init__(self):
        if False:
            while True:
                i = 10
        self.results = []

    def mark_region(self, name: str):
        if False:
            while True:
                i = 10
        self.results.append((name, (), ()))

    @staticmethod
    def flat_ids(args):
        if False:
            while True:
                i = 10
        flat_args = pytree.tree_leaves(args)
        return tuple(((t._cdata, t.storage().data_ptr()) for t in flat_args if isinstance(t, torch.Tensor) and t.storage()))

    def __torch_dispatch__(self, func, types, args=..., kwargs=None):
        if False:
            i = 10
            return i + 15
        args = args or []
        kwargs = kwargs or {}
        flat_inputs = self.flat_ids(args) + self.flat_ids(kwargs)
        out = func(*args, **kwargs)
        flat_outputs = self.flat_ids(out)
        if (flat_inputs or flat_outputs) and '_record_function_enter' not in func.name():
            self.results.append((func.name(), flat_inputs, flat_outputs))
        return out

@skipIfTorchDynamo('TorchDynamo changes Python calls that memory profiling relies on.')
class TestIdentifyGradients(TestCase):

    def gradient_detected(self, prof: torch.profiler.profile, ctx: _EventType, grad_tensor: torch.Tensor, parameter: Optional[torch.Tensor]=None) -> None:
        if False:
            print('Hello World!')

        def key_matches_tensor(key, tensor) -> bool:
            if False:
                i = 10
                return i + 15
            if tensor is None:
                return True
            if key is None:
                return False
            return tensor.storage().data_ptr() == key.storage.ptr
        tree = prof.profiler.kineto_results.experimental_event_tree()
        for node in _utils.traverse_dfs(tree):
            for (p_key, p_grad_key) in _memory_profiler.extract_gradients(node):
                if node.tag == ctx and key_matches_tensor(p_grad_key, grad_tensor):
                    if parameter is None:
                        return True
                    elif p_key is not None:
                        self.assertTrue(key_matches_tensor(p_key, parameter))
                        return True
        return False

    def assertGradientDetected(self, name: str, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        self.assertTrue(self.gradient_detected(*args, **kwargs), f'Failed to identify gradient `{name}` from profile.')

    def assertOnlyGradients(self, prof: torch.profiler.profile, tensors: Iterator[torch.Tensor]) -> None:
        if False:
            i = 10
            return i + 15
        allowed_set = {t.storage().data_ptr() for t in tensors}
        tree = prof.profiler.kineto_results.experimental_event_tree()
        for node in _utils.traverse_dfs(tree):
            for (_, p_grad_key) in _memory_profiler.extract_gradients(node):
                self.assertTrue(p_grad_key.storage.ptr in allowed_set, f'Tensor wrongly marked as gradient: {node.name}: {p_grad_key}')

    def test_extract_gradients_low_level(self) -> None:
        if False:
            while True:
                i = 10
        x = torch.ones((1,))
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        def check(cold_start: bool):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(w0.grad is None, cold_start)
            self.assertEqual(w1.grad is None, cold_start)
            with profile() as prof:
                z = x.expand(4) * w0
                (z * w1).sum().backward()
            self.assertGradientDetected('w0', prof, _EventType.TorchOp, w0.grad)
            self.assertGradientDetected('w1', prof, _EventType.TorchOp, w1.grad)
            self.assertOnlyGradients(prof, (w0.grad, w1.grad))
        check(cold_start=True)
        check(cold_start=False)

    def test_extract_gradients_from_module(self) -> None:
        if False:
            print('Hello World!')
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), ScaleLayer())
        named_parameters = dict(model.named_parameters())
        self.assertEqual(len(named_parameters), 3)

        def assert_only_gradients(prof: torch.profiler.profile):
            if False:
                i = 10
                return i + 15
            gradients = tuple((i.grad for i in named_parameters.values()))
            self.assertFalse(any((i is None for i in gradients)))
            self.assertOnlyGradients(prof, gradients)

        def check(cold_start: bool):
            if False:
                while True:
                    i = 10
            x = torch.ones((2, 2))
            with profile() as prof:
                model(x).sum().backward()
            for (name, p) in named_parameters.items():
                self.assertNotEqual(self.gradient_detected(prof, _EventType.PyCall, p.grad, p), cold_start, name)
                self.assertGradientDetected(name, prof, _EventType.TorchOp, p.grad)
            assert_only_gradients(prof)
            with profile() as prof:
                model(torch.ones((2, 2)))
            for (name, p) in named_parameters.items():
                self.assertGradientDetected(name, prof, _EventType.PyCall, p.grad, p)
                self.assertFalse(self.gradient_detected(prof, _EventType.TorchOp, p.grad), name)
            assert_only_gradients(prof)
        check(cold_start=True)
        check(cold_start=False)

    def _test_extract_gradients_from_optimizer(self, set_to_none: bool) -> None:
        if False:
            return 10
        x = torch.ones((1,))
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)
        optimizer = torch.optim.SGD((w0, w1), lr=0.1, momentum=0.9)

        def check(cold_start: bool):
            if False:
                return 10
            self.assertEqual(w0.grad is None, cold_start)
            self.assertEqual(w1.grad is None, cold_start)
            with profile() as prof:
                optimizer.zero_grad(set_to_none=set_to_none)
                z = x.expand(4) * w0
                (z * w1).sum().backward()
                optimizer.step()
            self.assertGradientDetected('w0', prof, _EventType.PyCall, w0.grad, w0)
            self.assertGradientDetected('w1', prof, _EventType.PyCall, w1.grad, w1)
            self.assertGradientDetected('w0', prof, _EventType.TorchOp, w0.grad)
            self.assertGradientDetected('w1', prof, _EventType.TorchOp, w1.grad)
            self.assertOnlyGradients(prof, (w0.grad, w1.grad))
            with profile() as prof:
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=set_to_none)
                    z = x.expand(4) * w0
                    (z * w1).sum().backward()
                    optimizer.step()
            self.assertNotEqual(self.gradient_detected(prof, _EventType.PyCall, w0.grad, w0), set_to_none)
            self.assertNotEqual(self.gradient_detected(prof, _EventType.PyCall, w1.grad, w1), set_to_none)
            if set_to_none:
                with self.assertRaisesRegex(AssertionError, 'Tensor wrongly marked'):
                    self.assertOnlyGradients(prof, (w0.grad, w1.grad))
        check(cold_start=True)
        check(cold_start=False)

    def test_extract_gradients_from_optimizer(self) -> None:
        if False:
            print('Hello World!')
        self._test_extract_gradients_from_optimizer(set_to_none=False)

    def test_extract_gradients_from_optimizer_set_to_none(self) -> None:
        if False:
            return 10
        self._test_extract_gradients_from_optimizer(set_to_none=True)

    def test_extract_gradients_from_module_and_optimizer(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Sequential(torch.nn.Linear(2, 1), ScaleLayer())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
        with profile() as prof:
            model(torch.ones((2, 2))).sum().backward()
            optimizer.step()
        self.assertGradientDetected('weight', prof, _EventType.PyCall, model[0].weight.grad, model[0].weight)

@skipIfTorchDynamo('TorchDynamo removes profiler altogether.')
class TestDataFlow(TestCase):

    def setUp(self) -> None:
        if False:
            while True:
                i = 10
        super().setUp()
        self.maxDiff = None

    @staticmethod
    def formatSchemas(prof: torch.profiler.profile, indent: int=12) -> Tuple[Tuple[str, Tuple[bool, ...]], ...]:
        if False:
            return 10
        tree = prof.profiler.kineto_results.experimental_event_tree()
        out: List[Tuple[str, Tuple[bool, ...]]] = []
        for node in _utils.traverse_dfs(tree):
            if node.tag == _EventType.TorchOp:
                e = node.extra_fields
                schemas = _memory_profiler.SchemaMatcher.match_schemas(e)
                name = node.name
                if len(schemas) == 1:
                    name = f'{name}.{schemas[0].overload_name}'
                elif len(schemas) > 1:
                    name = f"{name}.{{{', '.join((s.overload_name for s in schemas))}}}"
                out.append((name, _memory_profiler.SchemaMatcher.inputs_are_mutable(e)))
        return tuple(out)

    @staticmethod
    def _run_and_format_data_flow(inputs: Dict[str, torch.Tensor], f: Callable[..., Optional[Dict[str, torch.Tensor]]], indent: int=12) -> str:
        if False:
            return 10
        with profile() as prof:
            outputs = f(**inputs) or {}
            gc.collect()
        memory_profile = prof._memory_profile()
        graph = memory_profile._data_flow_graph
        storage_to_id = {key.storage.ptr: key.id for key in graph._active_version}
        lines: List[str] = []
        for (name, t) in it.chain(inputs.items(), outputs.items()):
            lines.append(f"{name + ':':<8} T{storage_to_id[t.storage().data_ptr()]}")
            if t.grad is not None:
                grad_id = storage_to_id[t.grad.storage().data_ptr()]
                lines.append(f"{name + '.grad:':<9} T{grad_id}")
        if lines:
            lines.append('')
        for node in graph.flow_nodes:
            destroyed = {k for (k, v) in node._edges.items() if v.is_deletion}
            inputs: List[str] = []
            for (key, (_, v)) in node.inputs.items():
                inputs.append(f"T{key.id}(v{v}{('*' if key in destroyed else '')})")
            outputs = [f'T{key.id}(v{v})' for (key, v) in node.outputs.items()]
            if inputs or outputs:
                event_name = node._event.name.replace('torch::autograd::', '')
                lines.append(f"{event_name:<25} {', '.join(inputs):<15}  ->  {', '.join(outputs)}")
        return textwrap.indent('\n'.join([l.rstrip() for l in lines]), ' ' * indent)

    def test_match_schemas(self) -> None:
        if False:
            print('Hello World!')
        with profile() as prof:
            x = torch.ones((1,)).mul(2).add_(2)
            _ = torch.sin(x, out=torch.empty_like(x))
        self.assertEqual(self.formatSchemas(prof), (('aten::ones.', (False,) * 5), ('aten::empty.memory_format', (False,) * 6), ('aten::fill_.Scalar', (True, False)), ('aten::mul.Tensor', (False, False)), ('aten::to.dtype', (False,) * 5), ('aten::_to_copy.', (False,) * 7), ('aten::empty_strided.', (False,) * 6), ('aten::copy_.', (True, False, False)), ('aten::add_.Tensor', (True, False, False)), ('aten::to.dtype', (False,) * 5), ('aten::_to_copy.', (False,) * 7), ('aten::empty_strided.', (False,) * 6), ('aten::copy_.', (True, False, False)), ('aten::empty_like.', (False,) * 6), ('aten::empty_strided.', (False,) * 6), ('aten::sin.out', (False, True))))

    def test_match_schemas_backward(self) -> None:
        if False:
            i = 10
            return i + 15
        x = torch.ones((1,))
        w = torch.ones((1,), requires_grad=True)
        with profile() as prof:
            torch.mul(x, w).backward()
        self.assertEqual(self.formatSchemas(prof), (('aten::mul.Tensor', (False, False)), ('aten::ones_like.', (False,) * 6), ('aten::empty_like.', (False,) * 6), ('aten::empty_strided.', (False,) * 6), ('aten::fill_.Scalar', (True, False)), ('autograd::engine::evaluate_function: MulBackward0', ()), ('MulBackward0', (None,)), ('aten::mul.Tensor', (False, False)), ('autograd::engine::evaluate_function: torch::autograd::AccumulateGrad', ()), ('torch::autograd::AccumulateGrad', (None,)), ('aten::detach.', (False,)), ('detach', (None,))))

    def test_match_schemas_tensorlist(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones((1,))
        y = torch.ones((1,))
        with profile() as prof:
            torch.cat([x, y], axis=0)
        self.assertEqual(self.formatSchemas(prof), (('aten::cat.', (False, False)),))

    def test_data_flow_graph_with_annotations(self) -> None:
        if False:
            return 10

        def f(x, y):
            if False:
                print('Hello World!')
            with torch.profiler.record_function('Namespaced::Annotation'):
                with torch.profiler.record_function('My Annotation'):
                    x.zero_()
                    y.zero_()
                    return {'x0': torch.ones_like(x), 'y0': torch.zeros_like(y)}
        inputs = {'x': torch.ones((1,)), 'y': torch.ones((1,))}
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f), '            x:       T0\n            y:       T1\n            x0:      T2\n            y0:      T3\n\n            aten::zero_               T0(v0)           ->  T0(v1)\n            aten::zero_               T1(v0)           ->  T1(v1)\n            aten::ones_like           T0(v1)           ->  T2(v0)\n            aten::zeros_like          T1(v1)           ->  T3(v0)')

    def test_data_flow_graph_non_op_allocations(self) -> None:
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            x.mul(2)
        self.assertExpectedInline(self._run_and_format_data_flow({'x': torch.ones((1,))}, f), '            x:       T1\n\n            [memory]                                   ->  T0(v0)\n            aten::mul                 T0(v0), T1(v0)   ->\n            [memory]                  T0(v0*)          ->')

    def test_data_flow_graph_simple(self) -> None:
        if False:
            print('Hello World!')
        inputs = {'x': torch.ones((25,)), 'y': torch.ones((25,), requires_grad=True)}

        def f0(x, y):
            if False:
                i = 10
                return i + 15
            z = x.mul(y)
            return {'z': z.view_as(z)}

        def f1(x, y):
            if False:
                while True:
                    i = 10
            with torch.no_grad():
                return f0(x, y)
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f0), '            x:       T0\n            y:       T1\n            z:       T2\n\n            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::view_as             T2(v0)           ->')
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f0), '            x:       T0\n            y:       T1\n            z:       T2\n\n            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::view_as             T2(v0)           ->')

    def test_data_flow_graph_simple_inplace(self) -> None:
        if False:
            i = 10
            return i + 15
        inputs = {'x': torch.ones((25,)), 'y': torch.ones((25,), requires_grad=True)}

        def f0(x, y):
            if False:
                print('Hello World!')
            x.mul_(y)

        def f1(x, y):
            if False:
                for i in range(10):
                    print('nop')
            with torch.no_grad():
                return f0(x, y)
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f0), '            x:       T0\n            y:       T1\n\n            aten::mul_                T0(v0), T1(v0)   ->  T0(v1), T2(v0)')
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f1), '            x:       T0\n            y:       T1\n\n            aten::mul_                T0(v0), T1(v0)   ->  T0(v1)')

    def test_data_flow_graph_simple_backward(self) -> None:
        if False:
            while True:
                i = 10
        inputs = {'x': torch.ones((1,)), 'w': torch.ones((1,), requires_grad=True)}
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, lambda x, w: (x * w).sin().backward()), '            x:       T0\n            w:       T1\n            w.grad:   T7\n\n            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::sin                 T2(v0)           ->  T3(v0)\n            aten::ones_like           T3(v0)           ->  T4(v0)\n            SinBackward0              T2(v0), T4(v0)   ->  T6(v0)\n            [memory]                  T2(v0*)          ->\n            MulBackward0              T0(v0), T6(v0)   ->  T7(v0)\n            [memory]                  T6(v0*)          ->\n            AccumulateGrad            T7(v0)           ->\n            [memory]                  T4(v0*)          ->\n            [memory]                  T3(v0*)          ->')

    def test_data_flow_graph_complicated(self) -> None:
        if False:
            print('Hello World!')

        def f():
            if False:
                print('Hello World!')
            x = torch.ones((25,))
            y = x.mul(2).add_(2)
            z = torch.sin(y, out=torch.empty_like(y))
            return {'x': x, 'y': y, 'z': z}
        self.assertExpectedInline(self._run_and_format_data_flow({}, f), '            x:       T0\n            y:       T3\n            z:       T6\n\n            aten::ones                                 ->  T0(v0)\n            [memory]                                   ->  T1(v0)\n            aten::mul                 T0(v0), T1(v0)   ->  T3(v0)\n            [memory]                  T1(v0*)          ->\n            [memory]                                   ->  T4(v0)\n            aten::add_                T3(v0), T4(v0)   ->  T3(v1)\n            [memory]                  T4(v0*)          ->\n            aten::empty_like          T3(v1)           ->  T6(v0)\n            aten::sin                 T3(v1), T6(v0)   ->  T6(v1)')
        with profile() as prof:
            f()
        mul_node = prof._memory_profile()._data_flow_graph.flow_nodes[2]
        self.assertEqual(mul_node._event.name, 'aten::mul')
        self.assertEqual(len(mul_node.intermediates), 1)
        self.assertEqual(mul_node.intermediates[0].id, 2)

    def test_data_flow_graph_stacked(self) -> None:
        if False:
            return 10
        inputs = {'x': torch.ones((25,)), 'w0': torch.ones((1,), requires_grad=True), 'w1': torch.ones((1,), requires_grad=True)}

        def f(x, w0, w1):
            if False:
                for i in range(10):
                    print('nop')
            return x.mul(w0).relu().mul(w1).relu().sum()

        def f_fwd(**kwargs):
            if False:
                print('Hello World!')
            with torch.no_grad():
                return {'loss': f(**kwargs)}

        def f_fwd_bwd(**kwargs):
            if False:
                while True:
                    i = 10
            loss = f(**kwargs)
            loss.backward()
            return {'loss': loss}
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f_fwd), '            x:       T0\n            w0:      T1\n            w1:      T4\n            loss:    T7\n\n            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::relu                T2(v0)           ->  T3(v0)\n            [memory]                  T2(v0*)          ->\n            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)\n            [memory]                  T3(v0*)          ->\n            aten::relu                T5(v0)           ->  T6(v0)\n            [memory]                  T5(v0*)          ->\n            aten::sum                 T6(v0)           ->  T7(v0)\n            [memory]                  T6(v0*)          ->')
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f_fwd_bwd), '            x:       T0\n            w0:      T1\n            w0.grad:  T15\n            w1:      T4\n            w1.grad:  T12\n            loss:    T7\n\n            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::relu                T2(v0)           ->  T3(v0)\n            [memory]                  T2(v0*)          ->\n            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)\n            aten::relu                T5(v0)           ->  T6(v0)\n            [memory]                  T5(v0*)          ->\n            aten::sum                 T6(v0)           ->  T7(v0)\n            aten::ones_like           T7(v0)           ->  T8(v0)\n            SumBackward0              T8(v0)           ->\n            ReluBackward0             T6(v0), T8(v0)   ->  T9(v0)\n            [memory]                  T6(v0*)          ->\n            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T10(v0), T11(v0)\n            aten::sum                 T10(v0)          ->  T12(v0)\n            [memory]                  T10(v0*)         ->\n            [memory]                  T9(v0*)          ->\n            AccumulateGrad            T12(v0)          ->\n            ReluBackward0             T3(v0), T11(v0)  ->  T13(v0)\n            [memory]                  T11(v0*)         ->\n            [memory]                  T3(v0*)          ->\n            MulBackward0              T0(v0), T13(v0)  ->  T14(v0)\n            aten::sum                 T14(v0)          ->  T15(v0)\n            [memory]                  T14(v0*)         ->\n            [memory]                  T13(v0*)         ->\n            AccumulateGrad            T15(v0)          ->\n            [memory]                  T8(v0*)          ->')
        self.assertExpectedInline(self._run_and_format_data_flow(inputs, f_fwd_bwd), '            x:       T0\n            w0:      T1\n            w0.grad:  T17\n            w1:      T4\n            w1.grad:  T13\n            loss:    T7\n\n            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::relu                T2(v0)           ->  T3(v0)\n            [memory]                  T2(v0*)          ->\n            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)\n            aten::relu                T5(v0)           ->  T6(v0)\n            [memory]                  T5(v0*)          ->\n            aten::sum                 T6(v0)           ->  T7(v0)\n            aten::ones_like           T7(v0)           ->  T8(v0)\n            SumBackward0              T8(v0)           ->\n            ReluBackward0             T6(v0), T8(v0)   ->  T9(v0)\n            [memory]                  T6(v0*)          ->\n            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T10(v0), T11(v0)\n            aten::sum                 T10(v0)          ->  T12(v0)\n            [memory]                  T10(v0*)         ->\n            [memory]                  T9(v0*)          ->\n            AccumulateGrad            T12(v0*), T13(v0)  ->  T13(v1)\n            ReluBackward0             T3(v0), T11(v0)  ->  T14(v0)\n            [memory]                  T11(v0*)         ->\n            [memory]                  T3(v0*)          ->\n            MulBackward0              T0(v0), T14(v0)  ->  T15(v0)\n            aten::sum                 T15(v0)          ->  T16(v0)\n            [memory]                  T15(v0*)         ->\n            [memory]                  T14(v0*)         ->\n            AccumulateGrad            T16(v0*), T17(v0)  ->  T17(v1)\n            [memory]                  T8(v0*)          ->')
        return
        x = torch.ones((25,))
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)
        with profile() as prof_no_grad:
            with torch.no_grad():
                x.mul(w0).relu().mul(w1).relu().sum()
        self.assertExpectedInline(self._format_graph(prof_no_grad), '            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::relu                T2(v0)           ->  T3(v0)\n            [memory]                  T2(v0*)          ->\n            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)\n            [memory]                  T3(v0*)          ->\n            aten::relu                T5(v0)           ->  T6(v0)\n            [memory]                  T5(v0*)          ->\n            aten::sum                 T6(v0)           ->  T7(v0)\n            [memory]                  T6(v0*)          ->\n            [memory]                  T7(v0*)          ->')
        with profile() as prof_grad:
            loss = x.mul(w0).relu().mul(w1).relu().sum()
            loss.backward()
        self.assertExpectedInline(self._format_graph(prof_grad), '            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::relu                T2(v0)           ->  T3(v0)\n            [memory]                  T2(v0*)          ->\n            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)\n            aten::relu                T5(v0)           ->  T6(v0)\n            [memory]                  T5(v0*)          ->\n            aten::sum                 T6(v0)           ->  T7(v0)\n            aten::ones_like           T7(v0)           ->  T8(v0)\n            SumBackward0              T8(v0)           ->  T8(v1)\n            ReluBackward0             T6(v0), T8(v1)   ->  T8(v2), T9(v0)\n            [memory]                  T6(v0*)          ->\n            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T9(v1), T10(v0), T11(v0)\n            aten::sum                 T10(v0)          ->  T12(v0)\n            [memory]                  T10(v0*)         ->\n            [memory]                  T9(v1*)          ->\n            AccumulateGrad            T12(v0)          ->  T12(v1)\n            ReluBackward0             T3(v0), T11(v0)  ->  T11(v1), T13(v0)\n            [memory]                  T11(v1*)         ->\n            [memory]                  T3(v0*)          ->\n            MulBackward0              T0(v0), T13(v0)  ->  T13(v1), T14(v0)\n            aten::sum                 T14(v0)          ->  T15(v0)\n            [memory]                  T14(v0*)         ->\n            [memory]                  T13(v1*)         ->\n            AccumulateGrad            T15(v0)          ->  T15(v1)\n            [memory]                  T8(v2*)          ->')
        with profile() as prof_grad:
            loss = x.mul(w0).relu().mul(w1).relu().sum()
            loss.backward()
        self.assertExpectedInline(self._format_graph(prof_grad), '            aten::mul                 T0(v0), T1(v0)   ->  T2(v0)\n            aten::relu                T2(v0)           ->  T3(v0)\n            [memory]                  T2(v0*)          ->\n            aten::mul                 T3(v0), T4(v0)   ->  T5(v0)\n            aten::relu                T5(v0)           ->  T6(v0)\n            [memory]                  T5(v0*)          ->\n            aten::sum                 T6(v0)           ->  T7(v0)\n            aten::ones_like           T7(v0)           ->  T8(v0)\n            SumBackward0              T8(v0)           ->  T8(v1)\n            ReluBackward0             T6(v0), T8(v1)   ->  T8(v2), T9(v0)\n            [memory]                  T6(v0*)          ->\n            MulBackward0              T3(v0), T4(v0), T9(v0)  ->  T9(v1), T10(v0), T11(v0)\n            aten::sum                 T10(v0)          ->  T12(v0)\n            [memory]                  T10(v0*)         ->\n            [memory]                  T9(v1*)          ->\n            AccumulateGrad            T12(v0*), T13(v0)  ->  T13(v1)\n            ReluBackward0             T3(v0), T11(v0)  ->  T11(v1), T14(v0)\n            [memory]                  T11(v1*)         ->\n            [memory]                  T3(v0*)          ->\n            MulBackward0              T0(v0), T14(v0)  ->  T14(v1), T15(v0)\n            aten::sum                 T15(v0)          ->  T16(v0)\n            [memory]                  T15(v0*)         ->\n            [memory]                  T14(v1*)         ->\n            AccumulateGrad            T16(v0*), T17(v0)  ->  T17(v1)\n            [memory]                  T8(v2*)          ->')

@skipIfTorchDynamo('TorchDynamo changes Python calls that memory profiling relies on.')
class TestMemoryProfilerE2E(TestCase):

    @staticmethod
    def _lookup_tensor_categories(t: torch.Tensor, memory_profile: _memory_profiler.MemoryProfile) -> Dict[_memory_profiler.TensorAndID, Optional[_memory_profiler.Category]]:
        if False:
            for i in range(10):
                print('nop')
        storage = t.storage()
        if storage is None:
            raise ValueError('Cannot look up uninitialized Tensor.')
        snapshot = memory_profile._category_snapshot()
        ids = {key.storage.allocation_id for (key, _) in snapshot if key.storage.ptr == storage.data_ptr() and key.device == storage.device}
        return {(key, version): category for ((key, version), category) in memory_profile._category_snapshot().items() if key.storage.allocation_id == max(ids | {-1})}

    def _run_and_check_parameters_and_gradients(self, inner_fn, model, grads_none: bool=False):
        if False:
            for i in range(10):
                print('nop')
        with profile() as prof:
            inner_fn()
        memory_profile = prof._memory_profile()

        def assert_category(t: torch.Tensor, category: _memory_profiler.Category, should_be_none: bool=False):
            if False:
                print('Hello World!')
            if should_be_none:
                assert t is None, 'tensor should be None but is not.'
                return
            self.assertIsNotNone(t)
            categories = self._lookup_tensor_categories(t, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all((c == category for c in categories.values())), categories)
        for p in model.parameters():
            assert_category(p, _memory_profiler.Category.PARAMETER)
            assert_category(p.grad, _memory_profiler.Category.GRADIENT, grads_none)
        _ = memory_profile.timeline

    def _run_and_format_categories(self, fn, indent=12):
        if False:
            return 10
        'Generate summary of assigned categories for expecttest.'
        with RecordInputOutputDispatchMode() as record_ops, profile() as prof:
            fn(lambda name: record_ops.mark_region(f'-- {name} '.ljust(105, '-')))
        memory_profile = prof._memory_profile()
        ptr_pair_to_key: Dict[Tuple[int, int], _memory_profiler.TensorKey] = {}
        snapshot = memory_profile._category_snapshot()
        for op in memory_profile._op_tree.dfs():
            if op.typed[0] == _EventType.TorchOp:
                inputs = pytree.tree_leaves(op.typed[1].inputs)
                for t in (i for i in inputs if isinstance(i, _TensorMetadata)):
                    key = _memory_profiler.TensorKey.from_tensor(t)
                    if key:
                        ptr_pair_to_key[t.impl_ptr, t.storage_data_ptr] = key

        def format_categories(ptr_pair: int):
            if False:
                i = 10
                return i + 15
            target_key = ptr_pair_to_key.get(ptr_pair, None)
            if target_key is None:
                return '???'
            matches = tuple(((version, category.name if category else '???') for ((key, version), category) in snapshot.items() if key == target_key))
            assert matches, 'Failed to lookup Tensor'
            categories = [matches[0][1]]
            for (_, category) in matches:
                if category != categories[-1]:
                    categories.append(category)
            return f"{target_key.storage.allocation_id} ({','.join(categories)})"
        out: List[str] = []
        for (name, inputs, outputs) in record_ops.results:
            if inputs or outputs:
                inputs_str = ', '.join((format_categories(i) for i in inputs))
                outputs_str = ', '.join((format_categories(i) for i in outputs))
                out.append(f'{name:<40} {inputs_str:<45} -> {outputs_str}')
            else:
                out.append(f'\n{name}')
        return textwrap.indent('\n'.join(out), ' ' * indent)

    def test_parameters_and_gradients(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), ScaleLayer(), torch.nn.Linear(2, 1), ScaleLayer())
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def fwd_only():
            if False:
                i = 10
                return i + 15
            _ = model(torch.ones((2, 2)))

        def fwd_bwd_step():
            if False:
                while True:
                    i = 10
            optimizer.zero_grad()
            y = model(torch.ones((2, 2)))
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            optimizer.step()
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_only, model=model, grads_none=True)
        self.assertTrue(all((p.grad is None for p in model.parameters())))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)
        self.assertTrue(not any((p.grad is None for p in model.parameters())))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_only, model=model)

    def test_parameters_and_gradients_set_to_none(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

        def fwd_bwd_step():
            if False:
                print('Hello World!')
            for _ in range(3):
                optimizer.zero_grad(set_to_none=True)
                y = model(torch.ones((2, 2)))
                torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
                optimizer.step()
        fwd_bwd_step()
        self.assertTrue(not any((p.grad is None for p in model.parameters())))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)
        optimizer.zero_grad(set_to_none=True)
        self.assertTrue(all((p.grad is None for p in model.parameters())))
        self._run_and_check_parameters_and_gradients(inner_fn=fwd_bwd_step, model=model)

    def test_inputs_fwd(self):
        if False:
            return 10
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        inputs = [torch.ones((2, 2)) for _ in range(2)]
        with profile() as prof:
            for x in inputs:
                _ = model(x)
            for _ in range(2):
                x = torch.ones((2, 2))
                inputs.append(x)
                _ = model(x)
        memory_profile = prof._memory_profile()
        for x in inputs:
            categories = self._lookup_tensor_categories(x, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all((i == _memory_profiler.Category.INPUT for i in categories.values())), categories)
        snapshot = memory_profile._category_snapshot()
        self.assertTrue(_memory_profiler.Category.INPUT in snapshot.values())

    def test_inputs_fwd_lazy(self):
        if False:
            for i in range(10):
                print('nop')
        model = torch.nn.Sequential(LazyLinear(2, 2), LazyLinear(2, 1))
        inputs = [torch.ones((2, 2)) for _ in range(2)]
        with profile() as prof:
            for x in inputs:
                _ = model(x)
            for _ in range(2):
                x = torch.ones((2, 2))
                inputs.append(x)
                _ = model(x)
        memory_profile = prof._memory_profile()
        for x in inputs:
            categories = self._lookup_tensor_categories(x, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all((i is None for i in categories.values())), categories)
        snapshot = memory_profile._category_snapshot()
        self.assertFalse(_memory_profiler.Category.INPUT in snapshot.values())

    def test_inputs_fwd_bwd(self):
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        inputs_targets = [(torch.ones((2, 2)), torch.rand((2, 1))) for _ in range(2)]

        def fwd_bwd_step(x, targets):
            if False:
                while True:
                    i = 10
            y = model(x)
            torch.nn.functional.mse_loss(y, targets).backward()
            optimizer.step()
            optimizer.zero_grad()
        with profile() as prof:
            for (x, targets) in inputs_targets:
                fwd_bwd_step(x, targets)
            for _ in range(2):
                x = torch.ones((2, 2))
                targets = torch.rand((2, 1))
                inputs_targets.append((x, targets))
                fwd_bwd_step(x, targets)
        memory_profile = prof._memory_profile()

        def check(t):
            if False:
                print('Hello World!')
            categories = self._lookup_tensor_categories(t, memory_profile)
            self.assertGreater(len(categories), 0)
            self.assertTrue(all((i == _memory_profiler.Category.INPUT for i in categories.values())))
        for (x, targets) in inputs_targets:
            check(x)
            check(targets)

    def test_lazily_initialized(self) -> None:
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), LazyLinear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
        self.assertEqual(len(list(model.parameters())), 4)

        def inner_fn():
            if False:
                return 10
            y = model(torch.ones((2, 2)))
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            optimizer.zero_grad()
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            optimizer.step()
        self._run_and_check_parameters_and_gradients(inner_fn=inner_fn, model=model)
        self.assertEqual(len(list(model.parameters())), 6)

    def test_manual_optimizer_step(self) -> None:
        if False:
            i = 10
            return i + 15
        model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))

        def inner_fn():
            if False:
                print('Hello World!')
            y = model(torch.ones((2, 2)))
            torch.nn.functional.mse_loss(y, torch.rand((2, 1))).backward()
            with torch.no_grad():
                for p in model.parameters():
                    grad = p.grad
                    self.assertIsNotNone(grad)
                    p.add_(grad, alpha=-0.1)
        self._run_and_check_parameters_and_gradients(inner_fn=inner_fn, model=model)

    def test_categories_e2e_simple_fwd(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        def step_fn(_):
            if False:
                while True:
                    i = 10
            x = torch.ones((2, 2))
            y = torch.cat([x * w0, x * w1], dim=1)
        self.assertExpectedInline(self._run_and_format_categories(step_fn), '            aten::ones                                                                             -> 1 (???)\n            aten::mul.Tensor                         1 (???), 2 (???)                              -> 3 (???)\n            aten::mul.Tensor                         1 (???), 4 (???)                              -> 5 (???)\n            aten::cat                                3 (???), 5 (???)                              -> ???')

    def test_categories_e2e_simple_fwd_bwd(self) -> None:
        if False:
            print('Hello World!')
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)

        def step_fn(mark_region):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.ones((2, 2))
            targets = torch.ones((2, 4))
            mark_region('Forward & loss')
            y = torch.cat([x * w0, x * w1], dim=1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y, targets)
            mark_region('Backward')
            loss.backward()
        self.assertExpectedInline(self._run_and_format_categories(step_fn), '            aten::ones                                                                             -> 1 (INPUT)\n            aten::ones                                                                             -> 2 (INPUT)\n\n            -- Forward & loss ---------------------------------------------------------------------------------------\n            aten::mul.Tensor                         1 (INPUT), 3 (INPUT)                          -> 4 (INPUT)\n            aten::mul.Tensor                         1 (INPUT), 5 (INPUT)                          -> 6 (INPUT)\n            aten::cat                                4 (INPUT), 6 (INPUT)                          -> 7 (INPUT)\n            aten::binary_cross_entropy_with_logits   7 (INPUT), 2 (INPUT)                          -> 13 (INPUT)\n\n            -- Backward ---------------------------------------------------------------------------------------------\n            aten::ones_like                          13 (INPUT)                                    -> 16 (INPUT)\n            aten::sigmoid                            7 (INPUT)                                     -> 17 (TEMPORARY)\n            aten::sub.Tensor                         17 (TEMPORARY), 2 (INPUT)                     -> 18 (TEMPORARY)\n            aten::mul.Tensor                         18 (TEMPORARY), 16 (INPUT)                    -> 19 (AUTOGRAD_DETAIL)\n            aten::div_.Scalar                        19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::slice.Tensor                       19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::slice.Tensor                       19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::mul.Tensor                         19 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 22 (AUTOGRAD_DETAIL)\n            aten::sum.dim_IntList                    22 (AUTOGRAD_DETAIL)                          -> 23 (GRADIENT)\n            aten::view                               23 (GRADIENT)                                 -> 23 (GRADIENT)\n            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)\n            aten::detach                             23 (GRADIENT)                                 -> ???\n            aten::mul.Tensor                         19 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 24 (AUTOGRAD_DETAIL)\n            aten::sum.dim_IntList                    24 (AUTOGRAD_DETAIL)                          -> 25 (GRADIENT)\n            aten::view                               25 (GRADIENT)                                 -> 25 (GRADIENT)\n            aten::detach                             25 (GRADIENT)                                 -> 25 (GRADIENT)\n            aten::detach                             25 (GRADIENT)                                 -> ???')

    def test_categories_e2e_simple_fwd_bwd_step(self) -> None:
        if False:
            print('Hello World!')
        w0 = torch.ones((1,), requires_grad=True)
        w1 = torch.ones((1,), requires_grad=True)
        optimizer = torch.optim.SGD([w0, w1], lr=0.1)

        def step_fn(mark_region):
            if False:
                while True:
                    i = 10
            x = torch.ones((2, 2))
            targets = torch.ones((2, 4))
            mark_region('Forward & loss')
            y = torch.cat([x * w0, x * w1], dim=1)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(y, targets)
            mark_region('Backward')
            loss.backward()
            mark_region('Optimizer')
            optimizer.step()
            optimizer.zero_grad()
        self.assertExpectedInline(self._run_and_format_categories(step_fn), '            aten::ones                                                                             -> 1 (INPUT)\n            aten::ones                                                                             -> 2 (INPUT)\n\n            -- Forward & loss ---------------------------------------------------------------------------------------\n            aten::mul.Tensor                         1 (INPUT), 3 (PARAMETER)                      -> 4 (ACTIVATION)\n            aten::mul.Tensor                         1 (INPUT), 5 (PARAMETER)                      -> 6 (ACTIVATION)\n            aten::cat                                4 (ACTIVATION), 6 (ACTIVATION)                -> 7 (ACTIVATION)\n            aten::binary_cross_entropy_with_logits   7 (ACTIVATION), 2 (INPUT)                     -> 13 (ACTIVATION)\n\n            -- Backward ---------------------------------------------------------------------------------------------\n            aten::ones_like                          13 (ACTIVATION)                               -> 16 (ACTIVATION)\n            aten::sigmoid                            7 (ACTIVATION)                                -> 17 (TEMPORARY)\n            aten::sub.Tensor                         17 (TEMPORARY), 2 (INPUT)                     -> 18 (TEMPORARY)\n            aten::mul.Tensor                         18 (TEMPORARY), 16 (ACTIVATION)               -> 19 (AUTOGRAD_DETAIL)\n            aten::div_.Scalar                        19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::slice.Tensor                       19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::slice.Tensor                       19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::mul.Tensor                         19 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 22 (AUTOGRAD_DETAIL)\n            aten::sum.dim_IntList                    22 (AUTOGRAD_DETAIL)                          -> 23 (GRADIENT)\n            aten::view                               23 (GRADIENT)                                 -> 23 (GRADIENT)\n            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)\n            aten::detach                             23 (GRADIENT)                                 -> 23 (GRADIENT)\n            aten::mul.Tensor                         19 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 24 (AUTOGRAD_DETAIL)\n            aten::sum.dim_IntList                    24 (AUTOGRAD_DETAIL)                          -> 25 (GRADIENT)\n            aten::view                               25 (GRADIENT)                                 -> 25 (GRADIENT)\n            aten::detach                             25 (GRADIENT)                                 -> 25 (GRADIENT)\n            aten::detach                             25 (GRADIENT)                                 -> 25 (GRADIENT)\n\n            -- Optimizer --------------------------------------------------------------------------------------------\n            aten::add_.Tensor                        3 (PARAMETER), 25 (GRADIENT)                  -> 3 (PARAMETER)\n            aten::add_.Tensor                        5 (PARAMETER), 23 (GRADIENT)                  -> 5 (PARAMETER)')

    def test_categories_e2e_simple_module_fwd(self) -> None:
        if False:
            i = 10
            return i + 15
        model = torch.nn.Linear(2, 4, bias=True)
        self.assertExpectedInline(self._run_and_format_categories(lambda _: model(torch.ones((2, 2)))), '            aten::ones                                                                             -> 1 (INPUT)\n            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)\n            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)')

    def test_categories_e2e_simple_module_fwd_bwd(self) -> None:
        if False:
            while True:
                i = 10
        model = torch.nn.Linear(2, 1, bias=True)

        def step_fn(mark_region):
            if False:
                i = 10
                return i + 15
            mark_region('Forward & loss')
            loss = model(torch.ones((2, 2))).sum()
            mark_region('Backward')
            loss.backward()
        self.assertExpectedInline(self._run_and_format_categories(step_fn), '\n            -- Forward & loss ---------------------------------------------------------------------------------------\n            aten::ones                                                                             -> 1 (INPUT)\n            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)\n            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)\n            aten::sum                                4 (ACTIVATION)                                -> 5 (ACTIVATION)\n\n            -- Backward ---------------------------------------------------------------------------------------------\n            aten::ones_like                          5 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::expand                             6 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::t                                  6 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::mm                                 6 (ACTIVATION), 1 (INPUT)                     -> 7 (GRADIENT)\n            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)\n            aten::sum.dim_IntList                    6 (ACTIVATION)                                -> 9 (GRADIENT)\n            aten::view                               9 (GRADIENT)                                  -> 9 (GRADIENT)\n            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)\n            aten::detach                             9 (GRADIENT)                                  -> ???\n            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)\n            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)\n            aten::detach                             7 (GRADIENT)                                  -> ???')

    def test_categories_e2e_simple_module_fwd_bwd_step(self) -> None:
        if False:
            print('Hello World!')
        model = torch.nn.Linear(2, 1, bias=True)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        def step_fn(mark_region):
            if False:
                for i in range(10):
                    print('nop')
            mark_region('Forward & loss')
            loss = model(torch.ones((2, 2))).sum()
            mark_region('Backward')
            loss.backward()
            mark_region('Optimizer')
            optimizer.step()
            optimizer.zero_grad()
        self.assertExpectedInline(self._run_and_format_categories(step_fn), '\n            -- Forward & loss ---------------------------------------------------------------------------------------\n            aten::ones                                                                             -> 1 (INPUT)\n            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)\n            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)\n            aten::sum                                4 (ACTIVATION)                                -> 5 (ACTIVATION)\n\n            -- Backward ---------------------------------------------------------------------------------------------\n            aten::ones_like                          5 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::expand                             6 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::t                                  6 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::mm                                 6 (ACTIVATION), 1 (INPUT)                     -> 7 (GRADIENT)\n            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)\n            aten::sum.dim_IntList                    6 (ACTIVATION)                                -> 9 (GRADIENT)\n            aten::view                               9 (GRADIENT)                                  -> 9 (GRADIENT)\n            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)\n            aten::detach                             9 (GRADIENT)                                  -> 9 (GRADIENT)\n            aten::t                                  7 (GRADIENT)                                  -> 7 (GRADIENT)\n            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)\n            aten::detach                             7 (GRADIENT)                                  -> 7 (GRADIENT)\n\n            -- Optimizer --------------------------------------------------------------------------------------------\n            aten::clone                              7 (GRADIENT)                                  -> 10 (OPTIMIZER_STATE)\n            aten::detach                             10 (OPTIMIZER_STATE)                          -> 10 (OPTIMIZER_STATE)\n            aten::detach                             10 (OPTIMIZER_STATE)                          -> 10 (OPTIMIZER_STATE)\n            aten::add_.Tensor                        2 (PARAMETER), 10 (OPTIMIZER_STATE)           -> 2 (PARAMETER)\n            aten::clone                              9 (GRADIENT)                                  -> 11 (OPTIMIZER_STATE)\n            aten::detach                             11 (OPTIMIZER_STATE)                          -> 11 (OPTIMIZER_STATE)\n            aten::detach                             11 (OPTIMIZER_STATE)                          -> 11 (OPTIMIZER_STATE)\n            aten::add_.Tensor                        3 (PARAMETER), 11 (OPTIMIZER_STATE)           -> 3 (PARAMETER)')

    def test_categories_e2e_sequential_fwd(self) -> None:
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(2, 4, bias=True), torch.nn.ReLU(), torch.nn.Linear(4, 4, bias=False), torch.nn.Softmax(dim=1))
        self.assertExpectedInline(self._run_and_format_categories(lambda _: model(torch.ones((2, 2)))), '            aten::ones                                                                             -> 1 (INPUT)\n            aten::t                                  2 (PARAMETER)                                 -> 2 (PARAMETER)\n            aten::addmm                              3 (PARAMETER), 1 (INPUT), 2 (PARAMETER)       -> 4 (ACTIVATION)\n            aten::relu                               4 (ACTIVATION)                                -> 5 (ACTIVATION)\n            aten::detach                             5 (ACTIVATION)                                -> ???\n            aten::t                                  6 (PARAMETER)                                 -> 6 (PARAMETER)\n            aten::mm                                 5 (ACTIVATION), 6 (PARAMETER)                 -> 7 (ACTIVATION)\n            aten::_softmax                           7 (ACTIVATION)                                -> 8 (ACTIVATION)\n            aten::detach                             8 (ACTIVATION)                                -> ???')

    def test_categories_e2e_sequential_fwd_bwd(self) -> None:
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(2, 4, bias=True), torch.nn.ReLU(), torch.nn.Linear(4, 4, bias=False), torch.nn.Softmax(dim=1))

        def step_fn(mark_region):
            if False:
                print('Hello World!')
            x = torch.ones((2, 2))
            targets = torch.ones((2, 4))
            mark_region('Forward')
            y = model(x)
            mark_region('Loss')
            loss = torch.sum((y - targets) ** 2).mean()
            mark_region('Backward')
            loss.backward()
        self.assertExpectedInline(self._run_and_format_categories(step_fn), '            aten::ones                                                                             -> 1 (INPUT)\n            aten::ones                                                                             -> 2 (INPUT)\n\n            -- Forward ----------------------------------------------------------------------------------------------\n            aten::t                                  3 (PARAMETER)                                 -> 3 (PARAMETER)\n            aten::addmm                              4 (PARAMETER), 1 (INPUT), 3 (PARAMETER)       -> 5 (ACTIVATION)\n            aten::relu                               5 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::detach                             6 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::t                                  7 (PARAMETER)                                 -> 7 (PARAMETER)\n            aten::mm                                 6 (ACTIVATION), 7 (PARAMETER)                 -> 8 (ACTIVATION)\n            aten::_softmax                           8 (ACTIVATION)                                -> 9 (ACTIVATION)\n            aten::detach                             9 (ACTIVATION)                                -> 9 (ACTIVATION)\n\n            -- Loss -------------------------------------------------------------------------------------------------\n            aten::sub.Tensor                         9 (ACTIVATION), 2 (INPUT)                     -> 10 (ACTIVATION)\n            aten::pow.Tensor_Scalar                  10 (ACTIVATION)                               -> 11 (ACTIVATION)\n            aten::sum                                11 (ACTIVATION)                               -> 12 (ACTIVATION)\n            aten::mean                               12 (ACTIVATION)                               -> 13 (ACTIVATION)\n\n            -- Backward ---------------------------------------------------------------------------------------------\n            aten::ones_like                          13 (ACTIVATION)                               -> 16 (ACTIVATION)\n            aten::expand                             16 (ACTIVATION)                               -> 16 (ACTIVATION)\n            aten::div.Scalar                         16 (ACTIVATION)                               -> 19 (AUTOGRAD_DETAIL)\n            aten::expand                             19 (AUTOGRAD_DETAIL)                          -> 19 (AUTOGRAD_DETAIL)\n            aten::pow.Tensor_Scalar                  10 (ACTIVATION)                               -> 20 (TEMPORARY)\n            aten::mul.Scalar                         20 (TEMPORARY)                                -> 23 (TEMPORARY)\n            aten::mul.Tensor                         19 (AUTOGRAD_DETAIL), 23 (TEMPORARY)          -> 24 (AUTOGRAD_DETAIL)\n            aten::detach                             9 (ACTIVATION)                                -> 9 (ACTIVATION)\n            aten::_softmax_backward_data             24 (AUTOGRAD_DETAIL), 9 (ACTIVATION)          -> 25 (AUTOGRAD_DETAIL)\n            aten::t                                  25 (AUTOGRAD_DETAIL)                          -> 25 (AUTOGRAD_DETAIL)\n            aten::mm                                 25 (AUTOGRAD_DETAIL), 6 (ACTIVATION)          -> 26 (GRADIENT)\n            aten::t                                  26 (GRADIENT)                                 -> 26 (GRADIENT)\n            aten::t                                  7 (PARAMETER)                                 -> 7 (PARAMETER)\n            aten::mm                                 25 (AUTOGRAD_DETAIL), 7 (PARAMETER)           -> 27 (AUTOGRAD_DETAIL)\n            aten::t                                  26 (GRADIENT)                                 -> 26 (GRADIENT)\n            aten::detach                             26 (GRADIENT)                                 -> 26 (GRADIENT)\n            aten::detach                             26 (GRADIENT)                                 -> ???\n            aten::detach                             6 (ACTIVATION)                                -> 6 (ACTIVATION)\n            aten::threshold_backward                 27 (AUTOGRAD_DETAIL), 6 (ACTIVATION)          -> 28 (AUTOGRAD_DETAIL)\n            aten::t                                  28 (AUTOGRAD_DETAIL)                          -> 28 (AUTOGRAD_DETAIL)\n            aten::mm                                 28 (AUTOGRAD_DETAIL), 1 (INPUT)               -> 29 (GRADIENT)\n            aten::t                                  29 (GRADIENT)                                 -> 29 (GRADIENT)\n            aten::sum.dim_IntList                    28 (AUTOGRAD_DETAIL)                          -> 30 (GRADIENT)\n            aten::view                               30 (GRADIENT)                                 -> 30 (GRADIENT)\n            aten::detach                             30 (GRADIENT)                                 -> 30 (GRADIENT)\n            aten::detach                             30 (GRADIENT)                                 -> ???\n            aten::t                                  29 (GRADIENT)                                 -> 29 (GRADIENT)\n            aten::detach                             29 (GRADIENT)                                 -> 29 (GRADIENT)\n            aten::detach                             29 (GRADIENT)                                 -> ???')

    def test_memory_timeline(self) -> None:
        if False:
            while True:
                i = 10
        model = torch.nn.Sequential(torch.nn.Linear(64, 512, bias=True), torch.nn.ReLU(), torch.nn.Linear(512, 512, bias=False), torch.nn.Softmax(dim=1))
        optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
        with profile() as prof:
            x = torch.ones((1024, 64))
            targets = torch.ones((1024, 512))
            y = model(x)
            loss = torch.nn.functional.mse_loss(y, targets)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        memory_profile = prof._memory_profile()
        timeline = memory_profile.timeline
        times = tuple((t for (t, _, _, _) in timeline))
        self.assertTrue(all((t1 >= t0 for (t0, t1) in zip(times, times[1:]))), times)
        self.assertTrue(all((t == -1 if action == _memory_profiler.Action.PREEXISTING else t > 0 for (t, action, _, _) in timeline)))

        def category_name(category):
            if False:
                while True:
                    i = 10
            return category.name if category else '???'

        def format_action(action, key, version):
            if False:
                return 10
            category = memory_profile._categories.get(key, version)
            if action == _memory_profiler.Action.INCREMENT_VERSION:
                new_category = memory_profile._categories.get(key, version + 1)
                if category != new_category:
                    return f'{category_name(category)} -> {category_name(new_category)}'
            return category_name(category)

        def format_size(size: int):
            if False:
                i = 10
                return i + 15
            if size < 1024:
                return f'{size / 1024:3.1f} kB'
            return f'{size // 1024} kB'
        id_map = {}

        def id_for_testing(key):
            if False:
                for i in range(10):
                    print('nop')
            return id_map.setdefault(key.storage.allocation_id, len(id_map))
        lines = [f'{action.name.lower():<25}  {format_action(action, key, version):<25}  {id_for_testing(key):>3}(v{version}) {format_size(size):>15}' for (_, action, (key, version), size) in prof._memory_profile().timeline if size > 1024]
        self.assertExpectedInline(textwrap.indent('\n'.join(lines), ' ' * 12), '            preexisting                PARAMETER                    0(v0)          128 kB\n            preexisting                PARAMETER                    1(v0)            2 kB\n            preexisting                PARAMETER                    2(v0)         1024 kB\n            create                     INPUT                        3(v0)          256 kB\n            create                     INPUT                        4(v0)         2048 kB\n            create                     ACTIVATION                   5(v0)         2048 kB\n            create                     ACTIVATION                   6(v0)         2048 kB\n            destroy                    ACTIVATION                   5(v0)         2048 kB\n            create                     ACTIVATION                   7(v0)         2048 kB\n            create                     ACTIVATION                   8(v0)         2048 kB\n            destroy                    ACTIVATION                   7(v0)         2048 kB\n            create                     ACTIVATION                   9(v0)         2048 kB\n            create                     TEMPORARY                   10(v0)         2048 kB\n            destroy                    TEMPORARY                   10(v0)         2048 kB\n            create                     AUTOGRAD_DETAIL             11(v0)         2048 kB\n            create                     AUTOGRAD_DETAIL             12(v0)         2048 kB\n            destroy                    AUTOGRAD_DETAIL             11(v0)         2048 kB\n            create                     GRADIENT                    13(v0)         1024 kB\n            create                     AUTOGRAD_DETAIL             14(v0)         2048 kB\n            destroy                    AUTOGRAD_DETAIL             12(v0)         2048 kB\n            create                     AUTOGRAD_DETAIL             15(v0)         2048 kB\n            destroy                    AUTOGRAD_DETAIL             14(v0)         2048 kB\n            destroy                    ACTIVATION                   6(v0)         2048 kB\n            create                     GRADIENT                    16(v0)          128 kB\n            create                     GRADIENT                    17(v0)            2 kB\n            destroy                    AUTOGRAD_DETAIL             15(v0)         2048 kB\n            create                     OPTIMIZER_STATE             18(v0)          128 kB\n            create                     OPTIMIZER_STATE             19(v0)          128 kB\n            create                     OPTIMIZER_STATE             20(v0)            2 kB\n            create                     OPTIMIZER_STATE             21(v0)            2 kB\n            create                     OPTIMIZER_STATE             22(v0)         1024 kB\n            create                     OPTIMIZER_STATE             23(v0)         1024 kB\n            increment_version          OPTIMIZER_STATE             18(v0)          128 kB\n            increment_version          OPTIMIZER_STATE             19(v0)          128 kB\n            increment_version          OPTIMIZER_STATE             19(v1)          128 kB\n            create                     ???                         24(v0)          128 kB\n            create                     ???                         25(v0)          128 kB\n            destroy                    ???                         24(v0)          128 kB\n            increment_version          ???                         25(v0)          128 kB\n            increment_version          PARAMETER                    0(v0)          128 kB\n            increment_version          OPTIMIZER_STATE             20(v0)            2 kB\n            increment_version          OPTIMIZER_STATE             21(v0)            2 kB\n            increment_version          OPTIMIZER_STATE             21(v1)            2 kB\n            create                     ???                         26(v0)            2 kB\n            create                     ???                         27(v0)            2 kB\n            destroy                    ???                         26(v0)            2 kB\n            increment_version          ???                         27(v0)            2 kB\n            destroy                    ???                         25(v1)          128 kB\n            increment_version          PARAMETER                    1(v0)            2 kB\n            increment_version          OPTIMIZER_STATE             22(v0)         1024 kB\n            increment_version          OPTIMIZER_STATE             23(v0)         1024 kB\n            increment_version          OPTIMIZER_STATE             23(v1)         1024 kB\n            create                     ???                         28(v0)         1024 kB\n            create                     ???                         29(v0)         1024 kB\n            destroy                    ???                         28(v0)         1024 kB\n            increment_version          ???                         29(v0)         1024 kB\n            destroy                    ???                         27(v1)            2 kB\n            increment_version          PARAMETER                    2(v0)         1024 kB\n            destroy                    ???                         29(v1)         1024 kB\n            destroy                    GRADIENT                    16(v0)          128 kB\n            destroy                    GRADIENT                    17(v0)            2 kB\n            destroy                    GRADIENT                    13(v0)         1024 kB')

    def test_memory_timeline_no_id(self) -> None:
        if False:
            while True:
                i = 10
        x = torch.ones((1024,), device='cuda' if torch.cuda.is_available() else 'cpu')
        with profile() as prof:
            del x
            y = torch.empty((64,))
            del y
            z = torch.empty((256,))
            z.view_as(z)
            del z
        memory_profile = prof._memory_profile()
        expected = [(_memory_profiler.Action.PREEXISTING, 4096), (_memory_profiler.Action.DESTROY, 4096), (_memory_profiler.Action.CREATE, 256), (_memory_profiler.Action.DESTROY, 256), (_memory_profiler.Action.CREATE, 1024), (_memory_profiler.Action.DESTROY, 1024)]
        actual = [(action, size) for (_, action, _, size) in memory_profile.timeline]
        if not torch.cuda.is_available():
            expected = expected[2:]
            for event in expected:
                self.assertTrue(event in actual, f'event: {event} was not found in actual.')
        else:
            self.assertEqual(actual, expected, f'expected does not match actual: {actual}')
if __name__ == '__main__':
    run_tests()