import builtins
import contextlib
import copy
import functools
import inspect
import math
import numbers
import io
import operator
import os
import pickle
import sys
import torch
import traceback
import typing
import types
import warnings
import unittest
from math import sqrt
from functorch.experimental import control_flow
from torch.multiprocessing import Process
from torch.testing import FileCheck
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_device_type import ops, onlyCPU, instantiate_device_type_tests
import torch.utils._pytree as pytree
import torch.fx._pytree as fx_pytree
from torch.fx import symbolic_trace, Proxy, Node, GraphModule, Interpreter, Tracer, Transformer, Graph, wrap, PH, CodeGen
from torch.fx.node import Target, Argument, _format_arg
from torch.fx.passes import shape_prop
from torch.fx.immutable_collections import immutable_dict, immutable_list
from torch.fx.experimental.rewriter import RewritingTracer
from torch.fx.operator_schemas import get_signature_for_torch_op
from copy import deepcopy
from collections import namedtuple
from torch.fx.proxy import TraceError
from torch.fx._compatibility import _BACK_COMPAT_OBJECTS, _MARKED_WITH_COMPATIBILITY
from torch.fx._symbolic_trace import PHBase, PHWithMeta
from fx.test_subgraph_rewriter import TestSubgraphRewriter
from fx.test_dce_pass import TestDCE
from fx.test_fx_const_fold import TestConstFold
from fx.test_fx_param_shape_control_flow import TestConstParamShapeInControlFlow
from fx.test_pass_infra import TestPassManager
from fx.test_common_passes import TestCommonPass
from fx.test_cse_pass import TestCSEPass
from fx.test_matcher_utils import TestMatcher
from fx.test_source_matcher_utils import TestSourceMatcher
from fx.test_gradual_type import AnnotationsTest
from fx.test_gradual_type import TypeCheckerTest
from typing import Any, Callable, Dict, NamedTuple, List, Optional, Tuple, Union
from torch.testing._internal.common_utils import IS_FBCODE, IS_MACOS, IS_WINDOWS, find_library_location, run_tests
from torch.testing._internal.jit_utils import JitTestCase
from fx.named_tup import MyNamedTup
try:
    from torchvision import models as torchvision_models
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport

class SimpleTest(torch.nn.Module):

    def forward(self, x):
        if False:
            while True:
                i = 10
        return torch.relu(x + 3.0)

def a_non_torch_leaf(a, b):
    if False:
        for i in range(10):
            print('nop')
    return a + b

def fx_int(x: float) -> int:
    if False:
        return 10
    return int(x)

def fx_int_x2(x: float) -> int:
    if False:
        while True:
            i = 10
    return int(x) * 2
Point = namedtuple('Point', ['x', 'y'])

def a_lifted_leaf(a, b):
    if False:
        i = 10
        return i + 15
    return a[0] + a[1] + b
wrap('a_lifted_leaf')
wrap('a_lifted_leaf')

def a_lifted_leaf2(a, b):
    if False:
        i = 10
        return i + 15
    return a[0] + a[1] + b
wrap(a_lifted_leaf2)
wrap('len')
wrap('getattr')

def wrapped_named_tup(p1, *, p2):
    if False:
        return 10
    return p1.x + p2.y
wrap(wrapped_named_tup)

@wrap
def wrapped_via_decorator(a):
    if False:
        while True:
            i = 10
    return a + 1
wrap('wrapped_with_submodule')

def wrapped_with_submodule(x: torch.Tensor, batchnorm1d: torch.nn.BatchNorm1d):
    if False:
        while True:
            i = 10
    return batchnorm1d(x)

def my_decorator(f):
    if False:
        i = 10
        return i + 15

    @functools.wraps(f)
    def wrapper_inside_decorator(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return f(*args, **kwargs)
    return wrapper_inside_decorator

@wrap
@my_decorator
def wrapped_decorated_fn(x):
    if False:
        while True:
            i = 10
    return x
real_wrapped_via_decorator = wrapped_via_decorator
real_a_lifed_leaf = a_lifted_leaf
real_a_lifed_leaf2 = a_lifted_leaf2
_sqrt = sqrt
wrap('wrapper_fn')

def wrapper_fn(x):
    if False:
        print('Hello World!')
    return torch.foo(x)

class Pair(NamedTuple):
    x: torch.Tensor
    y: torch.Tensor

    def _custom_fx_repr_fn(self) -> str:
        if False:
            while True:
                i = 10
        return f'Pair(x={_format_arg(self.x)}, y={_format_arg(self.y)})'

class Foo:

    def __init__(self, a, b):
        if False:
            print('Hello World!')
        self.a = a
        self.b = b

class Add(torch.nn.Module):

    def forward(self, x):
        if False:
            print('Hello World!')
        return x + x

@torch.fx.has_side_effect
@torch.fx.wrap
def side_effect_func(x: torch.Tensor):
    if False:
        print('Hello World!')
    print(x)

class TestFX(JitTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super().setUp()
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True
        if not (IS_FBCODE or IS_WINDOWS or IS_MACOS):
            lib_file_path = find_library_location('libtorchbind_test.so')
            torch.ops.load_library(str(lib_file_path))

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        super().tearDown()
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    def checkGraphModule(self, m: torch.nn.Module, args, kwargs=None):
        if False:
            while True:
                i = 10
        "Check that an nn.Module's results match the GraphModule version\n        for a given set of args/kwargs.\n        "
        kwargs = kwargs if kwargs else {}
        ref_outs = m(*args, **kwargs)
        gm = symbolic_trace(m)
        gm.graph.lint()
        test_outs = gm(*args, **kwargs)
        self.assertEqual(ref_outs, test_outs)

    def test_graph_module(self):
        if False:
            for i in range(10):
                print('nop')

        class MySub(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.w = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.w + x

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.lin = torch.nn.Linear(4, 3)
                self.sub_mod = MySub()
                self.w = torch.nn.Parameter(torch.rand(3))

            def forward(self, A, B, c):
                if False:
                    for i in range(10):
                        print('nop')
                t = torch.sigmoid(A) + self.lin(c)
                return self.sub_mod(t.data + self.w + t + 1 - A + B // A + -A + A.add(B, alpha=3))
        m = MyModule()
        gm = symbolic_trace(m)
        ms = torch.jit.script(gm)

        class M2(torch.nn.Module):

            def forward(self, A):
                if False:
                    print('Hello World!')
                (m, idx) = torch.max(A, 0)
                return (m + 1, idx + 1)
        m2 = M2()
        gm2 = symbolic_trace(m2)

        class T(torch.nn.Module):

            def forward(self, A, b=4, *args, c=5, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                x = A + 1 + args[0] + kwargs['3']
                return x
        t = T()
        symbolic_trace(t)

        class M3(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return torch.relu(x)
        m3 = M3()
        gm3 = symbolic_trace(m3)
        new_instance = gm3.__new__(type(gm3))
        new_instance.__init__(gm3, gm3.graph)
        x = torch.randn(5, 3)
        torch.testing.assert_close(new_instance(x), torch.relu(x))

    def test_informative_co_filename(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                return a * 2
        gm = symbolic_trace(MyModule())
        self.assertIn(os.path.basename(__file__), gm.forward.__code__.co_filename)

    def test_custom_import(self):
        if False:
            for i in range(10):
                print('nop')
        graph = torch.fx.Graph()
        a = graph.placeholder('x')
        b = graph.placeholder('y')
        c = graph.call_function(a_non_torch_leaf, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        (x, y) = (torch.rand(1), torch.rand(1))
        self.assertEqual(torch.sin(x + y), gm(x, y))

    def test_args_kwargs(self):
        if False:
            i = 10
            return i + 15

        class T(torch.nn.Module):

            def forward(self, *args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                x = args[0] + kwargs['foo']
                return x
        t = T()
        self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {'foo': torch.rand(1)})

    def test_args_kwargs_no_self(self):
        if False:
            for i in range(10):
                print('nop')

        class T(torch.nn.Module):

            def forward(*args, **kwargs):
                if False:
                    return 10
                self = args[0]
                return torch.relu(args[1])
        t = T()
        with self.assertRaisesRegex(RuntimeError, 'cannot be part of \\*args expansion'):
            self.checkGraphModule(t, (torch.rand(1), torch.rand(1)), {'foo': torch.rand(1)})

    def test_fx_shifts(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return (x << 3, x >> 3)
        input = torch.LongTensor(10).random_(0, 1024)
        m = MyModule()
        self.checkGraphModule(m, (input,))

    def test_fx_and_or(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return (x & x, x | x)
        input = torch.LongTensor(10).random_(0, 1024)
        m = MyModule()
        self.checkGraphModule(m, (input,))

    def test_dict(self):
        if False:
            for i in range(10):
                print('nop')

        class MyDictMod(torch.nn.Module):

            def forward(self, d):
                if False:
                    while True:
                        i = 10
                return (d['3'].relu(), {'4': d['3'].neg()})
        input_dict = {'3': torch.rand(3, 4)}
        m = MyDictMod()
        self.checkGraphModule(m, (input_dict,))

    def test_matmul_tracing(self):
        if False:
            while True:
                i = 10
        const = torch.randn(3)

        def matmul_f(x):
            if False:
                for i in range(10):
                    print('nop')
            return x @ const
        mod = symbolic_trace(matmul_f)
        inp = torch.randn(3)
        self.assertEqual(mod(inp), matmul_f(inp))

        def rmatmul_f(x):
            if False:
                while True:
                    i = 10
            return const @ x
        mod = symbolic_trace(rmatmul_f)
        inp = torch.randn(3)
        self.assertEqual(mod(inp), rmatmul_f(inp))

    @skipIfNoDynamoSupport
    def test_control_flow_tracing(self):
        if False:
            for i in range(10):
                print('nop')

        def true(x, y):
            if False:
                return 10
            return x + y

        def false(x, y):
            if False:
                while True:
                    i = 10
            return x - y

        def f(x, y):
            if False:
                return 10
            x = control_flow.cond(x[0] == 0, true, false, [x, y])
        with self.assertRaisesRegex(RuntimeError, 'Expected pred to be bool or tensor, but got Proxy\\(eq\\)'):
            _ = symbolic_trace(f)

    def test_disallow_override(self):
        if False:
            return 10

        class NoMutableCallTracer(Tracer):

            def create_node(self, kind: str, target: Union[str, Callable], args: Tuple[Argument, ...], kwargs: Dict[str, Any], name: Optional[str]=None, type_expr: Optional[Any]=None) -> Node:
                if False:
                    return 10
                name = target if isinstance(target, str) else torch.typename(target)
                if name[-1] == '_':
                    raise RuntimeError('In-place operations are not supported')
                return super().create_node(kind, target, args, kwargs, name)

        class MyInplaceMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                x.add_(3.0)
                return x
        m = MyInplaceMod()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m)

        class MyInplaceMod2(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                torch.log_(x)
                return x
        m2 = MyInplaceMod2()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m2)

        class MyInplaceMod3(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                y = torch.ones(3, 4)
                y.add_(x)
                return x
        m3 = MyInplaceMod3()
        with self.assertRaisesRegex(RuntimeError, 'In-place operations'):
            NoMutableCallTracer().trace(m3)

    def test_leaf_module(self):
        if False:
            while True:
                i = 10

        class NoLeafModulesTracer(Tracer):

            def is_leaf_module(self, m, qualname):
                if False:
                    for i in range(10):
                        print('nop')
                return False

        class MyReluMod(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.relu(x)
        mrm = MyReluMod()
        sym = NoLeafModulesTracer().trace(mrm)
        for node in sym.nodes:
            self.assertNotEqual(node.op, 'call_module')
        sym.lint()

    def test_wrap(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(3 + 4 + 5, a_lifted_leaf((3, 4), 5))

        def to_trace(y):
            if False:
                return 10
            return a_lifted_leaf((4, y), 3) + a_lifted_leaf((3, 4), 5) + a_lifted_leaf((y, y), y)
        m = symbolic_trace(to_trace)
        self.assertIn('a_lifted_leaf', m.code)
        self.assertEqual(27, m(2))
        self.assertIs(a_lifted_leaf, real_a_lifed_leaf)

    def test_wrap_fn_directly(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(3 + 4 + 5, a_lifted_leaf2((3, 4), 5))

        def to_trace(y):
            if False:
                return 10
            return a_lifted_leaf2((4, y), 3) + a_lifted_leaf2((3, 4), 5) + a_lifted_leaf2((y, y), y)
        m = symbolic_trace(to_trace)
        self.assertIn('a_lifted_leaf2', m.code)
        self.assertEqual(27, m(2))
        self.assertIs(a_lifted_leaf2, real_a_lifed_leaf2)

    def test_wrapped_via_decorator(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            if False:
                while True:
                    i = 10
            return wrapped_via_decorator(y)
        m = symbolic_trace(to_trace)
        self.assertIn('wrapped_via_decorator', m.code)
        self.assertEqual(m(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, '__fx_already_patched'))

    def test_wrapped_via_decorator_and_transformed(self):
        if False:
            while True:
                i = 10
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            if False:
                i = 10
                return i + 15
            return wrapped_via_decorator(y)
        m = symbolic_trace(to_trace)
        self.assertIn('wrapped_via_decorator', m.code)
        self.assertEqual(m(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, '__fx_already_patched'))
        transformed = torch.fx.Transformer(m).transform()
        self.assertIn('wrapped_via_decorator', transformed.code)
        self.assertEqual(transformed(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, '__fx_already_patched'))

    def test_wrap_with_submodule(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)

            def forward(self, x: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                return wrapped_with_submodule(x, self.batchnorm1d)
        m = symbolic_trace(M())
        self.assertIn('wrapped_with_submodule', m.code)
        input = torch.rand(3, 2)
        ref_batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)
        self.assertEqual(ref_batchnorm1d(input), m(input))

    def test_wrapped_retrace(self):
        if False:
            return 10

        def to_trace(y):
            if False:
                return 10
            return wrapped_via_decorator(y)
        m = symbolic_trace(to_trace)
        self.assertIn('wrapped_via_decorator', m.code)
        self.assertEqual(m(0), 1)
        retraced = symbolic_trace(m)
        self.assertIn('wrapped_via_decorator', retraced.code)
        self.assertEqual(retraced(0), 1)

    def test_wrap_decorated_function(self):
        if False:
            for i in range(10):
                print('nop')

        def to_trace(y):
            if False:
                i = 10
                return i + 15
            return wrapped_decorated_fn(y)
        m = symbolic_trace(to_trace)
        self.assertIn('wrapped_decorated_fn', m.code)
        self.assertEqual(m(1), 1)

    def test_graph_edit_with_proxy(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    print('Hello World!')
                return a + b
        m = M()
        g = symbolic_trace(m).graph
        new_g = torch.fx.Graph()
        val_map: Dict[Node, Node] = {}
        output_val = new_g.graph_copy(g, val_map)
        t = Proxy(output_val)
        new_g.output((t + t).node)
        gm = GraphModule(m, new_g)
        gm.graph.lint()
        self.assertEqual(gm(3, 4), 14)

    def test_concrete_arg_none_assert(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def forward(self, x, val=None):
                if False:
                    print('Hello World!')
                return x if val is None else x + val
        f = Foo()
        traced = torch.fx.symbolic_trace(f, concrete_args={'val': None})
        with self.assertRaisesRegex(AssertionError, 'val has been specialized to have value None'):
            traced(torch.randn(5), torch.randn(5))
        x = torch.randn(5)
        torch.testing.assert_close(traced(x), f(x))

    def test_trace_multiple_funcs(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    return 10
                return x + y

            def minus_forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x - y

            def multiply_forward(self, x, y):
                if False:
                    i = 10
                    return i + 15
                return x * y
        f = Foo()
        (x, y) = (torch.randn(5), torch.randn(5))
        print(torch.__version__)
        tracer = Tracer()
        torch.testing.assert_close(GraphModule(f, tracer.trace(f))(x, y), f(x, y))
        tracer.traced_func_name = 'minus_forward'
        torch.testing.assert_close(GraphModule(f, tracer.trace(f))(x, y), f.minus_forward(x, y))
        tracer.traced_func_name = 'multiply_forward'
        torch.testing.assert_close(GraphModule(f, tracer.trace(f))(x, y), f.multiply_forward(x, y))
        tracer.traced_func_name = 'add_forward'
        with self.assertRaisesRegex(AssertionError, "doesn't exist in"):
            tracer.trace(f)

    def test_graph_unique_names(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    return 10
                return a + b
        m = M()
        g = symbolic_trace(m).graph
        new_g = torch.fx.Graph()
        val_map: Dict[Node, Node] = {}
        output_val = new_g.graph_copy(g, val_map)
        t = Proxy(output_val)
        new_g.output((t + t).node)
        gm = GraphModule(m, new_g)
        seen_names: Set[str] = set()
        for node in gm.graph.nodes:
            assert node.name not in seen_names
            seen_names.add(node.name)

    def test_stack_traces(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    print('Hello World!')
                return a + b
        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(M())
        orig_graph_nodes = list(graph.nodes)
        for node in orig_graph_nodes:
            if node.op == 'output':
                continue
            self.assertTrue(node.stack_trace is not None)
            assert 'test_fx.py' in node.stack_trace
            new_node = graph.node_copy(node)
            self.assertTrue(new_node.stack_trace is not None)
            assert 'test_fx.py' in new_node.stack_trace

    def test_stack_traces_with_transformer(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    for i in range(10):
                        print('nop')
                return a + b
        tracer = torch.fx.Tracer()
        tracer.record_stack_traces = True
        graph = tracer.trace(M())
        gm = GraphModule(tracer.root, graph)
        new_gm = Transformer(gm).transform()
        for node in new_gm.graph.nodes:
            if node.op in {'placeholder', 'output'}:
                continue
            self.assertTrue(node.stack_trace is not None)
            assert 'test_fx.py' in node.stack_trace

    def test_lineno_map(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    i = 10
                    return i + 15
                a = torch.sin(a)
                b = torch.cos(b)
                return a + b
        tracer = torch.fx.Tracer()
        graph = tracer.trace(M())
        gm = GraphModule(tracer.root, graph)
        expected = {1: 2, 2: 3, 3: 4, 4: 5}
        self.assertTrue(set(expected.items()).issubset(set(gm._lineno_map.items())))

        def transform_code(code):
            if False:
                print('Hello World!')
            return ["print('hello!')\n", *code]
        gm.graph.on_generate_code(lambda _: transform_code)
        gm.recompile()
        expected = {2: 2, 3: 3, 4: 4, 5: 5}
        self.assertTrue(set(expected.items()).issubset(set(gm._lineno_map.items())))

    def test_graph_unique_names_manual(self):
        if False:
            while True:
                i = 10
        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_module', 'linear_mod', args=(a,), name='foo_1_1')
        c: torch.fx.Node = graph.create_node('get_attr', 'y_attr', name='foo_1')
        d: torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        graph.output(d)
        graph2 = torch.fx.Graph()
        val_map: Dict[Node, Node] = {}
        graph2.graph_copy(graph, val_map)
        seen_names: Set[str] = set()
        for node in graph2.nodes:
            assert node.name not in seen_names
            seen_names.add(node.name)

    def test_unpack(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    print('Hello World!')
                (c, d) = a
                return c + d + b
        a = (torch.rand(1), torch.rand(1))
        b = torch.rand(1)
        m = M()
        self.checkGraphModule(m, (a, b))

    def test_native_callable(self):
        if False:
            for i in range(10):
                print('nop')
        if IS_FBCODE or IS_WINDOWS or IS_MACOS:
            raise unittest.SkipTest('non-portable load_library call used in test')

        class MySimpleMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return 3.0 * x + x
        msm = MySimpleMod()

        def lower_to_elementwise_interpreter(orig_mod: torch.nn.Module) -> torch.nn.Module:
            if False:
                while True:
                    i = 10
            mod = symbolic_trace(orig_mod)
            instructions = []
            constant_idx = 0
            constants = {}
            fn_input_names = []
            target_to_name = {operator.add: 'add', operator.mul: 'mul'}
            output_node: Optional[Node] = None
            for n in mod.graph.nodes:
                (target, args, out_name) = (n.target, n.args, n.name)
                assert len(n.kwargs) == 0, 'kwargs currently not supported'
                if n.op == 'placeholder':
                    fn_input_names.append(target)
                elif n.op == 'call_function':
                    assert target in target_to_name, 'Unsupported call target ' + target
                    arg_names = []
                    for arg in args:
                        if not isinstance(arg, Node):
                            arg_name = f'constant_{constant_idx}'
                            constants[arg_name] = torch.tensor([arg] if isinstance(arg, numbers.Number) else arg)
                            arg_names.append(arg_name)
                            constant_idx += 1
                        else:
                            arg_names.append(arg.name)
                    instructions.append((target_to_name[target], arg_names, out_name))
                elif n.op == 'output':
                    if output_node is not None:
                        raise RuntimeError('Multiple output nodes!')
                    output_node = n
                else:
                    raise RuntimeError('Unsupported opcode ' + n.op)
            interpreter = torch.classes._TorchScriptTesting._ElementwiseInterpreter()
            for (k, v) in constants.items():
                interpreter.add_constant(k, v)
            interpreter.set_input_names(fn_input_names)
            interpreter.set_instructions(instructions)
            assert isinstance(output_node.args[0], torch.fx.Node)
            interpreter.set_output_name(output_node.args[0].name)

            class WrapperModule(torch.nn.Module):

                def __init__(self, interpreter):
                    if False:
                        i = 10
                        return i + 15
                    super().__init__()
                    self.interpreter = interpreter
            wrapper = WrapperModule(interpreter)
            graph = torch.fx.Graph()
            placeholder_nodes = []
            for name in fn_input_names:
                placeholder_nodes.append(graph.create_node('placeholder', name))
            interpreter_node = graph.create_node('get_attr', 'interpreter')
            output_node = graph.create_node(op='call_method', target='__call__', args=(interpreter_node, placeholder_nodes))
            graph.output(output_node)
            graph.lint()
            return GraphModule(wrapper, graph)
        lowered = lower_to_elementwise_interpreter(msm)
        x = torch.rand(3, 4)
        ref_out = msm(x)
        test_out = lowered(x)
        torch.testing.assert_close(test_out, ref_out)
        scripted_lowered = torch.jit.script(lowered)
        script_out = scripted_lowered(x)
        torch.testing.assert_close(script_out, ref_out)
        import_copy = self.getExportImportCopy(scripted_lowered)
        imported_out = import_copy(x)
        torch.testing.assert_close(imported_out, ref_out)

    def test_reserved_getattr(self):
        if False:
            return 10
        'Ensure that we do not name any nodes with a reserved builtin like `getattr`'

        class M(torch.nn.Module):

            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                return a.foo.bar.baz
        m = M()
        m_g = symbolic_trace(m)
        m_g.graph.lint()
        for node in m_g.graph.nodes:
            self.assertTrue(node.name != 'getattr')

    @unittest.skip('Hotfix for SEV remediation')
    def test_trace_buffer_slice(self):
        if False:
            return 10
        (bs, d_hid) = (10, 23)

        class ExampleCode(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.mm_param = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.mm_param2 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
                self.lin = torch.nn.Linear(d_hid, d_hid)
                self.register_buffer('buffer', torch.randn(bs + 100, d_hid))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = torch.mm(x, self.mm_param)
                skip_connection = x
                x = torch.relu(x)
                x = torch.mm(x, self.mm_param) + self.buffer[:x.shape[0]]
                x = self.lin(x)
                x = torch.relu(x)
                x = x + skip_connection
                x = torch.mm(x, self.mm_param2)
                x = self.lin(x)
                return x
        ec = ExampleCode()
        traced = torch.fx.symbolic_trace(ec)
        x = torch.randn(bs, d_hid)
        torch.testing.assert_close(ec(x), traced(x))

    def test_node_tagging(self):
        if False:
            while True:
                i = 10

        class TaggingTracer(Tracer):

            def create_node(self, kind: str, target: Union[str, Callable], args: Tuple[Argument, ...], kwargs: Dict[str, Any], name: Optional[str]=None, type_expr: Optional[Any]=None) -> Node:
                if False:
                    return 10
                n = super().create_node(kind, target, args, kwargs, name)
                n.tag = 'foo'
                return n

        class M(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    for i in range(10):
                        print('nop')
                return a + b
        m = M()
        g = TaggingTracer().trace(m)
        g.lint()
        for n in g.nodes:
            self.assertTrue(hasattr(n, 'tag'))
            self.assertEqual(n.tag, 'foo')

    def test_tensor_attribute(self):
        if False:
            for i in range(10):
                print('nop')

        class TensorAttribute(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.tensor = torch.rand(3, 4)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.nn.functional.linear(x, self.tensor)
        ta = TensorAttribute()
        traced = symbolic_trace(ta)
        traced(torch.rand(4, 4))

        class WrapperForQualname(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.ta = TensorAttribute()

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.nn.functional.linear(x, self.ta.tensor)
        wfq = WrapperForQualname()
        traced2 = symbolic_trace(wfq)
        traced2.graph.lint()
        traced2(torch.rand(4, 4))

    def test_tensor_attribute_coalseced(self):
        if False:
            while True:
                i = 10

        def count_attrs(fx_module):
            if False:
                print('Hello World!')
            targets = set()
            for node in traced.graph.nodes:
                if node.op == 'get_attr':
                    targets.add(node.target)
            return len(targets)
        val = torch.tensor(5)

        def f(x):
            if False:
                while True:
                    i = 10
            return x + val + val
        traced = symbolic_trace(f)
        traced.graph.lint()
        self.assertEqual(count_attrs(traced), 1)
        val2 = torch.tensor(5)

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            val = torch.tensor(5)
            return x + val + val2
        traced = symbolic_trace(f)
        traced.graph.lint()
        self.assertEqual(count_attrs(traced), 2)

    def test_symbolic_trace_sequential(self):
        if False:
            print('Hello World!')

        class Simple(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.neg(x)
        seq = torch.nn.Sequential(Simple(), Simple(), Simple())
        traced = symbolic_trace(seq)
        traced.graph.lint()
        x = torch.rand(3, 4)
        self.assertEqual(traced(x), seq(x))

    def test_tensor_constant(self):
        if False:
            while True:
                i = 10

        class ConstTensor(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.nn.functional.linear(x, torch.zeros(3, 4))
        ct = ConstTensor()
        traced = symbolic_trace(ct)
        traced.graph.lint()
        traced(torch.rand(4, 4))

    def test_pickle_graphmodule(self):
        if False:
            i = 10
            return i + 15

        class Nested(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.st = torch.nn.Linear(4, 4)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.st(x)
        n = Nested()
        traced = symbolic_trace(n)
        traced.graph.lint()
        pickled = pickle.dumps(traced)
        loaded = pickle.loads(pickled)
        loaded.graph.lint()
        x = torch.rand(3, 4)
        self.assertEqual(loaded(x), traced(x))

    def test_pickle_custom_import(self):
        if False:
            for i in range(10):
                print('nop')
        graph = torch.fx.Graph()
        a = graph.placeholder('x')
        b = graph.placeholder('y')
        c = graph.call_function(a_non_torch_leaf, (a, b))
        d = graph.call_function(torch.sin, (c,))
        graph.output(d)
        gm = GraphModule(torch.nn.Module(), graph)
        pickled = pickle.dumps(gm)
        loaded = pickle.loads(pickled)
        loaded.graph.lint()
        (x, y) = (torch.rand(1), torch.rand(1))
        self.assertEqual(loaded(x, y), gm(x, y))

    def test_all_input_nodes(self):
        if False:
            for i in range(10):
                print('nop')
        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.placeholder('x')
        b: torch.fx.Node = graph.call_module('linear_mod', args=(a,))
        c: torch.fx.Node = graph.get_attr('y_attr')
        d: torch.fx.Node = graph.call_function(operator.add, args=(b, c))
        e: torch.fx.Node = graph.call_function(torch.unsqueeze, args=(d, 0))
        graph.output(e)
        graph.lint()
        self.assertEqual(b.all_input_nodes, [a])
        self.assertEqual(c.all_input_nodes, [])
        self.assertEqual(d.all_input_nodes, [b, c])
        self.assertEqual(e.all_input_nodes, [d])

    def test_deepcopy_graphmodule_with_transform(self):
        if False:
            return 10
        st = SimpleTest()
        traced = symbolic_trace(st)
        traced.graph.lint()

        def transform(traced):
            if False:
                print('Hello World!')
            new_graph = torch.fx.Graph()
            val_map: Dict[Node, Node] = {}
            output_value = new_graph.graph_copy(traced.graph, val_map)
            relu_out = new_graph.create_node(op='call_method', target='neg', args=(output_value,), kwargs={})
            new_graph.output(relu_out)
            return GraphModule(traced, new_graph)
        transformed = transform(traced)
        transformed.graph.lint()
        copied = copy.deepcopy(transformed)
        self.assertNotEqual(id(type(transformed)), id(type(copied)))
        x = torch.randn(3, 4)
        self.assertEqual(copied(x), transformed(x))

    def test_deepcopy_with_submods_params(self):
        if False:
            for i in range(10):
                print('nop')

        class Bar(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))

            def forward(self, x):
                if False:
                    return 10
                return torch.relu(x) + self.param

        class Baz(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.bar = Bar()

            def forward(self, x):
                if False:
                    return 10
                return self.bar(x) - self.param
        baz = Baz()
        traced = symbolic_trace(baz)
        traced.graph.lint()
        copied = copy.deepcopy(traced)
        copied.graph.lint()

    def test_deepcopy_graph_with_tracer_cls(self):
        if False:
            return 10

        class TestTracer(Tracer):

            def is_leaf_module(self, module, name):
                if False:
                    print('Hello World!')
                return True
        g = Graph(tracer_cls=TestTracer)
        x = g.placeholder('x')
        g.output(x)
        h = copy.deepcopy(g)
        self.assertIsNotNone(h._tracer_cls)
        self.assertTrue(g._tracer_cls == h._tracer_cls)

    def test_unpack_list_better_error(self):
        if False:
            for i in range(10):
                print('nop')

        class SomeArgs(torch.nn.Module):

            def forward(self, a, b):
                if False:
                    i = 10
                    return i + 15
                return torch.rand(3, 4)

        class UnpacksList(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.sa = SomeArgs()

            def forward(self, x: list):
                if False:
                    for i in range(10):
                        print('nop')
                return self.sa(*x)
        ul = UnpacksList()
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ul)

    def test_unpack_dict_better_error(self):
        if False:
            for i in range(10):
                print('nop')

        class SomeKwargs(torch.nn.Module):

            def forward(self, x=3, y=4):
                if False:
                    i = 10
                    return i + 15
                return torch.rand(3, 4)

        class UnpacksDict(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.sk = SomeKwargs()

            def forward(self, x: dict):
                if False:
                    for i in range(10):
                        print('nop')
                return self.sk(**x)
        ud = UnpacksDict()
        with self.assertRaisesRegex(TraceError, 'Proxy object cannot be iterated.'):
            symbolic_trace(ud)

    def test_pretty_print_targets(self):
        if False:
            for i in range(10):
                print('nop')

        class SomeMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.add(x.foo + x.bar, 3.0)
        traced = symbolic_trace(SomeMod())
        graph_str = str(traced.graph)
        self.assertIn('builtins.getattr', graph_str)
        self.assertIn('operator.add', graph_str)
        self.assertIn('torch.add', graph_str)

    def test_pretty_print_node(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param: torch.nn.Parameter = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x: torch.Tensor, y: int=2):
                if False:
                    print('Hello World!')
                return self.linear(x[y] + self.param).clamp(min=0.0, max=1.0)
        traced = symbolic_trace(M())
        all_formatted = '\n'.join([n.format_node() for n in traced.graph.nodes])
        FileCheck().check('x').check('placeholder').check('y').check('placeholder').check('getitem').check('call_function').check('param').check('get_attr').check('add').check('call_function').check('linear').check('call_module').check('clamp').check('call_method').run(all_formatted)

    def test_script_tensor_constant(self):
        if False:
            return 10

        class IHaveATensorConstant(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + torch.rand(3, 4)
        traced = torch.fx.symbolic_trace(IHaveATensorConstant())
        torch.jit.script(traced)

    def test_autowrap_functions(self):
        if False:
            i = 10
            return i + 15

        class AutowrapFnTest(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return fx_int(x.shape[0] / 2)

        class AutowrapFnTest2(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return fx_int(x.shape[0] / 2) + fx_int_x2(x.shape[0] / 2)
        tracer = Tracer(autowrap_functions=(fx_int,))
        graph = tracer.trace(AutowrapFnTest())
        traced = GraphModule(tracer.root, graph, 'test')
        tracer_2 = Tracer(autowrap_functions=(fx_int, fx_int_x2))
        tracer_2.trace(AutowrapFnTest2())
        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(4)), 2)

    def test_tuple_no_subscript(self):
        if False:
            i = 10
            return i + 15

        def foo(x: Tuple):
            if False:
                i = 10
                return i + 15
            return x[0]
        traced = torch.fx.symbolic_trace(foo)
        x = (torch.randn(5, 3),)
        torch.testing.assert_close(traced(x), x[0])
        bio = io.BytesIO()
        torch.save(traced, bio)
        bio.seek(0)
        loaded = torch.load(bio)
        torch.testing.assert_close(loaded(x), x[0])

    def test_torch_fx_len(self):
        if False:
            print('Hello World!')

        class FXLenTest(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return len(x)
        traced = symbolic_trace(FXLenTest())
        self.assertEqual(traced(torch.rand(3, 4)), 3)
        scripted = torch.jit.script(FXLenTest())
        self.assertEqual(scripted(torch.rand(3)), 3)
        traced_scripted = torch.jit.script(traced)
        self.assertEqual(traced_scripted(torch.rand(3)), 3)

        class FXLenTest2(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.l = [3, 4, 5]

            def forward(self, x):
                if False:
                    return 10
                return x + len(self.l)
        traced2 = symbolic_trace(FXLenTest2())
        inp = torch.rand(3, 4)
        self.assertEqual(traced2(inp), inp + 3.0)
        self.assertIs(len, builtins.len)

    def test_torch_fx_getattr(self):
        if False:
            print('Hello World!')

        class FXGetattrTest(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return getattr(x, 'nonexistent_attr', torch.Tensor([2, 3]))
        traced = symbolic_trace(FXGetattrTest())
        self.assertEqual(traced(torch.rand(3, 4)), torch.Tensor([2, 3]))

    def test_sqrt(self):
        if False:
            for i in range(10):
                print('nop')

        class Sqrt1(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return sqrt(x.size(0))

        class Sqrt2(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return math.sqrt(x.size(0))

        class Sqrt3(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return x + math.sqrt(2) + sqrt(2)
        self.checkGraphModule(Sqrt1(), [torch.zeros(8)])
        self.checkGraphModule(Sqrt2(), [torch.zeros(8)])
        self.checkGraphModule(Sqrt3(), [torch.zeros(8)])
        self.assertIs(sqrt, _sqrt)
        self.assertIs(math.sqrt, _sqrt)

    def test_torch_custom_ops(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, a):
                if False:
                    i = 10
                    return i + 15
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))
        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint()
        out = gm(input)
        self.assertEqual(out, ref_out)

    def test_torch_op_overloads(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, a):
                if False:
                    for i in range(10):
                        print('nop')
                b = torch.ops.aten.add.Tensor(a, a)
                return b
        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint()
        out = gm(input)
        self.assertEqual(out, ref_out)
        for node in gm.graph.nodes:
            if node.op == 'call_function':
                assert isinstance(node.target, torch._ops.OpOverload)
                assert node.target.__name__ == 'add.Tensor'

    def test_pickle_torch_custom_ops(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, a):
                if False:
                    while True:
                        i = 10
                b = torch.ops.aten.sigmoid(a)
                c = torch.ops.aten.cat([a, b])
                return torch.ops.aten.cat((c, c))
        m = M()
        input = torch.randn(3)
        ref_out = m(input)
        gm = symbolic_trace(m)
        gm.graph.lint()
        pickled = pickle.dumps(gm)
        loaded = pickle.loads(pickled)
        self.assertEqual(loaded(input), gm(input))

    def test_pretty_print(self):
        if False:
            print('Hello World!')
        st = SimpleTest()
        traced = symbolic_trace(st)
        traced.graph.lint()
        printed = str(traced)
        assert 'SimpleTest()' in printed
        assert 'torch.relu' in printed

    def test_pretty_print_graph(self):
        if False:
            print('Hello World!')

        class KwargPrintTest(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.squeeze(x + 3.0, dim=2)
        st = KwargPrintTest()
        traced = symbolic_trace(st)
        traced.graph.lint()
        stringed = str(traced.graph)
        for s in ['args', 'kwargs', 'num_users']:
            assert s in stringed

    def test_custom_proxy_type(self):
        if False:
            while True:
                i = 10

        class TensorPair:

            def __init__(self, left, right):
                if False:
                    for i in range(10):
                        print('nop')
                (self.left, self.right) = (left, right)

            def add(self, other):
                if False:
                    print('Hello World!')
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                if False:
                    return 10
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair(x: TensorPair, y: TensorPair):
            if False:
                while True:
                    i = 10
            s = x.add(y)
            return s.mul(x)
        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        ref_out = use_tensor_pair(x, y)
        traced = symbolic_trace(use_tensor_pair)
        traced_out = traced(x, y)
        self.assertEqual(traced_out.left, ref_out.left)
        self.assertEqual(traced_out.right, ref_out.right)

    def test_custom_proxy_type_literal(self):
        if False:
            while True:
                i = 10

        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):

            def __init__(self, left, right):
                if False:
                    return 10
                (self.left, self.right) = (left, right)

            def add(self, other):
                if False:
                    i = 10
                    return i + 15
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                if False:
                    i = 10
                    return i + 15
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_literal(x: TensorPair):
            if False:
                i = 10
                return i + 15
            s = x.add(TensorPair(torch.zeros(5, 3), torch.zeros(5, 3)))
            return s.mul(x)
        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        ref_out = use_tensor_pair_literal(x)
        traced = symbolic_trace(use_tensor_pair_literal)
        traced_out = traced(x)
        self.assertEqual(traced_out.left, ref_out.left)
        self.assertEqual(traced_out.right, ref_out.right)

    def test_custom_proxy_dynamic_value(self):
        if False:
            return 10

        class TensorPair(metaclass=torch.fx.ProxyableClassMeta):

            def __init__(self, left, right):
                if False:
                    return 10
                (self.left, self.right) = (left, right)

            def add(self, other):
                if False:
                    print('Hello World!')
                l = self.left + other.left
                r = self.right + other.right
                return TensorPair(l, r)

            def mul(self, other):
                if False:
                    return 10
                l = self.left * other.left
                r = self.right * other.right
                return TensorPair(l, r)

        def use_tensor_pair_ctor(x: TensorPair, y: torch.Tensor):
            if False:
                print('Hello World!')
            s = x.add(TensorPair(y, y))
            return s.mul(x)
        x = TensorPair(torch.randn(5, 3), torch.randn(5, 3))
        y = torch.randn(5, 3)
        ref_out = use_tensor_pair_ctor(x, y)
        traced = symbolic_trace(use_tensor_pair_ctor)
        traced_out = traced(x, y)
        self.assertEqual(traced_out.left, ref_out.left)
        self.assertEqual(traced_out.right, ref_out.right)

    def test_custom_proxy_input_dependent_control_flow(self):
        if False:
            while True:
                i = 10

        class ZeroTensor(metaclass=torch.fx.ProxyableClassMeta):

            def __init__(self, inp):
                if False:
                    print('Hello World!')
                if inp.sum() == 0:
                    self.is_zero = True
                    self.tensor = torch.tensor([])
                else:
                    self.is_zero = False
                    self.tensor = inp

            def add(self, other):
                if False:
                    print('Hello World!')
                if self.is_zero:
                    return ZeroTensor(other.tensor)
                elif other.is_zero:
                    return self

        def use_zero_tensor(x: torch.Tensor, y: torch.Tensor):
            if False:
                print('Hello World!')
            return ZeroTensor(x + y)
        (x, y) = (torch.randn(5, 3), torch.randn(5, 3))
        ref_out = use_zero_tensor(x, y)
        traced = symbolic_trace(use_zero_tensor)
        traced_out = traced(x, y)
        self.assertEqual(traced_out.is_zero, ref_out.is_zero)
        self.assertEqual(traced_out.tensor, ref_out.tensor)

    def test_graph_fns(self):
        if False:
            i = 10
            return i + 15
        g = Graph()
        a = g.placeholder('a')
        b = g.call_module('linear', (a,))
        c = g.get_attr('bias')
        d = g.call_method('add', (b, c))
        e = g.call_function(torch.sin, (d,))
        g.output(e)
        mod = torch.nn.Module()
        mod.linear = torch.nn.Linear(3, 4)
        mod.bias = torch.rand(4)
        gm = GraphModule(mod, g)
        gm.graph.lint()
        input = torch.rand(3)
        r = gm(input)
        ref = torch.sin(mod.linear(input) + mod.bias)
        self.assertEqual(r, ref)

    def test_remove_uses(self):
        if False:
            print('Hello World!')
        g: torch.fx.Graph = Graph()
        x: torch.fx.Node = g.placeholder('x')
        relu: torch.fx.Node = g.call_function(torch.relu, (x,))
        neg: torch.fx.Node = g.call_function(torch.neg, (relu,))
        g.output(neg)
        neg.replace_all_uses_with(relu)
        g.erase_node(neg)
        self.assertTrue(neg not in relu.users)

    def test_remove_uses_with_custom_filter(self):
        if False:
            for i in range(10):
                print('nop')
        g: torch.fx.Graph = Graph()
        x: torch.fx.Node = g.placeholder('x')
        relu: torch.fx.Node = g.call_function(torch.relu, (x,))
        neg: torch.fx.Node = g.call_function(torch.neg, (relu,))
        g.output(neg)
        neg.replace_all_uses_with(relu, lambda x: x != neg)
        self.assertTrue(neg in relu.users)

    def test_nonetype_annotation(self):
        if False:
            for i in range(10):
                print('nop')
        eb = torch.nn.EmbeddingBag(3, 4)
        symbolic_trace(eb)

    def test_pickle_nonetype_annotation(self):
        if False:
            while True:
                i = 10
        eb = torch.nn.EmbeddingBag(10, 3, mode='sum')
        traced = symbolic_trace(eb)
        pickled = pickle.dumps(traced)
        loaded = pickle.loads(pickled)
        loaded.graph.lint()
        input = torch.LongTensor([1, 2, 4, 5, 4, 3, 2, 9])
        offsets = torch.LongTensor([0, 4])
        self.assertEqual(loaded(input, offsets), traced(input, offsets))

    def test_return_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                if False:
                    print('Hello World!')
                return (x, x + x)
        original = M()
        traced = symbolic_trace(original)
        self.assertEqual(traced(torch.ones(1)), original.forward(torch.ones(1)))

    def test_construct_root_dict(self):
        if False:
            while True:
                i = 10
        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_module', 'foo.bar.baz', args=(a,))
        c: torch.fx.Node = graph.create_node('get_attr', 'zip.zap.zam')
        d: torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        graph.output(d)
        linear_mod: torch.nn.Module = torch.nn.Linear(3, 4)
        add_param: torch.Tensor = torch.rand(3, 4)
        gm: torch.fx.GraphModule = torch.fx.GraphModule({'foo.bar.baz': linear_mod, 'zip.zap.zam': add_param}, graph)
        gm.graph.lint()
        assert 'self.foo.bar.baz' in gm.code
        x: torch.Tensor = torch.rand(3, 3)
        out: torch.Tensor = gm(x)
        ref_out: torch.Tensor = linear_mod(x) + add_param
        self.assertEqual(out, ref_out)

    def test_symbolic_trace_assert(self):
        if False:
            for i in range(10):
                print('nop')

        class AssertsTensorShape(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                torch._assert(x.shape[1] > 4, 'assert_foobar')
                return x
        m = AssertsTensorShape()
        traced = symbolic_trace(m)
        traced(torch.rand(4, 5))
        with self.assertRaisesRegex(AssertionError, 'assert_foobar'):
            traced(torch.rand(4, 3))
        ms = torch.jit.script(m)
        with self.assertRaisesRegex(torch.jit.Error, 'assert_foobar'):
            ms(torch.rand(4, 3))

    def test_fx_create_arg(self):
        if False:
            i = 10
            return i + 15

        class CustomArgObject:

            def __init__(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                self.x = x
                self.y = y

            def __fx_create_arg__(self, tracer: torch.fx.Tracer):
                if False:
                    while True:
                        i = 10
                return tracer.create_node('call_function', CustomArgObject, args=(tracer.create_arg(self.x), tracer.create_arg(self.y)), kwargs={})

        class HasCustomArgObjectWhenLeaf(torch.nn.Module):

            def forward(self, o: CustomArgObject):
                if False:
                    i = 10
                    return i + 15
                for x in o.x:
                    o.y += x
                return o.y

        class Root(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.inner = HasCustomArgObjectWhenLeaf()

            def forward(self, x, y):
                if False:
                    return 10
                o = CustomArgObject(x, y)
                return self.inner(o)

        class CreateArgTracer(torch.fx.Tracer):

            def is_leaf_module(self, m, module_qualified_name):
                if False:
                    while True:
                        i = 10
                return type(m) is HasCustomArgObjectWhenLeaf
        m = Root()
        graph = CreateArgTracer().trace(m)
        gm = torch.fx.GraphModule(m, graph)
        assert 'CustomArgObject(' in gm.code

    def test_trace_fn_constant(self):
        if False:
            while True:
                i = 10
        some_constant = torch.rand(3, 4)

        def add_const(x):
            if False:
                i = 10
                return i + 15
            return some_constant + x
        traced = symbolic_trace(add_const)
        input = torch.rand(3, 4)
        self.assertEqual(traced(input), add_const(input))

    def test_copy_no_remap(self):
        if False:
            for i in range(10):
                print('nop')
        traced = symbolic_trace(SimpleTest())
        g = traced.graph
        copied = torch.fx.Graph()
        for node in g.nodes:
            copied.node_copy(node)
        with self.assertRaisesRegex(RuntimeError, 'does not belong to this Graph'):
            copied.lint()

    def test_wrong_topo(self):
        if False:
            print('Hello World!')
        graph: torch.fx.Graph = torch.fx.Graph()
        a: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_module', 'foo.bar.baz', args=(a,))
        c: torch.fx.Node = graph.create_node('get_attr', 'zip.zap.zam')
        d: torch.fx.Node = graph.create_node('call_function', operator.add, args=(b, c))
        graph.output(d)
        nodes = list(graph.nodes)
        nodes[3].append(nodes[2])
        with self.assertRaisesRegex(RuntimeError, 'was used before it has been defined'):
            graph.lint()

    def test_wrong_target_type(self):
        if False:
            while True:
                i = 10
        graph: torch.fx.Graph = torch.fx.Graph()
        with self.assertRaises(ValueError):
            n = torch.fx.Node(graph=graph, name='foo', op='call_function', target='foo', args=(), kwargs={})

    def test_example_shape_prop(self):
        if False:
            return 10

        class TestCase(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.attr = torch.randn(3, 4)
                self.submod = torch.nn.Linear(4, 4)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.neg(self.submod(x.relu() + self.attr))
        tc = TestCase()
        tc_traced = symbolic_trace(tc)
        ref_out = tc_traced(torch.rand(3, 4))
        shape_prop.ShapeProp(tc_traced).propagate(torch.rand(3, 4))
        opcodes = set()
        output_shape: Optional[torch.Shape] = None
        output_stride: Optional[Tuple[int]] = None
        for node in tc_traced.graph.nodes:
            opcodes.add(node.op)
            if node.op == 'output':
                output_shape = node.args[0].meta['tensor_meta'].shape
                output_stride = node.args[0].meta['tensor_meta'].stride
        self.assertEqual(opcodes, {'placeholder', 'get_attr', 'call_function', 'call_method', 'call_module', 'output'})
        self.assertEqual(output_shape, ref_out.shape)
        self.assertEqual(output_stride, ref_out.stride())

    def test_shape_prop_layout(self):
        if False:
            i = 10
            return i + 15

        class ConvTest(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv_mod = torch.nn.Conv2d(5, 5, 3)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv_mod(x)
        test_mod = ConvTest()
        traced = symbolic_trace(test_mod)
        x = torch.randn(5, 5, 224, 224)
        shape_prop.ShapeProp(traced).propagate(x)
        assert all((node.meta['tensor_meta'].memory_format is torch.contiguous_format for node in traced.graph.nodes))
        x_channels_last = x.contiguous(memory_format=torch.channels_last)
        traced.to(memory_format=torch.channels_last)
        shape_prop.ShapeProp(traced).propagate(x_channels_last)
        for node in traced.graph.nodes:
            if node.op in {'placeholder'}:
                self.assertEqual(node.meta['tensor_meta'].memory_format, torch.channels_last)

    def test_shape_prop_aggregate(self):
        if False:
            return 10

        class ReturnTwo(torch.nn.Module):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return (3, torch.sum(x))

        class UnderTest(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.rt = ReturnTwo()

            def forward(self, x):
                if False:
                    return 10
                return self.rt(x)
        ut = UnderTest()

        class RTTracer(torch.fx.Tracer):

            def is_leaf_module(self, m, module_qualified_name):
                if False:
                    while True:
                        i = 10
                return type(m) is ReturnTwo
        graph = RTTracer().trace(ut)
        mod = torch.fx.GraphModule(ut, graph)
        shape_prop.ShapeProp(mod).propagate(torch.rand(3, 4))
        for node in mod.graph.nodes:
            if node.op == 'call_module':
                assert 'tensor_meta' in node.meta
                tensor_meta = node.meta['tensor_meta']
                assert tensor_meta[0] == 3
                assert tensor_meta[1].shape == torch.Size([])

    def test_shape_prop_layout_3d(self):
        if False:
            return 10

        class ConvTest3d(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.conv_mod = torch.nn.Conv3d(5, 5, 3)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.conv_mod(x)
        test_mod_3d = ConvTest3d()
        traced_3d = symbolic_trace(test_mod_3d)
        x_3d = torch.randn(5, 5, 224, 224, 15)
        shape_prop.ShapeProp(traced_3d).propagate(x_3d)
        assert all((node.meta['tensor_meta'].memory_format is torch.contiguous_format for node in traced_3d.graph.nodes))
        x_channels_last_3d = x_3d.contiguous(memory_format=torch.channels_last_3d)
        traced_3d.to(memory_format=torch.channels_last_3d)
        shape_prop.ShapeProp(traced_3d).propagate(x_channels_last_3d)
        for node in traced_3d.graph.nodes:
            if node.op in {'placeholder'}:
                self.assertEqual(node.meta['tensor_meta'].memory_format, torch.channels_last_3d)

    def test_nn_module_stack(self):
        if False:
            while True:
                i = 10

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv_mod = torch.nn.Conv2d(64, 64, (3, 3), padding=1, bias=False)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.conv_mod(x)

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.sub_mod = SubModule()

            def forward(self, x):
                if False:
                    return 10
                return self.sub_mod(x)
        m = MyModule()
        gm = torch.fx.symbolic_trace(m)
        mod_stack = {}
        expected_stack = [('sub_mod', type(m.sub_mod)), ('sub_mod.conv_mod', type(m.sub_mod.conv_mod))]
        for node in gm.graph.nodes:
            mod_stack = node.meta.get('nn_module_stack', {})
            if mod_stack:
                break
        stack_list = list(mod_stack.items())
        self.assertEqual(stack_list, expected_stack)

    def test_transformer_preserves_nn_module_stack_for_get_attr(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.weight = torch.nn.Parameter(torch.ones(1, 1))

            def forward(self, x):
                if False:
                    return 10
                return self.weight + x
        tracer = torch.fx.Tracer()
        graph = tracer.trace(M())
        gm = GraphModule(tracer.root, graph)
        for node in gm.graph.nodes:
            if node.op == 'get_attr':
                node.meta['nn_module_stack'] = 'self'
                node.meta['stack_trace'] = 'stack_trace'
                node.meta['source_fn_stack'] = 'source_fn_stack'
        new_gm = Transformer(gm).transform()
        for node in new_gm.graph.nodes:
            if node.op == 'get_attr':
                self.assertEqual(node.meta['nn_module_stack'], 'self')
                self.assertEqual(node.meta['stack_trace'], 'stack_trace')
                self.assertEqual(node.meta['source_fn_stack'], 'source_fn_stack')

    def test_interpreter(self):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                if False:
                    return 10
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)
        m = MyModule()
        gm = torch.fx.symbolic_trace(m)
        interpreter = Interpreter(gm)
        input = torch.randn(3, 4)
        self.assertEqual(interpreter.run(input), gm(input))
        self.assertEqual(interpreter.run(input), m(input))

    def test_interpreter_run_node_override(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                if False:
                    return 10
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)
        m = MyModule()
        gm = torch.fx.symbolic_trace(m)

        class RunNodeInterpreter(Interpreter):

            def __init__(self, module):
                if False:
                    print('Hello World!')
                super().__init__(module)

            def run_node(self, n: Node) -> Any:
                if False:
                    while True:
                        i = 10
                result = super().run_node(n)
                n.cached_value = result
                return result
        input = torch.randn(3, 4)
        RunNodeInterpreter(gm).run(input)
        for node in gm.graph.nodes:
            assert hasattr(node, 'cached_value')

    def test_interpreter_onthefly_swap(self):
        if False:
            return 10

        def fn(x):
            if False:
                return 10
            return torch.sigmoid(x).neg()
        gm = torch.fx.symbolic_trace(fn)

        class NegSigmSwapInterpreter(Interpreter):

            def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                if False:
                    while True:
                        i = 10
                if target == torch.sigmoid:
                    return torch.neg(*args, **kwargs)
                return super().call_function(n)

            def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                if False:
                    for i in range(10):
                        print('nop')
                if target == 'neg':
                    (call_self, *args_tail) = args
                    return call_self.sigmoid(*args_tail, **kwargs)
                return super().call_method(n)
        input = torch.randn(3, 4)
        result = NegSigmSwapInterpreter(gm).run(input)
        self.assertEqual(result, torch.neg(input).sigmoid())

    def test_interpreter_partial_eval(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                if False:
                    return 10
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)
        gm = torch.fx.symbolic_trace(MyModule())
        interp = Interpreter(gm)
        env = {}
        for node in gm.graph.nodes:
            if node.op == 'call_module' and node.target == 'linear':
                env[node] = torch.arange(0, 12, 1).reshape(3, 4) - 6.0
                break
        assert len(env) == 1
        x = torch.randn(3, 4)
        result = interp.run(x, initial_env=env)
        self.assertEqual(result, (torch.arange(0, 12, 1).reshape(3, 4) - 6.0).clamp(0.0, 1.0))

    def test_interpreter_star_args(self):
        if False:
            print('Hello World!')

        def with_star_args(x, *args):
            if False:
                i = 10
                return i + 15
            return x + args[0]
        gm = torch.fx.symbolic_trace(with_star_args)
        interp = Interpreter(gm)
        result = interp.run(torch.ones(3, 4), torch.ones(3, 4), torch.rand(3, 4))
        self.assertEqual(result, torch.ones(3, 4) * 2.0)

    @skipIfNoTorchVision
    def test_interpreter_noop_resnet18(self):
        if False:
            while True:
                i = 10
        rn18 = torchvision_models.resnet18()
        transformed = torch.fx.Transformer(symbolic_trace(rn18)).transform()
        inp = torch.randn(5, 3, 224, 224)
        self.assertEqual(transformed(inp), rn18(inp))

    @skipIfNoTorchVision
    def test_interpreter_gc_values(self):
        if False:
            for i in range(10):
                print('nop')
        rn18 = torchvision_models.resnet18()
        interp = Interpreter(symbolic_trace(rn18))
        inp = torch.rand(5, 3, 224, 224)
        out = interp.run(inp)
        env_key_names = {n.name for n in interp.env.keys()}
        self.assertEqual(env_key_names, {'output'})

    def test_interpreter_default_args(self):
        if False:
            return 10

        class Model(torch.nn.Module):

            def forward(self, x, y=3.14159):
                if False:
                    return 10
                return x + y
        model = Model()
        gm = torch.fx.symbolic_trace(model)
        interp = Interpreter(gm)
        x = torch.randn(5, 3)
        out = interp.run(x)
        torch.testing.assert_close(out, x + 3.14159)

    def test_interpreter_not_enough_args(self):
        if False:
            i = 10
            return i + 15

        class Model(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    while True:
                        i = 10
                return x + y
        model = Model()
        gm = torch.fx.symbolic_trace(model)
        interp = Interpreter(gm)
        x = torch.randn(5, 3)
        with self.assertRaisesRegex(RuntimeError, 'Expected positional argument for parameter y, but one was not passed in'):
            out = interp.run(x)

    def test_transformer_noop(self):
        if False:
            while True:
                i = 10

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.linear(x + self.param).clamp(min=0.0, max=1.0)
        m = MyModule()
        gm = torch.fx.symbolic_trace(m)
        new_gm = Transformer(gm).transform()
        input = torch.randn(3, 4)
        self.assertEqual(new_gm(input), gm(input))

    def test_transformer_op_swap(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.sigmoid(x).neg()
        gm = torch.fx.symbolic_trace(fn)

        class NegSigmSwapXformer(Transformer):

            def call_function(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                if False:
                    while True:
                        i = 10
                if target == torch.sigmoid:
                    return torch.neg(*args, **kwargs)
                return super().call_function(n)

            def call_method(self, target: Target, args: Tuple, kwargs: Dict) -> Any:
                if False:
                    while True:
                        i = 10
                if target == 'neg':
                    (call_self, *args_tail) = args
                    return call_self.sigmoid(*args_tail, **kwargs)
                return super().call_method(n)
        transformed = NegSigmSwapXformer(gm).transform()
        input = torch.randn(3, 4)
        self.assertEqual(transformed(input), torch.neg(input).sigmoid())

    def test_transformer_multi_outputs(self):
        if False:
            for i in range(10):
                print('nop')

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.param = torch.nn.Parameter(torch.rand(3, 4))
                self.linear = torch.nn.Linear(4, 5)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = x + self.param
                out = self.linear(x)
                return (x, out)
        m = MyModule()
        gm = torch.fx.symbolic_trace(m)
        new_gm = Transformer(gm).transform()
        input = torch.randn(3, 4)
        self.assertEqual(new_gm(input), gm(input))

    def test_fn_type_annotations(self):
        if False:
            return 10

        class Foo(torch.nn.Module):

            def forward(self, p: Pair, z: torch.Tensor, i: int) -> Dict[str, torch.Tensor]:
                if False:
                    i = 10
                    return i + 15
                return {'a': p.x + p.y + z + i}
        foo_scripted = torch.jit.script(Foo())
        foo_scripted(Pair(torch.rand(5), torch.rand(5)), torch.rand(5), 3)
        fxed = symbolic_trace(Foo())
        fxed_scripted = torch.jit.script(fxed)
        fxed_scripted(Pair(torch.rand(5), torch.rand(5)), torch.rand(5), 3)

    def test_fn_type_annotation_empty(self):
        if False:
            while True:
                i = 10

        def forward(a: List[torch.Tensor]):
            if False:
                for i in range(10):
                    print('nop')
            return a[0]
        torch.jit.script(symbolic_trace(forward))

    def test_wrapped_method(self):
        if False:
            for i in range(10):
                print('nop')

        def wrap_with_relu(fn):
            if False:
                i = 10
                return i + 15

            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                if False:
                    i = 10
                    return i + 15
                return torch.relu(fn(*args, **kwargs))
            return wrapper

        class Foo(torch.nn.Module):

            @wrap_with_relu
            def forward(self, x, w):
                if False:
                    return 10
                return torch.matmul(x, w)
        f = Foo()
        traced = symbolic_trace(f)
        (x, w) = (torch.rand(3, 4), torch.rand(4, 4))
        self.assertTrue(any((n.target == torch.relu for n in traced.graph.nodes)))

    def test_empty_graph_codegen(self):
        if False:
            while True:
                i = 10
        graph = torch.fx.Graph()
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(gm(), None)

    def test_sequential(self):
        if False:
            while True:
                i = 10
        m = torch.nn.Sequential(torch.nn.Conv2d(1, 1, 1))
        gm = torch.fx.symbolic_trace(m)
        gm_copy = copy.deepcopy(gm)

    def test_ctx_mgr(self):
        if False:
            while True:
                i = 10

        @contextlib.contextmanager
        def do_nothing():
            if False:
                for i in range(10):
                    print('nop')
            yield

        class M(torch.nn.Module):

            @do_nothing()
            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return torch.relu(x)
        m = M()
        self.checkGraphModule(m, (torch.rand(3, 4),))

    def test_typename_print(self):
        if False:
            return 10
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,), type_expr=List[float])
        output: torch.fx.Node = graph.output(b)
        self.assertTrue('typing.List[float]' in str(graph))

    def test_layout(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.empty_like(x, layout=torch.strided, pin_memory=False).fill_(0)
        traced = symbolic_trace(M())
        x = torch.rand(5, 9, 3, 4)
        self.assertEqual(traced(x), torch.zeros_like(x))

    def test_ellipsis(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y[:, 1:10, ...]
        traced = symbolic_trace(M())
        (x, y) = (torch.rand(5, 9, 3, 4), torch.rand(5, 15, 3, 4))
        self.assertEqual(traced(x, y), x + y[:, 1:10, ...])

    def test_inf_nan(self):
        if False:
            while True:
                i = 10

        class FooMod(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (x + float('inf'), x + float('-inf'), x + float('nan'))
        fm = FooMod()
        self.checkGraphModule(fm, (torch.rand(3, 4),))

    def test_inf_nan_kwds(self):
        if False:
            for i in range(10):
                print('nop')
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_function', operator.add, (x, float('inf')), {}, name='inf')
        c: torch.fx.Node = graph.create_node('call_function', operator.add, (x, float('nan')), {}, name='nan')
        graph.output((b, c))
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        x = torch.rand(3, 4)
        self.assertEqual(gm(x), (x + float('inf'), x + float('nan')))

    def test_deepcopy_recursion_depth(self):
        if False:
            for i in range(10):
                print('nop')
        depth = sys.getrecursionlimit() + 20
        g = torch.fx.Graph()
        x = g.placeholder('x')
        for i in range(depth):
            x = g.call_function(torch.relu, (x,))
        g.output(x)
        copied_graph = copy.deepcopy(g)
        val_map = {}
        for (orig_node, new_node) in zip(g.nodes, copied_graph.nodes):
            val_map[orig_node] = new_node
        for (orig_node, new_node) in zip(g.nodes, copied_graph.nodes):
            orig_users = set(orig_node.users.keys())
            orig_users_equiv = {val_map[u] for u in orig_users}
            new_users = set(new_node.users.keys())
            self.assertEqual(orig_users_equiv, new_users)

    @skipIfNoTorchVision
    def test_replace_uses(self):
        if False:
            for i in range(10):
                print('nop')
        rn18 = torchvision_models.resnet18()

        class LowerReluTracer(torch.fx.Tracer):

            def is_leaf_module(self, m: torch.nn.Module, qualname: str):
                if False:
                    i = 10
                    return i + 15
                if isinstance(m, torch.nn.ReLU):
                    return False
                return super().is_leaf_module(m, qualname)
        rn18_traced = GraphModule(rn18, LowerReluTracer().trace(rn18))
        to_erase = []
        for node in rn18_traced.graph.nodes:
            if node.op == 'call_function' and node.target in [torch.relu, torch.nn.functional.relu]:
                kwargs = node.kwargs.copy()
                kwargs.pop('inplace')
                with rn18_traced.graph.inserting_before(node):
                    new_node = rn18_traced.graph.call_function(the_function=torch.neg, args=node.args, kwargs=node.kwargs)
                node.replace_all_uses_with(replace_with=new_node)
                to_erase.append(node)
        for node in to_erase:
            rn18_traced.graph.erase_node(node)

    def test_replace_input(self):
        if False:
            for i in range(10):
                print('nop')
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        y: torch.fx.Node = graph.create_node('placeholder', 'y')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output: torch.fx.Node = graph.output(b)
        b.replace_input_with(x, y)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        input_x = torch.randn(33, 44)
        input_y = torch.randn(11, 22)
        self.assertEqual(gm(input_x, input_y), torch.relu(input_y))

    def test_insertion_point(self):
        if False:
            while True:
                i = 10
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output: torch.fx.Node = graph.output(b)
        with graph.inserting_before(b):
            neg: torch.fx.Node = graph.call_function(the_function=torch.neg, args=(x,))
            (_, *relu_args) = b.args
            b.args = (neg, *relu_args)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        input = torch.randn(33, 44)
        self.assertEqual(gm(input), torch.relu(torch.neg(input)))

    def test_update_args_api(self):
        if False:
            print('Hello World!')
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        y: torch.fx.Node = graph.create_node('placeholder', 'y')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output: torch.fx.Node = graph.output(b)
        orig_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        (inp_x, inp_y) = (torch.randn(5, 3), torch.randn(3, 5))
        self.assertEqual(orig_gm(inp_x, inp_y), torch.relu(inp_x))
        b.update_arg(0, y)
        new_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(new_gm(inp_x, inp_y), torch.relu(inp_y))

    def test_update_kwargs_api(self):
        if False:
            while True:
                i = 10
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        y: torch.fx.Node = graph.create_node('placeholder', 'y')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, kwargs={'input': x})
        output: torch.fx.Node = graph.output(b)
        orig_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        (inp_x, inp_y) = (torch.randn(5, 3), torch.randn(3, 5))
        self.assertEqual(orig_gm(inp_x, inp_y), torch.relu(inp_x))
        b.update_kwarg('input', y)
        new_gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        self.assertEqual(new_gm(inp_x, inp_y), torch.relu(inp_y))

    def test_immutable_list_pytree_ops(self):
        if False:
            i = 10
            return i + 15
        rand_tensor = torch.randn(5, 3)
        l = immutable_list([3, [rand_tensor, 42]])
        (flattened, spec) = pytree.tree_flatten(l)
        assert flattened == [3, rand_tensor, 42]
        unflattened = pytree.tree_unflatten(flattened, spec)
        assert unflattened == l
        assert isinstance(unflattened, immutable_list)

    def test_immutable_dict_pytree_ops(self):
        if False:
            return 10
        rand_tensor = torch.randn(5, 3)
        d = immutable_dict({'a': 3, 'b': [rand_tensor, 42]})
        (flattened, spec) = pytree.tree_flatten(d)
        assert flattened == [3, rand_tensor, 42]
        unflattened = pytree.tree_unflatten(flattened, spec)
        assert unflattened == d
        assert isinstance(unflattened, immutable_dict)

    def test_move_before(self):
        if False:
            return 10
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output: torch.fx.Node = graph.output(b)
        neg: torch.fx.Node = graph.call_function(the_function=torch.neg, args=(x,))
        (_, *relu_args) = b.args
        b.args = (neg, *relu_args)
        b.prepend(neg)
        gm = torch.fx.GraphModule(torch.nn.Module(), graph)
        input = torch.randn(33, 44)
        self.assertEqual(gm(input), torch.relu(torch.neg(input)))

    def test_prepend_self(self):
        if False:
            return 10
        graph: torch.fx.Graph = torch.fx.Graph()
        x: torch.fx.Node = graph.create_node('placeholder', 'x')
        b: torch.fx.Node = graph.create_node('call_function', target=torch.relu, args=(x,))
        output: torch.fx.Node = graph.output(b)
        b.prepend(b)
        x.append(b)
        self.assertEqual(len(graph.nodes), 3)

    def test_erase_node_error(self):
        if False:
            print('Hello World!')
        st = SimpleTest()
        traced = symbolic_trace(st)
        for node in traced.graph.nodes:
            if node.target in [operator.add, torch.relu]:
                with self.assertRaisesRegex(RuntimeError, 'but it still had .* users in the graph'):
                    traced.graph.erase_node(node)

    def test_copy_it(self):
        if False:
            return 10
        d = immutable_dict([(3, 4), (5, 6)])
        l = immutable_list([(3, 4), (5, 6)])
        self.assertEqual(d, deepcopy(d))
        self.assertEqual(l, deepcopy(l))

    def test_get_torch_func_signature(self):
        if False:
            print('Hello World!')
        for key in dir(torch):
            obj = getattr(torch, key)
            if callable(obj):
                schemas = get_signature_for_torch_op(obj)

    def test_find_uses(self):
        if False:
            while True:
                i = 10
        graph = torch.fx.Graph()
        x = torch.fx.Proxy(graph.placeholder('x'))
        y = torch.relu(x)
        z = x + x
        u = torch.neg(x)
        graph.output((y + z + u).node)
        graph.lint()
        users_of_x = x.node.users
        self.assertEqual(len(users_of_x), 3)
        expected_ops = {'relu', 'add', 'neg'}
        for use in users_of_x:
            assert any((use.name.startswith(prefix) for prefix in expected_ops))

    def test_inline_graph(self):
        if False:
            i = 10
            return i + 15

        class InlineInto(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.relu(x)

        class ToInline(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return torch.neg(x)
        inline_into = symbolic_trace(InlineInto())
        to_inline = symbolic_trace(ToInline())
        combined_graph = torch.fx.Graph()
        output_node = combined_graph.graph_copy(inline_into.graph, {})
        input_node = list(to_inline.graph.nodes)[0]
        assert input_node and input_node.op == 'placeholder'
        val_map = {input_node: output_node}
        output = combined_graph.graph_copy(to_inline.graph, val_map)
        combined_graph.output(output)
        combined_module = torch.fx.GraphModule(torch.nn.Module(), combined_graph)
        input = torch.rand(3, 4)
        self.assertEqual(combined_module(input), input.relu().neg())

    def test_multi_insert_point(self):
        if False:
            return 10
        graph = torch.fx.Graph()
        x = torch.fx.Proxy(graph.placeholder('x'))
        relu = torch.relu(x)
        with graph.inserting_before(relu.node):
            y = torch.neg(x)
            z = torch.tanh(y)
        graph.output((relu.node, z.node))
        graph.lint()
        expected_ops = ['x', 'neg', 'tanh', 'relu']
        for (node, expected) in zip(graph.nodes, expected_ops):
            assert expected in node.name

    def test_reassign_args_kwargs_uses(self):
        if False:
            for i in range(10):
                print('nop')
        graph = torch.fx.Graph()
        (x, y) = (Proxy(graph.placeholder('x')), Proxy(graph.placeholder('y')))
        z = x + y
        zed = z + z + z
        graph.output(zed.node)
        graph.lint()
        zed.node.args = (zed.node.args[0], x.node)
        self.assertEqual(list(x.node.users.keys()), [z.node, zed.node])
        z.node.args = (y.node, y.node)
        self.assertEqual(list(x.node.users.keys()), [zed.node])

    def test_trace_function(self):
        if False:
            i = 10
            return i + 15

        def foo(x, y):
            if False:
                while True:
                    i = 10
            return torch.relu(x) + y
        (x, y) = (torch.randn(3, 4), torch.randn(3, 4))
        self.checkGraphModule(foo, (x, y))

    def test_trace_return_dataclass(self):
        if False:
            i = 10
            return i + 15
        '\n        Test case for Module that return dataclass\n        '
        from dataclasses import dataclass

        @dataclass
        class MyOutput:
            foo: torch.Tensor
            bar: torch.Tensor

        class ModuleReturnDataclass(torch.nn.Module):

            def forward(self, d: torch.Tensor):
                if False:
                    print('Hello World!')
                return MyOutput(foo=d + d, bar=d * 3)
        module = ModuleReturnDataclass()
        traced_graph = symbolic_trace(module).graph
        print(traced_graph)
        gm = GraphModule(module, traced_graph)
        x = torch.rand(1)
        self.assertEqual(module(x), gm(x))

    def test_trace_return_dataclass_nested(self):
        if False:
            i = 10
            return i + 15
        '\n        Test case for Module that return dataclass\n        '
        from dataclasses import dataclass

        @dataclass
        class MyOutput:
            foo: torch.Tensor
            bar: torch.Tensor

        class ModuleReturnDataclass(torch.nn.Module):

            def forward(self, d: torch.Tensor):
                if False:
                    while True:
                        i = 10
                return MyOutput(foo=d + d, bar=d * 3)

        class CallsModule(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = ModuleReturnDataclass()

            def forward(self, x):
                if False:
                    return 10
                tmp = self.m(x)
                return MyOutput(foo=tmp.foo, bar=tmp.bar)
        module = CallsModule()
        traced_graph = symbolic_trace(module).graph
        print(traced_graph)
        gm = GraphModule(module, traced_graph)
        x = torch.rand(1)
        self.assertEqual(module(x), gm(x))

    def test_trace_return_namedtuple(self):
        if False:
            return 10
        '\n        Test case for Module that return namedtuple\n        '

        class MyOutput(NamedTuple):
            foo: torch.Tensor
            bar: torch.Tensor

        class ModuleReturnNamedTuple(torch.nn.Module):

            def forward(self, d: torch.Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                return MyOutput(foo=d, bar=d)
        module = ModuleReturnNamedTuple()
        traced_graph = symbolic_trace(module).graph
        print(traced_graph)
        gm = GraphModule(module, traced_graph)
        x = torch.rand(1)
        self.assertEqual(module(x), gm(x))

    def test_trace_dict_int_keys(self):
        if False:
            return 10

        class ModWithDictArg(torch.nn.Module):

            def forward(self, d: Dict[int, torch.Tensor]):
                if False:
                    i = 10
                    return i + 15
                return d[42]

        class CallsModWithDict(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = ModWithDictArg()

            def forward(self, x):
                if False:
                    return 10
                return self.m({42: x})

        class MyTracer(torch.fx.Tracer):

            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return isinstance(m, ModWithDictArg)
        traced_graph = MyTracer().trace(CallsModWithDict())

    def test_trace_dict_proxy_keys(self):
        if False:
            print('Hello World!')

        class ModWithDictArg(torch.nn.Module):

            def forward(self, d: Dict[torch.Tensor, torch.Tensor]):
                if False:
                    return 10
                return d[42]

        class CallsModWithDict(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.m = ModWithDictArg()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.m({x: x})

        class MyTracer(torch.fx.Tracer):

            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                if False:
                    return 10
                return isinstance(m, ModWithDictArg)
        with self.assertRaisesRegex(RuntimeError, 'cannot contain a Node'):
            traced_graph = MyTracer().trace(CallsModWithDict())

    def test_module_deepcopy_edit_nodes(self):
        if False:
            i = 10
            return i + 15

        class Foo(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.relu(x)
        traced1 = symbolic_trace(Foo())
        copied = copy.deepcopy(traced1)
        for node in copied.graph.nodes:
            if node.target == torch.relu:
                node.target = torch.neg
        copied.recompile()
        traced1.recompile()
        x = torch.randn(15, 15)
        torch.testing.assert_close(traced1(x), torch.relu(x))
        torch.testing.assert_close(copied(x), torch.neg(x))

    def test_direct_param_use(self):
        if False:
            for i in range(10):
                print('nop')

        class TransposeTest(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.b = torch.nn.Parameter(torch.rand(4, 3))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.b

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.a = TransposeTest()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return (self.a.b, self.a.b.t(), self.a.b.view(12))
        traced = torch.fx.symbolic_trace(Foo())
        assert all(('constant' not in node.target for node in traced.graph.nodes))

    def test_single_default_arg(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def forward(self, y=1):
                if False:
                    for i in range(10):
                        print('nop')
                return y
        m = M()
        self.checkGraphModule(m, ())
        self.checkGraphModule(m, (3,))

    def test_multiple_default_args(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, y=1, z=2):
                if False:
                    while True:
                        i = 10
                return y + z
        m = M()
        self.checkGraphModule(m, ())
        self.checkGraphModule(m, (3,))
        self.checkGraphModule(m, (3, 4))

    def test_regular_and_default_args(self):
        if False:
            while True:
                i = 10

        class M(torch.nn.Module):

            def forward(self, x, y=1):
                if False:
                    i = 10
                    return i + 15
                return x + y
        m = M()
        self.checkGraphModule(m, (2,))
        self.checkGraphModule(m, (2, 3))

    def test_string_literal_return(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self):
                if False:
                    for i in range(10):
                        print('nop')
                return 'foo'
        m = M()
        self.checkGraphModule(m, ())

    def test_namedtuple_return_qualname(self):
        if False:
            i = 10
            return i + 15

        class NamedTupReturn(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                return MyNamedTup(x, x)
        traced = symbolic_trace(NamedTupReturn())
        input = torch.rand(3, 4)
        self.assertEqual(traced(input), MyNamedTup(input, input))

    def test_update_args_kwargs_yells_at_you(self):
        if False:
            return 10
        symtraced = symbolic_trace(SimpleTest())
        node = next(iter(symtraced.graph.nodes))
        with self.assertRaisesRegex(AttributeError, '__update_args_kwargs'):
            node.__update_args_kwargs((), {})

    def test_torchbind_class_attribute_in_fx(self):
        if False:
            for i in range(10):
                print('nop')
        if IS_FBCODE or IS_WINDOWS or IS_MACOS:
            self.skipTest('torch.classes._TorchScriptTesting._StackString is registered, skipping')

        class FooBar1234(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._StackString(['3', '4'])

            def forward(self):
                if False:
                    return 10
                return self.f.top()
        m = FooBar1234()
        self.checkGraphModule(m, ())

    def test_torchbind_class_attribute_in_fx_tensor_arg(self):
        if False:
            print('Hello World!')
        if IS_FBCODE or IS_WINDOWS or IS_MACOS:
            self.skipTest('torch.classes._TorchScriptTesting._ReLUClass is registered, skipping')

        class FooBar2341(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.f = torch.classes._TorchScriptTesting._ReLUClass()

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.f.run(x)
        m = FooBar2341()
        traced = symbolic_trace(m)
        input = torch.randn(3, 4)
        self.assertEqual(traced(input), m(input))
        self.assertTrue(any((n.op == 'call_method' for n in traced.graph.nodes)))

    def test_script_method_trace(self):
        if False:
            for i in range(10):
                print('nop')

        class Scripted(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return torch.relu(x)

        class Holder(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.s = torch.jit.script(Scripted())

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.s(x)
        h = Holder()
        traced = symbolic_trace(h)
        input = torch.randn(3, 4)
        self.assertEqual(traced(input), h(input))
        self.assertTrue(any((n.op == 'call_method' for n in traced.graph.nodes)))

    def test_namedtuple_return_trace(self):
        if False:
            while True:
                i = 10

        class NamedTupReturn(torch.nn.Module):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return Pair(x, x)
        traced = symbolic_trace(NamedTupReturn())
        input = torch.rand(3, 4)
        self.assertEqual(traced(input), Pair(input, input))

    def test_named_tuple_inlined(self):
        if False:
            print('Hello World!')

        class NamedTupMod(torch.nn.Module):

            def forward(self, inp):
                if False:
                    for i in range(10):
                        print('nop')
                return wrapped_named_tup(Pair(inp, 1.2), p2=Pair(3.4, inp))
        m = NamedTupMod()
        input = torch.rand(3, 4)
        ref = m(input)
        traced = symbolic_trace(m)
        res = traced(input)
        self.assertEqual(ref, res)
        ph = call_func = None
        for node in traced.graph.nodes:
            if node.op == 'placeholder':
                ph = node
            elif node.op == 'call_function' and node.target == wrapped_named_tup:
                node.update_arg(0, Pair(ph, 1.2))
                node.update_kwarg('p2', Pair(3.4, ph))
                call_func = node
                break
        self.assertTrue(call_func is not None)
        self.assertTrue(isinstance(call_func.args[0], Pair))
        self.assertTrue(isinstance(call_func.kwargs['p2'], Pair))
        self.assertEqual(_format_arg(call_func.args[0]), 'Pair(x=%inp, y=1.2)')
        self.assertEqual(_format_arg(call_func.kwargs['p2']), 'Pair(x=3.4, y=%inp)')
        traced.graph.eliminate_dead_code()
        traced.recompile()
        res = traced(input)
        self.assertEqual(ref, res)

    def test_return_type_exists(self):
        if False:
            return 10

        class ReturnTypeModule(torch.nn.Module):

            def other(self, x: List[str]) -> List[str]:
                if False:
                    print('Hello World!')
                return x

            def forward(self, x: List[str]) -> List[str]:
                if False:
                    print('Hello World!')
                return self.other(x)
        traced = symbolic_trace(ReturnTypeModule())
        self.assertIn('-> typing_List[str]', traced._code)
        scripted = torch.jit.script(traced)
        self.assertIn('-> List[str]', scripted.code)

    def getitem_inner(self):
        if False:
            while True:
                i = 10

        class GetItemBase(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.register_buffer('pe', torch.randn(8, 8))

        class GetItem1(GetItemBase):

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.pe[:, :x.size(0)]

        class GetItem2(GetItemBase):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.pe[x.size(0)]

        class GetItem3(GetItemBase):

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.pe[4]
        self.checkGraphModule(GetItem1(), [torch.zeros(4)])
        self.checkGraphModule(GetItem2(), [torch.zeros(4)])
        self.checkGraphModule(GetItem3(), [torch.zeros(4)])

    @unittest.skipUnless(os.environ.get('FX_PATCH_GETITEM') == '1', 'Will be checked in test_getitem_subproc')
    def test_getitem(self):
        if False:
            for i in range(10):
                print('nop')
        self.getitem_inner()

    def test_getitem_subproc(self):
        if False:
            i = 10
            return i + 15
        proc = Process(target=run_getitem_target)
        proc.start()
        proc.join()
        self.assertEqual(proc.exitcode, 0)

    def test_user_friendly_call_provenance_with_function(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return wrapper_fn(x)
        traced = torch.fx.symbolic_trace(fn)
        with self.assertRaisesRegex(RuntimeError, "'wrapper_fn' is being compiled since it was called from 'fn.forward'"):
            scripted = torch.jit.script(traced)

    def test_user_friendly_call_provenance_with_module(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def forward(self, x):
                if False:
                    return 10
                return wrapper_fn(x)
        traced = torch.fx.symbolic_trace(M())
        with self.assertRaisesRegex(RuntimeError, "'wrapper_fn' is being compiled since it was called from 'M.forward'"):
            scripted = torch.jit.script(traced)

    def test_snake_case(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.activations = torch.nn.ModuleDict([['snake_case', torch.nn.ReLU()], ['PascalCase', torch.nn.LeakyReLU()], ['ALL_CAPS', torch.nn.PReLU()]])

            def forward(self, x):
                if False:
                    print('Hello World!')
                a = self.activations['snake_case'](x)
                b = self.activations['PascalCase'](x)
                c = self.activations['ALL_CAPS'](x)
                return (a, b, c)
        traced = symbolic_trace(M())
        check = [('activations_snake_case', 'activations.snake_case'), ('activations_pascal_case', 'activations.PascalCase'), ('activations_all_caps', 'activations.ALL_CAPS')]
        i = 0
        for node in traced.graph.nodes:
            if node.op == 'placeholder' or node.op == 'output':
                continue
            name = check[i][0]
            target = check[i][1]
            self.assertEqual(name, node.name)
            self.assertEqual(target, node.target)
            i += 1
        self.assertEqual(i, 3)

    def test_no_mutation(self):
        if False:
            return 10
        from torch.fx.immutable_collections import immutable_list
        x = immutable_list([3, 4])
        with self.assertRaisesRegex(NotImplementedError, 'new_args'):
            x[0] = 4

    def test_partial_trace(self):
        if False:
            print('Hello World!')

        class Foo(torch.nn.Module):

            def forward(self, x, y):
                if False:
                    print('Hello World!')
                if y:
                    return 2 * x
                else:
                    return x
        mod = Foo()
        mod_true = symbolic_trace(mod, concrete_args={'y': True})
        mod_false = symbolic_trace(mod, concrete_args={'y': False})
        self.assertEqual(mod_true(3, True), 6)
        print(mod_true.code)
        assert any((i.target == torch._assert for i in mod_true.graph.nodes))
        with self.assertRaises(AssertionError):
            mod_true(3, False)
        self.assertEqual(mod_false(3, False), 3)
        with self.assertRaises(AssertionError):
            mod_false(3, True)

        def f_higher(a, f):
            if False:
                for i in range(10):
                    print('nop')
            return f(a)
        nf = symbolic_trace(f_higher, concrete_args={'f': lambda x: x * 2})
        self.assertEqual(nf(3, lambda x: x * 2), 6)

    def test_custom_traceback_raised_when_exception_source_is_graphmodule(self):
        if False:
            for i in range(10):
                print('nop')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.W = torch.nn.Parameter(torch.randn(5))

            def forward(self, x):
                if False:
                    return 10
                return torch.dot(self.W, x)
        traced = torch.fx.symbolic_trace(M())
        out = [n for n in traced.graph.nodes if n.op == 'output'][-1]
        with traced.graph.inserting_before(out):
            relu_out = traced.graph.call_method(method_name='relu', args=(out.args[0],))
        out.args = (relu_out,)
        traced.recompile()
        with self.capture_stderr() as captured:
            with self.assertRaises(TypeError):
                traced(5)
        self.assertRegex(captured[0], "Call using an FX-traced Module, line .* of the traced Module's generated forward function:")

    def test_custom_traceback_not_raised_when_exception_source_is_submodule(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.linear = torch.nn.Linear(3, 4)

            def forward(self, x):
                if False:
                    print('Hello World!')
                return self.linear(x)
        traced = torch.fx.symbolic_trace(M())
        try:
            traced(torch.rand(5, 5))
        except RuntimeError:
            captured = traceback.format_exc()
        self.assertNotRegex(captured, "Call using an FX-traced Module, line .* of the traced Module's generated forward function:")

    def test_graph_module_replicate_for_dp(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.relu(x)
        gm = torch.fx.symbolic_trace(Foo())
        x = torch.randn(5, 3)
        out = gm(x)
        replica = gm._replicate_for_data_parallel()
        out_replica = replica(x)
        torch.testing.assert_close(out_replica, out)

    def test_ast_rewriter_rewrites_assert(self):
        if False:
            return 10

        class M(torch.nn.Module):

            def forward(self, x: torch.Tensor, y: int, z: int):
                if False:
                    return 10
                assert y == z
                return torch.add(x, x)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M())
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        traced.graph.lint()

    def test_ast_rewriter_rewrites_assert_with_message(self):
        if False:
            print('Hello World!')

        class M(torch.nn.Module):

            def forward(self, x: torch.Tensor, y: int, z: int):
                if False:
                    for i in range(10):
                        print('nop')
                assert y == z, 'msg'
                return torch.add(x, x)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M())
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        traced.graph.lint()

    def test_throw_out_variant(self):
        if False:
            for i in range(10):
                print('nop')

        def foo(x):
            if False:
                i = 10
                return i + 15
            y = torch.rand_like(x)
            torch.sigmoid(x, out=y)
            return y

        class MyTracer(torch.fx.Tracer):
            check_mutable_operations = True
        tracer = MyTracer()
        with self.assertRaisesRegex(RuntimeError, 'mutable operation aten::sigmoid.out'):
            traced_graph = tracer.trace(foo)

    def test_ast_rewriter_reassigns_submodules(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.bn = torch.nn.BatchNorm2d(100)

            def forward(self, x: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                return torch.add(x, x)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M())
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        traced.graph.lint()

    def test_ast_rewriter_wrap(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(3 + 4 + 5, a_lifted_leaf((3, 4), 5))

        def to_trace(y):
            if False:
                i = 10
                return i + 15
            return a_lifted_leaf((4, y), 3) + a_lifted_leaf((3, 4), 5) + a_lifted_leaf((y, y), y)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(to_trace)
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        self.assertIn('a_lifted_leaf', traced.code)
        self.assertEqual(27, traced(2))
        self.assertIs(a_lifted_leaf, real_a_lifed_leaf)

    def test_ast_rewriter_wrap_fn_directly(self):
        if False:
            return 10
        self.assertEqual(3 + 4 + 5, a_lifted_leaf2((3, 4), 5))

        def to_trace(y):
            if False:
                while True:
                    i = 10
            return a_lifted_leaf2((4, y), 3) + a_lifted_leaf2((3, 4), 5) + a_lifted_leaf2((y, y), y)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(to_trace)
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        self.assertIn('a_lifted_leaf2', traced.code)
        self.assertEqual(27, traced(2))
        self.assertIs(a_lifted_leaf2, real_a_lifed_leaf2)

    def test_profiler_ranges_side_effect(self):
        if False:
            print('Hello World!')
        g = torch.fx.Graph()
        handle = g.call_function(torch.ops.profiler._record_function_enter_new, ('test_range',))
        g.call_function(torch.ops.profiler._record_function_exit, (handle,))
        g.output(None)
        found_targets = {}
        for node in g.nodes:
            if node.op == 'call_function':
                found_targets.setdefault(node.target)
        self.assertEqual(list(found_targets.keys()), [torch.ops.profiler._record_function_enter_new, torch.ops.profiler._record_function_exit])
        g.eliminate_dead_code()
        found_targets = {}
        for node in g.nodes:
            if node.op == 'call_function':
                found_targets.setdefault(node.target)
        self.assertEqual(list(found_targets.keys()), [torch.ops.profiler._record_function_enter_new, torch.ops.profiler._record_function_exit])

    def test_ast_rewriter_wrapped_via_decorator(self):
        if False:
            while True:
                i = 10

        class F(torch.nn.Module):

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return wrapped_via_decorator(x)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(F())
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        self.assertIn('wrapped_via_decorator', traced.code)
        self.assertEqual(traced(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, '__fx_already_patched'))

    def test_ast_rewriter_wrapped_via_decorator_and_transformed(self):
        if False:
            i = 10
            return i + 15
        self.assertEqual(wrapped_via_decorator(0), 1)

        def to_trace(y):
            if False:
                for i in range(10):
                    print('nop')
            return wrapped_via_decorator(y)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(to_trace)
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        self.assertIn('wrapped_via_decorator', traced.code)
        self.assertEqual(traced(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, '__fx_already_patched'))
        transformed = torch.fx.Transformer(traced).transform()
        self.assertIn('wrapped_via_decorator', transformed.code)
        self.assertEqual(transformed(0), 1)
        self.assertIs(wrapped_via_decorator, real_wrapped_via_decorator)
        self.assertFalse(hasattr(wrapped_via_decorator, '__fx_already_patched'))

    def test_ast_rewriter_wrap_with_submodule(self):
        if False:
            i = 10
            return i + 15

        class M(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)

            def forward(self, x: torch.Tensor):
                if False:
                    print('Hello World!')
                return wrapped_with_submodule(x, self.batchnorm1d)
        ast_rewriter = RewritingTracer()
        graph = ast_rewriter.trace(M())
        traced = GraphModule(ast_rewriter.root, graph, 'gm')
        self.assertIn('wrapped_with_submodule', traced.code)
        input = torch.rand(3, 2)
        ref_batchnorm1d = torch.nn.BatchNorm1d(2, affine=False)
        self.assertEqual(ref_batchnorm1d(input), traced(input))

    def test_submodule_manipulation_API(self):
        if False:
            for i in range(10):
                print('nop')

        class C(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.conv = torch.nn.Conv2d(16, 33, 3, stride=2)
                self.param = torch.nn.Parameter(torch.rand(2, 3))

            def forward(self, x):
                if False:
                    return 10
                return self.conv(torch.cat([self.param, x]))

        class B(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(100, 200)
                self.register_buffer('buf', torch.randn(2, 3))
                self.net_c = C()

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.linear(torch.cat([self.buf, self.net_c(x)]))

        class A(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.net_b = B()
                self.param = torch.nn.Parameter(torch.rand(2, 3))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.net_b(x) + self.param
        a = symbolic_trace(A())
        a.add_submodule('net_b.net_c.dropout', torch.nn.Dropout(p=0.2))
        conv = [n for n in a.graph.nodes if n.target == 'net_b.net_c.conv'][-1]
        with a.graph.inserting_before(conv):
            with warnings.catch_warnings(record=True) as w:
                dropout = a.graph.call_module(module_name='net_b.net_c.dropout', args=conv.args)
                self.assertEqual(len(w), 0)
        conv.replace_all_uses_with(dropout)
        a.graph.erase_node(conv)
        a.recompile()

        def module_exists(gm: GraphModule, path: str) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return any((path == name for (name, _) in gm.named_modules()))

        def parameter_exists(gm: GraphModule, path: str) -> bool:
            if False:
                while True:
                    i = 10
            return any((path == name for (name, _) in gm.named_parameters())) and any((path == name for name in gm.state_dict().keys()))

        def buffer_exists(gm: GraphModule, path: str) -> bool:
            if False:
                for i in range(10):
                    print('nop')
            return any((path == name for (name, _) in gm.named_buffers())) and any((path == name for name in gm.state_dict().keys()))
        self.assertTrue(module_exists(a, 'net_b.net_c.dropout'))
        self.assertIsNotNone(a.get_submodule('net_b.net_c.dropout'))
        self.assertTrue(module_exists(a, 'net_b.net_c.conv'))
        self.assertIsNotNone(a.get_submodule('net_b.net_c.conv'))
        conv = [n for n in a.graph.nodes if n.target == 'net_b.net_c.conv']
        self.assertEqual(conv, [])
        a.delete_submodule('net_b.net_c.conv')
        self.assertFalse(module_exists(a, 'net_b.net_c.conv'))
        with self.assertRaisesRegex(AttributeError, 'has no attribute `conv`'):
            self.assertIsNone(a.get_submodule('net_b.net_c.conv'))
        cat = [n for n in a.graph.nodes if n.target == torch.cat][-1]
        with a.graph.inserting_before(cat):
            with warnings.catch_warnings(record=True) as w:
                param = a.graph.get_attr(qualified_name='net_b.net_c.param')
                self.assertEqual(len(w), 0)
            with self.assertWarnsRegex(UserWarning, 'Attempted to insert a get_attr Node with no underlying reference in the owning GraphModule'):
                bad_param = a.graph.get_attr(qualified_name='net_b.param')
                a.graph.erase_node(bad_param)
        cat.args = (*cat.args, param)
        a.recompile()
        a.graph.lint()
        a.get_parameter('net_b.net_c.param')
        with self.assertRaisesRegex(AttributeError, 'is not an nn.Parameter'):
            a.get_parameter('net_b.buf')
        with self.assertRaisesRegex(AttributeError, 'has no attribute `param`'):
            a.get_parameter('net_b.param')
        a.get_buffer('net_b.buf')
        with self.assertRaisesRegex(AttributeError, 'is not a buffer'):
            a.get_buffer('net_b.net_c.param')
        with self.assertRaisesRegex(AttributeError, 'has no attribute `buf`'):
            a.get_buffer('net_b.net_c.buf')
        a.get_submodule('')
        a.get_parameter('param')
        a.add_submodule('net_b.embedding', torch.nn.Embedding(10, 3))
        a.add_submodule('net_b.net_c.embedding', torch.nn.Embedding(10, 3))
        a.add_submodule('net_b.net_c.rnn', torch.nn.RNN(10, 20, 2))
        a.add_submodule('batch_norm_2d', torch.nn.BatchNorm2d(100))
        a.delete_all_unused_submodules()
        self.assertFalse(module_exists(a, 'net_b.embedding'))
        self.assertFalse(module_exists(a, 'net_b.net_c.embedding'))
        self.assertFalse(module_exists(a, 'net_b.net_c.rnn'))
        self.assertFalse(module_exists(a, 'batch_norm_2d'))
        self.assertTrue(parameter_exists(a, 'net_b.net_c.param'))
        self.assertTrue(buffer_exists(a, 'net_b.buf'))
        a.graph.lint()

    def test_delete_unused_submodules_leaf(self):
        if False:
            i = 10
            return i + 15

        class SubModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.relu = torch.nn.ReLU()

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                x = self.linear(x)
                x = self.relu(x)
                return x

        class Model(torch.nn.Module):

            def __init__(self):
                if False:
                    for i in range(10):
                        print('nop')
                super().__init__()
                self.submod = SubModule()

            def forward(self, x):
                if False:
                    return 10
                x = self.submod(x)
                return x
        model = Model()

        class MyCustomTracer(torch.fx.Tracer):

            def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
                if False:
                    for i in range(10):
                        print('nop')
                return module_qualified_name == 'submod'
        inputs = torch.randn(1, 10)
        traced_graph = MyCustomTracer().trace(model)
        gm2 = torch.fx.GraphModule(model, traced_graph)
        gm2.delete_all_unused_submodules()
        torch.testing.assert_close(gm2(inputs), model(inputs))

    def test_fx_stateless(self):
        if False:
            print('Hello World!')

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.l1 = torch.nn.Linear(1, 1)
                self.register_buffer('buffer', torch.ones(1))

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.l1(x) + self.buffer
        module = MockModule()
        x = torch.rand((1, 1))
        weight = torch.tensor([[1.0]], requires_grad=True)
        bias = torch.tensor([0.0], requires_grad=True)
        buffer = torch.tensor([0.0])
        parameters = {'l1.weight': weight, 'l1.bias': bias, 'buffer': buffer}
        fx_module = torch.fx.symbolic_trace(module)
        res = torch.func.functional_call(fx_module, parameters, x)
        res.backward()
        self.assertIsNotNone(weight.grad)
        self.assertIsNotNone(bias.grad)
        self.assertIsNone(buffer.grad)
        self.assertIsNone(module.l1.weight.grad)
        self.assertIsNone(module.l1.bias.grad)
        self.assertIsNone(module.buffer.grad)

    def test_tracing_graphmodules_as_leaf_submodules(self):
        if False:
            while True:
                i = 10

        class A(torch.nn.Module):

            def forward(self, t):
                if False:
                    i = 10
                    return i + 15
                return t + t

        class B(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super(type(self), self).__init__()
                self.calling = False
                self.called = False

            def forward(self, t):
                if False:
                    return 10
                if self.calling:
                    return t - t
                else:
                    return t + t

            def __call__(self, *args):
                if False:
                    i = 10
                    return i + 15
                self.called = True
                self.calling = True
                return super(type(self), self).__call__(*args)
                self.calling = False

        class M(torch.nn.Module):

            def __init__(self, a, b):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.a = a
                self.b = b

            def forward(self, t):
                if False:
                    i = 10
                    return i + 15
                x = self.a(t)
                y = self.b(t)
                return x + y

        class LeafTracer(Tracer):

            def is_leaf_module(self, module, name):
                if False:
                    i = 10
                    return i + 15
                return True

        class LeafTracerNotB(Tracer):

            def is_leaf_module(self, module, name):
                if False:
                    print('Hello World!')
                return False if 'b' in name else True
        a = symbolic_trace(A())
        a.recompile()
        m = M(a, B())
        graph = LeafTracerNotB().trace(m)
        gm = GraphModule(m, graph)
        gm.recompile()
        self.assertTrue(isinstance(gm.get_submodule('a'), GraphModule))
        match = [n for n in gm.graph.nodes if n.op == 'call_module' and n.target == 'a']
        self.assertTrue(len(match) == 1)
        self.assertFalse(hasattr(gm, 'b'))
        match = [n for n in gm.graph.nodes if n.op == 'call_function' and n.target == operator.sub]
        self.assertTrue(len(match) == 1)
        a = symbolic_trace(A())
        a.recompile()
        b = B()
        m = M(a, b)
        graph = LeafTracer().trace(m)
        gm = GraphModule(m, graph)
        gm.recompile()
        self.assertTrue(isinstance(gm.get_submodule('a'), GraphModule))
        match = [n for n in gm.graph.nodes if n.op == 'call_module' and n.target == 'a']
        self.assertTrue(len(match) == 1)
        self.assertTrue(isinstance(gm.get_submodule('b'), torch.nn.Module))
        match = [n for n in gm.graph.nodes if n.op == 'call_module' and n.target == 'b']
        self.assertTrue(len(match) == 1)
        self.assertTrue(b.called)
        self.assertTrue(gm.get_submodule('b').called)
        a = symbolic_trace(A())
        a.recompile()
        b = symbolic_trace(B())
        b.recompile()
        m = M(a, b)
        graph = LeafTracer().trace(m)
        gm = GraphModule(m, graph)
        gm.recompile()
        self.assertTrue(isinstance(gm.get_submodule('a'), GraphModule))
        match = [n for n in gm.graph.nodes if n.op == 'call_module' and n.target == 'a']
        self.assertTrue(len(match) == 1)
        self.assertTrue(isinstance(gm.get_submodule('b'), torch.nn.Module))
        match = [n for n in gm.graph.nodes if n.op == 'call_module' and n.target == 'b']
        self.assertTrue(len(match) == 1)

    def _test_graph_module_init_buffer_param_copied(self, use_dict_init: bool):
        if False:
            i = 10
            return i + 15

        class MyModule(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('my_buff', torch.rand(3, 4))
                self.register_parameter('my_param', torch.nn.Parameter(torch.rand(3, 4)))

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return x + self.my_buff + self.my_param
        mod = MyModule()
        mod_traced = symbolic_trace(mod)
        orig_buff = mod_traced.get_buffer('my_buff')
        orig_param = mod_traced.get_parameter('my_param')
        mod_traced_new = GraphModule({'my_buff': orig_buff, 'my_param': orig_param} if use_dict_init else mod, mod_traced.graph)
        try:
            new_buff = mod_traced_new.get_buffer('my_buff')
        except Exception:
            self.fail('Did not find my_buff')
        self.assertEqual(orig_buff, new_buff)
        try:
            new_param = mod_traced_new.get_parameter('my_param')
        except Exception:
            self.fail('Did not find my_param')
        self.assertEqual(orig_param, new_param)
        x = torch.rand(3, 4)
        orig_out = mod_traced(x)
        submodules_out = mod_traced_new(x)
        self.assertEqual(orig_out, submodules_out)

    def test_graph_module_init_buffer_param_copied_dict_init(self):
        if False:
            i = 10
            return i + 15
        self._test_graph_module_init_buffer_param_copied(use_dict_init=True)

    def test_graph_module_init_buffer_param_copied_mod_init(self):
        if False:
            return 10
        self._test_graph_module_init_buffer_param_copied(use_dict_init=False)

    def test_annotations_with_no_forward_references(self):
        if False:
            for i in range(10):
                print('nop')

        class A:

            def __call__(self, x: torch.Tensor):
                if False:
                    i = 10
                    return i + 15
                return torch.add(x, x)

        class M(torch.nn.Module):

            def forward(self, x: torch.Tensor, a: A) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return a(x)
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    def test_annotations_with_forward_references(self):
        if False:
            for i in range(10):
                print('nop')

        class A:

            def __call__(self, x: torch.Tensor):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.add(x, x)

        class M(torch.nn.Module):

            def forward(self, x: 'torch.Tensor', a: 'A') -> 'torch.Tensor':
                if False:
                    return 10
                return a(x)
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    def test_annotations_with_non_torch_reference_and_no_internal_forward_references(self):
        if False:
            return 10

        class A:

            def __call__(self, x: torch.Tensor):
                if False:
                    return 10
                return torch.add(x, x)

        class M(torch.nn.Module):

            def forward(self, x: List[torch.Tensor], a: A) -> torch.Tensor:
                if False:
                    print('Hello World!')
                return a(x[0])
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    def test_annotations_with_non_torch_reference_and_internal_forward_references(self):
        if False:
            i = 10
            return i + 15

        class A:

            def __call__(self, x: torch.Tensor):
                if False:
                    print('Hello World!')
                return torch.add(x, x)

        class M(torch.nn.Module):

            def forward(self, x: List['torch.Tensor'], a: A) -> 'torch.Tensor':
                if False:
                    print('Hello World!')
                return a(x)[0]
        self.checkGraphModule(M(), (torch.rand(2, 3), A()), kwargs=None)

    @unittest.skipIf(sys.version_info < (3, 7), '`__future__` feature `annotations` is not defined in Python <3.7')
    def test_annotation_with_future(self):
        if False:
            return 10
        try:
            import fx.test_future
        finally:
            del sys.modules['__future__']

    @unittest.skipIf(sys.version_info > (3, 11), 'Does not work in 3.11')
    def test_annotations_empty_tuple(self):
        if False:
            for i in range(10):
                print('nop')

        class Foo(torch.nn.Module):

            def forward(self, x: Tuple[()], y: Tuple[str, Tuple[()]]):
                if False:
                    return 10
                return 'foo'
        traced = torch.fx.symbolic_trace(Foo())
        x = ()
        y = ('bar', ())
        traced(x, y)
        FileCheck().check('_Tuple[()]').check('typing_Tuple[str,typing_Tuple[()]]').run(traced.code)
        scripted = torch.jit.script(traced)
        scripted(x, y)
        FileCheck().check('Tuple[()]').check('Tuple[str, Tuple[()]]').run(scripted.code)

    @unittest.skipIf(IS_WINDOWS, 'Python Windows bug? https://bugs.python.org/issue45108')
    @unittest.skipIf(sys.version_info >= (3, 10), 'Does not work on Python-3.10')
    def test_assert(self):
        if False:
            return 10

        def f(x):
            if False:
                print('Hello World!')
            assert x > 1
            return x + 1
        try:
            torch.fx.proxy.TracerBase.trace_asserts = True
            traced = symbolic_trace(f)
        finally:
            torch.fx.proxy.TracerBase.trace_asserts = False
        self.assertEqual(f(2), traced(2))
        with self.assertRaises(AssertionError):
            traced(0)

    def test_pytree(self):
        if False:
            for i in range(10):
                print('nop')

        class PHTest(PHBase):
            pass

        def f_sum(x):
            if False:
                while True:
                    i = 10
            return sum(x)

        def f_sum_dict(x):
            if False:
                for i in range(10):
                    print('nop')
            out = 0
            for v in x.values():
                out += v
            return out

        def f_dict_list_map(x):
            if False:
                i = 10
                return i + 15
            new_dict = {}
            for (k, v) in x.items():
                new_dict[k] = [i + 1 for i in v]
            return new_dict

        def f_dict_add(x):
            if False:
                return 10
            return x['a'] + sum(x['z'])

        def f_namedtuple_add(x):
            if False:
                i = 10
                return i + 15
            return x.x + x.y
        pytree._register_pytree_node(Foo, lambda x: ([x.a, x.b], None), lambda x, _: Foo(x[0], x[1]))
        fx_pytree.register_pytree_flatten_spec(Foo, lambda x, _: [x.a, x.b])

        def f_custom(x):
            if False:
                return 10
            return x.a + x.b

        def f_custom_dict(x):
            if False:
                for i in range(10):
                    print('nop')
            return f_sum_dict(x.a) + x.b

        def f_return_custom(x):
            if False:
                while True:
                    i = 10
            return Foo(x.b, x.a)
        tests = [(f_sum, [PH, PH, PH]), (f_sum, []), (f_sum, [PHTest(), PHTest(), PHTest()]), (f_sum_dict, {'a': PH, 'b': PH, 'c': PH}), (f_dict_list_map, {'a': (PH, PH), 'b': [PH], 'c': []}), (f_dict_list_map, {5: (PH, PH, PH)}), (f_dict_add, {'a': PH, 'z': (PH, PH, PH)}), (f_dict_add, {'a': PH, 'z': []}), (f_custom, Foo(PH, PH)), (f_custom, Foo(PH, 3)), (f_custom_dict, Foo({'a': PH, 'b': PH}, PH)), (f_namedtuple_add, Point(PH, PH))]

        def verify_pytree(f, inp):
            if False:
                i = 10
                return i + 15
            val = pytree.tree_map(lambda x: torch.randn(3) if isinstance(x, PHBase) else x, inp)
            num_flat_args = len([i == PH for i in pytree.tree_leaves(inp)])
            orig_out = f(val)
            nf = symbolic_trace(f, concrete_args={'x': inp})
            self.assertEqual(nf(val), orig_out)
            bare_fx = GraphModule({}, copy.deepcopy(nf.graph))
            bare_fx.graph.set_codegen(CodeGen())
            bare_fx.recompile()
            self.assertEqual(nf.graph.process_outputs(bare_fx(*nf.graph.process_inputs(val))), orig_out)
            assert num_flat_args == 0 or 'tree_flatten_spec' in nf.code
            assert sum([i.op == 'placeholder' for i in nf.graph.nodes]) == num_flat_args
            nf = symbolic_trace(nf)
            self.assertEqual(nf(val), orig_out)
            assert 'tree_flatten_spec' not in nf.code
            assert sum([i.op == 'placeholder' for i in nf.graph.nodes]) == 1
            nf = symbolic_trace(nf, concrete_args={'x': inp})
            self.assertEqual(nf(val), orig_out)
            assert num_flat_args == 0 or 'tree_flatten_spec' in nf.code
            assert sum([i.op == 'placeholder' for i in nf.graph.nodes]) == num_flat_args
            pickled = pickle.dumps(nf)
            nf = pickle.loads(pickled)
            self.assertEqual(nf(val), orig_out)
        for (f, inp) in tests:
            verify_pytree(f, inp)

    def test_pytree_concrete(self):
        if False:
            i = 10
            return i + 15

        def f(b, a):
            if False:
                i = 10
                return i + 15
            if b:
                return a['a']
            else:
                return a['z']
        inp = {'a': {'a': PH, 'z': PH}, 'b': True}
        nf = symbolic_trace(f, concrete_args=inp)
        val = pytree.tree_map(lambda x: torch.randn(3) if x == PH else x, inp)
        self.assertEqual(nf(**val), f(**val))
        nf = symbolic_trace(nf)
        self.assertEqual(nf(**val), f(**val))

    def test_metadata_on_ph(self):
        if False:
            print('Hello World!')

        def f_sum(a: int, b: int) -> int:
            if False:
                print('Hello World!')
            return a + b

        def f_dict(a: Dict[str, str]) -> bool:
            if False:
                i = 10
                return i + 15
            return a['f1'] == a['f2']

        def verify_metadata(gm: GraphModule, arg_names: List[str], metadata: List[str]):
            if False:
                for i in range(10):
                    print('nop')
            for node in gm.graph.nodes:
                if node.op == 'placeholder':
                    self.assertTrue(node.name in arg_names)
                    self.assertTrue(node.ph_key in metadata)
        verify_metadata(gm=symbolic_trace(f_sum, concrete_args={'a': PHWithMeta(ph_key='a'), 'b': PHWithMeta(ph_key='b')}), arg_names=['a_1', 'b_1'], metadata=['a', 'b'])
        verify_metadata(gm=symbolic_trace(f_dict, concrete_args={'a': {'f1': PHWithMeta(ph_key='f1'), 'f2': PHWithMeta(ph_key='f2')}}), arg_names=['a_1', 'a_2'], metadata=['f1', 'f2'])

        class TaggingTracer(Tracer):

            def create_node(self, kind: str, target: Union[str, Callable], args: Tuple[Argument, ...], kwargs: Dict[str, Any], name: Optional[str]=None, type_expr: Optional[Any]=None) -> Node:
                if False:
                    while True:
                        i = 10
                n = super().create_node(kind, target, args, kwargs, name)
                n.tag = 'foo'
                return n

        class PHWithTag(PHBase):

            def __init__(self, tag: str):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.tag = tag
        g = TaggingTracer().trace(f_sum, concrete_args={'a': PHWithTag(tag='bar'), 'b': PHWithTag(tag='bar')})
        for n in g.nodes:
            self.assertTrue(hasattr(n, 'tag'))
            self.assertEqual(n.tag, 'foo')

    def test_custom_codegen(self):
        if False:
            for i in range(10):
                print('nop')

        class ListCodeGen(CodeGen):

            def gen_fn_def(self, free_vars, maybe_return_annotation):
                if False:
                    print('Hello World!')
                lst_unpack = f"\ndef forward(self, args_list: List[torch.Tensor]){maybe_return_annotation}:\n    {', '.join(free_vars)} = args_list"
                return lst_unpack

            def additional_globals(self):
                if False:
                    for i in range(10):
                        print('nop')
                return [('List', typing.List)]

            def process_inputs(self, *inputs):
                if False:
                    i = 10
                    return i + 15
                assert len(inputs) == 1
                return inputs[0]

        def f(a, b):
            if False:
                print('Hello World!')
            return a + b
        nf = symbolic_trace(f)
        vals = [torch.randn(3), torch.randn(3)]
        self.assertEqual(nf(*vals), f(*vals))
        nf.graph.set_codegen(ListCodeGen())
        nf.recompile()
        bare_fx = GraphModule({}, copy.deepcopy(nf.graph))
        bare_fx.graph.set_codegen(CodeGen())
        bare_fx.recompile()
        self.assertEqual(nf(vals), f(*vals))
        self.assertEqual(nf.graph.process_outputs(bare_fx(*nf.graph.process_inputs(vals))), f(*vals))
        ts_f = torch.jit.script(nf)
        self.assertEqual(nf(vals), ts_f(vals))

    def test_custom_codegen_with_transformer(self):
        if False:
            for i in range(10):
                print('nop')

        class ListCodeGen(CodeGen):

            def gen_fn_def(self, free_vars, maybe_return_annotation):
                if False:
                    print('Hello World!')
                lst_unpack = f"\ndef forward(self, args_list: List[torch.Tensor]){maybe_return_annotation}:\n    {', '.join(free_vars)} = args_list"
                return lst_unpack

            def additional_globals(self):
                if False:
                    return 10
                return [('List', typing.List)]

            def process_inputs(self, *inputs):
                if False:
                    return 10
                assert len(inputs) == 1
                return inputs[0]

        def f(a, b):
            if False:
                return 10
            return a + b
        nf = symbolic_trace(f)
        vals = [torch.randn(3), torch.randn(3)]
        self.assertEqual(nf(*vals), f(*vals))
        nf.graph.set_codegen(ListCodeGen())
        nf.recompile()
        self.assertEqual(nf(vals), f(*vals))
        transformed_gm = Transformer(nf).transform()
        self.assertEqual(nf(vals), transformed_gm(vals))

    def test_interpreter_with_codegen(self):
        if False:
            print('Hello World!')

        class ListCodeGen(CodeGen):

            def gen_fn_def(self, free_vars, maybe_return_annotation):
                if False:
                    return 10
                lst_unpack = f"\ndef forward(self, args_list: List[torch.Tensor]){maybe_return_annotation}:\n    {', '.join(free_vars)} = args_list"
                return lst_unpack

            def additional_globals(self):
                if False:
                    i = 10
                    return i + 15
                return [('List', typing.List)]

            def process_inputs(self, *inputs):
                if False:
                    i = 10
                    return i + 15
                assert len(inputs) == 1
                return inputs[0]

            def generate_output(self, output_args):
                if False:
                    print('Hello World!')
                return f'return list({repr(output_args)})'

            def process_outputs(self, outputs):
                if False:
                    while True:
                        i = 10
                return list(outputs)

        def f(a, b):
            if False:
                i = 10
                return i + 15
            a = a + b
            b = a + b
            return (a, b)
        nf = symbolic_trace(f)
        vals = [torch.randn(3), torch.randn(3)]
        nf.graph.set_codegen(ListCodeGen())
        nf.recompile()
        self.assertEqual(Interpreter(nf).run(vals), nf(vals))

    def test_imul_code_print(self):
        if False:
            for i in range(10):
                print('nop')
        graph = torch.fx.Graph()
        a = graph.placeholder('a')
        b = graph.placeholder('b')
        graph.call_function(operator.imul, (a, b), {})
        graph.output(a)
        gm = torch.fx.GraphModule({}, graph)
        gm.recompile()
        self.assertEqual(gm(2, 3), 6)
        self.assertIn('a *= b', gm.code)

    def test_deepcopy_tracer(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return (x + y).relu().sin()
        tracer = Tracer()
        tracer_before = copy.deepcopy(tracer)
        tracer.trace(fn)
        tracer_after = copy.deepcopy(tracer)
        self.assertEqual(str(tracer.graph), str(tracer_after.graph))
        self.assertTrue(not hasattr(tracer_before, 'graph') or str(tracer.graph) != str(tracer_before.graph))

    def test_deepcopy_graphmodule(self):
        if False:
            for i in range(10):
                print('nop')
        m = symbolic_trace(SimpleTest())
        m.meta['hello'] = 'world'
        copy_m = copy.deepcopy(m)
        self.assertEqual(copy_m.meta['hello'], 'world')

    def test_deepcopy_no_recursion(self):
        if False:
            i = 10
            return i + 15
        m = symbolic_trace(SimpleTest())
        m.meta['hello'] = m
        copy_m = copy.deepcopy(m)
        self.assertEqual(id(copy_m), id(copy_m.meta['hello']))

    def test_enum(self):
        if False:
            while True:
                i = 10
        from enum import Enum

        class Foo(Enum):
            A = 1
            B = 2

        def leaf_fn(arr, enum_val):
            if False:
                i = 10
                return i + 15
            arr.append(enum_val)
            return arr[-1].value

        def foo(x):
            if False:
                return 10
            return leaf_fn(x, Foo.A)
        traced = torch.fx.symbolic_trace(foo)
        self.assertEqual(foo([]), traced([]))

    def test_insert_arg(self):
        if False:
            i = 10
            return i + 15
        m = symbolic_trace(SimpleTest())
        m.register_buffer('buf', torch.tensor(0))
        output_node = next(iter(reversed(m.graph.nodes)))
        with m.graph.inserting_before(output_node):
            a = m.graph.get_attr('buf')
        r = len(output_node.args)
        output_node.insert_arg(0, a)
        self.assertEqual(len(output_node.args), r + 1)
        self.assertEqual(len(a.users), 1)
        self.assertIs(output_node.args[0], a)
        self.assertIs(list(a.users.keys())[0], output_node)
        output_node.insert_arg(2, a)
        self.assertEqual(len(output_node.args), r + 2)
        self.assertEqual(len(a.users), 1)
        self.assertIs(output_node.args[2], a)
        self.assertIs(list(a.users.keys())[0], output_node)
        m.graph.lint()

def run_getitem_target():
    if False:
        i = 10
        return i + 15
    from torch.fx._symbolic_trace import _wrapped_methods_to_patch
    _wrapped_methods_to_patch.append((torch.Tensor, '__getitem__'))
    try:
        TestFX().getitem_inner()
    finally:
        _wrapped_methods_to_patch.pop()

class TestOperatorSignatures(JitTestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        if False:
            while True:
                i = 10
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    @onlyCPU
    @ops(op_db, allowed_dtypes=(torch.float,))
    def test_get_torch_func_signature_exhaustive(self, device, dtype, op):
        if False:
            return 10
        if not isinstance(op.op, types.BuiltinFunctionType):
            raise unittest.SkipTest("This path doesn't work on Python functions")
        sample_inputs_itr = op.sample_inputs(device, dtype, requires_grad=False)
        schemas = get_signature_for_torch_op(op.op)
        if not schemas:
            raise RuntimeError('No Schemas Returned')
        for sample_input in sample_inputs_itr:
            for schema in schemas:
                try:
                    bound_args = schema.bind(sample_input.input, *sample_input.args, **sample_input.kwargs)
                    bound_args.apply_defaults()
                    op(*bound_args.args, **bound_args.kwargs)
                    break
                except TypeError as e:
                    pass
            else:
                raise RuntimeError(f'Did not match any schemas for op {op.name}!')

class TestFXAPIBackwardCompatibility(JitTestCase):

    def setUp(self):
        if False:
            return 10
        super().setUp()
        self.maxDiff = None
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        if False:
            return 10
        super().tearDown()
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag

    def _fn_to_stable_annotation_str(self, obj):
        if False:
            print('Hello World!')
        '\n        Unfortunately we have to serialize function signatures manually since\n        serialization for `inspect.Signature` objects is not stable across\n        python versions\n        '
        fn_name = torch.typename(obj)
        signature = inspect.signature(obj)
        sig_str = f'{fn_name}{signature}'
        arg_strs = []
        for (k, v) in signature.parameters.items():
            maybe_type_annotation = f': {self._annotation_type_to_stable_str(v.annotation, sig_str)}' if v.annotation is not inspect.Signature.empty else ''

            def default_val_str(val):
                if False:
                    return 10
                if isinstance(val, (tuple, list)):
                    str_pieces = ['(' if isinstance(val, tuple) else '[']
                    str_pieces.append(', '.join((default_val_str(v) for v in val)))
                    if isinstance(val, tuple) and len(str_pieces) == 2:
                        str_pieces.append(',')
                    str_pieces.append(')' if isinstance(val, tuple) else ']')
                    return ''.join(str_pieces)
                if isinstance(val, types.ModuleType):
                    return f'<module {val.__name__}>'
                if callable(val):
                    return f'<function {val.__name__}>'
                return str(val)
            if v.default is not inspect.Signature.empty:
                default_val_str = default_val_str(v.default) if not isinstance(v.default, str) else f"'{v.default}'"
                maybe_default = f' = {default_val_str}'
            else:
                maybe_default = ''
            maybe_stars = ''
            if v.kind == inspect.Parameter.VAR_POSITIONAL:
                maybe_stars = '*'
            elif v.kind == inspect.Parameter.VAR_KEYWORD:
                maybe_stars = '**'
            arg_strs.append(f'{maybe_stars}{k}{maybe_type_annotation}{maybe_default}')
        return_annot = f' -> {self._annotation_type_to_stable_str(signature.return_annotation, sig_str)}' if signature.return_annotation is not inspect.Signature.empty else ''
        return f"{fn_name}({', '.join(arg_strs)}){return_annot}"

    def _annotation_type_to_stable_str(self, t, sig_str):
        if False:
            return 10
        if t is inspect.Signature.empty:
            return ''
        if isinstance(t, str):
            return f"'{t}'"
        if hasattr(typing, 'ForwardRef') and isinstance(t, typing.ForwardRef):
            return t.__forward_arg__
        if hasattr(typing, '_ForwardRef') and isinstance(t, typing._ForwardRef):
            return t.__forward_arg__
        trivial_mappings = {str: 'str', int: 'int', float: 'float', bool: 'bool', torch.dtype: 'torch.dtype', torch.Tensor: 'torch.Tensor', torch.device: 'torch.device', torch.memory_format: 'torch.memory_format', slice: 'slice', torch.nn.Module: 'torch.nn.modules.module.Module', torch.fx.Graph: 'torch.fx.graph.Graph', torch.fx.Node: 'torch.fx.node.Node', torch.fx.Proxy: 'torch.fx.proxy.Proxy', torch.fx.node.Target: 'torch.fx.node.Target', torch.fx.node.Argument: 'torch.fx.node.Argument', torch.fx.graph.PythonCode: 'torch.fx.graph.PythonCode', torch.fx.graph_module.GraphModule: 'torch.fx.graph_module.GraphModule', torch.fx.subgraph_rewriter.Match: 'torch.fx.subgraph_rewriter.Match', Ellipsis: '...', typing.Any: 'Any', type(None): 'NoneType', None: 'None', typing.Iterator: 'Iterator'}
        mapping = trivial_mappings.get(t, None)
        if mapping:
            return mapping
        contained = getattr(t, '__args__', None) or []
        contained = t if isinstance(t, list) else contained
        if all((isinstance(ct, typing.TypeVar) for ct in contained)):
            contained = []
        contained_type_annots = [self._annotation_type_to_stable_str(ct, sig_str) for ct in contained]
        contained_type_str = f"[{', '.join(contained_type_annots)}]" if len(contained_type_annots) > 0 else ''
        origin = getattr(t, '__origin__', None)
        if origin is None:
            origin = t if t in {typing.Tuple, typing.Union, typing.Dict, typing.List, typing.Type, typing.Callable} else origin
        if origin in {tuple, typing.Tuple}:
            return f'Tuple{contained_type_str}'
        if origin in {typing.Union}:
            if len(contained) == 2 and (contained[0] is type(None)) ^ (contained[1] is type(None)):
                not_none_param = contained[0] if contained[0] is not type(None) else contained[1]
                return f'Optional[{self._annotation_type_to_stable_str(not_none_param, sig_str)}]'
            return f'Union{contained_type_str}'
        if origin in {dict, typing.Dict}:
            return f'Dict{contained_type_str}'
        if origin in {list, typing.List}:
            return f'List{contained_type_str}'
        if origin in {type, typing.Type}:
            return f'Type{contained_type_str}'
        if isinstance(t, typing.Callable):
            if len(contained) > 0 and contained[0] is not Ellipsis:
                return f"Callable[[{', '.join(contained_type_annots[:-1])}], {contained_type_annots[-1]}]"
            else:
                return f'Callable{contained_type_str}'
        raise RuntimeError(f'Unrecognized type {t} used in BC-compatible type signature {sig_str}.Please add support for this type and confirm with the FX team that your signature change is valid.')

    def test_function_back_compat(self):
        if False:
            return 10
        '\n        Test backward compatibility for function signatures with\n        @compatibility(is_backward_compatible=True). Currently this checks for\n        exact signature matches, which may lead to false positives. If this\n        becomes too annoying, we can refine this check to actually parse out\n        the saved schema strings and check if the change is truly backward-\n        incompatible.\n        '
        signature_strs = []
        for obj in _BACK_COMPAT_OBJECTS:
            if not isinstance(obj, type):
                signature_strs.append(self._fn_to_stable_annotation_str(obj))
        signature_strs.sort()
        try:
            self.assertExpected('\n'.join(signature_strs) + '\n', 'fx_backcompat_function_signatures')
        except AssertionError as e:
            msg = f'{e}\n****** ERROR ******\nAn FX function that has been marked as backwards-compatible has experienced a signature change. See the above exception context for more information. If this change was unintended, please revert it. If it was intended, check with the FX team to ensure that the proper deprecation protocols have been followed and subsequently --accept the change.'
            raise AssertionError(msg)

    def test_class_member_back_compat(self):
        if False:
            print('Hello World!')
        '\n        Test backward compatibility for members of classes with\n        @compatibility(is_backward_compatible=True). Currently this checks for\n        exact matches on the publicly visible members of the class.\n        '
        class_method_strs = []
        for obj in _BACK_COMPAT_OBJECTS:
            if isinstance(obj, type):
                public_members = [name for name in obj.__dict__ if not name.startswith('_')]
                class_method_strs.append(f'{torch.typename(obj)} {sorted(public_members)}')
        class_method_strs.sort()
        try:
            self.assertExpected('\n'.join(class_method_strs), 'fx_backcompat_class_members')
        except AssertionError as e:
            msg = f'{e}\n****** ERROR ******\nAn FX class that has been marked as backwards-compatible has experienced change in its public members. See the above exception context for more information. If this change was unintended, please revert it. If it was intended, check with the FX team to ensure that the proper deprecation protocols have been followed and subsequently --accept the change.'
            raise AssertionError(msg) from e

    def test_public_api_surface(self):
        if False:
            return 10
        non_back_compat_objects = {}

        def check_symbols_have_bc_designation(m, prefix):
            if False:
                print('Hello World!')
            if not m.__name__.startswith('torch.fx'):
                return
            if m.__name__.startswith('torch.fx.experimental'):
                return
            for (k, v) in m.__dict__.items():
                if v is m:
                    continue
                if k.startswith('_'):
                    continue
                if isinstance(v, types.ModuleType):
                    check_symbols_have_bc_designation(v, prefix + [k])
                elif isinstance(v, (type, types.FunctionType)):
                    if v not in _MARKED_WITH_COMPATIBILITY:
                        non_back_compat_objects.setdefault(v)
        check_symbols_have_bc_designation(torch.fx, ['torch', 'fx'])
        check_symbols_have_bc_designation(torch.fx.passes, ['torch', 'fx', 'passes'])
        non_back_compat_strs = [torch.typename(obj) for obj in non_back_compat_objects.keys()]
        non_back_compat_strs = [s for s in non_back_compat_strs if s.startswith('torch.fx') and (not s.startswith('torch.fx.experimental'))]
        non_back_compat_strs = [s for s in non_back_compat_strs if all((not atom.startswith('_') for atom in s.split('.')))]
        non_back_compat_strs.sort()
        if len(non_back_compat_strs) != 0:
            raise AssertionError(f'Public FX API(s) {non_back_compat_strs} introduced but not given a backwards-compatibility classification! Please decorate these API(s) with `@torch.fx._compatibility.compatibility` to specify BC guarantees.')

    def test_adding_side_effect_function(self):
        if False:
            i = 10
            return i + 15

        class TestModule(torch.nn.Module):

            def forward(self, x):
                if False:
                    print('Hello World!')
                side_effect_func(x)
                return x
        gm = torch.fx.symbolic_trace(TestModule())
        self.assertEqual(len(gm.graph.nodes), 3)
        gm.graph.eliminate_dead_code()
        gm.recompile()
        self.assertEqual(len(gm.graph.nodes), 3)
        found = False
        for node in gm.graph.nodes:
            if node.op == 'call_function' and node.target == side_effect_func:
                found = True
        self.assertTrue(found)

    def test_preserve_unused_attr_after_unpickle(self):
        if False:
            print('Hello World!')
        gm = torch.fx.symbolic_trace(Add())
        gm.add_submodule('foo', Add())
        gm.register_buffer('dummy_buffer', torch.empty(1))
        gm.register_parameter('dummy_parameter', torch.nn.Parameter(torch.empty(1)))
        b = io.BytesIO()
        torch.save(gm, b)
        b.seek(0)
        reload_gm = torch.load(b)
        self.assertTrue(hasattr(reload_gm, 'foo'))
        self.assertTrue(hasattr(reload_gm, 'dummy_buffer'))
        self.assertTrue(hasattr(reload_gm, 'dummy_parameter'))

class TestFunctionalTracing(JitTestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        if False:
            while True:
                i = 10
        super().tearDown()
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag
    IGNORE_FUNCS = ('has_torch_function', 'has_torch_function_unary', 'has_torch_function_variadic', 'handle_torch_function', 'boolean_dispatch')
    TO_PATCH = {'has_torch_function': None, 'has_torch_function_unary': None, 'has_torch_function_variadic': None}
    BUILT_IN_FUNC = (AssertionError, '')
    PROXY_ITERABLE = (TypeError, "argument of type 'Proxy' is not iterable")
    PROXY_ITERATED = (TraceError, 'Proxy object cannot be iterated')
    LEN_ERROR = (RuntimeError, "'len' is not supported in symbolic tracing by default")
    ARG_TYPE_MISMATCH = (TypeError, ', not Proxy$')
    CONTROL_FLOW = (TraceError, 'symbolically traced variables cannot be used as inputs to control flow')
    INTERPOLATE_ARGS_CONFLICT = (ValueError, 'only one of size or scale_factor should be defined')
    MUTABLE = (RuntimeError, 'Tried to trace mutable operation')
    UNTRACEABLE_FUNCTIONALS = {'adaptive_avg_pool1d': BUILT_IN_FUNC, 'avg_pool1d': BUILT_IN_FUNC, 'avg_pool2d': BUILT_IN_FUNC, 'avg_pool3d': BUILT_IN_FUNC, 'bilinear': BUILT_IN_FUNC, 'celu_': BUILT_IN_FUNC, 'channel_shuffle': BUILT_IN_FUNC, 'native_channel_shuffle': BUILT_IN_FUNC, 'conv1d': BUILT_IN_FUNC, 'conv2d': BUILT_IN_FUNC, 'conv3d': BUILT_IN_FUNC, 'conv_tbc': BUILT_IN_FUNC, 'conv_transpose1d': BUILT_IN_FUNC, 'conv_transpose2d': BUILT_IN_FUNC, 'conv_transpose3d': BUILT_IN_FUNC, 'cosine_similarity': BUILT_IN_FUNC, 'elu_': BUILT_IN_FUNC, 'gelu': BUILT_IN_FUNC, 'hardshrink': BUILT_IN_FUNC, 'hardtanh_': BUILT_IN_FUNC, 'leaky_relu_': BUILT_IN_FUNC, 'linear': BUILT_IN_FUNC, 'logsigmoid': BUILT_IN_FUNC, 'one_hot': BUILT_IN_FUNC, 'pad': BUILT_IN_FUNC, 'pairwise_distance': BUILT_IN_FUNC, 'pdist': BUILT_IN_FUNC, 'pixel_shuffle': BUILT_IN_FUNC, 'pixel_unshuffle': BUILT_IN_FUNC, 'prelu': BUILT_IN_FUNC, 'relu_': BUILT_IN_FUNC, 'rrelu_': BUILT_IN_FUNC, 'selu_': BUILT_IN_FUNC, 'scaled_dot_product_attention': BUILT_IN_FUNC, 'softplus': BUILT_IN_FUNC, 'softshrink': BUILT_IN_FUNC, 'threshold_': BUILT_IN_FUNC, 'adaptive_avg_pool2d': LEN_ERROR, 'adaptive_avg_pool3d': LEN_ERROR, 'adaptive_max_pool2d_with_indices': LEN_ERROR, 'adaptive_max_pool3d_with_indices': LEN_ERROR, 'instance_norm': CONTROL_FLOW, 'adaptive_max_pool1d': PROXY_ITERABLE, 'adaptive_max_pool2d': PROXY_ITERABLE, 'adaptive_max_pool3d': PROXY_ITERABLE, 'fractional_max_pool2d': PROXY_ITERABLE, 'fractional_max_pool3d': PROXY_ITERABLE, 'max_pool1d': PROXY_ITERABLE, 'max_pool2d': PROXY_ITERABLE, 'max_pool3d': PROXY_ITERABLE, 'lp_pool2d': PROXY_ITERATED, 'max_unpool1d': PROXY_ITERATED, 'max_unpool2d': PROXY_ITERATED, 'max_unpool3d': PROXY_ITERATED, 'fold': PROXY_ITERATED, 'unfold': PROXY_ITERATED, 'adaptive_max_pool1d_with_indices': ARG_TYPE_MISMATCH, 'fractional_max_pool2d_with_indices': ARG_TYPE_MISMATCH, 'fractional_max_pool3d_with_indices': ARG_TYPE_MISMATCH, 'layer_norm': ARG_TYPE_MISMATCH, 'lp_pool1d': ARG_TYPE_MISMATCH, 'affine_grid': CONTROL_FLOW, 'alpha_dropout': CONTROL_FLOW, 'batch_norm': CONTROL_FLOW, 'binary_cross_entropy': CONTROL_FLOW, 'binary_cross_entropy_with_logits': CONTROL_FLOW, 'celu': CONTROL_FLOW, 'cosine_embedding_loss': CONTROL_FLOW, 'cross_entropy': CONTROL_FLOW, 'ctc_loss': CONTROL_FLOW, 'dropout': CONTROL_FLOW, 'dropout1d': CONTROL_FLOW, 'dropout2d': CONTROL_FLOW, 'dropout3d': CONTROL_FLOW, 'elu': CONTROL_FLOW, 'embedding': CONTROL_FLOW, 'embedding_bag': CONTROL_FLOW, 'feature_alpha_dropout': CONTROL_FLOW, 'gaussian_nll_loss': CONTROL_FLOW, 'glu': CONTROL_FLOW, 'grid_sample': CONTROL_FLOW, 'group_norm': CONTROL_FLOW, 'gumbel_softmax': CONTROL_FLOW, 'hardsigmoid': CONTROL_FLOW, 'hardswish': CONTROL_FLOW, 'hardtanh': CONTROL_FLOW, 'hinge_embedding_loss': CONTROL_FLOW, 'huber_loss': CONTROL_FLOW, 'interpolate': CONTROL_FLOW, 'kl_div': CONTROL_FLOW, 'l1_loss': CONTROL_FLOW, 'leaky_relu': CONTROL_FLOW, 'local_response_norm': CONTROL_FLOW, 'margin_ranking_loss': CONTROL_FLOW, 'max_pool1d_with_indices': ARG_TYPE_MISMATCH, 'max_pool2d_with_indices': ARG_TYPE_MISMATCH, 'max_pool3d_with_indices': ARG_TYPE_MISMATCH, 'mse_loss': CONTROL_FLOW, 'multi_head_attention_forward': CONTROL_FLOW, 'multi_margin_loss': CONTROL_FLOW, 'multilabel_margin_loss': CONTROL_FLOW, 'multilabel_soft_margin_loss': CONTROL_FLOW, 'nll_loss': CONTROL_FLOW, 'poisson_nll_loss': CONTROL_FLOW, 'relu': CONTROL_FLOW, 'relu6': CONTROL_FLOW, 'rrelu': CONTROL_FLOW, 'selu': CONTROL_FLOW, 'silu': CONTROL_FLOW, 'mish': CONTROL_FLOW, 'smooth_l1_loss': CONTROL_FLOW, 'soft_margin_loss': CONTROL_FLOW, 'threshold': CONTROL_FLOW, 'triplet_margin_loss': CONTROL_FLOW, 'triplet_margin_with_distance_loss': CONTROL_FLOW, 'upsample': CONTROL_FLOW, 'upsample_bilinear': INTERPOLATE_ARGS_CONFLICT, 'upsample_nearest': INTERPOLATE_ARGS_CONFLICT}
    FUNCTIONALS_WITHOUT_ANNOTATION = ('adaptive_max_pool1d', 'adaptive_max_pool2d', 'adaptive_max_pool3d', 'fractional_max_pool2d', 'fractional_max_pool3d', 'max_pool1d', 'max_pool2d', 'max_pool3d', 'gaussian_nll_loss', 'upsample', 'upsample_bilinear', 'upsample_nearest')
    UNTRACEABLE_FUNCTIONALS_PY38 = {'adaptive_max_pool1d': PROXY_ITERATED, 'adaptive_max_pool2d': PROXY_ITERATED, 'adaptive_max_pool3d': PROXY_ITERATED, 'fractional_max_pool2d': PROXY_ITERATED, 'fractional_max_pool3d': PROXY_ITERATED, 'max_pool1d': PROXY_ITERATED, 'max_pool2d': PROXY_ITERATED, 'max_pool3d': PROXY_ITERATED, 'group_norm': CONTROL_FLOW}

    @classmethod
    def _get_functional(cls):
        if False:
            print('Hello World!')
        functional_list = []
        for f in dir(torch.nn.functional):
            if not f.islower():
                continue
            if f.startswith('_'):
                continue
            if f in cls.IGNORE_FUNCS:
                continue
            fn = getattr(torch.nn.functional, f)
            if not isinstance(fn, Callable):
                continue
            if f not in cls.FUNCTIONALS_WITHOUT_ANNOTATION:
                try:
                    sig = inspect.signature(fn)
                    has_tensor_arg = False
                    for param in sig.parameters.values():
                        if isinstance(param.annotation, type) and issubclass(param.annotation, torch.Tensor):
                            has_tensor_arg = True
                    if not has_tensor_arg:
                        continue
                except ValueError:
                    pass
            functional_list.append((f, fn))
        return functional_list

    @classmethod
    def generate_test_func(cls, func_name, fn):
        if False:
            i = 10
            return i + 15

        def functional_test(self):
            if False:
                while True:
                    i = 10
            if func_name in self.UNTRACEABLE_FUNCTIONALS_PY38 and sys.version_info >= (3, 8) and (sys.version_info < (3, 12)):
                (exc, err) = self.UNTRACEABLE_FUNCTIONALS_PY38[func_name]
                with self.assertRaisesRegex(exc, err):
                    symbolic_trace(fn)
            elif func_name in self.UNTRACEABLE_FUNCTIONALS:
                (exc, err) = self.UNTRACEABLE_FUNCTIONALS[func_name]
                with self.assertRaisesRegex(exc, err):
                    symbolic_trace(fn)
            else:
                symbolic_trace(fn)
        return functional_test

    @classmethod
    def generate_tests(cls):
        if False:
            while True:
                i = 10
        functional_list = cls._get_functional()
        for (func_name, fn) in functional_list:
            test_name = 'test_nn_functional_' + func_name
            functional_test = cls.generate_test_func(func_name, fn)
            setattr(cls, test_name, functional_test)

    @classmethod
    def setUpClass(cls):
        if False:
            return 10

        def no(*args, **kwargs):
            if False:
                i = 10
                return i + 15
            return False
        for name in cls.TO_PATCH.keys():
            cls.TO_PATCH[name] = getattr(torch.nn.functional, name)
            setattr(torch.nn.functional, name, no)

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        for name in cls.TO_PATCH.keys():
            setattr(torch.nn.functional, name, cls.TO_PATCH[name])
TestFunctionalTracing.generate_tests()
instantiate_device_type_tests(TestOperatorSignatures, globals())

@skipIfNoTorchVision
class TestVisionTracing(JitTestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.orig_tracer_mutable_flag = torch.fx.proxy.TracerBase.check_mutable_operations
        torch.fx.proxy.TracerBase.check_mutable_operations = True

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        torch.fx.proxy.TracerBase.check_mutable_operations = self.orig_tracer_mutable_flag
    PROXY_ITERATED = (TraceError, 'Proxy object cannot be iterated')
    INCONSISTENT_TYPE = (RuntimeError, 'Return value was annotated as having type __torch__.torchvision.models[.\\w]+ but is actually of type Tensor')
    UNTRACEABLE_MODELS = {'fasterrcnn_resnet50_fpn': PROXY_ITERATED, 'fasterrcnn_resnet50_fpn_v2': PROXY_ITERATED, 'fasterrcnn_mobilenet_v3_large_320_fpn': PROXY_ITERATED, 'fasterrcnn_mobilenet_v3_large_fpn': PROXY_ITERATED, 'maskrcnn_resnet50_fpn': PROXY_ITERATED, 'maskrcnn_resnet50_fpn_v2': PROXY_ITERATED, 'keypointrcnn_resnet50_fpn': PROXY_ITERATED, 'retinanet_resnet50_fpn': PROXY_ITERATED, 'retinanet_resnet50_fpn_v2': PROXY_ITERATED, 'ssd300_vgg16': PROXY_ITERATED, 'fcos_resnet50_fpn': PROXY_ITERATED, 'ssdlite320_mobilenet_v3_large': PROXY_ITERATED}
    UNSCRIPTABLE_MODELS = {'googlenet': INCONSISTENT_TYPE, 'inception_v3': INCONSISTENT_TYPE}
    output_transform = {'fcn_resnet50': lambda x: x['out'], 'fcn_resnet101': lambda x: x['out'], 'deeplabv3_resnet50': lambda x: x['out'], 'deeplabv3_resnet101': lambda x: x['out'], 'deeplabv3_mobilenet_v3_large': lambda x: x['out'], 'lraspp_mobilenet_v3_large': lambda x: x['out'], 'fasterrcnn_resnet50_fpn': lambda x: x[1], 'fasterrcnn_mobilenet_v3_large_fpn': lambda x: x[1], 'fasterrcnn_mobilenet_v3_large_320_fpn': lambda x: x[1], 'maskrcnn_resnet50_fpn': lambda x: x[1], 'keypointrcnn_resnet50_fpn': lambda x: x[1], 'retinanet_resnet50_fpn': lambda x: x[1]}

    @classmethod
    def generate_test_fn(cls, name, x, kwargs):
        if False:
            return 10

        def run_test(self):
            if False:
                for i in range(10):
                    print('nop')
            model = torchvision_models.get_model(name, **kwargs)
            model = model.eval()
            if name in self.UNTRACEABLE_MODELS:
                (err, exc) = self.UNTRACEABLE_MODELS[name]
                with self.assertRaisesRegex(err, exc):
                    graph = symbolic_trace(model)
            else:
                out_transform = self.output_transform.get(name, lambda x: x)
                graph: torch.fx.GraphModule = symbolic_trace(model)
                a = out_transform(model(x))
                b = out_transform(graph(x))
                self.assertEqual(a, b)
                if name in self.UNSCRIPTABLE_MODELS:
                    (err, exc) = self.UNSCRIPTABLE_MODELS[name]
                    with self.assertRaisesRegex(err, exc):
                        script = torch.jit.script(graph)
                else:
                    script = torch.jit.script(graph)
                    c = out_transform(script(x))
                    self.assertEqual(a, c)
        return run_test

    @classmethod
    def generate_classification_tests(cls):
        if False:
            print('Hello World!')
        for k in torchvision_models.list_models(module=torchvision_models):
            test_name = 'test_torchvision_models_' + k
            x = torch.rand(1, 3, 299, 299) if k in ['inception_v3'] else torch.rand(1, 3, 224, 224)
            kwargs = dict(num_classes=50)
            model_test = cls.generate_test_fn(k, x, kwargs)
            setattr(cls, test_name, model_test)

    @classmethod
    def generate_segmentation_tests(cls):
        if False:
            print('Hello World!')
        for k in torchvision_models.list_models(module=torchvision_models.segmentation):
            test_name = 'test_torchvision_models_segmentation_' + k
            x = torch.rand(1, 3, 32, 32)
            kwargs = dict(num_classes=10, pretrained_backbone=False)
            model_test = cls.generate_test_fn(k, x, kwargs)
            setattr(cls, test_name, model_test)

    @classmethod
    def generate_detection_tests(cls):
        if False:
            return 10
        for k in torchvision_models.list_models(module=torchvision_models.detection):
            test_name = 'test_torchvision_models_detection_' + k
            x = [torch.rand(3, 300, 300)]
            kwargs = dict(num_classes=10, pretrained_backbone=False)
            model_test = cls.generate_test_fn(k, x, kwargs)
            setattr(cls, test_name, model_test)

    @classmethod
    def generate_video_tests(cls):
        if False:
            return 10
        for k in torchvision_models.list_models(module=torchvision_models.video):
            test_name = 'test_torchvision_models_video_' + k
            x = torch.rand(1, 3, 4, 112, 112) if k not in {'mvit_v1_b', 'mvit_v2_s', 's3d'} else torch.rand(1, 3, 16, 224, 224)
            kwargs = dict(num_classes=50)
            model_test = cls.generate_test_fn(k, x, kwargs)
            setattr(cls, test_name, model_test)

    @classmethod
    def generate_tests(cls):
        if False:
            return 10
        cls.generate_classification_tests()
        cls.generate_detection_tests()
        cls.generate_segmentation_tests()
        cls.generate_video_tests()
if HAS_TORCHVISION:
    TestVisionTracing.generate_tests()
if __name__ == '__main__':
    run_tests()