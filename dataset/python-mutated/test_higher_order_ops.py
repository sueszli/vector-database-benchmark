import functools
import pprint
import re
import unittest
import functorch.experimental.control_flow as control_flow
import torch
import torch._dynamo.config as config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn as nn
import torch.utils._pytree as pytree
import torch.utils.checkpoint
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend, EagerAndRecordGraphs, normalize_gm
from torch._dynamo.utils import counters, ifdynstaticdefault
from torch._higher_order_ops.wrap import wrap
from torch.testing._internal.inductor_utils import HAS_CUDA
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, 'requires cuda')

def check_dynamic_shape_capture():
    if False:
        print('Hello World!')
    if not config.assume_static_by_default:
        return True
    return False

def count_ops(gm, args, freq, op):
    if False:
        return 10
    assert [node.target for node in gm.graph.nodes].count(op) == freq
    return gm

class Obj:
    pass

class MyModule(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.existing = torch.nn.Parameter(torch.ones([]))

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.existing * x
global_obj = Obj()
global_module = MyModule()
global_var = torch.randn(3)
global_num = 3.14
global_list = []

def find_first_node(gm, func):
    if False:
        for i in range(10):
            print('nop')
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None

def op_count(gm):
    if False:
        while True:
            i = 10
    result = 0
    for node in gm.graph.nodes:
        if 'call' in node.op:
            result += 1
    return result

def assert_dict_matches_regex(self, dct, dct_with_regex_keys):
    if False:
        return 10
    regex_keys = dct_with_regex_keys.keys()
    regex_key_to_actual_key = {}
    for regex_key in regex_keys:
        for key in dct:
            if re.match(regex_key, key):
                if regex_key in regex_key_to_actual_key:
                    raise AssertionError(f"Single key regex mapped to multiple keys. Please improve your regex. Got: regex='{regex_key}' keys='{regex_key_to_actual_key[regex_key]}','{key}'")
                regex_key_to_actual_key[regex_key] = key
    new_dct = {}
    for regex_key in regex_keys:
        if regex_key not in regex_key_to_actual_key:
            raise AssertionError(f"Got regex '{regex_key}' but could not match any key in dict with keys {dct.keys()}")
        new_dct[regex_key_to_actual_key[regex_key]] = dct_with_regex_keys[regex_key]
    self.assertEqual(dct, new_dct)

def default_args_generator(seed_value):
    if False:
        print('Hello World!')
    (flat_args, args_spec) = pytree.tree_flatten(seed_value)
    for i in range(3):
        new_flat_arg = []
        for val in flat_args:
            if isinstance(val, torch.Tensor):
                new_val = val + 0.1 * i
            elif isinstance(val, int):
                new_val = val + 1 * i
            elif isinstance(val, float):
                new_val = val + 0.1 * i
            else:
                raise AssertionError('unexpected arg type')
            new_flat_arg.append(new_val)
        new_args = pytree.tree_unflatten(new_flat_arg, args_spec)
        yield new_args

class HigherOrderOpTests(torch._dynamo.test_case.TestCase):

    def _assert_wrap_fallback(self, func, args, setup=lambda : None):
        if False:
            print('Hello World!')
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        setup()
        expected = func(*args)
        setup()
        result = torch.compile(func, backend=cnt, fullgraph=False)(*args)
        num_graph_breaks = len(counters['graph_break'].keys())
        self.assertGreater(num_graph_breaks, 0)
        for gm in backend.graphs:
            for node in gm.graph.nodes:
                self.assertFalse(node.target is wrap)
        self.assertEqual(result, expected)

    def _test_wrap_simple(self, func, args_generator, expected_num_wrap_args, expected_opcount=2, return_graph=False):
        if False:
            i = 10
            return i + 15
        graph = None
        for (i, args) in enumerate(args_generator):
            backend = EagerAndRecordGraphs()
            cnt = CompileCounterWithBackend(backend)
            expected = func(*args)
            result = torch.compile(func, fullgraph=True, backend=cnt)(*args)
            self.assertEqual(result, expected)
            self.assertEqual(cnt.frame_count, 1)
            self.assertEqual(len(backend.graphs), 1)
            if i == 0:
                self.assertEqual(cnt.op_count, expected_opcount)
                graph = backend.graphs[0]
                wrap_node = find_first_node(graph, wrap)
                self.assertEqual(len(wrap_node.args), expected_num_wrap_args)
        if return_graph:
            return normalize_gm(graph.print_readable(print_output=False))

    def test_error_message_sane(self):
        if False:
            while True:
                i = 10
        foo = []

        def inner(x):
            if False:
                while True:
                    i = 10
            foo.append(x)
            return x.clone()

        @torch.compile(backend='eager', fullgraph=True)
        def f(x):
            if False:
                i = 10
                return i + 15
            return wrap(inner, x)
        x = torch.randn(3)
        with self.assertRaisesRegex(RuntimeError, 'while introspecting wrap, we were unable to trace function `inner`'):
            f(x)

    def test_no_freevars(self):
        if False:
            return 10

        def f(x):
            if False:
                i = 10
                return i + 15
            return wrap(lambda x: torch.sin(x), x)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_return_captured_var(self):
        if False:
            print('Hello World!')
        freevar = torch.randn(3)

        def test(x):
            if False:
                for i in range(10):
                    print('nop')
            return freevar

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(test, x)
        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 2)

    def test_return_captured_vars(self):
        if False:
            print('Hello World!')
        freevar1 = torch.randn(3)
        freevar2 = torch.randn(3)

        def test(x):
            if False:
                for i in range(10):
                    print('nop')
            return (freevar1, freevar2, freevar1)

        def fn(x):
            if False:
                return 10
            return wrap(test, x)
        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 3, 4)

    def test_return_captured_var_used_multiple_times(self):
        if False:
            print('Hello World!')
        freevar = torch.randn(3)

        def test(x):
            if False:
                for i in range(10):
                    print('nop')
            y = x + freevar
            return (y, freevar)

        def fn(x):
            if False:
                print('Hello World!')
            return wrap(test, x)
        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 3, 3)

    def test_capture_untracked_global(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            return wrap(lambda x: x + global_var, x)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_symint_input(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            i = x.size(0)
            return wrap(lambda x, i: x.view(i), x, i)
        x = torch.randn(3, 1)
        self._test_wrap_simple(f, default_args_generator((x,)), ifdynstaticdefault(2, 3), expected_opcount=ifdynstaticdefault(2, 3))

    def test_wrap_pytree_args_nested(self):
        if False:
            i = 10
            return i + 15

        def f(x, y, z):
            if False:
                print('Hello World!')

            def fn(d):
                if False:
                    return 10
                return d['x'].sin() + d['y'][0].cos() - d['y'][1][2].sin()
            return wrap(fn, d)
        x = torch.tensor(1.5)
        y = torch.tensor(2.0)
        z = torch.tensor(3.0)
        d = {'x': x, 'y': (y, [x, y, z])}

        def my_args_generator(t):
            if False:
                for i in range(10):
                    print('nop')
            yield t
            yield (t[0] + 0.1, t[1], t[2])
            yield (t[0], t[1] + 0.1, t[2])
        actual_graph = self._test_wrap_simple(f, my_args_generator((x, y, z)), 4, return_graph=True)
        self.assertExpectedInline(actual_graph, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_d_x_ : torch.Tensor, L_d_y_0_ : torch.Tensor, L_d_y_1_2_ : torch.Tensor):\n        l_d_x_ = L_d_x_\n        l_d_y_0_ = L_d_y_0_\n        l_d_y_1_2_ = L_d_y_1_2_\n\n        wrap_body_0 = self.wrap_body_0\n        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_d_x_, l_d_y_0_, l_d_y_1_2_);  wrap_body_0 = l_d_x_ = l_d_y_0_ = l_d_y_1_2_ = None\n        getitem = wrap[0];  wrap = None\n        return (getitem,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_d_x_, l_d_y_0_, l_d_y_1_2_):\n            sin = l_d_x_.sin();  l_d_x_ = None\n            cos = l_d_y_0_.cos();  l_d_y_0_ = None\n            add = sin + cos;  sin = cos = None\n            sin_1 = l_d_y_1_2_.sin();  l_d_y_1_2_ = None\n            sub = add - sin_1;  add = sin_1 = None\n            return (sub,)\n')

    def test_wrap_pytree_args_with_symint_constant(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                return 10
            i = x.size(0)
            return wrap(lambda t: t[0].view(t[2]) + t[1], (x, y, i))
        x = torch.randn(3, 1)
        y = 0.5
        actual_graph = self._test_wrap_simple(f, default_args_generator((x, y)), ifdynstaticdefault(2, 3), expected_opcount=ifdynstaticdefault(2, 3), return_graph=True)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(actual_graph, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        wrap_body_0 = self.wrap_body_0\n        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None\n        getitem = wrap[0];  wrap = None\n        return (getitem,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            view = l_x_.view(3);  l_x_ = None\n            add = view + 0.5;  view = None\n            return (add,)\n')
        else:
            self.assertExpectedInline(actual_graph, 'class GraphModule(torch.nn.Module):\n    def forward(self, s0 : torch.SymInt, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        size = l_x_.size(0)\n\n        wrap_body_0 = self.wrap_body_0\n        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_, size);  wrap_body_0 = l_x_ = size = None\n        getitem = wrap[0];  wrap = None\n        return (getitem,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_, size):\n            view = l_x_.view(size);  l_x_ = size = None\n            add = view + 0.5;  view = None\n            return (add,)\n')

    def test_wrap_pytree_kwargs(self):
        if False:
            i = 10
            return i + 15

        def f(x, y, z):
            if False:
                return 10

            def fn(*, x, y, z):
                if False:
                    while True:
                        i = 10
                (z1, z2) = z
                return x * 2 + y + z1
            return wrap(fn, x=x, y=y, z=z)
        x = torch.randn(3)
        y = torch.randn(3, 3)

        def my_args_generator(t):
            if False:
                i = 10
                return i + 15
            yield t
            x1 = t[0] + 0.1
            y1 = t[1] + 0.1
            yield (x1, y1, (x1, y1))
            x2 = t[0] + 0.2
            y2 = t[0] + 0.2
            yield (x2, y2, (x2, y2))
        self._test_wrap_simple(f, my_args_generator((x, y, (x, y))), 3)

    def test_wrap_pytree_args_not_const_symint_tensor(self):
        if False:
            while True:
                i = 10

        class MyClass:

            def __init__(self, x):
                if False:
                    return 10
                self.val = x

        def f(x, y):
            if False:
                while True:
                    i = 10
            return wrap(lambda z: z[0].sin() * z[1].val.cos(), (x, y))
        x = torch.tensor(1.2)
        y = MyClass(torch.tensor(3.4))
        self._assert_wrap_fallback(f, (x, y))

    def test_capture_constants(self):
        if False:
            print('Hello World!')
        x = torch.randn(3, 3)
        y = 4.0

        def fn(x, y, z):
            if False:
                for i in range(10):
                    print('nop')
            if z:
                return x + y
            return x * y

        def f(x, y, z):
            if False:
                while True:
                    i = 10
            return wrap(fn, x, y, z)
        args = (x, 4.0, None)
        opt_f = torch.compile(f, fullgraph=True, backend=CompileCounter())
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)
        args = (x, 5.0, None)
        expected = f(*args)
        result = opt_f(*args)
        self.assertEqual(result, expected)

    def test_capture_untracked_global_nested(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(lambda x: wrap(lambda x: x + global_var, x), x)
        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, x + global_var)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_capture_untracked_nonlocal(self):
        if False:
            return 10
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')

            def g(x):
                if False:
                    while True:
                        i = 10
                return wrap(lambda x: x + y, x)
            self._test_wrap_simple(g, default_args_generator((x,)), 3)
            return g(x)
        f(x, y)

    def test_capture_tracked(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            if False:
                while True:
                    i = 10
            return wrap(lambda x: x + y, x)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_capture_tracked_nested(self):
        if False:
            print('Hello World!')
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def f(x, y):
            if False:
                return 10
            return wrap(lambda x: wrap(lambda x: x + y, x), x)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_inlined_functions(self):
        if False:
            while True:
                i = 10

        def g(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return wrap(lambda x: g(x, y), x)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_same_freevar_twice(self):
        if False:
            i = 10
            return i + 15
        free = torch.randn(3)

        def g(x):
            if False:
                for i in range(10):
                    print('nop')
            y = free.sin()
            z = free.cos()
            return (y, z)

        def f(x):
            if False:
                print('Hello World!')
            return wrap(g, x)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 2, 3)

    def test_capture_value_created_in_subgraph(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            if False:
                print('Hello World!')
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            if False:
                i = 10
                return i + 15
            return wrap(inner, x, y)
        result = f(x, y)
        self.assertEqual(result, x + y + x)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 1)
        gm = backend.graphs[0]
        wrap_node = find_first_node(gm, wrap)
        self.assertTrue(len(wrap_node.args), 3)
        body_function = getattr(gm, wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 3)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)
        body_function = getattr(body_function, inner_wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)
        inner_wrap_node = find_first_node(body_function, wrap)
        self.assertTrue(len(inner_wrap_node.args), 3)

    def test_side_effect_set_new_attr_global_obj(self):
        if False:
            while True:
                i = 10

        def setup():
            if False:
                for i in range(10):
                    print('nop')
            global global_obj
            global_obj = Obj()

        def f(x):
            if False:
                for i in range(10):
                    print('nop')

            def h(x):
                if False:
                    print('Hello World!')

                def g(x):
                    if False:
                        print('Hello World!')
                    global_obj.foo = x + 1
                    return x.clone()
                y = wrap(g, x)
                return y + global_obj.foo
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_obj(self):
        if False:
            for i in range(10):
                print('nop')

        def setup():
            if False:
                i = 10
                return i + 15
            global global_obj
            global_obj = Obj()
            global_obj.foo = nn.Parameter(torch.tensor(4.0))

        def f(x):
            if False:
                for i in range(10):
                    print('nop')

            def h(x):
                if False:
                    while True:
                        i = 10

                def g(x):
                    if False:
                        while True:
                            i = 10
                    global_obj.foo = x + 1
                    return x.clone()
                y = wrap(g, x)
                return y + global_obj.foo
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_obj(self):
        if False:
            print('Hello World!')

        def setup():
            if False:
                for i in range(10):
                    print('nop')
            global global_obj
            global_obj = Obj()
            global_obj.foo = torch.tensor(4.0)

        def f(x):
            if False:
                print('Hello World!')

            def h(x):
                if False:
                    return 10

                def g(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    del global_obj.foo
                    return x.clone()
                y = wrap(g, x)
                return y
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_set_new_attr_global_module(self):
        if False:
            print('Hello World!')

        def setup():
            if False:
                print('Hello World!')
            global global_module
            global_module = MyModule()

        def h(x):
            if False:
                for i in range(10):
                    print('nop')

            def g(x):
                if False:
                    i = 10
                    return i + 15
                global_module.foo = nn.Parameter(x + 1)
                return x.clone()
            y = wrap(g, x)
            return y + global_module.foo
        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_set_existing_attr_global_module(self):
        if False:
            print('Hello World!')

        def setup():
            if False:
                while True:
                    i = 10
            global global_module
            global_module = MyModule()

        def h(x):
            if False:
                while True:
                    i = 10

            def g(x):
                if False:
                    while True:
                        i = 10
                global_module.existing = nn.Parameter(torch.tensor(4.0))
                return global_module(x)
            y = wrap(g, x)
            return y
        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_del_existing_attr_global_module(self):
        if False:
            while True:
                i = 10

        def setup():
            if False:
                i = 10
                return i + 15
            global global_module
            global_module = MyModule()

        def h(x):
            if False:
                i = 10
                return i + 15

            def g(x):
                if False:
                    i = 10
                    return i + 15
                del global_module.existing
                return x.clone()
            y = wrap(g, x)
            return y
        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,), setup=setup)

    def test_side_effect_mutate_global_num(self):
        if False:
            return 10

        def setup():
            if False:
                while True:
                    i = 10
            global global_num
            global_num = 3.14

        def f(x):
            if False:
                return 10

            def g(x):
                if False:
                    return 10
                global global_num
                global_num = global_num + 1
                return x + global_num
            y = wrap(g, x)
            return y + global_num
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_num_builtin(self):
        if False:
            while True:
                i = 10

        def setup():
            if False:
                for i in range(10):
                    print('nop')
            global global_num
            global_num = 3.14

        def f(x):
            if False:
                while True:
                    i = 10

            def g(x):
                if False:
                    while True:
                        i = 10
                global global_num
                global_num += 1
                return x + global_num
            y = wrap(g, x)
            return y + global_num
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def setup():
            if False:
                print('Hello World!')
            global global_var
            global_var = torch.ones(3)

        def f(x):
            if False:
                return 10

            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                global global_var
                global_var = global_var + 1
                return x + global_var
            y = wrap(g, x)
            return y + global_var
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_tensor_builtin(self):
        if False:
            print('Hello World!')

        def setup():
            if False:
                print('Hello World!')
            global global_var
            global_var = torch.ones(3)

        def f(x):
            if False:
                return 10

            def g(x):
                if False:
                    i = 10
                    return i + 15
                global global_var
                global_var += 1
                return x + global_var
            y = wrap(g, x)
            return y + global_var
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_global_list(self):
        if False:
            i = 10
            return i + 15

        def setup():
            if False:
                i = 10
                return i + 15
            global global_list
            global_list = []

        def f(x):
            if False:
                return 10

            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                val = x + 1
                global_list.append(val)
                return global_list[-1]
            y = wrap(g, x)
            z = y + global_list[-1]
            return z
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,), setup=setup)

    def test_side_effect_mutate_nonlocal_num(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                while True:
                    i = 10

            def h(x):
                if False:
                    while True:
                        i = 10
                val = 1

                def g(x):
                    if False:
                        while True:
                            i = 10
                    nonlocal val
                    val = val + 1
                    return x + val
                y = wrap(g, x)
                z = y + val
                return z
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_obj(self):
        if False:
            return 10

        def f(x):
            if False:
                print('Hello World!')

            def h(x):
                if False:
                    i = 10
                    return i + 15
                obj = Obj()

                def g(x):
                    if False:
                        print('Hello World!')
                    obj.val = x.dim()
                    return x.clone()
                y = wrap(g, x)
                z = y + obj.val
                return z
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_existing_attr_nonlocal_obj(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                i = 10
                return i + 15

            def h(x):
                if False:
                    while True:
                        i = 10
                obj = Obj()
                obj.val = 3

                def g(x):
                    if False:
                        i = 10
                        return i + 15
                    obj.val = x.dim()
                    return x.clone()
                y = wrap(g, x)
                z = y + obj.val
                return z
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_del_existing_attr_nonlocal_obj(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')

            def h(x):
                if False:
                    print('Hello World!')
                obj = Obj()
                obj.val = 3

                def g(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    del obj.val
                    return x.clone()
                y = wrap(g, x)
                return y
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_set_new_attr_nonlocal_module(self):
        if False:
            while True:
                i = 10

        def h(x):
            if False:
                for i in range(10):
                    print('nop')
            obj = MyModule()

            def g(x):
                if False:
                    i = 10
                    return i + 15
                obj.val = x.dim()
                return x.clone()
            y = wrap(g, x)
            z = y + obj.val
            return z
        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_set_existing_attr_nonlocal_module(self):
        if False:
            return 10

        def h(x):
            if False:
                print('Hello World!')
            obj = MyModule()

            def g(x):
                if False:
                    return 10
                obj.existing = nn.Parameter(torch.tensor(3.14))
                return obj(x)
            y = wrap(g, x)
            return y
        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_del_existing_attr_nonlocal_module(self):
        if False:
            while True:
                i = 10

        def h(x):
            if False:
                i = 10
                return i + 15
            obj = MyModule()

            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                del obj.existing
                return x.clone()
            y = wrap(g, x)
            return y
        x = torch.zeros([])
        self._assert_wrap_fallback(h, (x,))

    def test_side_effect_mutate_nonlocal_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')

            def h(x):
                if False:
                    while True:
                        i = 10
                val = torch.tensor(1.0)

                def g(x):
                    if False:
                        i = 10
                        return i + 15
                    nonlocal val
                    val = val + 1
                    return x + val
                y = wrap(g, x)
                z = y + val
                return z
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_num_builtin(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10

            def h(x):
                if False:
                    print('Hello World!')
                val = 1

                def g(x):
                    if False:
                        i = 10
                        return i + 15
                    nonlocal val
                    val += 1
                    return x + val
                y = wrap(g, x)
                z = y + val
                return z
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_mutate_nonlocal_tensor_builtin(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')

            def h(x):
                if False:
                    for i in range(10):
                        print('nop')
                val = torch.tensor(1.0)

                def g(x):
                    if False:
                        i = 10
                        return i + 15
                    nonlocal val
                    val += 1
                    return x + val
                y = wrap(g, x)
                z = y + val
                return z
            return h(x)
        x = torch.zeros([])
        self._assert_wrap_fallback(f, (x,))

    def test_side_effect_nonlocal_list_append_graph_break(self):
        if False:
            return 10

        def g(x):
            if False:
                while True:
                    i = 10
            y = []

            def f(k):
                if False:
                    i = 10
                    return i + 15
                m = k + 1
                y.append(m)
                return k
            wrap(f, x)
            return y[0]
        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_nested_nonlocal_list_append_graph_break(self):
        if False:
            for i in range(10):
                print('nop')

        def g(x):
            if False:
                while True:
                    i = 10

            def h(x):
                if False:
                    i = 10
                    return i + 15
                y = []

                def f(k):
                    if False:
                        return 10
                    m = k + 1
                    y.append(m)
                    return k
                wrap(f, x)
                return y[0]
            return h(x)
        x = torch.randn(3, 3)
        self._assert_wrap_fallback(g, (x,))

    def test_side_effect_local_list_append_no_graph_break(self):
        if False:
            while True:
                i = 10

        def g(x):
            if False:
                print('Hello World!')

            def f(k):
                if False:
                    return 10
                y = []
                y.append(k + 1)
                return y[0]
            return wrap(f, x)
        x = torch.randn(3, 3)
        self._test_wrap_simple(g, default_args_generator((x,)), 2)

    def test_wrap_kwarg(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                return 10
            return wrap(lambda x, y: x + y, x, y=y)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_int(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(lambda x, y: x + y, x, y=y)
        x = torch.randn(3)
        y = 8
        self._test_wrap_simple(f, default_args_generator((x, y)), ifdynstaticdefault(2, 3))

    def test_wrap_all_kwarg(self):
        if False:
            while True:
                i = 10

        def f(y, x):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(lambda x, y: x * 2 + y, x=x, y=y)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_only(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                while True:
                    i = 10

            def fn(*, x, y):
                if False:
                    return 10
                return x * 2 + y
            return wrap(fn, x=x, y=y)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_default(self):
        if False:
            return 10

        def f(x, y):
            if False:
                i = 10
                return i + 15

            def fn(*, x, y, z=8):
                if False:
                    return 10
                return x * 2 + y + z
            return wrap(fn, x=x, y=y)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_default_if_branch(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                return 10

            def fn(*, x, y, z=None):
                if False:
                    while True:
                        i = 10
                if z is None:
                    return x * 2 + y
                else:
                    return 2 * x
            return wrap(fn, x=x, y=y)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y)), 3)

    def test_wrap_kwarg_recompile(self):
        if False:
            i = 10
            return i + 15

        def f(x, y, z=None):
            if False:
                for i in range(10):
                    print('nop')

            def fn(*, x, y, z=None):
                if False:
                    while True:
                        i = 10
                if z is None:
                    return x * 2 + y
                else:
                    return 2 * x
            return wrap(fn, x=x, y=y, z=z)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        counters.clear()
        opt = torch.compile(f, backend='eager', fullgraph=True)
        opt(x, y)
        self.assertEqual(counters['stats']['calls_captured'], 2)
        opt(x, y)
        self.assertEqual(counters['stats']['calls_captured'], 2)
        output = opt(x, y, 8)
        self.assertEqual(counters['stats']['calls_captured'], 4)
        self.assertEqual(output, 2 * x)

    def test_wrap_kwarg_default_else_branch(self):
        if False:
            print('Hello World!')

        def f(x, y, z):
            if False:
                i = 10
                return i + 15

            def fn(*, x, y, z=None):
                if False:
                    i = 10
                    return i + 15
                if z is None:
                    return x * 2 + y
                else:
                    return 2 * x
            return wrap(fn, x=x, y=y, z=z)
        x = torch.randn(3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(f, default_args_generator((x, y, 8)), 2)

    def test_map_subgraph_name_is_valid(self):
        if False:
            i = 10
            return i + 15
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=cnt, fullgraph=True)
        def map_f(xs, y):
            if False:
                return 10

            def inner(x, y):
                if False:
                    return 10

                def inner2(x, y):
                    if False:
                        for i in range(10):
                            print('nop')
                    return x + y
                return control_flow.map(inner2, x, y)
            return control_flow.map(inner, xs, y)
        result = map_f(xs, y)
        self.assertEqual(result, xs + y)
        map_gm = backend.graphs[0]
        name_set = set()
        for (name, _) in map_gm.named_modules():
            name_set.add(name)
        self.assertEqual(name_set, {'', 'map_body_1.map_body_0', 'map_body_1'})

    def test_map_multi_return(self):
        if False:
            i = 10
            return i + 15
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            if False:
                print('Hello World!')
            return control_flow.map(lambda x: (x.sin(), x.sin()), x)
        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, (x.sin(), x.sin()))
        self.assertEqual(cnt.frame_count, 0)

    def test_map_kwargs(self):
        if False:
            return 10
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return control_flow.map(lambda x: x.sin(), x=x)
        x = torch.randn(3)
        self.assertRaises(TypeError, lambda : f(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_map_symint_input(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        def fn(x, y):
            if False:
                i = 10
                return i + 15

            def inner(x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return torch.sin(x + y)
            return control_flow.map(inner, x, y.size(0))
        x = torch.randn(3, 1)
        y = torch.randn(3, 1)
        compiled_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        ref = fn(x, y)
        res = compiled_fn(x, y)
        self.assertEqual(ref, res)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, ifdynstaticdefault(2, 3))

    def test_cond_subgraph_name_is_valid(self):
        if False:
            i = 10
            return i + 15
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            if False:
                for i in range(10):
                    print('nop')

            def true_fn(pred2, x, y):
                if False:
                    for i in range(10):
                        print('nop')
                return x + y

            def false_fn(pred2, x, y):
                if False:
                    print('Hello World!')

                def true_fn2(x, y):
                    if False:
                        print('Hello World!')
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    if False:
                        while True:
                            i = 10
                    return x.cos() - y.sin()
                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])
            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])
        result = cond_f(pred, pred2, xs, y)
        self.assertEqual(result, xs + y)
        cond_gm = backend.graphs[0]
        name_set = set()
        for (name, _) in cond_gm.named_modules():
            name_set.add(name)
        self.assertEqual(name_set, {'', 'cond_true_1', 'cond_false_1', 'cond_false_1.cond_false_0', 'cond_false_1.cond_true_0'})

    @torch._dynamo.config.patch(assume_static_by_default=True, dynamic_shapes=True)
    def test_cond_graph_break_in_one_branch(self):
        if False:
            for i in range(10):
                print('nop')
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()
                self.register_buffer('buffer', torch.ones(6, 4))

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')

                def true_fn(x):
                    if False:
                        print('Hello World!')
                    self.buffer += 1
                    return self.buffer.sum() + x.sum()

                def false_fn(x):
                    if False:
                        i = 10
                        return i + 15
                    return (x - 1).sum()
                return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [x])
        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True)
        mod_for_eager = Foo()
        with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
            mod_for_eager(torch.ones(6, 4))
        with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
            mod_for_compile(torch.ones(3, 4))

    def test_cond_free_variable_in_both_branches(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        z = torch.ones(4, 4)

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('buffer', torch.ones(6, 4))

            def forward(self, x, y):
                if False:
                    print('Hello World!')

                def true_fn(x):
                    if False:
                        return 10
                    return x.sum() + self.buffer.sum() + z.sum()

                def false_fn(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    return x.sum() - z.sum() - self.buffer.sum()
                return control_flow.cond(y, true_fn, false_fn, [x])
        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True, fullgraph=True)
        mod_for_eager = Foo()
        self.assertEqual(mod_for_compile(torch.tensor(True), torch.tensor(5)), mod_for_eager(torch.tensor(True), torch.tensor(5)))
        for node in backend.graphs[0].graph.nodes:
            if node.op == 'call_function' and node.target == torch.ops.higher_order.cond:
                (_, _, _, operands) = node.args
                self.assertEqual(len(operands), 4)
            if node.op == 'get_attr':
                if str(node.target) in 'cond_true_0, cond_false_0':
                    num_placeholders = len([node for node in getattr(backend.graphs[0], str(node.target)).graph.nodes if node.op == 'placeholder'])
                    self.assertEqual(num_placeholders, 4)

    def _check_cond_graph_and_extract(self, fn, args):
        if False:
            print('Hello World!')
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        out = torch.compile(fn, backend=cnt, fullgraph=True)(*args)
        self.assertEqual(out, fn(*args))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)
        if check_dynamic_shape_capture():
            return
        gm = backend.graphs[0]
        graph = gm.code.strip()
        true_graph = gm.cond_true_0.code.strip()
        false_graph = gm.cond_false_0.code.strip()
        return (graph, true_graph, false_graph)

    def test_cond_branches_no_arguments(self):
        if False:
            return 10

        def fn(x):
            if False:
                print('Hello World!')

            def true_fn():
                if False:
                    return 10
                return torch.sin(x)

            def false_fn():
                if False:
                    i = 10
                    return i + 15
                return torch.cos(x)
            return control_flow.cond(x.sum() > 0, true_fn, false_fn, tuple())
        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            (graph, true_graph, false_graph) = graphs
            self.assertExpectedInline(graph, 'def forward(self, L_x_ : torch.Tensor):\n    l_x_ = L_x_\n    sum_1 = l_x_.sum()\n    gt = sum_1 > 0;  sum_1 = None\n    cond_true_0 = self.cond_true_0\n    cond_false_0 = self.cond_false_0\n    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, [l_x_]);  gt = cond_true_0 = cond_false_0 = l_x_ = None\n    return (cond,)')
            self.assertExpectedInline(true_graph, 'def forward(self, l_x_):\n    l_x__1 = l_x_\n    sin = torch.sin(l_x__1);  l_x__1 = None\n    return sin')
            self.assertExpectedInline(false_graph, 'def forward(self, l_x_):\n    l_x__1 = l_x_\n    cos = torch.cos(l_x__1);  l_x__1 = None\n    return cos')

    def test_cond_branches_no_arguments_no_closure(self):
        if False:
            return 10

        def fn(x):
            if False:
                print('Hello World!')

            def true_fn():
                if False:
                    i = 10
                    return i + 15
                return torch.ones(3, 4)

            def false_fn():
                if False:
                    print('Hello World!')
                return torch.ones(3, 4).sin()
            return control_flow.cond(x.sum() > 0, true_fn, false_fn, tuple())
        self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        graphs = self._check_cond_graph_and_extract(fn, (torch.randn(4, 5),))
        if graphs is not None:
            (graph, true_graph, false_graph) = graphs
            self.assertExpectedInline(graph, 'def forward(self, L_x_ : torch.Tensor):\n    l_x_ = L_x_\n    sum_1 = l_x_.sum();  l_x_ = None\n    gt = sum_1 > 0;  sum_1 = None\n    cond_true_0 = self.cond_true_0\n    cond_false_0 = self.cond_false_0\n    cond = torch.ops.higher_order.cond(gt, cond_true_0, cond_false_0, []);  gt = cond_true_0 = cond_false_0 = None\n    return (cond,)')
            self.assertExpectedInline(true_graph, 'def forward(self):\n    ones = torch.ones(3, 4)\n    return ones')
            self.assertExpectedInline(false_graph, 'def forward(self):\n    ones = torch.ones(3, 4)\n    sin = ones.sin();  ones = None\n    return sin')

    def test_cond_side_effect_in_one_branches(self):
        if False:
            print('Hello World!')
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        z = [torch.ones(4, 4)]

        class Foo(torch.nn.Module):

            def __init__(self):
                if False:
                    return 10
                super().__init__()

            def forward(self, y, x):
                if False:
                    return 10

                def true_fn(x):
                    if False:
                        i = 10
                        return i + 15
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x.sum() + z[-1].sum()

                def false_fn(x):
                    if False:
                        while True:
                            i = 10
                    return x.sum() - z[0].sum()
                return control_flow.cond(y, true_fn, false_fn, [x])
        mod_for_eager = Foo()
        mod_for_compile = torch.compile(Foo(), backend=cnt, dynamic=True, fullgraph=False)
        with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
            mod_for_eager(torch.tensor(True), torch.tensor(5))
        with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
            mod_for_compile(torch.tensor(True), torch.tensor(5))

    def test_cond_with_constant_pred(self):
        if False:
            for i in range(10):
                print('nop')

        def test(pred, x):
            if False:
                for i in range(10):
                    print('nop')

            def true_fn(x):
                if False:
                    print('Hello World!')
                return x

            def false_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return -x
            return control_flow.cond(pred, true_fn, false_fn, [x])
        opt_test = torch.compile(test, backend='eager')
        inp = torch.ones(3, 3)
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))

    def test_map_graph_break(self):
        if False:
            return 10
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)

        class Module(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_buffer('w', torch.ones(6, 4))

            def forward(self, xs):
                if False:
                    while True:
                        i = 10

                def body(x):
                    if False:
                        i = 10
                        return i + 15
                    self.w += 1
                    return x
                return control_flow.map(body, xs)
        mod = Module()
        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)
        mod_for_eager = Module()
        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        self.assertEqual(len(backend.graphs), 0)
        self.assertEqual(res, mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]])))

    def test_map_side_effect(self):
        if False:
            i = 10
            return i + 15
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        z = [torch.ones(6, 4)]

        class Module(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.register_buffer('w', torch.ones(6, 4))

            def forward(self, xs):
                if False:
                    return 10

                def body(x):
                    if False:
                        return 10
                    z.append(x)
                    z.append(x)
                    z.pop()
                    return x + z[-1].sum()
                return control_flow.map(body, xs)
        mod = Module()
        mod_for_compile = torch.compile(mod, backend=cnt, dynamic=True, fullgraph=False)
        mod_for_eager = Module()
        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        res = mod_for_compile(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        eager = mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        eager = mod_for_eager(torch.Tensor([[6, 4, 5], [3, 4, 5], [6, 6, 6]]))
        self.assertEqual(len(backend.graphs), 0)
        self.assertEqual(res, eager)

    def test_wrap_subgraph_name_is_valid(self):
        if False:
            for i in range(10):
                print('nop')
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)

        def inner(x, y):
            if False:
                for i in range(10):
                    print('nop')
            z = x + y
            return wrap(lambda x: wrap(lambda x: x + z, x), x)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x, y):
            if False:
                print('Hello World!')
            return wrap(inner, x, y)
        result = f(x, y)
        self.assertEqual(result, x + y + x)
        wrap_gm = backend.graphs[0]
        names = set()
        for (mod_name, _) in wrap_gm.named_modules():
            names.add(mod_name)
        self.assertEqual(names, {'', 'wrap_body_2', 'wrap_body_2.wrap_body_1', 'wrap_body_2.wrap_body_1.wrap_body_0'})

    def test_wrap_allow_local_assign_in_body_fn(self):
        if False:
            print('Hello World!')

        def f(arg1, arg2):
            if False:
                for i in range(10):
                    print('nop')

            def inner_f(arg1, arg2):
                if False:
                    while True:
                        i = 10
                a = arg1
                b = arg2
                ret = []
                for x in a:
                    ret.append(x + 1)
                for x in b:
                    ret.append(x + 1)
                return ret
            return wrap(inner_f, arg1, arg2)
        x = torch.ones(3)

        def my_args_generator():
            if False:
                for i in range(10):
                    print('nop')
            yield ([x], [x.sin()])
            yield ((x,), (x.sin(),))
        actual_graph = self._test_wrap_simple(f, my_args_generator(), 3, 3, return_graph=True)
        if check_dynamic_shape_capture():
            return
        self.assertExpectedInline(actual_graph, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_arg1_0_ : torch.Tensor, L_arg2_0_ : torch.Tensor):\n        l_arg1_0_ = L_arg1_0_\n        l_arg2_0_ = L_arg2_0_\n\n        wrap_body_0 = self.wrap_body_0\n        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_arg1_0_, l_arg2_0_);  wrap_body_0 = l_arg1_0_ = l_arg2_0_ = None\n        getitem = wrap[0]\n        getitem_1 = wrap[1];  wrap = None\n        return (getitem, getitem_1)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_arg1_0_, l_arg2_0_):\n            add = l_arg1_0_ + 1;  l_arg1_0_ = None\n\n            add_1 = l_arg2_0_ + 1;  l_arg2_0_ = None\n            return (add, add_1)\n')

    def test_capture_global_num(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            return wrap(lambda x: x + global_num, x)
        x = torch.zeros([])
        self._test_wrap_simple(f, default_args_generator((x,)), 2)

    def test_capture_global_num_adds_guard(self):
        if False:
            i = 10
            return i + 15

        @torch.compile(backend='eager', fullgraph=True)
        def f(x):
            if False:
                while True:
                    i = 10
            return wrap(lambda x: x + global_num, x)
        global global_num
        x = torch.zeros([])
        result = f(x)
        self.assertEqual(result, x + global_num)
        global_num = torch.randn([]).item()
        result = f(x)
        self.assertEqual(result, x + global_num)

    def test_capture_input_num(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                while True:
                    i = 10
            return wrap(lambda x: x + y, x)
        x = torch.zeros([])
        y = 3.14
        self._test_wrap_simple(f, default_args_generator((x, y)), 2)

    def test_side_effect_in_body(self):
        if False:
            i = 10
            return i + 15
        counters.clear()
        backend = EagerAndRecordGraphs()
        x = torch.randn([])
        y = torch.randn([])

        def inner(x):
            if False:
                i = 10
                return i + 15
            nonlocal y
            y = x
            return x.clone()

        @torch.compile(backend=backend)
        def f(x):
            if False:
                return 10
            return wrap(inner, x)
        f(x)
        self.assertEqual(y, x)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {'.*HigherOrderOperator: Mutating a variable not in the current scope \\(SideEffects\\)': 1})

    def test_fallback_on_graph_break_simple(self):
        if False:
            i = 10
            return i + 15
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            if False:
                i = 10
                return i + 15
            y = x.sin()
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            if False:
                while True:
                    i = 10
            return wrap(inner, x)
        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 0)

    def test_fallback_on_graph_break_complicated(self):
        if False:
            while True:
                i = 10
        cnt = CompileCounter()
        x = torch.randn([])

        def inner(x):
            if False:
                i = 10
                return i + 15
            y = x.sin()
            y = y * global_var
            torch._dynamo.graph_break()
            z = y.sin()
            return z

        @torch.compile(backend=cnt)
        def f(x):
            if False:
                while True:
                    i = 10
            x = x.clone()
            result = wrap(inner, x)
            return result.clone()
        result = f(x)
        self.assertEqual(result, inner(x))
        self.assertEqual(cnt.frame_count, 2)

    def test_modules(self):
        if False:
            while True:
                i = 10
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            if False:
                return 10
            return wrap(lambda x: mod(x), x)
        result = f(x)
        self.assertEqual(result, mod(x))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)
        self.assertTrue(len(dict(backend.graphs[0].named_parameters())) == 2)
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        linear_node = find_first_node(body_function, torch._C._nn.linear)
        self.assertTrue(linear_node is not None)
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_flat_list_output(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            return wrap(lambda x: [torch.sin(x), torch.cos(x)], x)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 2, expected_opcount=3)

    def test_fallback_on_python_primitives_output(self):
        if False:
            i = 10
            return i + 15
        counters.clear()
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(lambda x: [1, torch.sin(x), 2.0], x)
        x = torch.randn(3)
        result = f(x)
        self.assertEqual(result, [1, torch.sin(x), 2.0])
        self.assertEqual(cnt.frame_count, 0)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {".*HigherOrderOperator body's output must consist of tensors only": 1})

    def test_nested_tuple_output(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            ((a, b),) = wrap(lambda x: ((x.sin(), x.cos()),), x)
            return a + b
        x = torch.randn(2, 3)
        counters.clear()
        graph = self._test_wrap_simple(f, default_args_generator((x,)), 2, 4, return_graph=True)
        self.assertEqual(len(counters['graph_break']), 0)
        if check_dynamic_shape_capture():
            return
        self.assertExpectedInline(graph, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        wrap_body_0 = self.wrap_body_0\n        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None\n        a = wrap[0]\n        b = wrap[1];  wrap = None\n\n        add = a + b;  a = b = None\n        return (add,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            sin = l_x_.sin()\n            cos = l_x_.cos();  l_x_ = None\n            return (sin, cos)\n')

    def test_output_with_dict(self):
        if False:
            return 10

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(lambda x: [{'a': -x}], x)
        x = torch.randn(3)
        counters.clear()
        graph = self._test_wrap_simple(f, default_args_generator((x,)), 2, 2, return_graph=True)
        self.assertEqual(len(counters['graph_break']), 0)
        if check_dynamic_shape_capture():
            return
        self.assertExpectedInline(graph, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        wrap_body_0 = self.wrap_body_0\n        wrap = torch._higher_order_ops.wrap.wrap(wrap_body_0, l_x_);  wrap_body_0 = l_x_ = None\n        getitem = wrap[0];  wrap = None\n        return (getitem,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            neg = -l_x_;  l_x_ = None\n            return (neg,)\n')

    def test_access_module_attr(self):
        if False:
            i = 10
            return i + 15
        counters.clear()
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        mod = torch.nn.Linear(3, 3)
        x = torch.randn(3, 3)

        @torch.compile(backend=cnt, fullgraph=True)
        def f(x):
            if False:
                print('Hello World!')
            y = mod(x)
            return wrap(lambda y: y - mod.bias, y)
        result = f(x)
        self.assertEqual(result, mod(x) - mod.bias)
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(backend.graphs), 1)
        wrap_node = find_first_node(backend.graphs[0], wrap)
        self.assertTrue(len(wrap_node.args), 3)
        self.assertTrue(len(dict(backend.graphs[0].named_parameters())) == 2)
        body_function = getattr(backend.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 1)
        self.assertTrue(len(dict(body_function.named_parameters())) == 0)
        self.assertTrue(len(dict(body_function.named_children())) == 0)

    def test_make_closure(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                i = 10
                return i + 15

            def g(x):
                if False:
                    return 10
                return x + y
            return g(x)

        def h(x, y):
            if False:
                while True:
                    i = 10
            return wrap(f, x, y)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(h, default_args_generator((x, y)), 3)

    def test_internal_nonlocal(self):
        if False:
            return 10

        def f(x, y):
            if False:
                print('Hello World!')
            w = 1

            def g(x):
                if False:
                    for i in range(10):
                        print('nop')
                nonlocal w
                w = x
                return x

            def h(x):
                if False:
                    i = 10
                    return i + 15
                nonlocal w
                w = w + 1
                return x
            g(x)
            h(x)
            return w + y

        def h(x, y):
            if False:
                return 10
            return wrap(f, x, y)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        self._test_wrap_simple(h, default_args_generator((x, y)), 3)

    def test_capture_numpy_number(self):
        if False:
            return 10
        import numpy as np
        y = np.float32(1.0)

        def f(x):
            if False:
                while True:
                    i = 10
            return wrap(lambda x: x + y, x)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_freevars_as_inputs_to_wrap(self):
        if False:
            return 10
        y = torch.randn(3)

        def f(x):
            if False:
                return 10
            return wrap(lambda x, y: x + y, x, y)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3)

    def test_lift_tensor_constant(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            y = torch.tensor(1.0)
            return wrap(lambda x: x + y, x)
        x = torch.randn(3)
        self._test_wrap_simple(f, default_args_generator((x,)), 3, expected_opcount=3)

    def test_nested_wrap(self):
        if False:
            print('Hello World!')

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    for i in range(10):
                        print('nop')
                return self.linear(x)
        mod = MockModule()

        def gn(x):
            if False:
                print('Hello World!')
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return wrap(gn, x)
        self._test_wrap_simple(fn, default_args_generator((torch.randn(10, 10),)), 4)

    def test_fn_with_kwargs_in_torch_ops(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                while True:
                    i = 10
            return wrap(lambda z: torch.cos(input=z), x)
        x = torch.randn(3)
        self._test_wrap_simple(fn, default_args_generator((x,)), 2)

    def test_hooks(self):
        if False:
            print('Hello World!')

        class ToyModel(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.net = torch.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    i = 10
                    return i + 15
                return self.net(x)
        model = ToyModel()
        forward_handles = {}
        activations = dict()

        def save_activations(mod, inp, out):
            if False:
                return 10
            activations[name] = inp
        for (name, module) in model.named_children():
            forward_handles[name] = module.register_forward_hook(save_activations)

        @torch.compile(backend='eager')
        def fn(x):
            if False:
                print('Hello World!')
            return wrap(lambda x: model(x), x)
        for i in range(2):
            activations.clear()
            x = torch.randn((10, 10))
            pred = fn(x)
            loss = pred.sum()
            loss.backward()
        self.assertTrue(activations.keys() == forward_handles.keys())

    def _get_source_fn_stack(self, gm, node_names):
        if False:
            return 10
        ret = {}
        for mod in gm.modules():
            for node in mod.graph.nodes:
                if node.name in node_names:
                    actual_stack = [name for (name, _) in node.meta.get('source_fn_stack', [])]
                    ret[node.name] = actual_stack
        return ret

    def test_wrap_source_fn_stack(self):
        if False:
            while True:
                i = 10

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    print('Hello World!')
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                if False:
                    while True:
                        i = 10
                return self.linear(x)
        mod = MockModule()

        def gn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.cos(x) + wrap(mod, x)

        def fn(x):
            if False:
                print('Hello World!')
            return wrap(gn, x)
        backend = EagerAndRecordGraphs()
        inp = torch.randn((4, 4))
        torch.compile(fn, backend=backend, fullgraph=True)(inp)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {'cos', 'add', 'linear'})
        self.assertExpectedInline(pprint.pformat(actual_stack), "{'add': ['wrap', 'add'],\n 'cos': ['wrap', 'cos'],\n 'linear': ['wrap', 'wrap', 'linear']}")

    def test_cond_source_fn_stack(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()

        @torch.compile(backend=backend, fullgraph=True)
        def cond_f(pred, pred2, x, y):
            if False:
                return 10

            def true_fn(pred2, x, y):
                if False:
                    print('Hello World!')
                return x + y

            def false_fn(pred2, x, y):
                if False:
                    return 10

                def true_fn2(x, y):
                    if False:
                        print('Hello World!')
                    return x.sin() - y.cos()

                def false_fn2(x, y):
                    if False:
                        return 10
                    return x.cos() - y.sin()
                return control_flow.cond(pred2, true_fn2, false_fn2, [x, y])
            return control_flow.cond(pred, true_fn, false_fn, [pred2, x, y])
        pred = torch.tensor(True)
        pred2 = torch.tensor(False)
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3, 3)
        cond_f(pred, pred2, xs, y)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {'cos', 'add', 'sin', 'sub'})
        self.assertExpectedInline(pprint.pformat(actual_stack), "{'add': ['cond', 'add'],\n 'cos': ['cond', 'cond', 'cos'],\n 'sin': ['cond', 'cond', 'sin'],\n 'sub': ['cond', 'cond', 'sub']}")

    def test_map_source_fn_stack(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()
        xs = torch.randn(2, 3, 3)
        y = torch.randn(3)

        @torch.compile(backend=backend, fullgraph=True)
        def map_f(xs, y):
            if False:
                print('Hello World!')

            def inner(x, y):
                if False:
                    for i in range(10):
                        print('nop')

                def inner2(x, y):
                    if False:
                        i = 10
                        return i + 15
                    return x + y
                return control_flow.map(inner2, x, y) * y.cos()
            return control_flow.map(inner, xs, y).sin()
        result = map_f(xs, y)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {'cos', 'add', 'sin'})
        self.assertExpectedInline(pprint.pformat(actual_stack), "{'add': ['map', 'map', 'add'], 'cos': ['map', 'cos'], 'sin': ['sin']}")

    def test_grad_source_fn_stack(self):
        if False:
            while True:
                i = 10
        backend = EagerAndRecordGraphs()

        def fn(x):
            if False:
                return 10
            return x.sin().sum()

        @torch.compile(backend=backend, fullgraph=False)
        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.grad(torch.func.grad(fn))(x)
        x = torch.randn(())
        wrapper_fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {'sum_1', 'sin'})
        self.assertExpectedInline(pprint.pformat(actual_stack), "{'sin': ['grad_impl', 'grad_impl', 'sin'],\n 'sum_1': ['grad_impl', 'grad_impl', 'sum_1']}")

    def test_vmap_source_fn_stack(self):
        if False:
            print('Hello World!')
        backend = EagerAndRecordGraphs()

        def inner_fn(x):
            if False:
                while True:
                    i = 10
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)

        @torch.compile(backend=backend, fullgraph=True)
        def fn(x):
            if False:
                return 10
            return torch.func.vmap(lambda x: inner_fn(x.cos()))(x)
        x = torch.randn(3, 3, 3, 3)
        fn(x)
        gm = backend.graphs[0]
        actual_stack = self._get_source_fn_stack(gm, {'sum_1', 'sum_2', 'add'})
        self.assertExpectedInline(pprint.pformat(actual_stack), "{'add': ['vmap_impl', 'vmap_impl', 'add'],\n 'sum_1': ['vmap_impl', 'vmap_impl', 'sum_1'],\n 'sum_2': ['vmap_impl', 'vmap_impl', 'sum_2']}")

    def test_cond_pytree_operands(self):
        if False:
            print('Hello World!')

        def _construct_pytree():
            if False:
                print('Hello World!')
            a = torch.randn(3, 3)
            b = torch.randn(3, 3)
            c = torch.randn(3, 3)
            d = torch.randn(3, 3)
            e = torch.randn(3, 3)
            f = torch.randn(3, 3)
            g = torch.randn(3, 3)
            return (a, [[[b]]], c, (d, (e,), f), {'g': g})
        pred = torch.tensor(True)
        inp = _construct_pytree()

        def _reduce_sum(flattened):
            if False:
                print('Hello World!')
            init = 0
            for val in flattened:
                init += val
            return init

        def _reduce_max(flattened):
            if False:
                print('Hello World!')
            init = flattened[0]
            for val in flattened:
                init = max(val, init)
            return init

        def true_fn(pytree_in):
            if False:
                print('Hello World!')
            (flattened, spec) = pytree.tree_flatten(pytree_in)
            return _reduce_sum(flattened)

        def false_fn(pytree_in):
            if False:
                while True:
                    i = 10
            (flattened, spec) = pytree.tree_flatten(pytree_in)
            return _reduce_max(flattened)

        def fn(pred, pytree_in):
            if False:
                return 10
            return torch.cond(pred, true_fn, false_fn, [pytree_in])
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        compiled_res = torch.compile(fn, backend=backend)(pred, inp)
        eager_res = fn(pred, inp)
        self.assertEqual(compiled_res, eager_res)
        graph = backend.graphs[0]
        if check_dynamic_shape_capture():
            return
        self.assertExpectedInline(graph.code.strip(), 'def forward(self, L_pred_ : torch.Tensor, L_pytree_in_0_ : torch.Tensor, L_pytree_in_1_0_0_0_ : torch.Tensor, L_pytree_in_2_ : torch.Tensor, L_pytree_in_3_0_ : torch.Tensor, L_pytree_in_3_1_0_ : torch.Tensor, L_pytree_in_3_2_ : torch.Tensor, L_pytree_in_4_g_ : torch.Tensor):\n    l_pred_ = L_pred_\n    l_pytree_in_0_ = L_pytree_in_0_\n    l_pytree_in_1_0_0_0_ = L_pytree_in_1_0_0_0_\n    l_pytree_in_2_ = L_pytree_in_2_\n    l_pytree_in_3_0_ = L_pytree_in_3_0_\n    l_pytree_in_3_1_0_ = L_pytree_in_3_1_0_\n    l_pytree_in_3_2_ = L_pytree_in_3_2_\n    l_pytree_in_4_g_ = L_pytree_in_4_g_\n    cond_true_0 = self.cond_true_0\n    cond_false_0 = self.cond_false_0\n    cond = torch.ops.higher_order.cond(l_pred_, cond_true_0, cond_false_0, [l_pytree_in_0_, l_pytree_in_1_0_0_0_, l_pytree_in_2_, l_pytree_in_3_0_, l_pytree_in_3_1_0_, l_pytree_in_3_2_, l_pytree_in_4_g_]);  l_pred_ = cond_true_0 = cond_false_0 = l_pytree_in_0_ = l_pytree_in_1_0_0_0_ = l_pytree_in_2_ = l_pytree_in_3_0_ = l_pytree_in_3_1_0_ = l_pytree_in_3_2_ = l_pytree_in_4_g_ = None\n    return (cond,)')

    def test_cond_pytree_operands_with_non_tensor_leaves(self):
        if False:
            while True:
                i = 10

        def fn(pred, pytree_in):
            if False:
                return 10
            return torch.cond(pred, lambda x: x[0] + 1, lambda x: x[0] * 2, (pytree_in,))
        pred = torch.tensor(True)
        for pytree_in in [(1,), ('string',), (1.0,)]:
            with self.assertRaisesRegex(RuntimeError, 'Expect operands to be a tuple of possibly nested dict/list/tuple'):
                fn(pred, pytree_in)
        for pytree_in in [(1,), ('string',), (1.0,)]:
            with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
                torch.compile(fn, backend='eager')(pred, pytree_in)

class FuncTorchHigherOrderOpTests(torch._dynamo.test_case.TestCase):

    def run(self, result=None):
        if False:
            return 10
        with config.patch(capture_func_transforms=True):
            super().run(result)

    def _compile_check(self, fn, inputs, fullgraph=True, graph_idx=0):
        if False:
            for i in range(10):
                print('nop')
        backend = EagerAndRecordGraphs()
        actual = fn(*inputs)
        expected = torch.compile(fn, backend=backend, fullgraph=fullgraph)(*inputs)
        self.assertEqual(actual, expected)
        wrapped_gm = backend.graphs[graph_idx]
        return wrapped_gm

    def test_grad(self):
        if False:
            return 10
        counters.clear()

        def fn(x):
            if False:
                return 10
            return x.sin().sum()

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None\n        contiguous = call.contiguous();  call = None\n        return (contiguous,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            sin = l_x_.sin();  l_x_ = None\n            sum_1 = sin.sum();  sin = None\n            return sum_1\n')

    def test_grad_freevar_tensor(self):
        if False:
            return 10
        counters.clear()
        y = torch.randn(3, 3)

        def fn(x):
            if False:
                i = 10
                return i + 15
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        expected = wrapper_fn(x)
        actual = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=True)(x)
        self.assertEqual(actual, expected)

    def test_grad_freevar_python_scalar(self):
        if False:
            return 10
        counters.clear()
        y = 3

        def fn(x):
            if False:
                return 10
            return (x.sin() + y).sum()

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None\n        contiguous = call.contiguous();  call = None\n        return (contiguous,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            sin = l_x_.sin();  l_x_ = None\n            add = sin + 3;  sin = None\n            sum_1 = add.sum();  add = None\n            return sum_1\n')

    def test_grad_capture_tensor(self):
        if False:
            i = 10
            return i + 15
        counters.clear()

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            y = torch.randn(3)

            def fn(x):
                if False:
                    print('Hello World!')
                return (x.sin() + y).sum()
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False, graph_idx=1)
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        l_x_ = L_x_\n        l_y_ = L_y_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None\n        contiguous = call.contiguous();  call = None\n        return (contiguous,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_, l_y_):\n            sin = l_x_.sin();  l_x_ = None\n            add = sin + l_y_;  sin = l_y_ = None\n            sum_1 = add.sum();  add = None\n            return sum_1\n')

    def test_grad_closure_scalar(self):
        if False:
            while True:
                i = 10
        counters.clear()

        def wrapper_fn(x):
            if False:
                return 10
            y = 3.14

            def fn(x):
                if False:
                    print('Hello World!')
                return (x.sin() + y).sum()
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None\n        contiguous = call.contiguous();  call = None\n        return (contiguous,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            sin = l_x_.sin();  l_x_ = None\n            add = sin + 3.14;  sin = None\n            sum_1 = add.sum();  add = None\n            return sum_1\n')

    def test_grad_has_aux(self):
        if False:
            for i in range(10):
                print('nop')
        counters.clear()
        y = 3.14

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.grad(fn, has_aux=True)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, True);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None\n        getitem = call[0]\n        getitem_1 = call[1];  call = None\n        contiguous = getitem.contiguous();  getitem = None\n        return (contiguous, getitem_1)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            sin = l_x_.sin()\n            add = sin + 3.14;  sin = None\n            sum_1 = add.sum();  add = None\n            cos = l_x_.cos();  l_x_ = None\n            return (sum_1, cos)\n')

    def test_grad_two_tensor_has_aux(self):
        if False:
            while True:
                i = 10
        counters.clear()

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn(x, y):
            if False:
                while True:
                    i = 10
            return torch.func.grad(fn, has_aux=True)(x, y)
        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        l_x_ = L_x_\n        l_y_ = L_y_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, True);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None\n        getitem = call[0]\n        getitem_1 = call[1];  call = None\n        contiguous = getitem.contiguous();  getitem = None\n        return (contiguous, getitem_1)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_, l_y_):\n            sin = l_x_.sin()\n            add = sin + l_y_;  sin = l_y_ = None\n            sum_1 = add.sum();  add = None\n            cos = l_x_.cos();  l_x_ = None\n            return (sum_1, cos)\n')

    def test_grad_two_tensor_all_grad_has_aux(self):
        if False:
            return 10
        counters.clear()
        nums = (0, 1)

        def fn(x, y):
            if False:
                print('Hello World!')
            return ((x.sin() + y).sum(), x.cos())

        def wrapper_fn_const_var(x, y):
            if False:
                i = 10
                return i + 15
            return torch.func.grad(fn, argnums=(0, 1), has_aux=True)(x, y)

        def wrapper_fn_tuple_var(x, y):
            if False:
                return 10
            return torch.func.grad(fn, argnums=nums, has_aux=True)(x, y)
        y = torch.randn(3, 3, 3)
        x = torch.randn(3, 3, 3)
        wrapped_gm_const_var = self._compile_check(wrapper_fn_const_var, (x, y))
        wrapped_gm_tuple_var = self._compile_check(wrapper_fn_tuple_var, (x, y))
        if check_dynamic_shape_capture():
            return
        actual_const_var = normalize_gm(wrapped_gm_const_var.print_readable(print_output=False))
        actual_tuple_var = normalize_gm(wrapped_gm_tuple_var.print_readable(print_output=False))
        self.assertExpectedInline(actual_const_var, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        l_x_ = L_x_\n        l_y_ = L_y_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, (0, 1), True);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None\n        getitem = call[0]\n        getitem_1 = getitem[0]\n        getitem_2 = getitem[1];  getitem = None\n        getitem_3 = call[1];  call = None\n        contiguous = getitem_1.contiguous();  getitem_1 = None\n        contiguous_1 = getitem_2.contiguous();  getitem_2 = None\n        return (contiguous, contiguous_1, getitem_3)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_, l_y_):\n            sin = l_x_.sin()\n            add = sin + l_y_;  sin = l_y_ = None\n            sum_1 = add.sum();  add = None\n            cos = l_x_.cos();  l_x_ = None\n            return (sum_1, cos)\n')
        self.assertExpectedInline(actual_tuple_var, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        l_x_ = L_x_\n        l_y_ = L_y_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, (0, 1), True);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_, l_y_);  grad_proxy = l_x_ = l_y_ = None\n        getitem = call[0]\n        getitem_1 = getitem[0]\n        getitem_2 = getitem[1];  getitem = None\n        getitem_3 = call[1];  call = None\n        contiguous = getitem_1.contiguous();  getitem_1 = None\n        contiguous_1 = getitem_2.contiguous();  getitem_2 = None\n        return (contiguous, contiguous_1, getitem_3)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_, l_y_):\n            sin = l_x_.sin()\n            add = sin + l_y_;  sin = l_y_ = None\n            sum_1 = add.sum();  add = None\n            cos = l_x_.cos();  l_x_ = None\n            return (sum_1, cos)\n')

    def test_grad_over_grad(self):
        if False:
            return 10
        counters.clear()

        def fn(x):
            if False:
                return 10
            return x.sin().sum()

        def wrapper_fn(x):
            if False:
                while True:
                    i = 10
            return torch.func.grad(torch.func.grad(fn))(x)
        x = torch.randn(())
        wrapped_gm = self._compile_check(wrapper_fn, (x,), fullgraph=False)
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        grad_body_1 = self.grad_body_1\n        grad_proxy = torch.func.grad(grad_body_1, 0, False);  grad_body_1 = None\n        call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None\n        contiguous = call.contiguous();  call = None\n        return (contiguous,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_):\n            grad_body_0 = self.grad_body_0\n            grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None\n            call = grad_proxy.__call__(l_x_);  grad_proxy = l_x_ = None\n            contiguous = call.contiguous();  call = None\n            return contiguous\n\n        class GraphModule(torch.nn.Module):\n            def forward(self, l_x_):\n                sin = l_x_.sin();  l_x_ = None\n                sum_1 = sin.sum();  sin = None\n                return sum_1\n')

    def test_grad_with_graph_break(self):
        if False:
            for i in range(10):
                print('nop')
        counters.clear()

        def fn(x):
            if False:
                print('Hello World!')
            torch._dynamo.graph_break()
            return x.sin().sum()

        def wrapper_fn(x):
            if False:
                return 10
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x)
        self.assertEqual(len(counters['graph_break']), 1)
        self.assertEqual(actual, expected)

    def test_grad_with_side_effect(self):
        if False:
            while True:
                i = 10
        counters.clear()
        foo = [1, 2]

        def fn(x):
            if False:
                return 10
            foo.append(3)
            return x.sin().sum()

        def wrapper_fn(x):
            if False:
                print('Hello World!')
            return torch.func.grad(fn)(x)
        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x)
        self.assertEqual(len(counters['graph_break']), 1)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {'.*HigherOrderOperator: Mutating a variable not in the current scope \\(replace_all\\)': 2})
        self.assertEqual(actual, expected)

    def test_grad_pytree(self):
        if False:
            print('Hello World!')
        counters.clear()

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            (x1, x2) = x
            return x1.sin().sum() + x2

        def wrapper_fn(x):
            if False:
                print('Hello World!')
            return torch.func.grad(fn)(x)
        x1 = torch.randn(3, 3, 3)
        x2 = torch.randn(())
        actual = wrapper_fn((x1, x2))
        expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)((x1, x2))
        self.assertEqual(len(counters['graph_break']), 1)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {'.*HigherOrderOperator with body that accepts non-Tensors as input': 2})
        self.assertEqual(actual, expected)

    def test_grad_non_tensor_input(self):
        if False:
            i = 10
            return i + 15
        counters.clear()

        def fn(x, y):
            if False:
                print('Hello World!')
            return x.sin().sum() + y

        def wrapper_fn(x, y):
            if False:
                return 10
            return torch.func.grad(fn)(x, y)
        x = torch.randn(3, 3, 3)
        y = 3.0
        wrapped_gm = self._compile_check(wrapper_fn, (x, y))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, 'class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        l_x_ = L_x_\n\n        grad_body_0 = self.grad_body_0\n        grad_proxy = torch.func.grad(grad_body_0, 0, False);  grad_body_0 = None\n        call = grad_proxy.__call__(l_x_, 3.0);  grad_proxy = l_x_ = None\n        contiguous = call.contiguous();  call = None\n        return (contiguous,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, l_x_, const):\n            sin = l_x_.sin();  l_x_ = None\n            sum_1 = sin.sum();  sin = None\n            add = sum_1 + 3.0;  sum_1 = None\n            return add\n')

    def test_grad_disable_capture(self):
        if False:
            while True:
                i = 10
        counters.clear()
        with config.patch(capture_func_transforms=False):

            def fn(x):
                if False:
                    print('Hello World!')
                return x.sin().sum()

            def wrapper_fn(x):
                if False:
                    return 10
                return torch.func.grad(fn)(x)
            x = torch.randn(3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x)
            self.assertEqual(len(counters['graph_break']), 1)
            self.assertEqual(dict(counters['graph_break']), {'torch.func.grad capture is disabled, it can be turned on by setting `torch._dynamo.config.capture_func_transforms=True`': 2})
            self.assertEqual(actual, expected)

    def test_grad_fn_with_kwargs(self):
        if False:
            i = 10
            return i + 15

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return (x + y).sum()

        def wrapper_fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.grad(fn)(x, y=y)
        x = torch.randn(3, 3)
        y = torch.randn(3, 3)
        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x, y)
        self.assertEqual(len(counters['graph_break']), 1)
        self.assertEqual(dict(counters['graph_break']), {'torch.func.grad: kwargs arguments are currently unsupported.': 2})
        self.assertEqual(actual, expected)

    def test_vmap(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        child = L_x_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), 0, 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child);  vmap_proxy = child = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            add = sum_1 + sum_2;  sum_1 = sum_2 = None\n            return add\n")

    def test_vmap_free_const(self):
        if False:
            i = 10
            return i + 15
        y = 3

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        child = L_x_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), 0, 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child);  vmap_proxy = child = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            add = sum_1 + sum_2;  sum_1 = sum_2 = None\n            add_1 = add + 3;  add = None\n            return add_1\n")

    def test_vmap_free_tensor(self):
        if False:
            for i in range(10):
                print('nop')
        y = torch.randn(3, 3)

        def fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.vmap(lambda x: x.sum(0) + x.sum(1) + y)(x)
        x = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        child = L_x_\n        l_y_ = L_y_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0, None), 0, 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child, l_y_);  vmap_proxy = child = l_y_ = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select, l_y_):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            add = sum_1 + sum_2;  sum_1 = sum_2 = None\n            add_1 = add + l_y_;  add = l_y_ = None\n            return add_1\n")

    def test_vmap_two_inputs(self):
        if False:
            while True:
                i = 10

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return torch.func.vmap(lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=(0, 1))(x, y)
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        child = L_x_\n        child_1 = L_y_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        select_1 = child_1.select(1, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0, 1), 0, 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child, child_1);  vmap_proxy = child = child_1 = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select, select_1):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            add = sum_1 + sum_2;  sum_1 = sum_2 = None\n            add_1 = add + select_1;  add = select_1 = None\n            return add_1\n")

    def test_vmap_two_inputs_tuple_in_dims(self):
        if False:
            print('Hello World!')
        in_dims = (0, 1)

        def fn(x, y):
            if False:
                while True:
                    i = 10
            return torch.func.vmap(lambda x, y: x.sum(0) + x.sum(1) + y, in_dims=in_dims)(x, y)
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        child = L_x_\n        child_1 = L_y_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        select_1 = child_1.select(1, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0, 1), 0, 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child, child_1);  vmap_proxy = child = child_1 = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select, select_1):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            add = sum_1 + sum_2;  sum_1 = sum_2 = None\n            add_1 = add + select_1;  add = select_1 = None\n            return add_1\n")

    def test_vmap_over_vmap_two_inputs(self):
        if False:
            print('Hello World!')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return torch.func.vmap(torch.func.vmap(lambda x, y: x + y, in_dims=1))(x, y)
        x = torch.randn(3, 3, 3)
        y = torch.randn(3, 3, 3)
        wrapped_gm = self._compile_check(fn, (x, y))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor, L_y_ : torch.Tensor):\n        child = L_x_\n        child_1 = L_y_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n        _check_randomness_arg_1 = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        select_1 = child_1.select(0, 0)\n        vmap_body_1 = self.vmap_body_1\n        vmap_proxy = torch.func.vmap(vmap_body_1, (0, 0), 0, 'error');  vmap_body_1 = None\n        call = vmap_proxy.__call__(child, child_1);  vmap_proxy = child = child_1 = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select, select_1):\n            select_2 = select.select(1, 0)\n            select_3 = select_1.select(1, 0)\n            vmap_body_0 = self.vmap_body_0\n            vmap_proxy = torch.func.vmap(vmap_body_0, (1, 1), 0, 'error');  vmap_body_0 = None\n            call = vmap_proxy.__call__(select, select_1);  vmap_proxy = select = select_1 = None\n            return call\n\n        class GraphModule(torch.nn.Module):\n            def forward(self, select_2, select_3):\n                add = select_2 + select_3;  select_2 = select_3 = None\n                return add\n")

    def test_vmap_over_vmap_captured(self):
        if False:
            return 10
        x = torch.ones(2, 3)
        y = torch.ones(5, 3)

        def fn(x):
            if False:
                return 10
            return torch.func.vmap(torch.func.vmap(lambda y: x * y))(y)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_y_ : torch.Tensor, L_x_ : torch.Tensor):\n        child = L_y_\n        l_x_ = L_x_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n        _check_randomness_arg_1 = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_1 = self.vmap_body_1\n        vmap_proxy = torch.func.vmap(vmap_body_1, (0, None), 0, 'error');  vmap_body_1 = None\n        call = vmap_proxy.__call__(child, l_x_);  vmap_proxy = child = l_x_ = None\n        return (call,)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select, l_x_):\n            select_1 = select.select(0, 0)\n            vmap_body_0 = self.vmap_body_0\n            vmap_proxy = torch.func.vmap(vmap_body_0, (0, None), 0, 'error');  vmap_body_0 = None\n            call = vmap_proxy.__call__(select, l_x_);  vmap_proxy = select = l_x_ = None\n            return call\n\n        class GraphModule(torch.nn.Module):\n            def forward(self, select_1, l_x_):\n                mul = l_x_ * select_1;  l_x_ = select_1 = None\n                return mul\n")

    def test_vmap_multiple_outputs(self):
        if False:
            i = 10
            return i + 15
        x = torch.ones(2, 4, 3)

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)))(x)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        child = L_x_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), 0, 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child);  vmap_proxy = child = None\n        getitem = call[0]\n        getitem_1 = call[1];  call = None\n        return (getitem, getitem_1)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            return (sum_1, sum_2)\n")

    def test_vmap_multiple_outputs_diff_dims(self):
        if False:
            while True:
                i = 10
        x = torch.ones(2, 4, 3)

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=(1, 0))(x)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        child = L_x_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), (1, 0), 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child);  vmap_proxy = child = None\n        getitem = call[0]\n        getitem_1 = call[1];  call = None\n        return (getitem, getitem_1)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            return (sum_1, sum_2)\n")

    def test_vmap_multiple_outputs_out_dims_tuple(self):
        if False:
            while True:
                i = 10
        x = torch.ones(2, 4, 3)
        out_dims = (1, 0)

        def fn(x):
            if False:
                return 10
            return torch.vmap(lambda x: (x.sum(0), x.sum(1)), out_dims=out_dims)(x)
        wrapped_gm = self._compile_check(fn, (x,))
        if check_dynamic_shape_capture():
            return
        actual = normalize_gm(wrapped_gm.print_readable(print_output=False))
        self.assertExpectedInline(actual, "class GraphModule(torch.nn.Module):\n    def forward(self, L_x_ : torch.Tensor):\n        child = L_x_\n\n        _check_randomness_arg = torch._functorch.vmap._check_randomness_arg('error')\n\n        select = child.select(0, 0)\n        vmap_body_0 = self.vmap_body_0\n        vmap_proxy = torch.func.vmap(vmap_body_0, (0,), (1, 0), 'error');  vmap_body_0 = None\n        call = vmap_proxy.__call__(child);  vmap_proxy = child = None\n        getitem = call[0]\n        getitem_1 = call[1];  call = None\n        return (getitem, getitem_1)\n\n    class GraphModule(torch.nn.Module):\n        def forward(self, select):\n            sum_1 = select.sum(0)\n            sum_2 = select.sum(1);  select = None\n            return (sum_1, sum_2)\n")

    def test_vmap_kwargs(self):
        if False:
            while True:
                i = 10
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def fn(x, y):
            if False:
                print('Hello World!')
            return torch.func.vmap(lambda x, y: x + y)(x, y=y)
        actual = fn(x, y)
        expected = torch.compile(fn, backend='aot_eager', fullgraph=False)(x, y)
        self.assertEqual(len(counters['graph_break']), 1)
        self.assertEqual(dict(counters['graph_break']), {'NYI - torch.func.vmap: kwargs arguments are currently unsupported.': 2})
        self.assertEqual(actual, expected)

    def test_vmap_pytree_inputs(self):
        if False:
            return 10
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)

        def vmap_fn(inps):
            if False:
                for i in range(10):
                    print('nop')
            x = inps['x']
            y = inps['y']
            return x + y

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return torch.func.vmap(vmap_fn)({'x': x, 'y': y})
        actual = fn(x, y)
        expected = torch.compile(fn, backend='aot_eager', fullgraph=False)(x, y)
        self.assertEqual(len(counters['graph_break']), 2)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {'.*HigherOrderOperator with body that accepts non-Tensors as input': 2, 'Unsupported: meta converter nyi with fake tensor propagation.': 1})
        self.assertEqual(actual, expected)

    def test_vmap_side_effects(self):
        if False:
            while True:
                i = 10
        counters.clear()
        x = torch.ones(2, 3)
        y = torch.randn(2, 3)
        some_list = []

        def f(x, y):
            if False:
                print('Hello World!')
            some_list.append(1)
            return x + y

        def wrapper_fn(x, y):
            if False:
                print('Hello World!')
            return torch.func.vmap(f)(x, y)
        actual = wrapper_fn(x, y)
        expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x, y)
        self.assertEqual(len(counters['graph_break']), 1)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {'.*HigherOrderOperator: Mutating a variable not in the current scope \\(replace_all\\)': 2})
        self.assertEqual(actual, expected)

    def test_vmap_disable_capture(self):
        if False:
            while True:
                i = 10
        counters.clear()
        with config.patch(capture_func_transforms=False):

            def wrapper_fn(x):
                if False:
                    return 10
                return torch.func.vmap(lambda x: x.sum(0) + x.sum(1))(x)
            x = torch.randn(3, 3, 3)
            actual = wrapper_fn(x)
            expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x)
            self.assertEqual(len(counters['graph_break']), 1)
            self.assertEqual(dict(counters['graph_break']), {'torch.func.vmap capture is disabled, it can be turned on by setting `torch._dynamo.config.capture_func_transforms=True`': 2})
            self.assertEqual(actual, expected)

    def test_vmap_illegal_op_graph_break(self):
        if False:
            while True:
                i = 10
        counters.clear()

        def bad_fn(x):
            if False:
                return 10
            x.stride()
            return x

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.vmap(bad_fn)(x)
        x = torch.randn(3, 3, 3)
        actual = wrapper_fn(x)
        expected = torch.compile(wrapper_fn, backend='aot_eager', fullgraph=False)(x)
        self.assertEqual(len(counters['graph_break']), 1)
        assert_dict_matches_regex(self, dict(counters['graph_break']), {'.*Illegal getattr invocation stride in strict mode': 2})
        self.assertEqual(actual, expected)

    def test_vmap_multiple_invocation_in_dims(self):
        if False:
            while True:
                i = 10
        counters.clear()

        def wrapper_fn(x, in_dims):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.vmap(torch.sum, in_dims)(x)
        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = (wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2))
        actual = (opt(x, 0), opt(x, 1), opt(x, 2))
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 9)

    def test_vmap_multiple_invocation_out_dims(self):
        if False:
            return 10
        counters.clear()

        def wrapper_fn(x, out_dims):
            if False:
                while True:
                    i = 10
            return torch.func.vmap(lambda x: torch.sum(x, 0), out_dims=out_dims)(x)
        x = torch.randn(3, 3, 3, 3)
        cnt = CompileCounter()
        opt = torch.compile(wrapper_fn, backend=cnt, fullgraph=False, dynamic=True)
        expected = (wrapper_fn(x, 0), wrapper_fn(x, 1), wrapper_fn(x, 2))
        actual = (opt(x, 0), opt(x, 1), opt(x, 2))
        self.assertEqual(expected, actual)
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 9)

    def test_vmap_new_tensor_in_body(self):
        if False:
            print('Hello World!')

        def fn(x):
            if False:
                while True:
                    i = 10
            return x + torch.ones(3)

        def wrapper_fn(x):
            if False:
                i = 10
                return i + 15
            return torch.func.vmap(fn)(x)
        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend='eager', fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_unused_in_body(self):
        if False:
            while True:
                i = 10

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.tensor(0.5)

        def wrapper_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return torch.func.vmap(fn)(x)
        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend='eager', fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

    def test_vmap_new_tensor_implicit_via_op(self):
        if False:
            return 10

        def wrapper_fn(x):
            if False:
                print('Hello World!')
            return torch.func.vmap(lambda t: torch.add(t, 0.5))(x)
        x = torch.randn(3)
        opt = torch.compile(wrapper_fn, backend='eager', fullgraph=True)
        expected = wrapper_fn(x)
        actual = opt(x)
        self.assertEqual(expected, actual)

class ActivationCheckpointingTests(torch._dynamo.test_case.TestCase):

    def _validate(self, fn, backend, *args, skip_check=False, fullgraph=True):
        if False:
            i = 10
            return i + 15
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.clone().detach().requires_grad_(arg.requires_grad))
        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()
        opt_fn = torch.compile(fn, fullgraph=fullgraph, backend=backend)
        torch.manual_seed(0)
        result = opt_fn(*cloned_args)
        result.sum().backward()
        if not skip_check:
            self.assertEqual(result, expected)
            for (arg, cloned_arg) in zip(args, cloned_args):
                self.assertEqual(arg.grad, cloned_arg.grad)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function(self):
        if False:
            print('Hello World!')

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=3, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_function_with_kwargs(self):
        if False:
            for i in range(10):
                print('nop')

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                return 10
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False)
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(count_ops, freq=3, op=torch.ops.aten.mm.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout(self):
        if False:
            for i in range(10):
                print('nop')

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default)
        bw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.rngprims.philox_rand.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x, y, skip_check=True)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_dropout_inductor(self):
        if False:
            while True:
                i = 10

        def gn(x, y):
            if False:
                while True:
                    i = 10
            return torch.nn.functional.dropout(torch.matmul(x, y), p=0.2)

        def fn(x, y):
            if False:
                return 10
            return torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y)
        x = torch.randn(4, 4, device='cuda', requires_grad=True)
        y = torch.randn(4, 4, device='cuda', requires_grad=True)
        backend = 'inductor'
        self._validate(fn, backend, x, y, skip_check=True)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_fallback(self):
        if False:
            for i in range(10):
                print('nop')

        def gn(x, y):
            if False:
                print('Hello World!')
            torch._dynamo.graph_break()
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return torch.cos(torch.utils.checkpoint.checkpoint(gn, torch.sin(x), y))
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)
        args = (x, y)
        backend = EagerAndRecordGraphs()
        cnt = CompileCounterWithBackend(backend)
        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)
        self.assertEqual(result, expected)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(backend.graphs), 2)

    @requires_cuda()
    @torch._functorch.config.patch(functionalize_rng_ops=True)
    def test_module(self):
        if False:
            for i in range(10):
                print('nop')

        class MockModule(torch.nn.Module):

            def __init__(self):
                if False:
                    i = 10
                    return i + 15
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                if False:
                    return 10
                return torch.sigmoid(self.linear(x))
        mod = MockModule()

        def fn(x):
            if False:
                i = 10
                return i + 15
            return torch.utils.checkpoint.checkpoint(mod, torch.sin(x))
        x = torch.randn(10, 10, requires_grad=True)
        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.sigmoid.default)
        bw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.sigmoid.default)
        backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        self._validate(fn, backend, x)

    def test_override_fallthrough_dispatch_key(self):
        if False:
            for i in range(10):
                print('nop')
        test_op = torch._ops.HigherOrderOperator('_fallthrough_test_only')
        default_keys = torch._ops._HIGHER_ORDER_OP_DEFAULT_FALLTHROUGH_DISPATCH_KEYS
        self.assertTrue(not any((test_op.non_fallthrough_keys.has(key) for key in default_keys)))
        foos = [lambda x=i: x for (i, k) in enumerate(default_keys)]
        for (foo, fallthrough_key) in zip(foos, default_keys):
            test_op.py_impl(fallthrough_key)(foo)
        self.assertTrue(all((test_op.non_fallthrough_keys.has(key) for key in default_keys)))
        self.assertEqual(list(range(len(default_keys))), [test_op.py_kernels[key]() for key in default_keys])

    def test_cond_with_kwargs(self):
        if False:
            while True:
                i = 10
        from torch._higher_order_ops.cond import cond_op

        def test(pred, x):
            if False:
                while True:
                    i = 10

            def true_fn(x):
                if False:
                    i = 10
                    return i + 15
                return x

            def false_fn(x):
                if False:
                    print('Hello World!')
                return -x
            return cond_op(pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x])
        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt)
        inp = torch.ones(3, 3)
        self.assertTrue(torch.allclose(test(True, inp), opt_test(True, inp)))
        self.assertEqual(cnt.frame_count, 1)
        self.assertTrue(torch.allclose(test(False, inp), opt_test(False, inp)))
        self.assertEqual(cnt.frame_count, 2)

    def test_cond_with_invalid_kwargs(self):
        if False:
            i = 10
            return i + 15
        from torch._higher_order_ops.cond import cond_op

        def test(pred, mode, x):
            if False:
                while True:
                    i = 10

            def true_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x

            def false_fn(x):
                if False:
                    return 10
                return -x
            if mode:
                return cond_op(pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x], invalid=True)
            else:
                return cond_op(pred, pred=pred, true_fn=true_fn, false_fn=false_fn, operands=[x])
        cnt = CompileCounter()
        opt_test = torch.compile(test, backend=cnt)
        inp = torch.ones(3, 3)
        with self.assertRaises(torch._dynamo.exc.UncapturedHigherOrderOpError):
            opt_test(True, True, inp)
        with self.assertRaises(AssertionError):
            opt_test(True, False, inp)
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    run_tests()