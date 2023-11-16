import functools
import unittest
from torch.testing._internal.common_utils import TEST_WITH_TORCHDYNAMO, parametrize, instantiate_parametrized_tests
import torch
import torch.utils._pytree as pytree
from functorch.experimental import control_flow
from functorch.experimental.control_flow import UnsupportedAliasMutationException, cond
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import run_tests, TestCase
from torch._dynamo.exc import CondOpArgsMismatchError
from torch.testing._internal.common_quantization import skipIfNoDynamoSupport
from torch._subclasses.functional_tensor import FunctionalTensor

def to_fun(t):
    if False:
        i = 10
        return i + 15
    if isinstance(t, torch.Tensor):
        return FunctionalTensor.to_functional(t)
    return t

def from_fun(t):
    if False:
        return 10
    if not isinstance(t, FunctionalTensor):
        if isinstance(t, torch.Tensor):
            assert not torch._is_functional_tensor(t)
        return t
    torch._sync(t)
    return torch._from_functional_tensor(t.elem)

def to_fun_old(t):
    if False:
        return 10
    if isinstance(t, torch.Tensor) and (not torch._is_functional_tensor(t)):
        out = torch._to_functional_tensor(t)
        torch._mirror_autograd_meta_to(t, out)
        return out
    return t

def from_fun_old(t):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(t, torch.Tensor):
        assert torch._is_functional_tensor(t)
        torch._sync(t)
        return torch._from_functional_tensor(t)
    return t

def _fake_map(f, x, *args):
    if False:
        for i in range(10):
            print('nop')
    from functorch.experimental.control_flow import _stack_pytree, _unstack_pytree
    x_pytrees = _unstack_pytree(x)
    zs = []
    for xp in x_pytrees:
        zs.append(f(xp, *args))
    return _stack_pytree(zs)

def collect_meta_for_filtered_nodes(gm: torch.fx.GraphModule, node_names, meta_field_name):
    if False:
        i = 10
        return i + 15
    ret = []
    for mod in gm.modules():
        for node in mod.graph.nodes:
            if node.name in node_names:
                for field_name in meta_field_name:
                    ret.append(node.meta.get(field_name))
    return ret

def reduce_func(*operands):
    if False:
        i = 10
        return i + 15
    acc = 0
    for operand in operands:
        acc += operand
    return acc

class ReduceObj:

    def __call__(self, *operands):
        if False:
            i = 10
            return i + 15
        return reduce_func(*operands)

class ReduceMod(torch.nn.Module):

    def _reduce(self, *operands):
        if False:
            for i in range(10):
                print('nop')
        return reduce_func(*operands)

    def forward(self, *operands):
        if False:
            for i in range(10):
                print('nop')
        return self._reduce(*operands)

@skipIfNoDynamoSupport
class TestControlFlow(TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        torch._dynamo.reset()
        super().setUp()

    def test_cond_no_trace(self):
        if False:
            while True:
                i = 10

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return x.sin()

        def false_fn(x):
            if False:
                return 10
            return x.cos()
        x = torch.randn(4)
        result = cond(False, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    @unittest.skipIf(not torch.cuda.is_available(), 'Test requires CUDA.')
    def test_cond_gpu(self):
        if False:
            print('Hello World!')

        def true_fn(x):
            if False:
                print('Hello World!')
            return x.sin()

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.cos()
        x = torch.randn(4, device='cuda')
        pred = torch.tensor(False, device='cuda')
        result = cond(pred, true_fn, false_fn, [x])
        self.assertEqual(result, torch.cos(x))

    @unittest.skipIf(not torch.cuda.is_available(), 'Test requires CUDA.')
    def test_map_gpu(self):
        if False:
            return 10

        def f(x, y):
            if False:
                print('Hello World!')
            return x + y
        xs = torch.ones(3, 2, 2, device='cuda')
        y = torch.ones(2, device='cuda')
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(expected, res)

    def test_map_illegal_inputs(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                print('Hello World!')
            return x[0] + x[1] + y
        with self.assertRaisesRegex(RuntimeError, 'Mapped xs can only consist of tensors\\. Got xs \\[3, tensor\\(\\[1\\., 1\\.\\]\\)\\]\\.'):
            _ = control_flow.map(f, (3, torch.ones(2)), torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, 'Leading dimensions of mapped xs cannot be 0\\.'):
            _ = control_flow.map(f, (torch.ones(0, 1, 2), torch.ones(0, 1, 2)), torch.ones(2))
        with self.assertRaisesRegex(RuntimeError, 'Leading dimensions of mapped xs must be consistent\\. Got shapes \\[torch\\.Size\\(\\[3, 4, 5\\]\\), torch\\.Size\\(\\[4, 4, 5\\]\\)\\]\\.'):
            _ = control_flow.map(f, (torch.ones(3, 4, 5), torch.ones(4, 4, 5)), torch.ones(5))

    def test_map_illegal_outputs(self):
        if False:
            return 10

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x.item()

        def f1(x, y):
            if False:
                i = 10
                return i + 15
            return y.size()

        def f2(x, y):
            if False:
                i = 10
                return i + 15
            return None
        x = torch.ones([3])
        y = torch.ones([1, 2, 3])
        with self.assertRaisesRegex(RuntimeError, 'Expect outputs of map only contains tensors or None\\.'):
            _ = control_flow.map(f, x, y)
        with self.assertRaisesRegex(RuntimeError, 'Expect outputs of map only contains tensors or None\\.'):
            out = control_flow.map(f1, x, y)
        _ = control_flow.map(f2, x, y)

    def test_map_list_in_out(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return [[x[0][0] + y]]
        xs = [[torch.ones(3, 2, 2)]]
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertEqual(len(res[0]), 1)
        self.assertEqual(expected, res)

    def test_map_dict_in_out(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                return 10
            return {'c': x['a']['b'] + y}
        xs = {'a': {'b': torch.ones(3, 2, 2)}}
        y = torch.ones(2)
        res = control_flow.map(f, xs, y)
        expected = _fake_map(f, xs, y)
        self.assertEqual(len(res), 1)
        self.assertTrue('c' in res)
        self.assertEqual(expected, res)

    def test_map_autograd_simple(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return x.sin().cos() * y.cos().sin()
        xs = torch.ones(3, 2, 2, requires_grad=True)
        y = torch.ones(2, requires_grad=True)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs, y), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs, y), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_simple_partial_grad(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x.sin().cos() * y.cos().sin()
        xs = torch.ones(3, 2, 2, requires_grad=True)
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res)
        grads = torch.autograd.grad(res, (xs,), grad_out)
        expected_grads = torch.autograd.grad(expected_res, (xs,), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_no_grad_output(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                return 10
            return (x[0].sin().cos() + y, y.cos().sin())
        xs = [torch.ones(3, 2, 2, requires_grad=True), torch.ones(3, 3)]
        y = torch.ones(2, requires_grad=False)
        res = control_flow.map(f, xs, y)
        expected_res = _fake_map(f, xs, y)
        grad_out = torch.ones_like(res[0])
        grads = torch.autograd.grad(res[0], (xs[0],), grad_out)
        expected_grads = torch.autograd.grad(expected_res[0], (xs[0],), grad_out)
        self.assertEqual(expected_res, res)
        self.assertEqual(expected_grads, grads)

    def test_map_autograd_nested_list(self):
        if False:
            while True:
                i = 10
        import torch.utils._pytree as pytree

        def f(x, y):
            if False:
                while True:
                    i = 10
            (a, b) = x
            (c, d) = a
            return [[b.sin() * c.cos()], d.sin() * y.cos()]

        def fwbw(map_op, f, x, y):
            if False:
                for i in range(10):
                    print('nop')
            z = map_op(f, x, y)
            flat_x = pytree.tree_leaves(x)
            flat_z = pytree.tree_leaves(z)
            grads = torch.autograd.grad(flat_z, flat_x, [torch.ones_like(z) for z in flat_z])
            return (z, grads)
        x = [[torch.randn(3, 2, 2, requires_grad=True), torch.randn(3, 2, 1, requires_grad=True)], torch.ones(3, 1, 2, requires_grad=True)]
        y = torch.ones(1, requires_grad=True)
        true_outs = fwbw(control_flow.map, f, x, y)
        fake_outs = fwbw(_fake_map, f, x, y)
        self.assertEqual(true_outs, fake_outs)

@skipIfNoDynamoSupport
class TestControlFlowTraced(TestCase):

    def setUp(self):
        if False:
            return 10
        torch._dynamo.reset()
        super().setUp()

    def test_cond_traced_not_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def true_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.sin()

        def false_fn(x):
            if False:
                return 10
            return x.cos()

        def f(x, y):
            if False:
                print('Hello World!')
            return cond(y, true_fn, false_fn, [x])
        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertEqual(result_true, torch.sin(x))
        self.assertEqual(result_false, torch.cos(x))
        graph = make_fx(f, tracing_mode='symbolic')(x, torch.tensor(False))
        self.assertEqual(graph(x, torch.tensor(True)), f(x, torch.tensor(True)))

    def test_cond_nested_traced(self):
        if False:
            print('Hello World!')

        def true_nested(y):
            if False:
                i = 10
                return i + 15
            return y * y

        def false_nested(y):
            if False:
                return 10
            return y + y

        def true_fn(x, pred2):
            if False:
                print('Hello World!')
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            if False:
                print('Hello World!')
            return x.cos()

        def f(x, pred, pred2):
            if False:
                while True:
                    i = 10
            return cond(pred, true_fn, false_fn, [x, pred2])
        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))
        result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True))
        result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False))
        result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True))
        result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False))
        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))
        self.assertEqual(result_false_true, result_false_false)
        self.assertEqual(result_true_true, x * x + x)
        self.assertEqual(result_true_false, x + x + x)
        self.assertEqual(result_false_true, torch.cos(x))
        graph = make_fx(f, tracing_mode='symbolic')(x, torch.tensor(False), torch.tensor(False))
        self.assertEqual(graph(x, torch.tensor(True), torch.tensor(True)), f(x, torch.tensor(True), torch.tensor(True)))

    def test_cond_functionalized(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                print('Hello World!')
            y = x.sin()
            y.add_(4)
            return x.sin().max() + y.sum()

        def false_fn(x):
            if False:
                return 10
            return x.cos().min()

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))
        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))
        all_ops_in_true_branch = []
        for node in graph_module.true_graph_0.graph.nodes:
            if node.op == 'call_function':
                all_ops_in_true_branch.append(node.target)
        self.assertFalse(any((op._schema.is_mutable for op in all_ops_in_true_branch)))
        graph_module = make_fx(torch.func.functionalize(f), tracing_mode='symbolic')(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    def test_cond_retrace_functionalized(self):
        if False:
            return 10

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x.sin()

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return x.cos()

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            return cond(x.all(), true_fn, false_fn, (x,))
        inp = torch.ones(1, 2)
        gm_non_functional = make_fx(f, tracing_mode='real')(inp)
        gm_functional = make_fx(torch.func.functionalize(gm_non_functional), tracing_mode='real')(inp)
        self.assertEqual(gm_functional(torch.zeros(1, 2)), f(torch.zeros(1, 2)))

    def test_cond_functionalized_nested(self):
        if False:
            for i in range(10):
                print('nop')

        def true_true_fn(x):
            if False:
                print('Hello World!')
            y = x.cos()
            y.add_(4)
            return x.sin().max() + y.sin().max()

        def true_false_fn(x):
            if False:
                print('Hello World!')
            return x.cos().min()

        def true_fn(x):
            if False:
                print('Hello World!')
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return x.sum()

        def f(x):
            if False:
                while True:
                    i = 10
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))
        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))
        gm_true_true_branch = graph_module.true_graph_0.true_graph_0
        graph_module1 = make_fx(torch.func.functionalize(f), tracing_mode='symbolic')(*example_inputs)
        self.assertEqual(graph_module1(*example_inputs), f(*example_inputs))
        all_ops = []
        for node in gm_true_true_branch.graph.nodes:
            if node.op == 'call_function':
                all_ops.append(node.target)
        self.assertFalse(any((op._schema.is_mutable for op in all_ops)))

    def test_cond_functionalized_data_dependent_pred(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.sin().sum()

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.cos().sum()

        def f(x):
            if False:
                return 10
            pred = x.nonzero().shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))
        graph_module = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(graph_module(*example_inputs), f(*example_inputs))

    def test_cond_functionalized_input_mutation_on_true_branch(self):
        if False:
            return 10

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.sin().sum()

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.cos().sum()

        def f(x):
            if False:
                while True:
                    i = 10
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            functional_f(*example_inputs)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_input_mutation_on_false_branch(self):
        if False:
            for i in range(10):
                print('nop')

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x.sin().sum()

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            view_x = x.view(x.shape)
            view_x.add_(1)
            return view_x.cos().sum()

        def f(x):
            if False:
                while True:
                    i = 10
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(5, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            functional_f(*example_inputs)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_output_alias_input(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            if False:
                print('Hello World!')
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(5, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch might be aliasing'):
            functional_f(*example_inputs)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch might be aliasing'):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_nested_input_mutation(self):
        if False:
            for i in range(10):
                print('nop')

        def true_true_fn(x):
            if False:
                i = 10
                return i + 15
            x.add_(4)
            return x.sin().max()

        def true_false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x.cos().min()

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            if False:
                while True:
                    i = 10
            return x.sum()

        def f(x):
            if False:
                return 10
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])
        example_inputs = (torch.ones(4, 5),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            functional_f(*example_inputs)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            make_fx(torch.func.functionalize(f))(*example_inputs)

    def test_cond_functionalized_nested_input_mutation_with_aot_func(self):
        if False:
            i = 10
            return i + 15

        def true_true_fn(x):
            if False:
                while True:
                    i = 10
            x.add_(4)
            return x.sin().max()

        def true_false_fn(x):
            if False:
                return 10
            return x.cos().min()

        def true_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            pred = x.shape[0] == 1
            return cond(pred, true_true_fn, true_false_fn, [x])

        def false_fn(x):
            if False:
                print('Hello World!')
            return x.sum()

        def f(x):
            if False:
                print('Hello World!')
            pred = x.shape[0] == 1
            return cond(pred, true_fn, false_fn, [x])
        example_input = torch.ones(4, 5)
        try:
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
                f(example_input_func)
            with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
                make_fx(f)(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            if False:
                print('Hello World!')

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    print('Hello World!')
                torch._enable_functionalization(reapply_views=False)
                try:
                    return func(*args, **kwargs)
                finally:
                    torch._disable_functionalization()
            return wrapper
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch'):
            make_fx(f_wrapper(f))(example_input_func)

    def test_cond_functionalized_input_aliasing_with_aot_func(self):
        if False:
            print('Hello World!')

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x

        def false_fn(x):
            if False:
                print('Hello World!')
            view_x = x.view(x.shape)
            return view_x

        def f(x):
            if False:
                while True:
                    i = 10
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])
        example_input = torch.ones(5, 5)
        try:
            example_input_func = to_fun_old(example_input)
            torch._enable_functionalization(reapply_views=False)
            with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch might be aliasing'):
                f(example_input_func)
        finally:
            torch._disable_functionalization()

        def f_wrapper(func):
            if False:
                return 10

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(lambda x: torch._to_functional_tensor(x) if isinstance(x, torch.Tensor) else x, args)
                    func_kwargs = pytree.tree_map(lambda x: torch._to_functional_tensor(x) if isinstance(x, torch.Tensor) else x, kwargs)
                    return func(*func_args, **func_kwargs)
                finally:
                    torch._disable_functionalization()
            return wrapper
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'One of torch.cond branch might be aliasing'):
            make_fx(f_wrapper(f))(example_input)

    def test_cond_functionalized_aot_func_check_functional(self):
        if False:
            return 10

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x.cos()

        def false_fn(x):
            if False:
                while True:
                    i = 10
            y = x.sin()
            y.add_(5)
            return y

        def f(x):
            if False:
                while True:
                    i = 10
            pred = x.shape[0] == 4
            return cond(pred, true_fn, false_fn, [x])
        example_input = torch.ones(5, 5)

        def f_wrapper(func):
            if False:
                return 10

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    return 10
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x, args)
                    func_kwargs = pytree.tree_map(lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x, kwargs)
                    return pytree.tree_map(from_fun_old, func(*func_args, **func_kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper
        result_gm = make_fx(f_wrapper(f))(example_input)
        for node in result_gm.true_graph_0.graph.nodes:
            if node.op == 'call_function':
                self.assertTrue(not node.target._schema.is_mutable)
        for node in result_gm.false_graph_0.graph.nodes:
            if node.op == 'call_function':
                self.assertTrue(not node.target._schema.is_mutable)
        self.assertEqual(result_gm(torch.ones(5, 5)), f(torch.ones(5, 5)))

    def test_cond_nested_traced_other_inputs(self):
        if False:
            print('Hello World!')

        def true_nested(y):
            if False:
                for i in range(10):
                    print('nop')
            return y * y

        def false_nested(y):
            if False:
                while True:
                    i = 10
            return y + y

        def true_fn(k, pred2):
            if False:
                i = 10
                return i + 15
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([0.25, 0.25]), z)

        def false_fn(k, _):
            if False:
                return 10
            return k.cos()

        def f(k, pred, pred2):
            if False:
                return 10
            return cond(pred, true_fn, false_fn, [k, pred2])
        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))
        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, a * a + torch.tensor([0.25, 0.25]))
        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, b * b + torch.tensor([0.25, 0.25]))

    def test_cond_nested_traced_multi(self):
        if False:
            while True:
                i = 10

        def true_a(y):
            if False:
                while True:
                    i = 10
            return y * y

        def false_a(y):
            if False:
                print('Hello World!')
            return y + y

        def true_b(y, z):
            if False:
                print('Hello World!')
            return y + z

        def false_b(y, z):
            if False:
                while True:
                    i = 10
            return y * z

        def f(x, pred, pred2):
            if False:
                i = 10
                return i + 15
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out
        x = torch.randn(4)
        graph = make_fx(f)(x, torch.tensor(False), torch.tensor(False))
        self.assertExpectedInline(graph.code.strip(), 'def forward(self, x_1, pred_1, pred2_1):\n    true_graph_0 = self.true_graph_0\n    false_graph_0 = self.false_graph_0\n    conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);  pred_1 = true_graph_0 = false_graph_0 = None\n    true_graph_1 = self.true_graph_1\n    false_graph_1 = self.false_graph_1\n    conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1]);  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None\n    add = torch.ops.aten.add.Tensor(conditional, conditional_1);  conditional = conditional_1 = None\n    return add')
        self.assertExpectedInline(graph.true_graph_0.code.strip(), 'def forward(self, arg0_1):\n    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    return mul')

    def test_raise_error_on_mismatch_type_size(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                return 10
            return x.sin()

        def false_fn(x):
            if False:
                print('Hello World!')
            return (x, x)

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return cond(y, true_fn, false_fn, [x])
        x = torch.randn(4)
        with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
            make_fx(f)(x, torch.tensor(False))

    def test_raise_error_on_mismatch_tensor_size(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x.sin()

        def false_fn(x):
            if False:
                print('Hello World!')
            return torch.zeros([10, 10])

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return cond(y, true_fn, false_fn, [x])
        x = torch.randn(4)
        with self.assertRaisesRegex(CondOpArgsMismatchError, 'Expected each tensor to have same metadata but got'):
            make_fx(f)(x, torch.tensor(False))

    def test_cond_traced_not_nested_fake_tensor(self):
        if False:
            while True:
                i = 10

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return x.sin()

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return x.cos()

        def f(x, y):
            if False:
                return 10
            return cond(y, true_fn, false_fn, [x])
        x = torch.randn(4)
        graph = make_fx(f, tracing_mode='fake')(x, torch.tensor(False))
        result_true = graph.forward(x, torch.tensor(True))
        result_false = graph.forward(x, torch.tensor(False))
        self.assertFalse(torch.allclose(result_true, result_false))
        self.assertEqual(result_true, torch.sin(x))
        self.assertEqual(result_false, torch.cos(x))

    def test_cond_nested_traced_fake_tensor(self):
        if False:
            while True:
                i = 10

        def true_nested(y):
            if False:
                while True:
                    i = 10
            return y * y

        def false_nested(y):
            if False:
                while True:
                    i = 10
            return y + y

        def true_fn(x, pred2):
            if False:
                return 10
            z = cond(pred2, true_nested, false_nested, [x])
            return x + z

        def false_fn(x, _):
            if False:
                return 10
            return x.cos()

        def f(x, pred, pred2):
            if False:
                while True:
                    i = 10
            return cond(pred, true_fn, false_fn, [x, pred2])
        x = torch.randn(4)
        graph = make_fx(f, tracing_mode='fake')(x, torch.tensor(False), torch.tensor(False))
        result_true_true = graph.forward(x, torch.tensor(True), torch.tensor(True))
        result_true_false = graph.forward(x, torch.tensor(True), torch.tensor(False))
        result_false_true = graph.forward(x, torch.tensor(False), torch.tensor(True))
        result_false_false = graph.forward(x, torch.tensor(False), torch.tensor(False))
        self.assertNotEqual(result_true_true, result_true_false)
        self.assertFalse(torch.allclose(result_false_true, result_true_true))
        self.assertEqual(result_false_true, result_false_false)
        self.assertEqual(result_true_true, x * x + x)
        self.assertEqual(result_true_false, x + x + x)
        self.assertEqual(result_false_true, torch.cos(x))

    def test_cond_nested_traced_other_inputs_fake_tensor(self):
        if False:
            i = 10
            return i + 15

        def true_nested(y):
            if False:
                return 10
            return y * y

        def false_nested(y):
            if False:
                return 10
            return y + y

        def true_fn(k, pred2):
            if False:
                print('Hello World!')
            z = cond(pred2, true_nested, false_nested, [k])
            return torch.add(torch.tensor([0.25, 0.25]), z)

        def false_fn(k, _):
            if False:
                i = 10
                return i + 15
            return k.cos()

        def f(k, pred, pred2):
            if False:
                return 10
            return cond(pred, true_fn, false_fn, [k, pred2])
        x = torch.tensor([0.5, 0.5])
        graph = make_fx(f, tracing_mode='fake')(x, torch.tensor(False), torch.tensor(False))
        a = torch.tensor([1.0, 1.0])
        result_true_true = graph.forward(a, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, a * a + torch.tensor([0.25, 0.25]))
        b = torch.tensor([2.0, 2.0])
        result_true_true = graph.forward(b, torch.tensor(True), torch.tensor(True))
        self.assertEqual(result_true_true, b * b + torch.tensor([0.25, 0.25]))

    def test_cond_nested_traced_multi_fake_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def true_a(y):
            if False:
                i = 10
                return i + 15
            return y * y

        def false_a(y):
            if False:
                for i in range(10):
                    print('nop')
            return y + y

        def true_b(y, z):
            if False:
                i = 10
                return i + 15
            return y + z

        def false_b(y, z):
            if False:
                for i in range(10):
                    print('nop')
            return y * z

        def f(x, pred, pred2):
            if False:
                i = 10
                return i + 15
            a_out = cond(pred, true_a, false_a, [x])
            b_out = cond(pred2, true_b, false_b, [x, x])
            return a_out + b_out
        x = torch.randn(4)
        graph = make_fx(f, tracing_mode='fake')(x, torch.tensor(False), torch.tensor(False))
        self.assertExpectedInline(graph.code.strip(), 'def forward(self, x_1, pred_1, pred2_1):\n    true_graph_0 = self.true_graph_0\n    false_graph_0 = self.false_graph_0\n    conditional = torch.ops.higher_order.cond(pred_1, true_graph_0, false_graph_0, [x_1]);  pred_1 = true_graph_0 = false_graph_0 = None\n    true_graph_1 = self.true_graph_1\n    false_graph_1 = self.false_graph_1\n    conditional_1 = torch.ops.higher_order.cond(pred2_1, true_graph_1, false_graph_1, [x_1]);  pred2_1 = true_graph_1 = false_graph_1 = x_1 = None\n    add = torch.ops.aten.add.Tensor(conditional, conditional_1);  conditional = conditional_1 = None\n    return add')
        self.assertExpectedInline(graph.true_graph_0.code.strip(), 'def forward(self, arg0_1):\n    mul = torch.ops.aten.mul.Tensor(arg0_1, arg0_1);  arg0_1 = None\n    return mul')

    def test_raise_error_on_mismatch_type_size_fake_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return x.sin()

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return (x, x)

        def f(x, y):
            if False:
                print('Hello World!')
            return cond(y, true_fn, false_fn, [x])
        x = torch.randn(4)
        with self.assertRaisesRegex(torch._dynamo.exc.UncapturedHigherOrderOpError, "Cond doesn't work unless it is captured completely with torch.compile"):
            make_fx(f, tracing_mode='fake')(x, torch.tensor(False))

    def test_raise_error_on_mismatch_tensor_size_fake_tensor(self):
        if False:
            for i in range(10):
                print('nop')

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x.sin()

        def false_fn(x):
            if False:
                return 10
            return torch.zeros([10, 10])

        def f(x, y):
            if False:
                while True:
                    i = 10
            return cond(y, true_fn, false_fn, [x])
        x = torch.randn(4)
        with self.assertRaisesRegex(CondOpArgsMismatchError, 'Expected each tensor to have same metadata but got'):
            make_fx(f, tracing_mode='fake')(x, torch.tensor(False))

    def check_map_count(self, gm, op_count):
        if False:
            print('Hello World!')
        i = 0
        for m in gm.modules():
            for node in m.graph.nodes:
                if node.op == 'call_function' and node.target == torch.ops.higher_order.map_impl:
                    i += 1
        self.assertEqual(i, op_count)

    def test_tracing_map_real(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                print('Hello World!')
            return x + y

        def g(xs, y):
            if False:
                print('Hello World!')
            return control_flow.map(f, xs, y)
        gm = make_fx(g, tracing_mode='real')(torch.ones(3, 2, 2), torch.ones(2))
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_simple(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return x + y

        def g(xs, y):
            if False:
                return 10
            return control_flow.map(f, xs, y)
        gm = make_fx(g, tracing_mode='symbolic')(torch.ones(3, 2, 4), torch.ones(4))
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_list(self):
        if False:
            print('Hello World!')

        def f(x, y):
            if False:
                print('Hello World!')
            return [x[0][0] + y, x[1] * y]

        def g(xs, y, z):
            if False:
                while True:
                    i = 10
            out = control_flow.map(f, xs, y)
            return (out[0] + z, out[1] * z)
        example_x = [[torch.ones(3, 4, 5)], torch.ones(3, 4, 5)]
        gm = make_fx(g, tracing_mode='symbolic')(example_x, torch.ones(5), torch.ones(5))
        x = [[torch.randn(4, 5, 6)], torch.ones(4, 5, 6)]
        y = torch.randn(6)
        z = torch.ones(6)
        res = gm(x, y, z)
        self.assertEqual(res, g(x, y, z))
        self.check_map_count(gm, 1)

    def test_tracing_map_symbolic_dict(self):
        if False:
            i = 10
            return i + 15

        def f(x, y):
            if False:
                print('Hello World!')
            return {'d': x['b']['a'] + y, 'e': x['c'] * y}

        def g(xs, y, z):
            if False:
                return 10
            out = control_flow.map(f, xs, y)
            return {'f': out['d'] + z, 'g': out['e'] * z}
        example_x = {'b': {'a': torch.ones(3, 4, 5)}, 'c': torch.ones(3, 4, 5)}
        gm = make_fx(g, tracing_mode='symbolic')(example_x, torch.ones(5), torch.ones(5))
        x = {'b': {'a': torch.randn(4, 5, 6)}, 'c': torch.ones(4, 5, 6)}
        y = torch.randn(6)
        z = torch.ones(6)
        res = gm(x, y, z)
        self.assertEqual(res, g(x, y, z))
        self.check_map_count(gm, 1)

    def test_tracing_map_autograd_symbolic_simple(self):
        if False:
            return 10

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y

        def g(xs, y):
            if False:
                print('Hello World!')
            out = control_flow.map(f, xs, y)
            return torch.autograd.grad(out, (xs, y), torch.ones_like(out))
        gm = make_fx(g, tracing_mode='symbolic')(torch.ones(3, 4, 5, requires_grad=True), torch.ones(5, requires_grad=True))
        x = torch.randn(4, 5, 6, requires_grad=True)
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_symbolic_list(self):
        if False:
            for i in range(10):
                print('nop')
        import torch.utils._pytree as pytree

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return [x[0].cos() + y.sin(), x[1].sin() * y.cos()]

        def g(xs, y):
            if False:
                for i in range(10):
                    print('nop')
            out = control_flow.map(f, xs, y)
            flat_out = pytree.tree_leaves(out)
            flat_inp = pytree.tree_leaves((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out])
        gm = make_fx(g, tracing_mode='symbolic')([torch.ones(3, 4, 5), torch.ones(3, 4, 5, requires_grad=True)], torch.ones(5, requires_grad=True))
        x = [torch.randn(4, 5, 6), torch.ones(4, 5, 6, requires_grad=True)]
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_symbolic_dict(self):
        if False:
            while True:
                i = 10

        def f(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return [x['a'] + y, x['b'] * y]

        def g(xs, y):
            if False:
                while True:
                    i = 10
            out = control_flow.map(f, xs, y)
            flat_out = pytree.tree_leaves(out)
            flat_inp = pytree.tree_leaves((xs, y))
            requires_grad_inp = [inp for inp in flat_inp if inp.requires_grad]
            return torch.autograd.grad(flat_out, requires_grad_inp, [torch.ones_like(out) for out in flat_out])
        traced_x = {'a': torch.ones(3, 4, 5, requires_grad=True), 'b': torch.ones(3, 4, 5, requires_grad=True)}
        gm = make_fx(g, tracing_mode='symbolic')(traced_x, torch.ones(5, requires_grad=True))
        x = {'a': torch.randn(4, 5, 6, requires_grad=True), 'b': torch.ones(4, 5, 6, requires_grad=True)}
        y = torch.randn(6, requires_grad=True)
        res = gm(x, y)
        self.assertEqual(res, g(x, y))
        self.check_map_count(gm, 2)

    def test_tracing_map_autograd_aot_functionalized(self):
        if False:
            print('Hello World!')

        def inner(x, y):
            if False:
                print('Hello World!')
            z = x - 1
            z.add_(1)
            return z * y

        def f(xs, y):
            if False:
                for i in range(10):
                    print('nop')
            res = control_flow.map(inner, xs, y)
            grads = torch.autograd.grad(res, (xs, y), torch.ones_like(res))
            return grads

        def f_wrapper(func):
            if False:
                i = 10
                return i + 15

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    for i in range(10):
                        print('nop')
                torch._enable_functionalization(reapply_views=False)
                try:
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper
        example_inputs = (torch.ones(3, 2, 4, requires_grad=True), torch.ones(2, 4, requires_grad=True))
        gm = make_fx(f, tracing_mode='symbolic')(*example_inputs)
        fgm = make_fx(f_wrapper(f), tracing_mode='symbolic')(*example_inputs)
        xs = torch.ones(3, 4, 5, requires_grad=True)
        y = torch.ones(4, 5, requires_grad=True)
        self.assertEqual(gm(xs, y), f(xs, y))

        def count_mutable(gm):
            if False:
                print('Hello World!')
            c = 0
            for node in gm.graph.nodes:
                if node.op == 'call_function':
                    if node.target == torch.ops.higher_order.map_impl:
                        c += count_mutable(getattr(gm, str(node.args[0])))
                    elif (schema := getattr(node.target, '_schema', None)):
                        c += int(schema.is_mutable)
            return c
        self.assertEqual(count_mutable(fgm), 0)
        self.assertEqual(count_mutable(gm), 2)

    def test_map_functionalized(self):
        if False:
            for i in range(10):
                print('nop')

        def map_fn(x, y):
            if False:
                return 10
            z = x + y
            z.add_(4)
            return z

        def f(xs, y):
            if False:
                return 10
            return control_flow.map(map_fn, xs, y)
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        self.assertEqual(functional_f(*example_inputs), f(*example_inputs))
        gm = make_fx(torch.func.functionalize(f))(*example_inputs)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))
        gm = make_fx(torch.func.functionalize(f), tracing_mode='symbolic')(*example_inputs)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))
        for node in gm.body_graph_0.graph.nodes:
            if node.op == 'call_function':
                self.assertTrue(not node.target._schema.is_mutable)
        self.check_map_count(gm, 1)

    def test_map_functionalized_aot_func(self):
        if False:
            print('Hello World!')

        def map_fn(x, y):
            if False:
                while True:
                    i = 10
            z = x + y
            z.add_(4)
            return z

        def f(xs, y):
            if False:
                print('Hello World!')
            return control_flow.map(map_fn, xs, y)

        def f_wrapper(func):
            if False:
                print('Hello World!')

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    while True:
                        i = 10
                torch._enable_functionalization(reapply_views=False)
                try:
                    return pytree.tree_map(from_fun_old, func(*args, **kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        gm = make_fx(f_wrapper(f))(*example_inputs)
        for node in gm.body_graph_0.graph.nodes:
            if node.op == 'call_function':
                self.assertTrue(not node.target._schema.is_mutable)
        self.assertEqual(gm(*example_inputs), f(*example_inputs))

    def test_map_functionalized_arg_mutation(self):
        if False:
            i = 10
            return i + 15

        def map_fn(x, y):
            if False:
                print('Hello World!')
            y.add_(4)
            return x + y

        def f(xs, y):
            if False:
                return 10
            return control_flow.map(map_fn, xs, y)
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'torch.map is mutating the input!'):
            functional_f(*example_inputs)

    def test_map_functionalized_elem_mutation(self):
        if False:
            while True:
                i = 10

        def map_fn(x, y):
            if False:
                i = 10
                return i + 15
            x.add_(4)
            return x + y

        def f(xs, y):
            if False:
                return 10
            return control_flow.map(map_fn, xs, y)
        example_inputs = (torch.ones(3, 2, 4), torch.ones(4))
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'torch.map is mutating the input!'):
            functional_f(*example_inputs)

    def test_cond_autograd_fail(self):
        if False:
            for i in range(10):
                print('nop')

        def true_fn(x):
            if False:
                print('Hello World!')
            return x.cos()

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return x.sin()

        def f(x, y):
            if False:
                print('Hello World!')
            return control_flow.cond(x.shape[0] > 4, true_fn, false_fn, [y])
        example_inputs = (torch.ones(3, 2, 4, requires_grad=True), torch.ones(4, requires_grad=True))
        with self.assertRaisesRegex(RuntimeError, 'Autograd not implemented for cond'):
            f(*example_inputs).sum().backward()
        f(*example_inputs)

    def test_map_functionalized_elem_alias(self):
        if False:
            return 10

        def map_fn(x):
            if False:
                print('Hello World!')
            x.view(x.shape)
            return x

        def f(xs):
            if False:
                print('Hello World!')
            return control_flow.map(map_fn, xs)
        example_inputs = (torch.ones(3, 2, 4),)
        functional_f = torch.func.functionalize(f)
        with self.assertRaisesRegex(UnsupportedAliasMutationException, 'torch.map is aliasing the input!'):
            functional_f(*example_inputs)

    def test_nested_map_cond_real(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x * y

        def false_fn(x, y):
            if False:
                return 10
            return x + y

        def f(x, pred, y):
            if False:
                i = 10
                return i + 15
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            if False:
                for i in range(10):
                    print('nop')
            return control_flow.map(f, xs, pred, y)
        gm = make_fx(g, tracing_mode='real')(torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4))
        pred = torch.tensor(False)
        x = torch.randn(3, 2, 4)
        y = torch.randn(4)
        res = gm(pred, x, y)
        self.assertEqual(res, g(pred, x, y))
        self.check_map_count(gm, 1)

    def test_nested_map_cond_symbolic(self):
        if False:
            for i in range(10):
                print('nop')

        def true_fn(x, y):
            if False:
                while True:
                    i = 10
            return x * y

        def false_fn(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return x + y

        def f(x, pred, y):
            if False:
                return 10
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            if False:
                while True:
                    i = 10
            return control_flow.map(f, xs, pred, y)
        gm = make_fx(g, tracing_mode='symbolic')(torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4))
        pred = torch.tensor(False)
        x = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(pred, x, y)
        self.assertEqual(res, g(pred, x, y))
        self.check_map_count(gm, 1)

    def test_nested_cond_map_cond_symbolic(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x, y):
            if False:
                print('Hello World!')
            return x * y

        def false_fn(x, y):
            if False:
                while True:
                    i = 10
            return x + y

        def f(x, pred, y):
            if False:
                print('Hello World!')
            return cond(pred, true_fn, false_fn, [x, y])

        def g(pred, xs, y):
            if False:
                for i in range(10):
                    print('nop')
            return control_flow.map(f, xs, pred, y)

        def main_true_fn(pred, xs, y):
            if False:
                for i in range(10):
                    print('nop')
            return g(pred, xs, y) * 2

        def main_false_fn(pred, xs, y):
            if False:
                i = 10
                return i + 15
            return g(pred, xs, y) + 1

        def main(p, pred, xs, y):
            if False:
                for i in range(10):
                    print('nop')
            return cond(p, main_true_fn, main_false_fn, [pred, xs, y])
        gm = make_fx(main, tracing_mode='symbolic')(torch.tensor(True), torch.tensor(True), torch.ones(3, 2, 4), torch.ones(4))
        p = torch.tensor(False)
        pred = torch.tensor(False)
        xs = torch.randn(3, 2, 2)
        y = torch.randn(2)
        res = gm(p, pred, xs, y)
        self.assertEqual(res, main(p, pred, xs, y))
        self.check_map_count(gm, 2)

    def test_cond_with_sym_pred(self):
        if False:
            while True:
                i = 10

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return x + x

        def false_fn(x):
            if False:
                print('Hello World!')
            return x * x

        def foo(x):
            if False:
                print('Hello World!')
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        gm = make_fx(foo, tracing_mode='symbolic')(torch.ones(3, 2, 1))
        self.assertEqual(len(gm.shape_env.guards), 0)
        self.assertExpectedInline(gm.code.strip(), 'def forward(self, x_1):\n    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)\n    eq = sym_size_int == 4;  sym_size_int = None\n    true_graph_0 = self.true_graph_0\n    false_graph_0 = self.false_graph_0\n    conditional = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, [x_1]);  eq = true_graph_0 = false_graph_0 = x_1 = None\n    return conditional')
        x = torch.ones(4, 3, 2)
        self.assertEqual(gm(x), true_fn(x))
        self.assertEqual(foo(x), true_fn(x))

    def _check_closure_correctly_lifted(self, f, *, args, exp_res, exp_arg_num):
        if False:
            i = 10
            return i + 15
        assert isinstance(args, (tuple, list))
        self.assertEqual(f(*args), exp_res)
        gm = make_fx(f)(*args)
        gm.print_readable()
        self.assertEqual(gm(*args), exp_res)

        def cnt_placeholder(gm):
            if False:
                for i in range(10):
                    print('nop')
            return len([node for node in gm.graph.nodes if node.op == 'placeholder'])
        placeholder_cnts = [cnt_placeholder(mod) for mod in gm.children()]
        self.assertTrue(all((cnt == exp_arg_num for cnt in placeholder_cnts)))

    def _check_closure_correctly_lifted_with_mutation(self, f, closures_to_be_mutated, *, args, exp_arg_num):
        if False:
            while True:
                i = 10
        exp_res = f(*args)
        self._check_closure_correctly_lifted(f, args=args, exp_res=exp_res, exp_arg_num=exp_arg_num)
        for closure in closures_to_be_mutated:
            closure.add(-1)
        new_exp_res = f(*args)
        self._check_closure_correctly_lifted(f, args=args, exp_res=new_exp_res, exp_arg_num=exp_arg_num)

    def test_cond_with_tensor_closure(self):
        if False:
            for i in range(10):
                print('nop')
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            if False:
                print('Hello World!')
            return x + a

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + b

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        inp = torch.randn(2, 3)
        self._check_closure_correctly_lifted_with_mutation(foo, (a, b), args=(inp,), exp_arg_num=3)

    def test_cond_with_tensor_closure_graph_module(self):
        if False:
            while True:
                i = 10
        a = torch.ones(2, 3)
        b = torch.ones(2, 3) + 1

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x + a

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + b

        def foo(x):
            if False:
                return 10
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        inp = torch.randn(2, 3)
        gm = make_fx(foo)(inp)
        self.assertExpectedInline(gm.code.strip(), 'def forward(self, x_1):\n    true_graph_0 = self.true_graph_0\n    false_graph_0 = self.false_graph_0\n    _tensor_constant0 = self._tensor_constant0\n    _tensor_constant1 = self._tensor_constant1\n    conditional = torch.ops.higher_order.cond(False, true_graph_0, false_graph_0, [x_1, _tensor_constant0, _tensor_constant1]);  true_graph_0 = false_graph_0 = x_1 = _tensor_constant0 = _tensor_constant1 = None\n    return conditional')
        self.assertExpectedInline(gm.true_graph_0.code.strip(), 'def forward(self, arg0_1, arg1_1, arg2_1):\n    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None\n    return add')

    def test_cond_with_module_param_closure(self):
        if False:
            print('Hello World!')

        class Mod(torch.nn.Module):

            def __init__(self):
                if False:
                    while True:
                        i = 10
                super().__init__()
                self.register_parameter('param', torch.nn.Parameter(torch.ones(2, 3)))
                self.register_buffer('buffer', torch.ones(2, 3) + 1)
        my_mode = Mod()

        def true_fn(x):
            if False:
                while True:
                    i = 10
            return x + my_mode.param

        def false_fn(x):
            if False:
                return 10
            return x + my_mode.buffer

        def foo(x):
            if False:
                return 10
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        inp = torch.ones(2, 3)
        self._check_closure_correctly_lifted_with_mutation(foo, (my_mode.param, my_mode.buffer), args=(inp,), exp_arg_num=3)

    def test_cond_with_module_python_scalar_closure(self):
        if False:
            print('Hello World!')

        def foo(x):
            if False:
                while True:
                    i = 10
            a = torch.ones(1, 1)
            b = 1

            def true_fn(x):
                if False:
                    print('Hello World!')
                return x + a

            def false_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return x + b
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        inp = torch.ones(2, 3)
        res = inp + 1
        self._check_closure_correctly_lifted(foo, args=(inp,), exp_res=res, exp_arg_num=2)

    def test_cond_nested_with_closure(self):
        if False:
            i = 10
            return i + 15
        a = torch.ones(1, 1)
        b = torch.ones(1, 1) + 1

        def inner_true_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + a

        def inner_false_fn(x):
            if False:
                while True:
                    i = 10
            return x + b

        def foo(x):
            if False:
                for i in range(10):
                    print('nop')

            def true_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            def false_fn(x):
                if False:
                    print('Hello World!')
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        inp = torch.ones(2, 3)
        self._check_closure_correctly_lifted_with_mutation(foo, (a, b), args=(inp,), exp_arg_num=5)

    def test_cond_nested_with_closure_graph_module(self):
        if False:
            i = 10
            return i + 15
        a = torch.ones(1, 1)
        b = torch.ones(1, 1) + 1

        def inner_true_fn(x):
            if False:
                while True:
                    i = 10
            return x + a

        def inner_false_fn(x):
            if False:
                i = 10
                return i + 15
            return x + b

        def foo(x):
            if False:
                return 10

            def true_fn(x):
                if False:
                    return 10
                return cond(x.shape[0] == 2, inner_true_fn, inner_false_fn, [x])

            def false_fn(x):
                if False:
                    for i in range(10):
                        print('nop')
                return cond(x.shape[0] > 4, inner_true_fn, inner_false_fn, [x])
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])

    def test_map_unfunc_boolean_tensor_for_nested_map_cond(self):
        if False:
            while True:
                i = 10

        def map_fn(pred, x):
            if False:
                return 10

            def fn(x, pred):
                if False:
                    return 10
                return control_flow.cond(pred, lambda x: x * 2, lambda x: x / 2, (x,))
            return control_flow.map(fn, x, pred)

        def f_wrapper(func):
            if False:
                return 10

            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if False:
                    return 10
                torch._enable_functionalization(reapply_views=False)
                try:
                    func_args = pytree.tree_map(lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x, args)
                    func_kwargs = pytree.tree_map(lambda x: to_fun_old(x) if isinstance(x, torch.Tensor) else x, kwargs)
                    return pytree.tree_map(from_fun_old, func(*func_args, **func_kwargs))
                finally:
                    torch._disable_functionalization()
            return wrapper
        gm = make_fx(f_wrapper(map_fn))(torch.tensor(True), torch.ones([2, 3], requires_grad=False))
        self.assertExpectedInline(gm.code.strip(), 'def forward(self, pred_1, x_1):\n    body_graph_0 = self.body_graph_0\n    map_impl = torch.ops.higher_order.map_impl(body_graph_0, 1, x_1, pred_1);  body_graph_0 = x_1 = pred_1 = None\n    getitem = map_impl[0];  map_impl = None\n    return getitem')
        self.assertExpectedInline(gm.body_graph_0.code.strip(), 'def forward(self, arg0_1, arg1_1):\n    true_graph_0 = self.true_graph_0\n    false_graph_0 = self.false_graph_0\n    conditional = torch.ops.higher_order.cond(arg1_1, true_graph_0, false_graph_0, [arg0_1]);  arg1_1 = true_graph_0 = false_graph_0 = arg0_1 = None\n    return [conditional]')

    def test_cond_make_fx_preserve_stack_trace_for_nodes_in_subgraph(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                i = 10
                return i + 15
            return x + x.cos()

        def false_fn(x):
            if False:
                i = 10
                return i + 15
            return x * x.sin()

        def foo(x):
            if False:
                i = 10
                return i + 15
            return cond(x.shape[0] == 4, true_fn, false_fn, (x,))
        inp = torch.randn([4, 3])
        (gm, _) = torch._dynamo.export(foo)(inp)

        def run_with_interpreter(*args):
            if False:
                while True:
                    i = 10
            with torch.fx.traceback.preserve_node_meta():
                return torch.fx.Interpreter(gm).run(*args)
        new_gm = make_fx(run_with_interpreter)(inp)
        checked_ops = {'add', 'mul', 'sin', 'cos'}
        checked_meta = ['source_fn_stack', 'stack_trace']
        all_source_fns = collect_meta_for_filtered_nodes(gm, checked_ops, checked_meta)
        new_source_fns = collect_meta_for_filtered_nodes(new_gm, checked_ops, checked_meta)
        self.assertEqual(all_source_fns, new_source_fns)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, 'triggers cache limit for foo and changes unique_graphs count.')
    def test_cond_no_dynamo_cache_limit(self):
        if False:
            print('Hello World!')
        torch._dynamo.reset()
        counters = torch._dynamo.utils.counters
        counters.clear()

        def foo(x, true_fn, false_fn):
            if False:
                while True:
                    i = 10
            return cond(x.shape[0] == 4, true_fn, false_fn, (x,))
        inp = torch.ones(3, 4)
        exp_out = inp.sin()
        iter_n = torch._dynamo.config.cache_size_limit + 1

        def make_dummy_fn(op):
            if False:
                for i in range(10):
                    print('nop')
            exec(f'temp = lambda x: x.{op}()')
            return locals()['temp']
        for _ in range(iter_n):
            self.assertEqual(foo(inp, make_dummy_fn('cos'), make_dummy_fn('sin')), exp_out)
        self.assertEqual(counters['stats']['calls_captured'], iter_n)
        self.assertEqual(counters['stats']['unique_graphs'], iter_n)

    def test_cond_with_consecutive_make_fx_symbolic(self):
        if False:
            i = 10
            return i + 15

        def true_fn(x):
            if False:
                return 10
            return x - x.cos()

        def false_fn(x):
            if False:
                for i in range(10):
                    print('nop')
            return x + x.sin()

        def foo(x):
            if False:
                print('Hello World!')
            return cond(x.shape[0] == 4, true_fn, false_fn, [x])
        inps = (torch.ones(3, 4), torch.ones(3, 5), torch.ones(5, 4), torch.ones(5, 3))
        for inp in inps:
            gm = make_fx(foo, tracing_mode='symbolic')(torch.ones(3, 4))
            self.assertExpectedInline(gm.code.strip(), 'def forward(self, x_1):\n    sym_size_int = torch.ops.aten.sym_size.int(x_1, 0)\n    eq = sym_size_int == 4;  sym_size_int = None\n    true_graph_0 = self.true_graph_0\n    false_graph_0 = self.false_graph_0\n    conditional = torch.ops.higher_order.cond(eq, true_graph_0, false_graph_0, [x_1]);  eq = true_graph_0 = false_graph_0 = x_1 = None\n    return conditional')
            self.assertExpectedInline(gm.true_graph_0.code.strip(), 'def forward(self, arg0_1):\n    cos = torch.ops.aten.cos.default(arg0_1)\n    sub = torch.ops.aten.sub.Tensor(arg0_1, cos);  arg0_1 = cos = None\n    return sub')
            self.assertExpectedInline(gm.false_graph_0.code.strip(), 'def forward(self, arg0_1):\n    sin = torch.ops.aten.sin.default(arg0_1)\n    add = torch.ops.aten.add.Tensor(arg0_1, sin);  arg0_1 = sin = None\n    return add')

    def _create_test_fns_for_cond(self, pred, inner_most_fn, operands, closure_list, nested_level):
        if False:
            i = 10
            return i + 15
        if nested_level == 0:
            if len(closure_list) > 0:

                def true_fn(*operands):
                    if False:
                        for i in range(10):
                            print('nop')
                    return inner_most_fn(*operands) + inner_most_fn(*closure_list)

                def false_fn(*operands):
                    if False:
                        while True:
                            i = 10
                    return inner_most_fn(*operands) - inner_most_fn(*closure_list)
            else:

                def true_fn(*operands):
                    if False:
                        while True:
                            i = 10
                    return inner_most_fn(*operands)

                def false_fn(*operands):
                    if False:
                        while True:
                            i = 10
                    return inner_most_fn(*operands)

            def fn(pred, operands):
                if False:
                    print('Hello World!')
                if len(operands) == 0 and len(closure_list) == 0:
                    return torch.zeros(1)
                return cond(pred, true_fn, false_fn, operands)
            return ((pred, operands), fn)
        else:
            (args, inner_fn) = self._create_test_fns_for_cond(pred <= 0, inner_most_fn, operands, closure_list, nested_level - 1)

            def true_fn(*operands):
                if False:
                    return 10
                return inner_most_fn(*operands) + inner_fn(*args)

            def false_fn(*operands):
                if False:
                    for i in range(10):
                        print('nop')
                return inner_most_fn(*operands) - inner_fn(*args)

            def fn(pred, operands):
                if False:
                    print('Hello World!')
                if len(operands) == 0 and len(closure_list) == 0:
                    return torch.ones(1)
                return cond(pred, true_fn, false_fn, operands)
            return ((pred, operands), fn)

    def _init_predicate(self, pred_type):
        if False:
            for i in range(10):
                print('nop')
        if pred_type == 'bool':
            return True
        elif pred_type == 'intTensor':
            return torch.tensor(1)
        elif pred_type == 'floatTensor':
            return torch.tensor(1.0)
        elif pred_type == 'boolTensor':
            return torch.tensor(False)
        else:
            raise NotImplementedError

    def _init_fn(self, inner_fn_type):
        if False:
            while True:
                i = 10
        if inner_fn_type == 'function':
            return reduce_func
        elif inner_fn_type == 'module':
            return ReduceMod()
        elif inner_fn_type == 'object':
            return ReduceObj()
        else:
            raise NotImplementedError

    @parametrize('predType', ['bool', 'intTensor', 'floatTensor', 'boolTensor'])
    @parametrize('innerFnType', ['function', 'module', 'object'])
    @parametrize('nOperands', [0, 1])
    @parametrize('nClosure', [0, 1])
    @parametrize('nesting', [0, 2])
    def test_cond_tracing_with_valid_inputs(self, predType, innerFnType, nOperands, nClosure, nesting):
        if False:
            while True:
                i = 10
        pred = self._init_predicate(predType)
        inner_fn = self._init_fn(innerFnType)
        operands = [torch.ones(2, 3) + i for i in range(nOperands)]
        closure = [torch.ones(2, 3) - i for i in range(nClosure)]
        (args, fn) = self._create_test_fns_for_cond(pred, inner_fn, operands, closure, nesting)
        eager_res = fn(*args)
        for tracing_mode in ['symbolic', 'fake', 'real']:
            with self.subTest(tracing_mode=tracing_mode):
                gm = make_fx(fn, tracing_mode=tracing_mode, _allow_non_fake_inputs=True)(*args)
                self.assertEqual(gm(*args), eager_res)
instantiate_parametrized_tests(TestControlFlowTraced)
if __name__ == '__main__':
    run_tests()