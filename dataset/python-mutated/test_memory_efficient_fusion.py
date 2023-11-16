import torch
import torch.nn as nn
import torch.fx as fx
from functorch import make_fx
from torch.nn import functional as F
from functorch.compile import memory_efficient_fusion
from torch._functorch.compile_utils import fx_graph_cse
from torch.testing._internal.common_utils import TestCase, run_tests
import inspect
import random
from typing import Callable
import unittest
HAS_CUDA = torch.cuda.is_available()

def _num_args(fn: Callable):
    if False:
        return 10
    return len(inspect.signature(fn).parameters)

def gelu_bias(bias, y):
    if False:
        print('Hello World!')
    x = bias + y
    return x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

def swish(x):
    if False:
        while True:
            i = 10
    return x * torch.sigmoid(x)

def mish(x):
    if False:
        return 10
    return x.mul(torch.tanh(F.softplus(x)))

def hard_sigmoid(x):
    if False:
        print('Hello World!')
    return (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)

def hard_swish(x):
    if False:
        print('Hello World!')
    return x * (x + 3.0).clamp(min=0.0, max=6.0).div(6.0)

def hard_mish(x):
    if False:
        print('Hello World!')
    return 0.5 * x * (x + 2.0).clamp(min=0.0, max=2.0)

def run_and_compare_activation(self, fn, inps):
    if False:
        return 10
    with torch.jit.fuser('fuser1'):
        device = 'cuda'
        dtype = torch.float
        if isinstance(fn, nn.Module):
            fn = fn.to(device=device, dtype=dtype)
        ref_args = [torch.randn(shape, device=device, dtype=dtype, requires_grad=True) for shape in inps]
        res_args = [i.clone().detach().requires_grad_(True) for i in ref_args]
        ref = fn(*ref_args)
        ref.sum().backward()
        mem_optimized_fn = memory_efficient_fusion(fn)
        for _ in range(5):
            for i in res_args:
                i.grad = None
            res = mem_optimized_fn(*res_args)
            res.sum().backward()
        self.assertEqual(ref, res)
        for (ref_arg, res_arg) in zip(ref_args, res_args):
            self.assertEqual(ref_arg.grad, res_arg.grad)

@unittest.skipIf(not torch.cuda.is_available(), 'CUDA is unavailable')
class TestMemoryEfficientOpAuthoring(TestCase):

    def test_gelu_bias(self):
        if False:
            print('Hello World!')
        run_and_compare_activation(self, gelu_bias, [(1024,), (1024,)])

    def test_mish(self):
        if False:
            return 10
        run_and_compare_activation(self, mish, [(1024,)])

    def test_swish(self):
        if False:
            print('Hello World!')
        run_and_compare_activation(self, swish, [(1024,)])

    def test_hard_sigmoid(self):
        if False:
            while True:
                i = 10
        run_and_compare_activation(self, hard_sigmoid, [(1024,)])

    def test_hard_swish(self):
        if False:
            i = 10
            return i + 15
        run_and_compare_activation(self, hard_swish, [(1024,)])

    def test_layer_norm(self):
        if False:
            return 10

        def layer_norm(x, weight, bias):
            if False:
                for i in range(10):
                    print('nop')
            dim = -1
            eps = 1e-05
            mean = torch.mean(x, dim, keepdim=True)
            centered = x - mean
            var = torch.sum(centered * centered, dim, keepdim=True) / x.size(-1)
            rvar = 1.0 / torch.sqrt(var + eps)
            normed = (x - mean) * rvar
            return normed * weight + bias
        bs = 10
        ln_size = 16
        layer_norm_inps = [(bs, ln_size), (ln_size,), (ln_size,)]
        run_and_compare_activation(self, layer_norm, layer_norm_inps)

    def test_rmsnorm(self):
        if False:
            while True:
                i = 10

        class T5LayerNorm(nn.Module):

            def __init__(self, hidden_size, eps=1e-06):
                if False:
                    print('Hello World!')
                '\n                Construct a layernorm module in the T5 style No bias and no subtraction of mean.\n                '
                super().__init__()
                self.weight = nn.Parameter(torch.ones(hidden_size))
                self.variance_epsilon = eps

            def forward(self, hidden_states):
                if False:
                    return 10
                variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
                hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
                if self.weight.dtype in [torch.float16, torch.bfloat16]:
                    hidden_states = hidden_states.to(self.weight.dtype)
                return self.weight * hidden_states
        bs = 256
        seq = 256
        hidden = 1024
        t5_norm = T5LayerNorm(hidden)
        t5_norm_inputs = [(bs, seq, hidden)]
        run_and_compare_activation(self, t5_norm, t5_norm_inputs)

def check(f, t, delta, check_val=True, graph_input=False):
    if False:
        while True:
            i = 10
    if graph_input:
        fx_g = f
    else:
        fx_g = make_fx(f)(t)
    new_graph = fx_graph_cse(fx_g.graph)
    new_g = fx.GraphModule(fx_g, new_graph)
    old_num_nodes = len(fx_g.graph.nodes)
    new_num_nodes = len(new_graph.nodes)
    if delta == -1:
        assert old_num_nodes >= new_num_nodes, f'number of nodes increased {old_num_nodes}, {new_num_nodes}'
    else:
        assert old_num_nodes == new_num_nodes + delta, f'number of nodes not the same {old_num_nodes - delta}, {new_num_nodes}\n {fx_g.graph} \n {new_graph}'
    pass_2_graph = fx_graph_cse(new_graph)
    pass_2_num_nodes = len(pass_2_graph.nodes)
    assert pass_2_num_nodes == new_num_nodes, f'second pass graph has less node {pass_2_num_nodes}, {new_num_nodes}\n {new_graph} \n {pass_2_graph}'
    if check_val:
        true_result = fx_g(t)
        our_result = new_g(t)
        if true_result is None:
            assert our_result is None, f'true result is None, CSE result is {our_result}'
        else:
            assert torch.all(true_result == our_result), f'results are different {true_result}, {our_result}'

class NoChangeTestCase(TestCase):

    def test_nochange(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                i = 10
                return i + 15
            a = x + 1
            b = x + a
            a = x
            d = x + a
            return b + d
        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_empty(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            pass
        t = torch.randn(2, 2)
        check(f, t, 0)

    def test_rand_like(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            a = torch.rand_like(x)
            b = torch.rand_like(x)
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

    def test_rand_n(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            a = torch.randn(4)
            b = torch.randn(4)
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 0, check_val=False)

class ReduceTestCase(TestCase):

    def test_immutable_list_type(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                while True:
                    i = 10
            a = x.sum(dim=1)
            b = x.sum(dim=1)
            c = x.sum()
            d = x.sum()
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_immutable_list_multiple_entries(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            a = x.sum(dim=[0, 1])
            b = x.sum(dim=[0, 1])
            c = x.sum(dim=1)
            d = x.sum(dim=1)
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            a = x.cos()
            b = x.cos()
            c = a + a
            d = b + b
            return c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_2(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                while True:
                    i = 10
            a = x.cos().sin()
            b = x.cos().sin()
            c = a + a
            d = b + b
            return c + d
        t = torch.randn(1)
        check(f, t, 3)

    def test_two_args_default(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=False)
            c = x.sum(dim=1, keepdim=False)
            d = x.sum(dim=1)
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_two_args(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                i = 10
                return i + 15
            a = x.sum(dim=1)
            b = x.sum(dim=1, keepdim=True)
            c = x.sum(dim=1, keepdim=True)
            d = x.sum(dim=1)
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 2)

    def test_simple_multiple_same_ops(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                while True:
                    i = 10
            a = x.sum()
            b = x.sum()
            c = x.sum()
            d = x.sum()
            return a + b + c + d
        t = torch.randn(2, 2)
        check(f, t, 3)

    def test_nested_immutable_list_type(self):
        if False:
            i = 10
            return i + 15

        def f(x):
            if False:
                return 10
            a = torch.cat((x, x))
            b = torch.cat((x, x))
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 1)

    def test_kwarg(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                print('Hello World!')
            a = torch.ones_like(x)
            b = torch.ones_like(x)
            return a + b
        t = torch.randn(2, 2)
        check(f, t, 1)

class RandomOpTestCase(TestCase):

    def test_random(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                while True:
                    i = 10
            vals = [x]
            ops = [torch.clone, torch.cos, torch.tanh, torch.nn.functional.gelu]
            for _ in range(100):
                new_val = random.choice(ops)(random.choice(vals))
                vals.append(new_val)
            return vals[-1]
        fx_g = fx.symbolic_trace(f)
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
        t = torch.randn(2, 2)
        for _ in range(30):
            check(fx_g, t, -1, graph_input=True)
if __name__ == '__main__':
    run_tests()