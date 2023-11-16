import unittest
from typing import Dict, Optional
import numpy as np
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from typing import List

class StaticModule:

    def __init__(self, scripted):
        if False:
            return 10
        if hasattr(scripted, '_c'):
            self.static_module = torch._C._jit_to_static_module(scripted._c)
        else:
            self.static_module = torch._C._jit_to_static_module(scripted.graph)

    def __call__(self, *args, **kwargs):
        if False:
            print('Hello World!')
        return self.static_module(*args, **kwargs)

    def benchmark(self, args, kwargs, warmup_runs, main_runs):
        if False:
            while True:
                i = 10
        self.static_module.benchmark(args, kwargs, warmup_runs, main_runs)

    def runAsync(self, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.static_module.runAsync(args, kwargs)

    def benchmark_individual_ops(self, args, kwargs, warmup_runs, main_runs):
        if False:
            while True:
                i = 10
        return self.static_module.benchmark_individual_ops(args, kwargs, warmup_runs, main_runs)

def linear_shim(input: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]=None) -> torch.Tensor:
    if False:
        print('Hello World!')
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    ret = output
    return ret
torch.nn.functional.linear = linear_shim

class MultiHeadAttentionLayer(nn.Module):

    def __init__(self, hid_dim, n_heads, dropout, device):
        if False:
            i = 10
            return i + 15
        super().__init__()
        assert hid_dim % n_heads == 0
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask):
        if False:
            return 10
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        attention = torch.softmax(energy, dim=-1)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return (x, attention)

def create_mlp(ln, sigmoid_layer):
    if False:
        i = 10
        return i + 15
    layers = nn.ModuleList()
    for i in range(0, len(ln) - 1):
        n = ln[i]
        m = ln[i + 1]
        LL = nn.Linear(int(n), int(m), bias=True)
        mean = 0.0
        std_dev = np.sqrt(2 / (m + n))
        W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        std_dev = np.sqrt(1 / m)
        bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
        LL.weight.data = torch.tensor(W, requires_grad=True)
        LL.bias.data = torch.tensor(bt, requires_grad=True)
        layers.append(LL)
        if i == sigmoid_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU())
    with torch.no_grad():
        s = torch.jit.script(torch.nn.Sequential(*layers))
    s.eval()
    return s

def trivial_graph(a, b, c):
    if False:
        while True:
            i = 10
    s = torch.tensor([[3, 3], [3, 3]])
    return a + b * c + s

def elementwise_square_addition(input1, input2):
    if False:
        i = 10
        return i + 15
    return input1 * input1 + input2 * input2

def fork_wait_graph1(input1, input2):
    if False:
        return 10
    fut = torch.jit.fork(elementwise_square_addition, input1, input2)
    return torch.jit.wait(fut)

def fork_wait_graph2(input1, input2):
    if False:
        return 10
    fut = torch.jit.fork(loop_graph, input1, input2, 5)
    return torch.jit.wait(fut)
'\n   graph with multiple fork/wait operations\n   :param input: torch.tensor input to forked subgraph\n   :param iters: number of future/wait pairs to be created\n'

def fork_wait_graph3(input, iters: int):
    if False:
        while True:
            i = 10
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for _ in range(iters):
        futures.append(torch.jit.fork(torch.neg, input))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.sum(torch.stack(results))
'\n   graph with multi-level fork/wait operations\n   :param input: torch.tensor input to forked subgraph\n   :param num_forks: number of top level forks\n   :param num_child_forks: number of child forks per parent fork\n'

def fork_wait_graph4(input, num_forks: int, num_child_forks: int):
    if False:
        print('Hello World!')
    futures: List[torch.jit.Future[torch.Tensor]] = []
    for _ in range(num_forks):
        futures.append(torch.jit.fork(fork_wait_graph3, input, num_child_forks))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.sum(torch.stack(results))

def add_tensor(input1, input2):
    if False:
        for i in range(10):
            print('nop')
    return input1 + input2

def fork_wait_graph_exception(input1, input2):
    if False:
        for i in range(10):
            print('nop')
    fut = torch.jit.fork(add_tensor, input1, input2)
    return torch.jit.wait(fut)

def loop_graph(a, b, iters: int):
    if False:
        i = 10
        return i + 15
    c = a + b * 2
    for i in range(iters):
        c = c + b
        c *= 2
        c -= a
    return c

def output_graph(a, b, c, iters: int):
    if False:
        return 10
    s = torch.tensor([[3, 3], [3, 3]])
    k = a + b * c + s
    d: Dict[int, torch.Tensor] = {}
    for i in range(iters):
        d[i] = k + i
    return d

class SubModule(nn.Module):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.a = 11
        self.b = 2

    def forward(self, x):
        if False:
            print('Hello World!')
        return self.a + self.b + x

class SubModule2(nn.Module):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self.a = 12
        self.b = 2

    def forward(self, x):
        if False:
            i = 10
            return i + 15
        self.b = 30
        return self.a + self.b + x

class TestModule(nn.Module):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self.sub1 = SubModule()
        self.sub2 = SubModule2()
        self.a = 3
        self.b = 4

    def forward(self, x):
        if False:
            return 10
        self.b = 20
        return self.sub1(x) + self.a + self.b + self.sub2(x)

class TestStaticModule(TestCase):
    """
    Test Case: To test simple fork/wait operation in a graph
    fork is called on simple addition operation on input tensors
    """

    def test_fork_wait_1(self):
        if False:
            for i in range(10):
                print('nop')
        inp1 = torch.ones(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph1)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module(inp1, inp2)
        torch.testing.assert_close(output_test, output_ref)
    '\n    Test Case: To test simple fork/wait operation with\n    StaticRuntime runAsync API returning future\n    '

    def test_fork_wait_1_async(self):
        if False:
            while True:
                i = 10
        inp1 = torch.ones(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph1)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module.runAsync((inp1, inp2), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)
    '\n    Test Case: To test fork/wait operation in a graph on\n    a loop subgraph performing mix of operations\n    '

    def test_fork_wait_2(self):
        if False:
            while True:
                i = 10
        inp1 = torch.randn(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph2)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module(inp1, inp2)
        torch.testing.assert_close(output_test, output_ref)
    '\n    Test Case: To test fork/wait operation on a loop\n    subgraph with StaticRuntime runAsync API returning future\n    '

    def test_fork_wait_2_async(self):
        if False:
            i = 10
            return i + 15
        inp1 = torch.randn(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph2)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module.runAsync((inp1, inp2), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)
    '\n    Test Case: To test fork/wait operation in a graph on\n    having multiple fork/wait operations\n    '

    def test_fork_wait_3(self):
        if False:
            i = 10
            return i + 15
        input = torch.ones(3, 3)
        num_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph3)
        output_ref = torch_graph(input, num_forks)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module(input, num_forks)
        torch.testing.assert_close(output_test, output_ref)
    '\n    Test Case: To test fork/wait operation in a graph with\n    multiple fork/wait operations on runAsync API returning future\n    '

    def test_fork_wait_3_async(self):
        if False:
            for i in range(10):
                print('nop')
        input = torch.ones(3, 3)
        num_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph3)
        output_ref = torch_graph(input, num_forks)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module.runAsync((input, num_forks), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)
    '\n    Test Case: To test fork/wait operation in a graph on\n    multiple nested fork/wait operations\n    '

    @unittest.skip('Broken test: https://github.com/pytorch/pytorch/issues/109782')
    def test_fork_wait_4(self):
        if False:
            while True:
                i = 10
        input = torch.ones(3, 3)
        num_forks = 10
        num_child_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph4)
        static_runtime_module = StaticModule(torch_graph)
        output_ref = torch_graph(input, num_forks, num_child_forks)
        output_test = static_runtime_module(input, num_forks, num_child_forks)
        torch.testing.assert_close(output_test, output_ref)
    '\n    Test Case: To test fork/wait operation in a graph with multiple\n    nested fork/wait operations on runAsync API returning future\n    '

    @unittest.skip('Broken test: https://github.com/pytorch/pytorch/issues/109782')
    def test_fork_wait_4_async(self):
        if False:
            for i in range(10):
                print('nop')
        input = torch.ones(3, 3)
        num_forks = 10
        num_child_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph4)
        static_runtime_module = StaticModule(torch_graph)
        output_ref = torch_graph(input, num_forks, num_child_forks)
        output_test = static_runtime_module.runAsync((input, num_forks, num_child_forks), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)
    '\n    Test Case: To test exception handling in fork/wait\n    operation. Add.Tensor op is called for tensors with\n    non-matching dims on the forked subgraph and the\n    exception raised by subgraph is set on future returned\n    by prim::fork to parent graph. Returned exception is\n    checked for substring expected_error_msg as declared below\n    '

    def test_fork_wait_exception(self):
        if False:
            return 10
        input1 = torch.randn(4, 7)
        input2 = torch.randn(4, 5)
        torch_graph = torch.jit.script(fork_wait_graph_exception)
        try:
            static_runtime_module = StaticModule(torch_graph)
            output_test = static_runtime_module(input1, input2)
        except Exception as error:
            expected_error_msg = 'The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 1'
            if str(error).find(expected_error_msg) == -1:
                raise RuntimeError(f'Tried execution of add.Tensors with incompatible shape. Exception raised by forked runtime execution does not contain expected substring: "{expected_error_msg}"') from error
    '\n    Test Case: To test exception handling in fork/wait\n    operation with runAsync API. Add.Tensor op is called for\n    tensors with non-matching dims on the forked subgraph\n    and the exception raised by subgraph is set on future returned\n    by prim::fork to parent graph. Returned exception is\n    checked for substring expected_error_msg as declared below\n    '

    def test_fork_wait_exception_async(self):
        if False:
            return 10
        input1 = torch.randn(4, 7)
        input2 = torch.randn(4, 5)
        torch_graph = torch.jit.script(fork_wait_graph_exception)
        try:
            static_runtime_module = StaticModule(torch_graph)
            output_test = static_runtime_module.runAsync((input1, input2), {})
        except Exception as error:
            expected_error_msg = 'The size of tensor a (7) must match the size of tensor b (5) at non-singleton dimension 1'
            if str(error).find(expected_error_msg) == -1:
                raise RuntimeError(f'Tried execution of add.Tensors with incompatible shape. Exception raised by forked runtime execution does not contain expected substring: "{expected_error_msg}"') from error

    def test_multihead_attention_layer(self):
        if False:
            print('Hello World!')
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device('cpu')
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)
        attention.eval()
        attention = torch.jit.script(attention)
        attention.eval()
        o_ref = attention(src, src, src, src_mask)
        attention_a = StaticModule(attention)
        o_test = attention_a(src, src, src, src_mask)
        o_test_kw = attention_a(src, src, value=src, mask=src_mask)
        for (a, b) in zip(o_ref, o_test):
            torch.testing.assert_close(a, b)
        for (a, b) in zip(o_ref, o_test_kw):
            torch.testing.assert_close(a, b)

    def test_multihead_attention_layer_benchmark(self):
        if False:
            while True:
                i = 10
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device('cpu')
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)
        attention.eval()
        attention = torch.jit.script(attention)
        attention_a = StaticModule(attention)
        attention_a.benchmark([src, src, src, src_mask], {}, 2, 2)
        metrics = attention_a.benchmark_individual_ops([src, src, src, src_mask], {}, 2, 2)

    def test_mlp(self):
        if False:
            return 10
        ln_bot = [512, 512, 64]
        sigmoid_bot = -1
        ln_top = [100, 1024, 1024, 1024, 1]
        sigmoid_top = 3
        bot_l = create_mlp(ln_bot, sigmoid_bot)
        bot_l_acc = StaticModule(bot_l)
        top_l = create_mlp(ln_top, sigmoid_top)
        top_l_acc = StaticModule(top_l)
        with torch.no_grad():
            bot_inp = torch.randn(2048, 512)
            top_inp = torch.randn(2048, 100)
        ref_bot = bot_l(bot_inp)
        acc_bot = bot_l_acc(bot_inp)
        torch.testing.assert_close(acc_bot, ref_bot)
        ref_top = top_l(top_inp)
        acc_top = top_l_acc(top_inp)
        torch.testing.assert_close(acc_top, ref_top)
        for _ in range(5):
            with torch.no_grad():
                bot_inp = torch.randn(2048, 512)
                top_inp = torch.randn(2048, 100)
            ref_bot = bot_l(bot_inp)
            acc_bot = bot_l_acc(bot_inp)
            torch.testing.assert_close(acc_bot, ref_bot)
            ref_top = top_l(top_inp)
            acc_top = top_l_acc(top_inp)
            torch.testing.assert_close(acc_top, ref_top)

    def test_trivial_graph(self):
        if False:
            for i in range(10):
                print('nop')
        s = torch.full((2, 2), 2)
        tg = torch.jit.script(trivial_graph)
        o_ref = tg(s, s, s)
        tg_a = StaticModule(tg)
        o_test = tg_a(s, s, s)
        torch.testing.assert_close(o_ref, o_test)

    def test_leaky_relu(self):
        if False:
            print('Hello World!')
        s = torch.randn(5, 5)
        tg = torch.jit.script(nn.LeakyReLU(0.1))
        o_ref = tg(s)
        tg_a = StaticModule(tg)
        o_test = tg_a(s)
        torch.testing.assert_close(o_ref, o_test)

    def test_attr(self):
        if False:
            print('Hello World!')
        '\n        TorchScript IR of TestModule() after freezing:\n        graph(%self : __torch__.test_static_runtime.___torch_mangle_0.TestModule,\n              %x.1 : Tensor):\n            %18 : int = prim::Constant[value=30]()\n            %30 : int = prim::Constant[value=13]()\n            %3 : int = prim::Constant[value=20]()\n            %2 : int = prim::Constant[value=1]()\n            %self.sub2.a : int = prim::Constant[value=12]()\n            %self.a : int = prim::Constant[value=3]()\n            = prim::SetAttr[name="b"](%self, %3)\n            %17 : Tensor = aten::add(%x.1, %30, %2)\n            %7 : Tensor = aten::add(%17, %self.a, %2)\n            %b.1 : int = prim::GetAttr[name="b"](%self)\n            %9 : Tensor = aten::add(%7, %b.1, %2)\n            %sub2 : __torch__.test_static_runtime.___torch_mangle_2.SubModule2 = prim::GetAttr[name="sub2"](%self)\n            = prim::SetAttr[name="b"](%sub2, %18)\n            %b : int = prim::GetAttr[name="b"](%sub2)\n            %22 : int = aten::add(%self.sub2.a, %b)\n            %23 : Tensor = aten::add(%x.1, %22, %2)\n            %12 : Tensor = aten::add(%9, %23, %2)\n            return (%12)\n        '
        m = TestModule()
        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)
        ms = torch.jit.script(m)
        sm = StaticModule(ms)
        output_sm = sm(input)
        torch.testing.assert_close(output_s, output_sm)
        sm.benchmark([input], {}, 2, 2)
        sm.benchmark_individual_ops([input], {}, 2, 2)
        sm.benchmark([], {'x': input}, 2, 2)
        sm.benchmark_individual_ops([], {'x': input}, 2, 2)

    @unittest.skip('Temporarily disabled')
    def test_fusion_trivial_graph(self):
        if False:
            return 10
        s = torch.full((2, 2), 2)
        tg = torch.jit.script(trivial_graph)
        o_ref = tg(s, s, s)
        torch._C._fuse_to_static_module(tg.graph)
        assert 'StaticSubgraph' in str(tg.graph)
        o_test = tg(s, s, s)
        torch.testing.assert_close(o_ref, o_test)

    @unittest.skip('Temporarily disabled')
    def test_fusion_multihead_attention_layer(self):
        if False:
            return 10
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device('cpu')
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)
        attention.eval()
        attention = torch.jit.script(attention)
        attention.eval()
        o_ref = attention(src, src, src, src_mask)
        torch._C._fuse_to_static_module(attention._c)
        o_test = attention(src, src, src, src_mask)
        for (a, b) in zip(o_ref, o_test):
            torch.testing.assert_close(a, b)

    @unittest.skip('Temporarily disabled')
    def test_fusion_loop(self):
        if False:
            print('Hello World!')
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        c = 4
        lg = torch.jit.script(loop_graph)
        o_ref = lg(a, b, c)
        torch._C._fuse_to_static_module(lg.graph)
        assert 'StaticSubgraph' in str(lg.graph)
        o_test = lg(a, b, c)
        torch.testing.assert_close(o_ref, o_test)

    @unittest.skip('Temporarily disabled')
    def test_fusion_outputs(self):
        if False:
            i = 10
            return i + 15
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        c = 4
        og = torch.jit.script(output_graph)
        o_ref = og(a, b, b, c)
        torch._C._fuse_to_static_module(og.graph)
        assert 'StaticSubgraph' in str(og.graph)
        o_test = og(a, b, b, c)
        for i in o_ref.keys():
            torch.testing.assert_close(o_ref[i], o_test[i])

    def test_create_object(self):
        if False:
            i = 10
            return i + 15

        class Foo:

            def __init__(self, x: torch.Tensor) -> None:
                if False:
                    return 10
                self.x = x

        class Mod(torch.nn.Module):

            def __init__(self) -> None:
                if False:
                    return 10
                super().__init__()

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                if False:
                    while True:
                        i = 10
                foo = Foo(y)
                return y * foo.x
        mod = torch.jit.script(Mod()).eval()
        y = torch.randn((1,))
        expected = mod(y)
        static_mod = StaticModule(torch.jit.freeze(mod))
        actual = static_mod(y)
        self.assertEqual(expected, actual)
if __name__ == '__main__':
    run_tests()