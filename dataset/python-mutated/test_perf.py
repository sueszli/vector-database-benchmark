import contextlib
from unittest.mock import patch
import functorch
import torch
import torch._inductor.config as config
from torch._inductor import metrics
from torch._inductor.compile_fx import compile_fx, count_bytes_inner
from torch.testing._internal.common_utils import IS_WINDOWS, skipIfRocm, TestCase as TorchTestCase
from torch.testing._internal.triton_utils import HAS_CUDA, requires_cuda
if HAS_CUDA:
    from torch.testing._internal.triton_utils import add_kernel
aten = torch.ops.aten

def count_bytes_inductor(gm, example_inputs):
    if False:
        for i in range(10):
            print('nop')
    return compile_fx(gm, example_inputs, inner_compile=count_bytes_inner)
if not IS_WINDOWS:

    @torch._dynamo.optimize(count_bytes_inductor)
    def f(x):
        if False:
            print('Hello World!')
        return torch.cat([x, x.cos()])
else:

    def f(x):
        if False:
            while True:
                i = 10
        return torch.cat([x, x.cos()])

def count_numel(f, *args):
    if False:
        print('Hello World!')
    '\n    Assumes all inputs are fp32\n    '
    metrics.reset()
    torch._dynamo.optimize(count_bytes_inductor)(f)(*args)
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)

def count_numel_train(f, *args):
    if False:
        return 10
    '\n    Assumes all inputs are fp32\n    '
    metrics.reset()
    f = torch._dynamo.optimize(count_bytes_inductor)(f)
    out = f(*args)
    res = 0
    for o in out:
        res += o.mean()
    res.backward()
    print(metrics.nodes_num_elem)
    return str(metrics.num_bytes_accessed // 4)
DEVICE = 'cuda'

def T(*size, dtype=torch.float32, device=DEVICE, grad=False):
    if False:
        for i in range(10):
            print('nop')
    return torch.randn(size, dtype=dtype, device=device, requires_grad=grad)

def TI(*size, mx=10, dtype=torch.int32, device=DEVICE):
    if False:
        return 10
    return torch.randint(0, mx, size, dtype=dtype, device=device)

class TestCase(TorchTestCase):
    device = DEVICE
    pass

class NumBytesMetricTests(TestCase):
    """
    Primarily used for sanity testing that the num_bytes_accessed metrics is correct.
    """

    def test_pointwise(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                print('Hello World!')
            return x.cos()
        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), '20')

        def f(x, y):
            if False:
                i = 10
                return i + 15
            return x + y
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '30')

        def f(x, y):
            if False:
                return 10
            return x + y
        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '210')

        def f(x):
            if False:
                i = 10
                return i + 15
            return x + x
        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), '20')

        def f(x):
            if False:
                print('Hello World!')
            return x + x.t()
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '200')

        def f(a, b, c):
            if False:
                print('Hello World!')
            return (a.cos(), b.sin() + c.sin())
        inp = (T(10), T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '50')

    def test_reduction(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.sum(dim=1)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '110')

        def f(x):
            if False:
                while True:
                    i = 10
            return x.sum(dim=0)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '110')

    def test_extern(self):
        if False:
            while True:
                i = 10

        def f(x):
            if False:
                return 10
            return torch.mm(x, x)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '200')

        def f(a, b):
            if False:
                return 10
            return torch.mm(a, b)
        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '300')

        def f(x):
            if False:
                for i in range(10):
                    print('nop')
            x = x.cos()
            x = torch.mm(x, x)
            x = x.cos()
            return x
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '600')

        def f(x):
            if False:
                i = 10
                return i + 15
            a = x.cos()
            b = x.sin()
            x = torch.mm(a, b)
            return x
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '600')

    def test_cat(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                while True:
                    i = 10
            return torch.cat([a.sin(), b.sin()])
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '40')

        def f(a, b):
            if False:
                print('Hello World!')
            return torch.cat([a, b])
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '40')

        def f(a, b):
            if False:
                while True:
                    i = 10
            return torch.cat([a.cos(), b])
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '40')

        def f(a):
            if False:
                i = 10
                return i + 15
            return torch.cat([a.cos(), a.sin()])
        inp = (T(10),)
        self.assertExpectedInline(count_numel(f, *inp), '30')

        def f(a, b):
            if False:
                return 10
            return torch.cat([torch.mm(a, a), b.sin()])
        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '400')

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return torch.cat((a + 1, b + 2, c + 3)) + 10
        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '600')

        def f(a, b, c, d, e):
            if False:
                return 10
            return torch.cat((a + 1, b + 2, c + 3, d + 4, e + 5)) + 10
        inp = [T(10, 10) for _ in range(5)]
        self.assertExpectedInline(count_numel(f, *inp), '2000')

        def f(a, b):
            if False:
                return 10
            return torch.cat([a.sum(dim=0), b.sum(dim=0)]) + 10
        inp = [T(10, 10, 10), T(10, 10, 10)]
        self.assertExpectedInline(count_numel(f, *inp), '2600')

    def test_index(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                return 10
            return a[b]
        inp = (T(10), TI(10, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), '30')

class FusionTests(TestCase):
    """
    Tests that things can be fused into a single kernel
    """

    def test_horizontal_reduction_pointwise(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                print('Hello World!')
            b = a.sum(dim=1)
            c = a.cos()
            return (b, c)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '210')

    def test_horizontal_reduction_reduction(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                while True:
                    i = 10
            b = a.sum(dim=1)
            c = a.amax(dim=1)
            return (b, c)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '120')

    def test_horizontal_reduction_pointwise2(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            c = a.sum(dim=1)
            b = b.cos()
            return b + c
        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '120')

    def test_horizontal_reduction_outer_pointwise(self):
        if False:
            i = 10
            return i + 15

        def f(a, b):
            if False:
                return 10
            c = a.sum(dim=0)
            b = b.cos()
            return b + c
        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '120')

    def test_horizontal_sum_pw_broadcast(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                while True:
                    i = 10
            a = a.sum(dim=1, keepdim=True)
            b = b.cos()
            return a * b
        inp = (T(10, 10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '210')

    def test_vertical_sum_pw(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                while True:
                    i = 10
            a = a.cos()
            a = a.sum(dim=1)
            return a.cos()
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '110')

    def test_norm_chain(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            b = a.sum(dim=1, keepdim=True)
            a = a * b
            return a
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '200')

    def test_softmax_inner(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                print('Hello World!')
            return torch.softmax(a, dim=1)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '200')

    def test_layer_norm(self):
        if False:
            for i in range(10):
                print('nop')
        mod = torch.nn.LayerNorm(10, device=self.device)

        def f(x):
            if False:
                while True:
                    i = 10
            return mod(x)
        inp = (T(10, 10),)
        with torch.no_grad():
            self.assertExpectedInline(count_numel(f, *inp), '220')

    def test_double_softmax(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x):
            if False:
                return 10
            x = torch.softmax(x, dim=1)
            x = torch.softmax(x, dim=1)
            return x
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '200')

    def test_softmax_backward(self):
        if False:
            for i in range(10):
                print('nop')

        def f(grad_out, out):
            if False:
                for i in range(10):
                    print('nop')
            return aten._softmax_backward_data(grad_out, out, 1, torch.float32)
        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '300')

    def test_neighbor(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                while True:
                    i = 10
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)
        inp = (T(10, 1, 4), T(1, 10, 4))
        self.assertExpectedInline(count_numel(f, *inp), '90')

    def test_factory_reduction(self):
        if False:
            while True:
                i = 10

        def f():
            if False:
                while True:
                    i = 10
            a = torch.ones(10, device=self.device)
            b = torch.ones(10, 10, device=self.device)
            return (a + b).sum(dim=-1)
        inp = ()
        self.assertExpectedInline(count_numel(f, *inp), '10')

    def test_index_pointwise(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                return 10
            return a[b].cos()
        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), '320')

    def test_index_reduction(self):
        if False:
            return 10

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return a[b].cos().sum(dim=1)
        inp = (T(10, 10), TI(20, mx=10))
        self.assertExpectedInline(count_numel(f, *inp), '140')

    def test_mutation_fusion(self):
        if False:
            while True:
                i = 10

        def f(a, b, c):
            if False:
                i = 10
                return i + 15
            a0 = a.add(c)
            b0 = b.add(a0)
            b.copy_(b0)
            a.copy_(a0)
        inp = (T(10, 10), T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '500')

    def test_reduction_pointwise_multi_level_reduction(self):
        if False:
            i = 10
            return i + 15
        hidden_size = 4096

        def f(x, scale, amax_keep_dim):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.nn.functional.layer_norm(x.to(dtype=torch.float), [hidden_size], weight=None, bias=None, eps=1e-05)
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            x_scaled = x * scale
            y = torch.nn.functional.sigmoid(x_scaled)
            return (y, amax)
        inp = (T(4, 2048, hidden_size, dtype=torch.float), T(1, dtype=torch.float))
        expected_amax_keep_dim_numel = 1 + 4 * 2048 * hidden_size * 3 + 4 * 2048 * 4 + 1
        self.assertGreaterAlmostEqual(count_numel(f, *inp, True), str(expected_amax_keep_dim_numel))
        expected_amax_no_keep_dim_numel = 1 + 4 * 2048 * hidden_size * 2 + 4 * 2048 * 2 + 1
        self.assertExpectedInline(count_numel(f, *inp, False), str(expected_amax_no_keep_dim_numel))

    def test_pointwise_multi_level_reduction(self):
        if False:
            print('Hello World!')
        hidden_size = 4096

        def f(x, scale, amax_keep_dim):
            if False:
                return 10
            x = x * 1.1
            amax = torch.amax(torch.abs(x), keepdim=amax_keep_dim)
            x_scaled = x * scale
            y = torch.nn.functional.sigmoid(x_scaled)
            return (y, amax)
        inp = (T(4, 2048, hidden_size, dtype=torch.float), T(1, dtype=torch.float))
        compiled_f = torch.compile(f)
        compiled_f(*inp, True)
        expected_numel = 1 + 4 * 2048 * hidden_size * 3 + 1
        actual_numel_amax_keep_dim = count_numel(f, *inp, True)
        actual_numel_amax_no_keep_dim = count_numel(f, *inp, False)
        self.assertEqual(actual_numel_amax_keep_dim, actual_numel_amax_no_keep_dim)
        self.assertGreaterAlmostEqual(actual_numel_amax_keep_dim, str(expected_numel))

class SchedulerFusionTests(TestCase):
    """
    Testing the fusion group creation heuristic (i.e. cases where we can't fuse
    everything into a single kernel)
    Disables inductor rematerialization for easier reasoning of tests.
    """

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        super().setUpClass()
        cls._stack = contextlib.ExitStack()
        cls._stack.enter_context(patch.object(config, 'realize_bytes_threshold', 0))

    @classmethod
    def tearDownClass(cls):
        if False:
            i = 10
            return i + 15
        cls._stack.close()
        super().tearDownClass()

    @patch.object(config, 'pattern_matcher', False)
    def test_fusion_choice1(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                return 10
            c = a.cos()
            d = torch.mm(c, c)
            e = c.cos()
            return d + e
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '700')

    @patch.object(config, 'pattern_matcher', False)
    def test_fusion_choice2(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                print('Hello World!')
            c = a.cos()
            d = torch.mm(c, c)
            e = c.sum(dim=1)
            f = d + e
            return f
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '620')

    @patch.object(config, 'pattern_matcher', False)
    def test_fusion_choice3(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                print('Hello World!')
            c = a.cos()
            d = torch.mm(c, c)
            e = c + a
            f = d + e
            return (f, e)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '800')

class TilingTests(TestCase):

    def test_tiling_simple(self):
        if False:
            while True:
                i = 10

        def f(a, b):
            if False:
                print('Hello World!')
            return a + b.t()
        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '300')

        def f(a, b):
            if False:
                print('Hello World!')
            return a.t() + b
        inp = (T(10, 10), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '300')

    def test_tiling_three(self):
        if False:
            return 10

        def f(a, b, c):
            if False:
                for i in range(10):
                    print('nop')
            return a + b.permute(1, 2, 0) + c.permute(2, 0, 1)
        inp = (T(10, 10, 10), T(10, 10, 10), T(10, 10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '4000')

class MinCutPartitioningTests(TestCase):

    def test_partitioning_full_remat(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                i = 10
                return i + 15
            return x.cos().cos().cos()
        inp = (T(10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), '50')

    def test_partitioning_partial_remat(self):
        if False:
            while True:
                i = 10

        def f(a, b, c, d):
            if False:
                while True:
                    i = 10
            x = a + b + c + d
            return x.cos().cos()
        inp = (T(10, grad=True), T(10, grad=True), T(10, grad=True), T(10, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), '90')

    def test_partitioning_dtype(self):
        if False:
            return 10

        def f(x):
            if False:
                return 10
            return (x < 0) * x
        inp = (T(100, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), '450')

    @patch.object(functorch.compile.config, 'max_dist_from_bw', 1000)
    def test_partitioning_unremat_bw(self):
        if False:
            print('Hello World!')

        def f(x):
            if False:
                return 10
            return torch.mm(x, x.new_ones(x.shape)).tanh().tanh()
        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), '1300')

    @patch.object(config, 'pattern_matcher', False)
    def test_partitioning_unremat_bw2(self):
        if False:
            return 10

        def f(a):
            if False:
                i = 10
                return i + 15
            a = torch.mm(a, a)
            a = a + 1
            b = a + 2
            c = torch.mm(a, b)
            return c
        inp = (T(10, 10, grad=True),)
        self.assertExpectedInline(count_numel_train(f, *inp), '2600')

    def test_partitioning_keops(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return (a * b).cos().sum(dim=1)
        inp = (T(20, 1, grad=True), T(1, 20, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), '220')

    def test_partitioning_cat(self):
        if False:
            print('Hello World!')

        def f(a, b):
            if False:
                while True:
                    i = 10
            a = torch.tanh(a)
            return torch.cat([a, b])
        inp = (T(10, grad=True), T(10, grad=True))
        self.assertExpectedInline(count_numel_train(f, *inp), '70')

def unfusible(x):
    if False:
        while True:
            i = 10
    return aten.special_bessel_j0(x)

class NoopTests(TestCase):

    def test_noop_clones(self):
        if False:
            print('Hello World!')

        def f(a):
            if False:
                i = 10
                return i + 15
            b = a.clone()
            b = unfusible(b)
            return b
        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), '20')

        def f(a):
            if False:
                print('Hello World!')
            b = a.clone()
            c = unfusible(b)
            return (b, c)
        self.assertExpectedInline(count_numel(f, inp), '40')

    def test_noop_slice_scatter(self):
        if False:
            while True:
                i = 10

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = aten.slice_scatter(a, a)
            c = unfusible(b)
            return c
        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), '20')

    def test_noop_dtype_conversion(self):
        if False:
            return 10

        def f(a):
            if False:
                while True:
                    i = 10
            b = torch.ops.prims.convert_element_type(a, torch.float32)
            c = unfusible(b)
            return c
        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), '20')

    def test_noop_device_conversion(self):
        if False:
            return 10

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = torch.ops.prims.device_put(a, 'cuda')
            c = unfusible(b)
            return c
        inp = T(10)
        self.assertExpectedInline(count_numel(f, inp), '20')

    def test_noop_int_ops(self):
        if False:
            i = 10
            return i + 15

        def f1(a):
            if False:
                while True:
                    i = 10
            b = torch.ceil(a)
            c = unfusible(b)
            return c

        def f2(a):
            if False:
                print('Hello World!')
            d = torch.floor(a)
            e = unfusible(d)
            return e

        def f3(a):
            if False:
                while True:
                    i = 10
            f = torch.round(a)
            g = unfusible(f)
            return g

        def f4(a):
            if False:
                while True:
                    i = 10
            f = torch.pow(a, 1)
            g = unfusible(f)
            return g
        inp = TI(10)
        self.assertExpectedInline(count_numel(f1, inp), '20')
        self.assertExpectedInline(count_numel(f2, inp), '20')
        self.assertExpectedInline(count_numel(f3, inp), '20')
        self.assertExpectedInline(count_numel(f4, inp), '20')

    def test_noop_cat(self):
        if False:
            i = 10
            return i + 15

        def f1(a):
            if False:
                i = 10
                return i + 15
            b = torch.cat([a])
            return unfusible(b)
        inp = T(10)
        self.assertExpectedInline(count_numel(f1, inp), '20')

        def f2(a):
            if False:
                i = 10
                return i + 15
            b = torch.cat([a])
            c = torch.cat([b])
            return c
        self.assertExpectedInline(count_numel(f2, inp), '20')

class InplacingTests(TestCase):

    def test_inplace_scatter(self):
        if False:
            return 10

        def f(a, b):
            if False:
                for i in range(10):
                    print('nop')
            a = a.cos()
            a[b] = 1
            return a
        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), '26')

        def f(a, b):
            if False:
                while True:
                    i = 10
            out = aten.index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)
        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), '6')

        def f(a, b):
            if False:
                i = 10
                return i + 15
            out = aten._unsafe_index_put(a, (b,), torch.tensor(1.0))
            return a.copy_(out)
        inp = (T(10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), '6')

    def test_inplace_scatter_noop_view(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a, b):
            if False:
                return 10
            a[:, b] = 1
            return a
        inp = (T(10, 10), TI(2, mx=5))
        self.assertExpectedInline(count_numel(f, *inp), '42')

    @requires_cuda()
    @skipIfRocm
    def test_inplace_triton_kernel_v1(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x: torch.Tensor, y: torch.Tensor):
            if False:
                return 10
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '40')

    @requires_cuda()
    @skipIfRocm
    def test_inplace_triton_kernel_v2(self):
        if False:
            i = 10
            return i + 15

        def f(x: torch.Tensor, y: torch.Tensor):
            if False:
                i = 10
                return i + 15
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            tmp = torch.add(x, 1)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return (output, tmp)
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '60')

    @requires_cuda()
    @skipIfRocm
    def test_inplace_triton_kernel_v3(self):
        if False:
            while True:
                i = 10

        def f(x: torch.Tensor, y: torch.Tensor):
            if False:
                print('Hello World!')
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            x.add_(1)
            return output
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '90')

    @requires_cuda()
    @skipIfRocm
    def test_inplace_triton_kernel_v4(self):
        if False:
            i = 10
            return i + 15

        def f(x: torch.Tensor, y: torch.Tensor):
            if False:
                i = 10
                return i + 15
            x_view = x.view(-1)
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            output2 = x_view.mul(2)
            return (output, output2)
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '60')

    @requires_cuda()
    @skipIfRocm
    def test_inplace_triton_kernel_v5(self):
        if False:
            for i in range(10):
                print('nop')

        def f(x: torch.Tensor, y: torch.Tensor):
            if False:
                for i in range(10):
                    print('nop')
            x_view = x.view(-1)
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            x_view.mul_(2)
            return output
        inp = (T(10), T(10))
        self.assertExpectedInline(count_numel(f, *inp), '90')

    @requires_cuda()
    @skipIfRocm
    def test_inplace_triton_kernel_v6(self):
        if False:
            while True:
                i = 10

        def f(x: torch.Tensor, y: torch.Tensor):
            if False:
                while True:
                    i = 10
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements,)
            add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=16)
            return output
        t = T(10)
        inp = (t, t.view(-1))
        self.assertExpectedInline(count_numel(f, *inp), '150')

    def test_inplace_randperm_scatter(self):
        if False:
            while True:
                i = 10

        def scaled_index_add(x, y, scale_y):
            if False:
                for i in range(10):
                    print('nop')
            index = torch.randperm(x.shape[0], device=x.device)[:y.shape[0]]
            out = x.index_add_(dim=0, source=y * scale_y, index=index)
            return out
        inp = (T(10, 10), T(5, 10), T(10))
        self.assertExpectedInline(count_numel(scaled_index_add, *inp), '240')

class WouldBeNiceIfItWorked:

    def test_horizontal(self):
        if False:
            i = 10
            return i + 15

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            b = a.sum(dim=0)
            c = a.cos()
            return (b, c)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '210')

    def test_softmax_outer(self):
        if False:
            for i in range(10):
                print('nop')

        def f(a):
            if False:
                for i in range(10):
                    print('nop')
            return torch.softmax(a, dim=0)
        inp = (T(10, 10),)
        self.assertExpectedInline(count_numel(f, *inp), '200')

    @patch.object(config, 'realize_bytes_threshold', 0)
    def test_fusion_choice4(self):
        if False:
            print('Hello World!')

        def f(a, b, b2):
            if False:
                print('Hello World!')
            c = a + b
            d = torch.mm(c, c)
            e = c + b + b2
            f = d + e + b2
            return (f, e)
        inp = (T(10, 10), T(10, 10, dtype=torch.float16), T(10, 10))
        self.assertExpectedInline(count_numel(f, *inp), '1000')

    def test_neighbor(self):
        if False:
            return 10

        def f(a, b):
            if False:
                while True:
                    i = 10
            return ((a - b) ** 2).sum(dim=-1).amax(dim=1)
        inp = (T(10, 1, 8), T(1, 10, 8))
        self.assertExpectedInline(count_numel(f, *inp), '170')
if __name__ == '__main__':
    from torch._dynamo.test_case import run_tests
    if HAS_CUDA:
        run_tests(needs='filelock')