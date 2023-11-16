from functorch.dim import Tensor, Dim, dims, dimlists, stack, DimensionBindError, DimList
from attn_ft import BertSelfAttention as BertSelfAttentionA, Linear
from attn_positional import BertSelfAttention as BertSelfAttentionB
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_CUDA
from unittest import skip, skipIf
import torch
import gc
from functorch._C import dim as _C
try:
    from torchvision.models import resnet18
except ImportError:
    resnet18 = None
(_test_c, _parse_test, _set_pointwise_optimize) = (_C._test_c, _C._parse_test, _C._set_pointwise_optimize)
from contextlib import contextmanager
from time import perf_counter
measure_perf = False
if measure_perf:
    from torchdim.magic_trace import magic_trace
else:

    @contextmanager
    def magic_trace(*args, **kwargs):
        if False:
            print('Hello World!')
        yield

@contextmanager
def measure(what):
    if False:
        while True:
            i = 10
    b = perf_counter()
    yield
    e = perf_counter()
    print(f'{what}: {e - b:.20f} seconds')

def triu(A):
    if False:
        while True:
            i = 10
    (i, j) = dims()
    a = A[i, j]
    zero = torch.tensor(0, dtype=torch.float)
    return torch.where(i <= j, a, zero).order(i, j)

def gpu_time(lmb, name, r=100):
    if False:
        print('Hello World!')
    b = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    for _ in range(r):
        lmb()
    b.record()
    for _ in range(r):
        lmb()
    e.record()
    e.synchronize()
    elapsed = b.elapsed_time(e)
    print(name, elapsed / r)
    return elapsed / r

class TestMin(TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        gc.disable()
        gc.collect()
        self.interesting = set()
        for o in gc.get_objects():
            if isinstance(o, (torch.Tensor, Dim, Tensor, DimList)):
                self.interesting.add(id(o))
        if 'cuda' in self._testMethodName:
            self.mem_allocated = torch.cuda.memory_allocated()

    def tearDown(self):
        if False:
            while True:
                i = 10
        interesting = []
        for o in gc.get_objects():
            if isinstance(o, (torch.Tensor, Dim, Tensor, DimList)) and id(o) not in self.interesting:
                interesting.append(o)
        extra_memory = 0
        if 'cuda' in self._testMethodName:
            extra_memory += torch.cuda.memory_allocated() - self.mem_allocated
        if extra_memory != 0 or len(interesting) != 0:
            import refcycle
            refcycle.garbage().export_image('garbage.pdf')
        gc.collect()
        assert extra_memory == 0, f'extra cuda memory left allocated: {extra_memory}'
        assert len(interesting) == 0, f'extra torch.Tensor, Dim, or Tensor left allocated: {len(interesting)} objects of types: {[type(t) for t in interesting]}'

    def test_manual_stuff(self):
        if False:
            i = 10
            return i + 15
        A_ = torch.rand(3, 4)
        B_ = torch.rand(4, 5)
        (i, j, k) = dims()
        A = A_[i, k]
        B = B_[k, j]
        C = (A.expand(j) * B.expand(i)).sum(k)
        self.assertTrue(torch.allclose(C.order(i, j), torch.mm(A_, B_)))
        self.assertTrue(torch.allclose(torch.triu(A_, 0), triu(A_)))
        D_ = torch.randint(0, 3, (6,))
        d = dims()
        D = D_[d]
        A.index([i], [D]).order(k, d)

    def attn(self, batch_size=1, sequence_length=4, hidden_size=6, num_attention_heads=3, linear=Linear, device=None, time=False):
        if False:
            print('Hello World!')

        def maybe_to(x):
            if False:
                return 10
            return x if device is None else x.to(device)
        attention_probs_dropout_prob = 0.0
        A = maybe_to(BertSelfAttentionA(hidden_size, num_attention_heads, attention_probs_dropout_prob, linear=linear))
        B = maybe_to(BertSelfAttentionB(hidden_size, num_attention_heads, attention_probs_dropout_prob))
        A.load_state_dict(B.state_dict())
        hidden_state = maybe_to(torch.rand(batch_size, sequence_length, hidden_size))
        b_out = B(hidden_state)
        a_out = A(hidden_state)
        self.assertTrue(torch.allclose(a_out, b_out))
        if time:
            gpu_time(lambda : B(hidden_state), 'positional', r=3)
            gpu_time(lambda : A(hidden_state), 'first_class', r=3)
        for approach in ('relative_key', 'relative_key_query'):
            A = maybe_to(BertSelfAttentionA(hidden_size, num_attention_heads, attention_probs_dropout_prob, approach, sequence_length, linear=linear))
            B = maybe_to(BertSelfAttentionB(hidden_size, num_attention_heads, attention_probs_dropout_prob, approach, sequence_length))
            A.load_state_dict(B.state_dict())
            hidden_state = maybe_to(torch.rand(batch_size, sequence_length, hidden_size))
            b_out = B(hidden_state)
            a_out = A(hidden_state)
            self.assertTrue(torch.allclose(a_out, b_out))
            if time:
                gpu_time(lambda : B(hidden_state), 'positional', r=3)
                gpu_time(lambda : A(hidden_state), 'first_class', r=3)
        A = maybe_to(BertSelfAttentionA(hidden_size, num_attention_heads, attention_probs_dropout_prob, None, None, linear=linear))
        B = maybe_to(BertSelfAttentionB(hidden_size, num_attention_heads, attention_probs_dropout_prob, None, None))
        A.load_state_dict(B.state_dict())
        hidden_state = maybe_to(torch.rand(batch_size, sequence_length, hidden_size))
        past_key_value = (maybe_to(torch.rand(batch_size, num_attention_heads, sequence_length, hidden_size // num_attention_heads)), maybe_to(torch.rand(batch_size, num_attention_heads, sequence_length, hidden_size // num_attention_heads)))
        b_out = B(hidden_state, past_key_value=past_key_value)
        a_out = A(hidden_state, past_key_value=past_key_value)
        self.assertTrue(torch.allclose(a_out, b_out))
        if time:
            gpu_time(lambda : B(hidden_state), 'positional', r=3)
            gpu_time(lambda : A(hidden_state), 'first_class', r=3)

    def test_attn(self):
        if False:
            for i in range(10):
                print('nop')
        self.attn()

    def test_inplace(self):
        if False:
            while True:
                i = 10
        embeddings = torch.zeros(10, 3)
        indices = torch.arange(2) + 1
        values = torch.rand(2, 3)
        (i, n, f) = dims()
        embeddings[indices[i], f] += values[i, f]

    def test_adapt(self):
        if False:
            return 10

        def f():
            if False:
                i = 10
                return i + 15
            (ci, co) = dims()
        for i in range(10):
            f()

    @skipIf(not TEST_CUDA, 'no CUDA')
    def test_attn_cuda(self):
        if False:
            return 10
        self.attn(batch_size=256, hidden_size=768, sequence_length=128, num_attention_heads=12, device='cuda', time=measure_perf, linear=torch.nn.Linear)

    def test_stack(self):
        if False:
            for i in range(10):
                print('nop')
        (i, j, d) = dims()
        A = torch.rand(4, 5)
        r = stack([A[i, j]], d, j)

    def test_max(self):
        if False:
            return 10
        ap = torch.rand(2, 3, 2)
        (i, j, k) = dims()
        a = ap[i, j, k]
        (r, i0) = a.max(dim=k)
        self.assertTrue(torch.allclose(r.order(i, j), ap.max(2)[0]))

    def test_mm(self):
        if False:
            print('Hello World!')
        (i, j, k, q) = dims()
        a = torch.rand(3, 4)
        b = torch.rand(4, 5)
        a_ = a[i, k]
        b_ = b[k, j]
        q.size = 1
        r = (a_.expand(j, q) * b_.expand(i, q)).sum(k).order(q, i, j)

    def test_with_dims_split(self):
        if False:
            while True:
                i = 10
        a = torch.arange(3 * 12).view(3, 12)
        (i, j, k) = dims()
        k.size = 4
        r = a[i, [j, k]]
        x = r.order(i, [j, k])
        self.assertTrue(torch.allclose(a, x))

    def test_hello(self):
        if False:
            print('Hello World!')
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        (i, j, k) = dims()
        r = (A[i, k] * B[k, j]).sum(k).order(i, j)
        assert torch.allclose(r, A @ B)
        assert A.sum() == A[i].sum((0, i))
        assert A.sum() == A[i].sum((-1, i))
        assert torch.allclose(A.sum(), A[i].sum(0, keepdim=True).sum((0, i)))
        assert torch.allclose(A[i].std(i, True), A.std(0, True))
        assert torch.allclose(A[i, k].max(i)[0].order(k), A.max(0)[0])
        assert torch.allclose(A.sort(1)[0], A[i, k].sort(k)[0].order(i, k))
        assert torch.allclose(A[i].renorm(1, i, 7).order(i), A.renorm(1, 0, 7))
        kk = dims()
        k2 = dims()
        assert torch.allclose(A.expand(5, -1, -1), A[i, k].expand(j).order(j, i, k))
        z = dims()
        C = torch.arange(2)
        assert torch.allclose(A[:, 0:2], A[i, k].index(k, C[z]).order(i, z))
        (o, l) = dims()
        o.size = 2
        r = A[i, k].index(k, (o, l))
        assert torch.allclose(r.order(i, o, l), A.view(-1, 2, 2))
        rr = r.index((o, l), k)
        assert torch.allclose(A, rr.order(i, k))
        r = i + k - 1
        r2 = torch.arange(3)[:, None] + torch.arange(4)[None, :] - 1
        assert torch.allclose(r.order(i, k), r2)
        assert torch.allclose(A.T, A[..., k].order(k))
        (a_, b_) = dimlists()
        assert torch.allclose(A[i, a_].order(*a_, i), A.T)
        assert torch.allclose(A[:, a_].order(*a_), A.T)
        assert torch.allclose(A[i, b_, k].order(i, k, *b_), A)
        A[i] + i
        assert torch.allclose((A[i] + i).order(i), A + torch.arange(3)[:, None])
        try:
            A[1, ..., 1, 1]
            raise NotImplementedError()
        except IndexError:
            pass
        (c, d) = dims()
        c.size = 2
        assert torch.allclose(A[i, [c, d]].order(i, c, d), A.view(3, 2, 2))
        assert torch.allclose(A[c + 1, c + 0].order(c), A[torch.arange(2) + 1, torch.arange(2)])
        try:
            A[..., 3, ...]
            raise NotImplementedError()
        except DimensionBindError:
            pass
        C = torch.rand(4, 7)
        (c_, x, y, z) = dims()
        (a, b, c) = C.split((3, 3, 1), dim=1)
        s = dims()
        ref = C.split((3, 3, 1), dim=1)
        t = C[s, c_].split((x, y, z), dim=c_)
        for (a, b, d) in zip(ref, t, (x, y, z)):
            assert torch.allclose(a, b.order(s, d))
        D = torch.rand(3, 4, 5)
        assert torch.allclose(D.transpose(0, 1).flatten(1, 2), D[i, k, j].order((i, j)).order(k))
        r = [id(x) for x in torch.rand_like(A[i, k]).dims]
        assert id(i) in r and id(k) in r
        r = [id(x) for x in torch.nn.functional.dropout(A[i, k]).dims]
        assert id(i) in r and id(k) in r

    def test_simple(self):
        if False:
            print('Hello World!')
        (i, j, k) = dims()
        x = torch.rand(3, 4)
        z = x[i, j]
        z + z + z + z
        z.order(i, j)

    def test_mm_fuse(self):
        if False:
            while True:
                i = 10
        (i, j, k) = dims()
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        C = (A[i, k] * B[k, j]).sum(k).order(i, j)
        assert torch.allclose(C, A @ B)

    def test_time_mm_fuse(self):
        if False:
            while True:
                i = 10
        (i, j, k) = dims()
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        for _ in range(10):
            r0 = A @ B
        for _ in range(10):
            a = A[i, k]
            b = B[k, j]
            r1 = (a * b).sum(k)
        with measure('pp'):
            for _ in range(10000):
                A @ B
        with measure('fc'):
            for _ in range(10000):
                (A[i, k] * B[k, j]).sum(k).order(i, j)
        with magic_trace('f.fxt'):
            for _ in range(10000):
                (A[i, k] * B[k, j]).sum(k).order(i, j)
        with magic_trace('p.fxt'):
            for _ in range(10000):
                A @ B
        assert torch.allclose(r1.order(i, j), r0)

    def test_compare_dims(self):
        if False:
            for i in range(10):
                print('nop')
        (i, j) = dims()
        i.size = 3
        j.size = 4
        i < j

    def test_c(self):
        if False:
            print('Hello World!')
        _test_c()

    def test_seg(self):
        if False:
            for i in range(10):
                print('nop')
        A = torch.rand(3, 4)
        (i, k) = dims()
        i.size = 4
        k.size = 3
        r = i + k - 1

    def test_expand(self):
        if False:
            while True:
                i = 10
        A = torch.rand(3, 4)
        i = dims()
        assert list(A[i].expand(2, 4).order(i).size()) == [3, 2, 4]

    def test_parse(self):
        if False:
            for i in range(10):
                print('nop')
        self.assertEqual(('x', None, None, None), _parse_test(1, 0, 'x'))
        self.assertEqual(('x', None, 'y', None), _parse_test(1, 0, 'x', c='y'))
        self.assertEqual(('x', None, 'y', 'z'), _parse_test(1, 0, 'x', d='z', c='y'))
        self.assertEqual(('x', '4', None, None), _parse_test(2, 0, 'x', b='4'))
        self.assertEqual(('x', 'y', 'z', 'q'), _parse_test(2, 0, 'x', 'y', 'z', 'q'))
        with self.assertRaises(TypeError):
            _parse_test(2, 0, 'x', 'y', 'z', 'q', '5')
        with self.assertRaises(TypeError):
            _parse_test(2, 0, 'x', 'y', b='y')
        with self.assertRaises(TypeError):
            _parse_test(2, 0, 'x', c='y')
        with self.assertRaises(TypeError):
            _parse_test(2, 0, 'x')

    def test_network(self):
        if False:
            for i in range(10):
                print('nop')
        if resnet18 is None:
            self.skipTest('no torchvision')
        rn = resnet18(norm_layer=lambda x: torch.nn.BatchNorm2d(x, track_running_stats=False))
        rn.train()
        img = torch.rand(1, 1, 2, 3, 224, 224)
        imgf = img.view(2, 3, 224, 224)
        (i, j) = dims()
        r = rn(img[i, j])
        r = r.order(i, j).view(2, 1000)
        r2 = rn(imgf)
        assert torch.allclose(r2, r, atol=1e-06)

    def test_dim_args(self):
        if False:
            i = 10
            return i + 15
        a = dimlists()
        assert isinstance(a, DimList)
        a = dims()
        b = dimlists()
        assert isinstance(a, Dim)
        assert isinstance(b, DimList)
        assert str(a) == 'a'
        (a, b) = dims(sizes=[3, 4])
        assert a.size == 3
        assert b.size == 4
        a = dims(sizes=[3])
        b = dimlists(sizes=[4])
        assert len(b) == 4
        a = dims()
        b = dimlists(sizes=[[4, 5]])
        assert b[0].size == 4
        assert b[1].size == 5

    def test_diag(self):
        if False:
            return 10
        i = dims()
        A = torch.rand(4, 4)
        A[i, i]

    def test_softmax_split(self):
        if False:
            return 10
        a = torch.rand(16)
        (g, i) = dims(sizes=[2, None])
        a2 = a[[i, g],]
        (m_b, _) = a2.max(i)
        f_b = torch.exp(a2 - m_b)
        l_b = f_b.sum(i)
        (m, _) = m_b.max(g)
        c = torch.exp(m_b - m)
        f = (c * f_b).order((i, g))
        l = (c * l_b).sum(g)
        assert torch.allclose(f / l, torch.nn.functional.softmax(a, dim=0))

    def test_index(self):
        if False:
            i = 10
            return i + 15
        A = torch.rand(3, 4)
        B = torch.rand(4, 5)
        (i, j, k) = dims()
        (o, l) = dims()
        o.size = 2
        r = A[i, k].index(k, [o, l])
        assert torch.allclose(r.order(i, o, l), A.view(-1, 2, 2))
        rr = r.index([o, l], k)
        assert torch.allclose(A, rr.order(i, k))
        z = dims()
        C = torch.arange(2)
        x = A[i, k].index(k, C[z]).order(i, z)
        assert torch.allclose(A[:, 0:2], x)
        C = torch.rand(3, 4, 5)
        ik = dims()
        assert torch.allclose(C.index((0, 2), ik).order(ik), C.permute(0, 2, 1).reshape(15, 4))

    def test_monkey(self):
        if False:
            while True:
                i = 10
        A = torch.rand(3, 4)
        A[0, 0] = 5
        x = torch.randn(3, 4, 4, 4, 3)
        x_clone1 = x.clone()
        ia = torch.tensor([0, 2, 1])
        ib = torch.tensor([0, 2, 1])
        first_shape = x[:, ia, None, ib, 0].shape
        x_clone1[:, ia, None, ib, 0] = torch.randn(first_shape).to(x_clone1)
        x = torch.autograd.Variable(torch.tensor([]))
        z = torch.autograd.Variable(torch.IntTensor([1, 2, 3]))
        a = [z[2], z[0] + 3]
        x.new(a)

    def test_index_placement(self):
        if False:
            return 10
        A = torch.rand(1, 2, 3, 4)
        (i, j) = dims(sizes=[2, 4])
        a = A[:, i + 0, :, j + 0]
        r = a.order(i, j)
        assert torch.allclose(A.permute(1, 3, 0, 2), r)

    def test_order(self):
        if False:
            print('Hello World!')
        (i, j) = dims()
        A = torch.rand(3, 4, 5)
        assert torch.allclose(A[i].order(1, i), A.permute(2, 0, 1))

    def test_mask(self):
        if False:
            i = 10
            return i + 15
        a = torch.rand(5)
        (i, j) = dims(sizes=[a.size(0), a.size(0)])
        ((i >= j) * a[i]).sum(j).order(i)

    def test_eq(self):
        if False:
            for i in range(10):
                print('nop')
        (i, j) = dims(sizes=[3, 3])
        assert (i == j).sum((i, j)) == 3

    def test_dims_with_size(self):
        if False:
            print('Hello World!')
        x = dims(3)
        assert len(x) == 3 and isinstance(x[0], Dim)

        class Foo:
            pass
        y = Foo()
        (z, y.x, q) = dims(3)
        assert str(z) == 'z'
        assert str(y.x) == 'd1'
        assert str(q) == 'd2'

    def test_dir(self):
        if False:
            print('Hello World!')
        (i, j) = dims(sizes=[3, 3])
        dir(i <= j)

    def test_doc(self):
        if False:
            i = 10
            return i + 15
        assert Tensor.clamp.__doc__ == torch.Tensor.clamp.__doc__

    def test_embed(self):
        if False:
            return 10
        embeddings = torch.rand(8, 32)
        ids = torch.tensor([1, 0, 3, 4])
        values_ = torch.empty(4, 32)
        for batch in range(ids.size(0)):
            for feature in range(embeddings.size(1)):
                values_[batch, feature] = embeddings[ids[batch], feature]
        (batch, feature) = dims(2)
        values = embeddings[ids[batch], feature].order(batch, feature)
        assert torch.allclose(values, values_)

    def test_functorch(self):
        if False:
            while True:
                i = 10
        A = torch.rand(3, 4, 5)
        B = torch.rand(3, 4, 5)
        C = torch.rand(5, 2)
        (i, j) = dims()
        AA = torch.mm(A[i], C)
        BB = torch.mm(B[j], C)
        assert list(torch.mm(AA.T, BB).order(i, j).shape) == [3, 3, 2, 2]

    def test_permute_orig(self):
        if False:
            for i in range(10):
                print('nop')
        d = dims(1)
        t_fc = torch.rand(1, 2, 3, 4)[d]
        assert t_fc.permute(dims=(1, 0, 2)).shape == t_fc.permute(1, 0, 2).shape

    def test_order_keyword(self):
        if False:
            print('Hello World!')
        d = dims(1)
        t = torch.rand(3)[d]
        self.assertRaises(TypeError, lambda : t.order(wrong=3))

    def test_big_split(self):
        if False:
            print('Hello World!')
        total = 0
        l = []
        while total < 6400:
            l.append(torch.randint(2, 10, (1,)).item())
            total += l[-1]
        x = torch.randn(total, 1)
        x.split(l, 0)
skip_functorch_only = ['test_time_mm_fuse', 'test_attn_cuda']

class TestMinFunctorchOnly(TestMin):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        _set_pointwise_optimize(False)

    def tearDown(self):
        if False:
            for i in range(10):
                print('nop')
        _set_pointwise_optimize(True)
        super().tearDown()
for n in skip_functorch_only:
    setattr(TestMinFunctorchOnly, n, skip('skip_functorch_only')(lambda self: None))
if __name__ == '__main__':
    run_tests()