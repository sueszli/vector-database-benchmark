import functools
import pickle
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.codecache import AsyncCompile, FxGraphCachePickler, FxGraphHashDetails, TensorMetadata, TensorMetadataAndValues
from torch.testing._internal.common_cuda import SM80OrLater
from torch.testing._internal.common_device_type import largeTensorTest
from torch.testing._internal.common_utils import instantiate_parametrized_tests, parametrize
from torch.testing._internal.inductor_utils import HAS_CUDA
from torch.utils._triton import has_triton
HAS_TRITON = has_triton()
requires_cuda = functools.partial(unittest.skipIf, not HAS_CUDA, 'requires cuda')
requires_triton = functools.partial(unittest.skipIf, not HAS_TRITON, 'requires triton')

class MyModel(torch.nn.Module):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10)

    def forward(self, inp):
        if False:
            i = 10
            return i + 15
        return self.fc1(inp)

def _run_codecache_test(start_method):
    if False:
        for i in range(10):
            print('nop')
    torch._inductor.config.worker_start_method = start_method
    torch._inductor.config.compile_threads = 16
    AsyncCompile.warm_pool()
    model = MyModel().cuda()
    model = torch.compile(model)
    inp = torch.rand(10, 10).cuda()
    model(inp).sum().backward()

@requires_cuda()
def test_codecache_spawn():
    if False:
        while True:
            i = 10
    _run_codecache_test('spawn')

@requires_cuda()
def test_codecache_fork():
    if False:
        while True:
            i = 10
    _run_codecache_test('fork')

class MyModelConv2d(torch.nn.Module):

    def __init__(self, dim=512):
        if False:
            while True:
                i = 10
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = torch.nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)

    def forward(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.conv1(x)
        torch._dynamo.graph_break()
        x = self.conv2(x)
        return x

@instantiate_parametrized_tests
class TestFxGraphCache(TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            return 10
        cls.tmpdir = tempfile.TemporaryDirectory()
        cls.cache_dir_patch = patch('torch._inductor.codecache.cache_dir')
        cls.cache_dir_patch.start().return_value = cls.tmpdir.name

    @classmethod
    def tearDownClass(cls):
        if False:
            while True:
                i = 10
        cls.cache_dir_patch.stop()
        cls.tmpdir.cleanup()

    def setUp(self):
        if False:
            while True:
                i = 10
        super().setUp()
        counters.clear()

    @requires_triton()
    @config.patch({'fx_graph_cache': True})
    @parametrize('device', ('cuda', 'cpu'))
    @parametrize('dtype', (torch.float32, torch.bfloat16))
    @parametrize('dynamic', (False, True))
    def test_cache_load_function(self, device, dtype, dynamic):
        if False:
            for i in range(10):
                print('nop')
        '\n        Verify that we can populate and load functions from the cache.\n        '
        if device == 'cuda' and (not HAS_CUDA):
            raise unittest.SkipTest('requires CUDA')
        if device == 'cuda' and dtype == torch.bfloat16 and (not SM80OrLater):
            raise unittest.SkipTest('requires SM80 or later')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return (x * 2, y @ y)
        a = torch.rand(25, dtype=dtype, device=device)
        b = torch.rand(5, 5, dtype=dtype, device=device)
        c = a.view(5, 5)
        compiled_fn = torch.compile(fn, dynamic=dynamic)
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 1)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 0)
        torch._dynamo.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 1)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 1)
        torch._dynamo.reset()
        self.assertEqual(fn(a, c), compiled_fn(a, c))
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 2)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 1)

    @requires_triton()
    @config.patch({'fx_graph_cache': True})
    @parametrize('device', ('cuda', 'cpu'))
    @parametrize('dtype', (torch.float32, torch.float64))
    @parametrize('dynamic', (False, True))
    def test_cache_load_model(self, device, dtype, dynamic):
        if False:
            return 10
        '\n        Verify that we can populate and load models from the cache.\n        '
        if device == 'cuda' and (not HAS_CUDA):
            raise unittest.SkipTest('requires CUDA')

        def fn(mod, x):
            if False:
                return 10
            mod.zero_grad()
            mod(x).sum().backward()
            return [p.grad for p in mod.parameters()]
        compiled_fn = torch.compile(fn, dynamic=dynamic)
        mod = MyModelConv2d().to(device=device, dtype=dtype)
        inp = torch.randn(2, 3, 16, 16, device=device, dtype=dtype)
        counters.clear()
        grads1 = compiled_fn(mod, inp)
        self.assertGreater(counters['inductor']['fxgraph_cache_miss'], 0)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 0)
        counters.clear()
        torch._dynamo.reset()
        grads2 = compiled_fn(mod, inp)
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 0)
        self.assertGreater(counters['inductor']['fxgraph_cache_hit'], 0)
        self.assertEqual(grads1, grads2)

    @largeTensorTest('64GB', device='cuda')
    @config.patch({'fx_graph_cache': True})
    @parametrize('device', ('cuda',))
    @parametrize('dtype', (torch.float16, torch.bfloat16))
    def test_cache_load_with_guards_int32_bounds(self, device, dtype):
        if False:
            print('Hello World!')
        '\n        Test caching the same graph, but under conditions that introduce guards\n        for tensor sizes < int32.\n        '
        if device == 'cuda' and (not HAS_CUDA):
            raise unittest.SkipTest('requires CUDA')
        if device == 'cuda' and dtype == torch.bfloat16 and (not SM80OrLater):
            raise unittest.SkipTest('requires SM80 or later')

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return (x + x, y + y)
        compiled_fn = torch.compile(fn, dynamic=True)
        shapes = (((5, 6), (7, 8)), ((5, 6), (47000, 47001)), ((47000, 47001), (5, 6)))
        for (a_shape, b_shape) in shapes:
            a = torch.rand(a_shape, device=device, dtype=dtype)
            b = torch.rand(b_shape, device=device, dtype=dtype)
            counters.clear()
            res1 = compiled_fn(a, b)
            self.assertGreater(counters['inductor']['fxgraph_cache_miss'], 0)
            self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 0)
            counters.clear()
            torch._dynamo.reset()
            res2 = compiled_fn(a, b)
            self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 0)
            self.assertGreater(counters['inductor']['fxgraph_cache_hit'], 0)
            self.assertEqual(res1, res2)

    @config.patch({'fx_graph_cache': True})
    @parametrize('device', ('cuda', 'cpu'))
    @parametrize('dtype', (torch.float32, torch.bfloat16))
    def test_cache_load_with_guards_static_bounds(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test caching the same graph, but under conditions that introduce guards\n        for static bounds.\n        '
        if device == 'cuda' and (not HAS_CUDA):
            raise unittest.SkipTest('requires CUDA')
        if device == 'cuda' and dtype == torch.bfloat16 and (not SM80OrLater):
            raise unittest.SkipTest('requires SM80 or later')

        def fn(x):
            if False:
                print('Hello World!')
            return torch.nn.functional.adaptive_avg_pool2d(x, [5, 7])
        compiled_fn = torch.compile(fn, dynamic=True)
        shapes = ((1, 64, 8, 9), (1, 64, 9, 10), (1, 64, 10, 11))
        for shape in shapes:
            x = torch.rand(shape, device=device, dtype=dtype)
            counters.clear()
            res1 = compiled_fn(x)
            self.assertGreater(counters['inductor']['fxgraph_cache_miss'], 0)
            self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 0)
            counters.clear()
            torch._dynamo.reset()
            res2 = compiled_fn(x)
            self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 0)
            self.assertGreater(counters['inductor']['fxgraph_cache_hit'], 0)
            self.assertEqual(res1, res2)

    @config.patch({'fx_graph_cache': True})
    def test_cache_clear(self):
        if False:
            while True:
                i = 10
        '\n        Test clearing the cache.\n        '

        def fn(x, y):
            if False:
                i = 10
                return i + 15
            return (x * y,)
        a = torch.rand(5, 5)
        b = torch.rand(5, 5)
        compiled_fn = torch.compile(fn)
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 1)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 0)
        counters.clear()
        torch._dynamo.reset()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 0)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 1)
        counters.clear()
        torch._dynamo.reset()
        torch._inductor.codecache.FxGraphCache.clear()
        self.assertEqual(fn(a, b), compiled_fn(a, b))
        self.assertEqual(counters['inductor']['fxgraph_cache_miss'], 1)
        self.assertEqual(counters['inductor']['fxgraph_cache_hit'], 0)

class TestFxGraphCacheHashing(TestCase):

    def test_tensor_constants(self):
        if False:
            while True:
                i = 10
        '\n        Test the handling of small vs. large tensor constants.\n        '
        data = FxGraphCachePickler.dumps(torch.tensor(list(range(9))))
        self.assertIsInstance(pickle.loads(data), TensorMetadata)
        data = FxGraphCachePickler.dumps(torch.tensor(list(range(8))))
        self.assertIsInstance(pickle.loads(data), TensorMetadataAndValues)

    def test_hash_fake_tensors(self):
        if False:
            return 10
        '\n        Test hashing (pickling) FakeTensors with various characteristics.\n        '
        with torch._subclasses.FakeTensorMode():
            data = FxGraphCachePickler.dumps(torch.randn(1))
            self.assertIsInstance(pickle.loads(data), TensorMetadata)
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3)), FxGraphCachePickler.dumps(torch.randn(3)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3)), FxGraphCachePickler.dumps(torch.randn(4)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3)), FxGraphCachePickler.dumps(torch.randn(3, 3)))
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3, 3)), FxGraphCachePickler.dumps(torch.randn(3, 3)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, 3)), FxGraphCachePickler.dumps(torch.randn(3, 4)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, 3)), FxGraphCachePickler.dumps(torch.randn(4, 3)))
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3, 3)), FxGraphCachePickler.dumps(torch.randn(3, 3).transpose(0, 1).transpose(0, 1)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, 3)), FxGraphCachePickler.dumps(torch.randn(3, 3).transpose(0, 1)))
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3)[1:]), FxGraphCachePickler.dumps(torch.randn(3)[1:]))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3)[1:]), FxGraphCachePickler.dumps(torch.randn(2)))
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)), FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float32)), FxGraphCachePickler.dumps(torch.randn(3, dtype=torch.float64)))
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)), FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, requires_grad=True)), FxGraphCachePickler.dumps(torch.randn(3, requires_grad=False)))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(1, 2, 3, 4)), FxGraphCachePickler.dumps(torch.randn(1, 2, 3, 4).to(memory_format=torch.channels_last)))
            self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3, device='meta')), FxGraphCachePickler.dumps(torch.randn(3, device='meta')))
            self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, device='meta')), FxGraphCachePickler.dumps(torch.randn(3, device='cpu')))
            if HAS_CUDA and torch.cuda.device_count() >= 2:
                self.assertEqual(FxGraphCachePickler.dumps(torch.randn(3, device='cuda:1')), FxGraphCachePickler.dumps(torch.randn(3, device='cuda:1')))
                self.assertNotEqual(FxGraphCachePickler.dumps(torch.randn(3, device='cuda:0')), FxGraphCachePickler.dumps(torch.randn(3, device='cuda:1')))

    def test_hash_kwargs(self):
        if False:
            print('Hello World!')
        '\n        Test the special handling of the kwargs when hashing, i.e.,\n        ordering of the kwargs dict and any set arguments.\n        '
        details1 = FxGraphHashDetails(None, [], {'a': 0, 'z': 1})
        details2 = FxGraphHashDetails(None, [], {'z': 1, 'a': 0})
        self.assertEqual(FxGraphCachePickler.dumps(details1), FxGraphCachePickler.dumps(details2))
        details1 = FxGraphHashDetails(None, [], {'a': 0})
        details2 = FxGraphHashDetails(None, [], {'a': 1})
        self.assertNotEqual(FxGraphCachePickler.dumps(details1), FxGraphCachePickler.dumps(details2))
        set1 = {'a', 'b', 'c', 'd', 'e', 'f', 'g'}
        set2 = set(sorted(set1))
        details1 = FxGraphHashDetails(None, [], {'a': set1})
        details2 = FxGraphHashDetails(None, [], {'a': set2})
        self.assertEqual(FxGraphCachePickler.dumps(details1), FxGraphCachePickler.dumps(details2))
        details1 = FxGraphHashDetails(None, [], {'a': {1, 2, 3}})
        details2 = FxGraphHashDetails(None, [], {'a': {1, 2}})
        self.assertNotEqual(FxGraphCachePickler.dumps(details1), FxGraphCachePickler.dumps(details2))

    def test_hash_config_changes(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Test that different config settings affect hashes.\n        '
        with config.patch({'max_autotune': False}):
            details1 = FxGraphHashDetails(None, [], {})
            details2 = FxGraphHashDetails(None, [], {})
        with config.patch({'max_autotune': True}):
            details3 = FxGraphHashDetails(None, [], {})
        self.assertEqual(FxGraphCachePickler.dumps(details1), FxGraphCachePickler.dumps(details2))
        self.assertNotEqual(FxGraphCachePickler.dumps(details1), FxGraphCachePickler.dumps(details3))
if __name__ == '__main__':
    run_tests()