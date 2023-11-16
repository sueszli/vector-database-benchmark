import torch
import numpy as np
import random
from torch import nan
from itertools import permutations, product
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import all_types, all_types_and, floating_types_and, integral_types
from torch.testing._internal.common_utils import TestCase, run_tests, slowTest
from torch.testing._internal.common_device_type import instantiate_device_type_tests, dtypes, onlyNativeDeviceTypes, onlyCUDA, dtypesIfCUDA, dtypesIfCPU, onlyCPU, largeTensorTest
SIZE = 100

class TestSortAndSelect(TestCase):

    def assertIsOrdered(self, order, x, mxx, ixx, task):
        if False:
            print('Hello World!')
        SIZE = x.size(1)
        if order == 'descending':

            def check_order(a, b):
                if False:
                    return 10
                return ((a != a) | (a >= b)).all().item()
        elif order == 'ascending':

            def check_order(a, b):
                if False:
                    print('Hello World!')
                return ((b != b) | (a <= b)).all().item()
        else:
            error(f'unknown order "{order}", must be "ascending" or "descending"')
        are_ordered = True
        for k in range(1, SIZE):
            self.assertTrue(check_order(mxx[:, k - 1], mxx[:, k]), f'torch.sort ({order}) values unordered for {task}')
        seen = set()
        indicesCorrect = True
        size0 = x.size(0)
        size = x.size(x.dim() - 1)
        x = x.tolist()
        mxx = mxx.tolist()
        ixx = ixx.tolist()
        for k in range(size0):
            seen.clear()
            for j in range(size):
                self.assertEqual(x[k][ixx[k][j]], mxx[k][j], msg=f'torch.sort ({order}) indices wrong for {task}')
                seen.add(ixx[k][j])
            self.assertEqual(len(seen), size)

    def test_sort(self, device):
        if False:
            while True:
                i = 10
        for SIZE in (4, 2049):
            x = torch.rand(4, SIZE, device=device)
            (res1val, res1ind) = torch.sort(x)
            y = x.clone()
            y_inds = torch.tensor((), dtype=torch.int64, device=device)
            torch.sort(y, out=(y, y_inds))
            (x_vals, x_inds) = torch.sort(x)
            self.assertEqual(x_vals, y)
            self.assertEqual(x_inds, y_inds)
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x), res1ind)
            self.assertEqual(x.argsort(), res1ind)
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random')
            self.assertEqual(torch.sort(torch.tensor((50, 40, 30, 20, 10), device=device))[0], torch.tensor((10, 20, 30, 40, 50), device=device), atol=0, rtol=0)
            x = torch.floor(torch.rand(4, SIZE, device=device) * 10)
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with duplicate keys')
            x = torch.rand(4, SIZE, device=device)
            (res1val, res1ind) = torch.sort(x, x.dim() - 1, True)
            res2val = torch.tensor((), device=device)
            res2ind = torch.tensor((), device=device, dtype=torch.long)
            torch.sort(x, x.dim() - 1, True, out=(res2val, res2ind))
            self.assertEqual(res1val, res2val, atol=0, rtol=0)
            self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
            self.assertEqual(torch.argsort(x, x.dim() - 1, True), res1ind)
            self.assertEqual(x.argsort(x.dim() - 1, True), res1ind)
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random')
            self.assertEqual(torch.sort(torch.tensor((10, 20, 30, 40, 50), device=device), 0, True)[0], torch.tensor((50, 40, 30, 20, 10), device=device), atol=0, rtol=0)
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random with duplicate keys')
            x = torch.tensor([1, 10, 2, 2, 3, 7, 7, 8, 9, 9] * 3)
            self.assertEqual(torch.argsort(x, stable=True), torch.sort(x, stable=True).indices)
            self.assertEqual(torch.argsort(x, stable=False), torch.sort(x, stable=False).indices)
            self.assertEqual(torch.argsort(x), torch.sort(x).indices)
            x = torch.rand(4, SIZE, device=device)
            x[1][2] = float('NaN')
            x[3][0] = float('NaN')
            torch.sort(x, out=(res2val, res2ind))
            self.assertIsOrdered('ascending', x, res2val, res2ind, 'random with NaNs')
            torch.sort(x, out=(res2val, res2ind), descending=True)
            self.assertIsOrdered('descending', x, res2val, res2ind, 'random with NaNs')

    @onlyCUDA
    def test_sort_large_slice(self, device):
        if False:
            print('Hello World!')
        x = torch.randn(4, 1024000, device=device)
        (res1val, res1ind) = torch.sort(x, stable=True)
        torch.cuda.synchronize()
        (res1val_cpu, res1ind_cpu) = torch.sort(x.cpu(), stable=True)
        self.assertEqual(res1val, res1val_cpu.cuda())
        self.assertEqual(res1ind, res1ind_cpu.cuda())
        (res1val, res1ind) = torch.sort(x, descending=True, stable=True)
        torch.cuda.synchronize()
        (res1val_cpu, res1ind_cpu) = torch.sort(x.cpu(), descending=True, stable=True)
        self.assertEqual(res1val, res1val_cpu.cuda())
        self.assertEqual(res1ind, res1ind_cpu.cuda())

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_stable_sort(self, device, dtype):
        if False:
            print('Hello World!')
        sizes = (100, 1000, 10000)
        for ncopies in sizes:
            x = torch.tensor([0, 1] * ncopies, dtype=dtype, device=device)
            (_, idx) = x.sort(stable=True)
            self.assertEqual(idx[:ncopies], torch.arange(start=0, end=2 * ncopies, step=2, device=device))
            self.assertEqual(idx[ncopies:], torch.arange(start=1, end=2 * ncopies, step=2, device=device))

    @onlyCUDA
    @dtypes(torch.uint8)
    @largeTensorTest('200GB')
    def test_sort_large(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        t0 = torch.randperm(8192, device=device).to(dtype)
        t = t0.view(1, 8192).expand(2 ** 18 + 1, -1).contiguous()
        (v, i) = t.sort()
        del t
        (iv, im) = i.var_mean(dim=0)
        del i
        (vv, vm) = v.var_mean(dim=0)
        del v
        self.assertEqual(vv, torch.zeros_like(vv))
        self.assertEqual(iv, torch.zeros_like(iv))
        self.assertEqual(vm, torch.arange(255, dtype=dtype, device=device))
        self.assertEqual(im, t0.sort().indices)

    @dtypes(torch.float32)
    def test_sort_restride(self, device, dtype):
        if False:
            print('Hello World!')
        tensor = torch.randn((3, 5), dtype=dtype, device=device)[:, 0]
        values = torch.tensor(0, dtype=dtype, device=device)
        indices = torch.tensor(0, dtype=torch.long, device=device)
        torch.sort(tensor, out=(values, indices))
        self.assertEqual(values.stride(), (1,))
        self.assertEqual(indices.stride(), (1,))
        self.assertEqual(tensor[indices], values)

    def _test_sort_discontiguous(self, device, dtype):
        if False:
            print('Hello World!')
        sizes = (5, 7, 2049)
        for shape in permutations(sizes):
            for perm in permutations((0, 1, 2)):
                for dim in range(3):
                    t = torch.randn(shape, device=device, dtype=dtype).permute(perm)
                    r1 = t.sort(dim=dim)
                    r2 = t.contiguous().sort(dim=dim)
                    self.assertEqual(r1, r2)
                    n = t.size(dim)
                    self.assertTrue((r1.values.narrow(dim, 1, n - 1) >= r1.values.narrow(dim, 0, n - 1)).all())
                    self.assertTrue((t.unsqueeze(-1).transpose(dim, -1) == r1.values.unsqueeze(-1)).any(dim=dim).any(dim=-1).all())
                    if self.device_type == 'cuda':
                        self.assertEqual(r1.values.stride(), t.stride())
                        self.assertEqual(r1.indices.stride(), t.stride())

    @onlyCUDA
    @dtypes(torch.float32)
    def test_sort_discontiguous(self, device, dtype):
        if False:
            print('Hello World!')
        self._test_sort_discontiguous(device, dtype)

    @slowTest
    @onlyCPU
    @dtypes(torch.float32)
    def test_sort_discontiguous_slow(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        self._test_sort_discontiguous(device, dtype)

    @dtypes(torch.float32)
    def test_sort_1d_output_discontiguous(self, device, dtype):
        if False:
            while True:
                i = 10
        tensor = torch.randn(12, device=device, dtype=dtype)[:6]
        values = torch.empty_like(tensor)[::2]
        indices = torch.empty(18, device=device, dtype=torch.long)[::3]
        torch.sort(tensor, out=(values, indices))
        (values_cont, indices_cont) = tensor.sort()
        self.assertEqual(indices, indices_cont)
        self.assertEqual(values, values_cont)

    @slowTest
    @onlyCPU
    @dtypes(*integral_types())
    def test_sort_1d_parallel(self, device, dtype):
        if False:
            i = 10
            return i + 15
        low = 0 if dtype == torch.uint8 else -128
        tensor = torch.randint(low=low, high=127, size=(100000,), device=device, dtype=dtype)
        (vals, _) = torch.sort(tensor, stable=True)
        self.assertEqual(True, torch.all(vals[:-1] <= vals[1:]))

    @dtypes(torch.float32)
    def test_topk_1d_output_discontiguous(self, device, dtype):
        if False:
            print('Hello World!')
        tensor = torch.randn(12, device=device, dtype=dtype)
        values = torch.empty_like(tensor)[::2]
        indices = torch.empty(18, device=device, dtype=torch.long)[::3]
        for sorted in (True, False):
            torch.topk(tensor, 6, sorted=sorted, out=(values, indices))
            (values_cont, indices_cont) = tensor.topk(6, sorted=sorted)
            self.assertEqual(indices, indices_cont)
            self.assertEqual(values, values_cont)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_stable_sort_against_numpy(self, device, dtype):
        if False:
            while True:
                i = 10
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            inf = float('inf')
            neg_inf = -float('inf')
            nan = float('nan')
        else:
            if dtype != torch.bool:
                inf = torch.iinfo(dtype).max
                neg_inf = torch.iinfo(dtype).min
            else:
                inf = True
                neg_inf = ~inf
            nan = inf

        def generate_samples():
            if False:
                return 10
            from itertools import chain, combinations
            for sizes in [(1025,), (10000,)]:
                size = sizes[0]
                yield (torch.tensor([0, 1] * size, dtype=dtype, device=device), 0)
            if self.device_type == 'cuda':
                return
            yield (torch.tensor([0, 1] * 100, dtype=dtype, device=device), 0)

            def repeated_index_fill(t, dim, idxs, vals):
                if False:
                    return 10
                res = t
                for (idx, val) in zip(idxs, vals):
                    res = res.index_fill(dim, idx, val)
                return res
            for sizes in [(1, 10), (10, 1), (10, 10), (10, 10, 10)]:
                size = min(*sizes)
                x = (torch.randn(*sizes, device=device) * size).to(dtype)
                yield (x, 0)
                n_fill_vals = 3
                for dim in range(len(sizes)):
                    idxs = (torch.randint(high=size, size=(size // 10,)) for i in range(n_fill_vals))
                    vals = (inf, neg_inf, nan)
                    subsets = chain.from_iterable((combinations(list(zip(idxs, vals)), r) for r in range(1, n_fill_vals + 1)))
                    for subset in subsets:
                        (idxs_subset, vals_subset) = zip(*subset)
                        yield (repeated_index_fill(x, dim, idxs_subset, vals_subset), dim)
        for (sample, dim) in generate_samples():
            (_, idx_torch) = sample.sort(dim=dim, stable=True)
            if dtype is torch.bfloat16:
                sample_numpy = sample.float().cpu().numpy()
            else:
                sample_numpy = sample.cpu().numpy()
            idx_numpy = np.argsort(sample_numpy, axis=dim, kind='stable')
            self.assertEqual(idx_torch, idx_numpy)

    @dtypes(*all_types_and(torch.half, torch.bfloat16))
    def test_msort(self, device, dtype):
        if False:
            i = 10
            return i + 15

        def test(shape):
            if False:
                print('Hello World!')
            tensor = make_tensor(shape, dtype=dtype, device=device, low=-9, high=9)
            if tensor.size() != torch.Size([]):
                if dtype is torch.bfloat16:
                    expected = torch.from_numpy(np.msort(tensor.float().cpu().numpy())).bfloat16()
                else:
                    expected = torch.from_numpy(np.msort(tensor.cpu().numpy()))
            else:
                expected = tensor
            result = torch.msort(tensor)
            self.assertEqual(result, expected)
            out = torch.empty_like(result)
            torch.msort(tensor, out=out)
            self.assertEqual(out, expected)
        shapes = ([], [0], [20], [1, 20], [30, 30], [10, 20, 30])
        for shape in shapes:
            test(shape)

    @dtypes(torch.float)
    def test_sort_expanded_tensor(self, device, dtype):
        if False:
            while True:
                i = 10
        data = torch.scalar_tensor(True, device=device, dtype=dtype)
        data = data.expand([1, 1, 1])
        ref = torch.Tensor([[[True]]])
        out = torch.sort(data, stable=True, dim=1, descending=True)
        expected = torch.sort(ref, stable=True, dim=1, descending=True)
        self.assertEqual(out, expected)
        data = torch.randn(4, 1, 10, device=device, dtype=dtype)
        data = data.expand([4, 8, 10])
        ref = data.contiguous()
        out = torch.sort(data, stable=True, dim=1, descending=True)
        expected = torch.sort(ref, stable=True, dim=1, descending=True)
        self.assertEqual(out, expected)

    def test_topk(self, device):
        if False:
            return 10

        def topKViaSort(t, k, dim, dir):
            if False:
                i = 10
                return i + 15
            (sorted, indices) = t.sort(dim, dir)
            return (sorted.narrow(dim, 0, k), indices.narrow(dim, 0, k))

        def compareTensors(t, res1, ind1, res2, ind2, dim):
            if False:
                for i in range(10):
                    print('nop')
            self.assertEqual(res1, res2, atol=0, rtol=0)
            if not ind1.eq(ind2).all():
                vals = t.gather(dim, ind2)
                self.assertEqual(res1, vals, atol=0, rtol=0)

        def compare(t, k, dim, dir):
            if False:
                while True:
                    i = 10
            (topKVal, topKInd) = t.topk(k, dim, dir, True)
            (sortKVal, sortKInd) = topKViaSort(t, k, dim, dir)
            compareTensors(t, sortKVal, sortKInd, topKVal, topKInd, dim)
        t = torch.rand(random.randint(1, SIZE), random.randint(1, SIZE), random.randint(1, SIZE), device=device)
        for _kTries in range(3):
            for _dimTries in range(3):
                for transpose in (True, False):
                    for dir in (True, False):
                        testTensor = t
                        if transpose:
                            dim1 = random.randrange(t.ndimension())
                            dim2 = dim1
                            while dim1 == dim2:
                                dim2 = random.randrange(t.ndimension())
                            testTensor = t.transpose(dim1, dim2)
                        dim = random.randrange(testTensor.ndimension())
                        k = random.randint(1, testTensor.size(dim))
                        compare(testTensor, k, dim, dir)
        t = torch.randn((2, 100000), device=device)
        compare(t, 2000, 1, True)
        compare(t, 2000, 1, False)
        t = torch.randn((2, 10000), device=device)
        compare(t, 2000, 1, True)
        compare(t, 2000, 1, False)

    def test_topk_arguments(self, device):
        if False:
            while True:
                i = 10
        q = torch.randn(10, 2, 10, device=device)
        self.assertRaises(TypeError, lambda : q.topk(4, True))

    def test_unique_dim(self, device):
        if False:
            while True:
                i = 10
        self.assertFalse(hasattr(torch, 'unique_dim'))

        def run_test(device, dtype):
            if False:
                for i in range(10):
                    print('nop')
            x = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]]], dtype=dtype, device=device)
            x_empty = torch.empty(5, 0, dtype=dtype, device=device)
            x_ill_formed_empty = torch.empty(5, 0, 0, dtype=dtype, device=device)
            x_ill_formed_empty_another = torch.empty(5, 0, 5, dtype=dtype, device=device)
            if dtype in floating_types_and(torch.float16, torch.bfloat16):
                x_nan = torch.tensor([float('nan'), 0, 0, float('nan'), float('nan'), 1], dtype=dtype, device=device)
            expected_unique_dim0 = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]]], dtype=dtype, device=device)
            expected_inverse_dim0 = torch.tensor([0, 0])
            expected_counts_dim0 = torch.tensor([2])
            expected_unique_dim1 = torch.tensor([[[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]], [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]], dtype=dtype, device=device)
            expected_unique_dim1_bool = torch.tensor([[[False, True], [True, True]], [[False, True], [True, True]]], dtype=torch.bool, device=device)
            expected_inverse_dim1 = torch.tensor([1, 0, 2, 0])
            expected_inverse_dim1_bool = torch.tensor([1, 0, 1, 0])
            expected_counts_dim1 = torch.tensor([2, 1, 1])
            expected_counts_dim1_bool = torch.tensor([2, 2])
            expected_unique_dim2 = torch.tensor([[[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0], [2.0, 1.0], [0.0, 1.0]]], dtype=dtype, device=device)
            expected_inverse_dim2 = torch.tensor([0, 1])
            expected_counts_dim2 = torch.tensor([1, 1])
            expected_unique_empty = torch.empty(5, 0, dtype=dtype, device=device)
            expected_inverse_empty = torch.tensor([], dtype=torch.long, device=device)
            expected_counts_empty = torch.tensor([], dtype=torch.long, device=device)
            if dtype in floating_types_and(torch.float16, torch.bfloat16):
                expected_unique_nan = torch.tensor([float('nan'), 0, float('nan'), float('nan'), 1], dtype=dtype, device=device)
                expected_inverse_nan = torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.long, device=device)
                expected_counts_nan = torch.tensor([1, 2, 1, 1, 1], dtype=torch.long, device=device)
            x_unique = torch.unique(x, dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            (x_unique, x_inverse) = torch.unique(x, return_inverse=True, dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            self.assertEqual(expected_inverse_dim0, x_inverse)
            (x_unique, x_counts) = torch.unique(x, return_inverse=False, return_counts=True, dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            self.assertEqual(expected_counts_dim0, x_counts)
            (x_unique, x_inverse, x_counts) = torch.unique(x, return_inverse=True, return_counts=True, dim=0)
            self.assertEqual(expected_unique_dim0, x_unique)
            self.assertEqual(expected_inverse_dim0, x_inverse)
            self.assertEqual(expected_counts_dim0, x_counts)
            x_unique = torch.unique(x, dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
            (x_unique, x_inverse) = torch.unique(x, return_inverse=True, dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
                self.assertEqual(expected_inverse_dim1_bool, x_inverse)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
                self.assertEqual(expected_inverse_dim1, x_inverse)
            (x_unique, x_counts) = torch.unique(x, return_inverse=False, return_counts=True, dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
                self.assertEqual(expected_counts_dim1_bool, x_counts)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
                self.assertEqual(expected_counts_dim1, x_counts)
            (x_unique, x_inverse, x_counts) = torch.unique(x, return_inverse=True, return_counts=True, dim=1)
            if x.dtype == torch.bool:
                self.assertEqual(expected_unique_dim1_bool, x_unique)
                self.assertEqual(expected_inverse_dim1_bool, x_inverse)
                self.assertEqual(expected_counts_dim1_bool, x_counts)
            else:
                self.assertEqual(expected_unique_dim1, x_unique)
                self.assertEqual(expected_inverse_dim1, x_inverse)
                self.assertEqual(expected_counts_dim1, x_counts)
            x_unique = torch.unique(x, dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            (x_unique, x_inverse) = torch.unique(x, return_inverse=True, dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            self.assertEqual(expected_inverse_dim2, x_inverse)
            (x_unique, x_counts) = torch.unique(x, return_inverse=False, return_counts=True, dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            self.assertEqual(expected_counts_dim2, x_counts)
            (x_unique, x_inverse, x_counts) = torch.unique(x, return_inverse=True, return_counts=True, dim=2)
            self.assertEqual(expected_unique_dim2, x_unique)
            self.assertEqual(expected_inverse_dim2, x_inverse)
            self.assertEqual(expected_counts_dim2, x_counts)
            (x_unique, x_inverse, x_counts) = torch.unique(x_empty, return_inverse=True, return_counts=True, dim=1)
            self.assertEqual(expected_unique_empty, x_unique)
            self.assertEqual(expected_inverse_empty, x_inverse)
            self.assertEqual(expected_counts_empty, x_counts)
            if dtype in floating_types_and(torch.float16, torch.bfloat16):
                (x_unique, x_inverse, x_counts) = torch.unique(x_nan, return_inverse=True, return_counts=True, dim=0)
                self.assertEqual(expected_unique_nan, x_unique)
                self.assertEqual(expected_inverse_nan, x_inverse)
                self.assertEqual(expected_counts_nan, x_counts)
            with self.assertRaises(RuntimeError):
                torch.unique(x_ill_formed_empty, return_inverse=True, return_counts=True, dim=1)
            with self.assertRaises(RuntimeError):
                torch.unique(x_ill_formed_empty_another, return_inverse=True, return_counts=True, dim=2)
            y = torch.tensor([[0, 1], [0, 1], [0, 1], [1, 2], [1, 2], [3, 4], [0, 1], [0, 1], [3, 4], [1, 2]], dtype=dtype, device=device)
            if dtype in floating_types_and(torch.float16, torch.bfloat16):
                y_nan = torch.tensor([float('nan'), 0, 0, float('nan'), float('nan'), 1], dtype=dtype, device=device)
            expected_y_unique = torch.tensor([[0, 1], [1, 2], [3, 4], [0, 1], [3, 4], [1, 2]], dtype=dtype, device=device)
            expected_y_inverse = torch.tensor([0, 0, 0, 1, 1, 2, 3, 3, 4, 5], dtype=torch.int64, device=device)
            expected_y_counts = torch.tensor([3, 2, 1, 2, 1, 1], dtype=torch.int64, device=device)
            expected_y_inverse_bool = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 3, 3], dtype=torch.int64, device=device)
            expected_y_counts_bool = torch.tensor([3, 3, 2, 2], dtype=torch.int64, device=device)
            if dtype in floating_types_and(torch.float16, torch.bfloat16):
                expected_y_unique_nan = torch.tensor([float('nan'), 0, float('nan'), float('nan'), 1], dtype=dtype, device=device)
                expected_y_inverse_nan = torch.tensor([0, 1, 1, 2, 3, 4], dtype=torch.long, device=device)
                expected_y_counts_nan = torch.tensor([1, 2, 1, 1, 1], dtype=torch.long, device=device)
            (y_unique, y_inverse, y_counts) = torch.unique_consecutive(y, return_inverse=True, return_counts=True, dim=0)
            if x.dtype == torch.bool:
                self.assertEqual(expected_y_inverse_bool, y_inverse)
                self.assertEqual(expected_y_counts_bool, y_counts)
            else:
                self.assertEqual(expected_y_inverse, y_inverse)
                self.assertEqual(expected_y_counts, y_counts)
            if dtype in floating_types_and(torch.float16, torch.bfloat16):
                (y_unique, y_inverse, y_counts) = torch.unique_consecutive(y_nan, return_inverse=True, return_counts=True, dim=0)
                self.assertEqual(expected_y_unique_nan, y_unique)
                self.assertEqual(expected_y_inverse_nan, y_inverse)
                self.assertEqual(expected_y_counts_nan, y_counts)
            x = torch.tensor([[[[1, 0, 1, 0, 1, 1], [0, 1, 1, 0, 1, 1]], [[0, 1, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0]]], [[[0, 1, 0, 1, 1, 1], [0, 1, 1, 0, 1, 1]], [[0, 0, 1, 1, 0, 1], [1, 1, 0, 0, 0, 0]]]], dtype=dtype, device=device)
            xn = x.cpu().numpy()
            for d in range(x.dim()):
                t = torch.unique(x, dim=d)
                n = np.unique(xn, axis=d)
                self.assertEqual(t.cpu().numpy(), n)
        run_test(device, torch.float)
        run_test(device, torch.double)
        run_test(device, torch.long)
        run_test(device, torch.uint8)
        run_test(device, torch.bool)

    @onlyCUDA
    def test_topk_noncontiguous_gpu(self, device):
        if False:
            return 10
        single_block_t = torch.randn(20, device=device)[::2]
        multi_block_t = torch.randn(20000, device=device)[::2]
        sort_t = torch.randn(200000, device=device)[::2]
        for t in (single_block_t, multi_block_t, sort_t):
            for k in (5, 2000, 10000):
                if k >= t.shape[0]:
                    continue
                (top1, idx1) = t.topk(k)
                (top2, idx2) = t.contiguous().topk(k)
                self.assertEqual(top1, top2)
                self.assertEqual(idx1, idx2)

    def _test_topk_dtype(self, device, dtype, integral, size):
        if False:
            for i in range(10):
                print('nop')
        if integral:
            a = torch.randint(torch.iinfo(dtype).min, torch.iinfo(dtype).max, size=(size,), dtype=dtype, device=device)
        else:
            a = torch.randn(size=(size,), dtype=dtype, device=device)
        sort_topk = a.sort()[0][-(size // 2):].flip(0)
        topk = a.topk(size // 2)
        self.assertEqual(sort_topk, topk[0])
        self.assertEqual(sort_topk, a[topk[1]])

    @dtypes(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)
    def test_topk_integral(self, device, dtype):
        if False:
            return 10
        small = 10
        large = 4096
        verylarge = 8192
        for curr_size in (small, large, verylarge):
            self._test_topk_dtype(device, dtype, True, curr_size)

    @dtypes(torch.bfloat16, torch.half)
    def test_topk_lower_precision(self, device, dtype):
        if False:
            return 10
        small = 10
        large = 4096
        verylarge = 8192
        for curr_size in (small, large, verylarge):
            self._test_topk_dtype(device, dtype, False, curr_size)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float, torch.double, torch.bfloat16, torch.half)
    def test_topk_nonfinite(self, device, dtype):
        if False:
            i = 10
            return i + 15
        x = torch.tensor([float('nan'), float('inf'), 10000.0, 0, -10000.0, -float('inf')], device=device, dtype=dtype)
        (val, idx) = x.topk(4)
        expect = torch.tensor([float('nan'), float('inf'), 10000.0, 0], device=device, dtype=dtype)
        self.assertEqual(val, expect)
        self.assertEqual(idx, [0, 1, 2, 3])
        (val, idx) = x.topk(4, largest=False)
        expect = torch.tensor([-float('inf'), -10000.0, 0, 10000.0], device=device, dtype=dtype)
        self.assertEqual(val, expect)
        self.assertEqual(idx, [5, 4, 3, 2])

    def test_topk_4d(self, device):
        if False:
            while True:
                i = 10
        small = 128
        large = 8192
        for size in (small, large):
            x = torch.ones(2, size, 2, 2, device=device)
            x[:, 1, :, :] *= 2.0
            x[:, 10, :, :] *= 1.5
            (val, ind) = torch.topk(x, k=2, dim=1)
            expected_ind = torch.ones(2, 2, 2, 2, dtype=torch.long, device=device)
            expected_ind[:, 1, :, :] = 10
            expected_val = torch.ones(2, 2, 2, 2, device=device)
            expected_val[:, 0, :, :] *= 2.0
            expected_val[:, 1, :, :] *= 1.5
            self.assertEqual(val, expected_val, atol=0, rtol=0)
            self.assertEqual(ind, expected_ind, atol=0, rtol=0)

    @onlyNativeDeviceTypes
    @dtypesIfCUDA(*all_types_and(torch.bfloat16))
    @dtypes(*all_types_and(torch.bfloat16, torch.half))
    def test_topk_zero(self, device, dtype):
        if False:
            i = 10
            return i + 15
        t = torch.rand(2, 2, device=device).to(dtype=dtype)
        (val, idx) = torch.topk(t, k=0, largest=False)
        self.assertEqual(val.size(), torch.Size([2, 0]))
        self.assertEqual(idx.size(), torch.Size([2, 0]))

    def _test_unique_scalar_empty(self, dtype, device, f):
        if False:
            while True:
                i = 10
        x = torch.tensor(0, dtype=dtype, device=device)
        (unique, inverse, counts) = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([0], dtype=dtype, device=device)
        expected_inverse = torch.tensor(0, device=device)
        expected_counts = torch.tensor([1], device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)
        x = torch.zeros((0, 0, 3), dtype=dtype, device=device)
        (unique, inverse, counts) = f(x, return_inverse=True, return_counts=True)
        expected_unique = torch.tensor([], dtype=dtype, device=device)
        expected_inverse = torch.empty((0, 0, 3), dtype=torch.long, device=device)
        expected_counts = torch.tensor([], dtype=torch.long, device=device)
        self.assertEqual(unique, expected_unique)
        self.assertEqual(inverse, expected_inverse)
        self.assertEqual(counts, expected_counts)

    def _test_unique_with_expects(self, device, dtype, f, x, expected_unique, expected_inverse, expected_counts, additional_shape):
        if False:
            i = 10
            return i + 15

        def ensure_tuple(x):
            if False:
                print('Hello World!')
            if isinstance(x, torch.Tensor):
                return (x,)
            return x
        for return_inverse in [True, False]:
            for return_counts in [True, False]:
                ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                self.assertEqual(expected_unique, ret[0])
                if return_inverse:
                    self.assertEqual(expected_inverse, ret[1])
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    self.assertEqual(expected_counts, ret[count_index])
                y = x.view(additional_shape)
                (y_unique, y_inverse, y_counts) = f(y, return_inverse=True, return_counts=True)
                self.assertEqual(expected_unique, y_unique)
                self.assertEqual(expected_inverse.view(additional_shape), y_inverse)
                self.assertEqual(expected_counts, y_counts)

    @dtypesIfCPU(*all_types_and(torch.bool, torch.float16, torch.bfloat16))
    @dtypes(*all_types_and(torch.half, torch.bool))
    def test_unique(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')

        def ensure_tuple(x):
            if False:
                for i in range(10):
                    print('nop')
            if isinstance(x, torch.Tensor):
                return (x,)
            return x
        if dtype is torch.bool:
            x = torch.tensor([True, False, False, False, True, False, True, False], dtype=torch.bool, device=device)
            expected_unique = torch.tensor([False, True], dtype=torch.bool, device=device)
            expected_inverse = torch.tensor([1, 0, 0, 0, 1, 0, 1, 0], dtype=torch.long, device=device)
            expected_counts = torch.tensor([5, 3], dtype=torch.long, device=device)
        else:
            x = torch.tensor([1, 2, 3, 2, 8, 5, 2, 3], dtype=dtype, device=device)
            expected_unique = torch.tensor([1, 2, 3, 5, 8], dtype=dtype, device=device)
            expected_inverse = torch.tensor([0, 1, 2, 1, 4, 3, 1, 2], device=device)
            expected_counts = torch.tensor([1, 3, 2, 1, 1], device=device)
        fs = (lambda x, **kwargs: torch.unique(x, sorted=True, **kwargs), lambda x, **kwargs: x.unique(sorted=True, **kwargs))
        x_sliced = torch.empty(x.size(0) * 2, dtype=dtype, device=device)[::2].copy_(x)
        xs = (x, x_sliced)
        for (f, x) in product(fs, xs):
            self._test_unique_with_expects(device, dtype, f, x, expected_unique, expected_inverse, expected_counts, (2, 2, 2))
            self._test_unique_scalar_empty(dtype, device, f)
        fs = (lambda x, **kwargs: torch.unique(x, sorted=False, **kwargs), lambda x, **kwargs: x.unique(sorted=False, **kwargs))
        for (f, x) in product(fs, xs):
            self._test_unique_scalar_empty(dtype, device, f)
            for (return_inverse, return_counts) in product((True, False), repeat=2):
                ret = ensure_tuple(f(x, return_inverse=return_inverse, return_counts=return_counts))
                self.assertEqual(len(ret), 1 + int(return_inverse) + int(return_counts))
                x_list = x.tolist()
                x_unique_list = ret[0].tolist()
                self.assertEqual(expected_unique.tolist(), sorted(x_unique_list))
                if return_inverse:
                    x_inverse_list = ret[1].tolist()
                    for (i, j) in enumerate(x_inverse_list):
                        self.assertEqual(x_list[i], x_unique_list[j])
                if return_counts:
                    count_index = 1 + int(return_inverse)
                    x_counts_list = ret[count_index].tolist()
                    for (i, j) in zip(x_unique_list, x_counts_list):
                        count = 0
                        for k in x_list:
                            if k == i:
                                count += 1
                        self.assertEqual(j, count)

    @dtypesIfCPU(*all_types_and(torch.bool, torch.float16, torch.bfloat16))
    @dtypes(*all_types_and(torch.half, torch.bool))
    def test_unique_consecutive(self, device, dtype):
        if False:
            while True:
                i = 10
        if dtype is torch.bool:
            x = torch.tensor([True, False, False, False, True, True, False, False, False], dtype=torch.bool, device=device)
            expected_unique = torch.tensor([True, False, True, False], dtype=torch.bool, device=device)
            expected_inverse = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 3], dtype=torch.long, device=device)
            expected_counts = torch.tensor([1, 3, 2, 3], dtype=torch.long, device=device)
        else:
            x = torch.tensor([1, 2, 2, 2, 5, 5, 2, 2, 3], dtype=dtype, device=device)
            expected_unique = torch.tensor([1, 2, 5, 2, 3], dtype=dtype, device=device)
            expected_inverse = torch.tensor([0, 1, 1, 1, 2, 2, 3, 3, 4], device=device)
            expected_counts = torch.tensor([1, 3, 2, 2, 1], device=device)
        for f in [torch.unique_consecutive, lambda x, **kwargs: x.unique_consecutive(**kwargs)]:
            self._test_unique_with_expects(device, dtype, f, x, expected_unique, expected_inverse, expected_counts, (3, 3))
            self._test_unique_scalar_empty(dtype, device, f)

    @dtypes(torch.double)
    def test_kthvalue(self, device, dtype):
        if False:
            i = 10
            return i + 15
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE, dtype=dtype, device=device)
        x0 = x.clone()
        k = random.randint(1, SIZE)
        (res1val, res1ind) = torch.kthvalue(x, k, keepdim=False)
        (res2val, res2ind) = torch.sort(x)
        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)
        k = random.randint(1, SIZE)
        res1val = torch.tensor([], dtype=dtype, device=device)
        res1ind = torch.tensor([], dtype=torch.long, device=device)
        torch.kthvalue(x, k, keepdim=False, out=(res1val, res1ind))
        (res2val, res2ind) = torch.sort(x)
        self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)
        k = random.randint(1, SIZE)
        (res1val, res1ind) = torch.kthvalue(x, k, 0, keepdim=False)
        (res2val, res2ind) = torch.sort(x, 0)
        self.assertEqual(res1val, res2val[k - 1], atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind[k - 1], atol=0, rtol=0)
        y = x.narrow(1, 0, 1)
        y0 = y.contiguous()
        k = random.randint(1, SIZE)
        (res1val, res1ind) = torch.kthvalue(y, k)
        (res2val, res2ind) = torch.kthvalue(y0, k)
        self.assertEqual(res1val, res2val, atol=0, rtol=0)
        self.assertEqual(res1ind, res2ind, atol=0, rtol=0)
        non_contig_t = torch.tensor([0, -1, 1, -2, 2], dtype=dtype, device=device)[::2]
        (expected_val, expected_ind) = non_contig_t.contiguous().kthvalue(2)
        non_contig_cpu_t = non_contig_t.cpu()
        (expected_val_cpu, expected_ind_cpu) = non_contig_cpu_t.kthvalue(2)
        (out_val, out_ind) = non_contig_t.kthvalue(2)
        self.assertEqual(expected_val, out_val, atol=0, rtol=0)
        self.assertEqual(expected_ind, out_ind, atol=0, rtol=0)
        self.assertEqual(expected_val_cpu, out_val, atol=0, rtol=0)
        self.assertEqual(expected_ind_cpu, out_ind, atol=0, rtol=0)
        self.assertEqual(x, x0, atol=0, rtol=0)
        y = torch.tensor((3.0, 5, 4, 1, 1, 5), dtype=dtype, device=device)
        self.assertEqual(torch.kthvalue(y, 3)[0], 3, atol=0, rtol=0)
        self.assertEqual(torch.kthvalue(y, 2)[0], 1, atol=0, rtol=0)
        SIZE = 50
        x = torch.rand(SIZE, SIZE, SIZE, dtype=dtype, device=device)
        x[torch.arange(SIZE), :, torch.randint(50, (50,))] = nan
        ks = [random.randint(1, SIZE), 1, SIZE, SIZE - 1]
        (res2val, res2ind) = torch.sort(x)
        for k in ks:
            (res1val, res1ind) = torch.kthvalue(x, k, keepdim=False)
            self.assertEqual(res1val[:, :], res2val[:, :, k - 1], atol=0, rtol=0)
            self.assertEqual(res1ind[:, :], res2ind[:, :, k - 1], atol=0, rtol=0)

    @dtypes(torch.float)
    @onlyNativeDeviceTypes
    def test_kthvalue_scalar(self, device, dtype):
        if False:
            while True:
                i = 10
        res = torch.tensor(2, device=device, dtype=dtype).kthvalue(1)
        ref = torch.tensor([2], device=device, dtype=dtype).kthvalue(1)
        self.assertEqual(res[0], ref[0].squeeze())
        self.assertEqual(res[1], ref[1].squeeze())

    @dtypes(*all_types())
    @dtypesIfCUDA(*all_types_and(torch.half))
    def test_isin(self, device, dtype):
        if False:
            return 10

        def assert_isin_equal(a, b):
            if False:
                i = 10
                return i + 15
            x = torch.isin(a, b)
            a = a.cpu().numpy() if torch.is_tensor(a) else np.array(a)
            b = b.cpu().numpy() if torch.is_tensor(b) else np.array(b)
            y = np.isin(a, b)
            self.assertEqual(x, y)
        a = torch.arange(24, device=device, dtype=dtype).reshape([2, 3, 4])
        b = torch.tensor([[10, 20, 30], [0, 1, 3], [11, 22, 33]], device=device, dtype=dtype)
        assert_isin_equal(a, b)
        zero_d = torch.tensor(3, device=device, dtype=dtype)
        assert_isin_equal(zero_d, b)
        assert_isin_equal(a, zero_d)
        assert_isin_equal(zero_d, zero_d)
        empty = torch.tensor([], device=device, dtype=dtype)
        assert_isin_equal(empty, b)
        assert_isin_equal(a, empty)
        assert_isin_equal(empty, empty)
        assert_isin_equal(a, 6)
        assert_isin_equal(5, b)

        def define_expected(lst, invert=False):
            if False:
                return 10
            expected = torch.tensor(lst, device=device)
            if invert:
                expected = expected.logical_not()
            return expected
        for mult in [1, 10]:
            for invert in [False, True]:
                a = torch.tensor([5, 7, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)
                a[0] = 8
                ec = define_expected([False, False, True, True], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)
                (a[0], a[3]) = (4, 8)
                ec = define_expected([True, False, True, False], invert=invert)
                c = torch.isin(a, b, assume_unique=True, invert=invert)
                self.assertEqual(c, ec)
                a = torch.tensor([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5], device=device, dtype=dtype)
                b = torch.tensor([2, 3, 4] * mult, device=device, dtype=dtype)
                ec = define_expected([False, True, False, True, True, True, True, True, True, False, True, False, False, False], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)
                b = torch.tensor([2, 3, 4] * mult + [5, 5, 4] * mult, device=device, dtype=dtype)
                ec = define_expected([True, True, True, True, True, True, True, True, True, True, True, False, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)
                a = torch.tensor([5, 7, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)
                a = torch.tensor([5, 7, 1, 1, 2], device=device, dtype=dtype)
                b = torch.tensor([2, 4, 3, 3, 1, 5] * mult, device=device, dtype=dtype)
                ec = define_expected([True, False, True, True, True], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)
                a = torch.tensor([5, 5], device=device, dtype=dtype)
                b = torch.tensor([2, 2] * mult, device=device, dtype=dtype)
                ec = define_expected([False, False], invert=invert)
                c = torch.isin(a, b, invert=invert)
                self.assertEqual(c, ec)
                for assume_unique in [False, True]:
                    a = torch.arange(6, device=device, dtype=dtype).reshape([2, 3])
                    b = torch.arange(3, 30, device=device, dtype=dtype)
                    ec = define_expected([[False, False, False], [True, True, True]], invert=invert)
                    c = torch.isin(a, b, invert=invert, assume_unique=assume_unique)
                    self.assertEqual(c, ec)

    def test_isin_different_dtypes(self, device):
        if False:
            for i in range(10):
                print('nop')
        supported_types = all_types() if device == 'cpu' else all_types_and(torch.half)
        for mult in [1, 10]:
            for assume_unique in [False, True]:
                for (dtype1, dtype2) in product(supported_types, supported_types):
                    a = torch.tensor([1, 2, 3], device=device, dtype=dtype1)
                    b = torch.tensor([3, 4, 5] * mult, device=device, dtype=dtype2)
                    ec = torch.tensor([False, False, True], device=device)
                    c = torch.isin(a, b, assume_unique=assume_unique)
                    self.assertEqual(c, ec)

    @onlyCUDA
    @dtypes(*all_types())
    def test_isin_different_devices(self, device, dtype):
        if False:
            while True:
                i = 10
        a = torch.arange(6, device=device, dtype=dtype).reshape([2, 3])
        b = torch.arange(3, 30, device='cpu', dtype=dtype)
        with self.assertRaises(RuntimeError):
            torch.isin(a, b)
        c = torch.arange(6, device='cpu', dtype=dtype).reshape([2, 3])
        d = torch.arange(3, 30, device=device, dtype=dtype)
        with self.assertRaises(RuntimeError):
            torch.isin(c, d)

    @dtypes(*integral_types())
    def test_sort_overflow(self, device, dtype):
        if False:
            return 10
        ' Regression test for https://github.com/pytorch/pytorch/issues/111189 '
        prev_num_threads = torch.get_num_threads()
        try:
            low = 0 if dtype == torch.uint8 else -1
            x = torch.full((32768,), low, dtype=dtype, device=device)
            x[:100] = torch.iinfo(x.dtype).max
            torch.set_num_threads(1)
            uv = x.sort().values.unique()
            self.assertEqual(uv.size(0), 2)
        finally:
            torch.set_num_threads(prev_num_threads)
instantiate_device_type_tests(TestSortAndSelect, globals())
if __name__ == '__main__':
    run_tests()