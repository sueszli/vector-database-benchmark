from functools import reduce
from functools import partial
from itertools import repeat
import unittest
import subprocess
import sys
import os
import random
import itertools
import math
from torch import inf, nan
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_utils import TestCase, run_tests, TEST_WITH_UBSAN, set_default_dtype, instantiate_parametrized_tests, slowTest, parametrize as parametrize_test, subtest, skipIfMps, gcIfJetson, skipIfTorchDynamo
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_nn import NNTestCase, _test_bfloat16_ops, _test_module_empty_input
from torch.testing._internal.common_device_type import largeTensorTest, onlyNativeDeviceTypes, dtypes, instantiate_device_type_tests, skipCUDAIfRocm, expectedFailureMeta, dtypesIfCUDA, onlyCPU, onlyCUDA, TEST_WITH_ROCM
from torch.testing._internal.common_dtype import floating_types_and
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import gradcheck, gradgradcheck

class TestAvgPool(TestCase):

    def _sum_pool2d(self, x, kernel_size):
        if False:
            for i in range(10):
                print('nop')
        windows = torch.nn.functional.unfold(x, kernel_size=kernel_size, stride=kernel_size)
        return torch.sum(windows, dim=1)

    def _sum_pool3d(self, x, kernel_size):
        if False:
            return 10
        h = kernel_size[0]
        splited_x = [t.sum(0) for t in x.split(h) if t.size(0) == h]
        splited_x = [self._sum_pool2d(t.unsqueeze(0).unsqueeze(0), kernel_size[1:]) for t in splited_x]
        joined_x = torch.cat(splited_x)
        return joined_x.view(1, joined_x.numel())

    def _avg_pool2d(self, x, kernel_size):
        if False:
            for i in range(10):
                print('nop')
        size = reduce(lambda x, y: x * y, kernel_size)
        return self._sum_pool2d(x, kernel_size) / size

    def _avg_pool3d(self, x, kernel_size):
        if False:
            while True:
                i = 10
        size = reduce(lambda x, y: x * y, kernel_size)
        return self._sum_pool3d(x, kernel_size) / size

    def test_doubletensor_avg_pool2d(self):
        if False:
            i = 10
            return i + 15
        (n, m) = (5, 8)
        input = torch.rand(1, 1, n, m, dtype=torch.double)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                actual = torch.nn.functional.avg_pool2d(input[0], (i, j))
                actual = actual.view(1, actual.numel())
                expected = self._avg_pool2d(input, (i, j))
                self.assertEqual(actual, expected, rtol=0, atol=1e-05)

    def test_doubletensor_avg_pool2d_with_divisor(self):
        if False:
            return 10
        (n, m) = (3, 3)
        input = torch.rand(1, 1, n, m, dtype=torch.double)
        for i in range(1, n + 1):
            for j in range(1, m + 1):
                for divisor in [1, 7, i * j]:
                    actual = F.avg_pool2d(input[0], (i, j), divisor_override=divisor)
                    actual = actual.view(1, actual.numel())
                    expected = self._sum_pool2d(input, (i, j)) / divisor
                    self.assertEqual(actual, expected, rtol=0, atol=1e-05)

    def test_doubletensor_avg_pool3d(self):
        if False:
            for i in range(10):
                print('nop')
        (h, w, d) = (5, 6, 7)
        input = torch.rand(h, w, d, dtype=torch.double)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    actual = torch.nn.functional.avg_pool3d(input.unsqueeze(0), (i, j, k))
                    actual = actual.view(1, actual.numel())
                    expected = self._avg_pool3d(input, (i, j, k))
                    self.assertEqual(actual, expected, rtol=0, atol=1e-05)

    def test_doubletensor_avg_pool3d_with_divisor(self):
        if False:
            print('Hello World!')
        (h, w, d) = (6, 5, 7)
        input = torch.rand(h, w, d, dtype=torch.double)
        for i in range(1, h + 1):
            for j in range(1, w + 1):
                for k in range(1, d + 1):
                    for divisor in [1, 7, i * j]:
                        actual = torch.nn.functional.avg_pool3d(input.unsqueeze(0), (i, j, k), divisor_override=divisor)
                        actual = actual.view(1, actual.numel())
                        expected = self._sum_pool3d(input, (i, j, k)) / divisor
                        self.assertEqual(actual, expected, rtol=0, atol=1e-05)

    def test_avg_pool1d_ceil_mode(self):
        if False:
            i = 10
            return i + 15
        x = 10 * torch.randn((1, 16, 4))
        y = torch.nn.functional.avg_pool1d(x, ceil_mode=True, count_include_pad=True, kernel_size=1, stride=2)
        self.assertTrue(not torch.isnan(y).any())
        if TEST_CUDA:
            y = torch.nn.functional.avg_pool1d(x.to('cuda'), ceil_mode=True, count_include_pad=True, kernel_size=1, stride=2)
            self.assertTrue(not torch.isnan(y).any())

    def test_avg_pool2d_ceil_mode(self):
        if False:
            while True:
                i = 10
        x = 10 * torch.randn((1, 16, 4, 4))
        y = torch.nn.functional.avg_pool2d(x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2), padding=(0, 1), stride=2)
        self.assertTrue(not torch.isnan(y).any())
        if TEST_CUDA:
            y = torch.nn.functional.avg_pool2d(x.to('cuda'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2), padding=(0, 1), stride=2)
            self.assertTrue(not torch.isnan(y).any())

    def test_avg_pool3d_ceil_mode(self):
        if False:
            print('Hello World!')
        x = 10 * torch.randn((1, 16, 4, 4, 4))
        y = torch.nn.functional.avg_pool3d(x, ceil_mode=True, count_include_pad=True, kernel_size=(1, 2, 3), stride=2)
        self.assertTrue(not torch.isnan(y).any())
        if TEST_CUDA:
            y = torch.nn.functional.avg_pool3d(x.to('cuda'), ceil_mode=True, count_include_pad=True, kernel_size=(1, 2, 3), stride=2)
            self.assertTrue(not torch.isnan(y).any())

class TestPoolingNN(NNTestCase):
    _do_cuda_memory_leak_check = True
    _do_cuda_non_default_stream = True

    def test_adaptive_pooling_size_none(self):
        if False:
            i = 10
            return i + 15
        for numel in (2, 3):
            for pool_type in ('Max', 'Avg'):
                cls_name = f'Adaptive{pool_type}Pool{numel}d'
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * (numel - 1) + (None,)
                module = module_cls(output_size)
                input = torch.randn((4,) * (numel + 1))
                output = module(input)
                self.assertEqual(output.size(), (4,) + (2,) * (numel - 1) + (4,))

    @unittest.skipIf(TEST_WITH_UBSAN, 'signed integer overflow error with UBSAN')
    def test_adaptive_pooling_size_overflow(self):
        if False:
            return 10
        self.assertRaises(RuntimeError, lambda : torch.nn.AdaptiveMaxPool1d(4611686018427387903)(torch.empty([2, 2, 2])))

    def test_adaptive_pooling_avg_nhwc(self):
        if False:
            for i in range(10):
                print('nop')
        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')
        for device in device_list:
            input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
            pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)
            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)

    def test_adaptive_pooling_avg_nhwc_non_contiguous(self):
        if False:
            while True:
                i = 10
        device_list = ['cpu']
        if TEST_CUDA:
            device_list.append('cuda')
        for device in device_list:
            input = torch.randint(1, 10, (4, 8, 8, 8), dtype=torch.float32).to(device)
            input = input.contiguous(memory_format=torch.channels_last)
            input = input[:, ::2, :, :].requires_grad_()
            grad = torch.randint(1, 10, (4, 8, 7, 7), dtype=torch.float32).to(device)
            grad = grad[:, ::2, :, :]
            pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AdaptiveAvgPool2d((7, 7)).to(device)
            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)

    def test_adaptive_pooling_bfloat16(self):
        if False:
            for i in range(10):
                print('nop')

        def _test_adaptive_pooling_bfloat16(self, device, mod, memory_format):
            if False:
                while True:
                    i = 10
            input = torch.randint(1, 10, (3, 19, 8, 8), dtype=torch.float32)
            input = input.to(device).to(memory_format=memory_format).requires_grad_()
            pool = mod((7, 7)).to(device)
            input2 = input.detach().clone().bfloat16().requires_grad_(True)
            out = pool(input)
            out.sum().backward()
            out2 = pool(input2)
            out2.sum().backward()
            self.assertTrue(out2.is_contiguous(memory_format=memory_format))
            self.assertEqual(out2.dtype, torch.bfloat16)
            self.assertEqual(input2.grad.dtype, torch.bfloat16)
            self.assertEqual(out, out2.float(), atol=0.1, rtol=0)
            self.assertEqual(input.grad, input2.grad.float(), atol=0.1, rtol=0)
        device_list = ['cpu']
        for device in device_list:
            _test_adaptive_pooling_bfloat16(self, device, torch.nn.AdaptiveAvgPool2d, torch.contiguous_format)
            _test_adaptive_pooling_bfloat16(self, device, torch.nn.AdaptiveAvgPool2d, torch.channels_last)
            _test_adaptive_pooling_bfloat16(self, device, torch.nn.AdaptiveMaxPool2d, torch.contiguous_format)
            _test_adaptive_pooling_bfloat16(self, device, torch.nn.AdaptiveMaxPool2d, torch.channels_last)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    @largeTensorTest('12GB', device='cuda')
    def test_adaptive_pooling_avg_nhwc_launch_config_backward(self):
        if False:
            i = 10
            return i + 15
        input = torch.randint(1, 10, (1, 32, 2 ** 17 + 1, 32), dtype=torch.float32, device='cuda')
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        grad = torch.randint(1, 10, (1, 32, 10, 32), dtype=torch.float32, device='cuda')
        pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()
        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_grad = grad.detach().clone().contiguous()
        ref_pool = torch.nn.AdaptiveAvgPool2d((10, 32)).cuda()
        out = pool(input)
        out.backward(grad)
        ref_out = ref_pool(ref_input)
        ref_out.backward(ref_grad)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)
        self.assertEqual(input.grad, ref_input.grad)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    @largeTensorTest('12GB', device='cuda')
    def test_adaptive_pooling_avg_nhwc_launch_config_forward(self):
        if False:
            return 10
        input = torch.randint(1, 10, (1, 32, 16, 16), dtype=torch.float32, device='cuda')
        input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
        pool = torch.nn.AdaptiveAvgPool2d((2 ** 17 + 1, 32)).cuda()
        ref_input = input.detach().clone().contiguous().requires_grad_(True)
        ref_pool = torch.nn.AdaptiveAvgPool2d((2 ** 17 + 1, 32)).cuda()
        out = pool(input)
        ref_out = ref_pool(ref_input)
        self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_out.is_contiguous())
        self.assertEqual(out, ref_out)

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_adaptive_avg_pooling_overflow(self):
        if False:
            i = 10
            return i + 15
        input = torch.randint(-256, 256, (20, 32, 256, 256), dtype=torch.half, device='cuda')
        avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        out = avg_pool(input)
        self.assertFalse(torch.isinf(out).any())
        self.assertFalse(torch.isnan(out).any())

    @unittest.skipIf(not TEST_CUDA, 'CUDA unavailable')
    def test_adaptive_avg_pooling_nhwc_overflow(self):
        if False:
            i = 10
            return i + 15
        input = torch.randint(-256, 256, (20, 32, 256, 256), dtype=torch.half, device='cuda')
        input = input.contiguous(memory_format=torch.channels_last)
        avg_pool = torch.nn.AdaptiveAvgPool2d((2, 2))
        out = avg_pool(input)
        self.assertFalse(torch.isinf(out).any())
        self.assertFalse(torch.isnan(out).any())

    def test_MaxUnpool2d_output_size(self):
        if False:
            for i in range(10):
                print('nop')
        m = nn.MaxPool2d(3, stride=2, return_indices=True)
        mu = nn.MaxUnpool2d(3, stride=2)
        big_t = torch.rand(1, 1, 6, 6)
        big_t[0][0][4][4] = 100
        (output_big, indices_big) = m(big_t)
        self.assertRaises(RuntimeError, lambda : mu(output_big, indices_big))
        small_t = torch.rand(1, 1, 5, 5)
        for i in range(0, 4, 2):
            for j in range(0, 4, 2):
                small_t[:, :, i, j] = 100
        (output_small, indices_small) = m(small_t)
        for h in range(3, 10):
            for w in range(3, 10):
                if 4 <= h <= 6 and 4 <= w <= 6:
                    size = (h, w)
                    if h == 6:
                        size = (1, 1) + size
                    mu(output_small, indices_small, output_size=size)
                else:
                    self.assertRaises(ValueError, lambda : mu(output_small, indices_small, (h, w)))

    def test_max_unpool2d_nhwc_cpu(self):
        if False:
            i = 10
            return i + 15
        input = torch.randn(2, 10, 9, 9).float().cpu()
        input = input.contiguous(memory_format=torch.channels_last)
        ref_input = input.clone().contiguous()
        pool = nn.MaxPool2d(3, stride=2, return_indices=True).cpu()
        ref_pool = nn.MaxPool2d(3, stride=2, return_indices=True).cpu()
        (out, ind) = pool(input)
        (ref_out, ref_ind) = ref_pool(ref_input)
        out.requires_grad_()
        ref_out.requires_grad_()
        unpool = nn.MaxUnpool2d(3, stride=2).cpu()
        ref_unpool = nn.MaxUnpool2d(3, stride=2).cpu()
        upout = unpool(out, ind)
        ref_upout = ref_unpool(ref_out, ref_ind)
        grad = torch.randn(upout.size()).float().cpu()
        grad = grad.contiguous(memory_format=torch.channels_last)
        ref_grad = grad.clone().contiguous()
        upout.backward(grad)
        ref_upout.backward(ref_grad)
        self.assertTrue(upout.is_contiguous(memory_format=torch.channels_last))
        self.assertTrue(ref_upout.is_contiguous())
        self.assertTrue(torch.allclose(upout, ref_upout))
        self.assertTrue(torch.allclose(out.grad, ref_out.grad))

    def test_max_unpool(self):
        if False:
            i = 10
            return i + 15
        with set_default_dtype(torch.double):
            (output, indices) = F.max_pool1d(torch.randn([1, 1, 4]), 2, stride=2, return_indices=True)
            self.assertEqual(F.max_unpool1d(output, indices, 2), F.max_unpool1d(output, indices, 2, stride=2))
            input = torch.randn([1, 1, 5], requires_grad=True)
            (output, indices) = F.max_pool1d(input, 2, stride=2, return_indices=True)
            self.assertEqual(F.max_unpool1d(output, indices, 2, stride=2, output_size=input.shape), F.max_unpool1d(output, indices, 2, stride=2, output_size=input.size()))
            gradcheck(F.max_unpool1d, (output, indices, 2), check_forward_ad=True)
            (output, indices) = F.max_pool2d(torch.randn([1, 1, 4, 4], requires_grad=True), 2, stride=2, return_indices=True)
            self.assertEqual(F.max_unpool2d(output, indices, 2), F.max_unpool2d(output, indices, 2, stride=2))
            gradcheck(F.max_unpool2d, (output, indices, 2), check_forward_ad=True)
            (output, indices) = F.max_pool3d(torch.randn([4, 4, 4, 4, 4], requires_grad=True), 2, stride=2, return_indices=True)
            self.assertEqual(F.max_unpool3d(output, indices, 2), F.max_unpool3d(output, indices, 2, stride=2))
            gradcheck(F.max_unpool3d, (output, indices, 2), check_forward_ad=True)

    def test_max_unpool3d_input_check(self):
        if False:
            for i in range(10):
                print('nop')
        x = torch.ones(1, 3, 1, 1, 1)
        with self.assertRaises(RuntimeError):
            F.max_unpool3d(x, torch.zeros(x.shape, dtype=int), [1, 1])

class TestPoolingNNDeviceType(NNTestCase):

    @onlyNativeDeviceTypes
    @dtypes(torch.float, torch.double)
    def test_adaptive_pooling_zero_batch(self, dtype, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.ones(0, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool1d(5).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        inp = torch.ones(0, 10, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool2d((5, 5)).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        inp = torch.ones(0, 10, 10, 10, dtype=dtype, device=device)
        mod = torch.nn.AdaptiveAvgPool3d((5, 5, 5)).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)

    @onlyNativeDeviceTypes
    @dtypes(torch.float32, torch.float64)
    @dtypesIfCUDA(torch.float32, torch.float64, torch.bfloat16, torch.float16)
    def test_adaptive_pooling_empty_output_size(self, dtype, device):
        if False:
            i = 10
            return i + 15
        error_msg = 'Expected grad_output to have non-zero size for non-batch dimensions'
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=True)
        input = make_arg((1, 64, 10, 9))
        output_size = 0
        fns = (nn.functional.adaptive_avg_pool2d, nn.functional.adaptive_avg_pool3d, nn.functional.adaptive_max_pool2d, nn.functional.adaptive_max_pool3d)
        for fn in fns:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                fn(input, output_size).sum().backward()
        fns2 = (nn.functional.adaptive_avg_pool1d, nn.functional.adaptive_max_pool1d)
        input2 = make_arg((1, 64))
        for fn in fns2:
            with self.assertRaisesRegex(RuntimeError, error_msg):
                fn(input2, output_size).sum().backward()

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool2d_zero_batch(self, device):
        if False:
            return 10
        mod = nn.FractionalMaxPool2d(3, output_ratio=(0.5, 0.5))
        inp = torch.ones(0, 16, 50, 32, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected input'):
            inp = torch.randn(1, 0, 50, 32, device=device)
            mod(inp)

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_zero_batch(self, device):
        if False:
            for i in range(10):
                print('nop')
        mod = nn.FractionalMaxPool3d(3, output_ratio=(0.5, 0.5, 0.5)).to(device)
        inp = torch.ones(0, 16, 50, 32, 32, device=device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected input'):
            inp = torch.randn(1, 0, 50, 32, 32, device=device)
            mod(inp)

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool2d_zero_out_size(self, device):
        if False:
            return 10
        mod = nn.FractionalMaxPool2d([2, 2], output_size=[0, 1])
        inp = torch.rand([16, 50, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((16, 50, 0, 1), device=device))

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_zero_out_size(self, device):
        if False:
            return 10
        mod = nn.FractionalMaxPool3d([3, 2, 2], output_size=[0, 1, 1])
        inp = torch.rand([16, 50, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((16, 0, 1, 1), device=device))

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool2d_zero_samples(self, device):
        if False:
            for i in range(10):
                print('nop')
        samples = torch.rand([0, 16, 2], device=device)
        mod = nn.FractionalMaxPool2d([2, 2], output_size=[1, 1], _random_samples=samples)
        inp = torch.randn([0, 16, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((0, 16, 1, 1), device=device))
        inp1 = torch.randn([1, 16, 32, 32], device=device)
        with self.assertRaisesRegex(RuntimeError, 'Expect _random_samples'):
            out1 = mod(inp1)

    @onlyNativeDeviceTypes
    def test_FractionalMaxPool3d_zero_samples(self, device):
        if False:
            while True:
                i = 10
        samples = torch.rand([0, 16, 3], device=device)
        mod = nn.FractionalMaxPool3d([3, 2, 2], output_size=[1, 1, 1], _random_samples=samples)
        inp = torch.randn([0, 16, 50, 32, 32], device=device)
        out = mod(inp)
        self.assertEqual(out, torch.empty((0, 16, 1, 1, 1), device=device))
        inp1 = torch.randn([1, 16, 50, 32, 32], device=device)
        with self.assertRaisesRegex(RuntimeError, 'Expect _random_samples'):
            out1 = mod(inp1)

    @onlyNativeDeviceTypes
    def test_MaxPool_zero_batch_dim(self, device):
        if False:
            i = 10
            return i + 15
        inp = torch.randn(0, 16, 50, device=device)
        mod = torch.nn.MaxPool1d(3, stride=2).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        inp = torch.randn(0, 16, 50, 32, device=device)
        mod = torch.nn.MaxPool2d(3, stride=2).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected'):
            inp = torch.randn(1, 0, 50, 32, device=device)
            mod(inp)
        inp = torch.ones(0, 16, 50, 44, 31, device=device)
        mod = torch.nn.MaxPool3d(3, stride=2).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected'):
            inp = torch.ones(1, 0, 50, 44, 31, device=device)
            mod(inp)

    @onlyNativeDeviceTypes
    def test_MaxUnpool_zero_batch_dim(self, device):
        if False:
            return 10
        pool = torch.nn.MaxPool1d(2, stride=2, return_indices=True).to(device)
        unpool = torch.nn.MaxUnpool1d(2, stride=2).to(device)
        inp = torch.randn(0, 10, 10, requires_grad=True, device=device)
        (output, indices) = pool(inp)
        output.requires_grad_(True)
        unpool_out = unpool(output, indices)
        unpool_out.sum().backward()
        self.assertEqual(inp.grad, torch.zeros_like(inp))
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))
        pool = torch.nn.MaxPool2d(2, stride=2, return_indices=True).to(device)
        unpool = torch.nn.MaxUnpool2d(2, stride=2).to(device)
        inp = torch.randn(0, 10, 10, 10, requires_grad=True, device=device)
        (output, indices) = pool(inp)
        unpool_out = unpool(output, indices)
        unpool_out.sum().backward()
        self.assertEqual(inp.grad, torch.zeros_like(inp))
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))
        pool = torch.nn.MaxPool3d(2, stride=2, return_indices=True).to(device)
        unpool = torch.nn.MaxUnpool3d(2, stride=2).to(device)
        inp = torch.randn(0, 10, 10, 10, 10, requires_grad=True, device=device)
        (output, indices) = pool(inp)
        output.requires_grad_(True)
        unpool_out = unpool(output, indices)
        unpool_out.sum().backward()
        self.assertEqual(inp.grad, torch.zeros_like(inp))
        self.assertEqual(unpool_out, torch.zeros_like(unpool_out))

    @slowTest
    @onlyNativeDeviceTypes
    @skipCUDAIfRocm
    @parametrize_test('module_name,module_size,output_size,test_index,should_error', [subtest(('MaxUnpool2d', (2, 2), (1, 3, 4, 5), -1, True), name='case1'), subtest(('MaxUnpool2d', (2, 2), (1, 3, 4, 5), 2 * 2 * 4 * 5, True), name='case2'), subtest(('MaxUnpool2d', (2, 2), (1, 3, 4, 5), 2 * 2 * 4 * 5 - 1, False), name='case3'), subtest(('MaxUnpool2d', (2, 3), (2, 1, 4, 2), 2 * 3 * 4 * 2, True), name='case4'), subtest(('MaxUnpool2d', (2, 3), (2, 1, 4, 2), 2 * 3 * 4 * 2 - 1, False), name='case5'), subtest(('MaxUnpool3d', (2, 2, 2), (1, 3, 4, 5), -1, True), name='case6'), subtest(('MaxUnpool3d', (2, 2, 2), (1, 3, 4, 5), 2 * 2 * 2 * 3 * 4 * 5, True), name='case7'), subtest(('MaxUnpool3d', (2, 2, 2), (1, 3, 4, 5), 2 * 2 * 2 * 3 * 4 * 5 - 1, False), name='case8'), subtest(('MaxUnpool3d', (2, 2, 2), (2, 3, 4, 1), 2 * 2 * 2 * 3 * 4 * 1, True), name='case9'), subtest(('MaxUnpool3d', (2, 2, 2), (2, 3, 4, 1), 2 * 2 * 2 * 3 * 4 * 1 - 1, False), name='case10')])
    def test_MaxUnpool_index_errors(self, device, module_name, module_size, output_size, test_index, should_error):
        if False:
            print('Hello World!')
        if torch.device(device).type == 'cuda':
            error_msgs = {'MaxUnpool2d': 'Assertion `maxind >= 0 && maxind < outputImageSize` failed', 'MaxUnpool3d': 'Assertion `index >= 0 && index < outputImageSize` failed'}
            script = f"\nimport torch\nunpool = torch.nn.{module_name}({module_size}).to('{device}')\noutput = torch.rand({output_size}, dtype=torch.float32, device='{device}')\nindices = torch.zeros({output_size}, dtype=torch.int64, device='{device}')\nindices.flatten()[0] = {test_index}\nunpool(output, indices)\ntorch.cuda.synchronize()\n"
            p = subprocess.run([sys.executable, '-c', script], cwd=os.path.dirname(os.path.realpath(__file__)), capture_output=True, text=True)
            output = p.stdout + '\n' + p.stderr
            error_msg = error_msgs[module_name]
            if should_error:
                self.assertIn(error_msg, output, 'The expected error was not found')
            else:
                self.assertNotIn('Error', output, 'Should not have produced an error')
        else:
            module_class = getattr(torch.nn, module_name)
            unpool = module_class(module_size).to(device)
            output = torch.rand(output_size, dtype=torch.float32, device=device)
            indices = torch.zeros(output_size, dtype=torch.int64, device=device)
            indices.flatten()[0] = test_index
            if should_error:
                with self.assertRaisesRegex(RuntimeError, 'Found an invalid max index:'):
                    unpool(output, indices)
            else:
                unpool(output, indices)

    @onlyNativeDeviceTypes
    def test_AdaptiveMaxPool_zero_batch_dim(self, device):
        if False:
            while True:
                i = 10
        inp = torch.randn(0, 16, 50, device=device)
        mod = torch.nn.AdaptiveMaxPool1d(3).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected'):
            inp = torch.randn(1, 0, 50, device=device)
            mod(inp)
        inp = torch.randn(0, 16, 50, 32, device=device)
        mod = torch.nn.AdaptiveMaxPool2d(3).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected'):
            inp = torch.randn(1, 0, 50, 32, device=device)
            mod(inp)
        inp = torch.ones(0, 16, 50, 44, 31, device=device)
        mod = torch.nn.AdaptiveMaxPool3d(3).to(device)
        _test_module_empty_input(self, mod, inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, 'Expected'):
            inp = torch.ones(1, 0, 50, 44, 31, device=device)
            mod(inp)

    @onlyNativeDeviceTypes
    def test_AvgPool2d_empty(self, device):
        if False:
            while True:
                i = 10
        avgpool = torch.nn.AvgPool2d(3, stride=2).to(device)
        inp = torch.randn(0, 16, 20, 32, device=device)
        _test_module_empty_input(self, avgpool, inp, check_size=False)
        clast_inp = torch.randn(0, 16, 20, 32, device=device).contiguous(memory_format=torch.channels_last)
        _test_module_empty_input(self, avgpool, clast_inp, check_size=False)
        with self.assertRaisesRegex(RuntimeError, '3D or 4D'):
            inp = torch.randn(16, 0, 20, 32, device=device)
            avgpool(inp)

    def test_pooling_shape(self, device):
        if False:
            print('Hello World!')
        ' Test the output shape calculation for pooling functions '

        def check(expected_out_shape, sizes, *args, **kwargs):
            if False:
                print('Hello World!')
            for kernel in ['max', 'avg']:
                for i in [1, 2, 3]:
                    if hasattr(torch.nn.functional, f'{kernel}_pool{i}d'):
                        op = getattr(torch.nn.functional, f'{kernel}_pool{i}d')
                        t = torch.randn(sizes[:i + 2], device=device)
                        self.assertEqual(op(t, *args, **kwargs).shape, expected_out_shape[:i + 2])
        check((1, 1, 3, 3, 4), (1, 1, 5, 6, 7), kernel_size=1, stride=2, padding=0, ceil_mode=True)
        check((1, 1, 2, 3, 3), (1, 1, 3, 4, 5), kernel_size=2, stride=2, padding=1, ceil_mode=False)
        check((1, 1, 2, 3, 3), (1, 1, 3, 4, 5), kernel_size=2, stride=2, padding=1, ceil_mode=True)
        x = torch.randn(1, 1, 6, 7, device=device)
        y = torch.nn.functional.max_pool2d(x, 1, stride=(2, 2), padding=0, ceil_mode=True)
        self.assertEqual(y.size(), (1, 1, 3, 4))

    @onlyNativeDeviceTypes
    def test_adaptive_avg_pool2d_output_size_one(self, device):
        if False:
            return 10

        def helper(size, memory_format):
            if False:
                print('Hello World!')
            x = torch.randint(1, 10, size, dtype=torch.float, device=device, requires_grad=True)
            if memory_format == 'non_contiguous':
                x = x[::2, ::2, ::2, ::2]
            else:
                x = x.to(memory_format=memory_format)
            net = torch.nn.AdaptiveAvgPool2d((1, 1))
            out = net(x)
            ref_out = x.contiguous().mean((-1, -2)).view((x.size(0), x.size(1), 1, 1))
            out.sum().backward()
            self.assertEqual(out, ref_out)
            if memory_format == torch.channels_last:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, c, c])
            else:
                self.assertTrue(out.is_contiguous())
                c = out.size(1)
                self.assertEqual(out.stride(), [c, 1, 1, 1])
        for mf in (torch.contiguous_format, torch.channels_last, 'non_contiguous'):
            helper((2, 3, 6, 6), mf)

    @onlyNativeDeviceTypes
    def test_adaptive_avg_pool3d_output_size_one(self, device):
        if False:
            return 10
        x = torch.randn((2, 3, 6, 6, 6), dtype=torch.float, device=device, requires_grad=True)
        net = torch.nn.AdaptiveAvgPool3d(1)
        out = net(x)
        ref_out = x.contiguous().mean((-1, -2, -3)).view(out.shape)
        out.sum().backward()
        self.assertEqual(out, ref_out)
        self.assertTrue(out.is_contiguous())
        c = out.size(1)
        self.assertEqual(out.stride(), [c, 1, 1, 1, 1])

    @expectedFailureMeta
    @onlyNativeDeviceTypes
    @dtypes(torch.uint8, torch.int8, torch.short, torch.int, torch.long)
    def test_adaptive_pooling_no_suppot_input(self, device, dtype):
        if False:
            i = 10
            return i + 15
        for numel in (2, 3):
            for pool_type in ('Max', 'Avg'):
                cls_name = f'Adaptive{pool_type}Pool{numel}d'
                module_cls = getattr(nn, cls_name)
                output_size = (2,) * numel
                module = module_cls(output_size)
                input = torch.randn((4,) * (numel + 1), device=device).to(dtype)
                with self.assertRaisesRegex(RuntimeError, 'not implemented'):
                    output = module(input)

    @onlyNativeDeviceTypes
    @gcIfJetson
    @dtypes(torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    def test_avg_pool2d_nhwc(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')

        def helper(n, c, h, w, kernel_size, stride=None, count_include_pad=True, divisor_override=None, padding=0):
            if False:
                print('Hello World!')
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(n, c, (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1, dtype=dtype, device=device)
            pool = torch.nn.AvgPool2d(kernel_size, stride=stride, count_include_pad=count_include_pad, divisor_override=divisor_override).to(device)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AvgPool2d(kernel_size, stride=stride, count_include_pad=count_include_pad, divisor_override=divisor_override).to(device)
            out = pool(input)
            out.backward(grad)
            ref_out = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(input.grad, ref_input.grad)
        helper(4, 8, 8, 8, 3)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=1)
        helper(4, 8, 8, 8, 3, count_include_pad=False, padding=2, stride=2)
        helper(4, 8, 8, 8, 3, divisor_override=42)
        helper(4, 8, 8, 8, 7)
        if TEST_WITH_ROCM and 'cuda' in device:
            torch.cuda.empty_cache()
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(4, 8, 7, 7, 3, padding=2, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    @onlyCPU
    @dtypes(torch.float, torch.double)
    def test_max_pool1d_corner_cases(self, device, dtype):
        if False:
            print('Hello World!')

        def check(x, args, expected):
            if False:
                return 10
            model = torch.nn.MaxPool1d(*args)
            if isinstance(x, list):
                x = torch.tensor(x, device=device, dtype=dtype)
                expected = torch.tensor(expected, device=device, dtype=dtype)
            self.assertEqual(model(x), expected)
        check([[1]], (1, None, 0, 1, False, False), [[1]])
        check([[1]], (2, None, 1, 2, False, False), [[float('-inf')]])
        check([[1], [1]], (2, None, 1, 2, False, False), [[float('-inf')], [float('-inf')]])
        check([[1, 2]], (2, 1, 1, 2, False, False), [[2, 1]])
        check([[1, 2]], (2, 2, 1, 2, False, True), [[2, 2]])

    @onlyCPU
    @dtypes(torch.float, torch.double)
    @skipIfTorchDynamo('OOMs https://github.com/pytorch/pytorch/issues/111320')
    def test_max_pool1d(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')

        def check(x, *args, **kwargs):
            if False:
                return 10
            model = torch.nn.MaxPool1d(*args, **kwargs)
            ref_model = torch.nn.MaxPool1d(*args, **kwargs, return_indices=True)
            self.assertEqual(model(x), ref_model(x)[0])
        sizes = [random.sample(range(8, 128), 3) for _ in range(3)]
        kernel_sizes = random.sample(range(1, 5), 3)
        strides = random.sample(range(1, 5), 3)
        dilations = random.sample(range(1, 5), 3)
        ceil_modes = [True, False]
        for (size, kernel_size, stride, dilation, ceil_mode) in itertools.product(sizes, kernel_sizes, strides, dilations, ceil_modes):
            padding = random.sample(range(0, math.floor(kernel_size / 2) + 1), 1)
            check(torch.randn(size, device=device, dtype=dtype), kernel_size, stride, padding, dilation, ceil_mode=ceil_mode)
        tensor = torch.randn(5, 151, 33, device=device, dtype=dtype)[::2, ::3, ::2]
        check(tensor, 3, 2, 1, 2, ceil_mode=True)
        check(tensor.transpose(1, 2), 3, 2, 1, 2, ceil_mode=True)

    @onlyCUDA
    @gcIfJetson
    def test_max_pool2d(self, device):
        if False:
            print('Hello World!')

        def helper(n, c, h, w, ks):
            if False:
                i = 10
                return i + 15
            x = torch.randn(n, c, h, w, device='cuda', dtype=torch.float, requires_grad=True)
            ref_x = x.detach().clone().cpu().requires_grad_()
            pool = torch.nn.MaxPool2d(kernel_size=ks)
            y = pool(x)
            ref_y = pool(ref_x)
            y.sum().backward()
            ref_y.sum().backward()
            self.assertEqual(y, ref_y)
            self.assertEqual(x.grad, ref_x.grad)
        helper(2, 8, 4, 4, ks=2)
        helper(1, 100000, 32, 32, ks=4)
        helper(1, 100000, 1, 4, ks=(1, 4))

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @gcIfJetson
    def test_max_pool2d_nhwc(self, device, dtype):
        if False:
            return 10

        def helper(n, c, h, w, kernel_size, stride=None):
            if False:
                i = 10
                return i + 15
            if stride is None:
                stride = kernel_size
            input = torch.randn(n, c, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last).requires_grad_()
            grad = torch.randn(n, c, (h - kernel_size) // stride + 1, (w - kernel_size) // stride + 1, dtype=dtype, device=device)
            pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(device)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(device)
            (out, ind) = pool(input)
            out.backward(grad)
            (ref_out, ref_ind) = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(ind, ref_ind)
            self.assertEqual(input.grad, ref_input.grad)
        helper(4, 8, 8, 8, 7)
        helper(200, 512, 28, 28, 2)
        helper(4, 8, 7, 7, 3, stride=1)
        helper(10, 512, 31, 31, 3, stride=2)
        helper(1, 129, 8, 8, 3, stride=2)

    @onlyNativeDeviceTypes
    @dtypes(torch.half, torch.bfloat16, torch.float, torch.double)
    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @gcIfJetson
    def test_max_pool3d_ndhwc(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')

        def helper(n, c, h, w, d, kernel_size, stride=None):
            if False:
                while True:
                    i = 10
            batch = n
            if not batch:
                batch = 1
            input = torch.randn(batch, c, d, h, w, dtype=dtype, device=device)
            input = input.contiguous(memory_format=torch.channels_last_3d).requires_grad_()
            if not n:
                input = input.squeeze(0).detach().clone().requires_grad_()
            if isinstance(kernel_size, int):
                kernel_size = [kernel_size] * 3
            if stride is None:
                stride = kernel_size
            elif isinstance(stride, int):
                stride = [stride] * 3
            grad = torch.randn(batch, c, (d - kernel_size[0]) // stride[0] + 1, (h - kernel_size[1]) // stride[1] + 1, (w - kernel_size[2]) // stride[2] + 1, dtype=dtype, device=device)
            grad = grad.contiguous(memory_format=torch.channels_last_3d)
            if not n:
                grad = grad.squeeze(0)
            pool = torch.nn.MaxPool3d(kernel_size, stride, return_indices=True).to(device)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool3d(kernel_size, stride, return_indices=True).to(device)
            (out, ind) = pool(input)
            out.backward(grad)
            (ref_out, ref_ind) = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            if len(out.shape) == 4:
                self.assertTrue(out.unsqueeze(0).is_contiguous(memory_format=torch.channels_last_3d))
            else:
                self.assertTrue(out.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(ref_out.is_contiguous())
            if len(ind.shape) == 4:
                self.assertTrue(ind.unsqueeze(0).is_contiguous(memory_format=torch.channels_last_3d))
            else:
                self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last_3d))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(ind, ref_ind)
            if dtype == torch.half:
                self.assertEqual(input.grad, ref_input.grad, atol=0.05, rtol=0.01)
            else:
                self.assertEqual(input.grad, ref_input.grad)
        helper(4, 8, 8, 8, 8, 7)
        helper(4, 8, 8, 8, 8, (5, 6, 7))
        helper(1, 8, 8, 8, 8, (5, 6, 7))
        helper(0, 6, 12, 13, 14, (5, 6, 7))
        helper(4, 8, 7, 7, 7, 3, stride=1)
        helper(10, 128, 19, 19, 19, 3, stride=2)
        helper(10, 128, 19, 19, 19, (1, 2, 3), stride=2)
        helper(1, 128, 19, 19, 19, (1, 2, 3), stride=2)
        helper(0, 128, 19, 19, 19, (1, 2, 3), stride=2)
        helper(1, 79, 4, 4, 4, 3, stride=2)
        helper(0, 79, 4, 4, 4, 3, stride=2)

    @onlyCPU
    @dtypes(torch.half, torch.bfloat16)
    def test_max_pool_bfloat16_half(self, device, dtype):
        if False:
            i = 10
            return i + 15

        def helper(shape, kernel_size, stride, memory_format, dtype):
            if False:
                i = 10
                return i + 15
            input = torch.randn(shape, dtype=dtype, device=device)
            input = input.to(memory_format=memory_format).requires_grad_()
            if len(shape) == 4:
                pool = torch.nn.MaxPool2d(kernel_size, stride, return_indices=True).to(device)
            else:
                pool = torch.nn.MaxPool3d(kernel_size, stride, return_indices=True).to(device)
            input2 = input.detach().clone().float().requires_grad_(True)
            (out, ind) = pool(input)
            out.sum().backward()
            (out2, ind2) = pool(input2)
            out2.sum().backward()
            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(out.dtype, dtype)
            self.assertEqual(input.grad.dtype, dtype)
            self.assertEqual(out, out2.to(dtype=dtype))
            self.assertEqual(ind, ind2)
            self.assertEqual(input.grad, input2.grad.to(dtype=dtype))
        helper((4, 30, 8, 8), 7, 1, torch.contiguous_format, dtype)
        helper((4, 65, 8, 8), 7, 1, torch.channels_last, dtype)
        helper((1, 19, 20, 10), 8, 2, torch.contiguous_format, dtype)
        helper((1, 19, 20, 10), 8, 2, torch.channels_last, dtype)
        helper((4, 30, 8, 8), 7, 1, torch.contiguous_format, dtype)
        helper((4, 65, 8, 8), 7, 1, torch.channels_last, dtype)
        helper((1, 19, 10, 10, 10), 8, 2, torch.contiguous_format, dtype)
        helper((1, 19, 10, 9, 14), 8, 2, torch.channels_last_3d, dtype)
        helper((4, 10, 3, 8, 8), 3, 1, torch.contiguous_format, dtype)
        helper((4, 10, 8, 8, 8), 7, 1, torch.channels_last_3d, dtype)

    @onlyCUDA
    @gcIfJetson
    def test_max_pool2d_indices(self, device):
        if False:
            print('Hello World!')

        def helper(n, c, h, w, ks):
            if False:
                for i in range(10):
                    print('nop')
            if n is None:
                x = torch.randn(c, h, w, device='cuda', dtype=torch.float, requires_grad=True)
            else:
                x = torch.randn(n, c, h, w, device='cuda', dtype=torch.float, requires_grad=True)
            ref_x = x.detach().clone().cpu().requires_grad_()
            pool = torch.nn.MaxPool2d(kernel_size=ks, return_indices=True)
            (y, idx) = pool(x)
            (ref_y, ref_idx) = pool(ref_x)
            y.sum().backward()
            ref_y.sum().backward()
            self.assertEqual(y, ref_y)
            self.assertEqual(idx, ref_idx)
            self.assertEqual(x.grad, ref_x.grad)
        helper(2, 8, 4, 4, ks=2)
        helper(None, 3, 50, 50, ks=5)

    @onlyCPU
    def test_avg_pool2d_bfloat16(self, device):
        if False:
            return 10

        def helper(n, c, h, w, kernel_size, stride, memory_format):
            if False:
                return 10
            input = torch.randn(n, c, h, w, dtype=torch.float32, device=device).bfloat16()
            input = input.to(memory_format=memory_format).requires_grad_()
            pool = torch.nn.AvgPool2d(kernel_size, stride).to(device)
            input2 = input.detach().clone().float().requires_grad_(True)
            out = pool(input)
            out.sum().backward()
            out2 = pool(input2)
            out2.sum().backward()
            self.assertTrue(out.is_contiguous(memory_format=memory_format))
            self.assertEqual(out.dtype, torch.bfloat16)
            self.assertEqual(input.grad.dtype, torch.bfloat16)
            self.assertEqual(out, out2.bfloat16())
            self.assertEqual(input.grad, input2.grad.bfloat16())
        helper(4, 30, 8, 8, 7, 1, torch.contiguous_format)
        helper(4, 65, 8, 8, 7, 1, torch.channels_last)
        helper(1, 19, 20, 10, 8, 2, torch.contiguous_format)
        helper(1, 19, 20, 10, 8, 2, torch.channels_last)

    @dtypes(torch.float, torch.double)
    def test_adaptive_pooling_max_nhwc(self, device, dtype):
        if False:
            while True:
                i = 10

        def helper(n, c, h, w, output_height, output_width, contig):
            if False:
                while True:
                    i = 10
            input = torch.randint(1, 10, (n, c, h, w), device=device, dtype=dtype)
            input = input.contiguous(memory_format=torch.channels_last)
            grad = torch.randint(1, 10, (4, 8, output_height, output_width), device=device, dtype=dtype)
            grad = grad.contiguous(memory_format=torch.channels_last)
            if not contig:
                input = input[:, ::2, :, :]
                grad = grad[:, ::2, :, :]
            input.requires_grad_(True)
            pool = torch.nn.AdaptiveMaxPool2d((output_height, output_width), return_indices=True).to(device)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.AdaptiveMaxPool2d((output_height, output_width), return_indices=True).to(device)
            (out, ind) = pool(input)
            out.backward(grad)
            (ref_out, ref_ind) = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(ind, ref_ind)
            self.assertEqual(input.grad, ref_input.grad)
        for contig in [True, False]:
            helper(4, 8, 10, 10, 7, 7, contig)
            helper(4, 8, 9, 14, 5, 8, contig)
            helper(4, 8, 11, 11, 1, 1, contig)

    @dtypes(torch.float, torch.double)
    def test_pooling_max_nhwc(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')

        def helper(n, c, h, w, kernel_size, stride, padding, dilation, contig, device):
            if False:
                return 10
            output_height = math.floor((h + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
            output_width = math.floor((w + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1)
            input = torch.randint(1, 10, (n, c, h, w), device=device, dtype=dtype)
            input = input.contiguous(memory_format=torch.channels_last)
            grad = torch.randint(1, 10, (n, c, output_height, output_width), device=device, dtype=dtype)
            grad = grad.contiguous(memory_format=torch.channels_last)
            if not contig:
                input = input[:, ::2, :, :]
                grad = grad[:, ::2, :, :]
            input.requires_grad_(True)
            pool = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices=True, ceil_mode=False)
            ref_input = input.detach().clone().contiguous().requires_grad_(True)
            ref_grad = grad.detach().clone().contiguous()
            ref_pool = torch.nn.MaxPool2d(kernel_size, stride, padding, dilation, return_indices=True, ceil_mode=False).to(device)
            (out, ind) = pool(input)
            out.backward(grad)
            (ref_out, ref_ind) = ref_pool(ref_input)
            ref_out.backward(ref_grad)
            self.assertTrue(out.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_out.is_contiguous())
            self.assertTrue(ind.is_contiguous(memory_format=torch.channels_last))
            self.assertTrue(ref_ind.is_contiguous())
            self.assertEqual(out, ref_out)
            self.assertEqual(ind, ref_ind)
            self.assertEqual(input.grad, ref_input.grad)
        for contig in [True, False]:
            helper(4, 8, 10, 10, (2, 2), (1, 1), (1, 1), (2, 2), contig, device)
            helper(4, 8, 9, 14, (2, 2), (1, 1), (1, 1), (2, 2), contig, device)
            helper(4, 8, 11, 11, (4, 4), (2, 2), (2, 2), (2, 2), contig, device)

    @onlyCUDA
    def test_pool3d_size_one_feature_dim(self, device):
        if False:
            print('Hello World!')
        x = torch.randn(7, 1, 5, 3, 2, device=device)
        strange_strides = [30, 1234, 6, 2, 1]
        y = x.as_strided(x.size(), strange_strides)
        x = x.cpu().as_strided(x.size(), strange_strides)
        to_test = {'max_pool3d': lambda t: F.max_pool3d(t, (5, 1, 1), stride=(5, 1, 1)), 'avg_pool3d': lambda t: F.avg_pool3d(t, (5, 1, 1), stride=(5, 1, 1))}
        for (test, fn) in to_test.items():
            out_y = fn(y)
            out_x = fn(x)
            self.assertEqual(out_y, out_x.to(device), msg=test)

    @onlyCUDA
    @largeTensorTest('18GB')
    @largeTensorTest('180GB', 'cpu')
    def test_pool3d_large_size_int64(self, device):
        if False:
            return 10
        x = torch.randn(70, 32, 100, 100, 100, dtype=torch.half, device=device, requires_grad=True)
        y = torch.nn.functional.max_pool3d(x, 5)
        g = torch.randn_like(y, dtype=torch.half)
        torch.cuda.synchronize()
        y.backward(g)
        torch.cuda.synchronize()
        ref_x = x.detach().cpu().float()
        ref_x.requires_grad = True
        ref_g = g.cpu().float()
        ref_y = torch.nn.functional.max_pool3d(ref_x, 5)
        ref_y.backward(ref_g)
        self.assertEqual(y, ref_y, exact_dtype=False)
        self.assertEqual(x.grad, ref_x.grad, exact_dtype=False)

    @onlyCUDA
    def test_AvgPool3d_backward_after_cat_dim1_device(self, device):
        if False:
            while True:
                i = 10
        x = torch.randn(1, 3, 4, 4, 4, device=device, requires_grad=True)
        y = F.avg_pool3d(x, kernel_size=3, padding=1, stride=2)
        grad = torch.randn(y.size(), device=device)
        stride = list(grad.stride())
        stride[0] = stride[0] * 2
        grad.set_(grad.storage(), 0, grad.size(), stride)
        assert grad.is_contiguous()
        y.backward(grad)

    def _test_maxpool_indices(self, num_dim, adaptive=False, device='cpu', dtype=torch.float):
        if False:
            return 10

        def expected_indices(dim, dtype):
            if False:
                return 10
            if dim == 1:
                return torch.tensor([1, 3], dtype=dtype).repeat(2, 2, 1)
            if dim == 2:
                return torch.tensor([[5, 7], [13, 15]], dtype=dtype).repeat(2, 2, 1, 1)

        def expected_grad(dim, dtype):
            if False:
                for i in range(10):
                    print('nop')
            if dim == 1:
                return torch.tensor([0, 1, 0, 1], dtype=dtype).repeat(2, 2, 1)
            grad = expected_grad(dim - 1, dtype=dtype)
            zero = torch.zeros(grad.size(), dtype=dtype)
            return torch.stack((zero, grad, zero, grad), 2)

        def expected_output(dim, dtype):
            if False:
                while True:
                    i = 10
            if dim == 1:
                return torch.arange(2, 17, 2, dtype=dtype).view(2, 2, 2)
            if dim == 2:
                col = torch.arange(6, 63, 8, dtype=dtype)
                return torch.stack([col, col + 2], 1).view(2, 2, 2, 2)
        if adaptive:
            cls_name = 'AdaptiveMaxPool{}d'.format(num_dim)
        else:
            cls_name = 'MaxPool{}d'.format(num_dim)
        module_cls = getattr(nn, cls_name)
        module = module_cls(2, return_indices=True).to(device, dtype=dtype)
        numel = 4 ** (num_dim + 1)
        input = torch.arange(1, numel + 1).view(2, 2, *repeat(4, num_dim)).to(device, dtype=dtype)
        input_var = input.clone().detach().requires_grad_()
        (output, indices) = module(input_var)
        if num_dim != 3:
            expected_indices = expected_indices(num_dim, dtype=indices.data.dtype)
            expected_output = expected_output(num_dim, dtype=output.data.dtype)
            self.assertEqual(indices.dim(), input.dim())
            self.assertEqual(indices.data.squeeze(), expected_indices)
            self.assertEqual(output.data.squeeze(), expected_output)
        self.assertTrue(output.requires_grad)
        self.assertFalse(indices.requires_grad)
        grad_output = torch.ones(output.size(), device=device, dtype=dtype)
        output.backward(grad_output, retain_graph=True)
        expected_grad = expected_grad(num_dim, dtype=input_var.grad.data.dtype)
        self.assertEqual(input_var.grad.data, expected_grad.view_as(input))
        indices.add_(1)
        self.assertRaises(RuntimeError, lambda : output.backward(grad_output))
        t = torch.tensor([[[float('-inf')]]])
        m = nn.MaxPool1d(kernel_size=1, return_indices=True)
        (output, indices) = m(t)
        self.assertEqual(output[0, 0, 0], float('-inf'))
        self.assertEqual(indices[0, 0, 0], 0)
        t = torch.tensor([[[float('-inf')]]])
        m = nn.MaxPool2d(kernel_size=1, return_indices=True)
        (output, indices) = m(t)
        self.assertEqual(output[0, 0, 0], float('-inf'))
        self.assertEqual(indices[0, 0, 0], 0)
        t = torch.tensor([[[[float('-inf')]]]])
        m = nn.MaxPool3d(kernel_size=1, return_indices=True)
        (output, indices) = m(t)
        self.assertEqual(output[0, 0, 0, 0], float('-inf'))
        self.assertEqual(indices[0, 0, 0, 0], 0)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_MaxPool1d_indices(self, device, dtype):
        if False:
            i = 10
            return i + 15
        self._test_maxpool_indices(1, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_MaxPool2d_indices(self, device, dtype):
        if False:
            print('Hello World!')
        self._test_maxpool_indices(2, device=device, dtype=dtype)

    @skipIfMps
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_MaxPool3d_indices(self, device, dtype):
        if False:
            return 10
        self._test_maxpool_indices(3, device=device, dtype=dtype)

    @skipIfMps
    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @dtypes(torch.float)
    def test_AdaptiveMaxPool1d_indices(self, device, dtype):
        if False:
            while True:
                i = 10
        self._test_maxpool_indices(1, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_AdaptiveMaxPool2d_indices(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        self._test_maxpool_indices(2, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_AdaptiveMaxPool3d_indices(self, device, dtype):
        if False:
            i = 10
            return i + 15
        self._test_maxpool_indices(3, adaptive=True, device=device, dtype=dtype)

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_maxpool_indices_no_batch_dim(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        'Check that indices with no batch dim is consistent with a single batch.'
        max_pool_cases = [(nn.MaxPool1d(3, return_indices=True), torch.randn(3, 5, device=device, dtype=dtype)), (nn.MaxPool2d(3, return_indices=True), torch.randn(3, 5, 6, device=device, dtype=dtype)), (nn.MaxPool3d(3, return_indices=True), torch.randn(3, 5, 6, 7, device=device, dtype=dtype)), (nn.AdaptiveMaxPool1d(3, return_indices=True), torch.randn(3, 5, device=device, dtype=dtype)), (nn.AdaptiveMaxPool2d(3, return_indices=True), torch.randn(3, 5, 6, device=device, dtype=dtype)), (nn.AdaptiveMaxPool3d(3, return_indices=True), torch.randn(3, 5, 6, 7, device=device, dtype=dtype))]
        for (module, input) in max_pool_cases:
            (_, indices_no_batch) = module(input)
            (_, indicies_single_batch) = module(input.unsqueeze(0))
            self.assertEqual(indices_no_batch, indicies_single_batch.squeeze(0))

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @onlyNativeDeviceTypes
    @gcIfJetson
    def test_max_pool_nan_inf(self, device, dtype):
        if False:
            print('Hello World!')
        for adaptive in ['', 'adaptive_']:
            for num_dim in [1, 2, 3]:
                fn_name = f'{adaptive}max_pool{num_dim}d'
                fn = getattr(F, fn_name)
                x = torch.full([1, 1] + num_dim * [3], nan, device=device, dtype=dtype, requires_grad=True)
                res = fn(x, 1 if adaptive else 3)
                res.backward(torch.randn_like(res))
                self.assertTrue(math.isnan(res.item()))
                x.requires_grad_(False)
                res = fn(x, 1 if adaptive else 3)
                self.assertTrue(math.isnan(res.item()))
                x2 = torch.full([1, 1] + num_dim * [3], -inf, device=device, dtype=dtype, requires_grad=True)
                res2 = fn(x2, 1 if adaptive else 3)
                res2.backward(torch.randn_like(res2))
                self.assertTrue(math.isinf(res2.item()))
                x2.requires_grad_(False)
                res2 = fn(x2, 1 if adaptive else 3)
                self.assertTrue(math.isinf(res2.item()))

    @expectedFailureMeta
    @onlyNativeDeviceTypes
    def test_fractional_max_pool2d(self, device):
        if False:
            while True:
                i = 10
        with set_default_dtype(torch.double):
            x = torch.randn(1, 2, 7, 7, requires_grad=True, device=device)
            samples = x.new(1, 2, 2).uniform_()

            def func(x):
                if False:
                    while True:
                        i = 10
                return F.fractional_max_pool2d(x, (2, 2), output_size=(3, 3), _random_samples=samples)
            self.assertEqual(func(x).shape, (1, 2, 3, 3))
            gradcheck(func, [x])
            gradgradcheck(func, [x])
            x = torch.randn(2, 7, 7, requires_grad=True, device=device)
            self.assertEqual(func(x).shape, (2, 3, 3))
            if self.device_type != 'cuda':
                gradcheck(func, [x])
                gradgradcheck(func, [x])
            for kernel_size in [(), (1,)]:
                with self.assertRaisesRegex(RuntimeError, 'kernel_size must either'):
                    F.fractional_max_pool2d(x, kernel_size=kernel_size, output_size=(3, 3), _random_samples=samples)
            err_large_msg = 'too large relative to input '
            err_out_size_msg = 'output_size must either'
            for (output_size, msg) in [((9, 3), err_large_msg + 'height'), ((3, 9), err_large_msg + 'width'), ((3,), err_out_size_msg), ((), err_out_size_msg)]:
                with self.assertRaisesRegex(RuntimeError, msg):
                    F.fractional_max_pool2d(x, (2, 2), output_size=output_size, _random_samples=samples)

    @expectedFailureMeta
    @onlyNativeDeviceTypes
    def test_fractional_max_pool3d(self, device):
        if False:
            i = 10
            return i + 15
        with set_default_dtype(torch.double):
            x = torch.randn(1, 2, 7, 7, 7, requires_grad=True, device=device)
            samples = x.new(1, 2, 3).uniform_()

            def func(x):
                if False:
                    return 10
                return F.fractional_max_pool3d(x, (2, 2, 2), output_size=(3, 3, 3), _random_samples=samples)
            self.assertEqual(func(x).shape, (1, 2, 3, 3, 3))
            gradcheck(func, [x])
            gradgradcheck(func, [x])
            x = torch.randn(2, 7, 7, 7, requires_grad=True, device=device)
            self.assertEqual(func(x).shape, (2, 3, 3, 3))
            gradcheck(func, [x])
            gradgradcheck(func, [x])
            for kernel_size in [(), (1,), (1, 1)]:
                with self.assertRaisesRegex(RuntimeError, 'kernel_size must either'):
                    F.fractional_max_pool3d(x, kernel_size=kernel_size, output_size=(3, 3, 3), _random_samples=samples)
            err_large_msg = 'too large relative to input '
            err_out_size_msg = 'output_size must either'
            for (output_size, msg) in [((9, 3, 3), err_large_msg + 'time'), ((3, 9, 3), err_large_msg + 'height'), ((3, 3, 9), err_large_msg + 'width'), ((3, 3), err_out_size_msg), ((3,), err_out_size_msg), ((), err_out_size_msg)]:
                with self.assertRaisesRegex(RuntimeError, msg):
                    F.fractional_max_pool3d(x, (2, 2, 2), output_size=output_size, _random_samples=samples)

    @dtypesIfCUDA(torch.half, torch.float, torch.double)
    @dtypes(torch.float)
    @onlyNativeDeviceTypes
    def test_fractional_max_pool_nan_inf(self, device, dtype):
        if False:
            return 10
        for num_dim in [2, 3]:
            fn_name = f'FractionalMaxPool{num_dim}d'
            fn = getattr(nn, fn_name)(kernel_size=2, output_size=1)
            x = torch.full([1, 1] + num_dim * [3], nan, device=device, dtype=dtype, requires_grad=True)
            res = fn(x)
            res.backward(torch.randn_like(res))
            self.assertTrue(math.isnan(res.item()))
            x2 = torch.full([1, 1] + num_dim * [3], -inf, device=device, dtype=dtype, requires_grad=True)
            res2 = fn(x2)
            res2.backward(torch.randn_like(res2))
            self.assertTrue(math.isinf(res2.item()))

    @onlyNativeDeviceTypes
    def test_pooling_zero_stride(self, device):
        if False:
            print('Hello World!')
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = f'{op}_pool{num_dim}d'
                fn = getattr(F, fn_name)
                x = torch.ones([1, 2] + num_dim * [4], device=device, dtype=torch.float)
                self.assertRaisesRegex(RuntimeError, 'stride should not be zero|stride must be greater than zero', lambda : fn(x, kernel_size=2, stride=0))
                fn_module_name = f'{op.title()}Pool{num_dim}d'
                fn_module = getattr(nn, fn_module_name)(kernel_size=2, stride=0)
                self.assertRaisesRegex(RuntimeError, 'stride should not be zero|stride must be greater than zero', lambda : fn_module(x))

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_pool_large_size(self, device, dtype):
        if False:
            while True:
                i = 10
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = f'{op}_pool{num_dim}d'
                fn = getattr(F, fn_name)
                x = torch.ones([1, 1, 16777217] + (num_dim - 1) * [1], device=device, dtype=dtype)
                res = fn(x, 1, stride=1, padding=0)
                self.assertEqual(x.shape[2], res.shape[2])

    @onlyCUDA
    @largeTensorTest('6GB')
    def test_pooling_large(self, device):
        if False:
            return 10

        def helper(pool):
            if False:
                for i in range(10):
                    print('nop')
            inp = torch.randn(2 ** 7 + 10, 2 ** 8, 2 ** 8, 2 ** 8, dtype=torch.half, device='cuda')
            self.assertTrue(inp.numel() > 2 ** 31 - 1)
            out = pool(inp)
            torch.cuda.synchronize()
        helper(nn.MaxPool2d(4, 4))
        helper(nn.AvgPool2d(4, 4))
        helper(nn.FractionalMaxPool2d(4, 4))
        helper(nn.AdaptiveMaxPool2d((2 ** 6, 2 ** 6)))
        helper(nn.AdaptiveAvgPool2d((2 ** 6, 2 ** 6)))

    @dtypesIfCUDA(*floating_types_and(torch.half, torch.bfloat16))
    @skipIfMps
    @dtypes(torch.float)
    def test_pool_invalid_size(self, device, dtype):
        if False:
            while True:
                i = 10
        for op in ('max', 'avg'):
            for num_dim in [1, 2, 3]:
                fn_name = f'{op}_pool{num_dim}d'
                if op == 'max':
                    fn_name += '_with_indices'
                fn = getattr(F, fn_name)
                x = torch.ones([1, 1] + num_dim * [4], device=device, dtype=dtype)
                with self.assertRaisesRegex(RuntimeError, 'too small|smaller than'):
                    try:
                        res = fn(x, 3, stride=2, padding=0, dilation=2)
                    except TypeError:
                        res = fn(x, 6, stride=2, padding=0)

    @onlyCUDA
    def test_pooling_bfloat16(self, device):
        if False:
            while True:
                i = 10
        _test_bfloat16_ops(self, torch.nn.AvgPool1d(3, stride=2), device, inp_dims=(8, 4, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AvgPool2d(3, stride=2), device, inp_dims=(8, 4, 16, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AvgPool3d(3, stride=2), device, inp_dims=(8, 4, 16, 16, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AdaptiveAvgPool1d(3), device, inp_dims=(8, 4, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AdaptiveAvgPool2d((3, 5)), device, inp_dims=(8, 4, 16, 16), prec=0.05)
        _test_bfloat16_ops(self, torch.nn.AdaptiveAvgPool3d((3, 5, 7)), device, inp_dims=(8, 4, 16, 16, 16), prec=0.05)

    def test_maxpool3d_non_square_backward(self, device):
        if False:
            print('Hello World!')
        for dim in (2, 3, 4):
            shape = tuple((32 if i != dim else 256 for i in range(4)))
            x = torch.randn(shape, device=device, requires_grad=True)
            F.max_pool3d(x, kernel_size=(1, 1, 1)).sum().backward()
            self.assertEqual(x.grad, torch.ones_like(x.grad))

    @slowTest
    def test_adaptive_pool_odd_size(self, device):
        if False:
            return 10
        (Ih, Iw, Oh, Ow) = (5873, 3693, 3527, 2219)
        imgs = torch.randint(low=0, high=256, size=(11, Ih, Iw), dtype=torch.float)
        imgs_ = F.adaptive_avg_pool2d(imgs, (Oh, Ow))
        imgs_ = F.adaptive_max_pool2d(imgs, (Oh, Ow))
        (Id, Ih, Iw, Od, Oh, Ow) = (3, 5873, 3693, 3, 3527, 2219)
        imgs = torch.randint(low=0, high=256, size=(3, Id, Ih, Iw), dtype=torch.float)
        imgs_ = F.adaptive_avg_pool3d(imgs, (Od, Oh, Ow))
        imgs_ = F.adaptive_max_pool3d(imgs, (Od, Oh, Ow))
instantiate_device_type_tests(TestPoolingNNDeviceType, globals())
instantiate_parametrized_tests(TestPoolingNN)
if __name__ == '__main__':
    run_tests()