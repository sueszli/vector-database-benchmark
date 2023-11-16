import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature.affine_shape import LAFAffineShapeEstimator, LAFAffNetShapeEstimator, PatchAffineShapeEstimator
from kornia.testing import assert_close

class TestPatchAffineShapeEstimator:

    def test_shape(self, device):
        if False:
            print('Hello World!')
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1, 1, 3])

    def test_shape_batch(self, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.rand(2, 1, 32, 32, device=device)
        ori = PatchAffineShapeEstimator(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([2, 1, 3])

    def test_print(self, device):
        if False:
            i = 10
            return i + 15
        sift = PatchAffineShapeEstimator(32)
        sift.__repr__()

    def test_toy(self, device):
        if False:
            for i in range(10):
                print('nop')
        aff = PatchAffineShapeEstimator(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, 5:-5, 1:-1] = 1
        abc = aff(inp)
        expected = torch.tensor([[[0.4146, 0.0, 1.0]]], device=device)
        assert_close(abc, expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        (batch_size, channels, height, width) = (1, 1, 13, 13)
        ori = PatchAffineShapeEstimator(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        assert gradcheck(ori, (patches,), raise_exception=True, nondet_tol=0.0001, fast_mode=True)

class TestLAFAffineShapeEstimator:

    def test_shape(self, device):
        if False:
            i = 10
            return i + 15
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        if False:
            print('Hello World!')
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFAffineShapeEstimator().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        if False:
            while True:
                i = 10
        sift = LAFAffineShapeEstimator()
        sift.__repr__()

    def test_toy(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[35.078, 0.0, 16.0], [0.0, 11.403, 16.0]]]], device=device, dtype=dtype)
        assert_close(new_laf, expected, atol=0.0001, rtol=0.0001)

    def test_toy_preserve(self, device, dtype):
        if False:
            while True:
                i = 10
        aff = LAFAffineShapeEstimator(32, preserve_orientation=True).to(device, dtype)
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[0.0, 35.078, 16.0], [-11.403, 0, 16.0]]]], device=device, dtype=dtype)
        assert_close(new_laf, expected, atol=0.0001, rtol=0.0001)

    def test_toy_not_preserve(self, device):
        if False:
            for i in range(10):
                print('nop')
        aff = LAFAffineShapeEstimator(32, preserve_orientation=False).to(device)
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[0.0, 20.0, 16.0], [-20.0, 0.0, 16.0]]]], device=device)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[35.078, 0, 16.0], [0, 11.403, 16.0]]]], device=device)
        assert_close(new_laf, expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            print('Hello World!')
        (batch_size, channels, height, width) = (1, 1, 40, 40)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        laf = torch.tensor([[[[5.0, 0.0, 26.0], [0.0, 5.0, 26.0]]]], device=device)
        laf = utils.tensor_to_gradcheck_var(laf)
        assert gradcheck(LAFAffineShapeEstimator(11).to(device), (laf, patches), raise_exception=True, rtol=0.001, atol=0.001, nondet_tol=0.0001, fast_mode=True)

class TestLAFAffNetShapeEstimator:

    def test_shape(self, device):
        if False:
            i = 10
            return i + 15
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(False).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_pretrained(self, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator(True).to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        if False:
            while True:
                i = 10
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 5, 2, 3, device=device)
        ori = LAFAffNetShapeEstimator().to(device).eval()
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        if False:
            while True:
                i = 10
        sift = LAFAffNetShapeEstimator()
        sift.__repr__()

    def test_toy(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        aff = LAFAffNetShapeEstimator(True).to(device, dtype).eval()
        inp = torch.zeros(1, 1, 32, 32, device=device, dtype=dtype)
        inp[:, :, 15:-15, 9:-9] = 1
        laf = torch.tensor([[[[20.0, 0.0, 16.0], [0.0, 20.0, 16.0]]]], device=device, dtype=dtype)
        new_laf = aff(laf, inp)
        expected = torch.tensor([[[[33.2073, 0.0, 16.0], [-1.3766, 12.0456, 16.0]]]], device=device, dtype=dtype)
        assert_close(new_laf, expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        (batch_size, channels, height, width) = (1, 1, 35, 35)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        laf = torch.tensor([[[[8.0, 0.0, 16.0], [0.0, 8.0, 16.0]]]], device=device)
        laf = utils.tensor_to_gradcheck_var(laf)
        assert gradcheck(LAFAffNetShapeEstimator(True).to(device, dtype=patches.dtype), (laf, patches), raise_exception=True, rtol=0.001, atol=0.001, nondet_tol=0.0001, fast_mode=True)