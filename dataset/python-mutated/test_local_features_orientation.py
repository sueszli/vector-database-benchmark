import pytest
import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature.orientation import LAFOrienter, OriNet, PassLAF, PatchDominantGradientOrientation
from kornia.geometry.conversions import rad2deg
from kornia.testing import assert_close

class TestPassLAF:

    def test_shape(self, device):
        if False:
            return 10
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        if False:
            i = 10
            return i + 15
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        if False:
            while True:
                i = 10
        sift = PassLAF()
        sift.__repr__()

    def test_pass(self, device):
        if False:
            return 10
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = PassLAF().to(device)
        out = ori(laf, inp)
        assert_close(out, laf)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        (batch_size, channels, height, width) = (1, 1, 21, 21)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        laf = torch.rand(batch_size, 4, 2, 3)
        assert gradcheck(PassLAF().to(device), (patches, laf), raise_exception=True, fast_mode=True)

class TestPatchDominantGradientOrientation:

    def test_shape(self, device):
        if False:
            while True:
                i = 10
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = PatchDominantGradientOrientation(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    def test_shape_batch(self, device):
        if False:
            return 10
        inp = torch.rand(10, 1, 32, 32, device=device)
        ori = PatchDominantGradientOrientation(32).to(device)
        ang = ori(inp)
        assert ang.shape == torch.Size([10])

    def test_print(self, device):
        if False:
            for i in range(10):
                print('nop')
        sift = PatchDominantGradientOrientation(32)
        sift.__repr__()

    def test_toy(self, device):
        if False:
            while True:
                i = 10
        ori = PatchDominantGradientOrientation(19).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, :10, :] = 1
        ang = ori(inp)
        expected = torch.tensor([90.0], device=device)
        assert_close(rad2deg(ang), expected)

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        (batch_size, channels, height, width) = (1, 1, 13, 13)
        ori = PatchDominantGradientOrientation(width).to(device)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        assert gradcheck(ori, (patches,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    @pytest.mark.skip(" Compiled functions can't take variable number")
    def test_jit(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (2, 1, 13, 13)
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = PatchDominantGradientOrientation(13).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(PatchDominantGradientOrientation(13).to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))

class TestOriNet:

    def test_shape(self, device):
        if False:
            print('Hello World!')
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = OriNet().to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    def test_pretrained(self, device):
        if False:
            print('Hello World!')
        inp = torch.rand(1, 1, 32, 32, device=device)
        ori = OriNet(True).to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        assert ang.shape == torch.Size([1])

    def test_shape_batch(self, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.rand(2, 1, 32, 32, device=device)
        ori = OriNet(True).to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        assert ang.shape == torch.Size([2])

    def test_print(self, device):
        if False:
            return 10
        sift = OriNet(32)
        sift.__repr__()

    def test_toy(self, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.zeros(1, 1, 32, 32, device=device)
        inp[:, :, :16, :] = 1
        ori = OriNet(True).to(device=device, dtype=inp.dtype).eval()
        ang = ori(inp)
        expected = torch.tensor([70.58], device=device)
        assert_close(rad2deg(ang), expected, atol=0.01, rtol=0.001)

    @pytest.mark.skip('jacobian not well computed')
    def test_gradcheck(self, device):
        if False:
            return 10
        (batch_size, channels, height, width) = (2, 1, 32, 32)
        patches = torch.rand(batch_size, channels, height, width, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        ori = OriNet().to(device=device, dtype=patches.dtype)
        assert gradcheck(ori, (patches,), raise_exception=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if False:
            print('Hello World!')
        (B, C, H, W) = (2, 1, 32, 32)
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = OriNet(True).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(OriNet(True).to(patches.device, patches.dtype).eval())
        assert_close(tfeat_jit(patches), tfeat(patches))

class TestLAFOrienter:

    def test_shape(self, device):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.rand(1, 1, 32, 32, device=device)
        laf = torch.rand(1, 1, 2, 3, device=device)
        ori = LAFOrienter().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_shape_batch(self, device):
        if False:
            i = 10
            return i + 15
        inp = torch.rand(2, 1, 32, 32, device=device)
        laf = torch.rand(2, 34, 2, 3, device=device)
        ori = LAFOrienter().to(device)
        out = ori(laf, inp)
        assert out.shape == laf.shape

    def test_print(self, device):
        if False:
            print('Hello World!')
        sift = LAFOrienter()
        sift.__repr__()

    def test_toy(self, device):
        if False:
            while True:
                i = 10
        ori = LAFOrienter(32).to(device)
        inp = torch.zeros(1, 1, 19, 19, device=device)
        inp[:, :, :, :10] = 1
        laf = torch.tensor([[[[5.0, 0.0, 8.0], [0.0, 5.0, 8.0]]]], device=device)
        new_laf = ori(laf, inp)
        expected = torch.tensor([[[[-5.0, 0.0, 8.0], [0.0, -5.0, 8.0]]]], device=device)
        assert_close(new_laf, expected)

    def test_gradcheck(self, device):
        if False:
            return 10
        (batch_size, channels, height, width) = (1, 1, 21, 21)
        patches = torch.rand(batch_size, channels, height, width, device=device).float()
        patches = utils.tensor_to_gradcheck_var(patches)
        laf = torch.ones(batch_size, 2, 2, 3, device=device).float()
        laf[:, :, 0, 1] = 0
        laf[:, :, 1, 0] = 0
        laf = utils.tensor_to_gradcheck_var(laf)
        assert gradcheck(LAFOrienter(8).to(device), (laf, patches), raise_exception=True, rtol=0.001, atol=0.001, fast_mode=True)