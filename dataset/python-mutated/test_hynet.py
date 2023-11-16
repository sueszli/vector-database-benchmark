import pytest
import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature import HyNet
from kornia.testing import assert_close

class TestHyNet:

    def test_shape(self, device):
        if False:
            i = 10
            return i + 15
        inp = torch.ones(1, 1, 32, 32, device=device)
        hynet = HyNet().to(device)
        out = hynet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        if False:
            while True:
                i = 10
        inp = torch.ones(16, 1, 32, 32, device=device)
        hynet = HyNet().to(device)
        out = hynet(inp)
        assert out.shape == (16, 128)

    def test_gradcheck(self, device):
        if False:
            return 10
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        hynet = HyNet().to(patches.device, patches.dtype)
        assert gradcheck(hynet, (patches,), eps=0.0001, atol=0.0001, nondet_tol=1e-08, raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if False:
            return 10
        (B, C, H, W) = (2, 1, 32, 32)
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = HyNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        assert_close(model(patches), model_jit(patches))