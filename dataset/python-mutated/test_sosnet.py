import pytest
import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature import SOSNet
from kornia.testing import assert_close

class TestSOSNet:

    def test_shape(self, device):
        if False:
            return 10
        inp = torch.ones(1, 1, 32, 32, device=device)
        sosnet = SOSNet(pretrained=False).to(device)
        sosnet.eval()
        out = sosnet(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        if False:
            while True:
                i = 10
        inp = torch.ones(16, 1, 32, 32, device=device)
        sosnet = SOSNet(pretrained=False).to(device)
        out = sosnet(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip('jacobian not well computed')
    def test_gradcheck(self, device):
        if False:
            return 10
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        sosnet = SOSNet(pretrained=False).to(patches.device, patches.dtype)
        assert gradcheck(sosnet, (patches,), eps=0.0001, atol=0.0001, raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if False:
            while True:
                i = 10
        (B, C, H, W) = (2, 1, 32, 32)
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        model = SOSNet().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(SOSNet().to(patches.device, patches.dtype).eval())
        assert_close(model(patches), model_jit(patches))