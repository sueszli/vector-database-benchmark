import pytest
import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature import TFeat
from kornia.testing import assert_close

class TestTFeat:

    def test_shape(self, device):
        if False:
            return 10
        inp = torch.ones(1, 1, 32, 32, device=device)
        tfeat = TFeat().to(device)
        tfeat.eval()
        out = tfeat(inp)
        assert out.shape == (1, 128)

    @pytest.mark.slow
    def test_pretrained(self, device):
        if False:
            while True:
                i = 10
        inp = torch.ones(1, 1, 32, 32, device=device)
        tfeat = TFeat(True).to(device)
        tfeat.eval()
        out = tfeat(inp)
        assert out.shape == (1, 128)

    def test_shape_batch(self, device):
        if False:
            print('Hello World!')
        inp = torch.ones(16, 1, 32, 32, device=device)
        tfeat = TFeat().to(device)
        out = tfeat(inp)
        assert out.shape == (16, 128)

    @pytest.mark.skip('jacobian not well computed')
    def test_gradcheck(self, device):
        if False:
            for i in range(10):
                print('nop')
        patches = torch.rand(2, 1, 32, 32, device=device)
        patches = utils.tensor_to_gradcheck_var(patches)
        tfeat = TFeat().to(patches.device, patches.dtype)
        assert gradcheck(tfeat, (patches,), eps=0.01, atol=0.01, raise_exception=True, fast_mode=True)

    @pytest.mark.slow
    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (2, 1, 32, 32)
        patches = torch.ones(B, C, H, W, device=device, dtype=dtype)
        tfeat = TFeat(True).to(patches.device, patches.dtype).eval()
        tfeat_jit = torch.jit.script(TFeat(True).to(patches.device, patches.dtype).eval())
        assert_close(tfeat_jit(patches), tfeat(patches))