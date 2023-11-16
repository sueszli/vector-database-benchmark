import pytest
import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature import DeFMO
from kornia.testing import assert_close

class TestDeFMO:

    @pytest.mark.slow
    def test_shape(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.ones(1, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        defmo.eval()
        out = defmo(inp)
        assert out.shape == (1, 24, 4, 128, 160)

    @pytest.mark.slow
    def test_shape_batch(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.ones(2, 6, 128, 160, device=device, dtype=dtype)
        defmo = DeFMO().to(device, dtype)
        out = defmo(inp)
        with torch.no_grad():
            assert out.shape == (2, 24, 4, 128, 160)

    @pytest.mark.slow
    @pytest.mark.grad
    def test_gradcheck(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        patches = torch.rand(2, 6, 64, 64, device=device, dtype=dtype)
        patches = utils.tensor_to_gradcheck_var(patches)
        defmo = DeFMO().to(patches.device, patches.dtype)
        assert gradcheck(defmo, (patches,), eps=0.0001, atol=0.0001, nondet_tol=1e-08, raise_exception=True, fast_mode=True)

    @pytest.mark.slow
    @pytest.mark.jit
    def test_jit(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 6, 128, 160)
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        model = DeFMO(True).to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(DeFMO(True).to(patches.device, patches.dtype).eval())
        with torch.no_grad():
            assert_close(model(patches), model_jit(patches))