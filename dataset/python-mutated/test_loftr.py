import sys
import pytest
import torch
from torch.autograd import gradcheck
import kornia.testing as utils
from kornia.feature import LoFTR
from kornia.geometry import resize
from kornia.testing import assert_close
from kornia.utils._compat import torch_version_ge

class TestLoFTR:

    @pytest.mark.slow
    def test_pretrained_outdoor_smoke(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        loftr = LoFTR('outdoor').to(device, dtype)
        assert loftr is not None

    @pytest.mark.slow
    def test_pretrained_indoor_smoke(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        loftr = LoFTR('indoor').to(device, dtype)
        assert loftr is not None

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_ge(1, 10), reason='RuntimeError: CUDA out of memory with pytorch>=1.10')
    @pytest.mark.skipif(sys.platform == 'win32', reason='this test takes so much memory in the CI with Windows')
    @pytest.mark.parametrize('data', ['loftr_fund'], indirect=True)
    def test_pretrained_indoor(self, device, dtype, data):
        if False:
            i = 10
            return i + 15
        loftr = LoFTR('indoor').to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        assert_close(out['keypoints0'], data_dev['loftr_indoor_tentatives0'])
        assert_close(out['keypoints1'], data_dev['loftr_indoor_tentatives1'])

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_ge(1, 10), reason='RuntimeError: CUDA out of memory with pytorch>=1.10')
    @pytest.mark.skipif(sys.platform == 'win32', reason='this test takes so much memory in the CI with Windows')
    @pytest.mark.parametrize('data', ['loftr_homo'], indirect=True)
    def test_pretrained_outdoor(self, device, dtype, data):
        if False:
            for i in range(10):
                print('nop')
        loftr = LoFTR('outdoor').to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        with torch.no_grad():
            out = loftr(data_dev)
        assert_close(out['keypoints0'], data_dev['loftr_outdoor_tentatives0'])
        assert_close(out['keypoints1'], data_dev['loftr_outdoor_tentatives1'])

    @pytest.mark.slow
    def test_mask(self, device):
        if False:
            return 10
        patches = torch.rand(1, 1, 32, 32, device=device)
        mask = torch.rand(1, 32, 32, device=device)
        loftr = LoFTR().to(patches.device, patches.dtype)
        sample = {'image0': patches, 'image1': patches, 'mask0': mask, 'mask1': mask}
        with torch.no_grad():
            out = loftr(sample)
        assert out is not None

    @pytest.mark.slow
    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        patches = torch.rand(1, 1, 32, 32, device=device)
        patches05 = resize(patches, (48, 48))
        patches = utils.tensor_to_gradcheck_var(patches)
        patches05 = utils.tensor_to_gradcheck_var(patches05)
        loftr = LoFTR().to(patches.device, patches.dtype)

        def proxy_forward(x, y):
            if False:
                return 10
            return loftr.forward({'image0': x, 'image1': y})['keypoints0']
        assert gradcheck(proxy_forward, (patches, patches05), eps=0.0001, atol=0.0001, raise_exception=True, fast_mode=True)

    @pytest.mark.skip('does not like transformer.py:L99, zip iteration')
    def test_jit(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 1, 32, 32)
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        sample = {'image0': patches, 'image1': patches2x}
        model = LoFTR().to(patches.device, patches.dtype).eval()
        model_jit = torch.jit.script(model)
        out = model(sample)
        out_jit = model_jit(sample)
        for (k, v) in out.items():
            assert_close(v, out_jit[k])