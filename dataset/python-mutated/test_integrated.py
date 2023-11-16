import sys
import pytest
import torch
from torch import nn
from torch.autograd import gradcheck
import kornia
import kornia.testing as utils
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, KeyNetHardNet, LAFDescriptor, LocalFeature, ScaleSpaceDetector, SIFTDescriptor, SIFTFeature, extract_patches_from_pyramid, get_laf_center, get_laf_descriptors, get_laf_orientation, get_laf_scale
from kornia.feature.integrated import LocalFeatureMatcher
from kornia.geometry import RANSAC, resize, transform_points
from kornia.testing import assert_close
from kornia.utils._compat import torch_version_le

class TestGetLAFDescriptors:

    def test_same(self, device, dtype):
        if False:
            i = 10
            return i + 15
        (B, C, H, W) = (1, 3, 64, 64)
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        descs_test_from_rgb = get_laf_descriptors(img, lafs, sift, PS, True)
        descs_test_from_gray = get_laf_descriptors(img_gray, lafs, sift, PS, True)
        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        (B1, N1, CH1, H1, W1) = patches.size()
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        assert_close(descs_test_from_rgb, descs_reference)
        assert_close(descs_test_from_gray, descs_reference)

    def test_gradcheck(self, device, dtype=torch.float64):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 1, 32, 32)
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 2.0, W / 2.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 5.0, (H + W) / 6.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        img = utils.tensor_to_gradcheck_var(img)
        lafs = utils.tensor_to_gradcheck_var(lafs)

        class _MeanPatch(nn.Module):

            def forward(self, inputs):
                if False:
                    print('Hello World!')
                return inputs.mean(dim=(2, 3))
        desc = _MeanPatch()
        assert gradcheck(get_laf_descriptors, (img, lafs, desc, PS, True), eps=0.001, atol=0.001, raise_exception=True, nondet_tol=0.001, fast_mode=True)

class TestLAFDescriptor:

    def test_same(self, device, dtype):
        if False:
            return 10
        (B, C, H, W) = (1, 3, 64, 64)
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        img_gray = kornia.color.rgb_to_grayscale(img)
        centers = torch.tensor([[H / 3.0, W / 3.0], [2.0 * H / 3.0, W / 2.0]], device=device, dtype=dtype).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 4.0, (H + W) / 8.0], device=device, dtype=dtype).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device, dtype=dtype).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        sift = SIFTDescriptor(PS).to(device, dtype)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        patches = extract_patches_from_pyramid(img_gray, lafs, PS)
        (B1, N1, CH1, H1, W1) = patches.size()
        descs_reference = sift(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        assert_close(descs_test, descs_reference)

    def test_empty(self, device):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 1, 32, 32)
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        lafs = torch.zeros(B, 0, 2, 3, device=device)
        sift = SIFTDescriptor(PS).to(device)
        lafsift = LAFDescriptor(sift, PS)
        descs_test = lafsift(img, lafs)
        assert descs_test.shape == (B, 0, 128)

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        (B, C, H, W) = (1, 1, 32, 32)
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        centers = torch.tensor([[H / 2.0, W / 2.0], [2.0 * H / 3.0, W / 2.0]], device=device).view(1, 2, 2)
        scales = torch.tensor([(H + W) / 5.0, (H + W) / 6.0], device=device).view(1, 2, 1, 1)
        ori = torch.tensor([0.0, 30.0], device=device).view(1, 2, 1)
        lafs = kornia.feature.laf_from_center_scale_ori(centers, scales, ori)
        img = utils.tensor_to_gradcheck_var(img)
        lafs = utils.tensor_to_gradcheck_var(lafs)

        class _MeanPatch(nn.Module):

            def forward(self, inputs):
                if False:
                    while True:
                        i = 10
                return inputs.mean(dim=(2, 3))
        lafdesc = LAFDescriptor(_MeanPatch(), PS)
        assert gradcheck(lafdesc, (img, lafs), eps=0.001, atol=0.001, raise_exception=True, nondet_tol=0.001, fast_mode=True)

class TestLocalFeature:

    def test_smoke(self, device, dtype):
        if False:
            print('Hello World!')
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(32)
        local_feature = LocalFeature(det, desc).to(device, dtype)
        assert local_feature is not None

    def test_same(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 1, 64, 64)
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS)).to(device, dtype)
        (lafs, responses, descs) = local_feature(img)
        (lafs1, responses1) = det(img)
        assert_close(lafs, lafs1)
        assert_close(responses, responses1)
        patches = extract_patches_from_pyramid(img, lafs1, PS)
        (B1, N1, CH1, H1, W1) = patches.size()
        descs1 = desc(patches.view(B1 * N1, CH1, H1, W1)).view(B1, N1, -1)
        assert_close(descs, descs1)

    def test_scale(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 1, 64, 64)
        PS = 16
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        det = ScaleSpaceDetector(10)
        desc = SIFTDescriptor(PS)
        local_feature = LocalFeature(det, LAFDescriptor(desc, PS), 1.0).to(device, dtype)
        local_feature2 = LocalFeature(det, LAFDescriptor(desc, PS), 2.0).to(device, dtype)
        (lafs, responses, descs) = local_feature(img)
        (lafs2, responses2, descs2) = local_feature2(img)
        assert_close(get_laf_center(lafs), get_laf_center(lafs2))
        assert_close(get_laf_orientation(lafs), get_laf_orientation(lafs2))
        assert_close(2.0 * get_laf_scale(lafs), get_laf_scale(lafs2))

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        (B, C, H, W) = (1, 1, 32, 32)
        PS = 16
        img = torch.rand(B, C, H, W, device=device)
        img = utils.tensor_to_gradcheck_var(img)
        local_feature = LocalFeature(ScaleSpaceDetector(2), LAFDescriptor(SIFTDescriptor(PS), PS)).to(device, img.dtype)
        assert gradcheck(local_feature, img, eps=0.0001, atol=0.0001, nondet_tol=1e-08, raise_exception=True, fast_mode=True)

class TestSIFTFeature:

    def test_smoke(self, device, dtype):
        if False:
            while True:
                i = 10
        sift = SIFTFeature()
        assert sift is not None

    @pytest.mark.skip('jacobian not well computed')
    def test_gradcheck(self, device):
        if False:
            return 10
        (B, C, H, W) = (1, 1, 32, 32)
        img = torch.rand(B, C, H, W, device=device)
        local_feature = SIFTFeature(2, True).to(device).to(device)
        img = utils.tensor_to_gradcheck_var(img)
        assert gradcheck(local_feature, img, eps=0.0001, atol=0.0001, raise_exception=True)

class TestKeyNetHardNetFeature:

    def test_smoke(self, device, dtype):
        if False:
            i = 10
            return i + 15
        sift = KeyNetHardNet(2).to(device, dtype)
        (B, C, H, W) = (1, 1, 32, 32)
        img = torch.rand(B, C, H, W, device=device, dtype=dtype)
        out = sift(img)
        assert out is not None

    @pytest.mark.skip('jacobian not well computed')
    def test_gradcheck(self, device):
        if False:
            print('Hello World!')
        (B, C, H, W) = (1, 1, 32, 32)
        img = torch.rand(B, C, H, W, device=device)
        local_feature = KeyNetHardNet(2, True).to(device).to(device)
        img = utils.tensor_to_gradcheck_var(img)
        assert gradcheck(local_feature, img, eps=0.0001, atol=0.0001, raise_exception=True)

class TestGFTTAffNetHardNet:

    def test_smoke(self, device, dtype):
        if False:
            return 10
        feat = GFTTAffNetHardNet().to(device, dtype)
        assert feat is not None

    @pytest.mark.skip('jacobian not well computed')
    def test_gradcheck(self, device):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (1, 1, 32, 32)
        img = torch.rand(B, C, H, W, device=device)
        img = utils.tensor_to_gradcheck_var(img)
        local_feature = GFTTAffNetHardNet(2, True).to(device, img.dtype)
        assert gradcheck(local_feature, img, eps=0.0001, atol=0.0001, raise_exception=True)

class TestLocalFeatureMatcher:

    def test_smoke(self, device):
        if False:
            while True:
                i = 10
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher('snn', 0.8)).to(device)
        assert matcher is not None

    @pytest.mark.slow
    @pytest.mark.parametrize('data', ['loftr_homo'], indirect=True)
    def test_nomatch(self, device, dtype, data):
        if False:
            while True:
                i = 10
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(100), DescriptorMatcher('snn', 0.8)).to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        with torch.no_grad():
            out = matcher({'image0': data_dev['image0'], 'image1': 0 * data_dev['image0']})
        assert len(out['keypoints0']) == 0

    @pytest.mark.skip('Takes too long time (but works)')
    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        matcher = LocalFeatureMatcher(SIFTFeature(5), DescriptorMatcher('nn', 1.0)).to(device)
        patches = torch.rand(1, 1, 32, 32, device=device)
        patches05 = resize(patches, (48, 48))
        patches = utils.tensor_to_gradcheck_var(patches)
        patches05 = utils.tensor_to_gradcheck_var(patches05)

        def proxy_forward(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return matcher({'image0': x, 'image1': y})['keypoints0']
        assert gradcheck(proxy_forward, (patches, patches05), eps=0.0001, atol=0.0001, raise_exception=True, fast_mode=True)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason='Fails for bached torch.linalg.solve')
    @pytest.mark.parametrize('data', ['loftr_homo'], indirect=True)
    def test_real_sift(self, device, dtype, data):
        if False:
            for i in range(10):
                print('nop')
        torch.random.manual_seed(0)
        matcher = LocalFeatureMatcher(SIFTFeature(1000), DescriptorMatcher('snn', 0.8)).to(device, dtype)
        ransac = RANSAC('homography', 1.0, 1024, 5).to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        with torch.no_grad():
            out = matcher(data_dev)
        (homography, inliers) = ransac(out['keypoints0'], out['keypoints1'])
        assert inliers.sum().item() > 50
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=0.05, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason='Fails for bached torch.linalg.solve')
    @pytest.mark.parametrize('data', ['loftr_homo'], indirect=True)
    def test_real_sift_preextract(self, device, dtype, data):
        if False:
            print('Hello World!')
        torch.random.manual_seed(0)
        feat = SIFTFeature(1000).to(device, dtype)
        matcher = LocalFeatureMatcher(feat, DescriptorMatcher('snn', 0.8)).to(device)
        ransac = RANSAC('homography', 1.0, 1024, 5).to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        (lafs, _, descs) = feat(data_dev['image0'])
        data_dev['lafs0'] = lafs
        data_dev['descriptors0'] = descs
        (lafs2, _, descs2) = feat(data_dev['image1'])
        data_dev['lafs1'] = lafs2
        data_dev['descriptors1'] = descs2
        with torch.no_grad():
            out = matcher(data_dev)
        (homography, inliers) = ransac(out['keypoints0'], out['keypoints1'])
        assert inliers.sum().item() > 50
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=0.05, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason='Fails for bached torch.linalg.solve')
    @pytest.mark.skipif(sys.platform == 'win32', reason='this test takes so much memory in the CI with Windows')
    @pytest.mark.parametrize('data', ['loftr_homo'], indirect=True)
    def test_real_gftt(self, device, dtype, data):
        if False:
            print('Hello World!')
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000), DescriptorMatcher('snn', 0.8)).to(device, dtype)
        ransac = RANSAC('homography', 1.0, 1024, 5).to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        with torch.no_grad():
            torch.manual_seed(0)
            out = matcher(data_dev)
        (homography, inliers) = ransac(out['keypoints0'], out['keypoints1'])
        assert inliers.sum().item() > 50
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=0.05, atol=5)

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason='Fails for bached torch.linalg.solve')
    @pytest.mark.skipif(sys.platform == 'win32', reason='this test takes so much memory in the CI with Windows')
    @pytest.mark.parametrize('data', ['loftr_homo'], indirect=True)
    def test_real_keynet(self, device, dtype, data):
        if False:
            for i in range(10):
                print('nop')
        torch.random.manual_seed(0)
        matcher = LocalFeatureMatcher(KeyNetHardNet(500), DescriptorMatcher('snn', 0.9)).to(device, dtype)
        ransac = RANSAC('homography', 1.0, 1024, 5).to(device, dtype)
        data_dev = utils.dict_to(data, device, dtype)
        pts_src = data_dev['pts0']
        pts_dst = data_dev['pts1']
        with torch.no_grad():
            out = matcher(data_dev)
        (homography, inliers) = ransac(out['keypoints0'], out['keypoints1'])
        assert inliers.sum().item() > 50
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=0.05, atol=5)

    @pytest.mark.skip('ScaleSpaceDetector now is not jittable')
    def test_jit(self, device, dtype):
        if False:
            while True:
                i = 10
        (B, C, H, W) = (1, 1, 32, 32)
        patches = torch.rand(B, C, H, W, device=device, dtype=dtype)
        patches2x = resize(patches, (48, 48))
        inputs = {'image0': patches, 'image1': patches2x}
        model = LocalFeatureMatcher(SIFTDescriptor(32), DescriptorMatcher('snn', 0.8)).to(device).eval()
        model_jit = torch.jit.script(model)
        out = model(inputs)
        out_jit = model_jit(inputs)
        for (k, v) in out.items():
            assert_close(v, out_jit[k])