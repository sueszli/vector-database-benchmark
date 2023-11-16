import pytest
import torch
from kornia.feature import DescriptorMatcher, GFTTAffNetHardNet, LocalFeatureMatcher, SIFTFeature
from kornia.geometry import rescale, transform_points
from kornia.testing import assert_close
from kornia.tracking import HomographyTracker
from kornia.utils._compat import torch_version_le

@pytest.fixture()
def data_url():
    if False:
        for i in range(10):
            print('nop')
    url = 'https://github.com/kornia/data_test/blob/main/loftr_outdoor_and_homography_data.pt?raw=true'
    return url

class TestHomographyTracker:

    @pytest.mark.slow
    def test_smoke(self, device):
        if False:
            print('Hello World!')
        tracker = HomographyTracker().to(device)
        assert tracker is not None

    @pytest.mark.slow
    def test_nomatch(self, device, dtype, data_url):
        if False:
            for i in range(10):
                print('nop')
        data = torch.hub.load_state_dict_from_url(data_url)
        matcher = LocalFeatureMatcher(SIFTFeature(100), DescriptorMatcher('smnn', 0.95)).to(device, dtype)
        tracker = HomographyTracker(matcher, matcher, minimum_inliers_num=100)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        tracker.set_target(data['image0'])
        torch.random.manual_seed(0)
        (_, success) = tracker(torch.zeros_like(data['image0']))
        assert not success

    @pytest.mark.slow
    @pytest.mark.skipif(torch_version_le(1, 9, 1), reason='Fails for bached torch.linalg.solve')
    def test_real(self, device, dtype, data_url):
        if False:
            print('Hello World!')
        data = torch.hub.load_state_dict_from_url(data_url)
        for k in data.keys():
            if isinstance(data[k], torch.Tensor):
                data[k] = data[k].to(device, dtype)
        data['image0'] = rescale(data['image0'], 0.5, interpolation='bilinear', align_corners=False)
        data['image1'] = rescale(data['image1'], 0.5, interpolation='bilinear', align_corners=False)
        matcher = LocalFeatureMatcher(GFTTAffNetHardNet(1000), DescriptorMatcher('snn', 0.8)).to(device, dtype)
        torch.manual_seed(8)
        tracker = HomographyTracker(matcher, matcher).to(device, dtype)
        with torch.no_grad():
            tracker.set_target(data['image0'])
            torch.manual_seed(8)
            (homography, success) = tracker(data['image1'])
        assert success
        pts_src = data['pts0'].to(device, dtype) / 2.0
        pts_dst = data['pts1'].to(device, dtype) / 2.0
        assert_close(transform_points(homography[None], pts_src[None]), pts_dst[None], rtol=0.05, atol=5)