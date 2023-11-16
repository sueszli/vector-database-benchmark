import math
import pytest
import torch
from torch.autograd import gradcheck
import kornia
from kornia.testing import BaseTester

class TestRgbToHsv(BaseTester):

    def test_smoke(self, device, dtype):
        if False:
            while True:
                i = 10
        (C, H, W) = (3, 4, 5)
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.rgb_to_hsv(img), torch.Tensor)

    @pytest.mark.parametrize('shape', [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        if False:
            print('Hello World!')
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.rgb_to_hsv(img).shape == shape

    def test_exception(self, device, dtype):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            assert kornia.color.rgb_to_hsv([0.0])
        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_hsv(img)
        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.rgb_to_hsv(img)

    def test_unit(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        data = torch.tensor([[[0.3944633, 0.8597369, 0.1670904, 0.2825457, 0.0953912], [0.1251704, 0.8020709, 0.8933256, 0.9170977, 0.1497008], [0.2711633, 0.1111478, 0.0783281, 0.2771807, 0.5487481], [0.0086008, 0.8288748, 0.9647092, 0.892202, 0.7614344], [0.2898048, 0.1282895, 0.7621747, 0.5657831, 0.9918593]], [[0.5414237, 0.9962701, 0.8947155, 0.5900949, 0.9483274], [0.0468036, 0.3933847, 0.8046577, 0.3640994, 0.06321], [0.6171775, 0.862478, 0.4126036, 0.7600935, 0.7279997], [0.4237089, 0.5365476, 0.5591233, 0.1523191, 0.1382165], [0.8932794, 0.8517839, 0.7152701, 0.8983801, 0.5905426]], [[0.286958, 0.4700376, 0.2743714, 0.8135023, 0.2229074], [0.930656, 0.3734594, 0.4566821, 0.7599275, 0.7557513], [0.7415742, 0.6115875, 0.3317572, 0.0379378, 0.131577], [0.8692724, 0.0809556, 0.7767404, 0.8742208, 0.1522012], [0.7708948, 0.4509611, 0.0481175, 0.2358997, 0.6900532]]], device=device, dtype=dtype)
        expected = torch.tensor([[[1.6519808, 1.3188975, 2.2487938, 3.582216, 2.250954], [4.28164, 0.04868213, 0.83454597, 5.533617, 4.319574], [3.4185164, 2.7919037, 2.8883224, 1.7474692, 1.3619272], [3.6837196, 0.6378961, 5.7213116, 5.2614374, 6.259687], [2.929221, 2.5614352, 0.97840965, 1.5729411, 6.0235224]], [[0.4699935, 0.52820253, 0.8132473, 0.65267974, 0.899411], [0.9497089, 0.534381, 0.48878422, 0.60298723, 0.9163612], [0.6343409, 0.87112963, 0.8101612, 0.9500878, 0.8192622], [0.9901055, 0.9023306, 0.42042294, 0.8292772, 0.81847864], [0.6755719, 0.8493871, 0.93686795, 0.73741645, 0.40461043]], [[0.5414237, 0.99627006, 0.89471555, 0.81350225, 0.9483274], [0.930656, 0.80207086, 0.8933256, 0.9170977, 0.75575125], [0.7415741, 0.86247796, 0.41260356, 0.76009345, 0.7279997], [0.8692723, 0.8288748, 0.9647092, 0.892202, 0.7614344], [0.8932794, 0.8517839, 0.7621747, 0.8983801, 0.99185926]]], device=device, dtype=dtype)
        self.assert_close(kornia.color.rgb_to_hsv(data), expected)

    def test_nan_rgb_to_hsv(self, device, dtype):
        if False:
            i = 10
            return i + 15
        if dtype == torch.float16:
            pytest.skip('not work for half-precision')
        data = torch.zeros(3, 5, 5, device=device, dtype=dtype)
        expected = torch.zeros_like(data)
        self.assert_close(kornia.color.rgb_to_hsv(data), expected)

    def test_gradcheck(self, device, dtype):
        if False:
            while True:
                i = 10
        (B, C, H, W) = (2, 3, 4, 4)
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.rgb_to_hsv, (img,), raise_exception=True, fast_mode=True)

    def test_jit(self, device, dtype):
        if False:
            while True:
                i = 10
        (B, C, H, W) = (2, 3, 4, 4)
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.rgb_to_hsv
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (2, 3, 4, 4)
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.RgbToHsv().to(device, dtype)
        fcn = kornia.color.rgb_to_hsv
        self.assert_close(ops(img), fcn(img))

class TestHsvToRgb(BaseTester):

    def test_smoke(self, device, dtype):
        if False:
            print('Hello World!')
        (C, H, W) = (3, 4, 5)
        img = torch.rand(C, H, W, device=device, dtype=dtype)
        assert isinstance(kornia.color.hsv_to_rgb(img), torch.Tensor)

    @pytest.mark.parametrize('shape', [(1, 3, 4, 4), (2, 3, 2, 4), (3, 3, 4, 1), (3, 2, 1)])
    def test_cardinality(self, device, dtype, shape):
        if False:
            for i in range(10):
                print('nop')
        img = torch.ones(shape, device=device, dtype=dtype)
        assert kornia.color.hsv_to_rgb(img).shape == shape

    def test_exception(self, device, dtype):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError):
            assert kornia.color.hsv_to_rgb([0.0])
        with pytest.raises(ValueError):
            img = torch.ones(1, 1, device=device, dtype=dtype)
            assert kornia.color.hsv_to_rgb(img)
        with pytest.raises(ValueError):
            img = torch.ones(2, 1, 1, device=device, dtype=dtype)
            assert kornia.color.hsv_to_rgb(img)

    def test_unit(self, device, dtype):
        if False:
            return 10
        data = torch.tensor([[[[3.5433271, 5.6390061, 1.3766849, 2.5384088, 4.6848912], [5.7209363, 5.326263, 6.2059994, 4.1164689, 2.38726], [0.6370091, 3.6186798, 5.9170871, 2.8275447, 5.4289737], [0.2751994, 1.6632686, 1.0049511, 0.7046204, 1.3791083], [0.7863123, 4.4852505, 4.3064494, 2.5573561, 5.9083076]], [[0.5026655, 0.9453601, 0.5929778, 0.2632897, 0.4590443], [0.6201433, 0.5610679, 0.965326, 0.0830478, 0.5000827], [0.6067343, 0.6422323, 0.677794, 0.7705711, 0.6050767], [0.5495264, 0.5573426, 0.4683768, 0.2268902, 0.2116482], [0.6525245, 0.0022379, 0.490998, 0.1682271, 0.6327152]], [[0.847168, 0.9302199, 0.3265766, 0.794457, 0.7038843], [0.4833369, 0.2088473, 0.1169234, 0.4966302, 0.6448684], [0.2713015, 0.589338, 0.6015301, 0.6801558, 0.2322258], [0.5704236, 0.6797268, 0.4755683, 0.4811209, 0.5317836], [0.3236262, 0.0999796, 0.3614958, 0.5117705, 0.8194097]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[0.4213259, 0.93021995, 0.26564622, 0.58528465, 0.5338429], [0.48333693, 0.20884734, 0.11692339, 0.45538613, 0.32238087], [0.2713015, 0.2108461, 0.60153013, 0.15604737, 0.23222584], [0.5704236, 0.4568531, 0.4755683, 0.48112088, 0.49611038], [0.32362622, 0.09981924, 0.20394461, 0.42567685, 0.81940967]], [[0.6838029, 0.0508271, 0.3265766, 0.794457, 0.3807702], [0.18359877, 0.0916698, 0.00405421, 0.45823452, 0.6448684], [0.20682439, 0.41690278, 0.1938166, 0.68015575, 0.0917114], [0.33933756, 0.6797268, 0.4665822, 0.44541004, 0.5317836], [0.27101707, 0.09975589, 0.18400209, 0.51177055, 0.30095676]], [[0.84716797, 0.5917818, 0.13292392, 0.6739741, 0.7038843], [0.34453064, 0.19874583, 0.01237347, 0.4966302, 0.41256943], [0.10669357, 0.589338, 0.3363524, 0.5229789, 0.20633064], [0.25696078, 0.30088606, 0.25282317, 0.37195927, 0.41923255], [0.11245217, 0.09997964, 0.3614958, 0.46373847, 0.4865534]]]], device=device, dtype=dtype)
        f = kornia.color.hsv_to_rgb
        self.assert_close(f(data), expected)
        data[:, 0] += 2 * math.pi
        self.assert_close(f(data), expected, low_tolerance=True)
        data[:, 0] -= 4 * math.pi
        self.assert_close(f(data), expected, low_tolerance=True)

    @pytest.mark.grad()
    def test_gradcheck(self, device, dtype):
        if False:
            print('Hello World!')
        (B, C, H, W) = (2, 3, 4, 4)
        img = torch.rand(B, C, H, W, device=device, dtype=torch.float64, requires_grad=True)
        assert gradcheck(kornia.color.hsv_to_rgb, (img,), raise_exception=True, fast_mode=True)

    @pytest.mark.jit()
    def test_jit(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, C, H, W) = (2, 3, 4, 4)
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        op = kornia.color.hsv_to_rgb
        op_jit = torch.jit.script(op)
        self.assert_close(op(img), op_jit(img))

    def test_module(self, device, dtype):
        if False:
            i = 10
            return i + 15
        (B, C, H, W) = (2, 3, 4, 4)
        img = torch.ones(B, C, H, W, device=device, dtype=dtype)
        ops = kornia.color.HsvToRgb().to(device, dtype)
        fcn = kornia.color.hsv_to_rgb
        self.assert_close(ops(img), fcn(img))