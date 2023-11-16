import pytest
import torch
from torch.autograd import gradcheck
import kornia
import kornia.testing as utils
from kornia.testing import assert_close
from kornia.utils._compat import torch_version

class TestCropAndResize3D:

    def test_crop(self, device, dtype):
        if False:
            return 10
        inp = torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        (depth, height, width) = (2, 2, 2)
        expected = torch.tensor([[[[[25.1667, 27.1667], [30.5, 32.5]], [[46.5, 48.5], [51.8333, 53.8333]]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_and_resize3d(inp, boxes, (depth, height, width))
        assert_close(patches, expected)

    def test_crop_batch(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.cat([torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4), torch.arange(0.0, 128.0, step=2, device=device, dtype=dtype).view(1, 1, 4, 4, 4)], dim=0)
        (depth, height, width) = (2, 2, 2)
        expected = torch.tensor([[[[[16.0, 19.0], [24.0, 27.0]], [[48.0, 51.0], [56.0, 59.0]]]], [[[[0.0, 6.0], [16.0, 22.0]], [[64.0, 70.0], [80.0, 86.0]]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]], [[0, 0, 0], [3, 0, 0], [3, 2, 0], [0, 2, 0], [0, 0, 2], [3, 0, 2], [3, 2, 2], [0, 2, 2]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_and_resize3d(inp, boxes, (depth, height, width), align_corners=True)
        assert_close(patches, expected)

    def test_gradcheck(self, device, dtype):
        if False:
            while True:
                i = 10
        img = torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        img = utils.tensor_to_gradcheck_var(img)
        boxes = torch.tensor([[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]], device=device, dtype=dtype)
        boxes = utils.tensor_to_gradcheck_var(boxes, requires_grad=False)
        assert gradcheck(kornia.geometry.transform.crop_and_resize3d, (img, boxes, (4, 3, 2)), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if False:
            while True:
                i = 10
        op = kornia.geometry.transform.crop_and_resize3d
        op_script = torch_optimizer(op)
        img = torch.arange(0.0, 64.0, device=device, dtype=dtype).view(1, 1, 4, 4, 4)
        boxes = torch.tensor([[[0, 0, 1], [3, 0, 1], [3, 2, 1], [0, 2, 1], [0, 0, 3], [3, 0, 3], [3, 2, 3], [0, 2, 3]]], device=device, dtype=dtype)
        actual = op_script(img, boxes, (4, 3, 2))
        expected = op(img, boxes, (4, 3, 2))
        assert_close(actual, expected)

class TestCenterCrop3D:

    @pytest.mark.parametrize('crop_size', [(3, 5, 7), (5, 3, 7), (7, 3, 5)])
    def test_center_crop_357(self, crop_size, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        expected = inp[:, :, inp.size(2) // 2 - crop_size[0] // 2:inp.size(2) // 2 + crop_size[0] // 2 + 1, inp.size(3) // 2 - crop_size[1] // 2:inp.size(3) // 2 + crop_size[1] // 2 + 1, inp.size(4) // 2 - crop_size[2] // 2:inp.size(4) // 2 + crop_size[2] // 2 + 1]
        out_crop = kornia.geometry.transform.center_crop3d(inp, crop_size, align_corners=True)
        assert_close(out_crop, expected, rtol=0.0001, atol=0.0001)

    @pytest.mark.parametrize('crop_size', [(3, 5, 7), (5, 3, 7), (7, 3, 5)])
    def test_center_crop_357_batch(self, crop_size, device, dtype):
        if False:
            return 10
        inp = torch.cat([torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7), torch.arange(343.0, 686.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)])
        expected = inp[:, :, inp.size(2) // 2 - crop_size[0] // 2:inp.size(2) // 2 + crop_size[0] // 2 + 1, inp.size(3) // 2 - crop_size[1] // 2:inp.size(3) // 2 + crop_size[1] // 2 + 1, inp.size(4) // 2 - crop_size[2] // 2:inp.size(4) // 2 + crop_size[2] // 2 + 1]
        out_crop = kornia.geometry.transform.center_crop3d(inp, crop_size, align_corners=True)
        assert_close(out_crop, expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            i = 10
            return i + 15
        img = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        img = utils.tensor_to_gradcheck_var(img)
        assert gradcheck(kornia.geometry.transform.center_crop3d, (img, (3, 5, 7)), raise_exception=True, fast_mode=True)

    @pytest.mark.skipif(torch_version() == '2.1.0', reason="https://github.com/pytorch/pytorch/issues/110680 -  unsupported operand type(s) for @: 'FakeTensor' and 'FakeTensor' on `normalize_homography3d`")
    def test_dynamo(self, device, dtype, torch_optimizer):
        if False:
            print('Hello World!')
        op = kornia.geometry.transform.center_crop3d
        op_script = torch_optimizer(op)
        img = torch.ones(4, 3, 5, 6, 7, device=device, dtype=dtype)
        actual = op_script(img, (4, 3, 2))
        expected = kornia.geometry.transform.center_crop3d(img, (4, 3, 2))
        assert_close(actual, expected, rtol=0.0001, atol=0.0001)

class TestCropByBoxes3D:

    def test_crop_by_boxes_no_resizing(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        src_box = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0], [3.0, 1.0, 2.0], [3.0, 3.0, 2.0], [1.0, 3.0, 2.0]]], device=device, dtype=dtype)
        dst_box = torch.tensor([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 2.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0], [2.0, 0.0, 1.0], [2.0, 2.0, 1.0], [0.0, 2.0, 1.0]]], device=device, dtype=dtype)
        expected = inp[:, :, 1:3, 1:4, 1:4]
        patches = kornia.geometry.transform.crop_by_boxes3d(inp, src_box, dst_box, align_corners=True)
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_crop_by_boxes_resizing(self, device, dtype):
        if False:
            return 10
        inp = torch.arange(0.0, 343.0, device=device, dtype=dtype).view(1, 1, 7, 7, 7)
        src_box = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0], [3.0, 1.0, 2.0], [3.0, 3.0, 2.0], [1.0, 3.0, 2.0]]], device=device, dtype=dtype)
        dst_box = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[[57.0, 59.0], [71.0, 73.0]], [[106.0, 108.0], [120.0, 122.0]]]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_by_boxes3d(inp, src_box, dst_box, align_corners=True)
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if False:
            while True:
                i = 10
        op = kornia.geometry.transform.crop_by_boxes3d
        op_script = torch_optimizer(op)
        inp = torch.randn((1, 1, 7, 7, 7), device=device, dtype=dtype)
        src_box = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0], [3.0, 1.0, 2.0], [3.0, 3.0, 2.0], [1.0, 3.0, 2.0]]], device=device, dtype=dtype)
        dst_box = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]], device=device, dtype=dtype)
        actual = op_script(inp, src_box, dst_box, align_corners=True)
        expected = op(inp, src_box, dst_box, align_corners=True)
        assert_close(actual, expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.randn((1, 1, 7, 7, 7), device=device, dtype=dtype)
        src_box = torch.tensor([[[1.0, 1.0, 1.0], [3.0, 1.0, 1.0], [3.0, 3.0, 1.0], [1.0, 3.0, 1.0], [1.0, 1.0, 2.0], [3.0, 1.0, 2.0], [3.0, 3.0, 2.0], [1.0, 3.0, 2.0]]], device=device, dtype=dtype)
        dst_box = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0, 1.0], [1.0, 1.0, 1.0], [0.0, 1.0, 1.0]]], device=device, dtype=dtype)
        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)
        assert gradcheck(kornia.geometry.transform.crop_by_boxes3d, (inp, src_box, dst_box), raise_exception=True, fast_mode=True)