import torch
from torch.autograd import gradcheck
import kornia
import kornia.testing as utils
from kornia.testing import assert_close

class TestCropAndResize:

    def test_align_corners_true(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        (height, width) = (2, 3)
        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (height, width))
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_align_corners_false(self, device, dtype):
        if False:
            print('Hello World!')
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        (height, width) = (2, 3)
        expected = torch.tensor([[[[6.7222, 7.1667, 7.6111], [9.3889, 9.8333, 10.2778]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (height, width), align_corners=False)
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_crop_batch(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]], [[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]], [[[7.0, 15.0], [8.0, 16.0]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]], [[1.0, 2.0], [3.0, 2.0], [3.0, 3.0], [1.0, 3.0]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (2, 2))
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_crop_batch_broadcast(self, device, dtype):
        if False:
            while True:
                i = 10
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]], [[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]], [[[6.0, 10.0], [7.0, 11.0]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_and_resize(inp, boxes, (2, 2))
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            return 10
        img = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        boxes = utils.tensor_to_gradcheck_var(boxes, requires_grad=False)
        assert gradcheck(kornia.geometry.transform.crop_and_resize, (img, boxes, (4, 2)), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if False:
            return 10
        op = kornia.geometry.transform.crop_and_resize
        op_optimized = torch_optimizer(op)
        img = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        boxes = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        (crop_height, crop_width) = (4, 2)
        actual = op_optimized(img, boxes, (crop_height, crop_width))
        expected = op(img, boxes, (crop_height, crop_width))
        assert_close(actual, expected, rtol=0.0001, atol=0.0001)

class TestCenterCrop:

    def test_center_crop_h2_w4(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]]]], device=device, dtype=dtype)
        out_crop = kornia.geometry.transform.center_crop(inp, (2, 4))
        assert_close(out_crop, expected, rtol=0.0001, atol=0.0001)

    def test_center_crop_h4_w2(self, device, dtype):
        if False:
            i = 10
            return i + 15
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        (height, width) = (4, 2)
        expected = torch.tensor([[[[2.0, 3.0], [6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]]], device=device, dtype=dtype)
        out_crop = kornia.geometry.transform.center_crop(inp, (height, width))
        assert_close(out_crop, expected, rtol=0.0001, atol=0.0001)

    def test_center_crop_h4_w2_batch(self, device, dtype):
        if False:
            i = 10
            return i + 15
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]], [[[1.0, 5.0, 9.0, 13.0], [2.0, 6.0, 10.0, 14.0], [3.0, 7.0, 11.0, 15.0], [4.0, 8.0, 12.0, 16.0]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[2.0, 3.0], [6.0, 7.0], [10.0, 11.0], [14.0, 15.0]]], [[[5.0, 9.0], [6.0, 10.0], [7.0, 11.0], [8.0, 12.0]]]], device=device, dtype=dtype)
        out_crop = kornia.geometry.transform.center_crop(inp, (4, 2))
        assert_close(out_crop, expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        img = torch.rand(1, 2, 5, 4, device=device, dtype=dtype)
        img = utils.tensor_to_gradcheck_var(img)
        assert gradcheck(kornia.geometry.transform.center_crop, (img, (4, 2)), raise_exception=True, fast_mode=True)

    def test_dynamo(self, device, dtype, torch_optimizer):
        if False:
            return 10
        op = kornia.geometry.transform.center_crop
        op_script = torch_optimizer(op)
        img = torch.ones(1, 2, 5, 4, device=device, dtype=dtype)
        actual = op_script(img, (4, 2))
        expected = op(img, (4, 2))
        assert_close(actual, expected, rtol=0.0001, atol=0.0001)

class TestCropByBoxes:

    def test_crop_by_boxes_no_resizing(self, device, dtype):
        if False:
            return 10
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        src = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        dst = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_by_boxes(inp, src, dst)
        assert_close(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        if False:
            print('Hello World!')
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        src = torch.tensor([[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0]]], device=device, dtype=dtype)
        dst = torch.tensor([[[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_by_boxes(inp, src, dst)
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            return 10
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=dtype)
        src = torch.tensor([[[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]]], device=device, dtype=dtype)
        dst = torch.tensor([[[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]], device=device, dtype=dtype)
        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)
        assert gradcheck(kornia.geometry.transform.crop_by_boxes, (inp, src, dst), raise_exception=True, fast_mode=True)

class TestCropByTransform:

    def test_crop_by_transform_no_resizing(self, device, dtype):
        if False:
            print('Hello World!')
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        transform = torch.tensor([[[1.0, 0.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[6.0, 7.0], [10.0, 11.0]]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_by_transform_mat(inp, transform, (2, 2))
        assert_close(patches, expected)

    def test_crop_by_boxes_resizing(self, device, dtype):
        if False:
            i = 10
            return i + 15
        inp = torch.tensor([[[[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0], [13.0, 14.0, 15.0, 16.0]]]], device=device, dtype=dtype)
        transform = torch.tensor([[[2.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[6.0, 6.5, 7.0], [10.0, 10.5, 11.0]]]], device=device, dtype=dtype)
        patches = kornia.geometry.transform.crop_by_transform_mat(inp, transform, (2, 3))
        assert_close(patches, expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        inp = torch.randn((1, 1, 3, 3), device=device, dtype=dtype)
        transform = torch.tensor([[[2.0, 0.0, -2.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        inp = utils.tensor_to_gradcheck_var(inp, requires_grad=True)
        assert gradcheck(kornia.geometry.transform.crop_by_transform_mat, (inp, transform, (2, 2)), raise_exception=True, fast_mode=True)