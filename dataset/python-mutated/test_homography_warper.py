import pytest
import torch
from torch.autograd import gradcheck
import kornia
import kornia.testing as utils
from kornia.testing import assert_close
from kornia.utils.helpers import _torch_inverse_cast

class TestHomographyWarper:
    num_tests = 10
    threshold = 0.1

    def test_identity(self, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (2, 5)
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=3, device=device, dtype=dtype)
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)
        assert_close(patch_src, patch_dst)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_normalize_homography_identity(self, batch_size, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (height, width) = (2, 5)
        dst_homo_src = utils.create_eye_batch(batch_size=batch_size, eye_size=3, device=device, dtype=dtype)
        res = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert_close(kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype), res)
        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        assert_close(norm_homo, dst_homo_src)
        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height * 2, width // 2))
        res = torch.tensor([[[4.0, 0.0, 3.0], [0.0, 1 / 3, -2 / 3], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).repeat(batch_size, 1, 1)
        assert_close(norm_homo, res, atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_denormalize_homography_identity(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (2, 5)
        dst_homo_src = utils.create_eye_batch(batch_size=batch_size, eye_size=3, device=device, dtype=dtype)
        res = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert_close(kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype), res)
        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        assert_close(denorm_homo, dst_homo_src)
        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height * 2, width // 2))
        res = torch.tensor([[[0.25, 0.0, 0.0], [0.0, 3.0, 0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).repeat(batch_size, 1, 1)
        assert_close(denorm_homo, res, atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_normalize_homography_general(self, batch_size, device, dtype):
        if False:
            print('Hello World!')
        (height, width) = (2, 5)
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 2.0
        dst_homo_src[..., 0, 2] = 1.0
        dst_homo_src[..., 1, 2] = 2.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)
        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        res = torch.tensor([[[0.5, 0.0, 0.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).expand(batch_size, -1, -1)
        assert_close(norm_homo, res)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_denormalize_homography_general(self, batch_size, device, dtype):
        if False:
            while True:
                i = 10
        (height, width) = (2, 5)
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 2.0
        dst_homo_src[..., 0, 2] = 1.0
        dst_homo_src[..., 1, 2] = 2.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)
        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        res = torch.tensor([[[0.5, 0.0, 3.0], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]]], device=device, dtype=dtype).expand(batch_size, -1, -1)
        assert_close(denorm_homo, res)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_consistency(self, batch_size, device, dtype):
        if False:
            print('Hello World!')
        (height, width) = (2, 5)
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 2.0
        dst_homo_src[..., 0, 2] = 1.0
        dst_homo_src[..., 1, 2] = 2.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)
        denorm_homo = kornia.geometry.conversions.denormalize_homography(dst_homo_src, (height, width), (height, width))
        norm_denorm_homo = kornia.geometry.conversions.normalize_homography(denorm_homo, (height, width), (height, width))
        assert_close(dst_homo_src, norm_denorm_homo)
        norm_homo = kornia.geometry.conversions.normalize_homography(dst_homo_src, (height, width), (height, width))
        denorm_norm_homo = kornia.geometry.conversions.denormalize_homography(norm_homo, (height, width), (height, width))
        assert_close(dst_homo_src, denorm_norm_homo)

    @pytest.mark.parametrize('offset', [1, 3, 7])
    @pytest.mark.parametrize('shape', [(4, 5), (2, 6), (4, 3), (5, 7)])
    def test_warp_grid_translation(self, shape, offset, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (height, width) = shape
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 2] = offset
        grid = kornia.utils.create_meshgrid(height, width, normalized_coordinates=False)
        flow = kornia.geometry.transform.warp_grid(grid, dst_homo_src)
        assert_close(grid[..., 0].to(device=device, dtype=dtype) + offset, flow[..., 0])
        assert_close(grid[..., 1].to(device=device, dtype=dtype), flow[..., 1])

    @pytest.mark.parametrize('batch_shape', [(1, 1, 4, 5), (2, 2, 4, 6), (3, 1, 5, 7)])
    def test_identity_resize(self, batch_shape, device, dtype):
        if False:
            print('Hello World!')
        (batch_size, channels, height, width) = batch_shape
        patch_src = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size=3, device=device, dtype=dtype)
        warper = kornia.geometry.transform.HomographyWarper(height // 2, width // 2, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)
        assert_close(patch_src[..., 0, 0], patch_dst[..., 0, 0], atol=0.0001, rtol=0.0001)
        assert_close(patch_src[..., 0, -1], patch_dst[..., 0, -1], atol=0.0001, rtol=0.0001)
        assert_close(patch_src[..., -1, 0], patch_dst[..., -1, 0], atol=0.0001, rtol=0.0001)
        assert_close(patch_src[..., -1, -1], patch_dst[..., -1, -1], atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('shape', [(4, 5), (2, 6), (4, 3), (5, 7)])
    def test_translation(self, shape, device, dtype):
        if False:
            return 10
        offset = 2.0
        (height, width) = shape
        patch_src = torch.rand(1, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 2] = offset / (width - 1)
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)
        assert_close(patch_src[..., 1:], patch_dst[..., :-1], atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('batch_shape', [(1, 1, 3, 5), (2, 2, 4, 3), (3, 1, 2, 3)])
    def test_rotation(self, batch_shape, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (batch_size, channels, height, width) = batch_shape
        patch_src = torch.rand(batch_size, channels, height, width, device=device, dtype=dtype)
        dst_homo_src = torch.eye(3, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.0
        dst_homo_src[..., 0, 1] = 1.0
        dst_homo_src[..., 1, 0] = -1.0
        dst_homo_src[..., 1, 1] = 0.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        patch_dst = warper(patch_src, dst_homo_src)
        assert_close(patch_src[..., 0, 0], patch_dst[..., 0, -1], atol=0.0001, rtol=0.0001)
        assert_close(patch_src[..., 0, -1], patch_dst[..., -1, -1], atol=0.0001, rtol=0.0001)
        assert_close(patch_src[..., -1, 0], patch_dst[..., 0, 0], atol=0.0001, rtol=0.0001)
        assert_close(patch_src[..., -1, -1], patch_dst[..., -1, 0], atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('batch_size', [1, 2, 3])
    def test_homography_warper(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (128, 64)
        eye_size = 3
        patch_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size, device=device, dtype=dtype)
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        for _ in range(self.num_tests):
            homo_delta = torch.rand_like(dst_homo_src) * 0.3
            dst_homo_src_i = dst_homo_src + homo_delta
            patch_dst = warper(patch_src, dst_homo_src_i)
            patch_dst_to_src = warper(patch_dst, _torch_inverse_cast(dst_homo_src_i))
            warper.precompute_warp_grid(_torch_inverse_cast(dst_homo_src_i))
            patch_dst_to_src_precomputed = warper(patch_dst)
            assert_close(patch_dst_to_src_precomputed, patch_dst_to_src, atol=0.0001, rtol=0.0001)
            error = utils.compute_patch_error(patch_src, patch_dst_to_src, height, width)
            assert error.item() < self.threshold
            patch_dst_to_src_functional = kornia.geometry.transform.homography_warp(patch_dst, _torch_inverse_cast(dst_homo_src_i), (height, width), align_corners=True)
            assert_close(patch_dst_to_src, patch_dst_to_src_functional, atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('batch_shape', [(1, 1, 7, 5), (2, 3, 8, 5), (1, 1, 7, 16)])
    def test_gradcheck(self, batch_shape, device, dtype):
        if False:
            i = 10
            return i + 15
        eye_size = 3
        patch_src = torch.rand(batch_shape, device=device, dtype=dtype)
        patch_src = utils.tensor_to_gradcheck_var(patch_src)
        (batch_size, _, height, width) = patch_src.shape
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size, device=device, dtype=dtype)
        dst_homo_src = utils.tensor_to_gradcheck_var(dst_homo_src, requires_grad=False)
        warper = kornia.geometry.transform.HomographyWarper(height, width, align_corners=True)
        assert gradcheck(warper, (patch_src, dst_homo_src), nondet_tol=1e-08, raise_exception=True, fast_mode=True)

    @pytest.mark.parametrize('batch_size', [1, 2, 3])
    @pytest.mark.parametrize('align_corners', [True, False])
    @pytest.mark.parametrize('normalized_coordinates', [True, False])
    def test_dynamo(self, batch_size, align_corners, normalized_coordinates, device, dtype, torch_optimizer):
        if False:
            while True:
                i = 10
        (height, width) = (128, 64)
        eye_size = 3
        patch_src = torch.rand(batch_size, 1, height, width, device=device, dtype=dtype)
        dst_homo_src = utils.create_eye_batch(batch_size, eye_size, device=device, dtype=dtype)
        for _ in range(self.num_tests):
            homo_delta = torch.rand_like(dst_homo_src) * 0.3
            dst_homo_src_i = dst_homo_src + homo_delta
            patch_dst = kornia.geometry.transform.homography_warp(patch_src, dst_homo_src_i, (height, width), align_corners=align_corners, normalized_coordinates=normalized_coordinates)
            patch_dst_optimized = torch_optimizer(kornia.geometry.transform.homography_warp)(patch_src, dst_homo_src_i, (height, width), align_corners=align_corners, normalized_coordinates=normalized_coordinates)
            assert_close(patch_dst, patch_dst_optimized, atol=0.0001, rtol=0.0001)

class TestHomographyNormalTransform:
    expected_2d_0 = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 2.0, -1.0], [0.0, 0.0, 1.0]]])
    expected_2d_1 = torch.tensor([[[0.5, 0.0, -1.0], [0.0, 200000000000000.0, -1.0], [0.0, 0.0, 1.0]]])
    expected_3d_0 = expected = torch.tensor([[[0.4, 0.0, 0.0, -1.0], [0.0, 2.0, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]])
    expected_3d_1 = torch.tensor([[[0.4, 0.0, 0.0, -1.0], [0.0, 200000000000000.0, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]])

    @pytest.mark.parametrize('height,width,expected', [(2, 5, expected_2d_0), (1, 5, expected_2d_1)])
    def test_transform2d(self, height, width, expected, device, dtype):
        if False:
            while True:
                i = 10
        output = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)
        assert_close(output, expected.to(device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('height', [1, 2, 5])
    @pytest.mark.parametrize('width', [1, 2, 5])
    def test_divide_by_zero2d(self, height, width, device, dtype):
        if False:
            i = 10
            return i + 15
        output = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)
        assert torch.isinf(output).sum().item() == 0

    def test_transform2d_apply(self, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (2, 5)
        input = torch.tensor([[0.0, 0.0], [width - 1, height - 1]], device=device, dtype=dtype)
        expected = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], device=device, dtype=dtype)
        transform = kornia.geometry.conversions.normal_transform_pixel(height, width, device=device, dtype=dtype)
        output = kornia.geometry.linalg.transform_points(transform, input)
        assert_close(output, expected.to(device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('height,width,depth,expected', [(2, 6, 4, expected_3d_0), (1, 6, 4, expected_3d_1)])
    def test_transform3d(self, height, width, depth, expected, device, dtype):
        if False:
            while True:
                i = 10
        output = kornia.geometry.conversions.normal_transform_pixel3d(depth, height, width, device=device, dtype=dtype)
        assert_close(output, expected.to(device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('height', [1, 2, 5])
    @pytest.mark.parametrize('width', [1, 2, 5])
    @pytest.mark.parametrize('depth', [1, 2, 5])
    def test_divide_by_zero3d(self, height, width, depth, device, dtype):
        if False:
            print('Hello World!')
        output = kornia.geometry.conversions.normal_transform_pixel3d(depth, height, width, device=device, dtype=dtype)
        assert torch.isinf(output).sum().item() == 0

    def test_transform3d_apply(self, device, dtype):
        if False:
            return 10
        (depth, height, width) = (3, 2, 5)
        input = torch.tensor([[0.0, 0.0, 0.0], [width - 1, height - 1, depth - 1]], device=device, dtype=dtype)
        expected = torch.tensor([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]], device=device, dtype=dtype)
        transform = kornia.geometry.conversions.normal_transform_pixel3d(depth, height, width, device=device, dtype=dtype)
        output = kornia.geometry.linalg.transform_points(transform, input)
        assert_close(output, expected.to(device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

class TestHomographyWarper3D:
    num_tests = 10
    threshold = 0.1

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_normalize_homography_identity(self, batch_size, device, dtype):
        if False:
            print('Hello World!')
        input_shape = (4, 8, 5)
        dst_homo_src = utils.create_eye_batch(batch_size=batch_size, eye_size=4).to(device=device, dtype=dtype)
        res = torch.tensor([[[0.5, 0.0, 0.0, -1.0], [0.0, 0.2857, 0.0, -1.0], [0.0, 0.0, 0.6667, -1.0], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        norm = kornia.geometry.conversions.normal_transform_pixel3d(input_shape[0], input_shape[1], input_shape[2]).to(device=device, dtype=dtype)
        assert_close(norm, res, rtol=0.0001, atol=0.0001)
        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, input_shape).to(device=device, dtype=dtype)
        assert_close(norm_homo, dst_homo_src, rtol=0.0001, atol=0.0001)
        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, input_shape).to(device=device, dtype=dtype)
        assert_close(norm_homo, dst_homo_src, rtol=0.0001, atol=0.0001)
        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, input_shape, (input_shape[0] // 2, input_shape[1] * 2, input_shape[2] // 2)).to(device=device, dtype=dtype)
        res = torch.tensor([[[4.0, 0.0, 0.0, 3.0], [0.0, 0.4667, 0.0, -0.5333], [0.0, 0.0, 3.0, 2.0], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype).repeat(batch_size, 1, 1)
        assert_close(norm_homo, res, rtol=0.0001, atol=0.0001)

    @pytest.mark.parametrize('batch_size', [1, 3])
    def test_normalize_homography_general(self, batch_size, device, dtype):
        if False:
            return 10
        dst_homo_src = torch.eye(4, device=device, dtype=dtype)
        dst_homo_src[..., 0, 0] = 0.5
        dst_homo_src[..., 1, 1] = 0.5
        dst_homo_src[..., 2, 2] = 2.0
        dst_homo_src[..., 0, 3] = 1.0
        dst_homo_src[..., 1, 3] = 2.0
        dst_homo_src[..., 2, 3] = 3.0
        dst_homo_src = dst_homo_src.expand(batch_size, -1, -1)
        norm_homo = kornia.geometry.conversions.normalize_homography3d(dst_homo_src, (2, 2, 5), (2, 2, 5))
        res = torch.tensor([[[0.5, 0.0, 0.0, 0.0], [0.0, 0.5, 0.0, 3.5], [0.0, 0.0, 2.0, 7.0], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype).expand(batch_size, -1, -1)
        assert_close(norm_homo, res)

    @pytest.mark.parametrize('offset', [1, 3, 7])
    @pytest.mark.parametrize('shape', [(4, 5, 6), (2, 4, 6), (4, 3, 9), (5, 7, 8)])
    def test_warp_grid_translation(self, shape, offset, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (depth, height, width) = shape
        dst_homo_src = utils.create_eye_batch(batch_size=1, eye_size=4, device=device, dtype=dtype)
        dst_homo_src[..., 0, 3] = offset
        grid = kornia.utils.create_meshgrid3d(depth, height, width, normalized_coordinates=False)
        flow = kornia.geometry.transform.warp_grid3d(grid, dst_homo_src)
        assert_close(grid[..., 0].to(device=device, dtype=dtype) + offset, flow[..., 0], atol=0.0001, rtol=0.0001)
        assert_close(grid[..., 1].to(device=device, dtype=dtype), flow[..., 1], atol=0.0001, rtol=0.0001)
        assert_close(grid[..., 2].to(device=device, dtype=dtype), flow[..., 2], atol=0.0001, rtol=0.0001)