import pytest
import torch
from torch.autograd import gradcheck
import kornia
import kornia.testing as utils
from kornia.geometry.conversions import normalize_pixel_coordinates
from kornia.testing import assert_close

class TestDepthWarper:
    eps = 1e-06

    def _create_pinhole_pair(self, batch_size, device, dtype):
        if False:
            while True:
                i = 10
        (fx, fy) = (1.0, 1.0)
        (height, width) = (3, 5)
        (cx, cy) = (width / 2, height / 2)
        (tx, ty, tz) = (0, 0, 0)
        pinhole_src = kornia.geometry.camera.PinholeCamera.from_parameters(fx, fy, cx, cy, height, width, tx, ty, tz, batch_size, device=device, dtype=dtype)
        pinhole_dst = kornia.geometry.camera.PinholeCamera.from_parameters(fx, fy, cx, cy, height, width, tx, ty, tz, batch_size, device=device, dtype=dtype)
        return (pinhole_src, pinhole_dst)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_compute_projection_matrix(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        assert warper._dst_proj_src is None
        warper.compute_projection_matrix(pinhole_src)
        assert warper._dst_proj_src is not None
        dst_proj_src = warper._dst_proj_src
        dst_proj_src_expected = torch.eye(4, device=device, dtype=dtype)[None].repeat(batch_size, 1, 1)
        dst_proj_src_expected[..., 0, -2] += pinhole_src.cx
        dst_proj_src_expected[..., 1, -2] += pinhole_src.cy
        dst_proj_src_expected[..., 0, -1] += 1.0
        assert_close(dst_proj_src, dst_proj_src_expected)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_warp_grid_offset_x1_depth1(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)
        grid_warped = warper.warp_grid(depth_src)
        assert grid_warped.shape == (batch_size, height, width, 2)
        grid = warper.grid[..., :2].to(device=device, dtype=dtype)
        grid_norm = normalize_pixel_coordinates(grid, height, width)
        assert_close(grid_warped[..., -2, 0], grid_norm[..., -1, 0].repeat(batch_size, 1), atol=0.0001, rtol=0.0001)
        assert_close(grid_warped[..., -1, 1], grid_norm[..., -1, 1].repeat(batch_size, 1), rtol=0.0001, atol=0.0001)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_warp_grid_offset_x1y1_depth1(self, batch_size, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (height, width) = (3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0
        pinhole_dst.ty += 1.0
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)
        grid_warped = warper.warp_grid(depth_src)
        assert grid_warped.shape == (batch_size, height, width, 2)
        grid = warper.grid[..., :2].to(device=device, dtype=dtype)
        grid_norm = normalize_pixel_coordinates(grid, height, width)
        assert_close(grid_warped[..., -2, 0], grid_norm[..., -1, 0].repeat(batch_size, 1), atol=0.0001, rtol=0.0001)
        assert_close(grid_warped[..., -2, :, 1], grid_norm[..., -1, :, 1].repeat(batch_size, 1), rtol=0.0001, atol=0.0001)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_warp_tensor_offset_x1y1(self, batch_size, device, dtype):
        if False:
            while True:
                i = 10
        (channels, height, width) = (3, 3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        pinhole_dst.tx += 1.0
        pinhole_dst.ty += 1.0
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)
        patch_dst = torch.arange(float(height * width), device=device, dtype=dtype).view(1, 1, height, width).expand(batch_size, channels, -1, -1)
        patch_src = warper(depth_src, patch_dst)
        assert_close(patch_dst[..., 1:, 1:], patch_src[..., :2, :4], atol=0.0001, rtol=0.0001)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_compute_projection(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        (height, width) = (3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)
        xy_projected = warper._compute_projection(0.0, 0.0, 1.0)
        assert xy_projected.shape == (batch_size, 2)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_compute_subpixel_step(self, batch_size, device, dtype):
        if False:
            while True:
                i = 10
        (height, width) = (3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        warper = kornia.geometry.depth.DepthWarper(pinhole_dst, height, width)
        warper.compute_projection_matrix(pinhole_src)
        subpixel_step = warper.compute_subpixel_step()
        assert_close(subpixel_step.item(), 0.1715, rtol=0.001, atol=0.001)

    @pytest.mark.parametrize('batch_size', (1, 2))
    def test_gradcheck(self, batch_size, device, dtype):
        if False:
            print('Hello World!')
        (channels, height, width) = (3, 3, 5)
        (pinhole_src, pinhole_dst) = self._create_pinhole_pair(batch_size, device, dtype)
        depth_src = torch.ones(batch_size, 1, height, width, device=device, dtype=dtype)
        depth_src = utils.tensor_to_gradcheck_var(depth_src)
        img_dst = torch.ones(batch_size, channels, height, width, device=device, dtype=dtype)
        img_dst = utils.tensor_to_gradcheck_var(img_dst)
        assert gradcheck(kornia.geometry.depth.depth_warp, (pinhole_dst, pinhole_src, depth_src, img_dst, height, width), raise_exception=True, fast_mode=True)