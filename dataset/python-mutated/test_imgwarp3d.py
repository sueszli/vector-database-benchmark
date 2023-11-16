import pytest
import torch
from torch.autograd import gradcheck
import kornia
import kornia.geometry.transform as proj
import kornia.testing as utils
from kornia.testing import assert_close
from kornia.utils.helpers import _torch_inverse_cast

class TestWarpAffine3d:

    def test_smoke(self, device, dtype):
        if False:
            print('Hello World!')
        sample = torch.rand(1, 3, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(1, 3, 4, device=device, dtype=dtype)
        output = proj.warp_affine3d(sample, P, (3, 4, 5))
        assert output.shape == (1, 3, 3, 4, 5)

    @pytest.mark.parametrize('batch_size', [1, 3])
    @pytest.mark.parametrize('num_channels', [1, 3, 5])
    @pytest.mark.parametrize('out_shape', [(3, 3, 3), (4, 5, 6)])
    def test_batch(self, batch_size, num_channels, out_shape, device, dtype):
        if False:
            return 10
        (B, C) = (batch_size, num_channels)
        sample = torch.rand(B, C, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(B, 3, 4, device=device, dtype=dtype)
        output = proj.warp_affine3d(sample, P, out_shape)
        assert list(output.shape) == [B, C, *list(out_shape)]

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        sample = torch.rand(1, 3, 3, 4, 5, device=device, dtype=torch.float64, requires_grad=True)
        P = torch.rand(1, 3, 4, device=device, dtype=torch.float64)
        assert gradcheck(proj.warp_affine3d, (sample, P, (3, 3, 3)), raise_exception=True, fast_mode=True)

    def test_forth_back(self, device, dtype):
        if False:
            i = 10
            return i + 15
        out_shape = (3, 4, 5)
        sample = torch.rand(2, 5, 3, 4, 5, device=device, dtype=dtype)
        P = torch.rand(2, 3, 4, device=device, dtype=dtype)
        P = kornia.geometry.convert_affinematrix_to_homography3d(P)
        P_hat = (_torch_inverse_cast(P) @ P)[:, :3]
        output = proj.warp_affine3d(sample, P_hat, out_shape, flags='nearest')
        assert_close(output, sample, rtol=0.0001, atol=0.0001)

    def test_rotate_x(self, device, dtype):
        if False:
            while True:
                i = 10
        sample = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 2.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        (_, _, D, H, W) = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)
        angles = torch.tensor([[90.0, 0.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        assert_close(output, expected, rtol=0.0001, atol=0.0001)

    def test_rotate_y(self, device, dtype):
        if False:
            return 10
        sample = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        (_, _, D, H, W) = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)
        angles = torch.tensor([[0.0, 90.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        assert_close(output, expected, rtol=0.0001, atol=0.0001)

    def test_rotate_z(self, device, dtype):
        if False:
            print('Hello World!')
        sample = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        (_, _, D, H, W) = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)
        angles = torch.tensor([[0.0, 0.0, 90.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        assert_close(output, expected, rtol=0.0001, atol=0.0001)

    def test_rotate_y_large(self, device, dtype):
        if False:
            print('Hello World!')
        'Rotates 90deg anti-clockwise.'
        sample = torch.tensor([[[[[0.0, 4.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 9.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 6.0, 7.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        expected = torch.tensor([[[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[4.0, 2.0, 0.0], [3.0, 1.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]], [[[0.0, 0.0, 0.0], [0.0, 7.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 6.0, 8.0], [9.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]]]], device=device, dtype=dtype)
        (_, _, D, H, W) = sample.shape
        center = torch.tensor([[(W - 1) / 2, (H - 1) / 2, (D - 1) / 2]], device=device, dtype=dtype)
        angles = torch.tensor([[0.0, 90.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angles, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angles, scales)
        output = proj.warp_affine3d(sample, P, (3, 3, 3))
        assert_close(output, expected, rtol=0.0001, atol=0.0001)

class TestGetRotationMatrix3d:

    def test_smoke(self, device, dtype):
        if False:
            return 10
        center = torch.rand(1, 3, device=device, dtype=dtype)
        angle = torch.rand(1, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        assert P.shape == (1, 3, 4)

    @pytest.mark.parametrize('batch_size', [1, 3, 6])
    def test_batch(self, batch_size, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        B: int = batch_size
        center = torch.rand(B, 3, device=device, dtype=dtype)
        angle = torch.rand(B, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        assert P.shape == (B, 3, 4)

    def test_identity(self, device, dtype):
        if False:
            print('Hello World!')
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.zeros(1, 3, device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype).unsqueeze(0)
        assert_close(P, P_expected, atol=0.0001, rtol=0.0001)

    def test_rot90x(self, device, dtype):
        if False:
            print('Hello World!')
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[90.0, 0.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, -1.0, 0.0], [0.0, 1.0, 0.0, 0.0]], device=device, dtype=dtype).unsqueeze(0)
        assert_close(P, P_expected, atol=0.0001, rtol=0.0001)

    def test_rot90y(self, device, dtype):
        if False:
            i = 10
            return i + 15
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[0.0, 90.0, 0.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([[0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 0.0], [-1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype).unsqueeze(0)
        assert_close(P, P_expected, atol=0.0001, rtol=0.0001)

    def test_rot90z(self, device, dtype):
        if False:
            return 10
        center = torch.zeros(1, 3, device=device, dtype=dtype)
        angle = torch.tensor([[0.0, 0.0, 90.0]], device=device, dtype=dtype)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=dtype)
        P = proj.get_projective_transform(center, angle, scales)
        P_expected = torch.tensor([[0.0, -1.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]], device=device, dtype=dtype).unsqueeze(0)
        assert_close(P, P_expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            while True:
                i = 10
        center = torch.rand(1, 3, device=device, dtype=torch.float64, requires_grad=True)
        angle = torch.rand(1, 3, device=device, dtype=torch.float64)
        scales: torch.Tensor = torch.ones_like(angle, device=device, dtype=torch.float64)
        assert gradcheck(proj.get_projective_transform, (center, angle, scales), raise_exception=True, fast_mode=True)

class TestPerspectiveTransform3D:

    @pytest.mark.skip('Not working')
    @pytest.mark.parametrize('batch_size', [1, 2, 5])
    def test_get_perspective_transform3d(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        norm = torch.rand(batch_size, 8, 3, device=device, dtype=dtype)
        points_src = torch.rand_like(norm, device=device, dtype=dtype)
        points_dst = points_src + norm
        dst_homo_src = kornia.geometry.transform.get_perspective_transform3d(points_src, points_dst)
        assert_close(kornia.geometry.linalg.transform_points(dst_homo_src, points_src), points_dst, rtol=0.0001, atol=0.0001)
        points_src = utils.tensor_to_gradcheck_var(points_src)
        points_dst = utils.tensor_to_gradcheck_var(points_dst)
        assert gradcheck(kornia.geometry.transform.get_perspective_transform3d, (points_src, points_dst), raise_exception=True)

    @pytest.mark.parametrize('batch_size', [1, 2])
    def test_get_perspective_transform3d_2(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        torch.manual_seed(0)
        src = kornia.geometry.bbox.bbox_generator3d(torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype), torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype), torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype), torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size), torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size), torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size)).to(device=device, dtype=dtype)
        dst = kornia.geometry.bbox.bbox_generator3d(torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype), torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype), torch.randint_like(torch.ones(batch_size), 0, 50, dtype=dtype), torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size), torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size), torch.randint(0, 50, (1,), dtype=dtype).repeat(batch_size)).to(device=device, dtype=dtype)
        out = kornia.geometry.transform.get_perspective_transform3d(src, dst)
        if batch_size == 1:
            expected = torch.tensor([[[3.3, 0.0, 0.0, -118.2], [0.0, 0.0769, 0.0, 0.0], [0.0, 0.0, 0.5517, 28.793], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        if batch_size == 2:
            expected = torch.tensor([[[0.963, 0.0, 0.0, -9.3702], [0.0, 2.0, 0.0, -49.9999], [0.0, 0.0, 0.383, 44.0213], [0.0, 0.0, 0.0, 1.0]], [[0.963, 0.0, 0.0, -36.5555], [0.0, 2.0, 0.0, -14.0], [0.0, 0.0, 0.383, 16.894], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        assert_close(out, expected, rtol=0.0001, atol=0.0001)
        points_src = utils.tensor_to_gradcheck_var(src)
        points_dst = utils.tensor_to_gradcheck_var(dst)
        assert gradcheck(kornia.geometry.transform.get_perspective_transform3d, (points_src, points_dst), raise_exception=True, fast_mode=True)