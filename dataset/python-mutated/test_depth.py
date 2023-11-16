import pytest
import torch
from torch.autograd import gradcheck
import kornia
import kornia.testing as utils
from kornia.testing import assert_close

class TestDepthTo3d:

    def test_smoke(self, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize('batch_size', [2, 4, 5])
    def test_shapes(self, batch_size, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(batch_size, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    @pytest.mark.parametrize('batch_size', [1, 2, 4, 5])
    def test_shapes_broadcast(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    def test_depth_to_3d_v2(self, device, dtype):
        if False:
            print('Hello World!')
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        points3d_v2 = kornia.geometry.depth.depth_to_3d_v2(depth[0, 0], camera_matrix[0])
        assert_close(points3d[0].permute(1, 2, 0), points3d_v2)

    def test_unproject_meshgrid(self, device, dtype):
        if False:
            while True:
                i = 10
        camera_matrix = torch.eye(3, device=device, dtype=dtype)
        grid = kornia.geometry.unproject_meshgrid(3, 4, camera_matrix, device=device, dtype=dtype)
        assert grid.shape == (3, 4, 3)
        assert_close(grid[..., 2], torch.ones_like(grid[..., 2]))

    def test_unproject_denormalized(self, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = 2 * torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        points3d_expected = torch.tensor([[[[0.0, 2.0, 4.0], [0.0, 2.0, 4.0], [0.0, 2.0, 4.0], [0.0, 2.0, 4.0]], [[0.0, 0.0, 0.0], [2.0, 2.0, 2.0], [4.0, 4.0, 4.0], [6.0, 6.0, 6.0]], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0], [2.0, 2.0, 2.0]]]], device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        assert_close(points3d, points3d_expected, atol=0.0001, rtol=0.0001)

    def test_unproject_normalized(self, device, dtype):
        if False:
            return 10
        depth = 2 * torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        points3d_expected = torch.tensor([[[[0.0, 1.4142, 1.7889], [0.0, 1.1547, 1.633], [0.0, 0.8165, 1.3333], [0.0, 0.603, 1.069]], [[0.0, 0.0, 0.0], [1.4142, 1.1547, 0.8165], [1.7889, 1.633, 1.3333], [1.8974, 1.8091, 1.6036]], [[2.0, 1.4142, 0.8944], [1.4142, 1.1547, 0.8165], [0.8944, 0.8165, 0.6667], [0.6325, 0.603, 0.5345]]]], device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix, normalize_points=True)
        assert_close(points3d, points3d_expected, atol=0.0001, rtol=0.0001)

    def test_unproject_and_project(self, device, dtype):
        if False:
            while True:
                i = 10
        depth = 2 * torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_3d(depth, camera_matrix)
        points2d = kornia.geometry.camera.project_points(points3d.permute(0, 2, 3, 1), camera_matrix[:, None, None])
        points2d_expected = kornia.utils.create_meshgrid(4, 3, False, device=device).to(dtype=dtype)
        assert_close(points2d, points2d_expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        depth = utils.tensor_to_gradcheck_var(depth)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.geometry.depth.depth_to_3d, (depth, camera_matrix), raise_exception=True, fast_mode=True)

class TestDepthToNormals:

    def test_smoke(self, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize('batch_size', [2, 4, 5])
    def test_shapes(self, batch_size, device, dtype):
        if False:
            while True:
                i = 10
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(batch_size, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    @pytest.mark.parametrize('batch_size', [2, 4, 5])
    def test_shapes_broadcast(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert points3d.shape == (batch_size, 3, 3, 4)

    def test_simple(self, device, dtype):
        if False:
            print('Hello World!')
        depth = 2 * torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        normals_expected = torch.tensor([[[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        normals = kornia.geometry.depth.depth_to_normals(depth, camera_matrix)
        assert_close(normals, normals_expected, rtol=0.001, atol=0.001)

    def test_simple_normalized(self, device, dtype):
        if False:
            i = 10
            return i + 15
        depth = 2 * torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        camera_matrix = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        normals_expected = torch.tensor([[[[0.3432, 0.4861, 0.7628], [0.2873, 0.426, 0.6672], [0.2284, 0.3683, 0.5596], [0.1695, 0.298, 0.4496]], [[0.3432, 0.2873, 0.2363], [0.4861, 0.426, 0.3785], [0.8079, 0.7261, 0.6529], [0.8948, 0.8237, 0.7543]], [[0.8743, 0.8253, 0.6019], [0.8253, 0.7981, 0.6415], [0.5432, 0.5807, 0.5105], [0.4129, 0.4824, 0.4784]]]], device=device, dtype=dtype)
        normals = kornia.geometry.depth.depth_to_normals(depth, camera_matrix, normalize_points=True)
        assert_close(normals, normals_expected, rtol=0.001, atol=0.001)

    def test_gradcheck(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        depth = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        depth = utils.tensor_to_gradcheck_var(depth)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.geometry.depth.depth_to_normals, (depth, camera_matrix), raise_exception=True, fast_mode=True)

class TestWarpFrameDepth:

    def test_smoke(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        image_src = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        depth_dst = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        src_trans_dst = torch.rand(1, 4, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (1, 3, 3, 4)

    @pytest.mark.parametrize('batch_size', [2, 4, 5])
    @pytest.mark.parametrize('num_features', [1, 3, 5])
    def test_shape(self, batch_size, num_features, device, dtype):
        if False:
            while True:
                i = 10
        image_src = torch.rand(batch_size, num_features, 3, 4, device=device, dtype=dtype)
        depth_dst = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        src_trans_dst = torch.rand(batch_size, 4, 4, device=device, dtype=dtype)
        camera_matrix = torch.rand(batch_size, 3, 3, device=device, dtype=dtype)
        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert image_dst.shape == (batch_size, num_features, 3, 4)

    def test_translation(self, device, dtype):
        if False:
            i = 10
            return i + 15
        image_src = torch.tensor([[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]], device=device, dtype=dtype)
        depth_dst = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        src_trans_dst = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        (h, w) = image_src.shape[-2:]
        camera_matrix = torch.tensor([[[1.0, 0.0, w / 2], [0.0, 1.0, h / 2], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        image_dst_expected = torch.tensor([[[[2.0, 3.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 0.0], [2.0, 3.0, 0.0]]]], device=device, dtype=dtype)
        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix)
        assert_close(image_dst, image_dst_expected, rtol=0.001, atol=0.001)

    def test_translation_normalized(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        image_src = torch.tensor([[[[1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]]], device=device, dtype=dtype)
        depth_dst = torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        src_trans_dst = torch.tensor([[[1.0, 0.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        (h, w) = image_src.shape[-2:]
        camera_matrix = torch.tensor([[[1.0, 0.0, w / 2], [0.0, 1.0, h / 2], [0.0, 0.0, 1.0]]], device=device, dtype=dtype)
        image_dst_expected = torch.tensor([[[[0.9223, 0.0, 0.0], [2.8153, 1.5, 0.0], [2.8028, 2.6459, 0.0], [2.8153, 1.5, 0.0]]]], device=device, dtype=dtype)
        image_dst = kornia.geometry.depth.warp_frame_depth(image_src, depth_dst, src_trans_dst, camera_matrix, normalize_points=True)
        assert_close(image_dst, image_dst_expected, rtol=0.001, atol=0.001)

    def test_gradcheck(self, device, dtype):
        if False:
            print('Hello World!')
        image_src = torch.rand(1, 3, 3, 4, device=device, dtype=dtype)
        image_src = utils.tensor_to_gradcheck_var(image_src)
        depth_dst = torch.rand(1, 1, 3, 4, device=device, dtype=dtype)
        depth_dst = utils.tensor_to_gradcheck_var(depth_dst)
        src_trans_dst = torch.rand(1, 4, 4, device=device, dtype=dtype)
        src_trans_dst = utils.tensor_to_gradcheck_var(src_trans_dst)
        camera_matrix = torch.rand(1, 3, 3, device=device, dtype=dtype)
        camera_matrix = utils.tensor_to_gradcheck_var(camera_matrix)
        assert gradcheck(kornia.geometry.depth.warp_frame_depth, (image_src, depth_dst, src_trans_dst, camera_matrix), raise_exception=True, fast_mode=True)

class TestDepthFromDisparity:

    def test_smoke(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        disparity = 2 * torch.tensor([[[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]]]], device=device, dtype=dtype)
        baseline = torch.tensor([1.0], device=device, dtype=dtype)
        focal = torch.tensor([1.0], device=device, dtype=dtype)
        depth_expected = torch.tensor([[[[0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]]]], device=device, dtype=dtype)
        depth = kornia.geometry.depth.depth_from_disparity(disparity, baseline, focal)
        assert_close(depth, depth_expected, rtol=0.001, atol=0.001)

    @pytest.mark.parametrize('batch_size', [2, 4, 5])
    def test_cardinality(self, batch_size, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        disparity = torch.rand(batch_size, 1, 3, 4, device=device, dtype=dtype)
        baseline = torch.rand(1, device=device, dtype=dtype)
        focal = torch.rand(1, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_from_disparity(disparity, baseline, focal)
        assert points3d.shape == (batch_size, 1, 3, 4)

    @pytest.mark.parametrize('shape', [(1, 1, 3, 4), (4, 1, 3, 4), (4, 3, 4), (1, 3, 4), (3, 4)])
    def test_shapes(self, shape, device, dtype):
        if False:
            i = 10
            return i + 15
        disparity = torch.randn(shape, device=device, dtype=dtype)
        baseline = torch.rand(1, device=device, dtype=dtype)
        focal = torch.rand(1, device=device, dtype=dtype)
        points3d = kornia.geometry.depth.depth_from_disparity(disparity, baseline, focal)
        assert points3d.shape == shape

    def test_gradcheck(self, device):
        if False:
            for i in range(10):
                print('nop')
        disparity = torch.rand(1, 1, 3, 4, device=device)
        disparity = utils.tensor_to_gradcheck_var(disparity)
        baseline = torch.rand(1, device=device)
        baseline = utils.tensor_to_gradcheck_var(baseline)
        focal = torch.rand(1, device=device)
        focal = utils.tensor_to_gradcheck_var(focal)
        assert gradcheck(kornia.geometry.depth.depth_from_disparity, (disparity, baseline, focal), raise_exception=True, fast_mode=True)