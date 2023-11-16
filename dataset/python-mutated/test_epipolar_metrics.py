import torch
from torch.autograd import gradcheck
import kornia.geometry.epipolar as epi
import kornia.testing as utils
from kornia.testing import assert_close

class TestSymmetricalEpipolarDistance:

    def test_smoke(self, device, dtype):
        if False:
            print('Hello World!')
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        if False:
            while True:
                i = 10
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.symmetrical_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (batch_size, num_frames, num_points) = (5, 3, 4)
        pts1 = torch.rand(batch_size, num_frames, num_points, 3)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3)
        Fm = torch.stack([utils.create_random_fundamental_matrix(1).type_as(pts1) for _ in range(num_frames)], dim=1)
        dist_frame_by_frame = torch.stack([epi.symmetrical_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...]) for t in range(num_frames)], dim=1)
        dist_all_frames = epi.symmetrical_epipolar_distance(pts1, pts2, Fm)
        assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-06, rtol=1e-06)

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        (batch_size, num_points, num_dims) = (2, 3, 2)
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = utils.create_random_fundamental_matrix(batch_size).type_as(points2)
        assert gradcheck(epi.symmetrical_epipolar_distance, (points1, points2, Fm), raise_exception=True, fast_mode=True)

    def test_shift(self, device, dtype):
        if False:
            return 10
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 2.0, 8.0], device=device, dtype=dtype)[None]
        assert_close(epi.symmetrical_epipolar_distance(pts1, pts2, Fm), expected, atol=0.0001, rtol=0.0001)

class TestSampsonEpipolarDistance:

    def test_smoke(self, device, dtype):
        if False:
            while True:
                i = 10
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.sampson_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        if False:
            return 10
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.sampson_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        if False:
            i = 10
            return i + 15
        (batch_size, num_frames, num_points) = (5, 3, 4)
        pts1 = torch.rand(batch_size, num_frames, num_points, 3)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3)
        Fm = torch.stack([utils.create_random_fundamental_matrix(1).type_as(pts1) for _ in range(num_frames)], dim=1)
        dist_frame_by_frame = torch.stack([epi.sampson_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...]) for t in range(num_frames)], dim=1)
        dist_all_frames = epi.sampson_epipolar_distance(pts1, pts2, Fm)
        assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-06, rtol=1e-06)

    def test_shift(self, device, dtype):
        if False:
            return 10
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 0.5, 2.0], device=device, dtype=dtype)[None]
        assert_close(epi.sampson_epipolar_distance(pts1, pts2, Fm), expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        (batch_size, num_points, num_dims) = (2, 3, 2)
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = utils.create_random_fundamental_matrix(batch_size).type_as(points2)
        assert gradcheck(epi.sampson_epipolar_distance, (points1, points2, Fm), raise_exception=True, fast_mode=True)

class TestLeftToRightEpipolarDistance:

    def test_smoke(self, device, dtype):
        if False:
            while True:
                i = 10
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.left_to_right_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        if False:
            return 10
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.left_to_right_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        if False:
            while True:
                i = 10
        (batch_size, num_frames, num_points) = (5, 3, 4)
        pts1 = torch.rand(batch_size, num_frames, num_points, 3)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3)
        Fm = torch.stack([utils.create_random_fundamental_matrix(1).type_as(pts1) for _ in range(num_frames)], dim=1)
        dist_frame_by_frame = torch.stack([epi.left_to_right_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...]) for t in range(num_frames)], dim=1)
        dist_all_frames = epi.left_to_right_epipolar_distance(pts1, pts2, Fm)
        assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-06, rtol=1e-06)

    def test_shift(self, device, dtype):
        if False:
            i = 10
            return i + 15
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)[None]
        assert_close(epi.left_to_right_epipolar_distance(pts1, pts2, Fm), expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            return 10
        (batch_size, num_points, num_dims) = (2, 3, 2)
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = utils.create_random_fundamental_matrix(batch_size).type_as(points2)
        assert gradcheck(epi.left_to_right_epipolar_distance, (points1, points2, Fm), raise_exception=True, fast_mode=True)

class TestRightToLeftEpipolarDistance:

    def test_smoke(self, device, dtype):
        if False:
            i = 10
            return i + 15
        pts1 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(1, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.right_to_left_epipolar_distance(pts1, pts2, Fm).shape == (1, 4)

    def test_batch(self, device, dtype):
        if False:
            return 10
        batch_size = 5
        pts1 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        pts2 = torch.rand(batch_size, 4, 3, device=device, dtype=dtype)
        Fm = utils.create_random_fundamental_matrix(1).type_as(pts1)
        assert epi.right_to_left_epipolar_distance(pts1, pts2, Fm).shape == (5, 4)

    def test_frames(self, device, dtype):
        if False:
            while True:
                i = 10
        (batch_size, num_frames, num_points) = (5, 3, 4)
        pts1 = torch.rand(batch_size, num_frames, num_points, 3)
        pts2 = torch.rand(batch_size, num_frames, num_points, 3)
        Fm = torch.stack([utils.create_random_fundamental_matrix(1).type_as(pts1) for _ in range(num_frames)], dim=1)
        dist_frame_by_frame = torch.stack([epi.right_to_left_epipolar_distance(pts1[:, t, ...], pts2[:, t, ...], Fm[:, t, ...]) for t in range(num_frames)], dim=1)
        dist_all_frames = epi.right_to_left_epipolar_distance(pts1, pts2, Fm)
        assert_close(dist_frame_by_frame, dist_all_frames, atol=1e-06, rtol=1e-06)

    def test_shift(self, device, dtype):
        if False:
            while True:
                i = 10
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 0.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        expected = torch.tensor([0.0, 1.0, 2.0], device=device, dtype=dtype)[None]
        assert_close(epi.right_to_left_epipolar_distance(pts1, pts2, Fm), expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        (batch_size, num_points, num_dims) = (2, 3, 2)
        points1 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(batch_size, num_points, num_dims, device=device, dtype=torch.float64)
        Fm = utils.create_random_fundamental_matrix(batch_size).type_as(points2)
        assert gradcheck(epi.right_to_left_epipolar_distance, (points1, points2, Fm), raise_exception=True, fast_mode=True)