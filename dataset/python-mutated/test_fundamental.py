from typing import Dict
import pytest
import test_common as utils
import torch
from torch.autograd import gradcheck
import kornia.geometry.epipolar as epi
import kornia.testing as utils2
from kornia.testing import assert_close

class TestNormalizePoints:

    def test_smoke(self, device, dtype):
        if False:
            return 10
        points = torch.rand(1, 1, 2, device=device, dtype=dtype)
        output = epi.normalize_points(points)
        assert len(output) == 2
        assert output[0].shape == (1, 1, 2)
        assert output[1].shape == (1, 3, 3)

    @pytest.mark.parametrize('batch_size, num_points', [(1, 2), (2, 3), (3, 2)])
    def test_shape(self, batch_size, num_points, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        (B, N) = (batch_size, num_points)
        points = torch.rand(B, N, 2, device=device, dtype=dtype)
        output = epi.normalize_points(points)
        assert output[0].shape == (B, N, 2)
        assert output[1].shape == (B, 3, 3)

    def test_mean_std(self, device, dtype):
        if False:
            i = 10
            return i + 15
        points = torch.tensor([[[0.0, 0.0], [0.0, 2.0], [1.0, 1.0], [1.0, 3.0]]], device=device, dtype=dtype)
        (points_norm, _) = epi.normalize_points(points)
        (points_std, points_mean) = torch.std_mean(points_norm, dim=1)
        assert_close(points_mean, torch.zeros_like(points_mean))
        assert (points_std < 2.0).all()

    def test_gradcheck(self, device):
        if False:
            i = 10
            return i + 15
        points = torch.rand(2, 3, 2, device=device, requires_grad=True, dtype=torch.float64)
        assert gradcheck(epi.normalize_points, (points,), raise_exception=True, fast_mode=True)

class TestNormalizeTransformation:

    def test_smoke(self, device, dtype):
        if False:
            i = 10
            return i + 15
        trans = torch.rand(2, 2, device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert trans_norm.shape == (2, 2)

    @pytest.mark.parametrize('batch_size, rows, cols', [(1, 2, 2), (2, 3, 3), (3, 4, 4), (2, 1, 2)])
    def test_shape(self, batch_size, rows, cols, device, dtype):
        if False:
            while True:
                i = 10
        (B, N, M) = (batch_size, rows, cols)
        trans = torch.rand(B, N, M, device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert trans_norm.shape == (B, N, M)

    def test_check_last_val(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        trans = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.5, 0.0, 0.5]]], device=device, dtype=dtype)
        trans_expected = torch.tensor([[[0.0, 0.0, 2.0], [0.0, 4.0, 0.0], [1.0, 0.0, 1.0]]], device=device, dtype=dtype)
        trans_norm = epi.normalize_transformation(trans)
        assert_close(trans_norm, trans_expected, atol=0.0001, rtol=0.0001)

    def test_check_corner_case(self, device, dtype):
        if False:
            return 10
        trans = torch.tensor([[[0.0, 0.0, 1.0], [0.0, 2.0, 0.0], [0.5, 0.0, 0.0]]], device=device, dtype=dtype)
        trans_expected = trans.clone()
        trans_norm = epi.normalize_transformation(trans)
        assert_close(trans_norm, trans_expected, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        trans = torch.rand(2, 3, 3, device=device, requires_grad=True, dtype=torch.float64)
        assert gradcheck(epi.normalize_transformation, (trans,), raise_exception=True, fast_mode=True)

class TestFindFundamental:

    def test_smoke(self, device, dtype):
        if False:
            return 10
        points1 = torch.rand(1, 8, 2, device=device, dtype=dtype)
        points2 = torch.rand(1, 8, 2, device=device, dtype=dtype)
        weights = torch.ones(1, 8, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize('batch_size, num_points', [(1, 8), (2, 9), (3, 10)])
    def test_shape(self, batch_size, num_points, device, dtype):
        if False:
            i = 10
            return i + 15
        (B, N) = (batch_size, num_points)
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = torch.ones(B, N, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (B, 3, 3)

    @pytest.mark.parametrize('batch_size, num_points', [(1, 8), (2, 8), (3, 10)])
    def test_shape_noweights(self, batch_size, num_points, device, dtype):
        if False:
            print('Hello World!')
        (B, N) = (batch_size, num_points)
        points1 = torch.rand(B, N, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, N, 2, device=device, dtype=dtype)
        weights = None
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert F_mat.shape == (B, 3, 3)

    @pytest.mark.parametrize('batch_size', [1, 2, 3])
    def test_shape_7point(self, batch_size, device, dtype):
        if False:
            print('Hello World!')
        B = batch_size
        points1 = torch.rand(B, 7, 2, device=device, dtype=dtype)
        points2 = torch.rand(B, 7, 2, device=device, dtype=dtype)
        torch.ones(B, 7, device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, method='7POINT')
        assert F_mat.shape == (B, 3, 3, 3)

    def test_opencv_svd(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        points1 = torch.tensor([[[0.8569, 0.5982], [0.0059, 0.9649], [0.1968, 0.8846], [0.6084, 0.3467], [0.9633, 0.5274], [0.8941, 0.8939], [0.0863, 0.5133], [0.2645, 0.8882], [0.2411, 0.3045], [0.8199, 0.4107]]], device=device, dtype=dtype)
        points2 = torch.tensor([[[0.0928, 0.3013], [0.0989, 0.9649], [0.0341, 0.4827], [0.8294, 0.4469], [0.223, 0.2998], [0.1722, 0.8182], [0.5264, 0.8869], [0.8908, 0.1233], [0.2338, 0.7663], [0.4466, 0.5696]]], device=device, dtype=dtype)
        weights = torch.ones(1, 10, device=device, dtype=dtype)
        Fm_expected = torch.tensor([[[-0.47408533, 0.22033807, -0.00346677], [0.54935973, 1.31080955, -1.25028275], [-0.36690215, -1.08143769, 1.0]]], device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, weights)
        assert_close(F_mat, Fm_expected, rtol=0.0001, atol=0.0001)

    def test_7point_opencv(self, device, dtype):
        if False:
            return 10
        points1 = torch.tensor([[[0.8569, 0.5982], [0.0059, 0.9649], [0.1968, 0.8846], [0.6084, 0.3467], [0.9633, 0.5274], [0.8941, 0.8939], [0.0863, 0.5133]]], device=device, dtype=dtype)
        points2 = torch.tensor([[[0.0928, 0.3013], [0.0989, 0.9649], [0.0341, 0.4827], [0.8294, 0.4469], [0.223, 0.2998], [0.1722, 0.8182], [0.5264, 0.8869]]], device=device, dtype=dtype)
        Fm_expected = torch.tensor([[[[-2.87490907, 5.41934672, 0.73871396], [0.34010174, 3.70371623, -4.65517276], [-0.1809933, -0.56577107, 1.0]], [[0.14465888, 0.68711702, -0.65570944], [0.53424758, 0.7988479, -0.75446946], [-0.48201197, -1.05375511, 1.0]], [[-0.0901827, 1.05515785, -0.54726062], [0.51914823, 1.02476892, -1.05783979], [-0.45860077, -1.01580301, 1.0]]]], device=device, dtype=dtype)
        F_mat = epi.find_fundamental(points1, points2, method='7POINT')
        assert_close(F_mat, Fm_expected, rtol=0.001, atol=0.001)

    def test_synthetic_sampson_7point(self, device, dtype):
        if False:
            while True:
                i = 10
        scene: Dict[str, torch.Tensor] = utils.generate_two_view_random_scene(device, dtype)
        x1 = scene['x1'][:, :7, :]
        x2 = scene['x2'][:, :7, :]
        F_est = epi.find_fundamental(x1, x2, None, '7POINT')
        for i in range(3):
            F = F_est[0][i].unsqueeze(0)
            if torch.all(F != 0):
                error = epi.sampson_epipolar_distance(x1, x2, F)
                assert_close(error, torch.zeros((F.shape[0], 7), device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

    def test_epipolar_constraint_7point(self, device, dtype):
        if False:
            return 10
        scene: Dict[str, torch.Tensor] = utils.generate_two_view_random_scene(device, dtype)
        x1 = scene['x1'][:, :7, :]
        x2 = scene['x2'][:, :7, :]
        F_est = epi.find_fundamental(x1, x2, None, '7POINT')
        for i in range(3):
            F = F_est[0][i].unsqueeze(0)
            if torch.all(F != 0):
                distance = epi.symmetrical_epipolar_distance(x1, x2, F)
                mean_error = distance.mean()
                assert_close(mean_error, torch.tensor(0.0, device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

    def test_synthetic_sampson(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        scene: Dict[str, torch.Tensor] = utils.generate_two_view_random_scene(device, dtype)
        x1 = scene['x1']
        x2 = scene['x2']
        weights = torch.ones_like(x1)[..., 0]
        F_est = epi.find_fundamental(x1, x2, weights)
        error = epi.sampson_epipolar_distance(x1, x2, F_est)
        assert_close(error, torch.zeros(x1.shape[:2], device=device, dtype=dtype), atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        points1 = torch.rand(1, 10, 2, device=device, dtype=torch.float64, requires_grad=True)
        points2 = torch.rand(1, 10, 2, device=device, dtype=torch.float64)
        weights = torch.ones(1, 10, device=device, dtype=torch.float64)
        assert gradcheck(epi.find_fundamental, (points1, points2, weights), raise_exception=True, fast_mode=True)

class TestComputeCorrespondEpilines:

    def test_smoke(self, device, dtype):
        if False:
            i = 10
            return i + 15
        point = torch.rand(1, 1, 2, device=device, dtype=dtype)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        lines = epi.compute_correspond_epilines(point, F_mat)
        assert lines.shape == (1, 1, 3)

    @pytest.mark.parametrize('batch_size, num_points', [(1, 2), (2, 3), (3, 2)])
    def test_shape(self, batch_size, num_points, device, dtype):
        if False:
            print('Hello World!')
        (B, N) = (batch_size, num_points)
        point = torch.rand(B, N, 2, device=device, dtype=dtype)
        F_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        lines = epi.compute_correspond_epilines(point, F_mat)
        assert lines.shape == (B, N, 3)

    @pytest.mark.parametrize('batch_size, num_frames, num_points', [(1, 1, 1), (1, 1, 2), (1, 2, 1), (1, 2, 2), (2, 1, 1), (2, 1, 2), (2, 2, 1), (2, 2, 2)])
    def test_volumetric(self, batch_size, num_frames, num_points, device, dtype):
        if False:
            print('Hello World!')
        (B, T, N) = (batch_size, num_frames, num_points)
        point = torch.rand(B, T, N, 2, device=device, dtype=dtype)
        F_mat = torch.rand(B, T, 3, 3, device=device, dtype=dtype)
        lines_T_hops = torch.zeros(B, T, N, 3, device=device, dtype=dtype)
        for i in range(T):
            lines_T_hops[:, i, ...] = epi.compute_correspond_epilines(point[:, i, ...], F_mat[:, i, ...])
        lines_one_hop = epi.compute_correspond_epilines(point, F_mat)
        assert_close(lines_T_hops, lines_one_hop, atol=2e-07, rtol=2e-07)

    def test_opencv(self, device, dtype):
        if False:
            return 10
        point = torch.rand(1, 2, 2, device=device, dtype=dtype)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        point = torch.tensor([[[0.9794, 0.7994], [0.8163, 0.85]]], device=device, dtype=dtype)
        F_mat = torch.tensor([[[0.1185, 0.4438, 0.9869], [0.567, 0.9447, 0.41], [0.1546, 0.2554, 0.4485]]], device=device, dtype=dtype)
        lines_expected = torch.tensor([[[0.64643687, 0.7629675, 0.35658622], [0.65710586, 0.7537983, 0.35616538]]], device=device, dtype=dtype)
        lines_est = epi.compute_correspond_epilines(point, F_mat)
        assert_close(lines_est, lines_expected, rtol=0.0001, atol=0.0001)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        point = torch.rand(1, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        F_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.compute_correspond_epilines, (point, F_mat), raise_exception=True, fast_mode=True)

class TestFundamentlFromEssential:

    def test_smoke(self, device, dtype):
        if False:
            print('Hello World!')
        E_mat = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize('batch_size', [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        if False:
            print('Hello World!')
        B: int = batch_size
        E_mat = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(B, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 3, 3, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (B, 3, 3)

    def test_shape_large(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        E_mat = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K1 = torch.rand(1, 2, 3, 3, device=device, dtype=dtype)
        K2 = torch.rand(1, 1, 3, 3, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_essential(E_mat, K1, K2)
        assert F_mat.shape == (1, 2, 3, 3)

    def test_from_to_essential(self, device, dtype):
        if False:
            while True:
                i = 10
        scene = utils.generate_two_view_random_scene(device, dtype)
        F_mat = scene['F']
        E_mat = epi.essential_from_fundamental(F_mat, scene['K1'], scene['K2'])
        F_hat = epi.fundamental_from_essential(E_mat, scene['K1'], scene['K2'])
        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        assert_close(F_mat_norm, F_hat_norm, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        E_mat = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        K1 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        K2 = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.fundamental_from_essential, (E_mat, K1, K2), raise_exception=True, fast_mode=True)

class TestFundamentalFromProjections:

    def test_smoke(self, device, dtype):
        if False:
            print('Hello World!')
        P1 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (1, 3, 3)

    @pytest.mark.parametrize('batch_size', [1, 2, 4, 7])
    def test_shape(self, batch_size, device, dtype):
        if False:
            i = 10
            return i + 15
        B: int = batch_size
        P1 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(B, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (B, 3, 3)

    def test_shape_large(self, device, dtype):
        if False:
            return 10
        P1 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        P2 = torch.rand(1, 2, 3, 4, device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        assert F_mat.shape == (1, 2, 3, 3)

    def test_from_to_projections(self, device, dtype):
        if False:
            for i in range(10):
                print('nop')
        P1 = torch.tensor([[[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [1.0, 0.0, 1.0, 0.0]]], device=device, dtype=dtype)
        P2 = torch.tensor([[[1.0, 1.0, 1.0, 3.0], [0.0, 2.0, 0.0, 3.0], [0.0, 1.0, 1.0, 0.0]]], device=device, dtype=dtype)
        F_mat = epi.fundamental_from_projections(P1, P2)
        P_mat = epi.projections_from_fundamental(F_mat)
        F_hat = epi.fundamental_from_projections(P_mat[..., 0], P_mat[..., 1])
        F_mat_norm = epi.normalize_transformation(F_mat)
        F_hat_norm = epi.normalize_transformation(F_hat)
        assert_close(F_mat_norm, F_hat_norm, atol=0.0001, rtol=0.0001)

    def test_gradcheck(self, device):
        if False:
            print('Hello World!')
        P1 = torch.rand(1, 3, 4, device=device, dtype=torch.float64, requires_grad=True)
        P2 = torch.rand(1, 3, 4, device=device, dtype=torch.float64)
        assert gradcheck(epi.fundamental_from_projections, (P1, P2), raise_exception=True, fast_mode=True)

    def test_batch_support_check(self, device, dtype):
        if False:
            i = 10
            return i + 15
        P1_batch = torch.tensor([[[946.92, -966.58, 608.62, -230760.0], [-218.29, 541.63, 1344.5, -643870.0], [-0.60675, -0.69807, 0.38021, 388.96]], [[946.92, -966.58, 608.62, -230760.0], [-218.29, 541.63, 1344.5, -643870.0], [-0.60675, -0.69807, 0.38021, 388.96]]], device=device, dtype=dtype)
        P1 = torch.tensor([[[946.92, -966.58, 608.62, -230760.0], [-218.29, 541.63, 1344.5, -643870.0], [-0.60675, -0.69807, 0.38021, 388.96]]], device=device, dtype=dtype)
        P2_batch = torch.tensor([[[1151.8, -758.22, 547.64, -197640.0], [-215.48, 531.02, 1349.2, -647310.0], [-0.43727, -0.78632, 0.43646, 345.15]], [[995.95, -864.64, 679.59, -275170.0], [-81.716, 778.26, 1239.5, -581370.0], [-0.5709, -0.60416, 0.55594, 281.11]]], device=device, dtype=dtype)
        P2 = torch.tensor([[[1151.8, -758.22, 547.64, -197640.0], [-215.48, 531.02, 1349.2, -647310.0], [-0.43727, -0.78632, 0.43646, 345.15]]], device=device, dtype=dtype)
        F_batch = epi.fundamental_from_projections(P1_batch, P2_batch)
        F = epi.fundamental_from_projections(P1, P2)
        assert_close(F_batch[0], F[0])

class TestPerpendicular:

    def test_shape(self, device, dtype):
        if False:
            while True:
                i = 10
        lines = torch.rand(2, 4, 3, device=device, dtype=dtype)
        points = torch.rand(2, 4, 2, device=device, dtype=dtype)
        perp = epi.get_perpendicular(lines, points)
        assert perp.shape == (2, 4, 3)

    def test_result(self, device, dtype):
        if False:
            while True:
                i = 10
        points = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], device=device, dtype=dtype)
        lines = torch.tensor([[[1.0, -1.0, 0.0], [0.0, 1.0, 1.0]]], device=device, dtype=dtype)
        perp = epi.get_perpendicular(lines, points)
        expected = torch.tensor([[[1.0, 1.0, -1.0], [-1.0, 0.0, 0.0]]], device=device, dtype=dtype)
        assert_close(perp, expected)

    def test_gradcheck(self, device):
        if False:
            print('Hello World!')
        pt = torch.rand(1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
        line = torch.rand(1, 3, 3, device=device, dtype=torch.float64)
        assert gradcheck(epi.get_perpendicular, (pt, line), raise_exception=True, fast_mode=True)

class TestGetClosestPointOnEpipolarLine:

    def test_shape(self, device, dtype):
        if False:
            while True:
                i = 10
        pts1 = torch.rand(2, 4, 2, device=device, dtype=dtype)
        pts2 = torch.rand(2, 4, 2, device=device, dtype=dtype)
        Fm = utils2.create_random_fundamental_matrix(1).type_as(pts1)
        perp = epi.get_closest_point_on_epipolar_line(pts1, pts2, Fm)
        assert perp.shape == (2, 4, 2)

    def test_shift(self, device, dtype):
        if False:
            while True:
                i = 10
        pts1 = torch.zeros(3, 2, device=device, dtype=dtype)[None]
        pts2 = torch.tensor([[2, 4.0], [2, 1], [2, 2.0]], device=device, dtype=dtype)[None]
        Fm = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]], dtype=dtype, device=device)[None]
        cp = epi.get_closest_point_on_epipolar_line(pts1, pts2, Fm)
        expected = torch.tensor([[[2.0, 0.0], [2.0, 0.0], [2.0, 0.0]]], device=device, dtype=dtype)
        assert_close(cp, expected)

    def test_gradcheck(self, device):
        if False:
            while True:
                i = 10
        pts1 = torch.rand(2, 4, 2, device=device, dtype=torch.float64, requires_grad=True)
        pts2 = torch.rand(2, 4, 2, device=device, dtype=torch.float64)
        Fm = utils2.create_random_fundamental_matrix(1).type_as(pts1)
        assert gradcheck(epi.get_closest_point_on_epipolar_line, (pts1, pts2, Fm), raise_exception=True, fast_mode=True)