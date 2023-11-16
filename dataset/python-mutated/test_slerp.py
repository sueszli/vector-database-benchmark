import numpy as np
from numpy.testing import assert_allclose
import pytest
from scipy.spatial import geometric_slerp

def _generate_spherical_points(ndim=3, n_pts=2):
    if False:
        return 10
    np.random.seed(123)
    points = np.random.normal(size=(n_pts, ndim))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return (points[0], points[1])

class TestGeometricSlerp:

    @pytest.mark.parametrize('n_dims', [2, 3, 5, 7, 9])
    @pytest.mark.parametrize('n_pts', [0, 3, 17])
    def test_shape_property(self, n_dims, n_pts):
        if False:
            while True:
                i = 10
        (start, end) = _generate_spherical_points(n_dims, 2)
        actual = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, n_pts))
        assert actual.shape == (n_pts, n_dims)

    @pytest.mark.parametrize('n_dims', [2, 3, 5, 7, 9])
    @pytest.mark.parametrize('n_pts', [3, 17])
    def test_include_ends(self, n_dims, n_pts):
        if False:
            i = 10
            return i + 15
        (start, end) = _generate_spherical_points(n_dims, 2)
        actual = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, n_pts))
        assert_allclose(actual[0], start)
        assert_allclose(actual[-1], end)

    @pytest.mark.parametrize('start, end', [(np.zeros((1, 3)), np.ones((1, 3))), (np.zeros((1, 3)), np.ones(3)), (np.zeros(1), np.ones((3, 1)))])
    def test_input_shape_flat(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='one-dimensional'):
            geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize('start, end', [(np.zeros(7), np.ones(3)), (np.zeros(2), np.ones(1)), (np.array([]), np.ones(3))])
    def test_input_dim_mismatch(self, start, end):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='dimensions'):
            geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize('start, end', [(np.array([]), np.array([]))])
    def test_input_at_least1d(self, start, end):
        if False:
            return 10
        with pytest.raises(ValueError, match='at least two-dim'):
            geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))

    @pytest.mark.parametrize('start, end, expected', [(np.array([0, 0, 1.0]), np.array([0, 0, -1.0]), 'warning'), (np.array([0.0, -0.0006108652, 0.999999813]), np.array([0, 0, -1.0]), 'warning'), (np.array([0.0, -0.000959930941, 0.999999539]), np.array([0, 0, -1.0]), 'success')])
    def test_handle_antipodes(self, start, end, expected):
        if False:
            print('Hello World!')
        if expected == 'warning':
            with pytest.warns(UserWarning, match='antipodes'):
                res = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))
        else:
            res = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 10))
        assert_allclose(np.linalg.norm(res, axis=1), 1.0)

    @pytest.mark.parametrize('start, end, expected', [(np.array([1, 0]), np.array([0, 1]), np.array([[1, 0], [np.sqrt(3) / 2, 0.5], [0.5, np.sqrt(3) / 2], [0, 1]])), (np.array([1, 0, 0]), np.array([0, 1, 0]), np.array([[1, 0, 0], [np.sqrt(3) / 2, 0.5, 0], [0.5, np.sqrt(3) / 2, 0], [0, 1, 0]])), (np.array([1, 0, 0, 0, 0]), np.array([0, 1, 0, 0, 0]), np.array([[1, 0, 0, 0, 0], [np.sqrt(3) / 2, 0.5, 0, 0, 0], [0.5, np.sqrt(3) / 2, 0, 0, 0], [0, 1, 0, 0, 0]]))])
    def test_straightforward_examples(self, start, end, expected):
        if False:
            print('Hello World!')
        actual = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 4))
        assert_allclose(actual, expected, atol=1e-16)

    @pytest.mark.parametrize('t', [np.linspace(-20, 20, 300), np.linspace(-0.0001, 0.0001, 17)])
    def test_t_values_limits(self, t):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='interpolation parameter'):
            _ = geometric_slerp(start=np.array([1, 0]), end=np.array([0, 1]), t=t)

    @pytest.mark.parametrize('start, end', [(np.array([1]), np.array([0])), (np.array([0]), np.array([1])), (np.array([-17.7]), np.array([165.9]))])
    def test_0_sphere_handling(self, start, end):
        if False:
            while True:
                i = 10
        with pytest.raises(ValueError, match='at least two-dim'):
            _ = geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 4))

    @pytest.mark.parametrize('tol', [5, '7', [5, 6, 7], np.array(9.0)])
    def test_tol_type(self, tol):
        if False:
            i = 10
            return i + 15
        with pytest.raises(ValueError, match='must be a float'):
            _ = geometric_slerp(start=np.array([1, 0]), end=np.array([0, 1]), t=np.linspace(0, 1, 5), tol=tol)

    @pytest.mark.parametrize('tol', [-5e-06, -7e-10])
    def test_tol_sign(self, tol):
        if False:
            i = 10
            return i + 15
        _ = geometric_slerp(start=np.array([1, 0]), end=np.array([0, 1]), t=np.linspace(0, 1, 5), tol=tol)

    @pytest.mark.parametrize('start, end', [(np.array([1, 0]), np.array([0, 0])), (np.array([1 + 1e-06, 0, 0]), np.array([0, 1 - 1e-06, 0])), (np.array([1 + 1e-06, 0, 0, 0]), np.array([0, 1 - 1e-06, 0, 0]))])
    def test_unit_sphere_enforcement(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError, match='unit n-sphere'):
            geometric_slerp(start=start, end=end, t=np.linspace(0, 1, 5))

    @pytest.mark.parametrize('start, end', [(np.array([1, 0]), np.array([np.sqrt(2) / 2.0, np.sqrt(2) / 2.0])), (np.array([1, 0]), np.array([-np.sqrt(2) / 2.0, np.sqrt(2) / 2.0]))])
    @pytest.mark.parametrize('t_func', [np.linspace, np.logspace])
    def test_order_handling(self, start, end, t_func):
        if False:
            for i in range(10):
                print('nop')
        num_t_vals = 20
        np.random.seed(789)
        forward_t_vals = t_func(0, 10, num_t_vals)
        forward_t_vals /= forward_t_vals.max()
        reverse_t_vals = np.flipud(forward_t_vals)
        shuffled_indices = np.arange(num_t_vals)
        np.random.shuffle(shuffled_indices)
        scramble_t_vals = forward_t_vals.copy()[shuffled_indices]
        forward_results = geometric_slerp(start=start, end=end, t=forward_t_vals)
        reverse_results = geometric_slerp(start=start, end=end, t=reverse_t_vals)
        scrambled_results = geometric_slerp(start=start, end=end, t=scramble_t_vals)
        assert_allclose(forward_results, np.flipud(reverse_results))
        assert_allclose(forward_results[shuffled_indices], scrambled_results)

    @pytest.mark.parametrize('t', ['15, 5, 7'])
    def test_t_values_conversion(self, t):
        if False:
            for i in range(10):
                print('nop')
        with pytest.raises(ValueError):
            _ = geometric_slerp(start=np.array([1]), end=np.array([0]), t=t)

    def test_accept_arraylike(self):
        if False:
            return 10
        actual = geometric_slerp([1, 0], [0, 1], [0, 1 / 3, 0.5, 2 / 3, 1])
        expected = np.array([[1, 0], [np.sqrt(3) / 2, 0.5], [np.sqrt(2) / 2, np.sqrt(2) / 2], [0.5, np.sqrt(3) / 2], [0, 1]], dtype=np.float64)
        assert_allclose(actual, expected, atol=1e-16)

    def test_scalar_t(self):
        if False:
            for i in range(10):
                print('nop')
        actual = geometric_slerp([1, 0], [0, 1], 0.5)
        expected = np.array([np.sqrt(2) / 2, np.sqrt(2) / 2], dtype=np.float64)
        assert actual.shape == (2,)
        assert_allclose(actual, expected)

    @pytest.mark.parametrize('start', [np.array([1, 0, 0]), np.array([0, 1])])
    @pytest.mark.parametrize('t', [np.array(1), np.array([1]), np.array([[1]]), np.array([[[1]]]), np.array([]), np.linspace(0, 1, 5)])
    def test_degenerate_input(self, start, t):
        if False:
            i = 10
            return i + 15
        if np.asarray(t).ndim > 1:
            with pytest.raises(ValueError):
                geometric_slerp(start=start, end=start, t=t)
        else:
            shape = (t.size,) + start.shape
            expected = np.full(shape, start)
            actual = geometric_slerp(start=start, end=start, t=t)
            assert_allclose(actual, expected)
            non_degenerate = geometric_slerp(start=start, end=start[::-1], t=t)
            assert actual.size == non_degenerate.size

    @pytest.mark.parametrize('k', np.logspace(-10, -1, 10))
    def test_numerical_stability_pi(self, k):
        if False:
            return 10
        angle = np.pi - k
        ts = np.linspace(0, 1, 100)
        P = np.array([1, 0, 0, 0])
        Q = np.array([np.cos(angle), np.sin(angle), 0, 0])
        with np.testing.suppress_warnings() as sup:
            sup.filter(UserWarning)
            result = geometric_slerp(P, Q, ts, 1e-18)
            norms = np.linalg.norm(result, axis=1)
            error = np.max(np.abs(norms - 1))
            assert error < 4e-15

    @pytest.mark.parametrize('t', [[[0, 0.5]], [[[[[[[[[0, 0.5]]]]]]]]]])
    def test_interpolation_param_ndim(self, t):
        if False:
            for i in range(10):
                print('nop')
        arr1 = np.array([0, 1])
        arr2 = np.array([1, 0])
        with pytest.raises(ValueError):
            geometric_slerp(start=arr1, end=arr2, t=t)
        with pytest.raises(ValueError):
            geometric_slerp(start=arr1, end=arr1, t=t)