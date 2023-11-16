import numpy as np
from numpy.testing import assert_allclose, assert_array_equal, assert_equal
import pytest
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import distance
from scipy._lib._util import check_random_state

class TestHausdorff:

    def setup_method(self):
        if False:
            return 10
        np.random.seed(1234)
        random_angles = np.random.random(100) * np.pi * 2
        random_columns = np.column_stack((random_angles, random_angles, np.zeros(100)))
        random_columns[..., 0] = np.cos(random_columns[..., 0])
        random_columns[..., 1] = np.sin(random_columns[..., 1])
        random_columns_2 = np.column_stack((random_angles, random_angles, np.zeros(100)))
        random_columns_2[1:, 0] = np.cos(random_columns_2[1:, 0]) * 2.0
        random_columns_2[1:, 1] = np.sin(random_columns_2[1:, 1]) * 2.0
        random_columns_2[0, 0] = np.cos(random_columns_2[0, 0]) * 3.3
        random_columns_2[0, 1] = np.sin(random_columns_2[0, 1]) * 3.3
        self.path_1 = random_columns
        self.path_2 = random_columns_2
        self.path_1_4d = np.insert(self.path_1, 3, 5, axis=1)
        self.path_2_4d = np.insert(self.path_2, 3, 27, axis=1)

    def test_symmetry(self):
        if False:
            while True:
                i = 10
        forward = directed_hausdorff(self.path_1, self.path_2)[0]
        reverse = directed_hausdorff(self.path_2, self.path_1)[0]
        assert forward != reverse

    def test_brute_force_comparison_forward(self):
        if False:
            return 10
        actual = directed_hausdorff(self.path_1, self.path_2)[0]
        expected = max(np.amin(distance.cdist(self.path_1, self.path_2), axis=1))
        assert_allclose(actual, expected)

    def test_brute_force_comparison_reverse(self):
        if False:
            for i in range(10):
                print('nop')
        actual = directed_hausdorff(self.path_2, self.path_1)[0]
        expected = max(np.amin(distance.cdist(self.path_1, self.path_2), axis=0))
        assert_allclose(actual, expected)

    def test_degenerate_case(self):
        if False:
            for i in range(10):
                print('nop')
        actual = directed_hausdorff(self.path_1, self.path_1)[0]
        assert_allclose(actual, 0.0)

    def test_2d_data_forward(self):
        if False:
            i = 10
            return i + 15
        actual = directed_hausdorff(self.path_1[..., :2], self.path_2[..., :2])[0]
        expected = max(np.amin(distance.cdist(self.path_1[..., :2], self.path_2[..., :2]), axis=1))
        assert_allclose(actual, expected)

    def test_4d_data_reverse(self):
        if False:
            i = 10
            return i + 15
        actual = directed_hausdorff(self.path_2_4d, self.path_1_4d)[0]
        expected = max(np.amin(distance.cdist(self.path_1_4d, self.path_2_4d), axis=0))
        assert_allclose(actual, expected)

    def test_indices(self):
        if False:
            for i in range(10):
                print('nop')
        path_simple_1 = np.array([[-1, -12], [0, 0], [1, 1], [3, 7], [1, 2]])
        path_simple_2 = np.array([[0, 0], [1, 1], [4, 100], [10, 9]])
        actual = directed_hausdorff(path_simple_2, path_simple_1)[1:]
        expected = (2, 3)
        assert_array_equal(actual, expected)

    def test_random_state(self):
        if False:
            i = 10
            return i + 15
        rs = check_random_state(None)
        old_global_state = rs.get_state()
        directed_hausdorff(self.path_1, self.path_2)
        rs2 = check_random_state(None)
        new_global_state = rs2.get_state()
        assert_equal(new_global_state, old_global_state)

    @pytest.mark.parametrize('seed', [None, 27870671])
    def test_random_state_None_int(self, seed):
        if False:
            return 10
        rs = check_random_state(None)
        old_global_state = rs.get_state()
        directed_hausdorff(self.path_1, self.path_2, seed)
        rs2 = check_random_state(None)
        new_global_state = rs2.get_state()
        assert_equal(new_global_state, old_global_state)

    def test_invalid_dimensions(self):
        if False:
            return 10
        rng = np.random.default_rng(189048172503940875434364128139223470523)
        A = rng.random((3, 2))
        B = rng.random((3, 5))
        msg = 'need to have the same number of columns'
        with pytest.raises(ValueError, match=msg):
            directed_hausdorff(A, B)

    @pytest.mark.parametrize('A, B, seed, expected', [([(0, 0)], [(0, 1), (0, 0)], 0, (0.0, 0, 1)), ([(0, 0)], [(0, 1), (0, 0)], 1, (0.0, 0, 1)), ([(-5, 3), (0, 0)], [(0, 1), (0, 0), (-5, 3)], 77098, (0.0, 1, 1))])
    def test_subsets(self, A, B, seed, expected):
        if False:
            for i in range(10):
                print('nop')
        actual = directed_hausdorff(u=A, v=B, seed=seed)
        assert_allclose(actual[0], expected[0])
        assert actual[1:] == expected[1:]

@pytest.mark.xslow
def test_massive_arr_overflow():
    if False:
        i = 10
        return i + 15
    try:
        import psutil
    except ModuleNotFoundError:
        pytest.skip('psutil required to check available memory')
    if psutil.virtual_memory().available < 80 * 2 ** 30:
        pytest.skip('insufficient memory available to run this test')
    size = int(3000000000.0)
    arr1 = np.zeros(shape=(size, 2))
    arr2 = np.zeros(shape=(3, 2))
    arr1[size - 1] = [5, 5]
    actual = directed_hausdorff(u=arr1, v=arr2)
    assert_allclose(actual[0], 7.0710678118654755)
    assert_allclose(actual[1], size - 1)