import logging
import unittest
import numpy as np
from numpy.testing import assert_array_equal
from scipy import sparse
from scipy.sparse import csc_matrix
from scipy.special import psi
import gensim.matutils as matutils

def logsumexp(x):
    if False:
        while True:
            i = 10
    "Log of sum of exponentials.\n\n    Parameters\n    ----------\n    x : numpy.ndarray\n        Input 2d matrix.\n\n    Returns\n    -------\n    float\n        log of sum of exponentials of elements in `x`.\n\n    Warnings\n    --------\n    By performance reasons, doesn't support NaNs or 1d, 3d, etc arrays like :func:`scipy.special.logsumexp`.\n\n    "
    x_max = np.max(x)
    x = np.log(np.sum(np.exp(x - x_max)))
    x += x_max
    return x

def mean_absolute_difference(a, b):
    if False:
        return 10
    'Mean absolute difference between two arrays.\n\n    Parameters\n    ----------\n    a : numpy.ndarray\n        Input 1d array.\n    b : numpy.ndarray\n        Input 1d array.\n\n    Returns\n    -------\n    float\n        mean(abs(a - b)).\n\n    '
    return np.mean(np.abs(a - b))

def dirichlet_expectation(alpha):
    if False:
        while True:
            i = 10
    'For a vector :math:`\\theta \\sim Dir(\\alpha)`, compute :math:`E[log \\theta]`.\n\n    Parameters\n    ----------\n    alpha : numpy.ndarray\n        Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector.\n\n    Returns\n    -------\n    numpy.ndarray:\n        :math:`E[log \\theta]`\n\n    '
    if len(alpha.shape) == 1:
        result = psi(alpha) - psi(np.sum(alpha))
    else:
        result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
    return result.astype(alpha.dtype, copy=False)
dirichlet_expectation_1d = dirichlet_expectation
dirichlet_expectation_2d = dirichlet_expectation

class TestLdaModelInner(unittest.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        self.random_state = np.random.RandomState()
        self.num_runs = 100
        self.num_topics = 100

    def test_log_sum_exp(self):
        if False:
            i = 10
            return i + 15
        rs = self.random_state
        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input = rs.uniform(-1000, 1000, size=(self.num_topics, 1))
                known_good = logsumexp(input)
                test_values = matutils.logsumexp(input)
                msg = 'logsumexp failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def test_mean_absolute_difference(self):
        if False:
            print('Hello World!')
        rs = self.random_state
        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input1 = rs.uniform(-10000, 10000, size=(self.num_topics,))
                input2 = rs.uniform(-10000, 10000, size=(self.num_topics,))
                known_good = mean_absolute_difference(input1, input2)
                test_values = matutils.mean_absolute_difference(input1, input2)
                msg = 'mean_absolute_difference failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

    def test_dirichlet_expectation(self):
        if False:
            for i in range(10):
                print('nop')
        rs = self.random_state
        for dtype in [np.float16, np.float32, np.float64]:
            for i in range(self.num_runs):
                input_1d = rs.uniform(0.01, 10000, size=(self.num_topics,))
                known_good = dirichlet_expectation(input_1d)
                test_values = matutils.dirichlet_expectation(input_1d)
                msg = 'dirichlet_expectation_1d failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)
                input_2d = rs.uniform(0.01, 10000, size=(1, self.num_topics))
                known_good = dirichlet_expectation(input_2d)
                test_values = matutils.dirichlet_expectation(input_2d)
                msg = 'dirichlet_expectation_2d failed for dtype={}'.format(dtype)
                self.assertTrue(np.allclose(known_good, test_values), msg)

def manual_unitvec(vec):
    if False:
        return 10
    vec = vec.astype(float)
    if sparse.issparse(vec):
        vec_sum_of_squares = vec.multiply(vec)
        unit = 1.0 / np.sqrt(vec_sum_of_squares.sum())
        return vec.multiply(unit)
    elif not sparse.issparse(vec):
        sum_vec_squared = np.sum(vec ** 2)
        vec /= np.sqrt(sum_vec_squared)
        return vec

class UnitvecTestCase(unittest.TestCase):

    def test_sparse_npfloat32(self):
        if False:
            while True:
                i = 10
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.float32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_sparse_npfloat64(self):
        if False:
            i = 10
            return i + 15
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.float64)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_sparse_npint32(self):
        if False:
            i = 10
            return i + 15
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.int32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_sparse_npint64(self):
        if False:
            while True:
                i = 10
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(np.int64)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_dense_npfloat32(self):
        if False:
            while True:
                i = 10
        input_vector = np.random.uniform(size=(5,)).astype(np.float32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_dense_npfloat64(self):
        if False:
            while True:
                i = 10
        input_vector = np.random.uniform(size=(5,)).astype(np.float64)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_dense_npint32(self):
        if False:
            i = 10
            return i + 15
        input_vector = np.random.randint(10, size=5).astype(np.int32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_dense_npint64(self):
        if False:
            for i in range(10):
                print('nop')
        input_vector = np.random.randint(10, size=5).astype(np.int32)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_sparse_python_float(self):
        if False:
            print('Hello World!')
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(float)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_sparse_python_int(self):
        if False:
            return 10
        input_vector = sparse.csr_matrix(np.asarray([[1, 0, 0, 0, 3], [0, 0, 4, 3, 0]])).astype(int)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector.data, man_unit_vector.data, atol=0.001))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_dense_python_float(self):
        if False:
            for i in range(10):
                print('nop')
        input_vector = np.random.uniform(size=(5,)).astype(float)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertEqual(input_vector.dtype, unit_vector.dtype)

    def test_dense_python_int(self):
        if False:
            while True:
                i = 10
        input_vector = np.random.randint(10, size=5).astype(int)
        unit_vector = matutils.unitvec(input_vector)
        man_unit_vector = manual_unitvec(input_vector)
        self.assertTrue(np.allclose(unit_vector, man_unit_vector))
        self.assertTrue(np.issubdtype(unit_vector.dtype, np.floating))

    def test_return_norm_zero_vector_scipy_sparse(self):
        if False:
            for i in range(10):
                print('nop')
        input_vector = sparse.csr_matrix([[]], dtype=np.int32)
        return_value = matutils.unitvec(input_vector, return_norm=True)
        self.assertTrue(isinstance(return_value, tuple))
        norm = return_value[1]
        self.assertTrue(isinstance(norm, float))
        self.assertEqual(norm, 1.0)

    def test_return_norm_zero_vector_numpy(self):
        if False:
            return 10
        input_vector = np.array([], dtype=np.int32)
        return_value = matutils.unitvec(input_vector, return_norm=True)
        self.assertTrue(isinstance(return_value, tuple))
        norm = return_value[1]
        self.assertTrue(isinstance(norm, float))
        self.assertEqual(norm, 1.0)

    def test_return_norm_zero_vector_gensim_sparse(self):
        if False:
            i = 10
            return i + 15
        input_vector = []
        return_value = matutils.unitvec(input_vector, return_norm=True)
        self.assertTrue(isinstance(return_value, tuple))
        norm = return_value[1]
        self.assertTrue(isinstance(norm, float))
        self.assertEqual(norm, 1.0)

class TestSparse2Corpus(unittest.TestCase):

    def setUp(self):
        if False:
            i = 10
            return i + 15
        self.orig_array = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.s2c = matutils.Sparse2Corpus(csc_matrix(self.orig_array))

    def test_getitem_slice(self):
        if False:
            return 10
        assert_array_equal(self.s2c[:2].sparse.toarray(), self.orig_array[:, :2])
        assert_array_equal(self.s2c[1:3].sparse.toarray(), self.orig_array[:, 1:3])

    def test_getitem_index(self):
        if False:
            return 10
        self.assertListEqual(self.s2c[1], [(0, 2), (1, 5), (2, 8)])

    def test_getitem_list_of_indices(self):
        if False:
            for i in range(10):
                print('nop')
        assert_array_equal(self.s2c[[1, 2]].sparse.toarray(), self.orig_array[:, [1, 2]])
        assert_array_equal(self.s2c[[1]].sparse.toarray(), self.orig_array[:, [1]])

    def test_getitem_ndarray(self):
        if False:
            i = 10
            return i + 15
        assert_array_equal(self.s2c[np.array([1, 2])].sparse.toarray(), self.orig_array[:, [1, 2]])
        assert_array_equal(self.s2c[np.array([1])].sparse.toarray(), self.orig_array[:, [1]])

    def test_getitem_range(self):
        if False:
            i = 10
            return i + 15
        assert_array_equal(self.s2c[range(1, 3)].sparse.toarray(), self.orig_array[:, [1, 2]])
        assert_array_equal(self.s2c[range(1, 2)].sparse.toarray(), self.orig_array[:, [1]])

    def test_getitem_ellipsis(self):
        if False:
            while True:
                i = 10
        assert_array_equal(self.s2c[...].sparse.toarray(), self.orig_array)
if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.DEBUG)
    unittest.main()