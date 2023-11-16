"""Test that Python numpy arrays can be passed to C++ Eigen library."""
import time
from absl.testing import absltest
import numpy as np
import pyspiel_eigen_test

class PyEigenTest(absltest.TestCase):

    def test_square_matrix_elements(self):
        if False:
            i = 10
            return i + 15
        x = np.array([[1, 2], [3, 4]]).astype(float)
        expected = np.array([[1, 2], [3, 4]]) ** 2
        actual = pyspiel_eigen_test.square(x)
        np.testing.assert_array_equal(expected, actual)

    def test_transpose_and_square_matrix_elements(self):
        if False:
            i = 10
            return i + 15
        x = np.array([[1, 2], [3, 4]]).astype(float)
        x = x.transpose()
        expected = np.array([[1, 9], [4, 16]])
        actual = pyspiel_eigen_test.square(x)
        np.testing.assert_array_equal(expected, actual)

    def test_transpose_then_slice_and_square_matrix_elements(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([[1, 2], [3, 4]]).astype(float)
        x = x.transpose()
        expected = np.array([[9], [16]])
        actual = pyspiel_eigen_test.square(x[0:, 1:])
        np.testing.assert_array_equal(expected, actual)

    def test_square_vector_elements(self):
        if False:
            for i in range(10):
                print('nop')
        x = np.array([1, 2, 3]).astype(float)
        expected = np.array([[1], [4], [9]])
        actual = pyspiel_eigen_test.square(x)
        np.testing.assert_array_equal(expected, actual)

    def test_allocate_cxx(self):
        if False:
            i = 10
            return i + 15
        actual = pyspiel_eigen_test.matrix()
        expected = np.array([[1, 2], [3, 4]])
        np.testing.assert_array_equal(expected, actual)

    def test_flags_copy_or_reference(self):
        if False:
            i = 10
            return i + 15
        start = time.time()
        a = pyspiel_eigen_test.BigMatrix()
        print('Alloc: ', time.time() - start)
        start = time.time()
        m = a.get_matrix()
        print('Ref get: ', time.time() - start)
        self.assertTrue(m.flags.writeable)
        self.assertFalse(m.flags.owndata)
        start = time.time()
        v = a.view_matrix()
        print('Ref view: ', time.time() - start)
        self.assertFalse(v.flags.writeable)
        self.assertFalse(v.flags.owndata)
        start = time.time()
        c = a.copy_matrix()
        print('Copy: ', time.time() - start)
        self.assertTrue(c.flags.writeable)
        self.assertTrue(c.flags.owndata)
if __name__ == '__main__':
    absltest.main()