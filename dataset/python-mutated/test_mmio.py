from tempfile import mkdtemp
import os
import io
import shutil
import textwrap
import numpy as np
from numpy import array, transpose, pi
from numpy.testing import assert_equal, assert_allclose, assert_array_equal, assert_array_almost_equal
import pytest
from pytest import raises as assert_raises
import scipy.sparse
import scipy.io._mmio
import scipy.io._fast_matrix_market as fmm
parametrize_args = [('integer', 'int'), ('unsigned-integer', 'uint')]

@pytest.fixture(scope='module', params=(scipy.io._mmio, fmm), autouse=True)
def implementations(request):
    if False:
        return 10
    global mminfo
    global mmread
    global mmwrite
    mminfo = request.param.mminfo
    mmread = request.param.mmread
    mmwrite = request.param.mmwrite

class TestMMIOArray:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        if False:
            return 10
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a, b)

    def check_exact(self, a, info):
        if False:
            print('Hello World!')
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a, b)

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        if False:
            print('Hello World!')
        self.check_exact(array([[1, 2], [3, 4]], dtype=dtype), (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_32bit_integer(self, typeval, dtype):
        if False:
            return 10
        a = array([[2 ** 31 - 1, 2 ** 31 - 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=dtype)
        self.check_exact(a, (2, 2, 4, 'array', typeval, 'general'))

    def test_64bit_integer(self):
        if False:
            print('Hello World!')
        a = array([[2 ** 31, 2 ** 32], [2 ** 63 - 2, 2 ** 63 - 1]], dtype=np.int64)
        if np.intp(0).itemsize < 8 and mmwrite == scipy.io._mmio.mmwrite:
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'array', 'integer', 'general'))

    def test_64bit_unsigned_integer(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([[2 ** 31, 2 ** 32], [2 ** 64 - 2, 2 ** 64 - 1]], dtype=np.uint64)
        self.check_exact(a, (2, 2, 4, 'array', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        if False:
            print('Hello World!')
        self.check_exact(array([[0, 1], [0, 0]], dtype=dtype), (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        if False:
            print('Hello World!')
        self.check_exact(array([[0, 0], [1, 0]], dtype=dtype), (2, 2, 4, 'array', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        if False:
            print('Hello World!')
        self.check_exact(array([[1, 2, 3], [4, 5, 6]], dtype=dtype), (2, 3, 6, 'array', typeval, 'general'))

    def test_simple_rectangular_float(self):
        if False:
            for i in range(10):
                print('nop')
        self.check([[1, 2], [3.5, 4], [5, 6]], (3, 2, 6, 'array', 'real', 'general'))

    def test_simple_float(self):
        if False:
            for i in range(10):
                print('nop')
        self.check([[1, 2], [3, 4.0]], (2, 2, 4, 'array', 'real', 'general'))

    def test_simple_complex(self):
        if False:
            while True:
                i = 10
        self.check([[1, 2], [3, 4j]], (2, 2, 4, 'array', 'complex', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        if False:
            return 10
        self.check_exact(array([[1, 2], [2, 4]], dtype=dtype), (2, 2, 4, 'array', typeval, 'symmetric'))

    def test_simple_skew_symmetric_integer(self):
        if False:
            print('Hello World!')
        self.check_exact([[0, 2], [-2, 0]], (2, 2, 4, 'array', 'integer', 'skew-symmetric'))

    def test_simple_skew_symmetric_float(self):
        if False:
            i = 10
            return i + 15
        self.check(array([[0, 2], [-2.0, 0.0]], 'f'), (2, 2, 4, 'array', 'real', 'skew-symmetric'))

    def test_simple_hermitian_complex(self):
        if False:
            print('Hello World!')
        self.check([[1, 2 + 3j], [2 - 3j, 4]], (2, 2, 4, 'array', 'complex', 'hermitian'))

    def test_random_symmetric_float(self):
        if False:
            return 10
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        self.check(a, (20, 20, 400, 'array', 'real', 'symmetric'))

    def test_random_rectangular_float(self):
        if False:
            for i in range(10):
                print('nop')
        sz = (20, 15)
        a = np.random.random(sz)
        self.check(a, (20, 15, 300, 'array', 'real', 'general'))

    def test_bad_number_of_array_header_fields(self):
        if False:
            i = 10
            return i + 15
        s = '            %%MatrixMarket matrix array real general\n              3  3 999\n            1.0\n            2.0\n            3.0\n            4.0\n            5.0\n            6.0\n            7.0\n            8.0\n            9.0\n            '
        text = textwrap.dedent(s).encode('ascii')
        with pytest.raises(ValueError, match='not of length 2'):
            scipy.io.mmread(io.BytesIO(text))

    def test_gh13634_non_skew_symmetric_int(self):
        if False:
            return 10
        self.check_exact(array([[1, 2], [-2, 99]], dtype=np.int32), (2, 2, 4, 'array', 'integer', 'general'))

    def test_gh13634_non_skew_symmetric_float(self):
        if False:
            i = 10
            return i + 15
        self.check(array([[1, 2], [-2, 99.0]], dtype=np.float32), (2, 2, 4, 'array', 'real', 'general'))

class TestMMIOSparseCSR(TestMMIOArray):

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        if False:
            while True:
                i = 10
        shutil.rmtree(self.tmpdir)

    def check(self, a, info):
        if False:
            return 10
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(a.toarray(), b.toarray())

    def check_exact(self, a, info):
        if False:
            print('Hello World!')
        mmwrite(self.fn, a)
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_equal(a.toarray(), b.toarray())

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_integer(self, typeval, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [3, 4]], dtype=dtype), (2, 2, 4, 'coordinate', typeval, 'general'))

    def test_32bit_integer(self):
        if False:
            return 10
        a = scipy.sparse.csr_matrix(array([[2 ** 31 - 1, -2 ** 31 + 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=np.int32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_64bit_integer(self):
        if False:
            return 10
        a = scipy.sparse.csr_matrix(array([[2 ** 32 + 1, 2 ** 32 + 1], [-2 ** 63 + 2, 2 ** 63 - 2]], dtype=np.int64))
        if np.intp(0).itemsize < 8 and mmwrite == scipy.io._mmio.mmwrite:
            assert_raises(OverflowError, mmwrite, self.fn, a)
        else:
            self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_32bit_unsigned_integer(self):
        if False:
            return 10
        a = scipy.sparse.csr_matrix(array([[2 ** 31 - 1, 2 ** 31 - 2], [2 ** 31 - 3, 2 ** 31 - 4]], dtype=np.uint32))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    def test_64bit_unsigned_integer(self):
        if False:
            for i in range(10):
                print('nop')
        a = scipy.sparse.csr_matrix(array([[2 ** 32 + 1, 2 ** 32 + 1], [2 ** 64 - 2, 2 ** 64 - 1]], dtype=np.uint64))
        self.check_exact(a, (2, 2, 4, 'coordinate', 'unsigned-integer', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_upper_triangle_integer(self, typeval, dtype):
        if False:
            while True:
                i = 10
        self.check_exact(scipy.sparse.csr_matrix([[0, 1], [0, 0]], dtype=dtype), (2, 2, 1, 'coordinate', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_lower_triangle_integer(self, typeval, dtype):
        if False:
            for i in range(10):
                print('nop')
        self.check_exact(scipy.sparse.csr_matrix([[0, 0], [1, 0]], dtype=dtype), (2, 2, 1, 'coordinate', typeval, 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_rectangular_integer(self, typeval, dtype):
        if False:
            return 10
        self.check_exact(scipy.sparse.csr_matrix([[1, 2, 3], [4, 5, 6]], dtype=dtype), (2, 3, 6, 'coordinate', typeval, 'general'))

    def test_simple_rectangular_float(self):
        if False:
            while True:
                i = 10
        self.check(scipy.sparse.csr_matrix([[1, 2], [3.5, 4], [5, 6]]), (3, 2, 6, 'coordinate', 'real', 'general'))

    def test_simple_float(self):
        if False:
            return 10
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4.0]]), (2, 2, 4, 'coordinate', 'real', 'general'))

    def test_simple_complex(self):
        if False:
            return 10
        self.check(scipy.sparse.csr_matrix([[1, 2], [3, 4j]]), (2, 2, 4, 'coordinate', 'complex', 'general'))

    @pytest.mark.parametrize('typeval, dtype', parametrize_args)
    def test_simple_symmetric_integer(self, typeval, dtype):
        if False:
            return 10
        self.check_exact(scipy.sparse.csr_matrix([[1, 2], [2, 4]], dtype=dtype), (2, 2, 3, 'coordinate', typeval, 'symmetric'))

    def test_simple_skew_symmetric_integer(self):
        if False:
            while True:
                i = 10
        self.check_exact(scipy.sparse.csr_matrix([[0, 2], [-2, 0]]), (2, 2, 1, 'coordinate', 'integer', 'skew-symmetric'))

    def test_simple_skew_symmetric_float(self):
        if False:
            i = 10
            return i + 15
        self.check(scipy.sparse.csr_matrix(array([[0, 2], [-2.0, 0]], 'f')), (2, 2, 1, 'coordinate', 'real', 'skew-symmetric'))

    def test_simple_hermitian_complex(self):
        if False:
            for i in range(10):
                print('nop')
        self.check(scipy.sparse.csr_matrix([[1, 2 + 3j], [2 - 3j, 4]]), (2, 2, 3, 'coordinate', 'complex', 'hermitian'))

    def test_random_symmetric_float(self):
        if False:
            print('Hello World!')
        sz = (20, 20)
        a = np.random.random(sz)
        a = a + transpose(a)
        a = scipy.sparse.csr_matrix(a)
        self.check(a, (20, 20, 210, 'coordinate', 'real', 'symmetric'))

    def test_random_rectangular_float(self):
        if False:
            print('Hello World!')
        sz = (20, 15)
        a = np.random.random(sz)
        a = scipy.sparse.csr_matrix(a)
        self.check(a, (20, 15, 300, 'coordinate', 'real', 'general'))

    def test_simple_pattern(self):
        if False:
            for i in range(10):
                print('nop')
        a = scipy.sparse.csr_matrix([[0, 1.5], [3.0, 2.5]])
        p = np.zeros_like(a.toarray())
        p[a.toarray() > 0] = 1
        info = (2, 2, 3, 'coordinate', 'pattern', 'general')
        mmwrite(self.fn, a, field='pattern')
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn)
        assert_array_almost_equal(p, b.toarray())

    def test_gh13634_non_skew_symmetric_int(self):
        if False:
            print('Hello World!')
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99]], dtype=np.int32)
        self.check_exact(a, (2, 2, 4, 'coordinate', 'integer', 'general'))

    def test_gh13634_non_skew_symmetric_float(self):
        if False:
            return 10
        a = scipy.sparse.csr_matrix([[1, 2], [-2, 99.0]], dtype=np.float32)
        self.check(a, (2, 2, 4, 'coordinate', 'real', 'general'))
_32bit_integer_dense_example = '%%MatrixMarket matrix array integer general\n2  2\n2147483647\n2147483646\n2147483647\n2147483646\n'
_32bit_integer_sparse_example = '%%MatrixMarket matrix coordinate integer symmetric\n2  2  2\n1  1  2147483647\n2  2  2147483646\n'
_64bit_integer_dense_example = '%%MatrixMarket matrix array integer general\n2  2\n          2147483648\n-9223372036854775806\n         -2147483648\n 9223372036854775807\n'
_64bit_integer_sparse_general_example = '%%MatrixMarket matrix coordinate integer general\n2  2  3\n1  1           2147483648\n1  2  9223372036854775807\n2  2  9223372036854775807\n'
_64bit_integer_sparse_symmetric_example = '%%MatrixMarket matrix coordinate integer symmetric\n2  2  3\n1  1            2147483648\n1  2  -9223372036854775807\n2  2   9223372036854775807\n'
_64bit_integer_sparse_skew_example = '%%MatrixMarket matrix coordinate integer skew-symmetric\n2  2  3\n1  1            2147483648\n1  2  -9223372036854775807\n2  2   9223372036854775807\n'
_over64bit_integer_dense_example = '%%MatrixMarket matrix array integer general\n2  2\n         2147483648\n9223372036854775807\n         2147483648\n9223372036854775808\n'
_over64bit_integer_sparse_example = '%%MatrixMarket matrix coordinate integer symmetric\n2  2  2\n1  1            2147483648\n2  2  19223372036854775808\n'

class TestMMIOReadLargeIntegers:

    def setup_method(self):
        if False:
            while True:
                i = 10
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        if False:
            for i in range(10):
                print('nop')
        shutil.rmtree(self.tmpdir)

    def check_read(self, example, a, info, dense, over32, over64):
        if False:
            return 10
        with open(self.fn, 'w') as f:
            f.write(example)
        assert_equal(mminfo(self.fn), info)
        if over32 and np.intp(0).itemsize < 8 and (mmwrite == scipy.io._mmio.mmwrite) or over64:
            assert_raises(OverflowError, mmread, self.fn)
        else:
            b = mmread(self.fn)
            if not dense:
                b = b.toarray()
            assert_equal(a, b)

    def test_read_32bit_integer_dense(self):
        if False:
            i = 10
            return i + 15
        a = array([[2 ** 31 - 1, 2 ** 31 - 1], [2 ** 31 - 2, 2 ** 31 - 2]], dtype=np.int64)
        self.check_read(_32bit_integer_dense_example, a, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=False, over64=False)

    def test_read_32bit_integer_sparse(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([[2 ** 31 - 1, 0], [0, 2 ** 31 - 2]], dtype=np.int64)
        self.check_read(_32bit_integer_sparse_example, a, (2, 2, 2, 'coordinate', 'integer', 'symmetric'), dense=False, over32=False, over64=False)

    def test_read_64bit_integer_dense(self):
        if False:
            i = 10
            return i + 15
        a = array([[2 ** 31, -2 ** 31], [-2 ** 63 + 2, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_dense_example, a, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=True, over64=False)

    def test_read_64bit_integer_sparse_general(self):
        if False:
            for i in range(10):
                print('nop')
        a = array([[2 ** 31, 2 ** 63 - 1], [0, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_general_example, a, (2, 2, 3, 'coordinate', 'integer', 'general'), dense=False, over32=True, over64=False)

    def test_read_64bit_integer_sparse_symmetric(self):
        if False:
            print('Hello World!')
        a = array([[2 ** 31, -2 ** 63 + 1], [-2 ** 63 + 1, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_symmetric_example, a, (2, 2, 3, 'coordinate', 'integer', 'symmetric'), dense=False, over32=True, over64=False)

    def test_read_64bit_integer_sparse_skew(self):
        if False:
            while True:
                i = 10
        a = array([[2 ** 31, -2 ** 63 + 1], [2 ** 63 - 1, 2 ** 63 - 1]], dtype=np.int64)
        self.check_read(_64bit_integer_sparse_skew_example, a, (2, 2, 3, 'coordinate', 'integer', 'skew-symmetric'), dense=False, over32=True, over64=False)

    def test_read_over64bit_integer_dense(self):
        if False:
            print('Hello World!')
        self.check_read(_over64bit_integer_dense_example, None, (2, 2, 4, 'array', 'integer', 'general'), dense=True, over32=True, over64=True)

    def test_read_over64bit_integer_sparse(self):
        if False:
            print('Hello World!')
        self.check_read(_over64bit_integer_sparse_example, None, (2, 2, 2, 'coordinate', 'integer', 'symmetric'), dense=False, over32=True, over64=True)
_general_example = '%%MatrixMarket matrix coordinate real general\n%=================================================================================\n%\n% This ASCII file represents a sparse MxN matrix with L\n% nonzeros in the following Matrix Market format:\n%\n% +----------------------------------------------+\n% |%%MatrixMarket matrix coordinate real general | <--- header line\n% |%                                             | <--+\n% |% comments                                    |    |-- 0 or more comment lines\n% |%                                             | <--+\n% |    M  N  L                                   | <--- rows, columns, entries\n% |    I1  J1  A(I1, J1)                         | <--+\n% |    I2  J2  A(I2, J2)                         |    |\n% |    I3  J3  A(I3, J3)                         |    |-- L lines\n% |        . . .                                 |    |\n% |    IL JL  A(IL, JL)                          | <--+\n% +----------------------------------------------+\n%\n% Indices are 1-based, i.e. A(1,1) is the first element.\n%\n%=================================================================================\n  5  5  8\n    1     1   1.000e+00\n    2     2   1.050e+01\n    3     3   1.500e-02\n    1     4   6.000e+00\n    4     2   2.505e+02\n    4     4  -2.800e+02\n    4     5   3.332e+01\n    5     5   1.200e+01\n'
_hermitian_example = '%%MatrixMarket matrix coordinate complex hermitian\n  5  5  7\n    1     1     1.0      0\n    2     2    10.5      0\n    4     2   250.5     22.22\n    3     3     1.5e-2   0\n    4     4    -2.8e2    0\n    5     5    12.       0\n    5     4     0       33.32\n'
_skew_example = '%%MatrixMarket matrix coordinate real skew-symmetric\n  5  5  7\n    1     1     1.0\n    2     2    10.5\n    4     2   250.5\n    3     3     1.5e-2\n    4     4    -2.8e2\n    5     5    12.\n    5     4     0\n'
_symmetric_example = '%%MatrixMarket matrix coordinate real symmetric\n  5  5  7\n    1     1     1.0\n    2     2    10.5\n    4     2   250.5\n    3     3     1.5e-2\n    4     4    -2.8e2\n    5     5    12.\n    5     4     8\n'
_symmetric_pattern_example = '%%MatrixMarket matrix coordinate pattern symmetric\n  5  5  7\n    1     1\n    2     2\n    4     2\n    3     3\n    4     4\n    5     5\n    5     4\n'
_empty_lines_example = '%%MatrixMarket  MATRIX    Coordinate    Real General\n\n   5  5         8\n\n1 1  1.0\n2 2       10.5\n3 3             1.5e-2\n4 4                     -2.8E2\n5 5                              12.\n     1      4      6\n     4      2      250.5\n     4      5      33.32\n\n'

class TestMMIOCoordinate:

    def setup_method(self):
        if False:
            for i in range(10):
                print('nop')
        self.tmpdir = mkdtemp()
        self.fn = os.path.join(self.tmpdir, 'testfile.mtx')

    def teardown_method(self):
        if False:
            print('Hello World!')
        shutil.rmtree(self.tmpdir)

    def check_read(self, example, a, info):
        if False:
            print('Hello World!')
        f = open(self.fn, 'w')
        f.write(example)
        f.close()
        assert_equal(mminfo(self.fn), info)
        b = mmread(self.fn).toarray()
        assert_array_almost_equal(a, b)

    def test_read_general(self):
        if False:
            i = 10
            return i + 15
        a = [[1, 0, 0, 6, 0], [0, 10.5, 0, 0, 0], [0, 0, 0.015, 0, 0], [0, 250.5, 0, -280, 33.32], [0, 0, 0, 0, 12]]
        self.check_read(_general_example, a, (5, 5, 8, 'coordinate', 'real', 'general'))

    def test_read_hermitian(self):
        if False:
            for i in range(10):
                print('nop')
        a = [[1, 0, 0, 0, 0], [0, 10.5, 0, 250.5 - 22.22j, 0], [0, 0, 0.015, 0, 0], [0, 250.5 + 22.22j, 0, -280, -33.32j], [0, 0, 0, 33.32j, 12]]
        self.check_read(_hermitian_example, a, (5, 5, 7, 'coordinate', 'complex', 'hermitian'))

    def test_read_skew(self):
        if False:
            while True:
                i = 10
        a = [[1, 0, 0, 0, 0], [0, 10.5, 0, -250.5, 0], [0, 0, 0.015, 0, 0], [0, 250.5, 0, -280, 0], [0, 0, 0, 0, 12]]
        self.check_read(_skew_example, a, (5, 5, 7, 'coordinate', 'real', 'skew-symmetric'))

    def test_read_symmetric(self):
        if False:
            print('Hello World!')
        a = [[1, 0, 0, 0, 0], [0, 10.5, 0, 250.5, 0], [0, 0, 0.015, 0, 0], [0, 250.5, 0, -280, 8], [0, 0, 0, 8, 12]]
        self.check_read(_symmetric_example, a, (5, 5, 7, 'coordinate', 'real', 'symmetric'))

    def test_read_symmetric_pattern(self):
        if False:
            return 10
        a = [[1, 0, 0, 0, 0], [0, 1, 0, 1, 0], [0, 0, 1, 0, 0], [0, 1, 0, 1, 1], [0, 0, 0, 1, 1]]
        self.check_read(_symmetric_pattern_example, a, (5, 5, 7, 'coordinate', 'pattern', 'symmetric'))

    def test_read_empty_lines(self):
        if False:
            print('Hello World!')
        a = [[1, 0, 0, 6, 0], [0, 10.5, 0, 0, 0], [0, 0, 0.015, 0, 0], [0, 250.5, 0, -280, 33.32], [0, 0, 0, 0, 12]]
        self.check_read(_empty_lines_example, a, (5, 5, 8, 'coordinate', 'real', 'general'))

    def test_empty_write_read(self):
        if False:
            return 10
        b = scipy.sparse.coo_matrix((10, 10))
        mmwrite(self.fn, b)
        assert_equal(mminfo(self.fn), (10, 10, 0, 'coordinate', 'real', 'symmetric'))
        a = b.toarray()
        b = mmread(self.fn).toarray()
        assert_array_almost_equal(a, b)

    def test_bzip2_py3(self):
        if False:
            i = 10
            return i + 15
        try:
            import bz2
        except ImportError:
            return
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
        mmwrite(self.fn, b)
        fn_bzip2 = '%s.bz2' % self.fn
        with open(self.fn, 'rb') as f_in:
            f_out = bz2.BZ2File(fn_bzip2, 'wb')
            f_out.write(f_in.read())
            f_out.close()
        a = mmread(fn_bzip2).toarray()
        assert_array_almost_equal(a, b.toarray())

    def test_gzip_py3(self):
        if False:
            print('Hello World!')
        try:
            import gzip
        except ImportError:
            return
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
        mmwrite(self.fn, b)
        fn_gzip = '%s.gz' % self.fn
        with open(self.fn, 'rb') as f_in:
            f_out = gzip.open(fn_gzip, 'wb')
            f_out.write(f_in.read())
            f_out.close()
        a = mmread(fn_gzip).toarray()
        assert_array_almost_equal(a, b.toarray())

    def test_real_write_read(self):
        if False:
            return 10
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
        mmwrite(self.fn, b)
        assert_equal(mminfo(self.fn), (5, 5, 8, 'coordinate', 'real', 'general'))
        a = b.toarray()
        b = mmread(self.fn).toarray()
        assert_array_almost_equal(a, b)

    def test_complex_write_read(self):
        if False:
            for i in range(10):
                print('nop')
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0 + 3j, 6.0 + 2j, 10.5 + 0.9j, 0.015 + -4.4j, 250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.0 + 0.8j])
        b = scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5))
        mmwrite(self.fn, b)
        assert_equal(mminfo(self.fn), (5, 5, 8, 'coordinate', 'complex', 'general'))
        a = b.toarray()
        b = mmread(self.fn).toarray()
        assert_array_almost_equal(a, b)

    def test_sparse_formats(self, tmp_path):
        if False:
            i = 10
            return i + 15
        tmpdir = tmp_path / 'sparse_formats'
        tmpdir.mkdir()
        mats = []
        I = array([0, 0, 1, 2, 3, 3, 3, 4])
        J = array([0, 3, 1, 2, 1, 3, 4, 4])
        V = array([1.0, 6.0, 10.5, 0.015, 250.5, -280.0, 33.32, 12.0])
        mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))
        V = array([1.0 + 3j, 6.0 + 2j, 10.5 + 0.9j, 0.015 + -4.4j, 250.5 + 0j, -280.0 + 5j, 33.32 + 6.4j, 12.0 + 0.8j])
        mats.append(scipy.sparse.coo_matrix((V, (I, J)), shape=(5, 5)))
        for mat in mats:
            expected = mat.toarray()
            for fmt in ['csr', 'csc', 'coo']:
                fname = tmpdir / (fmt + '.mtx')
                mmwrite(fname, mat.asformat(fmt))
                result = mmread(fname).toarray()
                assert_array_almost_equal(result, expected)

    def test_precision(self):
        if False:
            i = 10
            return i + 15
        test_values = [pi] + [10 ** i for i in range(0, -10, -1)]
        test_precisions = range(1, 10)
        for value in test_values:
            for precision in test_precisions:
                n = 10 ** precision + 1
                A = scipy.sparse.dok_matrix((n, n))
                A[n - 1, n - 1] = value
                mmwrite(self.fn, A, precision=precision)
                A = scipy.io.mmread(self.fn)
                assert_array_equal(A.row, [n - 1])
                assert_array_equal(A.col, [n - 1])
                assert_allclose(A.data, [float('%%.%dg' % precision % value)])

    def test_bad_number_of_coordinate_header_fields(self):
        if False:
            print('Hello World!')
        s = '            %%MatrixMarket matrix coordinate real general\n              5  5  8 999\n                1     1   1.000e+00\n                2     2   1.050e+01\n                3     3   1.500e-02\n                1     4   6.000e+00\n                4     2   2.505e+02\n                4     4  -2.800e+02\n                4     5   3.332e+01\n                5     5   1.200e+01\n            '
        text = textwrap.dedent(s).encode('ascii')
        with pytest.raises(ValueError, match='not of length 3'):
            scipy.io.mmread(io.BytesIO(text))

def test_gh11389():
    if False:
        while True:
            i = 10
    mmread(io.StringIO('%%MatrixMarket matrix coordinate complex symmetric\n 1 1 1\n1 1 -2.1846000000000e+02  0.0000000000000e+00'))

def test_gh18123(tmp_path):
    if False:
        return 10
    lines = [' %%MatrixMarket matrix coordinate real general\n', '5 5 3\n', '2 3 1.0\n', '3 4 2.0\n', '3 5 3.0\n']
    test_file = tmp_path / 'test.mtx'
    with open(test_file, 'w') as f:
        f.writelines(lines)
    mmread(test_file)

def test_threadpoolctl():
    if False:
        while True:
            i = 10
    try:
        import threadpoolctl
        if not hasattr(threadpoolctl, 'register'):
            pytest.skip('threadpoolctl too old')
            return
    except ImportError:
        pytest.skip('no threadpoolctl')
        return
    with threadpoolctl.threadpool_limits(limits=4):
        assert_equal(fmm.PARALLELISM, 4)
    with threadpoolctl.threadpool_limits(limits=2, user_api='scipy'):
        assert_equal(fmm.PARALLELISM, 2)