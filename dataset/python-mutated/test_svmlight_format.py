import gzip
import os
import shutil
from bz2 import BZ2File
from io import BytesIO
from tempfile import NamedTemporaryFile
import numpy as np
import pytest
import scipy.sparse as sp
import sklearn
from sklearn.datasets import dump_svmlight_file, load_svmlight_file, load_svmlight_files
from sklearn.utils._testing import assert_allclose, assert_array_almost_equal, assert_array_equal, fails_if_pypy
from sklearn.utils.fixes import CSR_CONTAINERS, _open_binary, _path
TEST_DATA_MODULE = 'sklearn.datasets.tests.data'
datafile = 'svmlight_classification.txt'
multifile = 'svmlight_multilabel.txt'
invalidfile = 'svmlight_invalid.txt'
invalidfile2 = 'svmlight_invalid_order.txt'
pytestmark = fails_if_pypy

def _load_svmlight_local_test_file(filename, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Helper to load resource `filename` with `importlib.resources`\n    '
    with _open_binary(TEST_DATA_MODULE, filename) as f:
        return load_svmlight_file(f, **kwargs)

def test_load_svmlight_file():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = _load_svmlight_local_test_file(datafile)
    assert X.indptr.shape[0] == 7
    assert X.shape[0] == 6
    assert X.shape[1] == 21
    assert y.shape[0] == 6
    for (i, j, val) in ((0, 2, 2.5), (0, 10, -5.2), (0, 15, 1.5), (1, 5, 1.0), (1, 12, -3), (2, 20, 27)):
        assert X[i, j] == val
    assert X[0, 3] == 0
    assert X[0, 5] == 0
    assert X[1, 8] == 0
    assert X[1, 16] == 0
    assert X[2, 18] == 0
    X[0, 2] *= 2
    assert X[0, 2] == 5
    assert_array_equal(y, [1, 2, 3, 4, 1, 2])

def test_load_svmlight_file_fd():
    if False:
        i = 10
        return i + 15
    with _path(TEST_DATA_MODULE, datafile) as data_path:
        data_path = str(data_path)
        (X1, y1) = load_svmlight_file(data_path)
        fd = os.open(data_path, os.O_RDONLY)
        try:
            (X2, y2) = load_svmlight_file(fd)
            assert_array_almost_equal(X1.data, X2.data)
            assert_array_almost_equal(y1, y2)
        finally:
            os.close(fd)

def test_load_svmlight_pathlib():
    if False:
        for i in range(10):
            print('nop')
    with _path(TEST_DATA_MODULE, datafile) as data_path:
        (X1, y1) = load_svmlight_file(str(data_path))
        (X2, y2) = load_svmlight_file(data_path)
    assert_allclose(X1.data, X2.data)
    assert_allclose(y1, y2)

def test_load_svmlight_file_multilabel():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = _load_svmlight_local_test_file(multifile, multilabel=True)
    assert y == [(0, 1), (2,), (), (1, 2)]

def test_load_svmlight_files():
    if False:
        i = 10
        return i + 15
    with _path(TEST_DATA_MODULE, datafile) as data_path:
        (X_train, y_train, X_test, y_test) = load_svmlight_files([str(data_path)] * 2, dtype=np.float32)
    assert_array_equal(X_train.toarray(), X_test.toarray())
    assert_array_almost_equal(y_train, y_test)
    assert X_train.dtype == np.float32
    assert X_test.dtype == np.float32
    with _path(TEST_DATA_MODULE, datafile) as data_path:
        (X1, y1, X2, y2, X3, y3) = load_svmlight_files([str(data_path)] * 3, dtype=np.float64)
    assert X1.dtype == X2.dtype
    assert X2.dtype == X3.dtype
    assert X3.dtype == np.float64

def test_load_svmlight_file_n_features():
    if False:
        i = 10
        return i + 15
    (X, y) = _load_svmlight_local_test_file(datafile, n_features=22)
    assert X.indptr.shape[0] == 7
    assert X.shape[0] == 6
    assert X.shape[1] == 22
    for (i, j, val) in ((0, 2, 2.5), (0, 10, -5.2), (1, 5, 1.0), (1, 12, -3)):
        assert X[i, j] == val
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(datafile, n_features=20)

def test_load_compressed():
    if False:
        return 10
    (X, y) = _load_svmlight_local_test_file(datafile)
    with NamedTemporaryFile(prefix='sklearn-test', suffix='.gz') as tmp:
        tmp.close()
        with _open_binary(TEST_DATA_MODULE, datafile) as f:
            with gzip.open(tmp.name, 'wb') as fh_out:
                shutil.copyfileobj(f, fh_out)
        (Xgz, ygz) = load_svmlight_file(tmp.name)
        os.remove(tmp.name)
    assert_array_almost_equal(X.toarray(), Xgz.toarray())
    assert_array_almost_equal(y, ygz)
    with NamedTemporaryFile(prefix='sklearn-test', suffix='.bz2') as tmp:
        tmp.close()
        with _open_binary(TEST_DATA_MODULE, datafile) as f:
            with BZ2File(tmp.name, 'wb') as fh_out:
                shutil.copyfileobj(f, fh_out)
        (Xbz, ybz) = load_svmlight_file(tmp.name)
        os.remove(tmp.name)
    assert_array_almost_equal(X.toarray(), Xbz.toarray())
    assert_array_almost_equal(y, ybz)

def test_load_invalid_file():
    if False:
        print('Hello World!')
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(invalidfile)

def test_load_invalid_order_file():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError):
        _load_svmlight_local_test_file(invalidfile2)

def test_load_zero_based():
    if False:
        print('Hello World!')
    f = BytesIO(b'-1 4:1.\n1 0:1\n')
    with pytest.raises(ValueError):
        load_svmlight_file(f, zero_based=False)

def test_load_zero_based_auto():
    if False:
        for i in range(10):
            print('nop')
    data1 = b'-1 1:1 2:2 3:3\n'
    data2 = b'-1 0:0 1:1\n'
    f1 = BytesIO(data1)
    (X, y) = load_svmlight_file(f1, zero_based='auto')
    assert X.shape == (1, 3)
    f1 = BytesIO(data1)
    f2 = BytesIO(data2)
    (X1, y1, X2, y2) = load_svmlight_files([f1, f2], zero_based='auto')
    assert X1.shape == (1, 4)
    assert X2.shape == (1, 4)

def test_load_with_qid():
    if False:
        i = 10
        return i + 15
    data = b'\n    3 qid:1 1:0.53 2:0.12\n    2 qid:1 1:0.13 2:0.1\n    7 qid:2 1:0.87 2:0.12'
    (X, y) = load_svmlight_file(BytesIO(data), query_id=False)
    assert_array_equal(y, [3, 2, 7])
    assert_array_equal(X.toarray(), [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]])
    res1 = load_svmlight_files([BytesIO(data)], query_id=True)
    res2 = load_svmlight_file(BytesIO(data), query_id=True)
    for (X, y, qid) in (res1, res2):
        assert_array_equal(y, [3, 2, 7])
        assert_array_equal(qid, [1, 1, 2])
        assert_array_equal(X.toarray(), [[0.53, 0.12], [0.13, 0.1], [0.87, 0.12]])

@pytest.mark.skip('testing the overflow of 32 bit sparse indexing requires a large amount of memory')
def test_load_large_qid():
    if False:
        for i in range(10):
            print('nop')
    '\n    load large libsvm / svmlight file with qid attribute. Tests 64-bit query ID\n    '
    data = b'\n'.join(('3 qid:{0} 1:0.53 2:0.12\n2 qid:{0} 1:0.13 2:0.1'.format(i).encode() for i in range(1, 40 * 1000 * 1000)))
    (X, y, qid) = load_svmlight_file(BytesIO(data), query_id=True)
    assert_array_equal(y[-4:], [3, 2, 3, 2])
    assert_array_equal(np.unique(qid), np.arange(1, 40 * 1000 * 1000))

def test_load_invalid_file2():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ValueError):
        with _path(TEST_DATA_MODULE, datafile) as data_path, _path(TEST_DATA_MODULE, invalidfile) as invalid_path:
            load_svmlight_files([str(data_path), str(invalid_path), str(data_path)])

def test_not_a_filename():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(TypeError):
        load_svmlight_file(0.42)

def test_invalid_filename():
    if False:
        i = 10
        return i + 15
    with pytest.raises(OSError):
        load_svmlight_file('trou pic nic douille')

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dump(csr_container):
    if False:
        return 10
    (X_sparse, y_dense) = _load_svmlight_local_test_file(datafile)
    X_dense = X_sparse.toarray()
    y_sparse = csr_container(y_dense)
    X_sliced = X_sparse[np.arange(X_sparse.shape[0])]
    y_sliced = y_sparse[np.arange(y_sparse.shape[0])]
    for X in (X_sparse, X_dense, X_sliced):
        for y in (y_sparse, y_dense, y_sliced):
            for zero_based in (True, False):
                for dtype in [np.float32, np.float64, np.int32, np.int64]:
                    f = BytesIO()
                    if sp.issparse(y) and y.shape[0] == 1:
                        y = y.T
                    X_input = X.astype(dtype)
                    dump_svmlight_file(X_input, y, f, comment='test', zero_based=zero_based)
                    f.seek(0)
                    comment = f.readline()
                    comment = str(comment, 'utf-8')
                    assert 'scikit-learn %s' % sklearn.__version__ in comment
                    comment = f.readline()
                    comment = str(comment, 'utf-8')
                    assert ['one', 'zero'][zero_based] + '-based' in comment
                    (X2, y2) = load_svmlight_file(f, dtype=dtype, zero_based=zero_based)
                    assert X2.dtype == dtype
                    assert_array_equal(X2.sorted_indices().indices, X2.indices)
                    X2_dense = X2.toarray()
                    if sp.issparse(X_input):
                        X_input_dense = X_input.toarray()
                    else:
                        X_input_dense = X_input
                    if dtype == np.float32:
                        assert_array_almost_equal(X_input_dense, X2_dense, 4)
                        assert_array_almost_equal(y_dense.astype(dtype, copy=False), y2, 4)
                    else:
                        assert_array_almost_equal(X_input_dense, X2_dense, 15)
                        assert_array_almost_equal(y_dense.astype(dtype, copy=False), y2, 15)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_dump_multilabel(csr_container):
    if False:
        i = 10
        return i + 15
    X = [[1, 0, 3, 0, 5], [0, 0, 0, 0, 0], [0, 5, 0, 1, 0]]
    y_dense = [[0, 1, 0], [1, 0, 1], [1, 1, 0]]
    y_sparse = csr_container(y_dense)
    for y in [y_dense, y_sparse]:
        f = BytesIO()
        dump_svmlight_file(X, y, f, multilabel=True)
        f.seek(0)
        assert f.readline() == b'1 0:1 2:3 4:5\n'
        assert f.readline() == b'0,2 \n'
        assert f.readline() == b'0,1 1:5 3:1\n'

def test_dump_concise():
    if False:
        return 10
    one = 1
    two = 2.1
    three = 3.01
    exact = 1.000000000000001
    almost = 1.0
    X = [[one, two, three, exact, almost], [1000000000.0, 2e+18, 3e+27, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]]
    y = [one, two, three, exact, almost]
    f = BytesIO()
    dump_svmlight_file(X, y, f)
    f.seek(0)
    assert f.readline() == b'1 0:1 1:2.1 2:3.01 3:1.000000000000001 4:1\n'
    assert f.readline() == b'2.1 0:1000000000 1:2e+18 2:3e+27\n'
    assert f.readline() == b'3.01 \n'
    assert f.readline() == b'1.000000000000001 \n'
    assert f.readline() == b'1 \n'
    f.seek(0)
    (X2, y2) = load_svmlight_file(f)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)

def test_dump_comment():
    if False:
        return 10
    (X, y) = _load_svmlight_local_test_file(datafile)
    X = X.toarray()
    f = BytesIO()
    ascii_comment = 'This is a comment\nspanning multiple lines.'
    dump_svmlight_file(X, y, f, comment=ascii_comment, zero_based=False)
    f.seek(0)
    (X2, y2) = load_svmlight_file(f, zero_based=False)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)
    utf8_comment = b'It is true that\n\xc2\xbd\xc2\xb2 = \xc2\xbc'
    f = BytesIO()
    with pytest.raises(UnicodeDecodeError):
        dump_svmlight_file(X, y, f, comment=utf8_comment)
    unicode_comment = utf8_comment.decode('utf-8')
    f = BytesIO()
    dump_svmlight_file(X, y, f, comment=unicode_comment, zero_based=False)
    f.seek(0)
    (X2, y2) = load_svmlight_file(f, zero_based=False)
    assert_array_almost_equal(X, X2.toarray())
    assert_array_almost_equal(y, y2)
    f = BytesIO()
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y, f, comment="I've got a \x00.")

def test_dump_invalid():
    if False:
        while True:
            i = 10
    (X, y) = _load_svmlight_local_test_file(datafile)
    f = BytesIO()
    y2d = [y]
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y2d, f)
    f = BytesIO()
    with pytest.raises(ValueError):
        dump_svmlight_file(X, y[:-1], f)

def test_dump_query_id():
    if False:
        for i in range(10):
            print('nop')
    (X, y) = _load_svmlight_local_test_file(datafile)
    X = X.toarray()
    query_id = np.arange(X.shape[0]) // 2
    f = BytesIO()
    dump_svmlight_file(X, y, f, query_id=query_id, zero_based=True)
    f.seek(0)
    (X1, y1, query_id1) = load_svmlight_file(f, query_id=True, zero_based=True)
    assert_array_almost_equal(X, X1.toarray())
    assert_array_almost_equal(y, y1)
    assert_array_almost_equal(query_id, query_id1)

def test_load_with_long_qid():
    if False:
        for i in range(10):
            print('nop')
    data = b'\n    1 qid:0 0:1 1:2 2:3\n    0 qid:72048431380967004 0:1440446648 1:72048431380967004 2:236784985\n    0 qid:-9223372036854775807 0:1440446648 1:72048431380967004 2:236784985\n    3 qid:9223372036854775807  0:1440446648 1:72048431380967004 2:236784985'
    (X, y, qid) = load_svmlight_file(BytesIO(data), query_id=True)
    true_X = [[1, 2, 3], [1440446648, 72048431380967004, 236784985], [1440446648, 72048431380967004, 236784985], [1440446648, 72048431380967004, 236784985]]
    true_y = [1, 0, 0, 3]
    trueQID = [0, 72048431380967004, -9223372036854775807, 9223372036854775807]
    assert_array_equal(y, true_y)
    assert_array_equal(X.toarray(), true_X)
    assert_array_equal(qid, trueQID)
    f = BytesIO()
    dump_svmlight_file(X, y, f, query_id=qid, zero_based=True)
    f.seek(0)
    (X, y, qid) = load_svmlight_file(f, query_id=True, zero_based=True)
    assert_array_equal(y, true_y)
    assert_array_equal(X.toarray(), true_X)
    assert_array_equal(qid, trueQID)
    f.seek(0)
    (X, y) = load_svmlight_file(f, query_id=False, zero_based=True)
    assert_array_equal(y, true_y)
    assert_array_equal(X.toarray(), true_X)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_load_zeros(csr_container):
    if False:
        while True:
            i = 10
    f = BytesIO()
    true_X = csr_container(np.zeros(shape=(3, 4)))
    true_y = np.array([0, 1, 0])
    dump_svmlight_file(true_X, true_y, f)
    for zero_based in ['auto', True, False]:
        f.seek(0)
        (X, y) = load_svmlight_file(f, n_features=4, zero_based=zero_based)
        assert_array_almost_equal(y, true_y)
        assert_array_almost_equal(X.toarray(), true_X.toarray())

@pytest.mark.parametrize('sparsity', [0, 0.1, 0.5, 0.99, 1])
@pytest.mark.parametrize('n_samples', [13, 101])
@pytest.mark.parametrize('n_features', [2, 7, 41])
@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_load_with_offsets(sparsity, n_samples, n_features, csr_container):
    if False:
        print('Hello World!')
    rng = np.random.RandomState(0)
    X = rng.uniform(low=0.0, high=1.0, size=(n_samples, n_features))
    if sparsity:
        X[X < sparsity] = 0.0
    X = csr_container(X)
    y = rng.randint(low=0, high=2, size=n_samples)
    f = BytesIO()
    dump_svmlight_file(X, y, f)
    f.seek(0)
    size = len(f.getvalue())
    mark_0 = 0
    mark_1 = size // 3
    length_0 = mark_1 - mark_0
    mark_2 = 4 * size // 5
    length_1 = mark_2 - mark_1
    (X_0, y_0) = load_svmlight_file(f, n_features=n_features, offset=mark_0, length=length_0)
    (X_1, y_1) = load_svmlight_file(f, n_features=n_features, offset=mark_1, length=length_1)
    (X_2, y_2) = load_svmlight_file(f, n_features=n_features, offset=mark_2)
    y_concat = np.concatenate([y_0, y_1, y_2])
    X_concat = sp.vstack([X_0, X_1, X_2])
    assert_array_almost_equal(y, y_concat)
    assert_array_almost_equal(X.toarray(), X_concat.toarray())

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_load_offset_exhaustive_splits(csr_container):
    if False:
        while True:
            i = 10
    rng = np.random.RandomState(0)
    X = np.array([[0, 0, 0, 0, 0, 0], [1, 2, 3, 4, 0, 6], [1, 2, 3, 4, 0, 6], [0, 0, 0, 0, 0, 0], [1, 0, 3, 0, 0, 0], [0, 0, 0, 0, 0, 1], [1, 0, 0, 0, 0, 0]])
    X = csr_container(X)
    (n_samples, n_features) = X.shape
    y = rng.randint(low=0, high=2, size=n_samples)
    query_id = np.arange(n_samples) // 2
    f = BytesIO()
    dump_svmlight_file(X, y, f, query_id=query_id)
    f.seek(0)
    size = len(f.getvalue())
    for mark in range(size):
        f.seek(0)
        (X_0, y_0, q_0) = load_svmlight_file(f, n_features=n_features, query_id=True, offset=0, length=mark)
        (X_1, y_1, q_1) = load_svmlight_file(f, n_features=n_features, query_id=True, offset=mark, length=-1)
        q_concat = np.concatenate([q_0, q_1])
        y_concat = np.concatenate([y_0, y_1])
        X_concat = sp.vstack([X_0, X_1])
        assert_array_almost_equal(y, y_concat)
        assert_array_equal(query_id, q_concat)
        assert_array_almost_equal(X.toarray(), X_concat.toarray())

def test_load_with_offsets_error():
    if False:
        while True:
            i = 10
    with pytest.raises(ValueError, match='n_features is required'):
        _load_svmlight_local_test_file(datafile, offset=3, length=3)

@pytest.mark.parametrize('csr_container', CSR_CONTAINERS)
def test_multilabel_y_explicit_zeros(tmp_path, csr_container):
    if False:
        for i in range(10):
            print('nop')
    '\n    Ensure that if y contains explicit zeros (i.e. elements of y.data equal to\n    0) then those explicit zeros are not encoded.\n    '
    save_path = str(tmp_path / 'svm_explicit_zero')
    rng = np.random.RandomState(42)
    X = rng.randn(3, 5).astype(np.float64)
    indptr = np.array([0, 2, 3, 6])
    indices = np.array([0, 2, 2, 0, 1, 2])
    data = np.array([0, 1, 1, 1, 1, 0])
    y = csr_container((data, indices, indptr), shape=(3, 3))
    dump_svmlight_file(X, y, save_path, multilabel=True)
    (_, y_load) = load_svmlight_file(save_path, multilabel=True)
    y_true = [(2.0,), (2.0,), (0.0, 1.0)]
    assert y_load == y_true