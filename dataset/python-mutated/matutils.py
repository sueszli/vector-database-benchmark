"""Math helper functions."""
from __future__ import with_statement
import logging
import math
from gensim import utils
import numpy as np
import scipy.sparse
from scipy.stats import entropy
from scipy.linalg import get_blas_funcs, triu
from scipy.linalg.lapack import get_lapack_funcs
from scipy.special import psi
logger = logging.getLogger(__name__)

def blas(name, ndarray):
    if False:
        for i in range(10):
            print('nop')
    'Helper for getting the appropriate BLAS function, using :func:`scipy.linalg.get_blas_funcs`.\n\n    Parameters\n    ----------\n    name : str\n        Name(s) of BLAS functions, without the type prefix.\n    ndarray : numpy.ndarray\n        Arrays can be given to determine optimal prefix of BLAS routines.\n\n    Returns\n    -------\n    object\n        BLAS function for the needed operation on the given data type.\n\n    '
    return get_blas_funcs((name,), (ndarray,))[0]

def argsort(x, topn=None, reverse=False):
    if False:
        i = 10
        return i + 15
    'Efficiently calculate indices of the `topn` smallest elements in array `x`.\n\n    Parameters\n    ----------\n    x : array_like\n        Array to get the smallest element indices from.\n    topn : int, optional\n        Number of indices of the smallest (greatest) elements to be returned.\n        If not given, indices of all elements will be returned in ascending (descending) order.\n    reverse : bool, optional\n        Return the `topn` greatest elements in descending order,\n        instead of smallest elements in ascending order?\n\n    Returns\n    -------\n    numpy.ndarray\n        Array of `topn` indices that sort the array in the requested order.\n\n    '
    x = np.asarray(x)
    if topn is None:
        topn = x.size
    if topn <= 0:
        return []
    if reverse:
        x = -x
    if topn >= x.size or not hasattr(np, 'argpartition'):
        return np.argsort(x)[:topn]
    most_extreme = np.argpartition(x, topn)[:topn]
    return most_extreme.take(np.argsort(x.take(most_extreme)))

def corpus2csc(corpus, num_terms=None, dtype=np.float64, num_docs=None, num_nnz=None, printprogress=0):
    if False:
        print('Hello World!')
    'Convert a streamed corpus in bag-of-words format into a sparse matrix `scipy.sparse.csc_matrix`,\n    with documents as columns.\n\n    Notes\n    -----\n    If the number of terms, documents and non-zero elements is known, you can pass\n    them here as parameters and a (much) more memory efficient code path will be taken.\n\n    Parameters\n    ----------\n    corpus : iterable of iterable of (int, number)\n        Input corpus in BoW format\n    num_terms : int, optional\n        Number of terms in `corpus`. If provided, the `corpus.num_terms` attribute (if any) will be ignored.\n    dtype : data-type, optional\n        Data type of output CSC matrix.\n    num_docs : int, optional\n        Number of documents in `corpus`. If provided, the `corpus.num_docs` attribute (in any) will be ignored.\n    num_nnz : int, optional\n        Number of non-zero elements in `corpus`. If provided, the `corpus.num_nnz` attribute (if any) will be ignored.\n    printprogress : int, optional\n        Log a progress message at INFO level once every `printprogress` documents. 0 to turn off progress logging.\n\n    Returns\n    -------\n    scipy.sparse.csc_matrix\n        `corpus` converted into a sparse CSC matrix.\n\n    See Also\n    --------\n    :class:`~gensim.matutils.Sparse2Corpus`\n        Convert sparse format to Gensim corpus format.\n\n    '
    try:
        if num_terms is None:
            num_terms = corpus.num_terms
        if num_docs is None:
            num_docs = corpus.num_docs
        if num_nnz is None:
            num_nnz = corpus.num_nnz
    except AttributeError:
        pass
    if printprogress:
        logger.info('creating sparse matrix from corpus')
    if num_terms is not None and num_docs is not None and (num_nnz is not None):
        (posnow, indptr) = (0, [0])
        indices = np.empty((num_nnz,), dtype=np.int32)
        data = np.empty((num_nnz,), dtype=dtype)
        for (docno, doc) in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info('PROGRESS: at document #%i/%i', docno, num_docs)
            posnext = posnow + len(doc)
            (indices[posnow:posnext], data[posnow:posnext]) = zip(*doc) if doc else ([], [])
            indptr.append(posnext)
            posnow = posnext
        assert posnow == num_nnz, 'mismatch between supplied and computed number of non-zeros'
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    else:
        (num_nnz, data, indices, indptr) = (0, [], [], [0])
        for (docno, doc) in enumerate(corpus):
            if printprogress and docno % printprogress == 0:
                logger.info('PROGRESS: at document #%i', docno)
            (doc_indices, doc_data) = zip(*doc) if doc else ([], [])
            indices.extend(doc_indices)
            data.extend(doc_data)
            num_nnz += len(doc)
            indptr.append(num_nnz)
        if num_terms is None:
            num_terms = max(indices) + 1 if indices else 0
        num_docs = len(indptr) - 1
        data = np.asarray(data, dtype=dtype)
        indices = np.asarray(indices)
        result = scipy.sparse.csc_matrix((data, indices, indptr), shape=(num_terms, num_docs), dtype=dtype)
    return result

def pad(mat, padrow, padcol):
    if False:
        while True:
            i = 10
    'Add additional rows/columns to `mat`. The new rows/columns will be initialized with zeros.\n\n    Parameters\n    ----------\n    mat : numpy.ndarray\n        Input 2D matrix\n    padrow : int\n        Number of additional rows\n    padcol : int\n        Number of additional columns\n\n    Returns\n    -------\n    numpy.matrixlib.defmatrix.matrix\n        Matrix with needed padding.\n\n    '
    if padrow < 0:
        padrow = 0
    if padcol < 0:
        padcol = 0
    (rows, cols) = mat.shape
    return np.block([[mat, np.zeros((rows, padcol))], [np.zeros((padrow, cols + padcol))]])

def zeros_aligned(shape, dtype, order='C', align=128):
    if False:
        while True:
            i = 10
    "Get array aligned at `align` byte boundary in memory.\n\n    Parameters\n    ----------\n    shape : int or (int, int)\n        Shape of array.\n    dtype : data-type\n        Data type of array.\n    order : {'C', 'F'}, optional\n        Whether to store multidimensional data in C- or Fortran-contiguous (row- or column-wise) order in memory.\n    align : int, optional\n        Boundary for alignment in bytes.\n\n    Returns\n    -------\n    numpy.ndarray\n        Aligned array.\n\n    "
    nbytes = np.prod(shape, dtype=np.int64) * np.dtype(dtype).itemsize
    buffer = np.zeros(nbytes + align, dtype=np.uint8)
    start_index = -buffer.ctypes.data % align
    return buffer[start_index:start_index + nbytes].view(dtype).reshape(shape, order=order)

def ismatrix(m):
    if False:
        print('Hello World!')
    'Check whether `m` is a 2D `numpy.ndarray` or `scipy.sparse` matrix.\n\n    Parameters\n    ----------\n    m : object\n        Object to check.\n\n    Returns\n    -------\n    bool\n        Is `m` a 2D `numpy.ndarray` or `scipy.sparse` matrix.\n\n    '
    return isinstance(m, np.ndarray) and m.ndim == 2 or scipy.sparse.issparse(m)

def any2sparse(vec, eps=1e-09):
    if False:
        i = 10
        return i + 15
    'Convert a numpy.ndarray or `scipy.sparse` vector into the Gensim bag-of-words format.\n\n    Parameters\n    ----------\n    vec : {`numpy.ndarray`, `scipy.sparse`}\n        Input vector\n    eps : float, optional\n        Value used for threshold, all coordinates less than `eps` will not be presented in result.\n\n    Returns\n    -------\n    list of (int, float)\n        Vector in BoW format.\n\n    '
    if isinstance(vec, np.ndarray):
        return dense2vec(vec, eps)
    if scipy.sparse.issparse(vec):
        return scipy2sparse(vec, eps)
    return [(int(fid), float(fw)) for (fid, fw) in vec if np.abs(fw) > eps]

def scipy2scipy_clipped(matrix, topn, eps=1e-09):
    if False:
        while True:
            i = 10
    "Get the 'topn' elements of the greatest magnitude (absolute value) from a `scipy.sparse` vector or matrix.\n\n    Parameters\n    ----------\n    matrix : `scipy.sparse`\n        Input vector or matrix (1D or 2D sparse array).\n    topn : int\n        Number of greatest elements, in absolute value, to return.\n    eps : float\n        Ignored.\n\n    Returns\n    -------\n    `scipy.sparse.csr.csr_matrix`\n        Clipped matrix.\n\n    "
    if not scipy.sparse.issparse(matrix):
        raise ValueError("'%s' is not a scipy sparse vector." % matrix)
    if topn <= 0:
        return scipy.sparse.csr_matrix([])
    if matrix.shape[0] == 1:
        biggest = argsort(abs(matrix.data), topn, reverse=True)
        (indices, data) = (matrix.indices.take(biggest), matrix.data.take(biggest))
        return scipy.sparse.csr_matrix((data, indices, [0, len(indices)]))
    else:
        matrix_indices = []
        matrix_data = []
        matrix_indptr = [0]
        matrix_abs = abs(matrix)
        for i in range(matrix.shape[0]):
            v = matrix.getrow(i)
            v_abs = matrix_abs.getrow(i)
            biggest = argsort(v_abs.data, topn, reverse=True)
            (indices, data) = (v.indices.take(biggest), v.data.take(biggest))
            matrix_data.append(data)
            matrix_indices.append(indices)
            matrix_indptr.append(matrix_indptr[-1] + min(len(indices), topn))
        matrix_indices = np.concatenate(matrix_indices).ravel()
        matrix_data = np.concatenate(matrix_data).ravel()
        return scipy.sparse.csr.csr_matrix((matrix_data, matrix_indices, matrix_indptr), shape=(matrix.shape[0], np.max(matrix_indices) + 1))

def scipy2sparse(vec, eps=1e-09):
    if False:
        for i in range(10):
            print('nop')
    'Convert a scipy.sparse vector into the Gensim bag-of-words format.\n\n    Parameters\n    ----------\n    vec : `scipy.sparse`\n        Sparse vector.\n\n    eps : float, optional\n        Value used for threshold, all coordinates less than `eps` will not be presented in result.\n\n    Returns\n    -------\n    list of (int, float)\n        Vector in Gensim bag-of-words format.\n\n    '
    vec = vec.tocsr()
    assert vec.shape[0] == 1
    return [(int(pos), float(val)) for (pos, val) in zip(vec.indices, vec.data) if np.abs(val) > eps]

class Scipy2Corpus:
    """Convert a sequence of dense/sparse vectors into a streamed Gensim corpus object.

    See Also
    --------
    :func:`~gensim.matutils.corpus2csc`
        Convert corpus in Gensim format to `scipy.sparse.csc` matrix.

    """

    def __init__(self, vecs):
        if False:
            print('Hello World!')
        '\n\n        Parameters\n        ----------\n        vecs : iterable of {`numpy.ndarray`, `scipy.sparse`}\n            Input vectors.\n\n        '
        self.vecs = vecs

    def __iter__(self):
        if False:
            return 10
        for vec in self.vecs:
            if isinstance(vec, np.ndarray):
                yield full2sparse(vec)
            else:
                yield scipy2sparse(vec)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.vecs)

def sparse2full(doc, length):
    if False:
        return 10
    'Convert a document in Gensim bag-of-words format into a dense numpy array.\n\n    Parameters\n    ----------\n    doc : list of (int, number)\n        Document in BoW format.\n    length : int\n        Vector dimensionality. This cannot be inferred from the BoW, and you must supply it explicitly.\n        This is typically the vocabulary size or number of topics, depending on how you created `doc`.\n\n    Returns\n    -------\n    numpy.ndarray\n        Dense numpy vector for `doc`.\n\n    See Also\n    --------\n    :func:`~gensim.matutils.full2sparse`\n        Convert dense array to gensim bag-of-words format.\n\n    '
    result = np.zeros(length, dtype=np.float32)
    doc = ((int(id_), float(val_)) for (id_, val_) in doc)
    doc = dict(doc)
    result[list(doc)] = list(doc.values())
    return result

def full2sparse(vec, eps=1e-09):
    if False:
        while True:
            i = 10
    "Convert a dense numpy array into the Gensim bag-of-words format.\n\n    Parameters\n    ----------\n    vec : numpy.ndarray\n        Dense input vector.\n    eps : float\n        Feature weight threshold value. Features with `abs(weight) < eps` are considered sparse and\n        won't be included in the BOW result.\n\n    Returns\n    -------\n    list of (int, float)\n        BoW format of `vec`, with near-zero values omitted (sparse vector).\n\n    See Also\n    --------\n    :func:`~gensim.matutils.sparse2full`\n        Convert a document in Gensim bag-of-words format into a dense numpy array.\n\n    "
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    return list(zip(nnz, vec.take(nnz)))
dense2vec = full2sparse

def full2sparse_clipped(vec, topn, eps=1e-09):
    if False:
        print('Hello World!')
    'Like :func:`~gensim.matutils.full2sparse`, but only return the `topn` elements of the greatest magnitude (abs).\n\n    This is more efficient that sorting a vector and then taking the greatest values, especially\n    where `len(vec) >> topn`.\n\n    Parameters\n    ----------\n    vec : numpy.ndarray\n        Input dense vector\n    topn : int\n        Number of greatest (abs) elements that will be presented in result.\n    eps : float\n        Threshold value, if coordinate in `vec` < eps, this will not be presented in result.\n\n    Returns\n    -------\n    list of (int, float)\n        Clipped vector in BoW format.\n\n    See Also\n    --------\n    :func:`~gensim.matutils.full2sparse`\n        Convert dense array to gensim bag-of-words format.\n\n    '
    if topn <= 0:
        return []
    vec = np.asarray(vec, dtype=float)
    nnz = np.nonzero(abs(vec) > eps)[0]
    biggest = nnz.take(argsort(abs(vec).take(nnz), topn, reverse=True))
    return list(zip(biggest, vec.take(biggest)))

def corpus2dense(corpus, num_terms, num_docs=None, dtype=np.float32):
    if False:
        print('Hello World!')
    'Convert corpus into a dense numpy 2D array, with documents as columns.\n\n    Parameters\n    ----------\n    corpus : iterable of iterable of (int, number)\n        Input corpus in the Gensim bag-of-words format.\n    num_terms : int\n        Number of terms in the dictionary. X-axis of the resulting matrix.\n    num_docs : int, optional\n        Number of documents in the corpus. If provided, a slightly more memory-efficient code path is taken.\n        Y-axis of the resulting matrix.\n    dtype : data-type, optional\n        Data type of the output matrix.\n\n    Returns\n    -------\n    numpy.ndarray\n        Dense 2D array that presents `corpus`.\n\n    See Also\n    --------\n    :class:`~gensim.matutils.Dense2Corpus`\n        Convert dense matrix to Gensim corpus format.\n\n    '
    if num_docs is not None:
        (docno, result) = (-1, np.empty((num_terms, num_docs), dtype=dtype))
        for (docno, doc) in enumerate(corpus):
            result[:, docno] = sparse2full(doc, num_terms)
        assert docno + 1 == num_docs
    else:
        result = np.column_stack([sparse2full(doc, num_terms) for doc in corpus])
    return result.astype(dtype)

class Dense2Corpus:
    """Treat dense numpy array as a streamed Gensim corpus in the bag-of-words format.

    Notes
    -----
    No data copy is made (changes to the underlying matrix imply changes in the streamed corpus).

    See Also
    --------
    :func:`~gensim.matutils.corpus2dense`
        Convert Gensim corpus to dense matrix.
    :class:`~gensim.matutils.Sparse2Corpus`
        Convert sparse matrix to Gensim corpus format.

    """

    def __init__(self, dense, documents_columns=True):
        if False:
            return 10
        '\n\n        Parameters\n        ----------\n        dense : numpy.ndarray\n            Corpus in dense format.\n        documents_columns : bool, optional\n            Documents in `dense` represented as columns, as opposed to rows?\n\n        '
        if documents_columns:
            self.dense = dense.T
        else:
            self.dense = dense

    def __iter__(self):
        if False:
            print('Hello World!')
        'Iterate over the corpus.\n\n        Yields\n        ------\n        list of (int, float)\n            Document in BoW format.\n\n        '
        for doc in self.dense:
            yield full2sparse(doc.flat)

    def __len__(self):
        if False:
            print('Hello World!')
        return len(self.dense)

class Sparse2Corpus:
    """Convert a matrix in scipy.sparse format into a streaming Gensim corpus.

    See Also
    --------
    :func:`~gensim.matutils.corpus2csc`
        Convert gensim corpus format to `scipy.sparse.csc` matrix
    :class:`~gensim.matutils.Dense2Corpus`
        Convert dense matrix to gensim corpus.

    """

    def __init__(self, sparse, documents_columns=True):
        if False:
            print('Hello World!')
        '\n\n        Parameters\n        ----------\n        sparse : `scipy.sparse`\n            Corpus scipy sparse format\n        documents_columns : bool, optional\n            Documents will be column?\n\n        '
        if documents_columns:
            self.sparse = sparse.tocsc()
        else:
            self.sparse = sparse.tocsr().T

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        '\n\n        Yields\n        ------\n        list of (int, float)\n            Document in BoW format.\n\n        '
        for (indprev, indnow) in zip(self.sparse.indptr, self.sparse.indptr[1:]):
            yield list(zip(self.sparse.indices[indprev:indnow], self.sparse.data[indprev:indnow]))

    def __len__(self):
        if False:
            for i in range(10):
                print('nop')
        return self.sparse.shape[1]

    def __getitem__(self, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        Retrieve a document vector or subset from the corpus by key.\n\n        Parameters\n        ----------\n        key: int, ellipsis, slice, iterable object\n            Index of the document retrieve.\n            Less commonly, the key can also be a slice, ellipsis, or an iterable\n            to retrieve multiple documents.\n\n        Returns\n        -------\n        list of (int, number), Sparse2Corpus\n            Document in BoW format when `key` is an integer. Otherwise :class:`~gensim.matutils.Sparse2Corpus`.\n        '
        sparse = self.sparse
        if isinstance(key, int):
            iprev = self.sparse.indptr[key]
            inow = self.sparse.indptr[key + 1]
            return list(zip(sparse.indices[iprev:inow], sparse.data[iprev:inow]))
        sparse = self.sparse.__getitem__((slice(None, None, None), key))
        return Sparse2Corpus(sparse)

def veclen(vec):
    if False:
        print('Hello World!')
    'Calculate L2 (euclidean) length of a vector.\n\n    Parameters\n    ----------\n    vec : list of (int, number)\n        Input vector in sparse bag-of-words format.\n\n    Returns\n    -------\n    float\n        Length of `vec`.\n\n    '
    if len(vec) == 0:
        return 0.0
    length = 1.0 * math.sqrt(sum((val ** 2 for (_, val) in vec)))
    assert length > 0.0, 'sparse documents must not contain any explicit zero entries'
    return length

def ret_normalized_vec(vec, length):
    if False:
        i = 10
        return i + 15
    'Normalize a vector in L2 (Euclidean unit norm).\n\n    Parameters\n    ----------\n    vec : list of (int, number)\n        Input vector in BoW format.\n    length : float\n        Length of vector\n\n    Returns\n    -------\n    list of (int, number)\n        L2-normalized vector in BoW format.\n\n    '
    if length != 1.0:
        return [(termid, val / length) for (termid, val) in vec]
    else:
        return list(vec)

def ret_log_normalize_vec(vec, axis=1):
    if False:
        print('Hello World!')
    log_max = 100.0
    if len(vec.shape) == 1:
        max_val = np.max(vec)
        log_shift = log_max - np.log(len(vec) + 1.0) - max_val
        tot = np.sum(np.exp(vec + log_shift))
        log_norm = np.log(tot) - log_shift
        vec -= log_norm
    elif axis == 1:
        max_val = np.max(vec, 1)
        log_shift = log_max - np.log(vec.shape[1] + 1.0) - max_val
        tot = np.sum(np.exp(vec + log_shift[:, np.newaxis]), 1)
        log_norm = np.log(tot) - log_shift
        vec = vec - log_norm[:, np.newaxis]
    elif axis == 0:
        k = ret_log_normalize_vec(vec.T)
        return (k[0].T, k[1])
    else:
        raise ValueError("'%s' is not a supported axis" % axis)
    return (vec, log_norm)
blas_nrm2 = blas('nrm2', np.array([], dtype=float))
blas_scal = blas('scal', np.array([], dtype=float))

def unitvec(vec, norm='l2', return_norm=False):
    if False:
        print('Hello World!')
    "Scale a vector to unit length.\n\n    Parameters\n    ----------\n    vec : {numpy.ndarray, scipy.sparse, list of (int, float)}\n        Input vector in any format\n    norm : {'l1', 'l2', 'unique'}, optional\n        Metric to normalize in.\n    return_norm : bool, optional\n        Return the length of vector `vec`, in addition to the normalized vector itself?\n\n    Returns\n    -------\n    numpy.ndarray, scipy.sparse, list of (int, float)}\n        Normalized vector in same format as `vec`.\n    float\n        Length of `vec` before normalization, if `return_norm` is set.\n\n    Notes\n    -----\n    Zero-vector will be unchanged.\n\n    "
    supported_norms = ('l1', 'l2', 'unique')
    if norm not in supported_norms:
        raise ValueError("'%s' is not a supported norm. Currently supported norms are %s." % (norm, supported_norms))
    if scipy.sparse.issparse(vec):
        vec = vec.tocsr()
        if norm == 'l1':
            veclen = np.sum(np.abs(vec.data))
        if norm == 'l2':
            veclen = np.sqrt(np.sum(vec.data ** 2))
        if norm == 'unique':
            veclen = vec.nnz
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            vec /= veclen
            if return_norm:
                return (vec, veclen)
            else:
                return vec
        elif return_norm:
            return (vec, 1.0)
        else:
            return vec
    if isinstance(vec, np.ndarray):
        if norm == 'l1':
            veclen = np.sum(np.abs(vec))
        if norm == 'l2':
            if vec.size == 0:
                veclen = 0.0
            else:
                veclen = blas_nrm2(vec)
        if norm == 'unique':
            veclen = np.count_nonzero(vec)
        if veclen > 0.0:
            if np.issubdtype(vec.dtype, np.integer):
                vec = vec.astype(float)
            if return_norm:
                return (blas_scal(1.0 / veclen, vec).astype(vec.dtype), veclen)
            else:
                return blas_scal(1.0 / veclen, vec).astype(vec.dtype)
        elif return_norm:
            return (vec, 1.0)
        else:
            return vec
    try:
        first = next(iter(vec))
    except StopIteration:
        if return_norm:
            return (vec, 1.0)
        else:
            return vec
    if isinstance(first, (tuple, list)) and len(first) == 2:
        if norm == 'l1':
            length = float(sum((abs(val) for (_, val) in vec)))
        if norm == 'l2':
            length = 1.0 * math.sqrt(sum((val ** 2 for (_, val) in vec)))
        if norm == 'unique':
            length = 1.0 * len(vec)
        assert length > 0.0, 'sparse documents must not contain any explicit zero entries'
        if return_norm:
            return (ret_normalized_vec(vec, length), length)
        else:
            return ret_normalized_vec(vec, length)
    else:
        raise ValueError('unknown input type')

def cossim(vec1, vec2):
    if False:
        return 10
    'Get cosine similarity between two sparse vectors.\n\n    Cosine similarity is a number between `<-1.0, 1.0>`, higher means more similar.\n\n    Parameters\n    ----------\n    vec1 : list of (int, float)\n        Vector in BoW format.\n    vec2 : list of (int, float)\n        Vector in BoW format.\n\n    Returns\n    -------\n    float\n        Cosine similarity between `vec1` and `vec2`.\n\n    '
    (vec1, vec2) = (dict(vec1), dict(vec2))
    if not vec1 or not vec2:
        return 0.0
    vec1len = 1.0 * math.sqrt(sum((val * val for val in vec1.values())))
    vec2len = 1.0 * math.sqrt(sum((val * val for val in vec2.values())))
    assert vec1len > 0.0 and vec2len > 0.0, 'sparse documents must not contain any explicit zero entries'
    if len(vec2) < len(vec1):
        (vec1, vec2) = (vec2, vec1)
    result = sum((value * vec2.get(index, 0.0) for (index, value) in vec1.items()))
    result /= vec1len * vec2len
    return result

def isbow(vec):
    if False:
        i = 10
        return i + 15
    'Checks if a vector is in the sparse Gensim bag-of-words format.\n\n    Parameters\n    ----------\n    vec : object\n        Object to check.\n\n    Returns\n    -------\n    bool\n        Is `vec` in BoW format.\n\n    '
    if scipy.sparse.issparse(vec):
        vec = vec.todense().tolist()
    try:
        (id_, val_) = vec[0]
        (int(id_), float(val_))
    except IndexError:
        return True
    except (ValueError, TypeError):
        return False
    return True

def _convert_vec(vec1, vec2, num_features=None):
    if False:
        while True:
            i = 10
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        if num_features is not None:
            dense1 = sparse2full(vec1, num_features)
            dense2 = sparse2full(vec2, num_features)
            return (dense1, dense2)
        else:
            max_len = max(len(vec1), len(vec2))
            dense1 = sparse2full(vec1, max_len)
            dense2 = sparse2full(vec2, max_len)
            return (dense1, dense2)
    else:
        if len(vec1) == 1:
            vec1 = vec1[0]
        if len(vec2) == 1:
            vec2 = vec2[0]
        return (vec1, vec2)

def kullback_leibler(vec1, vec2, num_features=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculate Kullback-Leibler distance between two probability distributions using `scipy.stats.entropy`.\n\n    Parameters\n    ----------\n    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n    num_features : int, optional\n        Number of features in the vectors.\n\n    Returns\n    -------\n    float\n        Kullback-Leibler distance between `vec1` and `vec2`.\n        Value in range [0, +∞) where values closer to 0 mean less distance (higher similarity).\n\n    '
    (vec1, vec2) = _convert_vec(vec1, vec2, num_features=num_features)
    return entropy(vec1, vec2)

def jensen_shannon(vec1, vec2, num_features=None):
    if False:
        while True:
            i = 10
    'Calculate Jensen-Shannon distance between two probability distributions using `scipy.stats.entropy`.\n\n    Parameters\n    ----------\n    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n    num_features : int, optional\n        Number of features in the vectors.\n\n    Returns\n    -------\n    float\n        Jensen-Shannon distance between `vec1` and `vec2`.\n\n    Notes\n    -----\n    This is a symmetric and finite "version" of :func:`gensim.matutils.kullback_leibler`.\n\n    '
    (vec1, vec2) = _convert_vec(vec1, vec2, num_features=num_features)
    avg_vec = 0.5 * (vec1 + vec2)
    return 0.5 * (entropy(vec1, avg_vec) + entropy(vec2, avg_vec))

def hellinger(vec1, vec2):
    if False:
        i = 10
        return i + 15
    'Calculate Hellinger distance between two probability distributions.\n\n    Parameters\n    ----------\n    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n\n    Returns\n    -------\n    float\n        Hellinger distance between `vec1` and `vec2`.\n        Value in range `[0, 1]`, where 0 is min distance (max similarity) and 1 is max distance (min similarity).\n\n    '
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        (vec1, vec2) = (dict(vec1), dict(vec2))
        indices = set(list(vec1.keys()) + list(vec2.keys()))
        sim = np.sqrt(0.5 * sum(((np.sqrt(vec1.get(index, 0.0)) - np.sqrt(vec2.get(index, 0.0))) ** 2 for index in indices)))
        return sim
    else:
        sim = np.sqrt(0.5 * ((np.sqrt(vec1) - np.sqrt(vec2)) ** 2).sum())
        return sim

def jaccard(vec1, vec2):
    if False:
        while True:
            i = 10
    'Calculate Jaccard distance between two vectors.\n\n    Parameters\n    ----------\n    vec1 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n    vec2 : {scipy.sparse, numpy.ndarray, list of (int, float)}\n        Distribution vector.\n\n    Returns\n    -------\n    float\n        Jaccard distance between `vec1` and `vec2`.\n        Value in range `[0, 1]`, where 0 is min distance (max similarity) and 1 is max distance (min similarity).\n\n    '
    if scipy.sparse.issparse(vec1):
        vec1 = vec1.toarray()
    if scipy.sparse.issparse(vec2):
        vec2 = vec2.toarray()
    if isbow(vec1) and isbow(vec2):
        union = sum((weight for (id_, weight) in vec1)) + sum((weight for (id_, weight) in vec2))
        (vec1, vec2) = (dict(vec1), dict(vec2))
        intersection = 0.0
        for (feature_id, feature_weight) in vec1.items():
            intersection += min(feature_weight, vec2.get(feature_id, 0.0))
        return 1 - float(intersection) / float(union)
    else:
        if isinstance(vec1, np.ndarray):
            vec1 = vec1.tolist()
        if isinstance(vec2, np.ndarray):
            vec2 = vec2.tolist()
        vec1 = set(vec1)
        vec2 = set(vec2)
        intersection = vec1 & vec2
        union = vec1 | vec2
        return 1 - float(len(intersection)) / float(len(union))

def jaccard_distance(set1, set2):
    if False:
        i = 10
        return i + 15
    'Calculate Jaccard distance between two sets.\n\n    Parameters\n    ----------\n    set1 : set\n        Input set.\n    set2 : set\n        Input set.\n\n    Returns\n    -------\n    float\n        Jaccard distance between `set1` and `set2`.\n        Value in range `[0, 1]`, where 0 is min distance (max similarity) and 1 is max distance (min similarity).\n    '
    union_cardinality = len(set1 | set2)
    if union_cardinality == 0:
        return 1.0
    return 1.0 - float(len(set1 & set2)) / float(union_cardinality)
try:
    from gensim._matutils import logsumexp, mean_absolute_difference, dirichlet_expectation
except ImportError:

    def logsumexp(x):
        if False:
            for i in range(10):
                print('nop')
        "Log of sum of exponentials.\n\n        Parameters\n        ----------\n        x : numpy.ndarray\n            Input 2d matrix.\n\n        Returns\n        -------\n        float\n            log of sum of exponentials of elements in `x`.\n\n        Warnings\n        --------\n        For performance reasons, doesn't support NaNs or 1d, 3d, etc arrays like :func:`scipy.special.logsumexp`.\n\n        "
        x_max = np.max(x)
        x = np.log(np.sum(np.exp(x - x_max)))
        x += x_max
        return x

    def mean_absolute_difference(a, b):
        if False:
            return 10
        'Mean absolute difference between two arrays.\n\n        Parameters\n        ----------\n        a : numpy.ndarray\n            Input 1d array.\n        b : numpy.ndarray\n            Input 1d array.\n\n        Returns\n        -------\n        float\n            mean(abs(a - b)).\n\n        '
        return np.mean(np.abs(a - b))

    def dirichlet_expectation(alpha):
        if False:
            for i in range(10):
                print('nop')
        'Expected value of log(theta) where theta is drawn from a Dirichlet distribution.\n\n        Parameters\n        ----------\n        alpha : numpy.ndarray\n            Dirichlet parameter 2d matrix or 1d vector, if 2d - each row is treated as a separate parameter vector.\n\n        Returns\n        -------\n        numpy.ndarray\n            Log of expected values, dimension same as `alpha.ndim`.\n\n        '
        if len(alpha.shape) == 1:
            result = psi(alpha) - psi(np.sum(alpha))
        else:
            result = psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]
        return result.astype(alpha.dtype, copy=False)

def qr_destroy(la):
    if False:
        while True:
            i = 10
    'Get QR decomposition of `la[0]`.\n\n    Parameters\n    ----------\n    la : list of numpy.ndarray\n        Run QR decomposition on the first elements of `la`. Must not be empty.\n\n    Returns\n    -------\n    (numpy.ndarray, numpy.ndarray)\n        Matrices :math:`Q` and :math:`R`.\n\n    Notes\n    -----\n    Using this function is less memory intense than calling `scipy.linalg.qr(la[0])`,\n    because the memory used in `la[0]` is reclaimed earlier. This makes a difference when\n    decomposing very large arrays, where every memory copy counts.\n\n    Warnings\n    --------\n    Content of `la` as well as `la[0]` gets destroyed in the process. Again, for memory-effiency reasons.\n\n    '
    a = np.asfortranarray(la[0])
    del la[0], la
    (m, n) = a.shape
    logger.debug('computing QR of %s dense matrix', str(a.shape))
    (geqrf,) = get_lapack_funcs(('geqrf',), (a,))
    (qr, tau, work, info) = geqrf(a, lwork=-1, overwrite_a=True)
    (qr, tau, work, info) = geqrf(a, lwork=work[0], overwrite_a=True)
    del a
    assert info >= 0
    r = triu(qr[:n, :n])
    if m < n:
        qr = qr[:, :m]
    (gorgqr,) = get_lapack_funcs(('orgqr',), (qr,))
    (q, work, info) = gorgqr(qr, tau, lwork=-1, overwrite_a=True)
    (q, work, info) = gorgqr(qr, tau, lwork=work[0], overwrite_a=True)
    assert info >= 0, 'qr failed'
    assert q.flags.f_contiguous
    return (q, r)

class MmWriter:
    """Store a corpus in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_,
    using :class:`~gensim.corpora.mmcorpus.MmCorpus`.

    Notes
    -----
    The output is written one document at a time, not the whole matrix at once (unlike e.g. `scipy.io.mmread`).
    This allows you to write corpora which are larger than the available RAM.

    The output file is created in a single pass through the input corpus, so that the input can be
    a once-only stream (generator).

    To achieve this, a fake MM header is written first, corpus statistics are collected
    during the pass (shape of the matrix, number of non-zeroes), followed by a seek back to the beginning of the file,
    rewriting the fake header with the final values.

    """
    HEADER_LINE = b'%%MatrixMarket matrix coordinate real general\n'

    def __init__(self, fname):
        if False:
            print('Hello World!')
        '\n\n        Parameters\n        ----------\n        fname : str\n            Path to output file.\n\n        '
        self.fname = fname
        if fname.endswith('.gz') or fname.endswith('.bz2'):
            raise NotImplementedError('compressed output not supported with MmWriter')
        self.fout = utils.open(self.fname, 'wb+')
        self.headers_written = False

    def write_headers(self, num_docs, num_terms, num_nnz):
        if False:
            for i in range(10):
                print('nop')
        'Write headers to file.\n\n        Parameters\n        ----------\n        num_docs : int\n            Number of documents in corpus.\n        num_terms : int\n            Number of term in corpus.\n        num_nnz : int\n            Number of non-zero elements in corpus.\n\n        '
        self.fout.write(MmWriter.HEADER_LINE)
        if num_nnz < 0:
            logger.info('saving sparse matrix to %s', self.fname)
            self.fout.write(utils.to_utf8(' ' * 50 + '\n'))
        else:
            logger.info('saving sparse %sx%s matrix with %i non-zero entries to %s', num_docs, num_terms, num_nnz, self.fname)
            self.fout.write(utils.to_utf8('%s %s %s\n' % (num_docs, num_terms, num_nnz)))
        self.last_docno = -1
        self.headers_written = True

    def fake_headers(self, num_docs, num_terms, num_nnz):
        if False:
            for i in range(10):
                print('nop')
        'Write "fake" headers to file, to be rewritten once we\'ve scanned the entire corpus.\n\n        Parameters\n        ----------\n        num_docs : int\n            Number of documents in corpus.\n        num_terms : int\n            Number of term in corpus.\n        num_nnz : int\n            Number of non-zero elements in corpus.\n\n        '
        stats = '%i %i %i' % (num_docs, num_terms, num_nnz)
        if len(stats) > 50:
            raise ValueError('Invalid stats: matrix too large!')
        self.fout.seek(len(MmWriter.HEADER_LINE))
        self.fout.write(utils.to_utf8(stats))

    def write_vector(self, docno, vector):
        if False:
            for i in range(10):
                print('nop')
        'Write a single sparse vector to the file.\n\n        Parameters\n        ----------\n        docno : int\n            Number of document.\n        vector : list of (int, number)\n            Document in BoW format.\n\n        Returns\n        -------\n        (int, int)\n            Max word index in vector and len of vector. If vector is empty, return (-1, 0).\n\n        '
        assert self.headers_written, 'must write Matrix Market file headers before writing data!'
        assert self.last_docno < docno, 'documents %i and %i not in sequential order!' % (self.last_docno, docno)
        vector = sorted(((i, w) for (i, w) in vector if abs(w) > 1e-12))
        for (termid, weight) in vector:
            self.fout.write(utils.to_utf8('%i %i %s\n' % (docno + 1, termid + 1, weight)))
        self.last_docno = docno
        return (vector[-1][0], len(vector)) if vector else (-1, 0)

    @staticmethod
    def write_corpus(fname, corpus, progress_cnt=1000, index=False, num_terms=None, metadata=False):
        if False:
            while True:
                i = 10
        'Save the corpus to disk in `Matrix Market format <https://math.nist.gov/MatrixMarket/formats.html>`_.\n\n        Parameters\n        ----------\n        fname : str\n            Filename of the resulting file.\n        corpus : iterable of list of (int, number)\n            Corpus in streamed bag-of-words format.\n        progress_cnt : int, optional\n            Print progress for every `progress_cnt` number of documents.\n        index : bool, optional\n            Return offsets?\n        num_terms : int, optional\n            Number of terms in the corpus. If provided, the `corpus.num_terms` attribute (if any) will be ignored.\n        metadata : bool, optional\n            Generate a metadata file?\n\n        Returns\n        -------\n        offsets : {list of int, None}\n            List of offsets (if index=True) or nothing.\n\n        Notes\n        -----\n        Documents are processed one at a time, so the whole corpus is allowed to be larger than the available RAM.\n\n        See Also\n        --------\n        :func:`gensim.corpora.mmcorpus.MmCorpus.save_corpus`\n            Save corpus to disk.\n\n        '
        mw = MmWriter(fname)
        mw.write_headers(-1, -1, -1)
        (_num_terms, num_nnz) = (0, 0)
        (docno, poslast) = (-1, -1)
        offsets = []
        if hasattr(corpus, 'metadata'):
            orig_metadata = corpus.metadata
            corpus.metadata = metadata
            if metadata:
                docno2metadata = {}
        else:
            metadata = False
        for (docno, doc) in enumerate(corpus):
            if metadata:
                (bow, data) = doc
                docno2metadata[docno] = data
            else:
                bow = doc
            if docno % progress_cnt == 0:
                logger.info('PROGRESS: saving document #%i', docno)
            if index:
                posnow = mw.fout.tell()
                if posnow == poslast:
                    offsets[-1] = -1
                offsets.append(posnow)
                poslast = posnow
            (max_id, veclen) = mw.write_vector(docno, bow)
            _num_terms = max(_num_terms, 1 + max_id)
            num_nnz += veclen
        if metadata:
            utils.pickle(docno2metadata, fname + '.metadata.cpickle')
            corpus.metadata = orig_metadata
        num_docs = docno + 1
        num_terms = num_terms or _num_terms
        if num_docs * num_terms != 0:
            logger.info('saved %ix%i matrix, density=%.3f%% (%i/%i)', num_docs, num_terms, 100.0 * num_nnz / (num_docs * num_terms), num_nnz, num_docs * num_terms)
        mw.fake_headers(num_docs, num_terms, num_nnz)
        mw.close()
        if index:
            return offsets

    def __del__(self):
        if False:
            while True:
                i = 10
        'Close `self.fout` file. Alias for :meth:`~gensim.matutils.MmWriter.close`.\n\n        Warnings\n        --------\n        Closing the file explicitly via the close() method is preferred and safer.\n\n        '
        self.close()

    def close(self):
        if False:
            return 10
        'Close `self.fout` file.'
        logger.debug('closing %s', self.fname)
        if hasattr(self, 'fout'):
            self.fout.close()
try:
    from gensim.corpora._mmreader import MmReader
except ImportError:
    raise utils.NO_CYTHON