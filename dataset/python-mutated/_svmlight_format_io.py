"""This module implements a loader and dumper for the svmlight format

This format is a text-based format, with one sample per line. It does
not store zero valued features hence is suitable for sparse dataset.

The first element of each line can be used to store a target variable to
predict.

This format is used as the default format for both svmlight and the
libsvm command line programs.
"""
import os.path
from contextlib import closing
from numbers import Integral
import numpy as np
import scipy.sparse as sp
from .. import __version__
from ..utils import IS_PYPY, check_array
from ..utils._param_validation import HasMethods, Interval, StrOptions, validate_params
if not IS_PYPY:
    from ._svmlight_format_fast import _dump_svmlight_file, _load_svmlight_file
else:

    def _load_svmlight_file(*args, **kwargs):
        if False:
            return 10
        raise NotImplementedError('load_svmlight_file is currently not compatible with PyPy (see https://github.com/scikit-learn/scikit-learn/issues/11543 for the status updates).')

@validate_params({'f': [str, Interval(Integral, 0, None, closed='left'), os.PathLike, HasMethods('read')], 'n_features': [Interval(Integral, 1, None, closed='left'), None], 'dtype': 'no_validation', 'multilabel': ['boolean'], 'zero_based': ['boolean', StrOptions({'auto'})], 'query_id': ['boolean'], 'offset': [Interval(Integral, 0, None, closed='left')], 'length': [Integral]}, prefer_skip_nested_validation=True)
def load_svmlight_file(f, *, n_features=None, dtype=np.float64, multilabel=False, zero_based='auto', query_id=False, offset=0, length=-1):
    if False:
        while True:
            i = 10
    'Load datasets in the svmlight / libsvm format into sparse CSR matrix.\n\n    This format is a text-based format, with one sample per line. It does\n    not store zero valued features hence is suitable for sparse dataset.\n\n    The first element of each line can be used to store a target variable\n    to predict.\n\n    This format is used as the default format for both svmlight and the\n    libsvm command line programs.\n\n    Parsing a text based source can be expensive. When repeatedly\n    working on the same dataset, it is recommended to wrap this\n    loader with joblib.Memory.cache to store a memmapped backup of the\n    CSR results of the first call and benefit from the near instantaneous\n    loading of memmapped structures for the subsequent calls.\n\n    In case the file contains a pairwise preference constraint (known\n    as "qid" in the svmlight format) these are ignored unless the\n    query_id parameter is set to True. These pairwise preference\n    constraints can be used to constraint the combination of samples\n    when using pairwise loss functions (as is the case in some\n    learning to rank problems) so that only pairs with the same\n    query_id value are considered.\n\n    This implementation is written in Cython and is reasonably fast.\n    However, a faster API-compatible loader is also available at:\n\n      https://github.com/mblondel/svmlight-loader\n\n    Parameters\n    ----------\n    f : str, path-like, file-like or int\n        (Path to) a file to load. If a path ends in ".gz" or ".bz2", it will\n        be uncompressed on the fly. If an integer is passed, it is assumed to\n        be a file descriptor. A file-like or file descriptor will not be closed\n        by this function. A file-like object must be opened in binary mode.\n\n        .. versionchanged:: 1.2\n           Path-like objects are now accepted.\n\n    n_features : int, default=None\n        The number of features to use. If None, it will be inferred. This\n        argument is useful to load several files that are subsets of a\n        bigger sliced dataset: each subset might not have examples of\n        every feature, hence the inferred shape might vary from one\n        slice to another.\n        n_features is only required if ``offset`` or ``length`` are passed a\n        non-default value.\n\n    dtype : numpy data type, default=np.float64\n        Data type of dataset to be loaded. This will be the data type of the\n        output numpy arrays ``X`` and ``y``.\n\n    multilabel : bool, default=False\n        Samples may have several labels each (see\n        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).\n\n    zero_based : bool or "auto", default="auto"\n        Whether column indices in f are zero-based (True) or one-based\n        (False). If column indices are one-based, they are transformed to\n        zero-based to match Python/NumPy conventions.\n        If set to "auto", a heuristic check is applied to determine this from\n        the file contents. Both kinds of files occur "in the wild", but they\n        are unfortunately not self-identifying. Using "auto" or True should\n        always be safe when no ``offset`` or ``length`` is passed.\n        If ``offset`` or ``length`` are passed, the "auto" mode falls back\n        to ``zero_based=True`` to avoid having the heuristic check yield\n        inconsistent results on different segments of the file.\n\n    query_id : bool, default=False\n        If True, will return the query_id array for each file.\n\n    offset : int, default=0\n        Ignore the offset first bytes by seeking forward, then\n        discarding the following bytes up until the next new line\n        character.\n\n    length : int, default=-1\n        If strictly positive, stop reading any new line of data once the\n        position in the file has reached the (offset + length) bytes threshold.\n\n    Returns\n    -------\n    X : scipy.sparse matrix of shape (n_samples, n_features)\n        The data matrix.\n\n    y : ndarray of shape (n_samples,), or a list of tuples of length n_samples\n        The target. It is a list of tuples when ``multilabel=True``, else a\n        ndarray.\n\n    query_id : array of shape (n_samples,)\n       The query_id for each sample. Only returned when query_id is set to\n       True.\n\n    See Also\n    --------\n    load_svmlight_files : Similar function for loading multiple files in this\n        format, enforcing the same number of features/columns on all of them.\n\n    Examples\n    --------\n    To use joblib.Memory to cache the svmlight file::\n\n        from joblib import Memory\n        from .datasets import load_svmlight_file\n        mem = Memory("./mycache")\n\n        @mem.cache\n        def get_data():\n            data = load_svmlight_file("mysvmlightfile")\n            return data[0], data[1]\n\n        X, y = get_data()\n    '
    return tuple(load_svmlight_files([f], n_features=n_features, dtype=dtype, multilabel=multilabel, zero_based=zero_based, query_id=query_id, offset=offset, length=length))

def _gen_open(f):
    if False:
        print('Hello World!')
    if isinstance(f, int):
        return open(f, 'rb', closefd=False)
    elif isinstance(f, os.PathLike):
        f = os.fspath(f)
    elif not isinstance(f, str):
        raise TypeError('expected {str, int, path-like, file-like}, got %s' % type(f))
    (_, ext) = os.path.splitext(f)
    if ext == '.gz':
        import gzip
        return gzip.open(f, 'rb')
    elif ext == '.bz2':
        from bz2 import BZ2File
        return BZ2File(f, 'rb')
    else:
        return open(f, 'rb')

def _open_and_load(f, dtype, multilabel, zero_based, query_id, offset=0, length=-1):
    if False:
        return 10
    if hasattr(f, 'read'):
        (actual_dtype, data, ind, indptr, labels, query) = _load_svmlight_file(f, dtype, multilabel, zero_based, query_id, offset, length)
    else:
        with closing(_gen_open(f)) as f:
            (actual_dtype, data, ind, indptr, labels, query) = _load_svmlight_file(f, dtype, multilabel, zero_based, query_id, offset, length)
    if not multilabel:
        labels = np.frombuffer(labels, np.float64)
    data = np.frombuffer(data, actual_dtype)
    indices = np.frombuffer(ind, np.longlong)
    indptr = np.frombuffer(indptr, dtype=np.longlong)
    query = np.frombuffer(query, np.int64)
    data = np.asarray(data, dtype=dtype)
    return (data, indices, indptr, labels, query)

@validate_params({'files': ['array-like', str, os.PathLike, HasMethods('read'), Interval(Integral, 0, None, closed='left')], 'n_features': [Interval(Integral, 1, None, closed='left'), None], 'dtype': 'no_validation', 'multilabel': ['boolean'], 'zero_based': ['boolean', StrOptions({'auto'})], 'query_id': ['boolean'], 'offset': [Interval(Integral, 0, None, closed='left')], 'length': [Integral]}, prefer_skip_nested_validation=True)
def load_svmlight_files(files, *, n_features=None, dtype=np.float64, multilabel=False, zero_based='auto', query_id=False, offset=0, length=-1):
    if False:
        i = 10
        return i + 15
    'Load dataset from multiple files in SVMlight format.\n\n    This function is equivalent to mapping load_svmlight_file over a list of\n    files, except that the results are concatenated into a single, flat list\n    and the samples vectors are constrained to all have the same number of\n    features.\n\n    In case the file contains a pairwise preference constraint (known\n    as "qid" in the svmlight format) these are ignored unless the\n    query_id parameter is set to True. These pairwise preference\n    constraints can be used to constraint the combination of samples\n    when using pairwise loss functions (as is the case in some\n    learning to rank problems) so that only pairs with the same\n    query_id value are considered.\n\n    Parameters\n    ----------\n    files : array-like, dtype=str, path-like, file-like or int\n        (Paths of) files to load. If a path ends in ".gz" or ".bz2", it will\n        be uncompressed on the fly. If an integer is passed, it is assumed to\n        be a file descriptor. File-likes and file descriptors will not be\n        closed by this function. File-like objects must be opened in binary\n        mode.\n\n        .. versionchanged:: 1.2\n           Path-like objects are now accepted.\n\n    n_features : int, default=None\n        The number of features to use. If None, it will be inferred from the\n        maximum column index occurring in any of the files.\n\n        This can be set to a higher value than the actual number of features\n        in any of the input files, but setting it to a lower value will cause\n        an exception to be raised.\n\n    dtype : numpy data type, default=np.float64\n        Data type of dataset to be loaded. This will be the data type of the\n        output numpy arrays ``X`` and ``y``.\n\n    multilabel : bool, default=False\n        Samples may have several labels each (see\n        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).\n\n    zero_based : bool or "auto", default="auto"\n        Whether column indices in f are zero-based (True) or one-based\n        (False). If column indices are one-based, they are transformed to\n        zero-based to match Python/NumPy conventions.\n        If set to "auto", a heuristic check is applied to determine this from\n        the file contents. Both kinds of files occur "in the wild", but they\n        are unfortunately not self-identifying. Using "auto" or True should\n        always be safe when no offset or length is passed.\n        If offset or length are passed, the "auto" mode falls back\n        to zero_based=True to avoid having the heuristic check yield\n        inconsistent results on different segments of the file.\n\n    query_id : bool, default=False\n        If True, will return the query_id array for each file.\n\n    offset : int, default=0\n        Ignore the offset first bytes by seeking forward, then\n        discarding the following bytes up until the next new line\n        character.\n\n    length : int, default=-1\n        If strictly positive, stop reading any new line of data once the\n        position in the file has reached the (offset + length) bytes threshold.\n\n    Returns\n    -------\n    [X1, y1, ..., Xn, yn] or [X1, y1, q1, ..., Xn, yn, qn]: list of arrays\n        Each (Xi, yi) pair is the result from load_svmlight_file(files[i]).\n        If query_id is set to True, this will return instead (Xi, yi, qi)\n        triplets.\n\n    See Also\n    --------\n    load_svmlight_file: Similar function for loading a single file in this\n        format.\n\n    Notes\n    -----\n    When fitting a model to a matrix X_train and evaluating it against a\n    matrix X_test, it is essential that X_train and X_test have the same\n    number of features (X_train.shape[1] == X_test.shape[1]). This may not\n    be the case if you load the files individually with load_svmlight_file.\n    '
    if (offset != 0 or length > 0) and zero_based == 'auto':
        zero_based = True
    if (offset != 0 or length > 0) and n_features is None:
        raise ValueError('n_features is required when offset or length is specified.')
    r = [_open_and_load(f, dtype, multilabel, bool(zero_based), bool(query_id), offset=offset, length=length) for f in files]
    if zero_based is False or (zero_based == 'auto' and all((len(tmp[1]) and np.min(tmp[1]) > 0 for tmp in r))):
        for (_, indices, _, _, _) in r:
            indices -= 1
    n_f = max((ind[1].max() if len(ind[1]) else 0 for ind in r)) + 1
    if n_features is None:
        n_features = n_f
    elif n_features < n_f:
        raise ValueError('n_features was set to {}, but input file contains {} features'.format(n_features, n_f))
    result = []
    for (data, indices, indptr, y, query_values) in r:
        shape = (indptr.shape[0] - 1, n_features)
        X = sp.csr_matrix((data, indices, indptr), shape)
        X.sort_indices()
        result += (X, y)
        if query_id:
            result.append(query_values)
    return result

def _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id):
    if False:
        i = 10
        return i + 15
    if comment:
        f.write(('# Generated by dump_svmlight_file from scikit-learn %s\n' % __version__).encode())
        f.write(('# Column indices are %s-based\n' % ['zero', 'one'][one_based]).encode())
        f.write(b'#\n')
        f.writelines((b'# %s\n' % line for line in comment.splitlines()))
    X_is_sp = sp.issparse(X)
    y_is_sp = sp.issparse(y)
    if not multilabel and (not y_is_sp):
        y = y[:, np.newaxis]
    _dump_svmlight_file(X, y, f, multilabel, one_based, query_id, X_is_sp, y_is_sp)

@validate_params({'X': ['array-like', 'sparse matrix'], 'y': ['array-like', 'sparse matrix'], 'f': [str, HasMethods(['write'])], 'zero_based': ['boolean'], 'comment': [str, bytes, None], 'query_id': ['array-like', None], 'multilabel': ['boolean']}, prefer_skip_nested_validation=True)
def dump_svmlight_file(X, y, f, *, zero_based=True, comment=None, query_id=None, multilabel=False):
    if False:
        while True:
            i = 10
    'Dump the dataset in svmlight / libsvm file format.\n\n    This format is a text-based format, with one sample per line. It does\n    not store zero valued features hence is suitable for sparse dataset.\n\n    The first element of each line can be used to store a target variable\n    to predict.\n\n    Parameters\n    ----------\n    X : {array-like, sparse matrix} of shape (n_samples, n_features)\n        Training vectors, where `n_samples` is the number of samples and\n        `n_features` is the number of features.\n\n    y : {array-like, sparse matrix}, shape = (n_samples,) or (n_samples, n_labels)\n        Target values. Class labels must be an\n        integer or float, or array-like objects of integer or float for\n        multilabel classifications.\n\n    f : str or file-like in binary mode\n        If string, specifies the path that will contain the data.\n        If file-like, data will be written to f. f should be opened in binary\n        mode.\n\n    zero_based : bool, default=True\n        Whether column indices should be written zero-based (True) or one-based\n        (False).\n\n    comment : str or bytes, default=None\n        Comment to insert at the top of the file. This should be either a\n        Unicode string, which will be encoded as UTF-8, or an ASCII byte\n        string.\n        If a comment is given, then it will be preceded by one that identifies\n        the file as having been dumped by scikit-learn. Note that not all\n        tools grok comments in SVMlight files.\n\n    query_id : array-like of shape (n_samples,), default=None\n        Array containing pairwise preference constraints (qid in svmlight\n        format).\n\n    multilabel : bool, default=False\n        Samples may have several labels each (see\n        https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multilabel.html).\n\n        .. versionadded:: 0.17\n           parameter `multilabel` to support multilabel datasets.\n    '
    if comment is not None:
        if isinstance(comment, bytes):
            comment.decode('ascii')
        else:
            comment = comment.encode('utf-8')
        if b'\x00' in comment:
            raise ValueError('comment string contains NUL byte')
    yval = check_array(y, accept_sparse='csr', ensure_2d=False)
    if sp.issparse(yval):
        if yval.shape[1] != 1 and (not multilabel):
            raise ValueError('expected y of shape (n_samples, 1), got %r' % (yval.shape,))
    elif yval.ndim != 1 and (not multilabel):
        raise ValueError('expected y of shape (n_samples,), got %r' % (yval.shape,))
    Xval = check_array(X, accept_sparse='csr')
    if Xval.shape[0] != yval.shape[0]:
        raise ValueError('X.shape[0] and y.shape[0] should be the same, got %r and %r instead.' % (Xval.shape[0], yval.shape[0]))
    if yval is y and hasattr(yval, 'sorted_indices'):
        y = yval.sorted_indices()
    else:
        y = yval
        if hasattr(y, 'sort_indices'):
            y.sort_indices()
    if Xval is X and hasattr(Xval, 'sorted_indices'):
        X = Xval.sorted_indices()
    else:
        X = Xval
        if hasattr(X, 'sort_indices'):
            X.sort_indices()
    if query_id is None:
        query_id = np.array([], dtype=np.int32)
    else:
        query_id = np.asarray(query_id)
        if query_id.shape[0] != y.shape[0]:
            raise ValueError('expected query_id of shape (n_samples,), got %r' % (query_id.shape,))
    one_based = not zero_based
    if hasattr(f, 'write'):
        _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)
    else:
        with open(f, 'wb') as f:
            _dump_svmlight(X, y, f, multilabel, one_based, comment, query_id)