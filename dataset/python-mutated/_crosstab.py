import numpy as np
from scipy.sparse import coo_matrix
from scipy._lib._bunch import _make_tuple_bunch
CrosstabResult = _make_tuple_bunch('CrosstabResult', ['elements', 'count'])

def crosstab(*args, levels=None, sparse=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return table of counts for each possible unique combination in ``*args``.\n\n    When ``len(args) > 1``, the array computed by this function is\n    often referred to as a *contingency table* [1]_.\n\n    The arguments must be sequences with the same length.  The second return\n    value, `count`, is an integer array with ``len(args)`` dimensions.  If\n    `levels` is None, the shape of `count` is ``(n0, n1, ...)``, where ``nk``\n    is the number of unique elements in ``args[k]``.\n\n    Parameters\n    ----------\n    *args : sequences\n        A sequence of sequences whose unique aligned elements are to be\n        counted.  The sequences in args must all be the same length.\n    levels : sequence, optional\n        If `levels` is given, it must be a sequence that is the same length as\n        `args`.  Each element in `levels` is either a sequence or None.  If it\n        is a sequence, it gives the values in the corresponding sequence in\n        `args` that are to be counted.  If any value in the sequences in `args`\n        does not occur in the corresponding sequence in `levels`, that value\n        is ignored and not counted in the returned array `count`.  The default\n        value of `levels` for ``args[i]`` is ``np.unique(args[i])``\n    sparse : bool, optional\n        If True, return a sparse matrix.  The matrix will be an instance of\n        the `scipy.sparse.coo_matrix` class.  Because SciPy\'s sparse matrices\n        must be 2-d, only two input sequences are allowed when `sparse` is\n        True.  Default is False.\n\n    Returns\n    -------\n    res : CrosstabResult\n        An object containing the following attributes:\n\n        elements : tuple of numpy.ndarrays.\n            Tuple of length ``len(args)`` containing the arrays of elements\n            that are counted in `count`.  These can be interpreted as the\n            labels of the corresponding dimensions of `count`. If `levels` was\n            given, then if ``levels[i]`` is not None, ``elements[i]`` will\n            hold the values given in ``levels[i]``.\n        count : numpy.ndarray or scipy.sparse.coo_matrix\n            Counts of the unique elements in ``zip(*args)``, stored in an\n            array. Also known as a *contingency table* when ``len(args) > 1``.\n\n    See Also\n    --------\n    numpy.unique\n\n    Notes\n    -----\n    .. versionadded:: 1.7.0\n\n    References\n    ----------\n    .. [1] "Contingency table", http://en.wikipedia.org/wiki/Contingency_table\n\n    Examples\n    --------\n    >>> from scipy.stats.contingency import crosstab\n\n    Given the lists `a` and `x`, create a contingency table that counts the\n    frequencies of the corresponding pairs.\n\n    >>> a = [\'A\', \'B\', \'A\', \'A\', \'B\', \'B\', \'A\', \'A\', \'B\', \'B\']\n    >>> x = [\'X\', \'X\', \'X\', \'Y\', \'Z\', \'Z\', \'Y\', \'Y\', \'Z\', \'Z\']\n    >>> res = crosstab(a, x)\n    >>> avals, xvals = res.elements\n    >>> avals\n    array([\'A\', \'B\'], dtype=\'<U1\')\n    >>> xvals\n    array([\'X\', \'Y\', \'Z\'], dtype=\'<U1\')\n    >>> res.count\n    array([[2, 3, 0],\n           [1, 0, 4]])\n\n    So `(\'A\', \'X\')` occurs twice, `(\'A\', \'Y\')` occurs three times, etc.\n\n    Higher dimensional contingency tables can be created.\n\n    >>> p = [0, 0, 0, 0, 1, 1, 1, 0, 0, 1]\n    >>> res = crosstab(a, x, p)\n    >>> res.count\n    array([[[2, 0],\n            [2, 1],\n            [0, 0]],\n           [[1, 0],\n            [0, 0],\n            [1, 3]]])\n    >>> res.count.shape\n    (2, 3, 2)\n\n    The values to be counted can be set by using the `levels` argument.\n    It allows the elements of interest in each input sequence to be\n    given explicitly instead finding the unique elements of the sequence.\n\n    For example, suppose one of the arguments is an array containing the\n    answers to a survey question, with integer values 1 to 4.  Even if the\n    value 1 does not occur in the data, we want an entry for it in the table.\n\n    >>> q1 = [2, 3, 3, 2, 4, 4, 2, 3, 4, 4, 4, 3, 3, 3, 4]  # 1 does not occur.\n    >>> q2 = [4, 4, 2, 2, 2, 4, 1, 1, 2, 2, 4, 2, 2, 2, 4]  # 3 does not occur.\n    >>> options = [1, 2, 3, 4]\n    >>> res = crosstab(q1, q2, levels=(options, options))\n    >>> res.count\n    array([[0, 0, 0, 0],\n           [1, 1, 0, 1],\n           [1, 4, 0, 1],\n           [0, 3, 0, 3]])\n\n    If `levels` is given, but an element of `levels` is None, the unique values\n    of the corresponding argument are used. For example,\n\n    >>> res = crosstab(q1, q2, levels=(None, options))\n    >>> res.elements\n    [array([2, 3, 4]), [1, 2, 3, 4]]\n    >>> res.count\n    array([[1, 1, 0, 1],\n           [1, 4, 0, 1],\n           [0, 3, 0, 3]])\n\n    If we want to ignore the pairs where 4 occurs in ``q2``, we can\n    give just the values [1, 2] to `levels`, and the 4 will be ignored:\n\n    >>> res = crosstab(q1, q2, levels=(None, [1, 2]))\n    >>> res.elements\n    [array([2, 3, 4]), [1, 2]]\n    >>> res.count\n    array([[1, 1],\n           [1, 4],\n           [0, 3]])\n\n    Finally, let\'s repeat the first example, but return a sparse matrix:\n\n    >>> res = crosstab(a, x, sparse=True)\n    >>> res.count\n    <2x3 sparse matrix of type \'<class \'numpy.int64\'>\'\n            with 4 stored elements in COOrdinate format>\n    >>> res.count.A\n    array([[2, 3, 0],\n           [1, 0, 4]])\n\n    '
    nargs = len(args)
    if nargs == 0:
        raise TypeError('At least one input sequence is required.')
    len0 = len(args[0])
    if not all((len(a) == len0 for a in args[1:])):
        raise ValueError('All input sequences must have the same length.')
    if sparse and nargs != 2:
        raise ValueError('When `sparse` is True, only two input sequences are allowed.')
    if levels is None:
        (actual_levels, indices) = zip(*[np.unique(a, return_inverse=True) for a in args])
    else:
        if len(levels) != nargs:
            raise ValueError('len(levels) must equal the number of input sequences')
        args = [np.asarray(arg) for arg in args]
        mask = np.zeros((nargs, len0), dtype=np.bool_)
        inv = np.zeros((nargs, len0), dtype=np.intp)
        actual_levels = []
        for (k, (levels_list, arg)) in enumerate(zip(levels, args)):
            if levels_list is None:
                (levels_list, inv[k, :]) = np.unique(arg, return_inverse=True)
                mask[k, :] = True
            else:
                q = arg == np.asarray(levels_list).reshape(-1, 1)
                mask[k, :] = np.any(q, axis=0)
                qnz = q.T.nonzero()
                inv[k, qnz[0]] = qnz[1]
            actual_levels.append(levels_list)
        mask_all = mask.all(axis=0)
        indices = tuple(inv[:, mask_all])
    if sparse:
        count = coo_matrix((np.ones(len(indices[0]), dtype=int), (indices[0], indices[1])))
        count.sum_duplicates()
    else:
        shape = [len(u) for u in actual_levels]
        count = np.zeros(shape, dtype=int)
        np.add.at(count, indices, 1)
    return CrosstabResult(actual_levels, count)