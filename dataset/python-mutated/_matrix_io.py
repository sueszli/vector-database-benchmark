import numpy as np
import scipy.sparse
__all__ = ['save_npz', 'load_npz']
PICKLE_KWARGS = dict(allow_pickle=False)

def save_npz(file, matrix, compressed=True):
    if False:
        i = 10
        return i + 15
    " Save a sparse matrix to a file using ``.npz`` format.\n\n    Parameters\n    ----------\n    file : str or file-like object\n        Either the file name (string) or an open file (file-like object)\n        where the data will be saved. If file is a string, the ``.npz``\n        extension will be appended to the file name if it is not already\n        there.\n    matrix: spmatrix (format: ``csc``, ``csr``, ``bsr``, ``dia`` or coo``)\n        The sparse matrix to save.\n    compressed : bool, optional\n        Allow compressing the file. Default: True\n\n    See Also\n    --------\n    scipy.sparse.load_npz: Load a sparse matrix from a file using ``.npz`` format.\n    numpy.savez: Save several arrays into a ``.npz`` archive.\n    numpy.savez_compressed : Save several arrays into a compressed ``.npz`` archive.\n\n    Examples\n    --------\n    Store sparse matrix to disk, and load it again:\n\n    >>> import numpy as np\n    >>> import scipy.sparse\n    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n       with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.toarray()\n    array([[0, 0, 3],\n           [4, 0, 0]], dtype=int64)\n\n    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)\n    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')\n\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n       with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.toarray()\n    array([[0, 0, 3],\n           [4, 0, 0]], dtype=int64)\n    "
    arrays_dict = {}
    if matrix.format in ('csc', 'csr', 'bsr'):
        arrays_dict.update(indices=matrix.indices, indptr=matrix.indptr)
    elif matrix.format == 'dia':
        arrays_dict.update(offsets=matrix.offsets)
    elif matrix.format == 'coo':
        arrays_dict.update(row=matrix.row, col=matrix.col)
    else:
        raise NotImplementedError(f'Save is not implemented for sparse matrix of format {matrix.format}.')
    arrays_dict.update(format=matrix.format.encode('ascii'), shape=matrix.shape, data=matrix.data)
    if compressed:
        np.savez_compressed(file, **arrays_dict)
    else:
        np.savez(file, **arrays_dict)

def load_npz(file):
    if False:
        print('Hello World!')
    " Load a sparse matrix from a file using ``.npz`` format.\n\n    Parameters\n    ----------\n    file : str or file-like object\n        Either the file name (string) or an open file (file-like object)\n        where the data will be loaded.\n\n    Returns\n    -------\n    result : csc_matrix, csr_matrix, bsr_matrix, dia_matrix or coo_matrix\n        A sparse matrix containing the loaded data.\n\n    Raises\n    ------\n    OSError\n        If the input file does not exist or cannot be read.\n\n    See Also\n    --------\n    scipy.sparse.save_npz: Save a sparse matrix to a file using ``.npz`` format.\n    numpy.load: Load several arrays from a ``.npz`` archive.\n\n    Examples\n    --------\n    Store sparse matrix to disk, and load it again:\n\n    >>> import numpy as np\n    >>> import scipy.sparse\n    >>> sparse_matrix = scipy.sparse.csc_matrix(np.array([[0, 0, 3], [4, 0, 0]]))\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n       with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.toarray()\n    array([[0, 0, 3],\n           [4, 0, 0]], dtype=int64)\n\n    >>> scipy.sparse.save_npz('/tmp/sparse_matrix.npz', sparse_matrix)\n    >>> sparse_matrix = scipy.sparse.load_npz('/tmp/sparse_matrix.npz')\n\n    >>> sparse_matrix\n    <2x3 sparse matrix of type '<class 'numpy.int64'>'\n        with 2 stored elements in Compressed Sparse Column format>\n    >>> sparse_matrix.toarray()\n    array([[0, 0, 3],\n           [4, 0, 0]], dtype=int64)\n    "
    with np.load(file, **PICKLE_KWARGS) as loaded:
        try:
            matrix_format = loaded['format']
        except KeyError as e:
            raise ValueError(f'The file {file} does not contain a sparse matrix.') from e
        matrix_format = matrix_format.item()
        if not isinstance(matrix_format, str):
            matrix_format = matrix_format.decode('ascii')
        try:
            cls = getattr(scipy.sparse, f'{matrix_format}_matrix')
        except AttributeError as e:
            raise ValueError(f'Unknown matrix format "{matrix_format}"') from e
        if matrix_format in ('csc', 'csr', 'bsr'):
            return cls((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])
        elif matrix_format == 'dia':
            return cls((loaded['data'], loaded['offsets']), shape=loaded['shape'])
        elif matrix_format == 'coo':
            return cls((loaded['data'], (loaded['row'], loaded['col'])), shape=loaded['shape'])
        else:
            raise NotImplementedError('Load is not implemented for sparse matrix of format {}.'.format(matrix_format))