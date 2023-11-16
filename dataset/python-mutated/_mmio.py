"""
  Matrix Market I/O in Python.
  See http://math.nist.gov/MatrixMarket/formats.html
  for information about the Matrix Market format.
"""
import os
import numpy as np
from numpy import asarray, real, imag, conj, zeros, ndarray, concatenate, ones, can_cast
from scipy.sparse import coo_matrix, issparse
__all__ = ['mminfo', 'mmread', 'mmwrite', 'MMFile']

def asstr(s):
    if False:
        print('Hello World!')
    if isinstance(s, bytes):
        return s.decode('latin1')
    return str(s)

def mminfo(source):
    if False:
        for i in range(10):
            print('nop')
    "\n    Return size and storage parameters from Matrix Market file-like 'source'.\n\n    Parameters\n    ----------\n    source : str or file-like\n        Matrix Market filename (extension .mtx) or open file-like object\n\n    Returns\n    -------\n    rows : int\n        Number of matrix rows.\n    cols : int\n        Number of matrix columns.\n    entries : int\n        Number of non-zero entries of a sparse matrix\n        or rows*cols for a dense matrix.\n    format : str\n        Either 'coordinate' or 'array'.\n    field : str\n        Either 'real', 'complex', 'pattern', or 'integer'.\n    symmetry : str\n        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n\n    Examples\n    --------\n    >>> from io import StringIO\n    >>> from scipy.io import mminfo\n\n    >>> text = '''%%MatrixMarket matrix coordinate real general\n    ...  5 5 7\n    ...  2 3 1.0\n    ...  3 4 2.0\n    ...  3 5 3.0\n    ...  4 1 4.0\n    ...  4 2 5.0\n    ...  4 3 6.0\n    ...  4 4 7.0\n    ... '''\n\n\n    ``mminfo(source)`` returns the number of rows, number of columns,\n    format, field type and symmetry attribute of the source file.\n\n    >>> mminfo(StringIO(text))\n    (5, 5, 7, 'coordinate', 'real', 'general')\n    "
    return MMFile.info(source)

def mmread(source):
    if False:
        print('Hello World!')
    "\n    Reads the contents of a Matrix Market file-like 'source' into a matrix.\n\n    Parameters\n    ----------\n    source : str or file-like\n        Matrix Market filename (extensions .mtx, .mtz.gz)\n        or open file-like object.\n\n    Returns\n    -------\n    a : ndarray or coo_matrix\n        Dense or sparse matrix depending on the matrix format in the\n        Matrix Market file.\n\n    Examples\n    --------\n    >>> from io import StringIO\n    >>> from scipy.io import mmread\n\n    >>> text = '''%%MatrixMarket matrix coordinate real general\n    ...  5 5 7\n    ...  2 3 1.0\n    ...  3 4 2.0\n    ...  3 5 3.0\n    ...  4 1 4.0\n    ...  4 2 5.0\n    ...  4 3 6.0\n    ...  4 4 7.0\n    ... '''\n\n    ``mmread(source)`` returns the data as sparse matrix in COO format.\n\n    >>> m = mmread(StringIO(text))\n    >>> m\n    <5x5 sparse matrix of type '<class 'numpy.float64'>'\n    with 7 stored elements in COOrdinate format>\n    >>> m.A\n    array([[0., 0., 0., 0., 0.],\n           [0., 0., 1., 0., 0.],\n           [0., 0., 0., 2., 3.],\n           [4., 5., 6., 7., 0.],\n           [0., 0., 0., 0., 0.]])\n    "
    return MMFile().read(source)

def mmwrite(target, a, comment='', field=None, precision=None, symmetry=None):
    if False:
        for i in range(10):
            print('nop')
    "\n    Writes the sparse or dense array `a` to Matrix Market file-like `target`.\n\n    Parameters\n    ----------\n    target : str or file-like\n        Matrix Market filename (extension .mtx) or open file-like object.\n    a : array like\n        Sparse or dense 2-D array.\n    comment : str, optional\n        Comments to be prepended to the Matrix Market file.\n    field : None or str, optional\n        Either 'real', 'complex', 'pattern', or 'integer'.\n    precision : None or int, optional\n        Number of digits to display for real or complex values.\n    symmetry : None or str, optional\n        Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n        If symmetry is None the symmetry type of 'a' is determined by its\n        values.\n\n    Returns\n    -------\n    None\n\n    Examples\n    --------\n    >>> from io import BytesIO\n    >>> import numpy as np\n    >>> from scipy.sparse import coo_matrix\n    >>> from scipy.io import mmwrite\n\n    Write a small NumPy array to a matrix market file.  The file will be\n    written in the ``'array'`` format.\n\n    >>> a = np.array([[1.0, 0, 0, 0], [0, 2.5, 0, 6.25]])\n    >>> target = BytesIO()\n    >>> mmwrite(target, a)\n    >>> print(target.getvalue().decode('latin1'))\n    %%MatrixMarket matrix array real general\n    %\n    2 4\n    1.0000000000000000e+00\n    0.0000000000000000e+00\n    0.0000000000000000e+00\n    2.5000000000000000e+00\n    0.0000000000000000e+00\n    0.0000000000000000e+00\n    0.0000000000000000e+00\n    6.2500000000000000e+00\n\n    Add a comment to the output file, and set the precision to 3.\n\n    >>> target = BytesIO()\n    >>> mmwrite(target, a, comment='\\n Some test data.\\n', precision=3)\n    >>> print(target.getvalue().decode('latin1'))\n    %%MatrixMarket matrix array real general\n    %\n    % Some test data.\n    %\n    2 4\n    1.000e+00\n    0.000e+00\n    0.000e+00\n    2.500e+00\n    0.000e+00\n    0.000e+00\n    0.000e+00\n    6.250e+00\n\n    Convert to a sparse matrix before calling ``mmwrite``.  This will\n    result in the output format being ``'coordinate'`` rather than\n    ``'array'``.\n\n    >>> target = BytesIO()\n    >>> mmwrite(target, coo_matrix(a), precision=3)\n    >>> print(target.getvalue().decode('latin1'))\n    %%MatrixMarket matrix coordinate real general\n    %\n    2 4 3\n    1 1 1.00e+00\n    2 2 2.50e+00\n    2 4 6.25e+00\n\n    Write a complex Hermitian array to a matrix market file.  Note that\n    only six values are actually written to the file; the other values\n    are implied by the symmetry.\n\n    >>> z = np.array([[3, 1+2j, 4-3j], [1-2j, 1, -5j], [4+3j, 5j, 2.5]])\n    >>> z\n    array([[ 3. +0.j,  1. +2.j,  4. -3.j],\n           [ 1. -2.j,  1. +0.j, -0. -5.j],\n           [ 4. +3.j,  0. +5.j,  2.5+0.j]])\n\n    >>> target = BytesIO()\n    >>> mmwrite(target, z, precision=2)\n    >>> print(target.getvalue().decode('latin1'))\n    %%MatrixMarket matrix array complex hermitian\n    %\n    3 3\n    3.00e+00 0.00e+00\n    1.00e+00 -2.00e+00\n    4.00e+00 3.00e+00\n    1.00e+00 0.00e+00\n    0.00e+00 5.00e+00\n    2.50e+00 0.00e+00\n\n    "
    MMFile().write(target, a, comment, field, precision, symmetry)

class MMFile:
    __slots__ = ('_rows', '_cols', '_entries', '_format', '_field', '_symmetry')

    @property
    def rows(self):
        if False:
            while True:
                i = 10
        return self._rows

    @property
    def cols(self):
        if False:
            i = 10
            return i + 15
        return self._cols

    @property
    def entries(self):
        if False:
            while True:
                i = 10
        return self._entries

    @property
    def format(self):
        if False:
            print('Hello World!')
        return self._format

    @property
    def field(self):
        if False:
            for i in range(10):
                print('nop')
        return self._field

    @property
    def symmetry(self):
        if False:
            return 10
        return self._symmetry

    @property
    def has_symmetry(self):
        if False:
            print('Hello World!')
        return self._symmetry in (self.SYMMETRY_SYMMETRIC, self.SYMMETRY_SKEW_SYMMETRIC, self.SYMMETRY_HERMITIAN)
    FORMAT_COORDINATE = 'coordinate'
    FORMAT_ARRAY = 'array'
    FORMAT_VALUES = (FORMAT_COORDINATE, FORMAT_ARRAY)

    @classmethod
    def _validate_format(self, format):
        if False:
            i = 10
            return i + 15
        if format not in self.FORMAT_VALUES:
            raise ValueError('unknown format type %s, must be one of %s' % (format, self.FORMAT_VALUES))
    FIELD_INTEGER = 'integer'
    FIELD_UNSIGNED = 'unsigned-integer'
    FIELD_REAL = 'real'
    FIELD_COMPLEX = 'complex'
    FIELD_PATTERN = 'pattern'
    FIELD_VALUES = (FIELD_INTEGER, FIELD_UNSIGNED, FIELD_REAL, FIELD_COMPLEX, FIELD_PATTERN)

    @classmethod
    def _validate_field(self, field):
        if False:
            i = 10
            return i + 15
        if field not in self.FIELD_VALUES:
            raise ValueError('unknown field type %s, must be one of %s' % (field, self.FIELD_VALUES))
    SYMMETRY_GENERAL = 'general'
    SYMMETRY_SYMMETRIC = 'symmetric'
    SYMMETRY_SKEW_SYMMETRIC = 'skew-symmetric'
    SYMMETRY_HERMITIAN = 'hermitian'
    SYMMETRY_VALUES = (SYMMETRY_GENERAL, SYMMETRY_SYMMETRIC, SYMMETRY_SKEW_SYMMETRIC, SYMMETRY_HERMITIAN)

    @classmethod
    def _validate_symmetry(self, symmetry):
        if False:
            return 10
        if symmetry not in self.SYMMETRY_VALUES:
            raise ValueError('unknown symmetry type %s, must be one of %s' % (symmetry, self.SYMMETRY_VALUES))
    DTYPES_BY_FIELD = {FIELD_INTEGER: 'intp', FIELD_UNSIGNED: 'uint64', FIELD_REAL: 'd', FIELD_COMPLEX: 'D', FIELD_PATTERN: 'd'}

    @staticmethod
    def reader():
        if False:
            return 10
        pass

    @staticmethod
    def writer():
        if False:
            for i in range(10):
                print('nop')
        pass

    @classmethod
    def info(self, source):
        if False:
            return 10
        "\n        Return size, storage parameters from Matrix Market file-like 'source'.\n\n        Parameters\n        ----------\n        source : str or file-like\n            Matrix Market filename (extension .mtx) or open file-like object\n\n        Returns\n        -------\n        rows : int\n            Number of matrix rows.\n        cols : int\n            Number of matrix columns.\n        entries : int\n            Number of non-zero entries of a sparse matrix\n            or rows*cols for a dense matrix.\n        format : str\n            Either 'coordinate' or 'array'.\n        field : str\n            Either 'real', 'complex', 'pattern', or 'integer'.\n        symmetry : str\n            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n        "
        (stream, close_it) = self._open(source)
        try:
            line = stream.readline()
            (mmid, matrix, format, field, symmetry) = (asstr(part.strip()) for part in line.split())
            if not mmid.startswith('%%MatrixMarket'):
                raise ValueError('source is not in Matrix Market format')
            if not matrix.lower() == 'matrix':
                raise ValueError('Problem reading file header: ' + line)
            if format.lower() == 'array':
                format = self.FORMAT_ARRAY
            elif format.lower() == 'coordinate':
                format = self.FORMAT_COORDINATE
            while line:
                if line.lstrip() and line.lstrip()[0] in ['%', 37]:
                    line = stream.readline()
                else:
                    break
            while not line.strip():
                line = stream.readline()
            split_line = line.split()
            if format == self.FORMAT_ARRAY:
                if not len(split_line) == 2:
                    raise ValueError('Header line not of length 2: ' + line.decode('ascii'))
                (rows, cols) = map(int, split_line)
                entries = rows * cols
            else:
                if not len(split_line) == 3:
                    raise ValueError('Header line not of length 3: ' + line.decode('ascii'))
                (rows, cols, entries) = map(int, split_line)
            return (rows, cols, entries, format, field.lower(), symmetry.lower())
        finally:
            if close_it:
                stream.close()

    @staticmethod
    def _open(filespec, mode='rb'):
        if False:
            i = 10
            return i + 15
        ' Return an open file stream for reading based on source.\n\n        If source is a file name, open it (after trying to find it with mtx and\n        gzipped mtx extensions). Otherwise, just return source.\n\n        Parameters\n        ----------\n        filespec : str or file-like\n            String giving file name or file-like object\n        mode : str, optional\n            Mode with which to open file, if `filespec` is a file name.\n\n        Returns\n        -------\n        fobj : file-like\n            Open file-like object.\n        close_it : bool\n            True if the calling function should close this file when done,\n            false otherwise.\n        '
        try:
            filespec = os.fspath(filespec)
        except TypeError:
            return (filespec, False)
        if mode[0] == 'r':
            if not os.path.isfile(filespec):
                if os.path.isfile(filespec + '.mtx'):
                    filespec = filespec + '.mtx'
                elif os.path.isfile(filespec + '.mtx.gz'):
                    filespec = filespec + '.mtx.gz'
                elif os.path.isfile(filespec + '.mtx.bz2'):
                    filespec = filespec + '.mtx.bz2'
            if filespec.endswith('.gz'):
                import gzip
                stream = gzip.open(filespec, mode)
            elif filespec.endswith('.bz2'):
                import bz2
                stream = bz2.BZ2File(filespec, 'rb')
            else:
                stream = open(filespec, mode)
        else:
            if filespec[-4:] != '.mtx':
                filespec = filespec + '.mtx'
            stream = open(filespec, mode)
        return (stream, True)

    @staticmethod
    def _get_symmetry(a):
        if False:
            print('Hello World!')
        (m, n) = a.shape
        if m != n:
            return MMFile.SYMMETRY_GENERAL
        issymm = True
        isskew = True
        isherm = a.dtype.char in 'FD'
        if issparse(a):
            a = a.tocoo()
            (row, col) = a.nonzero()
            if (row < col).sum() != (row > col).sum():
                return MMFile.SYMMETRY_GENERAL
            a = a.todok()

            def symm_iterator():
                if False:
                    i = 10
                    return i + 15
                for ((i, j), aij) in a.items():
                    if i > j:
                        aji = a[j, i]
                        yield (aij, aji, False)
                    elif i == j:
                        yield (aij, aij, True)
        else:

            def symm_iterator():
                if False:
                    return 10
                for j in range(n):
                    for i in range(j, n):
                        (aij, aji) = (a[i][j], a[j][i])
                        yield (aij, aji, i == j)
        for (aij, aji, is_diagonal) in symm_iterator():
            if isskew and is_diagonal and (aij != 0):
                isskew = False
            else:
                if issymm and aij != aji:
                    issymm = False
                with np.errstate(over='ignore'):
                    if isskew and aij != -aji:
                        isskew = False
                if isherm and aij != conj(aji):
                    isherm = False
            if not (issymm or isskew or isherm):
                break
        if issymm:
            return MMFile.SYMMETRY_SYMMETRIC
        if isskew:
            return MMFile.SYMMETRY_SKEW_SYMMETRIC
        if isherm:
            return MMFile.SYMMETRY_HERMITIAN
        return MMFile.SYMMETRY_GENERAL

    @staticmethod
    def _field_template(field, precision):
        if False:
            print('Hello World!')
        return {MMFile.FIELD_REAL: '%%.%ie\n' % precision, MMFile.FIELD_INTEGER: '%i\n', MMFile.FIELD_UNSIGNED: '%u\n', MMFile.FIELD_COMPLEX: '%%.%ie %%.%ie\n' % (precision, precision)}.get(field, None)

    def __init__(self, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self._init_attrs(**kwargs)

    def read(self, source):
        if False:
            for i in range(10):
                print('nop')
        "\n        Reads the contents of a Matrix Market file-like 'source' into a matrix.\n\n        Parameters\n        ----------\n        source : str or file-like\n            Matrix Market filename (extensions .mtx, .mtz.gz)\n            or open file object.\n\n        Returns\n        -------\n        a : ndarray or coo_matrix\n            Dense or sparse matrix depending on the matrix format in the\n            Matrix Market file.\n        "
        (stream, close_it) = self._open(source)
        try:
            self._parse_header(stream)
            return self._parse_body(stream)
        finally:
            if close_it:
                stream.close()

    def write(self, target, a, comment='', field=None, precision=None, symmetry=None):
        if False:
            i = 10
            return i + 15
        "\n        Writes sparse or dense array `a` to Matrix Market file-like `target`.\n\n        Parameters\n        ----------\n        target : str or file-like\n            Matrix Market filename (extension .mtx) or open file-like object.\n        a : array like\n            Sparse or dense 2-D array.\n        comment : str, optional\n            Comments to be prepended to the Matrix Market file.\n        field : None or str, optional\n            Either 'real', 'complex', 'pattern', or 'integer'.\n        precision : None or int, optional\n            Number of digits to display for real or complex values.\n        symmetry : None or str, optional\n            Either 'general', 'symmetric', 'skew-symmetric', or 'hermitian'.\n            If symmetry is None the symmetry type of 'a' is determined by its\n            values.\n        "
        (stream, close_it) = self._open(target, 'wb')
        try:
            self._write(stream, a, comment, field, precision, symmetry)
        finally:
            if close_it:
                stream.close()
            else:
                stream.flush()

    def _init_attrs(self, **kwargs):
        if False:
            while True:
                i = 10
        '\n        Initialize each attributes with the corresponding keyword arg value\n        or a default of None\n        '
        attrs = self.__class__.__slots__
        public_attrs = [attr[1:] for attr in attrs]
        invalid_keys = set(kwargs.keys()) - set(public_attrs)
        if invalid_keys:
            raise ValueError('found {} invalid keyword arguments, please only\n                                use {}'.format(tuple(invalid_keys), public_attrs))
        for attr in attrs:
            setattr(self, attr, kwargs.get(attr[1:], None))

    def _parse_header(self, stream):
        if False:
            print('Hello World!')
        (rows, cols, entries, format, field, symmetry) = self.__class__.info(stream)
        self._init_attrs(rows=rows, cols=cols, entries=entries, format=format, field=field, symmetry=symmetry)

    def _parse_body(self, stream):
        if False:
            print('Hello World!')
        (rows, cols, entries, format, field, symm) = (self.rows, self.cols, self.entries, self.format, self.field, self.symmetry)
        dtype = self.DTYPES_BY_FIELD.get(field, None)
        has_symmetry = self.has_symmetry
        is_integer = field == self.FIELD_INTEGER
        is_unsigned_integer = field == self.FIELD_UNSIGNED
        is_complex = field == self.FIELD_COMPLEX
        is_skew = symm == self.SYMMETRY_SKEW_SYMMETRIC
        is_herm = symm == self.SYMMETRY_HERMITIAN
        is_pattern = field == self.FIELD_PATTERN
        if format == self.FORMAT_ARRAY:
            a = zeros((rows, cols), dtype=dtype)
            line = 1
            (i, j) = (0, 0)
            if is_skew:
                a[i, j] = 0
                if i < rows - 1:
                    i += 1
            while line:
                line = stream.readline()
                if not line or line[0] in ['%', 37] or (not line.strip()):
                    continue
                if is_integer:
                    aij = int(line)
                elif is_unsigned_integer:
                    aij = int(line)
                elif is_complex:
                    aij = complex(*map(float, line.split()))
                else:
                    aij = float(line)
                a[i, j] = aij
                if has_symmetry and i != j:
                    if is_skew:
                        a[j, i] = -aij
                    elif is_herm:
                        a[j, i] = conj(aij)
                    else:
                        a[j, i] = aij
                if i < rows - 1:
                    i = i + 1
                else:
                    j = j + 1
                    if not has_symmetry:
                        i = 0
                    else:
                        i = j
                        if is_skew:
                            a[i, j] = 0
                            if i < rows - 1:
                                i += 1
            if is_skew:
                if not (i in [0, j] and j == cols - 1):
                    raise ValueError('Parse error, did not read all lines.')
            elif not (i in [0, j] and j == cols):
                raise ValueError('Parse error, did not read all lines.')
        elif format == self.FORMAT_COORDINATE:
            if entries == 0:
                return coo_matrix((rows, cols), dtype=dtype)
            I = zeros(entries, dtype='intc')
            J = zeros(entries, dtype='intc')
            if is_pattern:
                V = ones(entries, dtype='int8')
            elif is_integer:
                V = zeros(entries, dtype='intp')
            elif is_unsigned_integer:
                V = zeros(entries, dtype='uint64')
            elif is_complex:
                V = zeros(entries, dtype='complex')
            else:
                V = zeros(entries, dtype='float')
            entry_number = 0
            for line in stream:
                if not line or line[0] in ['%', 37] or (not line.strip()):
                    continue
                if entry_number + 1 > entries:
                    raise ValueError("'entries' in header is smaller than number of entries")
                l = line.split()
                (I[entry_number], J[entry_number]) = map(int, l[:2])
                if not is_pattern:
                    if is_integer:
                        V[entry_number] = int(l[2])
                    elif is_unsigned_integer:
                        V[entry_number] = int(l[2])
                    elif is_complex:
                        V[entry_number] = complex(*map(float, l[2:]))
                    else:
                        V[entry_number] = float(l[2])
                entry_number += 1
            if entry_number < entries:
                raise ValueError("'entries' in header is larger than number of entries")
            I -= 1
            J -= 1
            if has_symmetry:
                mask = I != J
                od_I = I[mask]
                od_J = J[mask]
                od_V = V[mask]
                I = concatenate((I, od_J))
                J = concatenate((J, od_I))
                if is_skew:
                    od_V *= -1
                elif is_herm:
                    od_V = od_V.conjugate()
                V = concatenate((V, od_V))
            a = coo_matrix((V, (I, J)), shape=(rows, cols), dtype=dtype)
        else:
            raise NotImplementedError(format)
        return a

    def _write(self, stream, a, comment='', field=None, precision=None, symmetry=None):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(a, list) or isinstance(a, ndarray) or isinstance(a, tuple) or hasattr(a, '__array__'):
            rep = self.FORMAT_ARRAY
            a = asarray(a)
            if len(a.shape) != 2:
                raise ValueError('Expected 2 dimensional array')
            (rows, cols) = a.shape
            if field is not None:
                if field == self.FIELD_INTEGER:
                    if not can_cast(a.dtype, 'intp'):
                        raise OverflowError("mmwrite does not support integer dtypes larger than native 'intp'.")
                    a = a.astype('intp')
                elif field == self.FIELD_REAL:
                    if a.dtype.char not in 'fd':
                        a = a.astype('d')
                elif field == self.FIELD_COMPLEX:
                    if a.dtype.char not in 'FD':
                        a = a.astype('D')
        else:
            if not issparse(a):
                raise ValueError('unknown matrix type: %s' % type(a))
            rep = 'coordinate'
            (rows, cols) = a.shape
        typecode = a.dtype.char
        if precision is None:
            if typecode in 'fF':
                precision = 8
            else:
                precision = 16
        if field is None:
            kind = a.dtype.kind
            if kind == 'i':
                if not can_cast(a.dtype, 'intp'):
                    raise OverflowError("mmwrite does not support integer dtypes larger than native 'intp'.")
                field = 'integer'
            elif kind == 'f':
                field = 'real'
            elif kind == 'c':
                field = 'complex'
            elif kind == 'u':
                field = 'unsigned-integer'
            else:
                raise TypeError('unexpected dtype kind ' + kind)
        if symmetry is None:
            symmetry = self._get_symmetry(a)
        self.__class__._validate_format(rep)
        self.__class__._validate_field(field)
        self.__class__._validate_symmetry(symmetry)
        data = f'%%MatrixMarket matrix {rep} {field} {symmetry}\n'
        stream.write(data.encode('latin1'))
        for line in comment.split('\n'):
            data = '%%%s\n' % line
            stream.write(data.encode('latin1'))
        template = self._field_template(field, precision)
        if rep == self.FORMAT_ARRAY:
            data = '%i %i\n' % (rows, cols)
            stream.write(data.encode('latin1'))
            if field in (self.FIELD_INTEGER, self.FIELD_REAL, self.FIELD_UNSIGNED):
                if symmetry == self.SYMMETRY_GENERAL:
                    for j in range(cols):
                        for i in range(rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))
                elif symmetry == self.SYMMETRY_SKEW_SYMMETRIC:
                    for j in range(cols):
                        for i in range(j + 1, rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))
                else:
                    for j in range(cols):
                        for i in range(j, rows):
                            data = template % a[i, j]
                            stream.write(data.encode('latin1'))
            elif field == self.FIELD_COMPLEX:
                if symmetry == self.SYMMETRY_GENERAL:
                    for j in range(cols):
                        for i in range(rows):
                            aij = a[i, j]
                            data = template % (real(aij), imag(aij))
                            stream.write(data.encode('latin1'))
                else:
                    for j in range(cols):
                        for i in range(j, rows):
                            aij = a[i, j]
                            data = template % (real(aij), imag(aij))
                            stream.write(data.encode('latin1'))
            elif field == self.FIELD_PATTERN:
                raise ValueError('pattern type inconsisted with dense format')
            else:
                raise TypeError('Unknown field type %s' % field)
        else:
            coo = a.tocoo()
            if symmetry != self.SYMMETRY_GENERAL:
                lower_triangle_mask = coo.row >= coo.col
                coo = coo_matrix((coo.data[lower_triangle_mask], (coo.row[lower_triangle_mask], coo.col[lower_triangle_mask])), shape=coo.shape)
            data = '%i %i %i\n' % (rows, cols, coo.nnz)
            stream.write(data.encode('latin1'))
            template = self._field_template(field, precision - 1)
            if field == self.FIELD_PATTERN:
                for (r, c) in zip(coo.row + 1, coo.col + 1):
                    data = '%i %i\n' % (r, c)
                    stream.write(data.encode('latin1'))
            elif field in (self.FIELD_INTEGER, self.FIELD_REAL, self.FIELD_UNSIGNED):
                for (r, c, d) in zip(coo.row + 1, coo.col + 1, coo.data):
                    data = '%i %i ' % (r, c) + template % d
                    stream.write(data.encode('latin1'))
            elif field == self.FIELD_COMPLEX:
                for (r, c, d) in zip(coo.row + 1, coo.col + 1, coo.data):
                    data = '%i %i ' % (r, c) + template % (d.real, d.imag)
                    stream.write(data.encode('latin1'))
            else:
                raise TypeError('Unknown field type %s' % field)

def _is_fromfile_compatible(stream):
    if False:
        i = 10
        return i + 15
    "\n    Check whether `stream` is compatible with numpy.fromfile.\n\n    Passing a gzipped file object to ``fromfile/fromstring`` doesn't work with\n    Python 3.\n    "
    bad_cls = []
    try:
        import gzip
        bad_cls.append(gzip.GzipFile)
    except ImportError:
        pass
    try:
        import bz2
        bad_cls.append(bz2.BZ2File)
    except ImportError:
        pass
    bad_cls = tuple(bad_cls)
    return not isinstance(stream, bad_cls)