""" Classes for read / write of matlab (TM) 4 files
"""
import sys
import warnings
import numpy as np
import scipy.sparse
from ._miobase import MatFileReader, docfiller, matdims, read_dtype, convert_dtypes, arr_to_chars, arr_dtype_number
from ._mio_utils import squeeze_element, chars_to_strings
from functools import reduce
__all__ = ['MatFile4Reader', 'MatFile4Writer', 'SYS_LITTLE_ENDIAN', 'VarHeader4', 'VarReader4', 'VarWriter4', 'arr_to_2d', 'mclass_info', 'mdtypes_template', 'miDOUBLE', 'miINT16', 'miINT32', 'miSINGLE', 'miUINT16', 'miUINT8', 'mxCHAR_CLASS', 'mxFULL_CLASS', 'mxSPARSE_CLASS', 'np_to_mtypes', 'order_codes']
SYS_LITTLE_ENDIAN = sys.byteorder == 'little'
miDOUBLE = 0
miSINGLE = 1
miINT32 = 2
miINT16 = 3
miUINT16 = 4
miUINT8 = 5
mdtypes_template = {miDOUBLE: 'f8', miSINGLE: 'f4', miINT32: 'i4', miINT16: 'i2', miUINT16: 'u2', miUINT8: 'u1', 'header': [('mopt', 'i4'), ('mrows', 'i4'), ('ncols', 'i4'), ('imagf', 'i4'), ('namlen', 'i4')], 'U1': 'U1'}
np_to_mtypes = {'f8': miDOUBLE, 'c32': miDOUBLE, 'c24': miDOUBLE, 'c16': miDOUBLE, 'f4': miSINGLE, 'c8': miSINGLE, 'i4': miINT32, 'i2': miINT16, 'u2': miUINT16, 'u1': miUINT8, 'S1': miUINT8}
mxFULL_CLASS = 0
mxCHAR_CLASS = 1
mxSPARSE_CLASS = 2
order_codes = {0: '<', 1: '>', 2: 'VAX D-float', 3: 'VAX G-float', 4: 'Cray'}
mclass_info = {mxFULL_CLASS: 'double', mxCHAR_CLASS: 'char', mxSPARSE_CLASS: 'sparse'}

class VarHeader4:
    is_logical = False
    is_global = False

    def __init__(self, name, dtype, mclass, dims, is_complex):
        if False:
            return 10
        self.name = name
        self.dtype = dtype
        self.mclass = mclass
        self.dims = dims
        self.is_complex = is_complex

class VarReader4:
    """ Class to read matlab 4 variables """

    def __init__(self, file_reader):
        if False:
            print('Hello World!')
        self.file_reader = file_reader
        self.mat_stream = file_reader.mat_stream
        self.dtypes = file_reader.dtypes
        self.chars_as_strings = file_reader.chars_as_strings
        self.squeeze_me = file_reader.squeeze_me

    def read_header(self):
        if False:
            return 10
        ' Read and return header for variable '
        data = read_dtype(self.mat_stream, self.dtypes['header'])
        name = self.mat_stream.read(int(data['namlen'])).strip(b'\x00')
        if data['mopt'] < 0 or data['mopt'] > 5000:
            raise ValueError('Mat 4 mopt wrong format, byteswapping problem?')
        (M, rest) = divmod(data['mopt'], 1000)
        if M not in (0, 1):
            warnings.warn("We do not support byte ordering '%s'; returned data may be corrupt" % order_codes[M], UserWarning)
        (O, rest) = divmod(rest, 100)
        if O != 0:
            raise ValueError('O in MOPT integer should be 0, wrong format?')
        (P, rest) = divmod(rest, 10)
        T = rest
        dims = (data['mrows'], data['ncols'])
        is_complex = data['imagf'] == 1
        dtype = self.dtypes[P]
        return VarHeader4(name, dtype, T, dims, is_complex)

    def array_from_header(self, hdr, process=True):
        if False:
            i = 10
            return i + 15
        mclass = hdr.mclass
        if mclass == mxFULL_CLASS:
            arr = self.read_full_array(hdr)
        elif mclass == mxCHAR_CLASS:
            arr = self.read_char_array(hdr)
            if process and self.chars_as_strings:
                arr = chars_to_strings(arr)
        elif mclass == mxSPARSE_CLASS:
            return self.read_sparse_array(hdr)
        else:
            raise TypeError('No reader for class code %s' % mclass)
        if process and self.squeeze_me:
            return squeeze_element(arr)
        return arr

    def read_sub_array(self, hdr, copy=True):
        if False:
            i = 10
            return i + 15
        ' Mat4 read using header `hdr` dtype and dims\n\n        Parameters\n        ----------\n        hdr : object\n           object with attributes ``dtype``, ``dims``. dtype is assumed to be\n           the correct endianness\n        copy : bool, optional\n           copies array before return if True (default True)\n           (buffer is usually read only)\n\n        Returns\n        -------\n        arr : ndarray\n            of dtype given by `hdr` ``dtype`` and shape given by `hdr` ``dims``\n        '
        dt = hdr.dtype
        dims = hdr.dims
        num_bytes = dt.itemsize
        for d in dims:
            num_bytes *= d
        buffer = self.mat_stream.read(int(num_bytes))
        if len(buffer) != num_bytes:
            raise ValueError("Not enough bytes to read matrix '%s'; is this a badly-formed file? Consider listing matrices with `whosmat` and loading named matrices with `variable_names` kwarg to `loadmat`" % hdr.name)
        arr = np.ndarray(shape=dims, dtype=dt, buffer=buffer, order='F')
        if copy:
            arr = arr.copy()
        return arr

    def read_full_array(self, hdr):
        if False:
            while True:
                i = 10
        ' Full (rather than sparse) matrix getter\n\n        Read matrix (array) can be real or complex\n\n        Parameters\n        ----------\n        hdr : ``VarHeader4`` instance\n\n        Returns\n        -------\n        arr : ndarray\n            complex array if ``hdr.is_complex`` is True, otherwise a real\n            numeric array\n        '
        if hdr.is_complex:
            res = self.read_sub_array(hdr, copy=False)
            res_j = self.read_sub_array(hdr, copy=False)
            return res + res_j * 1j
        return self.read_sub_array(hdr)

    def read_char_array(self, hdr):
        if False:
            for i in range(10):
                print('nop')
        " latin-1 text matrix (char matrix) reader\n\n        Parameters\n        ----------\n        hdr : ``VarHeader4`` instance\n\n        Returns\n        -------\n        arr : ndarray\n            with dtype 'U1', shape given by `hdr` ``dims``\n        "
        arr = self.read_sub_array(hdr).astype(np.uint8)
        S = arr.tobytes().decode('latin-1')
        return np.ndarray(shape=hdr.dims, dtype=np.dtype('U1'), buffer=np.array(S)).copy()

    def read_sparse_array(self, hdr):
        if False:
            i = 10
            return i + 15
        ' Read and return sparse matrix type\n\n        Parameters\n        ----------\n        hdr : ``VarHeader4`` instance\n\n        Returns\n        -------\n        arr : ``scipy.sparse.coo_matrix``\n            with dtype ``float`` and shape read from the sparse matrix data\n\n        Notes\n        -----\n        MATLAB 4 real sparse arrays are saved in a N+1 by 3 array format, where\n        N is the number of non-zero values. Column 1 values [0:N] are the\n        (1-based) row indices of the each non-zero value, column 2 [0:N] are the\n        column indices, column 3 [0:N] are the (real) values. The last values\n        [-1,0:2] of the rows, column indices are shape[0] and shape[1]\n        respectively of the output matrix. The last value for the values column\n        is a padding 0. mrows and ncols values from the header give the shape of\n        the stored matrix, here [N+1, 3]. Complex data are saved as a 4 column\n        matrix, where the fourth column contains the imaginary component; the\n        last value is again 0. Complex sparse data do *not* have the header\n        ``imagf`` field set to True; the fact that the data are complex is only\n        detectable because there are 4 storage columns.\n        '
        res = self.read_sub_array(hdr)
        tmp = res[:-1, :]
        dims = (int(res[-1, 0]), int(res[-1, 1]))
        I = np.ascontiguousarray(tmp[:, 0], dtype='intc')
        J = np.ascontiguousarray(tmp[:, 1], dtype='intc')
        I -= 1
        J -= 1
        if res.shape[1] == 3:
            V = np.ascontiguousarray(tmp[:, 2], dtype='float')
        else:
            V = np.ascontiguousarray(tmp[:, 2], dtype='complex')
            V.imag = tmp[:, 3]
        return scipy.sparse.coo_matrix((V, (I, J)), dims)

    def shape_from_header(self, hdr):
        if False:
            print('Hello World!')
        'Read the shape of the array described by the header.\n        The file position after this call is unspecified.\n        '
        mclass = hdr.mclass
        if mclass == mxFULL_CLASS:
            shape = tuple(map(int, hdr.dims))
        elif mclass == mxCHAR_CLASS:
            shape = tuple(map(int, hdr.dims))
            if self.chars_as_strings:
                shape = shape[:-1]
        elif mclass == mxSPARSE_CLASS:
            dt = hdr.dtype
            dims = hdr.dims
            if not (len(dims) == 2 and dims[0] >= 1 and (dims[1] >= 1)):
                return ()
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            rows = np.ndarray(shape=(), dtype=dt, buffer=self.mat_stream.read(dt.itemsize))
            self.mat_stream.seek(dt.itemsize * (dims[0] - 1), 1)
            cols = np.ndarray(shape=(), dtype=dt, buffer=self.mat_stream.read(dt.itemsize))
            shape = (int(rows), int(cols))
        else:
            raise TypeError('No reader for class code %s' % mclass)
        if self.squeeze_me:
            shape = tuple([x for x in shape if x != 1])
        return shape

class MatFile4Reader(MatFileReader):
    """ Reader for Mat4 files """

    @docfiller
    def __init__(self, mat_stream, *args, **kwargs):
        if False:
            print('Hello World!')
        ' Initialize matlab 4 file reader\n\n    %(matstream_arg)s\n    %(load_args)s\n        '
        super().__init__(mat_stream, *args, **kwargs)
        self._matrix_reader = None

    def guess_byte_order(self):
        if False:
            while True:
                i = 10
        self.mat_stream.seek(0)
        mopt = read_dtype(self.mat_stream, np.dtype('i4'))
        self.mat_stream.seek(0)
        if mopt == 0:
            return '<'
        if mopt < 0 or mopt > 5000:
            return SYS_LITTLE_ENDIAN and '>' or '<'
        return SYS_LITTLE_ENDIAN and '<' or '>'

    def initialize_read(self):
        if False:
            print('Hello World!')
        ' Run when beginning read of variables\n\n        Sets up readers from parameters in `self`\n        '
        self.dtypes = convert_dtypes(mdtypes_template, self.byte_order)
        self._matrix_reader = VarReader4(self)

    def read_var_header(self):
        if False:
            while True:
                i = 10
        ' Read and return header, next position\n\n        Parameters\n        ----------\n        None\n\n        Returns\n        -------\n        header : object\n           object that can be passed to self.read_var_array, and that\n           has attributes ``name`` and ``is_global``\n        next_position : int\n           position in stream of next variable\n        '
        hdr = self._matrix_reader.read_header()
        n = reduce(lambda x, y: x * y, hdr.dims, 1)
        remaining_bytes = hdr.dtype.itemsize * n
        if hdr.is_complex and (not hdr.mclass == mxSPARSE_CLASS):
            remaining_bytes *= 2
        next_position = self.mat_stream.tell() + remaining_bytes
        return (hdr, next_position)

    def read_var_array(self, header, process=True):
        if False:
            return 10
        ' Read array, given `header`\n\n        Parameters\n        ----------\n        header : header object\n           object with fields defining variable header\n        process : {True, False}, optional\n           If True, apply recursive post-processing during loading of array.\n\n        Returns\n        -------\n        arr : array\n           array with post-processing applied or not according to\n           `process`.\n        '
        return self._matrix_reader.array_from_header(header, process)

    def get_variables(self, variable_names=None):
        if False:
            print('Hello World!')
        ' get variables from stream as dictionary\n\n        Parameters\n        ----------\n        variable_names : None or str or sequence of str, optional\n            variable name, or sequence of variable names to get from Mat file /\n            file stream. If None, then get all variables in file.\n        '
        if isinstance(variable_names, str):
            variable_names = [variable_names]
        elif variable_names is not None:
            variable_names = list(variable_names)
        self.mat_stream.seek(0)
        self.initialize_read()
        mdict = {}
        while not self.end_of_stream():
            (hdr, next_position) = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            if variable_names is not None and name not in variable_names:
                self.mat_stream.seek(next_position)
                continue
            mdict[name] = self.read_var_array(hdr)
            self.mat_stream.seek(next_position)
            if variable_names is not None:
                variable_names.remove(name)
                if len(variable_names) == 0:
                    break
        return mdict

    def list_variables(self):
        if False:
            print('Hello World!')
        ' list variables from stream '
        self.mat_stream.seek(0)
        self.initialize_read()
        vars = []
        while not self.end_of_stream():
            (hdr, next_position) = self.read_var_header()
            name = 'None' if hdr.name is None else hdr.name.decode('latin1')
            shape = self._matrix_reader.shape_from_header(hdr)
            info = mclass_info.get(hdr.mclass, 'unknown')
            vars.append((name, shape, info))
            self.mat_stream.seek(next_position)
        return vars

def arr_to_2d(arr, oned_as='row'):
    if False:
        i = 10
        return i + 15
    " Make ``arr`` exactly two dimensional\n\n    If `arr` has more than 2 dimensions, raise a ValueError\n\n    Parameters\n    ----------\n    arr : array\n    oned_as : {'row', 'column'}, optional\n       Whether to reshape 1-D vectors as row vectors or column vectors.\n       See documentation for ``matdims`` for more detail\n\n    Returns\n    -------\n    arr2d : array\n       2-D version of the array\n    "
    dims = matdims(arr, oned_as)
    if len(dims) > 2:
        raise ValueError('Matlab 4 files cannot save arrays with more than 2 dimensions')
    return arr.reshape(dims)

class VarWriter4:

    def __init__(self, file_writer):
        if False:
            i = 10
            return i + 15
        self.file_stream = file_writer.file_stream
        self.oned_as = file_writer.oned_as

    def write_bytes(self, arr):
        if False:
            return 10
        self.file_stream.write(arr.tobytes(order='F'))

    def write_string(self, s):
        if False:
            i = 10
            return i + 15
        self.file_stream.write(s)

    def write_header(self, name, shape, P=miDOUBLE, T=mxFULL_CLASS, imagf=0):
        if False:
            return 10
        ' Write header for given data options\n\n        Parameters\n        ----------\n        name : str\n            name of variable\n        shape : sequence\n            Shape of array as it will be read in matlab\n        P : int, optional\n            code for mat4 data type, one of ``miDOUBLE, miSINGLE, miINT32,\n            miINT16, miUINT16, miUINT8``\n        T : int, optional\n            code for mat4 matrix class, one of ``mxFULL_CLASS, mxCHAR_CLASS,\n            mxSPARSE_CLASS``\n        imagf : int, optional\n            flag indicating complex\n        '
        header = np.empty((), mdtypes_template['header'])
        M = not SYS_LITTLE_ENDIAN
        O = 0
        header['mopt'] = M * 1000 + O * 100 + P * 10 + T
        header['mrows'] = shape[0]
        header['ncols'] = shape[1]
        header['imagf'] = imagf
        header['namlen'] = len(name) + 1
        self.write_bytes(header)
        data = name + '\x00'
        self.write_string(data.encode('latin1'))

    def write(self, arr, name):
        if False:
            print('Hello World!')
        ' Write matrix `arr`, with name `name`\n\n        Parameters\n        ----------\n        arr : array_like\n           array to write\n        name : str\n           name in matlab workspace\n        '
        if scipy.sparse.issparse(arr):
            self.write_sparse(arr, name)
            return
        arr = np.asarray(arr)
        dt = arr.dtype
        if not dt.isnative:
            arr = arr.astype(dt.newbyteorder('='))
        dtt = dt.type
        if dtt is np.object_:
            raise TypeError('Cannot save object arrays in Mat4')
        elif dtt is np.void:
            raise TypeError('Cannot save void type arrays')
        elif dtt in (np.str_, np.bytes_):
            self.write_char(arr, name)
            return
        self.write_numeric(arr, name)

    def write_numeric(self, arr, name):
        if False:
            return 10
        arr = arr_to_2d(arr, self.oned_as)
        imagf = arr.dtype.kind == 'c'
        try:
            P = np_to_mtypes[arr.dtype.str[1:]]
        except KeyError:
            if imagf:
                arr = arr.astype('c128')
            else:
                arr = arr.astype('f8')
            P = miDOUBLE
        self.write_header(name, arr.shape, P=P, T=mxFULL_CLASS, imagf=imagf)
        if imagf:
            self.write_bytes(arr.real)
            self.write_bytes(arr.imag)
        else:
            self.write_bytes(arr)

    def write_char(self, arr, name):
        if False:
            for i in range(10):
                print('nop')
        arr = arr_to_chars(arr)
        arr = arr_to_2d(arr, self.oned_as)
        dims = arr.shape
        self.write_header(name, dims, P=miUINT8, T=mxCHAR_CLASS)
        if arr.dtype.kind == 'U':
            n_chars = np.prod(dims)
            st_arr = np.ndarray(shape=(), dtype=arr_dtype_number(arr, n_chars), buffer=arr)
            st = st_arr.item().encode('latin-1')
            arr = np.ndarray(shape=dims, dtype='S1', buffer=st)
        self.write_bytes(arr)

    def write_sparse(self, arr, name):
        if False:
            return 10
        ' Sparse matrices are 2-D\n\n        See docstring for VarReader4.read_sparse_array\n        '
        A = arr.tocoo()
        imagf = A.dtype.kind == 'c'
        ijv = np.zeros((A.nnz + 1, 3 + imagf), dtype='f8')
        ijv[:-1, 0] = A.row
        ijv[:-1, 1] = A.col
        ijv[:-1, 0:2] += 1
        if imagf:
            ijv[:-1, 2] = A.data.real
            ijv[:-1, 3] = A.data.imag
        else:
            ijv[:-1, 2] = A.data
        ijv[-1, 0:2] = A.shape
        self.write_header(name, ijv.shape, P=miDOUBLE, T=mxSPARSE_CLASS)
        self.write_bytes(ijv)

class MatFile4Writer:
    """ Class for writing matlab 4 format files """

    def __init__(self, file_stream, oned_as=None):
        if False:
            print('Hello World!')
        self.file_stream = file_stream
        if oned_as is None:
            oned_as = 'row'
        self.oned_as = oned_as
        self._matrix_writer = None

    def put_variables(self, mdict, write_header=None):
        if False:
            while True:
                i = 10
        ' Write variables in `mdict` to stream\n\n        Parameters\n        ----------\n        mdict : mapping\n           mapping with method ``items`` return name, contents pairs\n           where ``name`` which will appeak in the matlab workspace in\n           file load, and ``contents`` is something writeable to a\n           matlab file, such as a NumPy array.\n        write_header : {None, True, False}\n           If True, then write the matlab file header before writing the\n           variables. If None (the default) then write the file header\n           if we are at position 0 in the stream. By setting False\n           here, and setting the stream position to the end of the file,\n           you can append variables to a matlab file\n        '
        self._matrix_writer = VarWriter4(self)
        for (name, var) in mdict.items():
            self._matrix_writer.write(var, name)