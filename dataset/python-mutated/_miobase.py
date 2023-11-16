"""
Base classes for MATLAB file stream reading.

MATLAB is a registered trademark of the Mathworks inc.
"""
import numpy as np
from scipy._lib import doccer
from . import _byteordercodes as boc
__all__ = ['MatFileReader', 'MatReadError', 'MatReadWarning', 'MatVarReader', 'MatWriteError', 'arr_dtype_number', 'arr_to_chars', 'convert_dtypes', 'doc_dict', 'docfiller', 'get_matfile_version', 'matdims', 'read_dtype']

class MatReadError(Exception):
    """Exception indicating a read issue."""

class MatWriteError(Exception):
    """Exception indicating a write issue."""

class MatReadWarning(UserWarning):
    """Warning class for read issues."""
doc_dict = {'file_arg': 'file_name : str\n   Name of the mat file (do not need .mat extension if\n   appendmat==True) Can also pass open file-like object.', 'append_arg': 'appendmat : bool, optional\n   True to append the .mat extension to the end of the given\n   filename, if not already present. Default is True.', 'load_args': "byte_order : str or None, optional\n   None by default, implying byte order guessed from mat\n   file. Otherwise can be one of ('native', '=', 'little', '<',\n   'BIG', '>').\nmat_dtype : bool, optional\n   If True, return arrays in same dtype as would be loaded into\n   MATLAB (instead of the dtype with which they are saved).\nsqueeze_me : bool, optional\n   Whether to squeeze unit matrix dimensions or not.\nchars_as_strings : bool, optional\n   Whether to convert char arrays to string arrays.\nmatlab_compatible : bool, optional\n   Returns matrices as would be loaded by MATLAB (implies\n   squeeze_me=False, chars_as_strings=False, mat_dtype=True,\n   struct_as_record=True).", 'struct_arg': 'struct_as_record : bool, optional\n   Whether to load MATLAB structs as NumPy record arrays, or as\n   old-style NumPy arrays with dtype=object. Setting this flag to\n   False replicates the behavior of SciPy version 0.7.x (returning\n   numpy object arrays). The default setting is True, because it\n   allows easier round-trip load and save of MATLAB files.', 'matstream_arg': 'mat_stream : file-like\n   Object with file API, open for reading.', 'long_fields': 'long_field_names : bool, optional\n   * False - maximum field name length in a structure is 31 characters\n     which is the documented maximum length. This is the default.\n   * True - maximum field name length in a structure is 63 characters\n     which works for MATLAB 7.6', 'do_compression': 'do_compression : bool, optional\n   Whether to compress matrices on write. Default is False.', 'oned_as': "oned_as : {'row', 'column'}, optional\n   If 'column', write 1-D NumPy arrays as column vectors.\n   If 'row', write 1D NumPy arrays as row vectors.", 'unicode_strings': 'unicode_strings : bool, optional\n   If True, write strings as Unicode, else MATLAB usual encoding.'}
docfiller = doccer.filldoc(doc_dict)
'\n\n Note on architecture\n======================\n\nThere are three sets of parameters relevant for reading files. The\nfirst are *file read parameters* - containing options that are common\nfor reading the whole file, and therefore every variable within that\nfile. At the moment these are:\n\n* mat_stream\n* dtypes (derived from byte code)\n* byte_order\n* chars_as_strings\n* squeeze_me\n* struct_as_record (MATLAB 5 files)\n* class_dtypes (derived from order code, MATLAB 5 files)\n* codecs (MATLAB 5 files)\n* uint16_codec (MATLAB 5 files)\n\nAnother set of parameters are those that apply only to the current\nvariable being read - the *header*:\n\n* header related variables (different for v4 and v5 mat files)\n* is_complex\n* mclass\n* var_stream\n\nWith the header, we need ``next_position`` to tell us where the next\nvariable in the stream is.\n\nThen, for each element in a matrix, there can be *element read\nparameters*. An element is, for example, one element in a MATLAB cell\narray. At the moment, these are:\n\n* mat_dtype\n\nThe file-reading object contains the *file read parameters*. The\n*header* is passed around as a data object, or may be read and discarded\nin a single function. The *element read parameters* - the mat_dtype in\nthis instance, is passed into a general post-processing function - see\n``mio_utils`` for details.\n'

def convert_dtypes(dtype_template, order_code):
    if False:
        return 10
    ' Convert dtypes in mapping to given order\n\n    Parameters\n    ----------\n    dtype_template : mapping\n       mapping with values returning numpy dtype from ``np.dtype(val)``\n    order_code : str\n       an order code suitable for using in ``dtype.newbyteorder()``\n\n    Returns\n    -------\n    dtypes : mapping\n       mapping where values have been replaced by\n       ``np.dtype(val).newbyteorder(order_code)``\n\n    '
    dtypes = dtype_template.copy()
    for k in dtypes:
        dtypes[k] = np.dtype(dtypes[k]).newbyteorder(order_code)
    return dtypes

def read_dtype(mat_stream, a_dtype):
    if False:
        i = 10
        return i + 15
    '\n    Generic get of byte stream data of known type\n\n    Parameters\n    ----------\n    mat_stream : file_like object\n        MATLAB (tm) mat file stream\n    a_dtype : dtype\n        dtype of array to read. `a_dtype` is assumed to be correct\n        endianness.\n\n    Returns\n    -------\n    arr : ndarray\n        Array of dtype `a_dtype` read from stream.\n\n    '
    num_bytes = a_dtype.itemsize
    arr = np.ndarray(shape=(), dtype=a_dtype, buffer=mat_stream.read(num_bytes), order='F')
    return arr

def matfile_version(file_name, *, appendmat=True):
    if False:
        i = 10
        return i + 15
    '\n    Return major, minor tuple depending on apparent mat file type\n\n    Where:\n\n     #. 0,x -> version 4 format mat files\n     #. 1,x -> version 5 format mat files\n     #. 2,x -> version 7.3 format mat files (HDF format)\n\n    Parameters\n    ----------\n    file_name : str\n       Name of the mat file (do not need .mat extension if\n       appendmat==True). Can also pass open file-like object.\n    appendmat : bool, optional\n       True to append the .mat extension to the end of the given\n       filename, if not already present. Default is True.\n\n    Returns\n    -------\n    major_version : {0, 1, 2}\n        major MATLAB File format version\n    minor_version : int\n        minor MATLAB file format version\n\n    Raises\n    ------\n    MatReadError\n        If the file is empty.\n    ValueError\n        The matfile version is unknown.\n\n    Notes\n    -----\n    Has the side effect of setting the file read pointer to 0\n    '
    from ._mio import _open_file_context
    with _open_file_context(file_name, appendmat=appendmat) as fileobj:
        return _get_matfile_version(fileobj)
get_matfile_version = matfile_version

def _get_matfile_version(fileobj):
    if False:
        return 10
    fileobj.seek(0)
    mopt_bytes = fileobj.read(4)
    if len(mopt_bytes) == 0:
        raise MatReadError('Mat file appears to be empty')
    mopt_ints = np.ndarray(shape=(4,), dtype=np.uint8, buffer=mopt_bytes)
    if 0 in mopt_ints:
        fileobj.seek(0)
        return (0, 0)
    fileobj.seek(124)
    tst_str = fileobj.read(4)
    fileobj.seek(0)
    maj_ind = int(tst_str[2] == b'I'[0])
    maj_val = int(tst_str[maj_ind])
    min_val = int(tst_str[1 - maj_ind])
    ret = (maj_val, min_val)
    if maj_val in (1, 2):
        return ret
    raise ValueError('Unknown mat file type, version %s, %s' % ret)

def matdims(arr, oned_as='column'):
    if False:
        while True:
            i = 10
    '\n    Determine equivalent MATLAB dimensions for given array\n\n    Parameters\n    ----------\n    arr : ndarray\n        Input array\n    oned_as : {\'column\', \'row\'}, optional\n        Whether 1-D arrays are returned as MATLAB row or column matrices.\n        Default is \'column\'.\n\n    Returns\n    -------\n    dims : tuple\n        Shape tuple, in the form MATLAB expects it.\n\n    Notes\n    -----\n    We had to decide what shape a 1 dimensional array would be by\n    default. ``np.atleast_2d`` thinks it is a row vector. The\n    default for a vector in MATLAB (e.g., ``>> 1:12``) is a row vector.\n\n    Versions of scipy up to and including 0.11 resulted (accidentally)\n    in 1-D arrays being read as column vectors. For the moment, we\n    maintain the same tradition here.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.io.matlab._miobase import matdims\n    >>> matdims(np.array(1)) # NumPy scalar\n    (1, 1)\n    >>> matdims(np.array([1])) # 1-D array, 1 element\n    (1, 1)\n    >>> matdims(np.array([1,2])) # 1-D array, 2 elements\n    (2, 1)\n    >>> matdims(np.array([[2],[3]])) # 2-D array, column vector\n    (2, 1)\n    >>> matdims(np.array([[2,3]])) # 2-D array, row vector\n    (1, 2)\n    >>> matdims(np.array([[[2,3]]])) # 3-D array, rowish vector\n    (1, 1, 2)\n    >>> matdims(np.array([])) # empty 1-D array\n    (0, 0)\n    >>> matdims(np.array([[]])) # empty 2-D array\n    (0, 0)\n    >>> matdims(np.array([[[]]])) # empty 3-D array\n    (0, 0, 0)\n\n    Optional argument flips 1-D shape behavior.\n\n    >>> matdims(np.array([1,2]), \'row\') # 1-D array, 2 elements\n    (1, 2)\n\n    The argument has to make sense though\n\n    >>> matdims(np.array([1,2]), \'bizarre\')\n    Traceback (most recent call last):\n       ...\n    ValueError: 1-D option "bizarre" is strange\n\n    '
    shape = arr.shape
    if shape == ():
        return (1, 1)
    if len(shape) == 1:
        if shape[0] == 0:
            return (0, 0)
        elif oned_as == 'column':
            return shape + (1,)
        elif oned_as == 'row':
            return (1,) + shape
        else:
            raise ValueError('1-D option "%s" is strange' % oned_as)
    return shape

class MatVarReader:
    """ Abstract class defining required interface for var readers"""

    def __init__(self, file_reader):
        if False:
            i = 10
            return i + 15
        pass

    def read_header(self):
        if False:
            while True:
                i = 10
        ' Returns header '
        pass

    def array_from_header(self, header):
        if False:
            while True:
                i = 10
        ' Reads array given header '
        pass

class MatFileReader:
    """ Base object for reading mat files

    To make this class functional, you will need to override the
    following methods:

    matrix_getter_factory   - gives object to fetch next matrix from stream
    guess_byte_order        - guesses file byte order from file
    """

    @docfiller
    def __init__(self, mat_stream, byte_order=None, mat_dtype=False, squeeze_me=False, chars_as_strings=True, matlab_compatible=False, struct_as_record=True, verify_compressed_data_integrity=True, simplify_cells=False):
        if False:
            i = 10
            return i + 15
        '\n        Initializer for mat file reader\n\n        mat_stream : file-like\n            object with file API, open for reading\n    %(load_args)s\n        '
        self.mat_stream = mat_stream
        self.dtypes = {}
        if not byte_order:
            byte_order = self.guess_byte_order()
        else:
            byte_order = boc.to_numpy_code(byte_order)
        self.byte_order = byte_order
        self.struct_as_record = struct_as_record
        if matlab_compatible:
            self.set_matlab_compatible()
        else:
            self.squeeze_me = squeeze_me
            self.chars_as_strings = chars_as_strings
            self.mat_dtype = mat_dtype
        self.verify_compressed_data_integrity = verify_compressed_data_integrity
        self.simplify_cells = simplify_cells
        if simplify_cells:
            self.squeeze_me = True
            self.struct_as_record = False

    def set_matlab_compatible(self):
        if False:
            i = 10
            return i + 15
        ' Sets options to return arrays as MATLAB loads them '
        self.mat_dtype = True
        self.squeeze_me = False
        self.chars_as_strings = False

    def guess_byte_order(self):
        if False:
            while True:
                i = 10
        ' As we do not know what file type we have, assume native '
        return boc.native_code

    def end_of_stream(self):
        if False:
            while True:
                i = 10
        b = self.mat_stream.read(1)
        curpos = self.mat_stream.tell()
        self.mat_stream.seek(curpos - 1)
        return len(b) == 0

def arr_dtype_number(arr, num):
    if False:
        while True:
            i = 10
    ' Return dtype for given number of items per element'
    return np.dtype(arr.dtype.str[:2] + str(num))

def arr_to_chars(arr):
    if False:
        while True:
            i = 10
    ' Convert string array to char array '
    dims = list(arr.shape)
    if not dims:
        dims = [1]
    dims.append(int(arr.dtype.str[2:]))
    arr = np.ndarray(shape=dims, dtype=arr_dtype_number(arr, 1), buffer=arr)
    empties = [arr == np.array('', dtype=arr.dtype)]
    if not np.any(empties):
        return arr
    arr = arr.copy()
    arr[tuple(empties)] = ' '
    return arr