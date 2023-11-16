"""
Module for reading and writing matlab (TM) .mat files
"""
from contextlib import contextmanager
from ._miobase import _get_matfile_version, docfiller
from ._mio4 import MatFile4Reader, MatFile4Writer
from ._mio5 import MatFile5Reader, MatFile5Writer
__all__ = ['mat_reader_factory', 'loadmat', 'savemat', 'whosmat']

@contextmanager
def _open_file_context(file_like, appendmat, mode='rb'):
    if False:
        i = 10
        return i + 15
    (f, opened) = _open_file(file_like, appendmat, mode)
    try:
        yield f
    finally:
        if opened:
            f.close()

def _open_file(file_like, appendmat, mode='rb'):
    if False:
        while True:
            i = 10
    "\n    Open `file_like` and return as file-like object. First, check if object is\n    already file-like; if so, return it as-is. Otherwise, try to pass it\n    to open(). If that fails, and `file_like` is a string, and `appendmat` is true,\n    append '.mat' and try again.\n    "
    reqs = {'read'} if set(mode) & set('r+') else set()
    if set(mode) & set('wax+'):
        reqs.add('write')
    if reqs.issubset(dir(file_like)):
        return (file_like, False)
    try:
        return (open(file_like, mode), True)
    except OSError as e:
        if isinstance(file_like, str):
            if appendmat and (not file_like.endswith('.mat')):
                file_like += '.mat'
            return (open(file_like, mode), True)
        else:
            raise OSError('Reader needs file name or open file-like object') from e

@docfiller
def mat_reader_factory(file_name, appendmat=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    '\n    Create reader for matlab .mat format files.\n\n    Parameters\n    ----------\n    %(file_arg)s\n    %(append_arg)s\n    %(load_args)s\n    %(struct_arg)s\n\n    Returns\n    -------\n    matreader : MatFileReader object\n       Initialized instance of MatFileReader class matching the mat file\n       type detected in `filename`.\n    file_opened : bool\n       Whether the file was opened by this routine.\n\n    '
    (byte_stream, file_opened) = _open_file(file_name, appendmat)
    (mjv, mnv) = _get_matfile_version(byte_stream)
    if mjv == 0:
        return (MatFile4Reader(byte_stream, **kwargs), file_opened)
    elif mjv == 1:
        return (MatFile5Reader(byte_stream, **kwargs), file_opened)
    elif mjv == 2:
        raise NotImplementedError('Please use HDF reader for matlab v7.3 files, e.g. h5py')
    else:
        raise TypeError('Did not recognize version %s' % mjv)

@docfiller
def loadmat(file_name, mdict=None, appendmat=True, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    "\n    Load MATLAB file.\n\n    Parameters\n    ----------\n    file_name : str\n       Name of the mat file (do not need .mat extension if\n       appendmat==True). Can also pass open file-like object.\n    mdict : dict, optional\n        Dictionary in which to insert matfile variables.\n    appendmat : bool, optional\n       True to append the .mat extension to the end of the given\n       filename, if not already present. Default is True.\n    byte_order : str or None, optional\n       None by default, implying byte order guessed from mat\n       file. Otherwise can be one of ('native', '=', 'little', '<',\n       'BIG', '>').\n    mat_dtype : bool, optional\n       If True, return arrays in same dtype as would be loaded into\n       MATLAB (instead of the dtype with which they are saved).\n    squeeze_me : bool, optional\n       Whether to squeeze unit matrix dimensions or not.\n    chars_as_strings : bool, optional\n       Whether to convert char arrays to string arrays.\n    matlab_compatible : bool, optional\n       Returns matrices as would be loaded by MATLAB (implies\n       squeeze_me=False, chars_as_strings=False, mat_dtype=True,\n       struct_as_record=True).\n    struct_as_record : bool, optional\n       Whether to load MATLAB structs as NumPy record arrays, or as\n       old-style NumPy arrays with dtype=object. Setting this flag to\n       False replicates the behavior of scipy version 0.7.x (returning\n       NumPy object arrays). The default setting is True, because it\n       allows easier round-trip load and save of MATLAB files.\n    verify_compressed_data_integrity : bool, optional\n        Whether the length of compressed sequences in the MATLAB file\n        should be checked, to ensure that they are not longer than we expect.\n        It is advisable to enable this (the default) because overlong\n        compressed sequences in MATLAB files generally indicate that the\n        files have experienced some sort of corruption.\n    variable_names : None or sequence\n        If None (the default) - read all variables in file. Otherwise,\n        `variable_names` should be a sequence of strings, giving names of the\n        MATLAB variables to read from the file. The reader will skip any\n        variable with a name not in this sequence, possibly saving some read\n        processing.\n    simplify_cells : False, optional\n        If True, return a simplified dict structure (which is useful if the mat\n        file contains cell arrays). Note that this only affects the structure\n        of the result and not its contents (which is identical for both output\n        structures). If True, this automatically sets `struct_as_record` to\n        False and `squeeze_me` to True, which is required to simplify cells.\n\n    Returns\n    -------\n    mat_dict : dict\n       dictionary with variable names as keys, and loaded matrices as\n       values.\n\n    Notes\n    -----\n    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.\n\n    You will need an HDF5 Python library to read MATLAB 7.3 format mat\n    files. Because SciPy does not supply one, we do not implement the\n    HDF5 / 7.3 interface here.\n\n    Examples\n    --------\n    >>> from os.path import dirname, join as pjoin\n    >>> import scipy.io as sio\n\n    Get the filename for an example .mat file from the tests/data directory.\n\n    >>> data_dir = pjoin(dirname(sio.__file__), 'matlab', 'tests', 'data')\n    >>> mat_fname = pjoin(data_dir, 'testdouble_7.4_GLNX86.mat')\n\n    Load the .mat file contents.\n\n    >>> mat_contents = sio.loadmat(mat_fname)\n\n    The result is a dictionary, one key/value pair for each variable:\n\n    >>> sorted(mat_contents.keys())\n    ['__globals__', '__header__', '__version__', 'testdouble']\n    >>> mat_contents['testdouble']\n    array([[0.        , 0.78539816, 1.57079633, 2.35619449, 3.14159265,\n            3.92699082, 4.71238898, 5.49778714, 6.28318531]])\n\n    By default SciPy reads MATLAB structs as structured NumPy arrays where the\n    dtype fields are of type `object` and the names correspond to the MATLAB\n    struct field names. This can be disabled by setting the optional argument\n    `struct_as_record=False`.\n\n    Get the filename for an example .mat file that contains a MATLAB struct\n    called `teststruct` and load the contents.\n\n    >>> matstruct_fname = pjoin(data_dir, 'teststruct_7.4_GLNX86.mat')\n    >>> matstruct_contents = sio.loadmat(matstruct_fname)\n    >>> teststruct = matstruct_contents['teststruct']\n    >>> teststruct.dtype\n    dtype([('stringfield', 'O'), ('doublefield', 'O'), ('complexfield', 'O')])\n\n    The size of the structured array is the size of the MATLAB struct, not the\n    number of elements in any particular field. The shape defaults to 2-D\n    unless the optional argument `squeeze_me=True`, in which case all length 1\n    dimensions are removed.\n\n    >>> teststruct.size\n    1\n    >>> teststruct.shape\n    (1, 1)\n\n    Get the 'stringfield' of the first element in the MATLAB struct.\n\n    >>> teststruct[0, 0]['stringfield']\n    array(['Rats live on no evil star.'],\n      dtype='<U26')\n\n    Get the first element of the 'doublefield'.\n\n    >>> teststruct['doublefield'][0, 0]\n    array([[ 1.41421356,  2.71828183,  3.14159265]])\n\n    Load the MATLAB struct, squeezing out length 1 dimensions, and get the item\n    from the 'complexfield'.\n\n    >>> matstruct_squeezed = sio.loadmat(matstruct_fname, squeeze_me=True)\n    >>> matstruct_squeezed['teststruct'].shape\n    ()\n    >>> matstruct_squeezed['teststruct']['complexfield'].shape\n    ()\n    >>> matstruct_squeezed['teststruct']['complexfield'].item()\n    array([ 1.41421356+1.41421356j,  2.71828183+2.71828183j,\n        3.14159265+3.14159265j])\n    "
    variable_names = kwargs.pop('variable_names', None)
    with _open_file_context(file_name, appendmat) as f:
        (MR, _) = mat_reader_factory(f, **kwargs)
        matfile_dict = MR.get_variables(variable_names)
    if mdict is not None:
        mdict.update(matfile_dict)
    else:
        mdict = matfile_dict
    return mdict

@docfiller
def savemat(file_name, mdict, appendmat=True, format='5', long_field_names=False, do_compression=False, oned_as='row'):
    if False:
        while True:
            i = 10
    '\n    Save a dictionary of names and arrays into a MATLAB-style .mat file.\n\n    This saves the array objects in the given dictionary to a MATLAB-\n    style .mat file.\n\n    Parameters\n    ----------\n    file_name : str or file-like object\n        Name of the .mat file (.mat extension not needed if ``appendmat ==\n        True``).\n        Can also pass open file_like object.\n    mdict : dict\n        Dictionary from which to save matfile variables.\n    appendmat : bool, optional\n        True (the default) to append the .mat extension to the end of the\n        given filename, if not already present.\n    format : {\'5\', \'4\'}, string, optional\n        \'5\' (the default) for MATLAB 5 and up (to 7.2),\n        \'4\' for MATLAB 4 .mat files.\n    long_field_names : bool, optional\n        False (the default) - maximum field name length in a structure is\n        31 characters which is the documented maximum length.\n        True - maximum field name length in a structure is 63 characters\n        which works for MATLAB 7.6+.\n    do_compression : bool, optional\n        Whether or not to compress matrices on write. Default is False.\n    oned_as : {\'row\', \'column\'}, optional\n        If \'column\', write 1-D NumPy arrays as column vectors.\n        If \'row\', write 1-D NumPy arrays as row vectors.\n\n    Examples\n    --------\n    >>> from scipy.io import savemat\n    >>> import numpy as np\n    >>> a = np.arange(20)\n    >>> mdic = {"a": a, "label": "experiment"}\n    >>> mdic\n    {\'a\': array([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16,\n        17, 18, 19]),\n    \'label\': \'experiment\'}\n    >>> savemat("matlab_matrix.mat", mdic)\n    '
    with _open_file_context(file_name, appendmat, 'wb') as file_stream:
        if format == '4':
            if long_field_names:
                raise ValueError('Long field names are not available for version 4 files')
            MW = MatFile4Writer(file_stream, oned_as)
        elif format == '5':
            MW = MatFile5Writer(file_stream, do_compression=do_compression, unicode_strings=True, long_field_names=long_field_names, oned_as=oned_as)
        else:
            raise ValueError("Format should be '4' or '5'")
        MW.put_variables(mdict)

@docfiller
def whosmat(file_name, appendmat=True, **kwargs):
    if False:
        i = 10
        return i + 15
    "\n    List variables inside a MATLAB file.\n\n    Parameters\n    ----------\n    %(file_arg)s\n    %(append_arg)s\n    %(load_args)s\n    %(struct_arg)s\n\n    Returns\n    -------\n    variables : list of tuples\n        A list of tuples, where each tuple holds the matrix name (a string),\n        its shape (tuple of ints), and its data class (a string).\n        Possible data classes are: int8, uint8, int16, uint16, int32, uint32,\n        int64, uint64, single, double, cell, struct, object, char, sparse,\n        function, opaque, logical, unknown.\n\n    Notes\n    -----\n    v4 (Level 1.0), v6 and v7 to 7.2 matfiles are supported.\n\n    You will need an HDF5 python library to read matlab 7.3 format mat\n    files (e.g. h5py). Because SciPy does not supply one, we do not implement the\n    HDF5 / 7.3 interface here.\n\n    .. versionadded:: 0.12.0\n\n    Examples\n    --------\n    >>> from io import BytesIO\n    >>> import numpy as np\n    >>> from scipy.io import savemat, whosmat\n\n    Create some arrays, and use `savemat` to write them to a ``BytesIO``\n    instance.\n\n    >>> a = np.array([[10, 20, 30], [11, 21, 31]], dtype=np.int32)\n    >>> b = np.geomspace(1, 10, 5)\n    >>> f = BytesIO()\n    >>> savemat(f, {'a': a, 'b': b})\n\n    Use `whosmat` to inspect ``f``.  Each tuple in the output list gives\n    the name, shape and data type of the array in ``f``.\n\n    >>> whosmat(f)\n    [('a', (2, 3), 'int32'), ('b', (1, 5), 'double')]\n\n    "
    with _open_file_context(file_name, appendmat) as f:
        (ML, file_opened) = mat_reader_factory(f, **kwargs)
        variables = ML.list_variables()
    return variables