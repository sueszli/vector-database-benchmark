import os.path
import subprocess
from collections import defaultdict
from typing import Tuple, Optional
import numpy as np
import pandas as pd
from chardet.universaldetector import UniversalDetector
from Orange.data import is_discrete_values, MISSING_VALUES, Variable, DiscreteVariable, StringVariable, ContinuousVariable, TimeVariable, Table
from Orange.misc.collections import natural_sorted
__all__ = ['Compression', 'open_compressed', 'detect_encoding', 'isnastr', 'guess_data_type', 'sanitize_variable', 'update_origin']

class Compression:
    """Supported compression extensions"""
    GZIP = '.gz'
    BZIP2 = '.bz2'
    XZ = '.xz'
    all = (GZIP, BZIP2, XZ)

def open_compressed(filename, *args, _open=open, **kwargs):
    if False:
        return 10
    'Return seamlessly decompressed open file handle for `filename`'
    if isinstance(filename, str):
        if filename.endswith(Compression.GZIP):
            from gzip import open as _open
        elif filename.endswith(Compression.BZIP2):
            from bz2 import open as _open
        elif filename.endswith(Compression.XZ):
            from lzma import open as _open
        return _open(filename, *args, **kwargs)
    return filename

def detect_encoding(filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Detect encoding of `filename`, which can be a ``str`` filename, a\n    ``file``-like object, or ``bytes``.\n    '
    if isinstance(filename, str) and (not filename.endswith(Compression.all)):
        try:
            with subprocess.Popen(('file', '--brief', '--mime-encoding', filename), stdout=subprocess.PIPE) as proc:
                proc.wait()
                if proc.returncode == 0:
                    encoding = proc.stdout.read().strip()
                    if encoding in (b'utf-8', b'us-ascii', b'iso-8859-1', b'utf-7', b'utf-16le', b'utf-16be', b'ebcdic'):
                        return encoding.decode('us-ascii')
        except OSError:
            pass
    detector = UniversalDetector()
    MAX_BYTES = 4 * 1024 * 12

    def _from_file(f):
        if False:
            i = 10
            return i + 15
        detector.feed(f.read(MAX_BYTES))
        detector.close()
        return detector.result.get('encoding') if detector.result.get('confidence', 0) >= 0.85 else 'utf-8'
    if isinstance(filename, str):
        with open_compressed(filename, 'rb') as f:
            return _from_file(f)
    elif isinstance(filename, bytes):
        detector.feed(filename[:MAX_BYTES])
        detector.close()
        return detector.result.get('encoding')
    elif hasattr(filename, 'encoding'):
        return filename.encoding
    else:
        return _from_file(filename)
__isnastr = np.frompyfunc({v for v in MISSING_VALUES if isinstance(v, str)}.__contains__, 1, 1)

def isnastr(arr, out=None):
    if False:
        return 10
    '\n    Given an (object) array of string values, return a boolean mask array\n    that is True where the `arr` contains one of the string constants\n    considered as N/A.\n\n    Parameters\n    ----------\n    arr : np.ndarray\n        Input array of strings.\n    out : Optional[np.ndarray]\n        Optional output array of the same shape as arr\n\n    Returns\n    -------\n    mask : np.ndarray\n    '
    arr = np.asarray(arr)
    if out is None and arr.shape != ():
        out = np.empty_like(arr, dtype=bool)
    return __isnastr(arr, out=out, casting='unsafe')

def guess_data_type(orig_values, namask=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Use heuristics to guess data type.\n    '
    (valuemap, values) = (None, orig_values)
    is_discrete = is_discrete_values(orig_values)
    orig_values = np.asarray(orig_values, dtype=str)
    if namask is None:
        namask = isnastr(orig_values)
    if is_discrete:
        valuemap = natural_sorted(is_discrete)
        coltype = DiscreteVariable
    else:
        values = np.empty_like(orig_values, dtype=float)
        values[namask] = np.nan
        try:
            np.copyto(values, orig_values, where=~namask, casting='unsafe')
        except ValueError:
            values = orig_values
            coltype = StringVariable
        else:
            coltype = ContinuousVariable
    if coltype is not ContinuousVariable:
        tvar = TimeVariable('_')
        temp_values = np.empty_like(orig_values, dtype=float)
        try:
            temp_values[~namask] = [tvar.parse_exact_iso(i) for i in orig_values[~namask]]
        except ValueError:
            pass
        else:
            valuemap = None
            coltype = TimeVariable
            values = temp_values
    return (valuemap, values, coltype)

def sanitize_variable(valuemap, values, orig_values, coltype, coltype_kwargs, name=None):
    if False:
        while True:
            i = 10
    assert issubclass(coltype, Variable)

    def get_number_of_decimals(values):
        if False:
            for i in range(10):
                print('nop')
        len_ = len
        ndecimals = max((len_(value) - value.find('.') for value in values if '.' in value), default=1)
        return ndecimals - 1
    if issubclass(coltype, DiscreteVariable) and valuemap is not None:
        coltype_kwargs.update(values=valuemap)
    var = coltype.make(name, **coltype_kwargs)
    if isinstance(var, DiscreteVariable):
        mapping = defaultdict(lambda : np.nan, {val: i for (i, val) in enumerate(var.values)})
        mapping[''] = np.nan
        mapvalues_ = np.frompyfunc(mapping.__getitem__, 1, 1)

        def mapvalues(arr):
            if False:
                print('Hello World!')
            arr = np.asarray(arr, dtype=object)
            return mapvalues_(arr, out=np.empty_like(arr, dtype=float), casting='unsafe')
        values = mapvalues(orig_values)
    if coltype is StringVariable:
        values = orig_values
    if isinstance(var, ContinuousVariable) and var.adjust_decimals:
        ndecimals = get_number_of_decimals(orig_values)
        if var.adjust_decimals == 2 or ndecimals > var.number_of_decimals:
            var.number_of_decimals = ndecimals
            var.adjust_decimals = 1
    if isinstance(var, TimeVariable) or coltype is TimeVariable:
        _var = var if isinstance(var, TimeVariable) else TimeVariable('_')
        values = [_var.parse(i) for i in orig_values]
    return (values, var)

def _extract_new_origin(attr: Variable, table: Table, lookup_dirs: Tuple[str]) -> Optional[str]:
    if False:
        return 10
    if os.path.exists(attr.attributes['origin']):
        return attr.attributes['origin']
    dir_ = os.path.basename(os.path.normpath(attr.attributes['origin']))
    for ld in lookup_dirs:
        new_dir = os.path.join(ld, dir_)
        if os.path.isdir(new_dir):
            return new_dir
    for ld in lookup_dirs:
        if all((os.path.exists(os.path.join(ld, attr.str_val(v))) for v in table.get_column(attr) if v and (not pd.isna(v)))):
            return ld
    return None

def update_origin(table: Table, file_path: str):
    if False:
        while True:
            i = 10
    '\n    When a dataset with file paths in the column is moved to another computer,\n    the absolute path may not be correct. This function updates the path for all\n    columns with an "origin" attribute.\n\n    The process consists of two steps. First, we identify directories to search\n    for files, and in the second step, we check if paths exist.\n\n    Lookup directories:\n    1. The directory where the file from file_path is placed\n    2. The parent directory of 1. The situation when the user places dataset\n       file in the directory with files (for example, workflow in a directory\n       with images)\n\n    Possible situations for file search:\n    1. The last directory of origin (basedir) is in one of the lookup directories\n    2. Origin doesn\'t exist in any lookup directories, but paths in a column can\n       be found in one of the lookup directories. This is usually a situation\n       when paths in a column are complex (e.g. a/b/c/d/file.txt).\n\n    Note: This function updates the existing table\n\n    Parameters\n    ----------\n    table\n        Orange Table to be updated if origin exits in any column\n    file_path\n        Path of the loaded dataset for reference. Only paths inside datasets\n        directory or its parent directory will be considered for new origin.\n    '
    file_dir = os.path.dirname(file_path)
    parent_dir = os.path.dirname(file_dir)
    lookup_dirs = tuple({file_dir: 0, parent_dir: 0})
    for attr in table.domain.metas:
        if 'origin' in attr.attributes and (attr.is_string or attr.is_discrete):
            new_orig = _extract_new_origin(attr, table, lookup_dirs)
            if new_orig:
                attr.attributes['origin'] = new_orig