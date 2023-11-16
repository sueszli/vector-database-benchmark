import numpy as np
import pytest
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable
from astropy.table.index import SlicedIndex
from astropy.time import Time
from astropy.utils.data_info import dtype_info_name
STRING_TYPE_NAMES = {(True, 'S'): 'bytes', (True, 'U'): 'str'}
DTYPE_TESTS = ((np.array(b'abcd').dtype, STRING_TYPE_NAMES[True, 'S'] + '4'), (np.array('abcd').dtype, STRING_TYPE_NAMES[True, 'U'] + '4'), ('S4', STRING_TYPE_NAMES[True, 'S'] + '4'), ('U4', STRING_TYPE_NAMES[True, 'U'] + '4'), (np.void, 'void'), (np.int32, 'int32'), (bool, 'bool'), (float, 'float64'), ('<f4', 'float32'), ('u8', 'uint64'), ('c16', 'complex128'), ('object', 'object'))

@pytest.mark.parametrize('input,output', DTYPE_TESTS)
def test_dtype_info_name(input, output):
    if False:
        while True:
            i = 10
    "\n    Test that dtype_info_name is giving the expected output\n\n    Here the available types::\n\n      'b' boolean\n      'i' (signed) integer\n      'u' unsigned integer\n      'f' floating-point\n      'c' complex-floating point\n      'O' (Python) objects\n      'S', 'a' (byte-)string\n      'U' Unicode\n      'V' raw data (void)\n    "
    assert dtype_info_name(input) == output

def test_info_no_copy_numpy():
    if False:
        while True:
            i = 10
    'Test that getting a single item from Table column object does not copy info.\n    See #10889.\n    '
    col = [1, 2]
    t = QTable([col], names=['col'])
    t.add_index('col')
    val = t['col'][0]
    assert isinstance(val, np.number)
    with pytest.raises(AttributeError):
        val.info
    val = t['col'][:]
    assert val.info.indices == []
cols = [[1, 2] * u.m, Time([1, 2], format='cxcsec')]

@pytest.mark.parametrize('col', cols)
def test_info_no_copy_mixin_with_index(col):
    if False:
        print('Hello World!')
    'Test that getting a single item from Table column object does not copy info.\n    See #10889.\n    '
    t = QTable([col], names=['col'])
    t.add_index('col')
    val = t['col'][0]
    assert 'info' not in val.__dict__
    assert val.info.indices == []
    val = t['col'][:]
    assert 'info' in val.__dict__
    assert val.info.indices == []
    val = t[:]['col']
    assert 'info' in val.__dict__
    assert isinstance(val.info.indices[0], SlicedIndex)

def test_info_no_copy_skycoord():
    if False:
        i = 10
        return i + 15
    'Test that getting a single item from Table SkyCoord column object does\n    not copy info.  Cannot create an index on a SkyCoord currently.\n    '
    col = (SkyCoord([1, 2], [1, 2], unit='deg'),)
    t = QTable([col], names=['col'])
    val = t['col'][0]
    assert 'info' not in val.__dict__
    assert val.info.indices == []
    val = t['col'][:]
    assert val.info.indices == []
    val = t[:]['col']
    assert val.info.indices == []