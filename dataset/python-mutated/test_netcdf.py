""" Tests for netcdf """
import os
from os.path import join as pjoin, dirname
import shutil
import tempfile
import warnings
from io import BytesIO
from glob import glob
from contextlib import contextmanager
import numpy as np
from numpy.testing import assert_, assert_allclose, assert_equal, break_cycles, suppress_warnings, IS_PYPY
from pytest import raises as assert_raises
from scipy.io import netcdf_file
from scipy._lib._tmpdirs import in_tempdir
TEST_DATA_PATH = pjoin(dirname(__file__), 'data')
N_EG_ELS = 11
VARTYPE_EG = 'b'

@contextmanager
def make_simple(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    f = netcdf_file(*args, **kwargs)
    f.history = 'Created for a test'
    f.createDimension('time', N_EG_ELS)
    time = f.createVariable('time', VARTYPE_EG, ('time',))
    time[:] = np.arange(N_EG_ELS)
    time.units = 'days since 2008-01-01'
    f.flush()
    yield f
    f.close()

def check_simple(ncfileobj):
    if False:
        return 10
    'Example fileobj tests '
    assert_equal(ncfileobj.history, b'Created for a test')
    time = ncfileobj.variables['time']
    assert_equal(time.units, b'days since 2008-01-01')
    assert_equal(time.shape, (N_EG_ELS,))
    assert_equal(time[-1], N_EG_ELS - 1)

def assert_mask_matches(arr, expected_mask):
    if False:
        return 10
    "\n    Asserts that the mask of arr is effectively the same as expected_mask.\n\n    In contrast to numpy.ma.testutils.assert_mask_equal, this function allows\n    testing the 'mask' of a standard numpy array (the mask in this case is treated\n    as all False).\n\n    Parameters\n    ----------\n    arr : ndarray or MaskedArray\n        Array to test.\n    expected_mask : array_like of booleans\n        A list giving the expected mask.\n    "
    mask = np.ma.getmaskarray(arr)
    assert_equal(mask, expected_mask)

def test_read_write_files():
    if False:
        i = 10
        return i + 15
    cwd = os.getcwd()
    try:
        tmpdir = tempfile.mkdtemp()
        os.chdir(tmpdir)
        with make_simple('simple.nc', 'w') as f:
            pass
        with netcdf_file('simple.nc', 'a') as f:
            check_simple(f)
            f._attributes['appendRan'] = 1
        with netcdf_file('simple.nc') as f:
            assert_equal(f.use_mmap, not IS_PYPY)
            check_simple(f)
            assert_equal(f._attributes['appendRan'], 1)
        with netcdf_file('simple.nc', 'a') as f:
            assert_(not f.use_mmap)
            check_simple(f)
            assert_equal(f._attributes['appendRan'], 1)
        with netcdf_file('simple.nc', mmap=False) as f:
            assert_(not f.use_mmap)
            check_simple(f)
        with open('simple.nc', 'rb') as fobj:
            with netcdf_file(fobj) as f:
                assert_(not f.use_mmap)
                check_simple(f)
        with suppress_warnings() as sup:
            if IS_PYPY:
                sup.filter(RuntimeWarning, 'Cannot close a netcdf_file opened with mmap=True.*')
            with open('simple.nc', 'rb') as fobj:
                with netcdf_file(fobj, mmap=True) as f:
                    assert_(f.use_mmap)
                    check_simple(f)
        with open('simple.nc', 'r+b') as fobj:
            with netcdf_file(fobj, 'a') as f:
                assert_(not f.use_mmap)
                check_simple(f)
                f.createDimension('app_dim', 1)
                var = f.createVariable('app_var', 'i', ('app_dim',))
                var[:] = 42
        with netcdf_file('simple.nc') as f:
            check_simple(f)
            assert_equal(f.variables['app_var'][:], 42)
    finally:
        if IS_PYPY:
            break_cycles()
            break_cycles()
        os.chdir(cwd)
        shutil.rmtree(tmpdir)

def test_read_write_sio():
    if False:
        while True:
            i = 10
    eg_sio1 = BytesIO()
    with make_simple(eg_sio1, 'w'):
        str_val = eg_sio1.getvalue()
    eg_sio2 = BytesIO(str_val)
    with netcdf_file(eg_sio2) as f2:
        check_simple(f2)
    eg_sio3 = BytesIO(str_val)
    assert_raises(ValueError, netcdf_file, eg_sio3, 'r', True)
    eg_sio_64 = BytesIO()
    with make_simple(eg_sio_64, 'w', version=2) as f_64:
        str_val = eg_sio_64.getvalue()
    eg_sio_64 = BytesIO(str_val)
    with netcdf_file(eg_sio_64) as f_64:
        check_simple(f_64)
        assert_equal(f_64.version_byte, 2)
    eg_sio_64 = BytesIO(str_val)
    with netcdf_file(eg_sio_64, version=2) as f_64:
        check_simple(f_64)
        assert_equal(f_64.version_byte, 2)

def test_bytes():
    if False:
        i = 10
        return i + 15
    raw_file = BytesIO()
    f = netcdf_file(raw_file, mode='w')
    f.a = 'b'
    f.createDimension('dim', 1)
    var = f.createVariable('var', np.int16, ('dim',))
    var[0] = -9999
    var.c = 'd'
    f.sync()
    actual = raw_file.getvalue()
    expected = b'CDF\x01\x00\x00\x00\x00\x00\x00\x00\n\x00\x00\x00\x01\x00\x00\x00\x03dim\x00\x00\x00\x00\x01\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x01a\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01b\x00\x00\x00\x00\x00\x00\x0b\x00\x00\x00\x01\x00\x00\x00\x03var\x00\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00\x0c\x00\x00\x00\x01\x00\x00\x00\x01c\x00\x00\x00\x00\x00\x00\x02\x00\x00\x00\x01d\x00\x00\x00\x00\x00\x00\x03\x00\x00\x00\x04\x00\x00\x00x\xd8\xf1\x80\x01'
    assert_equal(actual, expected)

def test_encoded_fill_value():
    if False:
        print('Hello World!')
    with netcdf_file(BytesIO(), mode='w') as f:
        f.createDimension('x', 1)
        var = f.createVariable('var', 'S1', ('x',))
        assert_equal(var._get_encoded_fill_value(), b'\x00')
        var._FillValue = b'\x01'
        assert_equal(var._get_encoded_fill_value(), b'\x01')
        var._FillValue = b'\x00\x00'
        assert_equal(var._get_encoded_fill_value(), b'\x00')

def test_read_example_data():
    if False:
        print('Hello World!')
    for fname in glob(pjoin(TEST_DATA_PATH, '*.nc')):
        with netcdf_file(fname, 'r'):
            pass
        with netcdf_file(fname, 'r', mmap=False):
            pass

def test_itemset_no_segfault_on_readonly():
    if False:
        i = 10
        return i + 15
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist')
        with netcdf_file(filename, 'r', mmap=True) as f:
            time_var = f.variables['time']
    assert_raises(RuntimeError, time_var.assignValue, 42)

def test_appending_issue_gh_8625():
    if False:
        print('Hello World!')
    stream = BytesIO()
    with make_simple(stream, mode='w') as f:
        f.createDimension('x', 2)
        f.createVariable('x', float, ('x',))
        f.variables['x'][...] = 1
        f.flush()
        contents = stream.getvalue()
    stream = BytesIO(contents)
    with netcdf_file(stream, mode='a') as f:
        f.variables['x'][...] = 2

def test_write_invalid_dtype():
    if False:
        return 10
    dtypes = ['int64', 'uint64']
    if np.dtype('int').itemsize == 8:
        dtypes.append('int')
    if np.dtype('uint').itemsize == 8:
        dtypes.append('uint')
    with netcdf_file(BytesIO(), 'w') as f:
        f.createDimension('time', N_EG_ELS)
        for dt in dtypes:
            assert_raises(ValueError, f.createVariable, 'time', dt, ('time',))

def test_flush_rewind():
    if False:
        return 10
    stream = BytesIO()
    with make_simple(stream, mode='w') as f:
        f.createDimension('x', 4)
        v = f.createVariable('v', 'i2', ['x'])
        v[:] = 1
        f.flush()
        len_single = len(stream.getvalue())
        f.flush()
        len_double = len(stream.getvalue())
    assert_(len_single == len_double)

def test_dtype_specifiers():
    if False:
        while True:
            i = 10
    with make_simple(BytesIO(), mode='w') as f:
        f.createDimension('x', 4)
        f.createVariable('v1', 'i2', ['x'])
        f.createVariable('v2', np.int16, ['x'])
        f.createVariable('v3', np.dtype(np.int16), ['x'])

def test_ticket_1720():
    if False:
        print('Hello World!')
    io = BytesIO()
    items = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    with netcdf_file(io, 'w') as f:
        f.history = 'Created for a test'
        f.createDimension('float_var', 10)
        float_var = f.createVariable('float_var', 'f', ('float_var',))
        float_var[:] = items
        float_var.units = 'metres'
        f.flush()
        contents = io.getvalue()
    io = BytesIO(contents)
    with netcdf_file(io, 'r') as f:
        assert_equal(f.history, b'Created for a test')
        float_var = f.variables['float_var']
        assert_equal(float_var.units, b'metres')
        assert_equal(float_var.shape, (10,))
        assert_allclose(float_var[:], items)

def test_mmaps_segfault():
    if False:
        print('Hello World!')
    filename = pjoin(TEST_DATA_PATH, 'example_1.nc')
    if not IS_PYPY:
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            with netcdf_file(filename, mmap=True) as f:
                x = f.variables['lat'][:]
                del x

    def doit():
        if False:
            print('Hello World!')
        with netcdf_file(filename, mmap=True) as f:
            return f.variables['lat'][:]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning, 'Cannot close a netcdf_file opened with mmap=True, when netcdf_variables or arrays referring to its data still exist')
        x = doit()
    x.sum()

def test_zero_dimensional_var():
    if False:
        print('Hello World!')
    io = BytesIO()
    with make_simple(io, 'w') as f:
        v = f.createVariable('zerodim', 'i2', [])
        assert v.isrec is False, v.isrec
        f.flush()

def test_byte_gatts():
    if False:
        for i in range(10):
            print('nop')
    with in_tempdir():
        filename = 'g_byte_atts.nc'
        f = netcdf_file(filename, 'w')
        f._attributes['holy'] = b'grail'
        f._attributes['witch'] = 'floats'
        f.close()
        f = netcdf_file(filename, 'r')
        assert_equal(f._attributes['holy'], b'grail')
        assert_equal(f._attributes['witch'], b'floats')
        f.close()

def test_open_append():
    if False:
        i = 10
        return i + 15
    with in_tempdir():
        filename = 'append_dat.nc'
        f = netcdf_file(filename, 'w')
        f._attributes['Kilroy'] = 'was here'
        f.close()
        f = netcdf_file(filename, 'a')
        assert_equal(f._attributes['Kilroy'], b'was here')
        f._attributes['naughty'] = b'Zoot'
        f.close()
        f = netcdf_file(filename, 'r')
        assert_equal(f._attributes['Kilroy'], b'was here')
        assert_equal(f._attributes['naughty'], b'Zoot')
        f.close()

def test_append_recordDimension():
    if False:
        return 10
    dataSize = 100
    with in_tempdir():
        with netcdf_file('withRecordDimension.nc', 'w') as f:
            f.createDimension('time', None)
            f.createVariable('time', 'd', ('time',))
            f.createDimension('x', dataSize)
            x = f.createVariable('x', 'd', ('x',))
            x[:] = np.array(range(dataSize))
            f.createDimension('y', dataSize)
            y = f.createVariable('y', 'd', ('y',))
            y[:] = np.array(range(dataSize))
            f.createVariable('testData', 'i', ('time', 'x', 'y'))
            f.flush()
            f.close()
        for i in range(2):
            with netcdf_file('withRecordDimension.nc', 'a') as f:
                f.variables['time'].data = np.append(f.variables['time'].data, i)
                f.variables['testData'][i, :, :] = np.full((dataSize, dataSize), i)
                f.flush()
            with netcdf_file('withRecordDimension.nc') as f:
                assert_equal(f.variables['time'][-1], i)
                assert_equal(f.variables['testData'][-1, :, :].copy(), np.full((dataSize, dataSize), i))
                assert_equal(f.variables['time'].data.shape[0], i + 1)
                assert_equal(f.variables['testData'].data.shape[0], i + 1)
        with netcdf_file('withRecordDimension.nc') as f:
            with assert_raises(KeyError) as ar:
                f.variables['testData']._attributes['data']
            ex = ar.value
            assert_equal(ex.args[0], 'data')

def test_maskandscale():
    if False:
        i = 10
        return i + 15
    t = np.linspace(20, 30, 15)
    t[3] = 100
    tm = np.ma.masked_greater(t, 99)
    fname = pjoin(TEST_DATA_PATH, 'example_2.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        Temp = f.variables['Temperature']
        assert_equal(Temp.missing_value, 9999)
        assert_equal(Temp.add_offset, 20)
        assert_equal(Temp.scale_factor, np.float32(0.01))
        found = Temp[:].compressed()
        del Temp
        expected = np.round(tm.compressed(), 2)
        assert_allclose(found, expected)
    with in_tempdir():
        newfname = 'ms.nc'
        f = netcdf_file(newfname, 'w', maskandscale=True)
        f.createDimension('Temperature', len(tm))
        temp = f.createVariable('Temperature', 'i', ('Temperature',))
        temp.missing_value = 9999
        temp.scale_factor = 0.01
        temp.add_offset = 20
        temp[:] = tm
        f.close()
        with netcdf_file(newfname, maskandscale=True) as f:
            Temp = f.variables['Temperature']
            assert_equal(Temp.missing_value, 9999)
            assert_equal(Temp.add_offset, 20)
            assert_equal(Temp.scale_factor, np.float32(0.01))
            expected = np.round(tm.compressed(), 2)
            found = Temp[:].compressed()
            del Temp
            assert_allclose(found, expected)

def test_read_withValuesNearFillValue():
    if False:
        return 10
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var1_fillval0'][:]
        assert_mask_matches(vardata, [False, True, False])

def test_read_withNoFillValue():
    if False:
        for i in range(10):
            print('nop')
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var2_noFillval'][:]
        assert_mask_matches(vardata, [False, False, False])
        assert_equal(vardata, [1, 2, 3])

def test_read_withFillValueAndMissingValue():
    if False:
        return 10
    IRRELEVANT_VALUE = 9999
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var3_fillvalAndMissingValue'][:]
        assert_mask_matches(vardata, [True, False, False])
        assert_equal(vardata, [IRRELEVANT_VALUE, 2, 3])

def test_read_withMissingValue():
    if False:
        print('Hello World!')
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var4_missingValue'][:]
        assert_mask_matches(vardata, [False, True, False])

def test_read_withFillValNaN():
    if False:
        for i in range(10):
            print('nop')
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var5_fillvalNaN'][:]
        assert_mask_matches(vardata, [False, True, False])

def test_read_withChar():
    if False:
        return 10
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var6_char'][:]
        assert_mask_matches(vardata, [False, True, False])

def test_read_with2dVar():
    if False:
        i = 10
        return i + 15
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=True) as f:
        vardata = f.variables['var7_2d'][:]
        assert_mask_matches(vardata, [[True, False], [False, False], [False, True]])

def test_read_withMaskAndScaleFalse():
    if False:
        while True:
            i = 10
    fname = pjoin(TEST_DATA_PATH, 'example_3_maskedvals.nc')
    with netcdf_file(fname, maskandscale=False, mmap=False) as f:
        vardata = f.variables['var3_fillvalAndMissingValue'][:]
        assert_mask_matches(vardata, [False, False, False])
        assert_equal(vardata, [1, 2, 3])