""" Nose test generators

Need function load / save / roundtrip tests

"""
import os
from collections import OrderedDict
from os.path import join as pjoin, dirname
from glob import glob
from io import BytesIO
import re
from tempfile import mkdtemp
import warnings
import shutil
import gzip
from numpy.testing import assert_array_equal, assert_array_almost_equal, assert_equal, assert_, assert_warns, assert_allclose
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import array
import scipy.sparse as SP
import scipy.io
from scipy.io.matlab import MatlabOpaque, MatlabFunction, MatlabObject
import scipy.io.matlab._byteordercodes as boc
from scipy.io.matlab._miobase import matdims, MatWriteError, MatReadError, matfile_version
from scipy.io.matlab._mio import mat_reader_factory, loadmat, savemat, whosmat
from scipy.io.matlab._mio5 import MatFile5Writer, MatFile5Reader, varmats_from_mat, to_writeable, EmptyStructMarker
import scipy.io.matlab._mio5_params as mio5p
from scipy._lib._util import VisibleDeprecationWarning
test_data_path = pjoin(dirname(__file__), 'data')

def mlarr(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Convenience function to return matlab-compatible 2-D array.'
    arr = np.array(*args, **kwargs)
    arr.shape = matdims(arr)
    return arr
theta = np.pi / 4 * np.arange(9, dtype=float).reshape(1, 9)
case_table4 = [{'name': 'double', 'classes': {'testdouble': 'double'}, 'expected': {'testdouble': theta}}]
case_table4.append({'name': 'string', 'classes': {'teststring': 'char'}, 'expected': {'teststring': array(['"Do nine men interpret?" "Nine men," I nod.'])}})
case_table4.append({'name': 'complex', 'classes': {'testcomplex': 'double'}, 'expected': {'testcomplex': np.cos(theta) + 1j * np.sin(theta)}})
A = np.zeros((3, 5))
A[0] = list(range(1, 6))
A[:, 0] = list(range(1, 4))
case_table4.append({'name': 'matrix', 'classes': {'testmatrix': 'double'}, 'expected': {'testmatrix': A}})
case_table4.append({'name': 'sparse', 'classes': {'testsparse': 'sparse'}, 'expected': {'testsparse': SP.coo_matrix(A)}})
B = A.astype(complex)
B[0, 0] += 1j
case_table4.append({'name': 'sparsecomplex', 'classes': {'testsparsecomplex': 'sparse'}, 'expected': {'testsparsecomplex': SP.coo_matrix(B)}})
case_table4.append({'name': 'multi', 'classes': {'theta': 'double', 'a': 'double'}, 'expected': {'theta': theta, 'a': A}})
case_table4.append({'name': 'minus', 'classes': {'testminus': 'double'}, 'expected': {'testminus': mlarr(-1)}})
case_table4.append({'name': 'onechar', 'classes': {'testonechar': 'char'}, 'expected': {'testonechar': array(['r'])}})
CA = mlarr(([], mlarr([1]), mlarr([[1, 2]]), mlarr([[1, 2, 3]])), dtype=object).reshape(1, -1)
CA[0, 0] = array(['This cell contains this string and 3 arrays of increasing length'])
case_table5 = [{'name': 'cell', 'classes': {'testcell': 'cell'}, 'expected': {'testcell': CA}}]
CAE = mlarr((mlarr(1), mlarr(2), mlarr([]), mlarr([]), mlarr(3)), dtype=object).reshape(1, -1)
objarr = np.empty((1, 1), dtype=object)
objarr[0, 0] = mlarr(1)
case_table5.append({'name': 'scalarcell', 'classes': {'testscalarcell': 'cell'}, 'expected': {'testscalarcell': objarr}})
case_table5.append({'name': 'emptycell', 'classes': {'testemptycell': 'cell'}, 'expected': {'testemptycell': CAE}})
case_table5.append({'name': 'stringarray', 'classes': {'teststringarray': 'char'}, 'expected': {'teststringarray': array(['one  ', 'two  ', 'three'])}})
case_table5.append({'name': '3dmatrix', 'classes': {'test3dmatrix': 'double'}, 'expected': {'test3dmatrix': np.transpose(np.reshape(list(range(1, 25)), (4, 3, 2)))}})
st_sub_arr = array([np.sqrt(2), np.exp(1), np.pi]).reshape(1, 3)
dtype = [(n, object) for n in ['stringfield', 'doublefield', 'complexfield']]
st1 = np.zeros((1, 1), dtype)
st1['stringfield'][0, 0] = array(['Rats live on no evil star.'])
st1['doublefield'][0, 0] = st_sub_arr
st1['complexfield'][0, 0] = st_sub_arr * (1 + 1j)
case_table5.append({'name': 'struct', 'classes': {'teststruct': 'struct'}, 'expected': {'teststruct': st1}})
CN = np.zeros((1, 2), dtype=object)
CN[0, 0] = mlarr(1)
CN[0, 1] = np.zeros((1, 3), dtype=object)
CN[0, 1][0, 0] = mlarr(2, dtype=np.uint8)
CN[0, 1][0, 1] = mlarr([[3]], dtype=np.uint8)
CN[0, 1][0, 2] = np.zeros((1, 2), dtype=object)
CN[0, 1][0, 2][0, 0] = mlarr(4, dtype=np.uint8)
CN[0, 1][0, 2][0, 1] = mlarr(5, dtype=np.uint8)
case_table5.append({'name': 'cellnest', 'classes': {'testcellnest': 'cell'}, 'expected': {'testcellnest': CN}})
st2 = np.empty((1, 1), dtype=[(n, object) for n in ['one', 'two']])
st2[0, 0]['one'] = mlarr(1)
st2[0, 0]['two'] = np.empty((1, 1), dtype=[('three', object)])
st2[0, 0]['two'][0, 0]['three'] = array(['number 3'])
case_table5.append({'name': 'structnest', 'classes': {'teststructnest': 'struct'}, 'expected': {'teststructnest': st2}})
a = np.empty((1, 2), dtype=[(n, object) for n in ['one', 'two']])
a[0, 0]['one'] = mlarr(1)
a[0, 0]['two'] = mlarr(2)
a[0, 1]['one'] = array(['number 1'])
a[0, 1]['two'] = array(['number 2'])
case_table5.append({'name': 'structarr', 'classes': {'teststructarr': 'struct'}, 'expected': {'teststructarr': a}})
ODT = np.dtype([(n, object) for n in ['expr', 'inputExpr', 'args', 'isEmpty', 'numArgs', 'version']])
MO = MatlabObject(np.zeros((1, 1), dtype=ODT), 'inline')
m0 = MO[0, 0]
m0['expr'] = array(['x'])
m0['inputExpr'] = array([' x = INLINE_INPUTS_{1};'])
m0['args'] = array(['x'])
m0['isEmpty'] = mlarr(0)
m0['numArgs'] = mlarr(1)
m0['version'] = mlarr(1)
case_table5.append({'name': 'object', 'classes': {'testobject': 'object'}, 'expected': {'testobject': MO}})
fp_u_str = open(pjoin(test_data_path, 'japanese_utf8.txt'), 'rb')
u_str = fp_u_str.read().decode('utf-8')
fp_u_str.close()
case_table5.append({'name': 'unicode', 'classes': {'testunicode': 'char'}, 'expected': {'testunicode': array([u_str])}})
case_table5.append({'name': 'sparse', 'classes': {'testsparse': 'sparse'}, 'expected': {'testsparse': SP.coo_matrix(A)}})
case_table5.append({'name': 'sparsecomplex', 'classes': {'testsparsecomplex': 'sparse'}, 'expected': {'testsparsecomplex': SP.coo_matrix(B)}})
case_table5.append({'name': 'bool', 'classes': {'testbools': 'logical'}, 'expected': {'testbools': array([[True], [False]])}})
case_table5_rt = case_table5[:]
case_table5_rt.append({'name': 'objectarray', 'classes': {'testobjectarray': 'object'}, 'expected': {'testobjectarray': np.repeat(MO, 2).reshape(1, 2)}})

def types_compatible(var1, var2):
    if False:
        print('Hello World!')
    'Check if types are same or compatible.\n\n    0-D numpy scalars are compatible with bare python scalars.\n    '
    type1 = type(var1)
    type2 = type(var2)
    if type1 is type2:
        return True
    if type1 is np.ndarray and var1.shape == ():
        return type(var1.item()) is type2
    if type2 is np.ndarray and var2.shape == ():
        return type(var2.item()) is type1
    return False

def _check_level(label, expected, actual):
    if False:
        print('Hello World!')
    ' Check one level of a potentially nested array '
    if SP.issparse(expected):
        assert_(SP.issparse(actual))
        assert_array_almost_equal(actual.toarray(), expected.toarray(), err_msg=label, decimal=5)
        return
    assert_(types_compatible(expected, actual), 'Expected type %s, got %s at %s' % (type(expected), type(actual), label))
    if not isinstance(expected, (np.void, np.ndarray, MatlabObject)):
        assert_equal(expected, actual)
        return
    assert_(expected.shape == actual.shape, msg='Expected shape {}, got {} at {}'.format(expected.shape, actual.shape, label))
    ex_dtype = expected.dtype
    if ex_dtype.hasobject:
        if isinstance(expected, MatlabObject):
            assert_equal(expected.classname, actual.classname)
        for (i, ev) in enumerate(expected):
            level_label = '%s, [%d], ' % (label, i)
            _check_level(level_label, ev, actual[i])
        return
    if ex_dtype.fields:
        for fn in ex_dtype.fields:
            level_label = f'{label}, field {fn}, '
            _check_level(level_label, expected[fn], actual[fn])
        return
    if ex_dtype.type in (str, np.str_, np.bool_):
        assert_equal(actual, expected, err_msg=label)
        return
    assert_array_almost_equal(actual, expected, err_msg=label, decimal=5)

def _load_check_case(name, files, case):
    if False:
        print('Hello World!')
    for file_name in files:
        matdict = loadmat(file_name, struct_as_record=True)
        label = f'test {name}; file {file_name}'
        for (k, expected) in case.items():
            k_label = f'{label}, variable {k}'
            assert_(k in matdict, 'Missing key at %s' % k_label)
            _check_level(k_label, expected, matdict[k])

def _whos_check_case(name, files, case, classes):
    if False:
        while True:
            i = 10
    for file_name in files:
        label = f'test {name}; file {file_name}'
        whos = whosmat(file_name)
        expected_whos = [(k, expected.shape, classes[k]) for (k, expected) in case.items()]
        whos.sort()
        expected_whos.sort()
        assert_equal(whos, expected_whos, f'{label}: {whos!r} != {expected_whos!r}')

def _rt_check_case(name, expected, format):
    if False:
        return 10
    mat_stream = BytesIO()
    savemat(mat_stream, expected, format=format)
    mat_stream.seek(0)
    _load_check_case(name, [mat_stream], expected)

def _cases(version, filt='test%(name)s_*.mat'):
    if False:
        return 10
    if version == '4':
        cases = case_table4
    elif version == '5':
        cases = case_table5
    else:
        assert version == '5_rt'
        cases = case_table5_rt
    for case in cases:
        name = case['name']
        expected = case['expected']
        if filt is None:
            files = None
        else:
            use_filt = pjoin(test_data_path, filt % dict(name=name))
            files = glob(use_filt)
            assert len(files) > 0, f'No files for test {name} using filter {filt}'
        classes = case['classes']
        yield (name, files, expected, classes)

@pytest.mark.parametrize('version', ('4', '5'))
def test_load(version):
    if False:
        while True:
            i = 10
    for case in _cases(version):
        _load_check_case(*case[:3])

@pytest.mark.parametrize('version', ('4', '5'))
def test_whos(version):
    if False:
        return 10
    for case in _cases(version):
        _whos_check_case(*case)

@pytest.mark.parametrize('version, fmts', [('4', ['4', '5']), ('5_rt', ['5'])])
def test_round_trip(version, fmts):
    if False:
        return 10
    for case in _cases(version, filt=None):
        for fmt in fmts:
            _rt_check_case(case[0], case[2], fmt)

def test_gzip_simple():
    if False:
        while True:
            i = 10
    xdense = np.zeros((20, 20))
    xdense[2, 3] = 2.3
    xdense[4, 5] = 4.5
    x = SP.csc_matrix(xdense)
    name = 'gzip_test'
    expected = {'x': x}
    format = '4'
    tmpdir = mkdtemp()
    try:
        fname = pjoin(tmpdir, name)
        mat_stream = gzip.open(fname, mode='wb')
        savemat(mat_stream, expected, format=format)
        mat_stream.close()
        mat_stream = gzip.open(fname, mode='rb')
        actual = loadmat(mat_stream, struct_as_record=True)
        mat_stream.close()
    finally:
        shutil.rmtree(tmpdir)
    assert_array_almost_equal(actual['x'].toarray(), expected['x'].toarray(), err_msg=repr(actual))

def test_multiple_open():
    if False:
        i = 10
        return i + 15
    tmpdir = mkdtemp()
    try:
        x = dict(x=np.zeros((2, 2)))
        fname = pjoin(tmpdir, 'a.mat')
        savemat(fname, x)
        os.unlink(fname)
        savemat(fname, x)
        loadmat(fname)
        os.unlink(fname)
        f = open(fname, 'wb')
        savemat(f, x)
        f.seek(0)
        f.close()
        f = open(fname, 'rb')
        loadmat(f)
        f.seek(0)
        f.close()
    finally:
        shutil.rmtree(tmpdir)

def test_mat73():
    if False:
        i = 10
        return i + 15
    filenames = glob(pjoin(test_data_path, 'testhdf5*.mat'))
    assert_(len(filenames) > 0)
    for filename in filenames:
        fp = open(filename, 'rb')
        assert_raises(NotImplementedError, loadmat, fp, struct_as_record=True)
        fp.close()

def test_warnings():
    if False:
        return 10
    fname = pjoin(test_data_path, 'testdouble_7.1_GLNX86.mat')
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        loadmat(fname, struct_as_record=True)
        loadmat(fname, struct_as_record=False)

def test_regression_653():
    if False:
        print('Hello World!')
    sio = BytesIO()
    savemat(sio, {'d': {1: 2}}, format='5')
    back = loadmat(sio)['d']
    assert_equal(back.shape, (1, 1))
    assert_equal(back.dtype, np.dtype(object))
    assert_(back[0, 0] is None)

def test_structname_len():
    if False:
        i = 10
        return i + 15
    lim = 31
    fldname = 'a' * lim
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    savemat(BytesIO(), {'longstruct': st1}, format='5')
    fldname = 'a' * (lim + 1)
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    assert_raises(ValueError, savemat, BytesIO(), {'longstruct': st1}, format='5')

def test_4_and_long_field_names_incompatible():
    if False:
        while True:
            i = 10
    my_struct = np.zeros((1, 1), dtype=[('my_fieldname', object)])
    assert_raises(ValueError, savemat, BytesIO(), {'my_struct': my_struct}, format='4', long_field_names=True)

def test_long_field_names():
    if False:
        while True:
            i = 10
    lim = 63
    fldname = 'a' * lim
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    savemat(BytesIO(), {'longstruct': st1}, format='5', long_field_names=True)
    fldname = 'a' * (lim + 1)
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    assert_raises(ValueError, savemat, BytesIO(), {'longstruct': st1}, format='5', long_field_names=True)

def test_long_field_names_in_struct():
    if False:
        print('Hello World!')
    lim = 63
    fldname = 'a' * lim
    cell = np.ndarray((1, 2), dtype=object)
    st1 = np.zeros((1, 1), dtype=[(fldname, object)])
    cell[0, 0] = st1
    cell[0, 1] = st1
    savemat(BytesIO(), {'longstruct': cell}, format='5', long_field_names=True)
    assert_raises(ValueError, savemat, BytesIO(), {'longstruct': cell}, format='5', long_field_names=False)

def test_cell_with_one_thing_in_it():
    if False:
        for i in range(10):
            print('nop')
    cells = np.ndarray((1, 2), dtype=object)
    cells[0, 0] = 'Hello'
    cells[0, 1] = 'World'
    savemat(BytesIO(), {'x': cells}, format='5')
    cells = np.ndarray((1, 1), dtype=object)
    cells[0, 0] = 'Hello, world'
    savemat(BytesIO(), {'x': cells}, format='5')

def test_writer_properties():
    if False:
        return 10
    mfw = MatFile5Writer(BytesIO())
    assert_equal(mfw.global_vars, [])
    mfw.global_vars = ['avar']
    assert_equal(mfw.global_vars, ['avar'])
    assert_equal(mfw.unicode_strings, False)
    mfw.unicode_strings = True
    assert_equal(mfw.unicode_strings, True)
    assert_equal(mfw.long_field_names, False)
    mfw.long_field_names = True
    assert_equal(mfw.long_field_names, True)

def test_use_small_element():
    if False:
        print('Hello World!')
    sio = BytesIO()
    wtr = MatFile5Writer(sio)
    arr = np.zeros(10)
    wtr.put_variables({'aaaaa': arr})
    w_sz = len(sio.getvalue())
    sio.truncate(0)
    sio.seek(0)
    wtr.put_variables({'aaaa': arr})
    assert_(w_sz - len(sio.getvalue()) > 4)
    sio.truncate(0)
    sio.seek(0)
    wtr.put_variables({'aaaaaa': arr})
    assert_(len(sio.getvalue()) - w_sz < 4)

def test_save_dict():
    if False:
        return 10
    ab_exp = np.array([[(1, 2)]], dtype=[('a', object), ('b', object)])
    for dict_type in (dict, OrderedDict):
        d = dict_type([('a', 1), ('b', 2)])
        stream = BytesIO()
        savemat(stream, {'dict': d})
        stream.seek(0)
        vals = loadmat(stream)['dict']
        assert_equal(vals.dtype.names, ('a', 'b'))
        assert_array_equal(vals, ab_exp)

def test_1d_shape():
    if False:
        for i in range(10):
            print('nop')
    arr = np.arange(5)
    for format in ('4', '5'):
        stream = BytesIO()
        savemat(stream, {'oned': arr}, format=format)
        vals = loadmat(stream)
        assert_equal(vals['oned'].shape, (1, 5))
        stream = BytesIO()
        savemat(stream, {'oned': arr}, format=format, oned_as='column')
        vals = loadmat(stream)
        assert_equal(vals['oned'].shape, (5, 1))
        stream = BytesIO()
        savemat(stream, {'oned': arr}, format=format, oned_as='row')
        vals = loadmat(stream)
        assert_equal(vals['oned'].shape, (1, 5))

def test_compression():
    if False:
        return 10
    arr = np.zeros(100).reshape((5, 20))
    arr[2, 10] = 1
    stream = BytesIO()
    savemat(stream, {'arr': arr})
    raw_len = len(stream.getvalue())
    vals = loadmat(stream)
    assert_array_equal(vals['arr'], arr)
    stream = BytesIO()
    savemat(stream, {'arr': arr}, do_compression=True)
    compressed_len = len(stream.getvalue())
    vals = loadmat(stream)
    assert_array_equal(vals['arr'], arr)
    assert_(raw_len > compressed_len)
    arr2 = arr.copy()
    arr2[0, 0] = 1
    stream = BytesIO()
    savemat(stream, {'arr': arr, 'arr2': arr2}, do_compression=False)
    vals = loadmat(stream)
    assert_array_equal(vals['arr2'], arr2)
    stream = BytesIO()
    savemat(stream, {'arr': arr, 'arr2': arr2}, do_compression=True)
    vals = loadmat(stream)
    assert_array_equal(vals['arr2'], arr2)

def test_single_object():
    if False:
        for i in range(10):
            print('nop')
    stream = BytesIO()
    savemat(stream, {'A': np.array(1, dtype=object)})

def test_skip_variable():
    if False:
        return 10
    filename = pjoin(test_data_path, 'test_skip_variable.mat')
    d = loadmat(filename, struct_as_record=True)
    assert_('first' in d)
    assert_('second' in d)
    (factory, file_opened) = mat_reader_factory(filename, struct_as_record=True)
    d = factory.get_variables('second')
    assert_('second' in d)
    factory.mat_stream.close()

def test_empty_struct():
    if False:
        return 10
    filename = pjoin(test_data_path, 'test_empty_struct.mat')
    d = loadmat(filename, struct_as_record=True)
    a = d['a']
    assert_equal(a.shape, (1, 1))
    assert_equal(a.dtype, np.dtype(object))
    assert_(a[0, 0] is None)
    stream = BytesIO()
    arr = np.array((), dtype='U')
    savemat(stream, {'arr': arr})
    d = loadmat(stream)
    a2 = d['arr']
    assert_array_equal(a2, arr)

def test_save_empty_dict():
    if False:
        for i in range(10):
            print('nop')
    stream = BytesIO()
    savemat(stream, {'arr': {}})
    d = loadmat(stream)
    a = d['arr']
    assert_equal(a.shape, (1, 1))
    assert_equal(a.dtype, np.dtype(object))
    assert_(a[0, 0] is None)

def assert_any_equal(output, alternatives):
    if False:
        i = 10
        return i + 15
    ' Assert `output` is equal to at least one element in `alternatives`\n    '
    one_equal = False
    for expected in alternatives:
        if np.all(output == expected):
            one_equal = True
            break
    assert_(one_equal)

def test_to_writeable():
    if False:
        while True:
            i = 10
    res = to_writeable(np.array([1]))
    assert_equal(res.shape, (1,))
    assert_array_equal(res, 1)
    expected1 = np.array([(1, 2)], dtype=[('a', '|O8'), ('b', '|O8')])
    expected2 = np.array([(2, 1)], dtype=[('b', '|O8'), ('a', '|O8')])
    alternatives = (expected1, expected2)
    assert_any_equal(to_writeable({'a': 1, 'b': 2}), alternatives)
    assert_any_equal(to_writeable({'a': 1, 'b': 2, '_c': 3}), alternatives)
    assert_any_equal(to_writeable({'a': 1, 'b': 2, 100: 3}), alternatives)
    assert_any_equal(to_writeable({'a': 1, 'b': 2, '99': 3}), alternatives)

    class klass:
        pass
    c = klass
    c.a = 1
    c.b = 2
    assert_any_equal(to_writeable(c), alternatives)
    res = to_writeable([])
    assert_equal(res.shape, (0,))
    assert_equal(res.dtype.type, np.float64)
    res = to_writeable(())
    assert_equal(res.shape, (0,))
    assert_equal(res.dtype.type, np.float64)
    assert_(to_writeable(None) is None)
    assert_equal(to_writeable('a string').dtype.type, np.str_)
    res = to_writeable(1)
    assert_equal(res.shape, ())
    assert_equal(res.dtype.type, np.array(1).dtype.type)
    assert_array_equal(res, 1)
    assert_(to_writeable({}) is EmptyStructMarker)
    assert_(to_writeable(object()) is None)

    class C:
        pass
    assert_(to_writeable(c()) is EmptyStructMarker)
    res = to_writeable({'a': 1})['a']
    assert_equal(res.shape, (1,))
    assert_equal(res.dtype.type, np.object_)
    assert_(to_writeable({'1': 1}) is EmptyStructMarker)
    assert_(to_writeable({'_a': 1}) is EmptyStructMarker)
    assert_equal(to_writeable({'1': 1, 'f': 2}), np.array([(2,)], dtype=[('f', '|O8')]))

def test_recarray():
    if False:
        print('Hello World!')
    dt = [('f1', 'f8'), ('f2', 'S10')]
    arr = np.zeros((2,), dtype=dt)
    arr[0]['f1'] = 0.5
    arr[0]['f2'] = 'python'
    arr[1]['f1'] = 99
    arr[1]['f2'] = 'not perl'
    stream = BytesIO()
    savemat(stream, {'arr': arr})
    d = loadmat(stream, struct_as_record=False)
    a20 = d['arr'][0, 0]
    assert_equal(a20.f1, 0.5)
    assert_equal(a20.f2, 'python')
    d = loadmat(stream, struct_as_record=True)
    a20 = d['arr'][0, 0]
    assert_equal(a20['f1'], 0.5)
    assert_equal(a20['f2'], 'python')
    assert_equal(a20.dtype, np.dtype([('f1', 'O'), ('f2', 'O')]))
    a21 = d['arr'].flat[1]
    assert_equal(a21['f1'], 99)
    assert_equal(a21['f2'], 'not perl')

def test_save_object():
    if False:
        return 10

    class C:
        pass
    c = C()
    c.field1 = 1
    c.field2 = 'a string'
    stream = BytesIO()
    savemat(stream, {'c': c})
    d = loadmat(stream, struct_as_record=False)
    c2 = d['c'][0, 0]
    assert_equal(c2.field1, 1)
    assert_equal(c2.field2, 'a string')
    d = loadmat(stream, struct_as_record=True)
    c2 = d['c'][0, 0]
    assert_equal(c2['field1'], 1)
    assert_equal(c2['field2'], 'a string')

def test_read_opts():
    if False:
        while True:
            i = 10
    arr = np.arange(6).reshape(1, 6)
    stream = BytesIO()
    savemat(stream, {'a': arr})
    rdr = MatFile5Reader(stream)
    back_dict = rdr.get_variables()
    rarr = back_dict['a']
    assert_array_equal(rarr, arr)
    rdr = MatFile5Reader(stream, squeeze_me=True)
    assert_array_equal(rdr.get_variables()['a'], arr.reshape((6,)))
    rdr.squeeze_me = False
    assert_array_equal(rarr, arr)
    rdr = MatFile5Reader(stream, byte_order=boc.native_code)
    assert_array_equal(rdr.get_variables()['a'], arr)
    rdr = MatFile5Reader(stream, byte_order=boc.swapped_code)
    assert_raises(Exception, rdr.get_variables)
    rdr.byte_order = boc.native_code
    assert_array_equal(rdr.get_variables()['a'], arr)
    arr = np.array(['a string'])
    stream.truncate(0)
    stream.seek(0)
    savemat(stream, {'a': arr})
    rdr = MatFile5Reader(stream)
    assert_array_equal(rdr.get_variables()['a'], arr)
    rdr = MatFile5Reader(stream, chars_as_strings=False)
    carr = np.atleast_2d(np.array(list(arr.item()), dtype='U1'))
    assert_array_equal(rdr.get_variables()['a'], carr)
    rdr.chars_as_strings = True
    assert_array_equal(rdr.get_variables()['a'], arr)

def test_empty_string():
    if False:
        print('Hello World!')
    estring_fname = pjoin(test_data_path, 'single_empty_string.mat')
    fp = open(estring_fname, 'rb')
    rdr = MatFile5Reader(fp)
    d = rdr.get_variables()
    fp.close()
    assert_array_equal(d['a'], np.array([], dtype='U1'))
    stream = BytesIO()
    savemat(stream, {'a': np.array([''])})
    rdr = MatFile5Reader(stream)
    d = rdr.get_variables()
    assert_array_equal(d['a'], np.array([], dtype='U1'))
    stream.truncate(0)
    stream.seek(0)
    savemat(stream, {'a': np.array([], dtype='U1')})
    rdr = MatFile5Reader(stream)
    d = rdr.get_variables()
    assert_array_equal(d['a'], np.array([], dtype='U1'))
    stream.close()

def test_corrupted_data():
    if False:
        while True:
            i = 10
    import zlib
    for (exc, fname) in [(ValueError, 'corrupted_zlib_data.mat'), (zlib.error, 'corrupted_zlib_checksum.mat')]:
        with open(pjoin(test_data_path, fname), 'rb') as fp:
            rdr = MatFile5Reader(fp)
            assert_raises(exc, rdr.get_variables)

def test_corrupted_data_check_can_be_disabled():
    if False:
        return 10
    with open(pjoin(test_data_path, 'corrupted_zlib_data.mat'), 'rb') as fp:
        rdr = MatFile5Reader(fp, verify_compressed_data_integrity=False)
        rdr.get_variables()

def test_read_both_endian():
    if False:
        print('Hello World!')
    for fname in ('big_endian.mat', 'little_endian.mat'):
        fp = open(pjoin(test_data_path, fname), 'rb')
        rdr = MatFile5Reader(fp)
        d = rdr.get_variables()
        fp.close()
        assert_array_equal(d['strings'], np.array([['hello'], ['world']], dtype=object))
        assert_array_equal(d['floats'], np.array([[2.0, 3.0], [3.0, 4.0]], dtype=np.float32))

def test_write_opposite_endian():
    if False:
        return 10
    float_arr = np.array([[2.0, 3.0], [3.0, 4.0]])
    int_arr = np.arange(6).reshape((2, 3))
    uni_arr = np.array(['hello', 'world'], dtype='U')
    stream = BytesIO()
    savemat(stream, {'floats': float_arr.byteswap().view(float_arr.dtype.newbyteorder()), 'ints': int_arr.byteswap().view(int_arr.dtype.newbyteorder()), 'uni_arr': uni_arr.byteswap().view(uni_arr.dtype.newbyteorder())})
    rdr = MatFile5Reader(stream)
    d = rdr.get_variables()
    assert_array_equal(d['floats'], float_arr)
    assert_array_equal(d['ints'], int_arr)
    assert_array_equal(d['uni_arr'], uni_arr)
    stream.close()

def test_logical_array():
    if False:
        return 10
    with open(pjoin(test_data_path, 'testbool_8_WIN64.mat'), 'rb') as fobj:
        rdr = MatFile5Reader(fobj, mat_dtype=True)
        d = rdr.get_variables()
    x = np.array([[True], [False]], dtype=np.bool_)
    assert_array_equal(d['testbools'], x)
    assert_equal(d['testbools'].dtype, x.dtype)

def test_logical_out_type():
    if False:
        print('Hello World!')
    stream = BytesIO()
    barr = np.array([False, True, False])
    savemat(stream, {'barray': barr})
    stream.seek(0)
    reader = MatFile5Reader(stream)
    reader.initialize_read()
    reader.read_file_header()
    (hdr, _) = reader.read_var_header()
    assert_equal(hdr.mclass, mio5p.mxUINT8_CLASS)
    assert_equal(hdr.is_logical, True)
    var = reader.read_var_array(hdr, False)
    assert_equal(var.dtype.type, np.uint8)

def test_roundtrip_zero_dimensions():
    if False:
        for i in range(10):
            print('nop')
    stream = BytesIO()
    savemat(stream, {'d': np.empty((10, 0))})
    d = loadmat(stream)
    assert d['d'].shape == (10, 0)

def test_mat4_3d():
    if False:
        return 10
    stream = BytesIO()
    arr = np.arange(24).reshape((2, 3, 4))
    assert_raises(ValueError, savemat, stream, {'a': arr}, True, '4')

def test_func_read():
    if False:
        i = 10
        return i + 15
    func_eg = pjoin(test_data_path, 'testfunc_7.4_GLNX86.mat')
    fp = open(func_eg, 'rb')
    rdr = MatFile5Reader(fp)
    d = rdr.get_variables()
    fp.close()
    assert isinstance(d['testfunc'], MatlabFunction)
    stream = BytesIO()
    wtr = MatFile5Writer(stream)
    assert_raises(MatWriteError, wtr.put_variables, d)

def test_mat_dtype():
    if False:
        i = 10
        return i + 15
    double_eg = pjoin(test_data_path, 'testmatrix_6.1_SOL2.mat')
    fp = open(double_eg, 'rb')
    rdr = MatFile5Reader(fp, mat_dtype=False)
    d = rdr.get_variables()
    fp.close()
    assert_equal(d['testmatrix'].dtype.kind, 'u')
    fp = open(double_eg, 'rb')
    rdr = MatFile5Reader(fp, mat_dtype=True)
    d = rdr.get_variables()
    fp.close()
    assert_equal(d['testmatrix'].dtype.kind, 'f')

def test_sparse_in_struct():
    if False:
        i = 10
        return i + 15
    st = {'sparsefield': SP.coo_matrix(np.eye(4))}
    stream = BytesIO()
    savemat(stream, {'a': st})
    d = loadmat(stream, struct_as_record=True)
    assert_array_equal(d['a'][0, 0]['sparsefield'].toarray(), np.eye(4))

def test_mat_struct_squeeze():
    if False:
        return 10
    stream = BytesIO()
    in_d = {'st': {'one': 1, 'two': 2}}
    savemat(stream, in_d)
    loadmat(stream, struct_as_record=False)
    loadmat(stream, struct_as_record=False, squeeze_me=True)

def test_scalar_squeeze():
    if False:
        print('Hello World!')
    stream = BytesIO()
    in_d = {'scalar': [[0.1]], 'string': 'my name', 'st': {'one': 1, 'two': 2}}
    savemat(stream, in_d)
    out_d = loadmat(stream, squeeze_me=True)
    assert_(isinstance(out_d['scalar'], float))
    assert_(isinstance(out_d['string'], str))
    assert_(isinstance(out_d['st'], np.ndarray))

def test_str_round():
    if False:
        print('Hello World!')
    stream = BytesIO()
    in_arr = np.array(['Hello', 'Foob'])
    out_arr = np.array(['Hello', 'Foob '])
    savemat(stream, dict(a=in_arr))
    res = loadmat(stream)
    assert_array_equal(res['a'], out_arr)
    stream.truncate(0)
    stream.seek(0)
    in_str = in_arr.tobytes(order='F')
    in_from_str = np.ndarray(shape=a.shape, dtype=in_arr.dtype, order='F', buffer=in_str)
    savemat(stream, dict(a=in_from_str))
    assert_array_equal(res['a'], out_arr)
    stream.truncate(0)
    stream.seek(0)
    in_arr_u = in_arr.astype('U')
    out_arr_u = out_arr.astype('U')
    savemat(stream, {'a': in_arr_u})
    res = loadmat(stream)
    assert_array_equal(res['a'], out_arr_u)

def test_fieldnames():
    if False:
        i = 10
        return i + 15
    stream = BytesIO()
    savemat(stream, {'a': {'a': 1, 'b': 2}})
    res = loadmat(stream)
    field_names = res['a'].dtype.names
    assert_equal(set(field_names), {'a', 'b'})

def test_loadmat_varnames():
    if False:
        return 10
    mat5_sys_names = ['__globals__', '__header__', '__version__']
    for (eg_file, sys_v_names) in ((pjoin(test_data_path, 'testmulti_4.2c_SOL2.mat'), []), (pjoin(test_data_path, 'testmulti_7.4_GLNX86.mat'), mat5_sys_names)):
        vars = loadmat(eg_file)
        assert_equal(set(vars.keys()), set(['a', 'theta'] + sys_v_names))
        vars = loadmat(eg_file, variable_names='a')
        assert_equal(set(vars.keys()), set(['a'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=['a'])
        assert_equal(set(vars.keys()), set(['a'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=['theta'])
        assert_equal(set(vars.keys()), set(['theta'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=('theta',))
        assert_equal(set(vars.keys()), set(['theta'] + sys_v_names))
        vars = loadmat(eg_file, variable_names=[])
        assert_equal(set(vars.keys()), set(sys_v_names))
        vnames = ['theta']
        vars = loadmat(eg_file, variable_names=vnames)
        assert_equal(vnames, ['theta'])

def test_round_types():
    if False:
        return 10
    arr = np.arange(10)
    stream = BytesIO()
    for dts in ('f8', 'f4', 'i8', 'i4', 'i2', 'i1', 'u8', 'u4', 'u2', 'u1', 'c16', 'c8'):
        stream.truncate(0)
        stream.seek(0)
        savemat(stream, {'arr': arr.astype(dts)})
        vars = loadmat(stream)
        assert_equal(np.dtype(dts), vars['arr'].dtype)

def test_varmats_from_mat():
    if False:
        print('Hello World!')
    names_vars = (('arr', mlarr(np.arange(10))), ('mystr', mlarr('a string')), ('mynum', mlarr(10)))

    class C:

        def items(self):
            if False:
                i = 10
                return i + 15
            return names_vars
    stream = BytesIO()
    savemat(stream, C())
    varmats = varmats_from_mat(stream)
    assert_equal(len(varmats), 3)
    for i in range(3):
        (name, var_stream) = varmats[i]
        (exp_name, exp_res) = names_vars[i]
        assert_equal(name, exp_name)
        res = loadmat(var_stream)
        assert_array_equal(res[name], exp_res)

def test_one_by_zero():
    if False:
        return 10
    func_eg = pjoin(test_data_path, 'one_by_zero_char.mat')
    fp = open(func_eg, 'rb')
    rdr = MatFile5Reader(fp)
    d = rdr.get_variables()
    fp.close()
    assert_equal(d['var'].shape, (0,))

def test_load_mat4_le():
    if False:
        i = 10
        return i + 15
    mat4_fname = pjoin(test_data_path, 'test_mat4_le_floats.mat')
    vars = loadmat(mat4_fname)
    assert_array_equal(vars['a'], [[0.1, 1.2]])

def test_unicode_mat4():
    if False:
        for i in range(10):
            print('nop')
    bio = BytesIO()
    var = {'second_cat': 'Schrödinger'}
    savemat(bio, var, format='4')
    var_back = loadmat(bio)
    assert_equal(var_back['second_cat'], var['second_cat'])

def test_logical_sparse():
    if False:
        print('Hello World!')
    filename = pjoin(test_data_path, 'logical_sparse.mat')
    d = loadmat(filename, struct_as_record=True)
    log_sp = d['sp_log_5_4']
    assert_(isinstance(log_sp, SP.csc_matrix))
    assert_equal(log_sp.dtype.type, np.bool_)
    assert_array_equal(log_sp.toarray(), [[True, True, True, False], [False, False, True, False], [False, False, True, False], [False, False, False, False], [False, False, False, False]])

def test_empty_sparse():
    if False:
        i = 10
        return i + 15
    sio = BytesIO()
    import scipy.sparse
    empty_sparse = scipy.sparse.csr_matrix([[0, 0], [0, 0]])
    savemat(sio, dict(x=empty_sparse))
    sio.seek(0)
    res = loadmat(sio)
    assert_array_equal(res['x'].shape, empty_sparse.shape)
    assert_array_equal(res['x'].toarray(), 0)
    sio.seek(0)
    reader = MatFile5Reader(sio)
    reader.initialize_read()
    reader.read_file_header()
    (hdr, _) = reader.read_var_header()
    assert_equal(hdr.nzmax, 1)

def test_empty_mat_error():
    if False:
        i = 10
        return i + 15
    sio = BytesIO()
    assert_raises(MatReadError, loadmat, sio)

def test_miuint32_compromise():
    if False:
        return 10
    filename = pjoin(test_data_path, 'miuint32_for_miint32.mat')
    res = loadmat(filename)
    assert_equal(res['an_array'], np.arange(10)[None, :])
    filename = pjoin(test_data_path, 'bad_miuint32.mat')
    with assert_raises(ValueError):
        loadmat(filename)

def test_miutf8_for_miint8_compromise():
    if False:
        return 10
    filename = pjoin(test_data_path, 'miutf8_array_name.mat')
    res = loadmat(filename)
    assert_equal(res['array_name'], [[1]])
    filename = pjoin(test_data_path, 'bad_miutf8_array_name.mat')
    with assert_raises(ValueError):
        loadmat(filename)

def test_bad_utf8():
    if False:
        return 10
    filename = pjoin(test_data_path, 'broken_utf8.mat')
    res = loadmat(filename)
    assert_equal(res['bad_string'], b'\x80 am broken'.decode('utf8', 'replace'))

def test_save_unicode_field(tmpdir):
    if False:
        return 10
    filename = os.path.join(str(tmpdir), 'test.mat')
    test_dict = {'a': {'b': 1, 'c': 'test_str'}}
    savemat(filename, test_dict)

def test_save_custom_array_type(tmpdir):
    if False:
        for i in range(10):
            print('nop')

    class CustomArray:

        def __array__(self):
            if False:
                return 10
            return np.arange(6.0).reshape(2, 3)
    a = CustomArray()
    filename = os.path.join(str(tmpdir), 'test.mat')
    savemat(filename, {'a': a})
    out = loadmat(filename)
    assert_array_equal(out['a'], np.array(a))

def test_filenotfound():
    if False:
        return 10
    assert_raises(OSError, loadmat, 'NotExistentFile00.mat')
    assert_raises(OSError, loadmat, 'NotExistentFile00')

def test_simplify_cells():
    if False:
        for i in range(10):
            print('nop')
    filename = pjoin(test_data_path, 'testsimplecell.mat')
    res1 = loadmat(filename, simplify_cells=True)
    res2 = loadmat(filename, simplify_cells=False)
    assert_(isinstance(res1['s'], dict))
    assert_(isinstance(res2['s'], np.ndarray))
    assert_array_equal(res1['s']['mycell'], np.array(['a', 'b', 'c']))

@pytest.mark.parametrize('version, filt, regex', [(0, '_4*_*', None), (1, '_5*_*', None), (1, '_6*_*', None), (1, '_7*_*', '^((?!hdf5).)*$'), (2, '_7*_*', '.*hdf5.*'), (1, '8*_*', None)])
def test_matfile_version(version, filt, regex):
    if False:
        while True:
            i = 10
    use_filt = pjoin(test_data_path, 'test*%s.mat' % filt)
    files = glob(use_filt)
    if regex is not None:
        files = [file for file in files if re.match(regex, file) is not None]
    assert len(files) > 0, f'No files for version {version} using filter {filt}'
    for file in files:
        got_version = matfile_version(file)
        assert got_version[0] == version

def test_opaque():
    if False:
        return 10
    'Test that we can read a MatlabOpaque object.'
    data = loadmat(pjoin(test_data_path, 'parabola.mat'))
    assert isinstance(data['parabola'], MatlabFunction)
    assert isinstance(data['parabola'].item()[3].item()[3], MatlabOpaque)

def test_opaque_simplify():
    if False:
        print('Hello World!')
    'Test that we can read a MatlabOpaque object when simplify_cells=True.'
    data = loadmat(pjoin(test_data_path, 'parabola.mat'), simplify_cells=True)
    assert isinstance(data['parabola'], MatlabFunction)

def test_deprecation():
    if False:
        i = 10
        return i + 15
    'Test that access to previous attributes still works.'
    with assert_warns(DeprecationWarning):
        scipy.io.matlab.mio5_params.MatlabOpaque
    with assert_warns(DeprecationWarning):
        from scipy.io.matlab.miobase import MatReadError

def test_gh_17992(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    rng = np.random.default_rng(12345)
    outfile = tmp_path / 'lists.mat'
    array_one = rng.random((5, 3))
    array_two = rng.random((6, 3))
    list_of_arrays = [array_one, array_two]
    with np.testing.suppress_warnings() as sup:
        sup.filter(VisibleDeprecationWarning)
        savemat(outfile, {'data': list_of_arrays}, long_field_names=True, do_compression=True)
    new_dict = {}
    loadmat(outfile, new_dict)
    assert_allclose(new_dict['data'][0][0], array_one)
    assert_allclose(new_dict['data'][0][1], array_two)