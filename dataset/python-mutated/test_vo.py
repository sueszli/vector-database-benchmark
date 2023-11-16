"""
This is a set of regression tests for vo.
"""
import difflib
import gzip
import io
import pathlib
import sys
from unittest import mock
import numpy as np
import pytest
from numpy.testing import assert_array_equal
from astropy.io.votable import tree
from astropy.io.votable.exceptions import W39, VOTableSpecError, VOWarning
from astropy.io.votable.table import parse, parse_single_table, validate
from astropy.io.votable.xmlutil import validate_schema
from astropy.utils.data import get_pkg_data_filename, get_pkg_data_filenames
if hasattr(sys, 'float_repr_style'):
    legacy_float_repr = sys.float_repr_style == 'legacy'
else:
    legacy_float_repr = sys.platform.startswith('win')

def assert_validate_schema(filename, version):
    if False:
        i = 10
        return i + 15
    if sys.platform.startswith('win'):
        return
    try:
        (rc, stdout, stderr) = validate_schema(filename, version)
    except OSError:
        return
    assert rc == 0, 'File did not validate against VOTable schema'

def test_parse_single_table():
    if False:
        print('Hello World!')
    with np.errstate(over='ignore'):
        table = parse_single_table(get_pkg_data_filename('data/regression.xml'))
    assert isinstance(table, tree.TableElement)
    assert len(table.array) == 5

def test_parse_single_table2():
    if False:
        return 10
    with np.errstate(over='ignore'):
        table2 = parse_single_table(get_pkg_data_filename('data/regression.xml'), table_number=1)
    assert isinstance(table2, tree.TableElement)
    assert len(table2.array) == 1
    assert len(table2.array.dtype.names) == 28

def test_parse_single_table3():
    if False:
        return 10
    with pytest.raises(IndexError):
        parse_single_table(get_pkg_data_filename('data/regression.xml'), table_number=3)

def _test_regression(tmp_path, _python_based=False, binary_mode=1):
    if False:
        return 10
    votable = parse(get_pkg_data_filename('data/regression.xml'), _debug_python_based_parser=_python_based)
    table = votable.get_first_table()
    dtypes = [(('string test', 'string_test'), '|O8'), (('fixed string test', 'string_test_2'), '<U10'), ('unicode_test', '|O8'), (('unicode test', 'fixed_unicode_test'), '<U10'), (('string array test', 'string_array_test'), '<U4'), ('unsignedByte', '|u1'), ('short', '<i2'), ('int', '<i4'), ('long', '<i8'), ('double', '<f8'), ('float', '<f4'), ('array', '|O8'), ('bit', '|b1'), ('bitarray', '|b1', (3, 2)), ('bitvararray', '|O8'), ('bitvararray2', '|O8'), ('floatComplex', '<c8'), ('doubleComplex', '<c16'), ('doubleComplexArray', '|O8'), ('doubleComplexArrayFixed', '<c16', (2,)), ('boolean', '|b1'), ('booleanArray', '|b1', (4,)), ('nulls', '<i4'), ('nulls_array', '<i4', (2, 2)), ('precision1', '<f8'), ('precision2', '<f8'), ('doublearray', '|O8'), ('bitarray2', '|b1', (16,))]
    if sys.byteorder == 'big':
        new_dtypes = []
        for dtype in dtypes:
            dtype = list(dtype)
            dtype[1] = dtype[1].replace('<', '>')
            new_dtypes.append(tuple(dtype))
        dtypes = new_dtypes
    assert table.array.dtype == dtypes
    votable.to_xml(str(tmp_path / 'regression.tabledata.xml'), _debug_python_based_parser=_python_based)
    assert_validate_schema(str(tmp_path / 'regression.tabledata.xml'), votable.version)
    if binary_mode == 1:
        votable.get_first_table().format = 'binary'
        votable.version = '1.1'
    elif binary_mode == 2:
        votable.get_first_table()._config['version_1_3_or_later'] = True
        votable.get_first_table().format = 'binary2'
        votable.version = '1.3'
    with open(str(tmp_path / 'regression.binary.xml'), 'wb') as fd:
        votable.to_xml(fd, _debug_python_based_parser=_python_based)
    assert_validate_schema(str(tmp_path / 'regression.binary.xml'), votable.version)
    with open(str(tmp_path / 'regression.binary.xml'), 'rb') as fd:
        votable2 = parse(fd, _debug_python_based_parser=_python_based)
    votable2.get_first_table().format = 'tabledata'
    votable2.to_xml(str(tmp_path / 'regression.bin.tabledata.xml'), _astropy_version='testing', _debug_python_based_parser=_python_based)
    assert_validate_schema(str(tmp_path / 'regression.bin.tabledata.xml'), votable.version)
    with open(get_pkg_data_filename(f'data/regression.bin.tabledata.truth.{votable.version}.xml'), encoding='utf-8') as fd:
        truth = fd.readlines()
    with open(str(tmp_path / 'regression.bin.tabledata.xml'), encoding='utf-8') as fd:
        output = fd.readlines()
    sys.stdout.writelines(difflib.unified_diff(truth, output, fromfile='truth', tofile='output'))
    assert truth == output
    votable2.to_xml(str(tmp_path / 'regression.bin.tabledata.xml.gz'), _astropy_version='testing', _debug_python_based_parser=_python_based)
    with gzip.GzipFile(str(tmp_path / 'regression.bin.tabledata.xml.gz'), 'rb') as gzfd:
        output = gzfd.readlines()
    output = [x.decode('utf-8').rstrip() for x in output]
    truth = [x.rstrip() for x in truth]
    assert truth == output

@pytest.mark.xfail('legacy_float_repr')
def test_regression(tmp_path):
    if False:
        i = 10
        return i + 15
    with pytest.warns(W39), np.errstate(over='ignore'):
        _test_regression(tmp_path, False)

@pytest.mark.xfail('legacy_float_repr')
def test_regression_python_based_parser(tmp_path):
    if False:
        i = 10
        return i + 15
    with pytest.warns(W39), np.errstate(over='ignore'):
        _test_regression(tmp_path, True)

@pytest.mark.xfail('legacy_float_repr')
def test_regression_binary2(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with pytest.warns(W39), np.errstate(over='ignore'):
        _test_regression(tmp_path, False, 2)

class TestFixups:

    def setup_class(self):
        if False:
            i = 10
            return i + 15
        with np.errstate(over='ignore'):
            self.table = parse(get_pkg_data_filename('data/regression.xml')).get_first_table()
        self.array = self.table.array
        self.mask = self.table.array.mask

    def test_implicit_id(self):
        if False:
            return 10
        assert_array_equal(self.array['string_test_2'], self.array['fixed string test'])

class TestReferences:

    def setup_class(self):
        if False:
            return 10
        with np.errstate(over='ignore'):
            self.votable = parse(get_pkg_data_filename('data/regression.xml'))
        self.table = self.votable.get_first_table()
        self.array = self.table.array
        self.mask = self.table.array.mask

    def test_fieldref(self):
        if False:
            i = 10
            return i + 15
        fieldref = self.table.groups[1].entries[0]
        assert isinstance(fieldref, tree.FieldRef)
        assert fieldref.get_ref().name == 'boolean'
        assert fieldref.get_ref().datatype == 'boolean'

    def test_paramref(self):
        if False:
            print('Hello World!')
        paramref = self.table.groups[0].entries[0]
        assert isinstance(paramref, tree.ParamRef)
        assert paramref.get_ref().name == 'INPUT'
        assert paramref.get_ref().datatype == 'float'

    def test_iter_fields_and_params_on_a_group(self):
        if False:
            return 10
        assert len(list(self.table.groups[1].iter_fields_and_params())) == 2

    def test_iter_groups_on_a_group(self):
        if False:
            return 10
        assert len(list(self.table.groups[1].iter_groups())) == 1

    def test_iter_groups(self):
        if False:
            i = 10
            return i + 15
        assert len(list(self.votable.iter_groups())) == 9

    def test_ref_table(self):
        if False:
            while True:
                i = 10
        tables = list(self.votable.iter_tables())
        for (x, y) in zip(tables[0].array.data[0], tables[1].array.data[0]):
            assert_array_equal(x, y)

    def test_iter_coosys(self):
        if False:
            i = 10
            return i + 15
        assert len(list(self.votable.iter_coosys())) == 1

def test_select_columns_by_index():
    if False:
        print('Hello World!')
    columns = [0, 5, 13]
    table = parse(get_pkg_data_filename('data/regression.xml'), columns=columns).get_first_table()
    array = table.array
    mask = table.array.mask
    assert array['string_test'][0] == 'String & test'
    columns = ['string_test', 'unsignedByte', 'bitarray']
    for c in columns:
        assert not np.all(mask[c])
    assert np.all(mask['unicode_test'])

def test_select_columns_by_name():
    if False:
        i = 10
        return i + 15
    columns = ['string_test', 'unsignedByte', 'bitarray']
    table = parse(get_pkg_data_filename('data/regression.xml'), columns=columns).get_first_table()
    array = table.array
    mask = table.array.mask
    assert array['string_test'][0] == 'String & test'
    for c in columns:
        assert not np.all(mask[c])
    assert np.all(mask['unicode_test'])

class TestParse:

    def setup_class(self):
        if False:
            return 10
        with np.errstate(over='ignore'):
            self.votable = parse(get_pkg_data_filename('data/regression.xml'))
        self.table = self.votable.get_first_table()
        self.array = self.table.array
        self.mask = self.table.array.mask

    def test_string_test(self):
        if False:
            return 10
        assert issubclass(self.array['string_test'].dtype.type, np.object_)
        assert_array_equal(self.array['string_test'], ['String & test', 'String &amp; test', 'XXXX', '', ''])

    def test_fixed_string_test(self):
        if False:
            while True:
                i = 10
        assert issubclass(self.array['string_test_2'].dtype.type, np.str_)
        assert_array_equal(self.array['string_test_2'], ['Fixed stri', '0123456789', 'XXXX', '', ''])

    def test_unicode_test(self):
        if False:
            print('Hello World!')
        assert issubclass(self.array['unicode_test'].dtype.type, np.object_)
        assert_array_equal(self.array['unicode_test'], ["Ceçi n'est pas un pipe", 'வணக்கம்', 'XXXX', '', ''])

    def test_fixed_unicode_test(self):
        if False:
            i = 10
            return i + 15
        assert issubclass(self.array['fixed_unicode_test'].dtype.type, np.str_)
        assert_array_equal(self.array['fixed_unicode_test'], ["Ceçi n'est", 'வணக்கம்', '0123456789', '', ''])

    def test_unsignedByte(self):
        if False:
            while True:
                i = 10
        assert issubclass(self.array['unsignedByte'].dtype.type, np.uint8)
        assert_array_equal(self.array['unsignedByte'], [128, 255, 0, 255, 255])
        assert not np.any(self.mask['unsignedByte'])

    def test_short(self):
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(self.array['short'].dtype.type, np.int16)
        assert_array_equal(self.array['short'], [4096, 32767, -4096, 32767, 32767])
        assert not np.any(self.mask['short'])

    def test_int(self):
        if False:
            while True:
                i = 10
        assert issubclass(self.array['int'].dtype.type, np.int32)
        assert_array_equal(self.array['int'], [268435456, 2147483647, -268435456, 268435455, 123456789])
        assert_array_equal(self.mask['int'], [False, False, False, False, True])

    def test_long(self):
        if False:
            while True:
                i = 10
        assert issubclass(self.array['long'].dtype.type, np.int64)
        assert_array_equal(self.array['long'], [922337203685477, 123456789, -1152921504606846976, 1152921504606846975, 123456789])
        assert_array_equal(self.mask['long'], [False, True, False, False, True])

    def test_double(self):
        if False:
            print('Hello World!')
        assert issubclass(self.array['double'].dtype.type, np.float64)
        assert_array_equal(self.array['double'], [8.9990234375, 0.0, np.inf, np.nan, -np.inf])
        assert_array_equal(self.mask['double'], [False, False, False, True, False])

    def test_float(self):
        if False:
            while True:
                i = 10
        assert issubclass(self.array['float'].dtype.type, np.float32)
        assert_array_equal(self.array['float'], [1.0, 0.0, np.inf, np.inf, np.nan])
        assert_array_equal(self.mask['float'], [False, False, False, False, True])

    def test_array(self):
        if False:
            print('Hello World!')
        assert issubclass(self.array['array'].dtype.type, np.object_)
        match = [[], [[42, 32], [12, 32]], [[12, 34], [56, 78], [87, 65], [43, 21]], [[-1, 23]], [[31, -1]]]
        for (a, b) in zip(self.array['array'], match):
            for (a0, b0) in zip(a, b):
                assert issubclass(a0.dtype.type, np.int64)
                assert_array_equal(a0, b0)
        assert self.array.data['array'][3].mask[0][0]
        assert self.array.data['array'][4].mask[0][1]

    def test_bit(self):
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(self.array['bit'].dtype.type, np.bool_)
        assert_array_equal(self.array['bit'], [True, False, True, False, False])

    def test_bit_mask(self):
        if False:
            return 10
        assert_array_equal(self.mask['bit'], [False, False, False, False, True])

    def test_bitarray(self):
        if False:
            while True:
                i = 10
        assert issubclass(self.array['bitarray'].dtype.type, np.bool_)
        assert self.array['bitarray'].shape == (5, 3, 2)
        assert_array_equal(self.array['bitarray'], [[[True, False], [True, True], [False, True]], [[False, True], [False, False], [True, True]], [[True, True], [True, False], [False, False]], [[False, False], [False, False], [False, False]], [[False, False], [False, False], [False, False]]])

    def test_bitarray_mask(self):
        if False:
            print('Hello World!')
        assert_array_equal(self.mask['bitarray'], [[[False, False], [False, False], [False, False]], [[False, False], [False, False], [False, False]], [[False, False], [False, False], [False, False]], [[True, True], [True, True], [True, True]], [[True, True], [True, True], [True, True]]])

    def test_bitvararray(self):
        if False:
            i = 10
            return i + 15
        assert issubclass(self.array['bitvararray'].dtype.type, np.object_)
        match = [[True, True, True], [False, False, False, False, False], [True, False, True, False, True], [], []]
        for (a, b) in zip(self.array['bitvararray'], match):
            assert_array_equal(a, b)
        match_mask = [[False, False, False], [False, False, False, False, False], [False, False, False, False, False], False, False]
        for (a, b) in zip(self.array['bitvararray'], match_mask):
            assert_array_equal(a.mask, b)

    def test_bitvararray2(self):
        if False:
            print('Hello World!')
        assert issubclass(self.array['bitvararray2'].dtype.type, np.object_)
        match = [[], [[[False, True], [False, False], [True, False]], [[True, False], [True, False], [True, False]]], [[[True, True], [True, True], [True, True]]], [], []]
        for (a, b) in zip(self.array['bitvararray2'], match):
            for (a0, b0) in zip(a, b):
                assert a0.shape == (3, 2)
                assert issubclass(a0.dtype.type, np.bool_)
                assert_array_equal(a0, b0)

    def test_floatComplex(self):
        if False:
            i = 10
            return i + 15
        assert issubclass(self.array['floatComplex'].dtype.type, np.complex64)
        assert_array_equal(self.array['floatComplex'], [np.nan + 0j, 0 + 0j, 0 + -1j, np.nan + 0j, np.nan + 0j])
        assert_array_equal(self.mask['floatComplex'], [True, False, False, True, True])

    def test_doubleComplex(self):
        if False:
            return 10
        assert issubclass(self.array['doubleComplex'].dtype.type, np.complex128)
        assert_array_equal(self.array['doubleComplex'], [np.nan + 0j, 0 + 0j, 0 + -1j, np.nan + np.inf * 1j, np.nan + 0j])
        assert_array_equal(self.mask['doubleComplex'], [True, False, False, True, True])

    def test_doubleComplexArray(self):
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(self.array['doubleComplexArray'].dtype.type, np.object_)
        assert [len(x) for x in self.array['doubleComplexArray']] == [0, 2, 2, 0, 0]

    def test_boolean(self):
        if False:
            for i in range(10):
                print('nop')
        assert issubclass(self.array['boolean'].dtype.type, np.bool_)
        assert_array_equal(self.array['boolean'], [True, False, True, False, False])

    def test_boolean_mask(self):
        if False:
            return 10
        assert_array_equal(self.mask['boolean'], [False, False, False, False, True])

    def test_boolean_array(self):
        if False:
            i = 10
            return i + 15
        assert issubclass(self.array['booleanArray'].dtype.type, np.bool_)
        assert_array_equal(self.array['booleanArray'], [[True, True, True, True], [True, True, False, True], [True, True, False, True], [False, False, False, False], [False, False, False, False]])

    def test_boolean_array_mask(self):
        if False:
            while True:
                i = 10
        assert_array_equal(self.mask['booleanArray'], [[False, False, False, False], [False, False, False, False], [False, False, True, False], [True, True, True, True], [True, True, True, True]])

    def test_nulls(self):
        if False:
            print('Hello World!')
        assert_array_equal(self.array['nulls'], [0, -9, 2, -9, -9])
        assert_array_equal(self.mask['nulls'], [False, True, False, True, True])

    def test_nulls_array(self):
        if False:
            print('Hello World!')
        assert_array_equal(self.array['nulls_array'], [[[-9, -9], [-9, -9]], [[0, 1], [2, 3]], [[-9, 0], [-9, 1]], [[0, -9], [1, -9]], [[-9, -9], [-9, -9]]])
        assert_array_equal(self.mask['nulls_array'], [[[True, True], [True, True]], [[False, False], [False, False]], [[True, False], [True, False]], [[False, True], [False, True]], [[True, True], [True, True]]])

    def test_double_array(self):
        if False:
            i = 10
            return i + 15
        assert issubclass(self.array['doublearray'].dtype.type, np.object_)
        assert len(self.array['doublearray'][0]) == 0
        assert_array_equal(self.array['doublearray'][1], [0, 1, np.inf, -np.inf, np.nan, 0, -1])
        assert_array_equal(self.array.data['doublearray'][1].mask, [False, False, False, False, False, False, True])

    def test_bit_array2(self):
        if False:
            for i in range(10):
                print('nop')
        assert_array_equal(self.array['bitarray2'][0], [True, True, True, True, False, False, False, False, True, True, True, True, False, False, False, False])

    def test_bit_array2_mask(self):
        if False:
            print('Hello World!')
        assert not np.any(self.mask['bitarray2'][0])
        assert np.all(self.mask['bitarray2'][1:])

    def test_get_coosys_by_id(self):
        if False:
            return 10
        coosys = self.votable.get_coosys_by_id('J2000')
        assert coosys.system == 'eq_FK5'

    def test_get_field_by_utype(self):
        if False:
            i = 10
            return i + 15
        fields = list(self.votable.get_fields_by_utype('myint'))
        assert fields[0].name == 'int'
        assert fields[0].values.min == -1000

    def test_get_info_by_id(self):
        if False:
            for i in range(10):
                print('nop')
        info = self.votable.get_info_by_id('QUERY_STATUS')
        assert info.value == 'OK'
        if self.votable.version != '1.1':
            info = self.votable.get_info_by_id('ErrorInfo')
            assert info.value == 'One might expect to find some INFO here, too...'

    def test_repr(self):
        if False:
            while True:
                i = 10
        assert '3 tables' in repr(self.votable)
        assert repr(list(self.votable.iter_fields_and_params())[0]) == '<PARAM ID="awesome" arraysize="*" datatype="float" name="INPUT" unit="deg" value="[0.0 0.0]"/>'
        repr(list(self.votable.iter_groups()))
        assert repr(self.votable.resources) == '[</>]'
        assert repr(self.table).startswith('<VOTable')

class TestThroughTableData(TestParse):

    def setup_class(self):
        if False:
            print('Hello World!')
        with np.errstate(over='ignore'):
            votable = parse(get_pkg_data_filename('data/regression.xml'))
        self.xmlout = bio = io.BytesIO()
        with pytest.warns(W39):
            votable.to_xml(bio)
        bio.seek(0)
        self.votable = parse(bio)
        self.table = self.votable.get_first_table()
        self.array = self.table.array
        self.mask = self.table.array.mask

    def test_bit_mask(self):
        if False:
            i = 10
            return i + 15
        assert_array_equal(self.mask['bit'], [False, False, False, False, False])

    def test_bitarray_mask(self):
        if False:
            print('Hello World!')
        assert not np.any(self.mask['bitarray'])

    def test_bit_array2_mask(self):
        if False:
            print('Hello World!')
        assert not np.any(self.mask['bitarray2'])

    def test_schema(self, tmp_path):
        if False:
            while True:
                i = 10
        fn = tmp_path / 'test_through_tabledata.xml'
        with open(fn, 'wb') as f:
            f.write(self.xmlout.getvalue())
        assert_validate_schema(fn, '1.1')

class TestThroughBinary(TestParse):

    def setup_class(self):
        if False:
            print('Hello World!')
        with np.errstate(over='ignore'):
            votable = parse(get_pkg_data_filename('data/regression.xml'))
        votable.get_first_table().format = 'binary'
        self.xmlout = bio = io.BytesIO()
        with pytest.warns(W39):
            votable.to_xml(bio)
        bio.seek(0)
        self.votable = parse(bio)
        self.table = self.votable.get_first_table()
        self.array = self.table.array
        self.mask = self.table.array.mask

    def test_bit_mask(self):
        if False:
            while True:
                i = 10
        assert not np.any(self.mask['bit'])

    def test_bitarray_mask(self):
        if False:
            while True:
                i = 10
        assert not np.any(self.mask['bitarray'])

    def test_bit_array2_mask(self):
        if False:
            i = 10
            return i + 15
        assert not np.any(self.mask['bitarray2'])

class TestThroughBinary2(TestParse):

    def setup_class(self):
        if False:
            for i in range(10):
                print('nop')
        with np.errstate(over='ignore'):
            votable = parse(get_pkg_data_filename('data/regression.xml'))
        votable.version = '1.3'
        votable.get_first_table()._config['version_1_3_or_later'] = True
        votable.get_first_table().format = 'binary2'
        self.xmlout = bio = io.BytesIO()
        with pytest.warns(W39):
            votable.to_xml(bio)
        bio.seek(0)
        self.votable = parse(bio)
        self.table = self.votable.get_first_table()
        self.array = self.table.array
        self.mask = self.table.array.mask

    def test_get_coosys_by_id(self):
        if False:
            for i in range(10):
                print('nop')
        pass

def table_from_scratch():
    if False:
        while True:
            i = 10
    from astropy.io.votable.tree import Field, Resource, TableElement, VOTableFile
    votable = VOTableFile()
    resource = Resource()
    votable.resources.append(resource)
    table = TableElement(votable)
    resource.tables.append(table)
    table.fields.extend([Field(votable, ID='filename', datatype='char'), Field(votable, ID='matrix', datatype='double', arraysize='2x2')])
    table.create_arrays(2)
    table.array[0] = ('test1.xml', [[1, 0], [0, 1]])
    table.array[1] = ('test2.xml', [[0.5, 0.3], [0.2, 0.1]])
    out = io.StringIO()
    votable.to_xml(out)

@np.errstate(over='ignore')
def test_open_files():
    if False:
        return 10
    for filename in get_pkg_data_filenames('data', pattern='*.xml'):
        if not filename.endswith(('custom_datatype.xml', 'timesys_errors.xml', 'parquet_binary.xml')):
            parse(filename)

def test_too_many_columns():
    if False:
        while True:
            i = 10
    with pytest.raises(VOTableSpecError):
        parse(get_pkg_data_filename('data/too_many_columns.xml.gz'))

def test_build_from_scratch(tmp_path):
    if False:
        print('Hello World!')
    votable = tree.VOTableFile()
    resource = tree.Resource()
    votable.resources.append(resource)
    table = tree.TableElement(votable)
    resource.tables.append(table)
    table.fields.extend([tree.Field(votable, ID='filename', name='filename', datatype='char', arraysize='1'), tree.Field(votable, ID='matrix', name='matrix', datatype='double', arraysize='2x2')])
    table.create_arrays(2)
    table.array[0] = ('test1.xml', [[1, 0], [0, 1]])
    table.array[1] = ('test2.xml', [[0.5, 0.3], [0.2, 0.1]])
    votable.to_xml(str(tmp_path / 'new_votable.xml'))
    votable = parse(str(tmp_path / 'new_votable.xml'))
    table = votable.get_first_table()
    assert_array_equal(table.array.mask, np.array([(False, [[False, False], [False, False]]), (False, [[False, False], [False, False]])], dtype=[('filename', '?'), ('matrix', '?', (2, 2))]))

def test_validate(test_path_object=False):
    if False:
        i = 10
        return i + 15
    '\n    test_path_object is needed for test below ``test_validate_path_object``\n    so that file could be passed as pathlib.Path object.\n    '
    output = io.StringIO()
    fpath = get_pkg_data_filename('data/regression.xml')
    if test_path_object:
        fpath = pathlib.Path(fpath)
    result = validate(fpath, output, xmllint=False)
    assert result is False
    output.seek(0)
    output = output.readlines()
    with open(get_pkg_data_filename('data/validation.txt'), encoding='utf-8') as fd:
        truth = fd.readlines()
    truth = truth[1:]
    output = output[1:-1]
    sys.stdout.writelines(difflib.unified_diff(truth, output, fromfile='truth', tofile='output'))
    assert truth == output

@mock.patch('subprocess.Popen')
def test_validate_xmllint_true(mock_subproc_popen):
    if False:
        print('Hello World!')
    process_mock = mock.Mock()
    attrs = {'communicate.return_value': ('ok', 'ko'), 'returncode': 0}
    process_mock.configure_mock(**attrs)
    mock_subproc_popen.return_value = process_mock
    assert validate(get_pkg_data_filename('data/empty_table.xml'), xmllint=True)

def test_validate_path_object():
    if False:
        i = 10
        return i + 15
    'Validating when source is passed as path object (#4412).'
    test_validate(test_path_object=True)

def test_gzip_filehandles(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    with np.errstate(over='ignore'):
        votable = parse(get_pkg_data_filename('data/regression.xml'))
    with pytest.warns(W39):
        with open(tmp_path / 'regression.compressed.xml', 'wb') as fd:
            votable.to_xml(fd, compressed=True, _astropy_version='testing')
    with open(tmp_path / 'regression.compressed.xml', 'rb') as fd:
        votable = parse(fd)

def test_from_scratch_example():
    if False:
        while True:
            i = 10
    _run_test_from_scratch_example()

def _run_test_from_scratch_example():
    if False:
        print('Hello World!')
    from astropy.io.votable.tree import Field, Resource, TableElement, VOTableFile
    votable = VOTableFile()
    resource = Resource()
    votable.resources.append(resource)
    table = TableElement(votable)
    resource.tables.append(table)
    table.fields.extend([Field(votable, name='filename', datatype='char', arraysize='*'), Field(votable, name='matrix', datatype='double', arraysize='2x2')])
    table.create_arrays(2)
    table.array[0] = ('test1.xml', [[1, 0], [0, 1]])
    table.array[1] = ('test2.xml', [[0.5, 0.3], [0.2, 0.1]])
    assert table.array[0][0] == 'test1.xml'

def test_fileobj():
    if False:
        print('Hello World!')
    from astropy.utils.xml import iterparser
    filename = get_pkg_data_filename('data/regression.xml')
    with iterparser._convert_to_fd_or_read_function(filename) as fd:
        if sys.platform == 'win32':
            fd()
        else:
            assert isinstance(fd, io.FileIO)

def test_nonstandard_units():
    if False:
        while True:
            i = 10
    from astropy import units as u
    votable = parse(get_pkg_data_filename('data/nonstandard_units.xml'))
    assert isinstance(votable.get_first_table().fields[0].unit, u.UnrecognizedUnit)
    votable = parse(get_pkg_data_filename('data/nonstandard_units.xml'), unit_format='generic')
    assert not isinstance(votable.get_first_table().fields[0].unit, u.UnrecognizedUnit)

def test_resource_structure():
    if False:
        print('Hello World!')
    from astropy.io.votable import tree as vot
    vtf = vot.VOTableFile()
    r1 = vot.Resource()
    vtf.resources.append(r1)
    t1 = vot.TableElement(vtf)
    t1.name = 't1'
    t2 = vot.TableElement(vtf)
    t2.name = 't2'
    r1.tables.append(t1)
    r1.tables.append(t2)
    r2 = vot.Resource()
    vtf.resources.append(r2)
    t3 = vot.TableElement(vtf)
    t3.name = 't3'
    t4 = vot.TableElement(vtf)
    t4.name = 't4'
    r2.tables.append(t3)
    r2.tables.append(t4)
    r3 = vot.Resource()
    vtf.resources.append(r3)
    t5 = vot.TableElement(vtf)
    t5.name = 't5'
    t6 = vot.TableElement(vtf)
    t6.name = 't6'
    r3.tables.append(t5)
    r3.tables.append(t6)
    buff = io.BytesIO()
    vtf.to_xml(buff)
    buff.seek(0)
    vtf2 = parse(buff)
    assert len(vtf2.resources) == 3
    for r in range(len(vtf2.resources)):
        res = vtf2.resources[r]
        assert len(res.tables) == 2
        assert len(res.resources) == 0

def test_no_resource_check():
    if False:
        print('Hello World!')
    output = io.StringIO()
    result = validate(get_pkg_data_filename('data/no_resource.xml'), output, xmllint=False)
    assert result is False
    output.seek(0)
    output = output.readlines()
    with open(get_pkg_data_filename('data/no_resource.txt'), encoding='utf-8') as fd:
        truth = fd.readlines()
    truth = truth[1:]
    output = output[1:-1]
    sys.stdout.writelines(difflib.unified_diff(truth, output, fromfile='truth', tofile='output'))
    assert truth == output

def test_instantiate_vowarning():
    if False:
        print('Hello World!')
    VOWarning(())

def test_custom_datatype():
    if False:
        while True:
            i = 10
    votable = parse(get_pkg_data_filename('data/custom_datatype.xml'), datatype_mapping={'bar': 'int'})
    table = votable.get_first_table()
    assert table.array.dtype['foo'] == np.int32

def _timesys_tests(votable):
    if False:
        for i in range(10):
            print('nop')
    assert len(list(votable.iter_timesys())) == 4
    timesys = votable.get_timesys_by_id('time_frame')
    assert timesys.timeorigin == 2455197.5
    assert timesys.timescale == 'TCB'
    assert timesys.refposition == 'BARYCENTER'
    timesys = votable.get_timesys_by_id('mjd_origin')
    assert timesys.timeorigin == 'MJD-origin'
    assert timesys.timescale == 'TDB'
    assert timesys.refposition == 'EMBARYCENTER'
    timesys = votable.get_timesys_by_id('jd_origin')
    assert timesys.timeorigin == 'JD-origin'
    assert timesys.timescale == 'TT'
    assert timesys.refposition == 'HELIOCENTER'
    timesys = votable.get_timesys_by_id('no_origin')
    assert timesys.timeorigin is None
    assert timesys.timescale == 'UTC'
    assert timesys.refposition == 'TOPOCENTER'

def test_timesys():
    if False:
        i = 10
        return i + 15
    votable = parse(get_pkg_data_filename('data/timesys.xml'))
    _timesys_tests(votable)

def test_timesys_roundtrip():
    if False:
        while True:
            i = 10
    orig_votable = parse(get_pkg_data_filename('data/timesys.xml'))
    bio = io.BytesIO()
    orig_votable.to_xml(bio)
    bio.seek(0)
    votable = parse(bio)
    _timesys_tests(votable)

def test_timesys_errors():
    if False:
        print('Hello World!')
    output = io.StringIO()
    validate(get_pkg_data_filename('data/timesys_errors.xml'), output, xmllint=False)
    outstr = output.getvalue()
    assert "E23: Invalid timeorigin attribute 'bad-origin'" in outstr
    assert 'E22: ID attribute is required for all TIMESYS elements' in outstr
    assert "W48: Unknown attribute 'refposition_mispelled' on TIMESYS" in outstr

def test_get_infos_by_name():
    if False:
        print('Hello World!')
    vot = parse(io.BytesIO(b'\n        <VOTABLE xmlns="http://www.ivoa.net/xml/VOTable/v1.3"\n          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" version="1.4">\n          <RESOURCE type="results">\n            <INFO name="creator-name" value="Cannon, A."/>\n            <INFO name="creator-name" value="Fleming, W."/>\n          </RESOURCE>\n        </VOTABLE>'))
    infos = vot.get_infos_by_name('creator-name')
    assert [i.value for i in infos] == ['Cannon, A.', 'Fleming, W.']