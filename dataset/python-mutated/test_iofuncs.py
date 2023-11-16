"""
Tests for iofuncs.py.
"""
import io
import os
import copy
from PIL import ImageFile
import pytest
import numpy as np
import spyder_kernels.utils.iofuncs as iofuncs
LOCATION = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

def are_namespaces_equal(actual, expected):
    if False:
        i = 10
        return i + 15
    if actual is None and expected is None:
        return True
    are_equal = True
    for var in sorted(expected.keys()):
        try:
            are_equal = are_equal and bool(np.mean(expected[var] == actual[var]))
        except ValueError:
            are_equal = are_equal and all([np.all(obj1 == obj2) for (obj1, obj2) in zip(expected[var], actual[var])])
        print(str(var) + ': ' + str(are_equal))
    return are_equal

class CustomObj:
    """A custom class of objects for testing."""

    def __init__(self, data):
        if False:
            while True:
                i = 10
        self.data = None
        if data:
            self.data = data

    def __eq__(self, other):
        if False:
            return 10
        return self.__dict__ == other.__dict__

class UnDeepCopyableObj(CustomObj):
    """A class of objects that cannot be deepcopied."""

    def __getstate__(self):
        if False:
            for i in range(10):
                print('nop')
        raise RuntimeError()

class UnPickleableObj(UnDeepCopyableObj):
    """A class of objects that can deepcopied, but not pickled."""

    def __deepcopy__(self, memo):
        if False:
            for i in range(10):
                print('nop')
        new_one = self.__class__.__new__(self.__class__)
        new_one.__dict__.update(self.__dict__)
        return new_one

@pytest.fixture
def spydata_values():
    if False:
        while True:
            i = 10
    '\n    Define spydata file ground truth values.\n\n    The file export_data.spydata contains five variables to be loaded.\n    This fixture declares those variables in a static way.\n    '
    A = 1
    B = 'ham'
    C = np.eye(3)
    D = {'a': True, 'b': np.eye(4, dtype=np.complex128)}
    E = [np.eye(2, dtype=np.int64), 42.0, np.eye(3, dtype=np.bool_), np.eye(4, dtype=object)]
    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}

@pytest.fixture
def real_values():
    if False:
        return 10
    '\n    Load a Numpy pickled file.\n\n    The file numpy_data.npz contains six variables, each one represents the\n    expected test values after a manual conversion of the same variables\n    defined and evaluated in MATLAB. The manual type conversion was done\n    over several variable types, such as: Matrices/Vectors, Scalar and\n    Complex numbers, Structs, Strings and Cell Arrays. The set of variables\n    was defined to allow and test the deep conversion of a compound type,\n    i.e., a struct that contains other types that need to be converted,\n    like other structs, matrices and Cell Arrays.\n    '
    path = os.path.join(LOCATION, 'numpy_data.npz')
    file_s = np.load(path, allow_pickle=True)
    A = file_s['A'].item()
    B = file_s['B']
    C = file_s['C']
    D = file_s['D'].item()
    E = file_s['E']
    return {'A': A, 'B': B, 'C': C, 'D': D, 'E': E}

@pytest.fixture
def namespace_objects_full(spydata_values):
    if False:
        while True:
            i = 10
    '\n    Define a dictionary of objects of a variety of different types to be saved.\n\n    This fixture reprisents the state of the namespace before saving and\n    filtering out un-deep-copyable, un-pickleable, and uninteresting objects.\n    '
    namespace_dict = copy.deepcopy(spydata_values)
    namespace_dict['expected_error_string'] = 'Some objects could not be saved: undeepcopyable_instance, unpickleable_instance'
    namespace_dict['module_obj'] = io
    namespace_dict['class_obj'] = Exception
    namespace_dict['function_obj'] = os.path.join
    namespace_dict['unpickleable_instance'] = UnPickleableObj('spam')
    namespace_dict['undeepcopyable_instance'] = UnDeepCopyableObj('ham')
    namespace_dict['custom_instance'] = CustomObj('eggs')
    return namespace_dict

@pytest.fixture
def namespace_objects_filtered(spydata_values):
    if False:
        print('Hello World!')
    '\n    Define a dictionary of the objects from the namespace that can be saved.\n\n    This fixture reprisents the state of the namespace after saving and\n    filtering out un-deep-copyable, un-pickleable, and uninteresting objects.\n    '
    namespace_dict = copy.deepcopy(spydata_values)
    namespace_dict['custom_instance'] = CustomObj('eggs')
    return namespace_dict

@pytest.fixture
def namespace_objects_nocopyable():
    if False:
        i = 10
        return i + 15
    '\n    Define a dictionary of that cannot be deepcopied.\n    '
    namespace_dict = {}
    namespace_dict['expected_error_string'] = 'No supported objects to save'
    namespace_dict['class_obj'] = Exception
    namespace_dict['undeepcopyable_instance'] = UnDeepCopyableObj('ham')
    return namespace_dict

@pytest.fixture
def namespace_objects_nopickleable():
    if False:
        return 10
    '\n    Define a dictionary of objects that cannot be pickled.\n    '
    namespace_dict = {}
    namespace_dict['expected_error_string'] = 'No supported objects to save'
    namespace_dict['function_obj'] = os.path.join
    namespace_dict['unpickleable_instance'] = UnPickleableObj('spam')
    return namespace_dict

@pytest.fixture
def input_namespace(request):
    if False:
        print('Hello World!')
    if request.param is None:
        return None
    else:
        return request.getfixturevalue(request.param)

@pytest.fixture
def expected_namespace(request):
    if False:
        print('Hello World!')
    if request.param is None:
        return None
    else:
        return request.getfixturevalue(request.param)

def test_npz_import():
    if False:
        for i in range(10):
            print('nop')
    '\n    Test the load of .npz files as dictionaries.\n    '
    filename = os.path.join(LOCATION, 'import_data.npz')
    data = iofuncs.load_array(filename)
    assert isinstance(data, tuple)
    (variables, error) = data
    assert variables['val1'] == np.array(1) and (not error)

@pytest.mark.skipif(iofuncs.load_matlab is None, reason='SciPy required')
def test_matlab_import(real_values):
    if False:
        i = 10
        return i + 15
    '\n    Test the automatic conversion and import of variables from MATLAB.\n\n    This test loads a file stored in MATLAB, the variables defined are\n    equivalent to the manually converted values done over Numpy. This test\n    allows to evaluate the function which processes the conversion automa-\n    tically. i.e., The automatic conversion results should be equal to the\n    manual conversion of the variables.\n    '
    path = os.path.join(LOCATION, 'data.mat')
    (inf, _) = iofuncs.load_matlab(path)
    valid = True
    for var in sorted(real_values.keys()):
        valid = valid and bool(np.mean(real_values[var] == inf[var]))
    assert valid

@pytest.mark.parametrize('spydata_file_name', ['export_data.spydata', 'export_data_renamed.spydata'])
def test_spydata_import(spydata_file_name, spydata_values):
    if False:
        return 10
    '\n    Test spydata handling and variable importing.\n\n    This test loads all the variables contained inside a spydata tar\n    container and compares them against their static values.\n    It tests both a file with the original name, and one that has been renamed\n    in order to catch Issue #9 .\n    '
    path = os.path.join(LOCATION, spydata_file_name)
    (data, error) = iofuncs.load_dictionary(path)
    assert error is None
    assert are_namespaces_equal(data, spydata_values)

def test_spydata_import_witherror():
    if False:
        i = 10
        return i + 15
    '\n    Test that import fails gracefully with a fn not present in the namespace.\n\n    Checks that the error is caught, the message is passed back,\n    and the current working directory is restored afterwards.\n    '
    original_cwd = os.getcwd()
    path = os.path.join(LOCATION, 'export_data_withfunction.spydata')
    (data, error) = iofuncs.load_dictionary(path)
    assert error and isinstance(error, str)
    assert data is None
    assert os.getcwd() == original_cwd

def test_spydata_import_missing_file():
    if False:
        i = 10
        return i + 15
    '\n    Test that import fails properly when file is missing, and resets the cwd.\n    '
    original_cwd = os.getcwd()
    path = os.path.join(LOCATION, 'non_existant_path_2019-01-23.spydata')
    try:
        iofuncs.load_dictionary(path)
    except IOError:
        pass
    else:
        assert False
    assert os.getcwd() == original_cwd

@pytest.mark.skipif(iofuncs.load_matlab is None, reason='SciPy required')
def test_matlabstruct():
    if False:
        print('Hello World!')
    'Test support for matlab stlye struct.'
    a = iofuncs.MatlabStruct()
    a.b = 'spam'
    assert a['b'] == 'spam'
    a.c['d'] = 'eggs'
    assert a.c.d == 'eggs'
    assert a == {'c': {'d': 'eggs'}, 'b': 'spam'}
    a['d'] = [1, 2, 3]
    buf = io.BytesIO()
    iofuncs.save_matlab(a, buf)
    buf.seek(0)
    (data, error) = iofuncs.load_matlab(buf)
    assert error is None
    assert data['b'] == 'spam'
    assert data['c'].d == 'eggs'
    assert data['d'].tolist() == [[1, 2, 3]]

@pytest.mark.parametrize('input_namespace,expected_namespace,filename', [('spydata_values', 'spydata_values', 'export_data_copy'), ('namespace_objects_full', 'namespace_objects_filtered', 'export_data_2'), ('namespace_objects_nocopyable', None, 'export_data_none_1'), ('namespace_objects_nopickleable', None, 'export_data_none_2')], indirect=['input_namespace', 'expected_namespace'])
def test_spydata_export(input_namespace, expected_namespace, filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test spydata export and re-import.\n\n    This test saves the variables in ``spydata`` format and then\n    reloads and checks them to make sure they save/restore properly\n    and no errors occur during the process.\n    '
    path = os.path.join(LOCATION, filename + '.spydata')
    expected_error = None
    if 'expected_error_string' in input_namespace:
        expected_error = input_namespace['expected_error_string']
        del input_namespace['expected_error_string']
    cwd_original = os.getcwd()
    try:
        export_error = iofuncs.save_dictionary(input_namespace, path)
        assert export_error == expected_error
        if expected_namespace is None:
            assert not os.path.isfile(path)
        else:
            (data_actual, import_error) = iofuncs.load_dictionary(path)
            assert import_error is None
            print(data_actual.keys())
            print(expected_namespace.keys())
            assert are_namespaces_equal(data_actual, expected_namespace)
        assert cwd_original == os.getcwd()
    finally:
        if os.path.isfile(path):
            try:
                os.remove(path)
            except (IOError, OSError, PermissionError):
                pass

def test_save_load_hdf5_files(tmp_path):
    if False:
        for i in range(10):
            print('nop')
    'Simple test to check that we can save and load HDF5 files.'
    h5_file = tmp_path / 'test.h5'
    data = {'a': [1, 2, 3, 4], 'b': 4.5}
    iofuncs.save_hdf5(data, h5_file)
    expected = ({'a': np.array([1, 2, 3, 4]), 'b': np.array(4.5)}, None)
    assert repr(iofuncs.load_hdf5(h5_file)) == repr(expected)

def test_load_dicom_files():
    if False:
        while True:
            i = 10
    'Check that we can load DICOM files.'
    ImageFile.LOAD_TRUNCATED_IMAGES = True
    data = iofuncs.load_dicom(os.path.join(LOCATION, 'data.dcm'))
    assert data[0]['data'].shape == (512, 512)
if __name__ == '__main__':
    pytest.main()