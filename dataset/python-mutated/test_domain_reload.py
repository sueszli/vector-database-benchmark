import subprocess
import os
import platform
import pytest
from find_libpython import find_libpython
libpython = find_libpython()
pytestmark = pytest.mark.xfail(libpython is None, reason="Can't find suitable libpython")

def _run_test(testname):
    if False:
        for i in range(10):
            print('nop')
    dirname = os.path.split(__file__)[0]
    exename = os.path.join(dirname, 'bin', 'Python.DomainReloadTests.exe')
    args = [exename, testname]
    if platform.system() != 'Windows':
        args = ['mono'] + args
    env = os.environ.copy()
    env['PYTHONNET_PYDLL'] = libpython
    proc = subprocess.Popen(args, env=env)
    proc.wait()
    assert proc.returncode == 0

def test_rename_class():
    if False:
        for i in range(10):
            print('nop')
    _run_test('class_rename')

def test_rename_class_member_static_function():
    if False:
        while True:
            i = 10
    _run_test('static_member_rename')

def test_rename_class_member_function():
    if False:
        while True:
            i = 10
    _run_test('member_rename')

def test_rename_class_member_field():
    if False:
        return 10
    _run_test('field_rename')

def test_rename_class_member_property():
    if False:
        return 10
    _run_test('property_rename')

def test_rename_namespace():
    if False:
        return 10
    _run_test('namespace_rename')

def test_field_visibility_change():
    if False:
        while True:
            i = 10
    _run_test('field_visibility_change')

def test_method_visibility_change():
    if False:
        while True:
            i = 10
    _run_test('method_visibility_change')

def test_property_visibility_change():
    if False:
        for i in range(10):
            print('nop')
    _run_test('property_visibility_change')

def test_class_visibility_change():
    if False:
        return 10
    _run_test('class_visibility_change')

def test_method_parameters_change():
    if False:
        print('Hello World!')
    _run_test('method_parameters_change')

def test_method_return_type_change():
    if False:
        i = 10
        return i + 15
    _run_test('method_return_type_change')

def test_field_type_change():
    if False:
        for i in range(10):
            print('nop')
    _run_test('field_type_change')

def test_rename_event():
    if False:
        return 10
    _run_test('event_rename')

def test_construct_removed_class():
    if False:
        return 10
    _run_test('construct_removed_class')

def test_out_to_ref_param():
    if False:
        while True:
            i = 10
    _run_test('out_to_ref_param')

def test_ref_to_out_param():
    if False:
        return 10
    _run_test('ref_to_out_param')

def test_ref_to_in_param():
    if False:
        while True:
            i = 10
    _run_test('ref_to_in_param')

def test_in_to_ref_param():
    if False:
        i = 10
        return i + 15
    _run_test('in_to_ref_param')

def test_nested_type():
    if False:
        while True:
            i = 10
    _run_test('nested_type')

def test_import_after_reload():
    if False:
        for i in range(10):
            print('nop')
    _run_test('import_after_reload')