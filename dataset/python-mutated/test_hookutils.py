import os
import pytest
import shutil
import re
from PyInstaller.utils.hooks import collect_data_files, collect_submodules, get_module_file_attribute, remove_prefix, remove_suffix, remove_file_extension, is_module_or_submodule, check_requirement
from PyInstaller.compat import exec_python, is_win
from PyInstaller import log as logging

class TestRemovePrefix(object):

    def test_empty_string(self):
        if False:
            return 10
        assert '' == remove_prefix('', 'prefix')

    def test_emptystr_unmodif(self):
        if False:
            return 10
        assert 'test' == remove_prefix('test', '')

    def test_string_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        assert '' == remove_prefix('test', 'test')

    def test_just_prefix(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'ing' == remove_prefix('testing', 'test')

    def test_no_modific(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'atest' == remove_prefix('atest', 'test')

class TestRemoveSuffix(object):

    def test_empty_string(self):
        if False:
            i = 10
            return i + 15
        assert '' == remove_suffix('', 'suffix')

    def test_emptystr_unmodif(self):
        if False:
            for i in range(10):
                print('nop')
        assert 'test' == remove_suffix('test', '')

    def test_string_suffix(self):
        if False:
            i = 10
            return i + 15
        assert '' == remove_suffix('test', 'test')

    def test_just_suffix(self):
        if False:
            print('Hello World!')
        assert 'test' == remove_suffix('testing', 'ing')

    def test_no_modific(self):
        if False:
            print('Hello World!')
        assert 'testa' == remove_suffix('testa', 'test')

class TestRemoveExtension(object):

    def test_no_extension(self):
        if False:
            while True:
                i = 10
        assert 'file' == remove_file_extension('file')

    def test_two_extensions(self):
        if False:
            while True:
                i = 10
        assert 'file.1' == remove_file_extension('file.1.2')

    def test_remove_ext(self):
        if False:
            while True:
                i = 10
        assert 'file' == remove_file_extension('file.1')

    def test_unixstyle_not_ext(self):
        if False:
            return 10
        assert '.file' == remove_file_extension('.file')

    def test_unixstyle_ext(self):
        if False:
            while True:
                i = 10
        assert '.file' == remove_file_extension('.file.1')

    def test_unixstyle_path(self):
        if False:
            while True:
                i = 10
        assert '/a/b/c' == remove_file_extension('/a/b/c')
        assert '/a/b/c' == remove_file_extension('/a/b/c.1')

    def test_win32style_path(self):
        if False:
            i = 10
            return i + 15
        assert 'C:\\a\\b\\c' == remove_file_extension('C:\\a\\b\\c')
        assert 'C:\\a\\b\\c' == remove_file_extension('C:\\a\\b\\c.1')
TEST_MOD = 'hookutils_package'
TEST_MOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hookutils_files')

@pytest.fixture
def mod_list(monkeypatch):
    if False:
        print('Hello World!')
    monkeypatch.syspath_prepend(TEST_MOD_PATH)
    return collect_submodules(TEST_MOD)

class TestCollectSubmodules(object):

    def test_collect_submod_module(self, caplog):
        if False:
            print('Hello World!')
        with caplog.at_level(logging.DEBUG, logger='PyInstaller.utils.hooks'):
            assert collect_submodules('os') == ['os']
            assert 'collect_submodules - os is not a package.' in caplog.records[-1].getMessage()

    def test_not_a_string(self):
        if False:
            while True:
                i = 10
        with pytest.raises(TypeError, match='package must be a str'):
            collect_submodules(os)

    def test_collect_submod_itself(self, mod_list):
        if False:
            i = 10
            return i + 15
        assert TEST_MOD in mod_list

    def test_collect_submod_pyextension(self, mod_list):
        if False:
            return 10
        assert TEST_MOD + '.pyextension' in mod_list

    def test_collect_submod_all_included(self, mod_list):
        if False:
            for i in range(10):
                print('nop')
        mod_list.sort()
        assert mod_list == [TEST_MOD, TEST_MOD + '.pyextension', TEST_MOD + '.subpkg', TEST_MOD + '.subpkg.twelve', TEST_MOD + '.two']

    def test_collect_submod_no_dynamiclib(self, mod_list):
        if False:
            while True:
                i = 10
        assert TEST_MOD + '.dynamiclib' not in mod_list

    def test_collect_submod_subpkg_init(self, mod_list):
        if False:
            for i in range(10):
                print('nop')
        assert TEST_MOD + '.py_files_not_in_package.sub_pkg.three' not in mod_list

    def test_collect_submod_subpkg(self, mod_list):
        if False:
            while True:
                i = 10
        mod_list = collect_submodules(TEST_MOD + '.subpkg')
        mod_list.sort()
        assert mod_list == [TEST_MOD + '.subpkg', TEST_MOD + '.subpkg.twelve']

    def test_collect_submod_egg(self, tmpdir, monkeypatch):
        if False:
            print('Hello World!')
        dest_path = tmpdir.join('hookutils_package')
        shutil.copytree(TEST_MOD_PATH, dest_path.strpath)
        monkeypatch.chdir(dest_path)
        print(exec_python('setup.py', 'bdist_egg'))
        dist_path = dest_path.join('dist')
        fl = os.listdir(dist_path.strpath)
        assert len(fl) == 1
        egg_name = fl[0]
        assert egg_name.endswith('.egg')
        pth = dist_path.join(egg_name).strpath
        monkeypatch.setattr('PyInstaller.config.CONF', {'pathex': [pth]})
        monkeypatch.syspath_prepend(pth)
        ml = collect_submodules(TEST_MOD)
        self.test_collect_submod_all_included(ml)

    def test_collect_submod_stdout_interference(self, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        TEST_MOD = 'foo'
        TEST_MOD_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'hookutils_files2')
        monkeypatch.setattr('PyInstaller.config.CONF', {'pathex': [TEST_MOD_PATH]})
        monkeypatch.syspath_prepend(TEST_MOD_PATH)
        ml = collect_submodules(TEST_MOD)
        ml = sorted(ml)
        assert ml == ['foo', 'foo.bar']

    def test_error_propagation(self, capfd, monkeypatch):
        if False:
            for i in range(10):
                print('nop')
        monkeypatch.setattr('PyInstaller.config.CONF', {'pathex': [TEST_MOD_PATH]})
        monkeypatch.syspath_prepend(TEST_MOD_PATH)
        collect_submodules(TEST_MOD)
        error = capfd.readouterr().err
        assert re.match(".*Failed .* for 'hookutils_package.raises_error_on_import_[12]' because .* raised: AssertionError: I cannot be imported", error)
        assert error.count('Failed') == 1
        collect_submodules(TEST_MOD, on_error='ignore')
        assert capfd.readouterr().err == ''
        collect_submodules(TEST_MOD, on_error='warn')
        error = capfd.readouterr().err
        assert 'raises_error_on_import_1' in error
        assert 'raises_error_on_import_2' in error
        assert error.count('Failed') == 2
        with pytest.raises(RuntimeError) as ex_info:
            collect_submodules(TEST_MOD, on_error='raise')
            assert ex_info.match('(?s).* assert 0, "I cannot be imported!"')
            assert ex_info.match("Unable to load submodule 'hookutils_package.raises_error_on_import_[12]'")

def test_is_module_or_submodule():
    if False:
        return 10
    assert is_module_or_submodule('foo.bar', 'foo.bar')
    assert is_module_or_submodule('foo.bar.baz', 'foo.bar')
    assert not is_module_or_submodule('foo.bard', 'foo.bar')
    assert not is_module_or_submodule('foo', 'foo.bar')

def test_check_requirement_package_not_installed():
    if False:
        print('Hello World!')
    assert check_requirement('pytest')
    assert not check_requirement('magnumopus-no-package-test-case')

def test_collect_data_module():
    if False:
        print('Hello World!')
    with pytest.raises(TypeError):
        collect_data_files(__import__('os'))

@pytest.fixture(params=[([TEST_MOD], {}, ('dynamiclib.dll', 'dynamiclib.dylib', 'nine.dat', os.path.join('py_files_not_in_package', 'data', 'eleven.dat'), os.path.join('py_files_not_in_package', 'ten.dat'), 'pyextension.so' if is_win else 'pyextension.pyd', os.path.join('subpkg', 'thirteen.txt'))), ([TEST_MOD + '.subpkg'], {}, (os.path.join('subpkg', 'thirteen.txt'),)), ([TEST_MOD], dict(include_py_files=True, excludes=['**/__pycache__']), ('__init__.py', 'dynamiclib.dll', 'dynamiclib.dylib', 'nine.dat', os.path.join('py_files_not_in_package', 'data', 'eleven.dat'), os.path.join('py_files_not_in_package', 'one.py'), os.path.join('py_files_not_in_package', 'sub_pkg', '__init__.py'), os.path.join('py_files_not_in_package', 'sub_pkg', 'three.py'), os.path.join('py_files_not_in_package', 'ten.dat'), 'pyextension.so' if is_win else 'pyextension.pyd', os.path.join('raises_error_on_import_1', '__init__.py'), os.path.join('raises_error_on_import_1', 'foo.py'), os.path.join('raises_error_on_import_2', '__init__.py'), os.path.join('raises_error_on_import_2', 'foo.py'), os.path.join('subpkg', '__init__.py'), os.path.join('subpkg', 'thirteen.txt'), os.path.join('subpkg', 'twelve.py'), 'two.py')), ([TEST_MOD], dict(excludes=['py_files_not_in_package', '**/__pycache__']), ('dynamiclib.dll', 'dynamiclib.dylib', 'nine.dat', 'pyextension.so' if is_win else 'pyextension.pyd', os.path.join('subpkg', 'thirteen.txt'))), ([TEST_MOD], dict(includes=['**/*.dat', '**/*.txt']), ('nine.dat', os.path.join('py_files_not_in_package', 'data', 'eleven.dat'), os.path.join('py_files_not_in_package', 'ten.dat'), os.path.join('subpkg', 'thirteen.txt'))), ([TEST_MOD], dict(includes=['*.dat']), ('nine.dat',)), ([TEST_MOD], dict(subdir='py_files_not_in_package', excludes=['**/__pycache__']), (os.path.join('py_files_not_in_package', 'data', 'eleven.dat'), os.path.join('py_files_not_in_package', 'ten.dat')))], ids=['package', 'subpackage', 'package with py files', 'excludes', '** includes', 'includes', 'subdir'])
def data_lists(monkeypatch, request):
    if False:
        while True:
            i = 10

    def _sort(sequence):
        if False:
            print('Hello World!')
        sorted_list = sorted(list(sequence))
        return tuple(sorted_list)
    monkeypatch.setattr('PyInstaller.config.CONF', {'pathex': [TEST_MOD_PATH]})
    monkeypatch.syspath_prepend(TEST_MOD_PATH)
    (args, kwargs, subfiles) = request.param
    data = collect_data_files(*args, **kwargs)
    src = [item[0] for item in data]
    dst = [item[1] for item in data]
    return (subfiles, _sort(src), _sort(dst))

def test_collect_data_all_included(data_lists):
    if False:
        return 10
    (subfiles, src, dst) = data_lists
    src_compare = tuple([os.path.join(TEST_MOD_PATH, TEST_MOD, subpath) for subpath in subfiles])
    dst_compare = [os.path.dirname(os.path.join(TEST_MOD, subpath)) for subpath in subfiles]
    dst_compare.sort()
    dst_compare = tuple(dst_compare)
    assert src == src_compare
    assert dst == dst_compare

def test_get_module_file_attribute_non_exist_module():
    if False:
        for i in range(10):
            print('nop')
    with pytest.raises(ImportError):
        get_module_file_attribute('pyinst_nonexisting_module_name')