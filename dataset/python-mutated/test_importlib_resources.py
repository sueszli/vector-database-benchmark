import os
import shutil
import pytest
from PyInstaller.utils.tests import skipif
from PyInstaller.compat import is_darwin, is_py39, exec_python, exec_python_rc
from PyInstaller.utils.hooks import check_requirement
_MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules')

def __exec_python_script(script_filename, pathex):
    if False:
        i = 10
        return i + 15
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        pathex = os.pathsep.join([pathex, env['PYTHONPATH']])
    env['PYTHONPATH'] = pathex
    return exec_python_rc(script_filename, env=env)

def __get_test_package_path(package_type, tmpdir, monkeypatch):
    if False:
        while True:
            i = 10
    src_path = os.path.join(_MODULES_DIR, 'pyi_pkg_resources_provider', 'package')
    if package_type == 'pkg':
        return src_path
    dest_path = tmpdir.join('src')
    shutil.copytree(src_path, dest_path.strpath)
    monkeypatch.chdir(dest_path)
    print(exec_python('setup.py', 'bdist_egg'))
    dist_path = dest_path.join('dist')
    files = os.listdir(dist_path.strpath)
    assert len(files) == 1
    egg_name = files[0]
    assert egg_name.endswith('.egg')
    return dist_path.join(egg_name).strpath

@skipif(not is_py39 and (not check_requirement('importlib_resources')), reason='Python prior to 3.9 requires importlib_resources.')
@pytest.mark.parametrize('package_type', ['pkg'])
def test_importlib_resources_source(package_type, tmpdir, script_dir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    pathex = __get_test_package_path(package_type, tmpdir, monkeypatch)
    test_script = 'pyi_importlib_resources.py'
    test_script = os.path.join(str(script_dir), test_script)
    ret = __exec_python_script(test_script, pathex=pathex)
    assert ret == 0, 'Test script failed!'

@skipif(not is_py39 and (not check_requirement('importlib_resources')), reason='Python prior to 3.9 requires importlib_resources.')
@pytest.mark.parametrize('package_type', ['pkg'])
def test_importlib_resources_frozen(pyi_builder, package_type, tmpdir, script_dir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    pathex = __get_test_package_path(package_type, tmpdir, monkeypatch)
    test_script = 'pyi_importlib_resources.py'
    hooks_dir = os.path.join(_MODULES_DIR, 'pyi_pkg_resources_provider', 'hooks')
    pyi_args = ['--paths', pathex, '--hidden-import', 'pyi_pkgres_testpkg', '--additional-hooks-dir', hooks_dir]
    if is_darwin:
        pyi_args += ['--windowed']
    pyi_builder.test_script(test_script, pyi_args=pyi_args)

@skipif(not is_py39 and (not check_requirement('importlib_resources')), reason='Python prior to 3.9 requires importlib_resources.')
@pytest.mark.parametrize('as_package', [True, False])
def test_importlib_resources_namespace_package_data_files(pyi_builder, as_package):
    if False:
        print('Hello World!')
    pathex = os.path.join(_MODULES_DIR, 'pyi_namespace_package_with_data', 'package')
    hooks_dir = os.path.join(_MODULES_DIR, 'pyi_namespace_package_with_data', 'hooks')
    if as_package:
        hidden_imports = ['--hidden-import', 'pyi_test_nspkg', '--hidden-import', 'pyi_test_nspkg.data']
    else:
        hidden_imports = ['--hidden-import', 'pyi_test_nspkg']
    pyi_args = ['--paths', pathex, *hidden_imports, '--additional-hooks-dir', hooks_dir]
    if is_darwin:
        pyi_args += ['--windowed']
    pyi_builder.test_source('\n        import importlib\n        try:\n            import importlib_resources\n        except ModuleNotFoundError:\n            import importlib.resources as importlib_resources\n\n        # Get the package\'s directory (= our data directory)\n        data_dir = importlib_resources.files("pyi_test_nspkg.data")\n\n        # Sanity check; verify the directory\'s base name\n        assert data_dir.name == "data"\n\n        # Check that data files exist\n        assert (data_dir / "data_file1.txt").is_file()\n        assert (data_dir / "data_file2.txt").is_file()\n        assert (data_dir / "data_file3.txt").is_file()\n\n        # Force cache invalidation and check again.\n        # This verifies that our `PyiFrozenImporter` correctly sets the `path_finder` argument when constructing\n        # the `importlib._bootstrap_external._NamespacePath` for the namespace package. The `path_finder` is used\n        # during refresh triggered by cache invalidation.\n        importlib.invalidate_caches()\n\n        data_dir = importlib_resources.files("pyi_test_nspkg.data")\n        assert (data_dir / "data_file1.txt").is_file()\n        ', pyi_args=pyi_args)