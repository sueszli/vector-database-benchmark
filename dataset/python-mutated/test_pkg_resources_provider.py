import os
from PyInstaller.utils.tests import importorskip
from PyInstaller.compat import is_darwin, exec_python_rc
_MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules')

def __exec_python_script(script_filename, pathex):
    if False:
        for i in range(10):
            print('nop')
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        pathex = os.pathsep.join([pathex, env['PYTHONPATH']])
    env['PYTHONPATH'] = pathex
    return exec_python_rc(script_filename, env=env)

@importorskip('pkg_resources')
def test_pkg_resources_provider_source(tmpdir, script_dir, monkeypatch):
    if False:
        for i in range(10):
            print('nop')
    pathex = os.path.join(_MODULES_DIR, 'pyi_pkg_resources_provider', 'package')
    test_script = os.path.join(script_dir, 'pyi_pkg_resources_provider.py')
    ret = __exec_python_script(test_script, pathex=pathex)
    assert ret == 0, 'Test script failed!'

@importorskip('pkg_resources')
def test_pkg_resources_provider_frozen(pyi_builder, tmpdir, script_dir, monkeypatch):
    if False:
        while True:
            i = 10
    pathex = os.path.join(_MODULES_DIR, 'pyi_pkg_resources_provider', 'package')
    test_script = 'pyi_pkg_resources_provider.py'
    hooks_dir = os.path.join(_MODULES_DIR, 'pyi_pkg_resources_provider', 'hooks')
    pyi_args = ['--paths', pathex, '--hidden-import', 'pyi_pkgres_testpkg', '--additional-hooks-dir', hooks_dir]
    if is_darwin:
        pyi_args += ['--windowed']
    pyi_builder.test_script(test_script, pyi_args=pyi_args)