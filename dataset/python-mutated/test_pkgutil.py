import os
import pytest
from PyInstaller.compat import exec_python_rc
from PyInstaller.utils.tests import importable
_MODULES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'modules')

def _read_results_file(filename):
    if False:
        i = 10
        return i + 15
    output = []
    with open(filename, 'r') as fp:
        for line in fp:
            tokens = line.split(';')
            assert len(tokens) == 2
            output.append((tokens[0], int(tokens[1])))
    return sorted(output)

@pytest.mark.parametrize('package', ['json', 'xml.dom', 'psutil'])
@pytest.mark.parametrize('archive', ['archive', 'noarchive'])
def test_pkgutil_iter_modules(package, script_dir, tmpdir, pyi_builder, archive, resolve_pkg_path=False):
    if False:
        print('Hello World!')
    if not importable(package.split('.')[0]):
        pytest.skip('Needs ' + package)
    test_script = 'pyi_pkgutil_iter_modules.py'
    test_script = os.path.join(script_dir, test_script)
    out_unfrozen = os.path.join(tmpdir, 'output-unfrozen.txt')
    rc = exec_python_rc(test_script, package, '--output-file', out_unfrozen)
    assert rc == 0
    results_unfrozen = _read_results_file(out_unfrozen)
    out_frozen = os.path.join(tmpdir, 'output-frozen.txt')
    debug_args = ['--debug', 'noarchive'] if archive == 'noarchive' else []
    pyi_builder.test_script(test_script, pyi_args=['--collect-submodules', package, *debug_args], app_args=[package, '--output-file', out_frozen] + (['--resolve-pkg-path'] if resolve_pkg_path else []))
    results_frozen = _read_results_file(out_frozen)
    assert results_unfrozen == results_frozen

@pytest.mark.darwin
def test_pkgutil_iter_modules_resolve_pkg_path(script_dir, tmpdir, pyi_builder):
    if False:
        return 10
    if pyi_builder._mode != 'onefile':
        pytest.skip('The test is applicable only to onefile mode.')
    test_pkgutil_iter_modules('json', script_dir, tmpdir, pyi_builder, archive=True, resolve_pkg_path=True)

@pytest.mark.darwin
def test_pkgutil_iter_modules_macos_app_bundle(script_dir, tmpdir, pyi_builder, monkeypatch):
    if False:
        while True:
            i = 10
    if pyi_builder._mode != 'onedir':
        pytest.skip('The test is applicable only to onedir mode.')
    pathex = os.path.join(_MODULES_DIR, 'pyi_pkgutil_itermodules', 'package')
    hooks_dir = os.path.join(_MODULES_DIR, 'pyi_pkgutil_itermodules', 'hooks')
    package = 'mypackage'
    test_script = 'pyi_pkgutil_iter_modules.py'
    test_script = os.path.join(script_dir, test_script)
    env = os.environ.copy()
    if 'PYTHONPATH' in env:
        pathex = os.pathsep.join([pathex, env['PYTHONPATH']])
    env['PYTHONPATH'] = pathex
    out_unfrozen = os.path.join(tmpdir, 'output-unfrozen.txt')
    rc = exec_python_rc(test_script, package, '--output-file', out_unfrozen, env=env)
    assert rc == 0
    results_unfrozen = _read_results_file(out_unfrozen)
    pyi_builder.test_script(test_script, pyi_args=['--paths', pathex, '--hiddenimport', package, '--additional-hooks-dir', hooks_dir, '--windowed'], app_args=[package])
    executables = pyi_builder._find_executables('pyi_pkgutil_iter_modules')
    assert executables
    for (idx, exe) in enumerate(executables):
        out_frozen = os.path.join(tmpdir, f'output-frozen-{idx}.txt')
        rc = pyi_builder._run_executable(exe, args=[package, '--output-file', out_frozen], run_from_path=False, runtime=None)
        assert rc == 0
        results_frozen = _read_results_file(out_frozen)
        print('RESULTS', results_frozen, '\n\n')
        assert results_unfrozen == results_frozen