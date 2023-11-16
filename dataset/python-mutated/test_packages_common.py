import functools
import os
import time
from typing import Any
import pytest
from pytest_pyodide.runner import _BrowserBaseRunner
from conftest import ROOT_PATH, package_is_built
from pyodide_build.io import MetaConfig
PKG_DIR = ROOT_PATH / 'packages'

@functools.cache
def registered_packages() -> list[str]:
    if False:
        return 10
    'Returns a list of registered package names'
    packages = []
    for name in os.listdir(PKG_DIR):
        if (PKG_DIR / name).is_dir() and (PKG_DIR / name / 'meta.yaml').exists():
            packages.append(name)
    return packages
UNSUPPORTED_PACKAGES: dict[str, list[str]] = {'chrome': [], 'firefox': [], 'safari': [], 'node': ['cmyt', 'yt', 'galpy']}
if 'CI' in os.environ:
    UNSUPPORTED_PACKAGES['chrome'].extend(['statsmodels'])
XFAIL_PACKAGES: dict[str, str] = {'soupsieve': 'Importing soupsieve without installing beautifulsoup4 fails.'}

@pytest.mark.parametrize('name', registered_packages())
def test_parse_package(name: str) -> None:
    if False:
        print('Hello World!')
    meta = MetaConfig.from_yaml(PKG_DIR / name / 'meta.yaml')
    sharedlibrary = meta.build.package_type == 'shared_library'
    if name == 'sharedlib-test':
        assert sharedlibrary is True
    elif name == 'sharedlib-test-py':
        assert sharedlibrary is False

@pytest.mark.skip_refcount_check
@pytest.mark.driver_timeout(120)
@pytest.mark.parametrize('name', registered_packages())
@pytest.mark.benchmark(max_time=3, min_rounds=1, timer=time.perf_counter)
def test_import(name: str, selenium_standalone: _BrowserBaseRunner, benchmark: Any) -> None:
    if False:
        return 10
    if not package_is_built(name):
        raise AssertionError('Implementation error. Test for an unbuilt package should have been skipped in selenium_standalone fixture')
    if name in XFAIL_PACKAGES:
        pytest.xfail(XFAIL_PACKAGES[name])
    meta = MetaConfig.from_yaml(PKG_DIR / name / 'meta.yaml')
    if name in UNSUPPORTED_PACKAGES[selenium_standalone.browser]:
        pytest.xfail('{} fails to load and is not supported on {}.'.format(name, selenium_standalone.browser))
    selenium_standalone.run('import glob, os, site')

    def _get_file_count(expr):
        if False:
            print('Hello World!')
        return selenium_standalone.run(f"\n            len(list(glob.glob(\n                site.getsitepackages()[0] + '{expr}',\n                recursive=True)\n            ))\n            ")
    import_names = meta.test.imports
    if not import_names:
        import_names = meta.package.top_level
    if not import_names:
        return

    def _import_pkg():
        if False:
            return 10
        for import_name in import_names:
            selenium_standalone.run_async('import %s' % import_name)
    benchmark(_import_pkg)
    pyc_count = sum((_get_file_count(f'/{key}/**/*.pyc') for key in import_names))
    py_count = sum((_get_file_count(f'/{key}/**/*.py') for key in import_names))
    assert not (py_count > 0 and pyc_count > 0)
    assert _get_file_count('/**/*.exe') == 0