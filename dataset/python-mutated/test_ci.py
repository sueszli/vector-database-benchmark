from __future__ import annotations
import os
import importlib_metadata
import pytest
from packaging.version import Version

@pytest.mark.xfail(reason='https://github.com/dask/dask/issues/9735', strict=False)
@pytest.mark.skipif(not os.environ.get('UPSTREAM_DEV', False), reason='Only check for dev packages in `upstream` CI build')
def test_upstream_packages_installed():
    if False:
        print('Hello World!')
    packages = ['bokeh', 'numpy', 'pandas', 'pyarrow', 'scipy']
    for package in packages:
        v = Version(importlib_metadata.version(package))
        assert v.is_prerelease or v.local is not None, (package, str(v))