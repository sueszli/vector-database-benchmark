"""
This module provides decorator functions which can be applied to test objects
in order to skip those objects when certain conditions occur. A sample use case
is to detect if the platform is missing ``matplotlib``. If so, any test objects
which require ``matplotlib`` and decorated with ``@td.skip_if_no_mpl`` will be
skipped by ``pytest`` during the execution of the test suite.

To illustrate, after importing this module:

import pandas.util._test_decorators as td

The decorators can be applied to classes:

@td.skip_if_some_reason
class Foo:
    ...

Or individual functions:

@td.skip_if_some_reason
def test_foo():
    ...

For more information, refer to the ``pytest`` documentation on ``skipif``.
"""
from __future__ import annotations
import locale
from typing import TYPE_CHECKING, Callable
import pytest
from pandas._config import get_option
if TYPE_CHECKING:
    from pandas._typing import F
from pandas._config.config import _get_option
from pandas.compat import IS64, is_platform_windows
from pandas.core.computation.expressions import NUMEXPR_INSTALLED, USE_NUMEXPR
from pandas.util.version import Version

def safe_import(mod_name: str, min_version: str | None=None):
    if False:
        i = 10
        return i + 15
    '\n    Parameters\n    ----------\n    mod_name : str\n        Name of the module to be imported\n    min_version : str, default None\n        Minimum required version of the specified mod_name\n\n    Returns\n    -------\n    object\n        The imported module if successful, or False\n    '
    try:
        mod = __import__(mod_name)
    except ImportError:
        return False
    if not min_version:
        return mod
    else:
        import sys
        version = getattr(sys.modules[mod_name], '__version__')
        if version and Version(version) >= Version(min_version):
            return mod
    return False

def _skip_if_not_us_locale() -> bool:
    if False:
        i = 10
        return i + 15
    (lang, _) = locale.getlocale()
    if lang != 'en_US':
        return True
    return False

def _skip_if_no_scipy() -> bool:
    if False:
        i = 10
        return i + 15
    return not (safe_import('scipy.stats') and safe_import('scipy.sparse') and safe_import('scipy.interpolate') and safe_import('scipy.signal'))

def skip_if_installed(package: str) -> pytest.MarkDecorator:
    if False:
        for i in range(10):
            print('nop')
    '\n    Skip a test if a package is installed.\n\n    Parameters\n    ----------\n    package : str\n        The name of the package.\n\n    Returns\n    -------\n    pytest.MarkDecorator\n        a pytest.mark.skipif to use as either a test decorator or a\n        parametrization mark.\n    '
    return pytest.mark.skipif(safe_import(package), reason=f'Skipping because {package} is installed.')

def skip_if_no(package: str, min_version: str | None=None) -> pytest.MarkDecorator:
    if False:
        i = 10
        return i + 15
    '\n    Generic function to help skip tests when required packages are not\n    present on the testing system.\n\n    This function returns a pytest mark with a skip condition that will be\n    evaluated during test collection. An attempt will be made to import the\n    specified ``package`` and optionally ensure it meets the ``min_version``\n\n    The mark can be used as either a decorator for a test class or to be\n    applied to parameters in pytest.mark.parametrize calls or parametrized\n    fixtures. Use pytest.importorskip if an imported moduled is later needed\n    or for test functions.\n\n    If the import and version check are unsuccessful, then the test function\n    (or test case when used in conjunction with parametrization) will be\n    skipped.\n\n    Parameters\n    ----------\n    package: str\n        The name of the required package.\n    min_version: str or None, default None\n        Optional minimum version of the package.\n\n    Returns\n    -------\n    pytest.MarkDecorator\n        a pytest.mark.skipif to use as either a test decorator or a\n        parametrization mark.\n    '
    msg = f"Could not import '{package}'"
    if min_version:
        msg += f' satisfying a min_version of {min_version}'
    return pytest.mark.skipif(not safe_import(package, min_version=min_version), reason=msg)
skip_if_mpl = pytest.mark.skipif(bool(safe_import('matplotlib')), reason='matplotlib is present')
skip_if_32bit = pytest.mark.skipif(not IS64, reason='skipping for 32 bit')
skip_if_windows = pytest.mark.skipif(is_platform_windows(), reason='Running on Windows')
skip_if_not_us_locale = pytest.mark.skipif(_skip_if_not_us_locale(), reason=f'Specific locale is set {locale.getlocale()[0]}')
skip_if_no_scipy = pytest.mark.skipif(_skip_if_no_scipy(), reason='Missing SciPy requirement')
skip_if_no_ne = pytest.mark.skipif(not USE_NUMEXPR, reason=f'numexpr enabled->{USE_NUMEXPR}, installed->{NUMEXPR_INSTALLED}')

def parametrize_fixture_doc(*args) -> Callable[[F], F]:
    if False:
        for i in range(10):
            print('nop')
    '\n    Intended for use as a decorator for parametrized fixture,\n    this function will wrap the decorated function with a pytest\n    ``parametrize_fixture_doc`` mark. That mark will format\n    initial fixture docstring by replacing placeholders {0}, {1} etc\n    with parameters passed as arguments.\n\n    Parameters\n    ----------\n    args: iterable\n        Positional arguments for docstring.\n\n    Returns\n    -------\n    function\n        The decorated function wrapped within a pytest\n        ``parametrize_fixture_doc`` mark\n    '

    def documented_fixture(fixture):
        if False:
            return 10
        fixture.__doc__ = fixture.__doc__.format(*args)
        return fixture
    return documented_fixture

def mark_array_manager_not_yet_implemented(request) -> None:
    if False:
        return 10
    mark = pytest.mark.xfail(reason='Not yet implemented for ArrayManager')
    request.applymarker(mark)
skip_array_manager_not_yet_implemented = pytest.mark.xfail(_get_option('mode.data_manager', silent=True) == 'array', reason='Not yet implemented for ArrayManager')
skip_array_manager_invalid_test = pytest.mark.skipif(_get_option('mode.data_manager', silent=True) == 'array', reason='Test that relies on BlockManager internals or specific behaviour')
skip_copy_on_write_not_yet_implemented = pytest.mark.xfail(get_option('mode.copy_on_write') is True, reason='Not yet implemented/adapted for Copy-on-Write mode')
skip_copy_on_write_invalid_test = pytest.mark.skipif(get_option('mode.copy_on_write') is True, reason='Test not valid for Copy-on-Write mode')