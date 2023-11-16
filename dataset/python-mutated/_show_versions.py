"""
Utility methods to print system info for debugging

adapted from :func:`pandas.show_versions`
"""
import platform
import sys
from .. import __version__
from ..utils.fixes import threadpool_info
from ._openmp_helpers import _openmp_parallelism_enabled

def _get_sys_info():
    if False:
        i = 10
        return i + 15
    'System information\n\n    Returns\n    -------\n    sys_info : dict\n        system and Python version information\n\n    '
    python = sys.version.replace('\n', ' ')
    blob = [('python', python), ('executable', sys.executable), ('machine', platform.platform())]
    return dict(blob)

def _get_deps_info():
    if False:
        return 10
    'Overview of the installed version of main dependencies\n\n    This function does not import the modules to collect the version numbers\n    but instead relies on standard Python package metadata.\n\n    Returns\n    -------\n    deps_info: dict\n        version information on relevant Python libraries\n\n    '
    deps = ['pip', 'setuptools', 'numpy', 'scipy', 'Cython', 'pandas', 'matplotlib', 'joblib', 'threadpoolctl']
    deps_info = {'sklearn': __version__}
    from importlib.metadata import PackageNotFoundError, version
    for modname in deps:
        try:
            deps_info[modname] = version(modname)
        except PackageNotFoundError:
            deps_info[modname] = None
    return deps_info

def show_versions():
    if False:
        while True:
            i = 10
    'Print useful debugging information"\n\n    .. versionadded:: 0.20\n    '
    sys_info = _get_sys_info()
    deps_info = _get_deps_info()
    print('\nSystem:')
    for (k, stat) in sys_info.items():
        print('{k:>10}: {stat}'.format(k=k, stat=stat))
    print('\nPython dependencies:')
    for (k, stat) in deps_info.items():
        print('{k:>13}: {stat}'.format(k=k, stat=stat))
    print('\n{k}: {stat}'.format(k='Built with OpenMP', stat=_openmp_parallelism_enabled()))
    threadpool_results = threadpool_info()
    if threadpool_results:
        print()
        print('threadpoolctl info:')
        for (i, result) in enumerate(threadpool_results):
            for (key, val) in result.items():
                print(f'{key:>15}: {val}')
            if i != len(threadpool_results) - 1:
                print()