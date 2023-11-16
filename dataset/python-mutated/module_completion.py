"""
Module completion auxiliary functions.
"""
import pkgutil
from pickleshare import PickleShareDB
from spyder.config.base import get_conf_path
PREFERRED_MODULES = ['numpy', 'scipy', 'sympy', 'pandas', 'networkx', 'statsmodels', 'matplotlib', 'sklearn', 'skimage', 'mpmath', 'os', 'pillow', 'OpenGL', 'array', 'audioop', 'binascii', 'cPickle', 'cStringIO', 'cmath', 'collections', 'datetime', 'errno', 'exceptions', 'gc', 'importlib', 'itertools', 'math', 'mmap', 'msvcrt', 'nt', 'operator', 'ast', 'signal', 'sys', 'threading', 'time', 'wx', 'zipimport', 'zlib', 'pytest', 'PyQt4', 'PyQt5', 'PySide', 'PySide2', 'os.path']

def get_submodules(mod):
    if False:
        return 10
    'Get all submodules of a given module'

    def catch_exceptions(module):
        if False:
            i = 10
            return i + 15
        pass
    try:
        m = __import__(mod)
        submodules = [mod]
        submods = pkgutil.walk_packages(m.__path__, m.__name__ + '.', catch_exceptions)
        for sm in submods:
            sm_name = sm[1]
            submodules.append(sm_name)
    except ImportError:
        return []
    except:
        return [mod]
    return submodules

def get_preferred_submodules():
    if False:
        return 10
    '\n    Get all submodules of the main scientific modules and others of our\n    interest\n    '
    modules_path = get_conf_path('db')
    modules_db = PickleShareDB(modules_path)
    if 'submodules' in modules_db:
        return modules_db['submodules']
    submodules = []
    for m in PREFERRED_MODULES:
        submods = get_submodules(m)
        submodules += submods
    modules_db['submodules'] = submodules
    return submodules