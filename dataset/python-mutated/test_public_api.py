import sys
import sysconfig
import subprocess
import pkgutil
import types
import importlib
import inspect
import warnings
import numpy as np
import numpy
from numpy.testing import IS_WASM
import pytest
try:
    import ctypes
except ImportError:
    ctypes = None

def check_dir(module, module_name=None):
    if False:
        i = 10
        return i + 15
    'Returns a mapping of all objects with the wrong __module__ attribute.'
    if module_name is None:
        module_name = module.__name__
    results = {}
    for name in dir(module):
        if name == 'core':
            continue
        item = getattr(module, name)
        if hasattr(item, '__module__') and hasattr(item, '__name__') and (item.__module__ != module_name):
            results[name] = item.__module__ + '.' + item.__name__
    return results

def test_numpy_namespace():
    if False:
        for i in range(10):
            print('nop')
    allowlist = {'recarray': 'numpy.rec.recarray', 'show_config': 'numpy.__config__.show'}
    bad_results = check_dir(np)
    assert bad_results == allowlist

@pytest.mark.skipif(IS_WASM, reason="can't start subprocess")
@pytest.mark.parametrize('name', ['testing'])
def test_import_lazy_import(name):
    if False:
        print('Hello World!')
    "Make sure we can actually use the modules we lazy load.\n\n    While not exported as part of the public API, it was accessible.  With the\n    use of __getattr__ and __dir__, this isn't always true It can happen that\n    an infinite recursion may happen.\n\n    This is the only way I found that would force the failure to appear on the\n    badly implemented code.\n\n    We also test for the presence of the lazily imported modules in dir\n\n    "
    exe = (sys.executable, '-c', 'import numpy; numpy.' + name)
    result = subprocess.check_output(exe)
    assert not result
    assert name in dir(np)

def test_dir_testing():
    if False:
        while True:
            i = 10
    'Assert that output of dir has only one "testing/tester"\n    attribute without duplicate'
    assert len(dir(np)) == len(set(dir(np)))

def test_numpy_linalg():
    if False:
        i = 10
        return i + 15
    bad_results = check_dir(np.linalg)
    assert bad_results == {}

def test_numpy_fft():
    if False:
        i = 10
        return i + 15
    bad_results = check_dir(np.fft)
    assert bad_results == {}

@pytest.mark.skipif(ctypes is None, reason='ctypes not available in this python')
def test_NPY_NO_EXPORT():
    if False:
        i = 10
        return i + 15
    cdll = ctypes.CDLL(np._core._multiarray_tests.__file__)
    f = getattr(cdll, 'test_not_exported', None)
    assert f is None, "'test_not_exported' is mistakenly exported, NPY_NO_EXPORT does not work"
PUBLIC_MODULES = ['numpy.' + s for s in ['array_api', 'array_api.linalg', 'ctypeslib', 'dtypes', 'exceptions', 'f2py', 'fft', 'lib', 'lib.format', 'lib.mixins', 'lib.recfunctions', 'lib.scimath', 'lib.stride_tricks', 'lib.npyio', 'lib.introspect', 'lib.array_utils', 'linalg', 'ma', 'ma.extras', 'ma.mrecords', 'polynomial', 'polynomial.chebyshev', 'polynomial.hermite', 'polynomial.hermite_e', 'polynomial.laguerre', 'polynomial.legendre', 'polynomial.polynomial', 'random', 'testing', 'testing.overrides', 'typing', 'typing.mypy_plugin', 'version']]
if sys.version_info < (3, 12):
    PUBLIC_MODULES += ['numpy.' + s for s in ['distutils', 'distutils.cpuinfo', 'distutils.exec_command', 'distutils.misc_util', 'distutils.log', 'distutils.system_info']]
PUBLIC_ALIASED_MODULES = ['numpy.char', 'numpy.emath', 'numpy.rec']
PRIVATE_BUT_PRESENT_MODULES = ['numpy.' + s for s in ['compat', 'compat.py3k', 'conftest', 'core', 'core.multiarray', 'core.numeric', 'core.umath', 'core.arrayprint', 'core.defchararray', 'core.einsumfunc', 'core.fromnumeric', 'core.function_base', 'core.getlimits', 'core.numerictypes', 'core.overrides', 'core.records', 'core.shape_base', 'f2py.auxfuncs', 'f2py.capi_maps', 'f2py.cb_rules', 'f2py.cfuncs', 'f2py.common_rules', 'f2py.crackfortran', 'f2py.diagnose', 'f2py.f2py2e', 'f2py.f90mod_rules', 'f2py.func2subr', 'f2py.rules', 'f2py.symbolic', 'f2py.use_rules', 'fft.helper', 'lib.user_array', 'linalg.lapack_lite', 'linalg.linalg', 'ma.core', 'ma.testutils', 'ma.timer_comparison', 'matlib', 'matrixlib', 'matrixlib.defmatrix', 'polynomial.polyutils', 'random.mtrand', 'random.bit_generator', 'testing.print_coercion_tables']]
if sys.version_info < (3, 12):
    PRIVATE_BUT_PRESENT_MODULES += ['numpy.' + s for s in ['distutils.armccompiler', 'distutils.fujitsuccompiler', 'distutils.ccompiler', 'distutils.ccompiler_opt', 'distutils.command', 'distutils.command.autodist', 'distutils.command.bdist_rpm', 'distutils.command.build', 'distutils.command.build_clib', 'distutils.command.build_ext', 'distutils.command.build_py', 'distutils.command.build_scripts', 'distutils.command.build_src', 'distutils.command.config', 'distutils.command.config_compiler', 'distutils.command.develop', 'distutils.command.egg_info', 'distutils.command.install', 'distutils.command.install_clib', 'distutils.command.install_data', 'distutils.command.install_headers', 'distutils.command.sdist', 'distutils.conv_template', 'distutils.core', 'distutils.extension', 'distutils.fcompiler', 'distutils.fcompiler.absoft', 'distutils.fcompiler.arm', 'distutils.fcompiler.compaq', 'distutils.fcompiler.environment', 'distutils.fcompiler.g95', 'distutils.fcompiler.gnu', 'distutils.fcompiler.hpux', 'distutils.fcompiler.ibm', 'distutils.fcompiler.intel', 'distutils.fcompiler.lahey', 'distutils.fcompiler.mips', 'distutils.fcompiler.nag', 'distutils.fcompiler.none', 'distutils.fcompiler.pathf95', 'distutils.fcompiler.pg', 'distutils.fcompiler.nv', 'distutils.fcompiler.sun', 'distutils.fcompiler.vast', 'distutils.fcompiler.fujitsu', 'distutils.from_template', 'distutils.intelccompiler', 'distutils.lib2def', 'distutils.line_endings', 'distutils.mingw32ccompiler', 'distutils.msvccompiler', 'distutils.npy_pkg_config', 'distutils.numpy_distribution', 'distutils.pathccompiler', 'distutils.unixccompiler']]

def is_unexpected(name):
    if False:
        return 10
    'Check if this needs to be considered.'
    if '._' in name or '.tests' in name or '.setup' in name:
        return False
    if name in PUBLIC_MODULES:
        return False
    if name in PUBLIC_ALIASED_MODULES:
        return False
    if name in PRIVATE_BUT_PRESENT_MODULES:
        return False
    return True
if sys.version_info < (3, 12):
    SKIP_LIST = ['numpy.distutils.msvc9compiler']
else:
    SKIP_LIST = []

@pytest.mark.filterwarnings('ignore:.*np.compat.*:DeprecationWarning')
def test_all_modules_are_expected():
    if False:
        for i in range(10):
            print('nop')
    "\n    Test that we don't add anything that looks like a new public module by\n    accident.  Check is based on filenames.\n    "
    modnames = []
    for (_, modname, ispkg) in pkgutil.walk_packages(path=np.__path__, prefix=np.__name__ + '.', onerror=None):
        if is_unexpected(modname) and modname not in SKIP_LIST:
            modnames.append(modname)
    if modnames:
        raise AssertionError(f'Found unexpected modules: {modnames}')
SKIP_LIST_2 = ['numpy.math', 'numpy.lib.emath', 'numpy.lib.math', 'numpy.matlib.char', 'numpy.matlib.rec', 'numpy.matlib.emath', 'numpy.matlib.exceptions', 'numpy.matlib.math', 'numpy.matlib.linalg', 'numpy.matlib.fft', 'numpy.matlib.random', 'numpy.matlib.ctypeslib', 'numpy.matlib.ma']
if sys.version_info < (3, 12):
    SKIP_LIST_2 += ['numpy.distutils.log.sys', 'numpy.distutils.log.logging', 'numpy.distutils.log.warnings']

def test_all_modules_are_expected_2():
    if False:
        return 10
    "\n    Method checking all objects. The pkgutil-based method in\n    `test_all_modules_are_expected` does not catch imports into a namespace,\n    only filenames.  So this test is more thorough, and checks this like:\n\n        import .lib.scimath as emath\n\n    To check if something in a module is (effectively) public, one can check if\n    there's anything in that namespace that's a public function/object but is\n    not exposed in a higher-level namespace.  For example for a `numpy.lib`\n    submodule::\n\n        mod = np.lib.mixins\n        for obj in mod.__all__:\n            if obj in np.__all__:\n                continue\n            elif obj in np.lib.__all__:\n                continue\n\n            else:\n                print(obj)\n\n    "

    def find_unexpected_members(mod_name):
        if False:
            print('Hello World!')
        members = []
        module = importlib.import_module(mod_name)
        if hasattr(module, '__all__'):
            objnames = module.__all__
        else:
            objnames = dir(module)
        for objname in objnames:
            if not objname.startswith('_'):
                fullobjname = mod_name + '.' + objname
                if isinstance(getattr(module, objname), types.ModuleType):
                    if is_unexpected(fullobjname):
                        if fullobjname not in SKIP_LIST_2:
                            members.append(fullobjname)
        return members
    unexpected_members = find_unexpected_members('numpy')
    for modname in PUBLIC_MODULES:
        unexpected_members.extend(find_unexpected_members(modname))
    if unexpected_members:
        raise AssertionError('Found unexpected object(s) that look like modules: {}'.format(unexpected_members))

def test_api_importable():
    if False:
        print('Hello World!')
    '\n    Check that all submodules listed higher up in this file can be imported\n\n    Note that if a PRIVATE_BUT_PRESENT_MODULES entry goes missing, it may\n    simply need to be removed from the list (deprecation may or may not be\n    needed - apply common sense).\n    '

    def check_importable(module_name):
        if False:
            i = 10
            return i + 15
        try:
            importlib.import_module(module_name)
        except (ImportError, AttributeError):
            return False
        return True
    module_names = []
    for module_name in PUBLIC_MODULES:
        if not check_importable(module_name):
            module_names.append(module_name)
    if module_names:
        raise AssertionError('Modules in the public API that cannot be imported: {}'.format(module_names))
    for module_name in PUBLIC_ALIASED_MODULES:
        try:
            eval(module_name)
        except AttributeError:
            module_names.append(module_name)
    if module_names:
        raise AssertionError('Modules in the public API that were not found: {}'.format(module_names))
    with warnings.catch_warnings(record=True) as w:
        warnings.filterwarnings('always', category=DeprecationWarning)
        warnings.filterwarnings('always', category=ImportWarning)
        for module_name in PRIVATE_BUT_PRESENT_MODULES:
            if not check_importable(module_name):
                module_names.append(module_name)
    if module_names:
        raise AssertionError('Modules that are not really public but looked public and can not be imported: {}'.format(module_names))

@pytest.mark.xfail(sysconfig.get_config_var('Py_DEBUG') not in (None, 0, '0'), reason='NumPy possibly built with `USE_DEBUG=True ./tools/travis-test.sh`, which does not expose the `array_api` entry point. See https://github.com/numpy/numpy/pull/19800')
def test_array_api_entry_point():
    if False:
        while True:
            i = 10
    '\n    Entry point for Array API implementation can be found with importlib and\n    returns the numpy.array_api namespace.\n    '
    numpy_in_sitepackages = sysconfig.get_path('platlib') in np.__file__
    eps = importlib.metadata.entry_points()
    try:
        xp_eps = eps.select(group='array_api')
    except AttributeError:
        xp_eps = eps.get('array_api', [])
    if len(xp_eps) == 0:
        if numpy_in_sitepackages:
            msg = "No entry points for 'array_api' found"
            raise AssertionError(msg) from None
        return
    try:
        ep = next((ep for ep in xp_eps if ep.name == 'numpy'))
    except StopIteration:
        if numpy_in_sitepackages:
            msg = "'numpy' not in array_api entry points"
            raise AssertionError(msg) from None
        return
    xp = ep.load()
    msg = f"numpy entry point value '{ep.value}' does not point to our Array API implementation"
    assert xp is numpy.array_api, msg

def test_main_namespace_all_dir_coherence():
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks if `dir(np)` and `np.__all__` are consistent\n    and return same content, excluding private members.\n    '

    def _remove_private_members(member_set):
        if False:
            for i in range(10):
                print('nop')
        return {m for m in member_set if not m.startswith('_')}
    all_members = _remove_private_members(np.__all__)
    dir_members = _remove_private_members(np.__dir__())
    assert all_members == dir_members, f'Members that break symmetry: {all_members.symmetric_difference(dir_members)}'

@pytest.mark.filterwarnings('ignore:numpy.core(\\.\\w+)? is deprecated:DeprecationWarning')
def test_core_shims_coherence():
    if False:
        i = 10
        return i + 15
    '\n    Check that all "semi-public" members of `numpy._core` are also accessible\n    from `numpy.core` shims.\n    '
    import numpy.core as core
    for member_name in dir(np._core):
        if member_name.startswith('_') or member_name == 'tests':
            continue
        member = getattr(np._core, member_name)
        if inspect.ismodule(member):
            submodule = member
            submodule_name = member_name
            for submodule_member_name in dir(submodule):
                if submodule_member_name.startswith('__'):
                    continue
                submodule_member = getattr(submodule, submodule_member_name)
                core_submodule = __import__(f'numpy.core.{submodule_name}', fromlist=[submodule_member_name])
                assert submodule_member is getattr(core_submodule, submodule_member_name)
        else:
            assert member is getattr(core, member_name)

def test_functions_single_location():
    if False:
        print('Hello World!')
    "\n    Check that each public function is available from one location only.\n\n    Test performs BFS search traversing NumPy's public API. It flags\n    any function-like object that is accessible from more that one place.\n    "
    from typing import Any, Callable, Dict, List, Set, Tuple
    from numpy._core._multiarray_umath import _ArrayFunctionDispatcher as dispatched_function
    visited_modules: Set[types.ModuleType] = {np}
    visited_functions: Set[Callable[..., Any]] = set()
    functions_original_paths: Dict[Callable[..., Any], str] = dict()
    duplicated_functions: List[Tuple] = []
    modules_queue = [np]
    while len(modules_queue) > 0:
        module = modules_queue.pop()
        for member_name in dir(module):
            member = getattr(module, member_name)
            if inspect.ismodule(member) and 'numpy' in member.__name__ and (not member_name.startswith('_')) and ('numpy._core' not in member.__name__) and (member_name not in ['f2py', 'ma', 'testing', 'tests']) and (member not in visited_modules):
                modules_queue.append(member)
                visited_modules.add(member)
            elif inspect.isfunction(member) or isinstance(member, dispatched_function) or isinstance(member, np.ufunc):
                if member in visited_functions:
                    if member.__name__ in ['absolute', 'conjugate', 'invert', 'remainder', 'divide'] and module.__name__ == 'numpy':
                        continue
                    if member.__name__ == 'trimcoef' and module.__name__.startswith('numpy.polynomial'):
                        continue
                    duplicated_functions.append((member.__name__, module.__name__, functions_original_paths[member]))
                else:
                    visited_functions.add(member)
                    functions_original_paths[member] = module.__name__
    del visited_functions, visited_modules, functions_original_paths
    assert len(duplicated_functions) == 0, duplicated_functions