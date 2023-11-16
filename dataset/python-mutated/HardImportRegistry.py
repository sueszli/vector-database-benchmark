""" Registry for hard import data.

Part of it is static, but modules can get at during scan by plugins that
know how to handle these.
"""
import os
import sys
from nuitka.nodes.ConstantRefNodes import ExpressionConstantSysVersionInfoRef
from nuitka.PythonVersions import getFutureModuleKeys, getImportlibSubPackages, python_version
from nuitka.utils.Utils import isWin32Windows
hard_modules = set(('os', 'ntpath', 'posixpath', 'sys', 'types', 'typing', '__future__', 'importlib', 'importlib.resources', 'importlib.metadata', '_frozen_importlib', '_frozen_importlib_external', 'pkgutil', 'functools', 'sysconfig', 'unittest', 'unittest.mock', 'io', '_io', 'ctypes', 'ctypes.wintypes', 'ctypes.macholib', 'builtins'))
hard_modules_aliases = {'os.path': os.path.__name__}
hard_modules_stdlib = hard_modules
hard_modules_non_stdlib = set(('site', 'pkg_resources', 'importlib_metadata', 'importlib_resources'))
hard_modules = hard_modules | hard_modules_non_stdlib
hard_modules_version = {'cStringIO': (None, 768, None), 'typing': (848, None, None), '_frozen_importlib': (768, None, None), '_frozen_importlib_external': (848, None, None), 'importlib.resources': (880, None, None), 'importlib.metadata': (896, None, None), 'ctypes.wintypes': (None, None, 'win32'), 'builtin': (768, None, None)}
hard_modules_limited = ('importlib.metadata', 'ctypes.wintypes', 'importlib_metadata')
hard_modules_dynamic = set()

def isHardModule(module_name):
    if False:
        i = 10
        return i + 15
    if module_name not in hard_modules:
        return False
    (min_version, max_version, os_limit) = hard_modules_version.get(module_name, (None, None, None))
    if min_version is not None and python_version < min_version:
        return False
    if max_version is not None and python_version >= max_version:
        return False
    if os_limit is not None:
        if os_limit == 'win32':
            return isWin32Windows()
    return True
hard_modules_trust_with_side_effects = set(['site'])
if not isWin32Windows():
    hard_modules_trust_with_side_effects.add('ctypes.wintypes')

def isHardModuleWithoutSideEffect(module_name):
    if False:
        print('Hello World!')
    return module_name in hard_modules and module_name not in hard_modules_trust_with_side_effects
trust_undefined = 0
trust_constant = 1
trust_exist = 2
trust_module = trust_exist
trust_future = trust_exist
trust_importable = 3
trust_node = 4
trust_may_exist = 5
trust_not_exist = 6
trust_node_factory = {}
module_importlib_trust = dict(((key, trust_importable) for key in getImportlibSubPackages()))
if 'metadata' not in module_importlib_trust:
    module_importlib_trust['metadata'] = trust_undefined
if 'resources' not in module_importlib_trust:
    module_importlib_trust['resources'] = trust_undefined
module_sys_trust = {'version': trust_constant, 'hexversion': trust_constant, 'platform': trust_constant, 'maxsize': trust_constant, 'byteorder': trust_constant, 'builtin_module_names': trust_constant, 'stdout': trust_exist, 'stderr': trust_exist}
if python_version < 624:
    module_sys_trust['version_info'] = trust_constant
else:
    module_sys_trust['version_info'] = trust_node
    trust_node_factory['sys', 'version_info'] = ExpressionConstantSysVersionInfoRef
module_builtins_trust = {}
if python_version >= 768:
    module_builtins_trust['open'] = trust_node
if python_version < 768:
    module_sys_trust['exc_type'] = trust_may_exist
    module_sys_trust['exc_value'] = trust_may_exist
    module_sys_trust['exc_traceback'] = trust_may_exist
    module_sys_trust['maxint'] = trust_constant
    module_sys_trust['subversion'] = trust_constant
else:
    module_sys_trust['exc_type'] = trust_not_exist
    module_sys_trust['exc_value'] = trust_not_exist
    module_sys_trust['exc_traceback'] = trust_not_exist
module_typing_trust = {'TYPE_CHECKING': trust_constant}
module_os_trust = {'name': trust_constant, 'listdir': trust_node, 'curdir': trust_constant, 'pardir': trust_constant, 'sep': trust_constant, 'extsep': trust_constant, 'altsep': trust_constant, 'pathsep': trust_constant, 'linesep': trust_constant}
module_os_path_trust = {'exists': trust_node, 'isfile': trust_node, 'isdir': trust_node, 'basename': trust_node}
module_ctypes_trust = {'CDLL': trust_node}
hard_modules_trust = {'os': module_os_trust, 'ntpath': module_os_path_trust if os.path.__name__ == 'ntpath' else {}, 'posixpath': module_os_path_trust if os.path.__name__ == 'posixpath' else {}, 'sys': module_sys_trust, 'types': {}, 'typing': module_typing_trust, '__future__': dict(((key, trust_future) for key in getFutureModuleKeys())), 'importlib': module_importlib_trust, 'importlib.metadata': {'version': trust_node, 'distribution': trust_node, 'metadata': trust_node, 'entry_points': trust_node, 'PackageNotFoundError': trust_exist}, 'importlib_metadata': {'version': trust_node, 'distribution': trust_node, 'metadata': trust_node, 'entry_points': trust_node, 'PackageNotFoundError': trust_exist}, '_frozen_importlib': {}, '_frozen_importlib_external': {}, 'pkgutil': {'get_data': trust_node}, 'functools': {'partial': trust_exist}, 'sysconfig': {}, 'unittest': {'mock': trust_module}, 'unittest.mock': {}, 'io': {'BytesIO': trust_exist, 'StringIO': trust_exist}, '_io': {'BytesIO': trust_exist, 'StringIO': trust_exist}, 'pkg_resources': {'require': trust_node, 'get_distribution': trust_node, 'iter_entry_points': trust_node, 'resource_string': trust_node, 'resource_stream': trust_node}, 'importlib.resources': {'read_binary': trust_node, 'read_text': trust_node, 'files': trust_node}, 'importlib_resources': {'read_binary': trust_node, 'read_text': trust_node, 'files': trust_node}, 'ctypes': module_ctypes_trust, 'site': {}, 'ctypes.wintypes': {}, 'ctypes.macholib': {}, 'builtins': module_builtins_trust}

def _addHardImportNodeClasses():
    if False:
        return 10
    from nuitka.nodes.HardImportNodesGenerated import hard_import_node_classes
    for (hard_import_node_class, spec) in hard_import_node_classes.items():
        (module_name, function_name) = spec.name.rsplit('.', 1)
        if module_name in hard_modules_aliases:
            module_name = hard_modules_aliases.get(module_name)
        trust_node_factory[module_name, function_name] = hard_import_node_class
_addHardImportNodeClasses()
if isWin32Windows():
    module_os_trust['uname'] = trust_not_exist

def _checkHardModules():
    if False:
        for i in range(10):
            print('nop')
    for module_name in hard_modules:
        assert module_name in hard_modules_trust, module_name
    for (module_name, trust) in hard_modules_trust.items():
        assert module_name in hard_modules, module_name
        for (attribute_name, trust_value) in trust.items():
            if trust_value is trust_node:
                assert (module_name, attribute_name) in trust_node_factory or os.path.basename(sys.argv[0]).startswith('generate-'), (module_name, attribute_name)
_checkHardModules()

def addModuleTrust(module_name, attribute_name, trust_value):
    if False:
        while True:
            i = 10
    hard_modules_trust[module_name][attribute_name] = trust_value

def addModuleSingleAttributeNodeFactory(module_name, attribute_name, node_class):
    if False:
        i = 10
        return i + 15
    hard_modules_trust[module_name][attribute_name] = trust_node
    trust_node_factory[module_name, attribute_name] = node_class

def addModuleAttributeFactory(module_name, attribute_name, node_class):
    if False:
        print('Hello World!')
    trust_node_factory[module_name, attribute_name] = node_class

def addModuleDynamicHard(module_name):
    if False:
        return 10
    hard_modules.add(module_name)
    hard_modules_dynamic.add(module_name)
    hard_modules_non_stdlib.add(module_name)
    if module_name not in hard_modules_trust:
        hard_modules_trust[module_name] = {}

def isHardModuleDynamic(module_name):
    if False:
        return 10
    return module_name in hard_modules_dynamic