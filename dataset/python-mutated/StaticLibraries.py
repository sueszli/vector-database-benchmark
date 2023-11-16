""" This module deals with finding and information about static libraries.

"""
import os
from nuitka.containers.OrderedSets import OrderedSet
from nuitka.PythonFlavors import isAnacondaPython, isDebianPackagePython, isNuitkaPython
from nuitka.PythonVersions import getPythonABI, getSystemPrefixPath, python_version, python_version_str
from nuitka.Tracing import general
from .FileOperations import getFileContentByLine, getFileList
from .Utils import getLinuxDistribution, isDebianBasedLinux, isWin32Windows
_ldconf_paths = None
_static_lib_cache = {}

def locateStaticLinkLibrary(dll_name):
    if False:
        while True:
            i = 10
    if dll_name not in _static_lib_cache:
        _static_lib_cache[dll_name] = _locateStaticLinkLibrary(dll_name)
    return _static_lib_cache[dll_name]

def _locateStaticLinkLibrary(dll_name):
    if False:
        print('Hello World!')
    global _ldconf_paths
    if _ldconf_paths is None:
        _ldconf_paths = OrderedSet()
        for conf_filemame in getFileList('/etc/ld.so.conf.d', only_suffixes='.conf'):
            for conf_line in getFileContentByLine(conf_filemame):
                conf_line = conf_line.split('#', 1)[0]
                conf_line = conf_line.strip()
                if os.path.exists(conf_line):
                    _ldconf_paths.add(conf_line)
    for ld_config_path in _ldconf_paths:
        candidate = os.path.join(ld_config_path, 'lib%s.a' % dll_name)
        if os.path.exists(candidate):
            return candidate
    return None
_static_lib_python_path = False

def isDebianSuitableForStaticLinking():
    if False:
        for i in range(10):
            print('nop')
    (dist_name, _base, dist_version) = getLinuxDistribution()
    if dist_name == 'Debian':
        if dist_version is None:
            return True
        try:
            dist_version = tuple((int(x) for x in dist_version.split('.')))
        except ValueError:
            return True
        return dist_version >= (10,)
    else:
        return True

def _getSystemStaticLibPythonPath():
    if False:
        print('Hello World!')
    sys_prefix = getSystemPrefixPath()
    python_abi_version = python_version_str + getPythonABI()
    if isNuitkaPython():
        if isWin32Windows():
            return os.path.join(sys_prefix, 'libs', 'python' + python_abi_version.replace('.', '') + '.lib')
        else:
            return os.path.join(sys_prefix, 'lib', 'libpython' + python_abi_version + '.a')
    if isWin32Windows():
        if isAnacondaPython():
            return None
        candidates = [os.path.join(sys_prefix, 'libs', 'libpython' + python_abi_version.replace('.', '') + '.dll.a'), os.path.join(sys_prefix, 'lib', 'libpython' + python_abi_version + '.dll.a')]
        for candidate in candidates:
            if os.path.exists(candidate):
                return candidate
    else:
        candidate = os.path.join(sys_prefix, 'lib', 'libpython' + python_abi_version + '.a')
        if os.path.exists(candidate):
            return candidate
        if python_version < 768 and isDebianPackagePython() and isDebianSuitableForStaticLinking():
            candidate = locateStaticLinkLibrary('python' + python_abi_version)
        else:
            candidate = None
        if candidate is not None and os.path.exists(candidate):
            if not locateStaticLinkLibrary('z'):
                general.warning("Error, missing 'libz-dev' installation needed for static lib-python.")
            return candidate
        if python_version >= 768 and isDebianPackagePython() and isDebianBasedLinux():
            try:
                import sysconfig
                candidate = os.path.join(sysconfig.get_config_var('LIBPL'), 'libpython' + python_abi_version + '-pic.a')
                if os.path.exists(candidate):
                    return candidate
            except ImportError:
                pass
    return None

def getSystemStaticLibPythonPath():
    if False:
        while True:
            i = 10
    global _static_lib_python_path
    if _static_lib_python_path is False:
        _static_lib_python_path = _getSystemStaticLibPythonPath()
    return _static_lib_python_path