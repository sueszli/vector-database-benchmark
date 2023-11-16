""" Python version specifics.

This abstracts the Python version decisions. This makes decisions based on
the numbers, and attempts to give them meaningful names. Where possible it
should attempt to make run time detections.

"""
import __future__
import ctypes
import os
import re
import sys

def getSupportedPythonVersions():
    if False:
        i = 10
        return i + 15
    'Officially supported Python versions for Nuitka.'
    return ('2.6', '2.7', '3.3', '3.4', '3.5', '3.6', '3.7', '3.8', '3.9', '3.10', '3.11')

def getNotYetSupportedPythonVersions():
    if False:
        for i in range(10):
            print('nop')
    'Versions known to not work at all (yet).'
    return ('3.12',)

def getPartiallySupportedPythonVersions():
    if False:
        print('Hello World!')
    'Partially supported Python versions for Nuitka.'
    return ()

def getZstandardSupportingVersions():
    if False:
        print('Hello World!')
    result = getSupportedPythonVersions() + getPartiallySupportedPythonVersions()
    result = tuple((version for version in result if version not in ('2.6', '2.7', '3.3', '3.4')))
    return result

def getTestExecutionPythonVersions():
    if False:
        return 10
    return getSupportedPythonVersions() + getPartiallySupportedPythonVersions() + getNotYetSupportedPythonVersions()
assert len(set(getPartiallySupportedPythonVersions() + getNotYetSupportedPythonVersions() + getSupportedPythonVersions())) == len(getPartiallySupportedPythonVersions() + getNotYetSupportedPythonVersions() + getSupportedPythonVersions())

def getSupportedPythonVersionStr():
    if False:
        while True:
            i = 10
    supported_python_versions = getSupportedPythonVersions()
    supported_python_versions_str = repr(supported_python_versions)[1:-1]
    supported_python_versions_str = re.sub('(.*),(.*)$', '\\1, or\\2', supported_python_versions_str)
    return supported_python_versions_str

def _getPythonVersion():
    if False:
        i = 10
        return i + 15
    (big, major, minor) = sys.version_info[0:3]
    return big * 256 + major * 16 + min(15, minor)
python_version = _getPythonVersion()
python_version_full_str = '.'.join((str(s) for s in sys.version_info[0:3]))
python_version_str = '.'.join((str(s) for s in sys.version_info[0:2]))

def getErrorMessageExecWithNestedFunction():
    if False:
        print('Hello World!')
    'Error message of the concrete Python in case an exec occurs in a\n    function that takes a closure variable.\n    '
    assert python_version < 768
    try:
        exec('\ndef f():\n   exec ""\n   def nested():\n      return closure')
    except SyntaxError as e:
        return e.message.replace("'f'", "'%s'")

def getComplexCallSequenceErrorTemplate():
    if False:
        print('Hello World!')
    if not hasattr(getComplexCallSequenceErrorTemplate, 'result'):
        try:
            f = None
            f(*None)
        except TypeError as e:
            result = e.args[0].replace('NoneType object', '%s').replace('NoneType', '%s').replace('None ', '%s ')
            getComplexCallSequenceErrorTemplate.result = result
        else:
            sys.exit('Error, cannot detect expected error message.')
    return getComplexCallSequenceErrorTemplate.result

def getUnboundLocalErrorErrorTemplate():
    if False:
        for i in range(10):
            print('nop')
    if not hasattr(getUnboundLocalErrorErrorTemplate, 'result'):
        try:
            del _f
        except UnboundLocalError as e:
            result = e.args[0].replace('_f', '%s')
            getUnboundLocalErrorErrorTemplate.result = result
        else:
            sys.exit('Error, cannot detect expected error message.')
    return getUnboundLocalErrorErrorTemplate.result
_needs_set_literal_reverse_insertion = None

def needsSetLiteralReverseInsertion():
    if False:
        return 10
    'For Python3, until Python3.5 ca. the order of set literals was reversed.'
    global _needs_set_literal_reverse_insertion
    if _needs_set_literal_reverse_insertion is None:
        try:
            value = eval('{1,1.0}.pop()')
        except SyntaxError:
            _needs_set_literal_reverse_insertion = False
        else:
            _needs_set_literal_reverse_insertion = type(value) is float
    return _needs_set_literal_reverse_insertion

def needsDuplicateArgumentColOffset():
    if False:
        i = 10
        return i + 15
    if python_version < 851:
        return False
    else:
        return True

def getRunningPythonDLLPath():
    if False:
        return 10
    from nuitka.utils.SharedLibraries import getWindowsRunningProcessModuleFilename
    return getWindowsRunningProcessModuleFilename(ctypes.pythonapi._handle)

def getTargetPythonDLLPath():
    if False:
        print('Hello World!')
    dll_path = getRunningPythonDLLPath()
    from nuitka.Options import shallUsePythonDebug
    if dll_path.endswith('_d.dll'):
        if not shallUsePythonDebug():
            dll_path = dll_path[:-6] + '.dll'
        if not os.path.exists(dll_path):
            sys.exit('Error, cannot switch to non-debug Python, not installed.')
    else:
        if shallUsePythonDebug():
            dll_path = dll_path[:-4] + '_d.dll'
        if not os.path.exists(dll_path):
            sys.exit('Error, cannot switch to debug Python, not installed.')
    return dll_path

def isStaticallyLinkedPython():
    if False:
        while True:
            i = 10
    if os.name == 'nt':
        return ctypes.pythonapi is None
    try:
        import sysconfig
    except ImportError:
        return False
    result = sysconfig.get_config_var('Py_ENABLE_SHARED') == 0
    return result

def getPythonABI():
    if False:
        print('Hello World!')
    if hasattr(sys, 'abiflags'):
        abiflags = sys.abiflags
        from nuitka.Options import shallUsePythonDebug
        if shallUsePythonDebug() or hasattr(sys, 'getobjects'):
            if not abiflags.startswith('d'):
                abiflags = 'd' + abiflags
    else:
        abiflags = ''
    return abiflags
_the_sys_prefix = None

def getSystemPrefixPath():
    if False:
        print('Hello World!')
    'Return real sys.prefix as an absolute path breaking out of virtualenv.\n\n    Note:\n\n        For Nuitka, it often is OK to break out of the virtualenv, and use the\n        original install. Mind you, this is not about executing anything, this is\n        about building, and finding the headers to compile against that Python, we\n        do not care about any site packages, and so on.\n\n    Returns:\n        str - path to system prefix\n    '
    global _the_sys_prefix
    if _the_sys_prefix is None:
        sys_prefix = getattr(sys, 'real_prefix', getattr(sys, 'base_prefix', sys.prefix))
        sys_prefix = os.path.abspath(sys_prefix)
        for candidate in ('Lib/orig-prefix.txt', 'lib/python%s/orig-prefix.txt' % python_version_str):
            candidate = os.path.join(sys_prefix, candidate)
            if os.path.exists(candidate):
                with open(candidate) as f:
                    sys_prefix = f.read()
                assert sys_prefix == sys_prefix.strip()
        if os.name != 'nt' and os.path.islink(os.path.join(sys_prefix, '.Python')):
            sys_prefix = os.path.normpath(os.path.join(os.readlink(os.path.join(sys_prefix, '.Python')), '..'))
        if os.name != 'nt' and python_version >= 816 and os.path.exists(os.path.join(sys_prefix, 'bin/activate')):
            python_binary = os.path.join(sys_prefix, 'bin', 'python')
            python_binary = os.path.realpath(python_binary)
            sys_prefix = os.path.normpath(os.path.join(python_binary, '..', '..'))
        if os.name == 'nt':
            from nuitka.utils.FileOperations import getDirectoryRealPath
            sys_prefix = getDirectoryRealPath(sys_prefix)
        _the_sys_prefix = sys_prefix
    return _the_sys_prefix

def getFutureModuleKeys():
    if False:
        for i in range(10):
            print('nop')
    result = ['unicode_literals', 'absolute_import', 'division', 'print_function', 'generator_stop', 'nested_scopes', 'generators', 'with_statement']
    if hasattr(__future__, 'barry_as_FLUFL'):
        result.append('barry_as_FLUFL')
    if hasattr(__future__, 'annotations'):
        result.append('annotations')
    return result

def getImportlibSubPackages():
    if False:
        return 10
    result = []
    if python_version >= 624:
        import importlib
        import pkgutil
        for module_info in pkgutil.walk_packages(importlib.__path__):
            result.append(module_info[1])
    return result

def isDebugPython():
    if False:
        print('Hello World!')
    'Is this a debug build of Python.'
    return hasattr(sys, 'gettotalrefcount')

def _getFloatDigitBoundaryValue():
    if False:
        print('Hello World!')
    if python_version < 624:
        bits_per_digit = 15
    elif python_version < 768:
        bits_per_digit = sys.long_info.bits_per_digit
    else:
        bits_per_digit = sys.int_info.bits_per_digit
    return 2 ** bits_per_digit - 1
_float_digit_boundary = _getFloatDigitBoundaryValue()

def isPythonValidDigitValue(value):
    if False:
        for i in range(10):
            print('nop')
    'Does the given value fit into a float digit.\n\n    Note: Digits in long objects do not use 2-complement, but a boolean sign.\n    '
    return -_float_digit_boundary <= value <= _float_digit_boundary
sizeof_clong = ctypes.sizeof(ctypes.c_long)
_max_signed_long = 2 ** (sizeof_clong * 7) - 1
_min_signed_long = -2 ** (sizeof_clong * 7)
sizeof_clonglong = ctypes.sizeof(ctypes.c_longlong)
_max_signed_longlong = 2 ** (sizeof_clonglong * 8 - 1) - 1
_min_signed_longlong = -2 ** (sizeof_clonglong * 8 - 1)

def isPythonValidCLongValue(value):
    if False:
        while True:
            i = 10
    return _min_signed_long <= value <= _max_signed_long

def isPythonValidCLongLongValue(value):
    if False:
        while True:
            i = 10
    return _min_signed_longlong <= value <= _max_signed_longlong

def getInstalledPythonRegistryPaths(version):
    if False:
        i = 10
        return i + 15
    'Yield all Pythons as found in the Windows registry.'
    from nuitka.__past__ import WindowsError
    if str is bytes:
        import _winreg as winreg
    else:
        import winreg
    for hkey_branch in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
        for arch_key in (0, winreg.KEY_WOW64_32KEY, winreg.KEY_WOW64_64KEY):
            for suffix in ('', '-32', '-arm64'):
                try:
                    key = winreg.OpenKey(hkey_branch, 'SOFTWARE\\Python\\PythonCore\\%s%s\\InstallPath' % (version, suffix), 0, winreg.KEY_READ | arch_key)
                    install_dir = os.path.normpath(winreg.QueryValue(key, ''))
                except WindowsError:
                    pass
                else:
                    candidate = os.path.normpath(os.path.join(install_dir, 'python.exe'))
                    if os.path.exists(candidate):
                        yield candidate

def getTkInterVersion():
    if False:
        i = 10
        return i + 15
    'Get the tk-inter version or None if not installed.'
    try:
        if str is bytes:
            return str(__import__('TkInter').TkVersion)
        else:
            return str(__import__('tkinter').TkVersion)
    except ImportError:
        return None

def getModuleLinkerLibs():
    if False:
        return 10
    'Get static link libraries needed.'
    import sysconfig
    result = sysconfig.get_config_var('MODLIBS') or ''
    result = [entry[2:] for entry in result.split() if entry.startswith('-l:')]
    return result