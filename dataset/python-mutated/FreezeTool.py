""" This module contains code to freeze a number of Python modules
into a single (mostly) standalone DLL or EXE. """
import modulefinder
import sys
import os
import marshal
import platform
import struct
import io
import sysconfig
import zipfile
import importlib
import warnings
from importlib import machinery
from . import pefile
try:
    import p3extend_frozen
except ImportError:
    p3extend_frozen = None
from panda3d.core import Filename, Multifile, PandaSystem, StringStream
_PY_SOURCE = 1
_PY_COMPILED = 2
_C_EXTENSION = 3
_PKG_DIRECTORY = 5
_C_BUILTIN = 6
_PY_FROZEN = 7
_PKG_NAMESPACE_DIRECTORY = object()
python = os.path.splitext(os.path.split(sys.executable)[1])[0]
isDebugBuild = python.lower().endswith('_d')
startupModules = ['encodings', 'encodings.*', 'io', 'marshal', 'importlib.machinery', 'importlib.util']
builtinInitFuncs = {'builtins': None, 'sys': None, 'exceptions': None, '_warnings': '_PyWarnings_Init', 'marshal': 'PyMarshal_Init'}
if sys.version_info < (3, 7):
    builtinInitFuncs['_imp'] = 'PyInit_imp'
try:
    from pytest import freeze_includes as pytest_imports
except ImportError:

    def pytest_imports():
        if False:
            return 10
        return []
defaultHiddenImports = {'pytest': pytest_imports(), 'pkg_resources': ['pkg_resources.*.*'], 'xml.etree.cElementTree': ['xml.etree.ElementTree'], 'datetime': ['_strptime'], 'keyring.backends': ['keyring.backends.*'], 'matplotlib.font_manager': ['encodings.mac_roman'], 'matplotlib.backends._backend_tk': ['tkinter'], 'direct.particles': ['direct.particles.ParticleManagerGlobal'], 'numpy.core._multiarray_umath': ['numpy.core._internal', 'numpy.core._dtype_ctypes', 'numpy.core._methods'], 'pandas.compat': ['lzma', 'cmath'], 'pandas._libs.tslibs.conversion': ['pandas._libs.tslibs.base'], 'plyer': ['plyer.platforms'], 'scipy.linalg': ['scipy.linalg.cython_blas', 'scipy.linalg.cython_lapack'], 'scipy.sparse.csgraph': ['scipy.sparse.csgraph._validation'], 'scipy.spatial.qhull': ['scipy._lib.messagestream'], 'scipy.spatial._qhull': ['scipy._lib.messagestream'], 'scipy.spatial.transform.rotation': ['scipy.spatial.transform._rotation_groups'], 'scipy.spatial.transform._rotation': ['scipy.spatial.transform._rotation_groups'], 'scipy.special._ufuncs': ['scipy.special._ufuncs_cxx'], 'scipy.stats._stats': ['scipy.special.cython_special'], 'setuptools.monkey': ['setuptools.msvc'], 'shapely._geometry_helpers': ['shapely._geos']}
ignoreImports = {'direct.showbase.PythonUtil': ['pstats', 'profile'], 'toml.encoder': ['numpy'], 'py._builtin': ['__builtin__'], 'site': ['android_log']}
if sys.version_info >= (3, 8):
    ignoreImports['importlib._bootstrap_external'] = ['importlib.metadata']
    ignoreImports['importlib.metadata'] = ['pep517']
overrideModules = {'linecache': '__all__ = ["getline", "clearcache", "checkcache", "lazycache"]\n\ncache = {}\n\ndef getline(filename, lineno, module_globals=None):\n    return \'\'\n\ndef clearcache():\n    global cache\n    cache = {}\n\ndef getlines(filename, module_globals=None):\n    return []\n\ndef checkcache(filename=None):\n    pass\n\ndef updatecache(filename, module_globals=None):\n    pass\n\ndef lazycache(filename, module_globals):\n    pass\n', '_distutils_hack.override': ''}
reportedMissing = {}

class CompilationEnvironment:
    """ Create an instance of this class to record the commands to
    invoke the compiler on a given platform.  If needed, the caller
    can create a custom instance of this class (or simply set the
    compile strings directly) to customize the build environment. """

    def __init__(self, platform):
        if False:
            i = 10
            return i + 15
        self.platform = platform
        self.compileObj = 'error'
        self.linkExe = 'error'
        self.linkDll = 'error'
        self.Python = None
        self.PythonIPath = sysconfig.get_path('include')
        self.PythonVersion = sysconfig.get_config_var('LDVERSION') or sysconfig.get_python_version()
        self.MSVC = None
        self.PSDK = None
        self.MD = None
        self.suffix64 = ''
        self.dllext = ''
        self.arch = ''
        self.determineStandardSetup()

    def determineStandardSetup(self):
        if False:
            for i in range(10):
                print('nop')
        if self.platform.startswith('win'):
            self.Python = sysconf.PREFIX
            if 'VCINSTALLDIR' in os.environ:
                self.MSVC = os.environ['VCINSTALLDIR']
            elif Filename('/c/Program Files/Microsoft Visual Studio 9.0/VC').exists():
                self.MSVC = Filename('/c/Program Files/Microsoft Visual Studio 9.0/VC').toOsSpecific()
            elif Filename('/c/Program Files (x86)/Microsoft Visual Studio 9.0/VC').exists():
                self.MSVC = Filename('/c/Program Files (x86)/Microsoft Visual Studio 9.0/VC').toOsSpecific()
            elif Filename('/c/Program Files/Microsoft Visual Studio .NET 2003/Vc7').exists():
                self.MSVC = Filename('/c/Program Files/Microsoft Visual Studio .NET 2003/Vc7').toOsSpecific()
            else:
                print('Could not locate Microsoft Visual C++ Compiler! Try running from the Visual Studio Command Prompt.')
                sys.exit(1)
            if 'WindowsSdkDir' in os.environ:
                self.PSDK = os.environ['WindowsSdkDir']
            elif platform.architecture()[0] == '32bit' and Filename('/c/Program Files/Microsoft Platform SDK for Windows Server 2003 R2').exists():
                self.PSDK = Filename('/c/Program Files/Microsoft Platform SDK for Windows Server 2003 R2').toOsSpecific()
            elif os.path.exists(os.path.join(self.MSVC, 'PlatformSDK')):
                self.PSDK = os.path.join(self.MSVC, 'PlatformSDK')
            else:
                print('Could not locate the Microsoft Windows Platform SDK! Try running from the Visual Studio Command Prompt.')
                sys.exit(1)
            self.MD = '/MD'
            if isDebugBuild:
                self.MD = '/MDd'
                self.dllext = '_d'
            if self.platform == 'win_amd64':
                self.suffix64 = '\\amd64'
            if 'MAKEPANDA' in os.environ:
                self.compileObjExe = 'cl /wd4996 /Fo%(basename)s.obj /nologo /c %(MD)s /Zi /O2 /Ob2 /EHsc /Zm300 /W3 /I"%(pythonIPath)s" %(filename)s'
                self.compileObjDll = self.compileObjExe
                self.linkExe = 'link /nologo /MAP:NUL /FIXED:NO /OPT:REF /STACK:4194304 /INCREMENTAL:NO /LIBPATH:"%(python)s\\libs"  /out:%(basename)s.exe %(basename)s.obj'
                self.linkDll = 'link /nologo /DLL /MAP:NUL /FIXED:NO /OPT:REF /INCREMENTAL:NO /LIBPATH:"%(python)s\\libs"  /out:%(basename)s%(dllext)s.pyd %(basename)s.obj'
            else:
                os.environ['PATH'] += ';' + self.MSVC + '\\bin' + self.suffix64 + ';' + self.MSVC + '\\Common7\\IDE;' + self.PSDK + '\\bin'
                self.compileObjExe = 'cl /wd4996 /Fo%(basename)s.obj /nologo /c %(MD)s /Zi /O2 /Ob2 /EHsc /Zm300 /W3 /I"%(pythonIPath)s" /I"%(PSDK)s\\include" /I"%(MSVC)s\\include" %(filename)s'
                self.compileObjDll = self.compileObjExe
                self.linkExe = 'link /nologo /MAP:NUL /FIXED:NO /OPT:REF /STACK:4194304 /INCREMENTAL:NO /LIBPATH:"%(PSDK)s\\lib" /LIBPATH:"%(MSVC)s\\lib%(suffix64)s" /LIBPATH:"%(python)s\\libs"  /out:%(basename)s.exe %(basename)s.obj'
                self.linkDll = 'link /nologo /DLL /MAP:NUL /FIXED:NO /OPT:REF /INCREMENTAL:NO /LIBPATH:"%(PSDK)s\\lib" /LIBPATH:"%(MSVC)s\\lib%(suffix64)s" /LIBPATH:"%(python)s\\libs"  /out:%(basename)s%(dllext)s.pyd %(basename)s.obj'
        elif self.platform.startswith('osx_'):
            proc = self.platform.split('_', 1)[1]
            if proc == 'i386':
                self.arch = '-arch i386'
            elif proc == 'ppc':
                self.arch = '-arch ppc'
            elif proc == 'amd64':
                self.arch = '-arch x86_64'
            elif proc in ('arm64', 'aarch64'):
                self.arch = '-arch arm64'
            self.compileObjExe = 'clang -c %(arch)s -o %(basename)s.o -O2 -I%(pythonIPath)s %(filename)s'
            self.compileObjDll = 'clang -fPIC -c %(arch)s -o %(basename)s.o -O2 -I%(pythonIPath)s %(filename)s'
            self.linkExe = 'clang %(arch)s -o %(basename)s %(basename)s.o'
            if '/Python.framework/' in self.PythonIPath:
                framework_dir = self.PythonIPath.split('/Python.framework/', 1)[0]
                if framework_dir != '/System/Library/Frameworks':
                    self.linkExe += ' -F ' + framework_dir
            self.linkExe += ' -framework Python'
            self.linkDll = 'clang %(arch)s -undefined dynamic_lookup -bundle -o %(basename)s.so %(basename)s.o'
        else:
            lib_dir = sysconf.get_python_lib(plat_specific=1, standard_lib=1)
            self.compileObjExe = '%(CC)s %(CFLAGS)s -c -o %(basename)s.o -pthread -O2 %(filename)s -I%(pythonIPath)s'
            self.compileObjDll = '%(CC)s %(CFLAGS)s %(CCSHARED)s -c -o %(basename)s.o -O2 %(filename)s -I%(pythonIPath)s'
            self.linkExe = '%(CC)s -o %(basename)s %(basename)s.o -L/usr/local/lib -lpython%(pythonVersion)s'
            self.linkDll = '%(LDSHARED)s -o %(basename)s.so %(basename)s.o -L/usr/local/lib -lpython%(pythonVersion)s'
            if os.path.isdir('/usr/PCBSD/local/lib'):
                self.linkExe += ' -L/usr/PCBSD/local/lib'
                self.linkDll += ' -L/usr/PCBSD/local/lib'

    def compileExe(self, filename, basename, extraLink=[]):
        if False:
            print('Hello World!')
        compile = self.compileObjExe % dict({'python': self.Python, 'MSVC': self.MSVC, 'PSDK': self.PSDK, 'suffix64': self.suffix64, 'MD': self.MD, 'pythonIPath': self.PythonIPath, 'pythonVersion': self.PythonVersion, 'arch': self.arch, 'filename': filename, 'basename': basename}, **sysconf.get_config_vars())
        sys.stderr.write(compile + '\n')
        if os.system(compile) != 0:
            raise Exception('failed to compile %s.' % basename)
        link = self.linkExe % dict({'python': self.Python, 'MSVC': self.MSVC, 'PSDK': self.PSDK, 'suffix64': self.suffix64, 'pythonIPath': self.PythonIPath, 'pythonVersion': self.PythonVersion, 'arch': self.arch, 'filename': filename, 'basename': basename}, **sysconf.get_config_vars())
        link += ' ' + ' '.join(extraLink)
        sys.stderr.write(link + '\n')
        if os.system(link) != 0:
            raise Exception('failed to link %s.' % basename)

    def compileDll(self, filename, basename, extraLink=[]):
        if False:
            for i in range(10):
                print('nop')
        compile = self.compileObjDll % dict({'python': self.Python, 'MSVC': self.MSVC, 'PSDK': self.PSDK, 'suffix64': self.suffix64, 'MD': self.MD, 'pythonIPath': self.PythonIPath, 'pythonVersion': self.PythonVersion, 'arch': self.arch, 'filename': filename, 'basename': basename}, **sysconf.get_config_vars())
        sys.stderr.write(compile + '\n')
        if os.system(compile) != 0:
            raise Exception('failed to compile %s.' % basename)
        link = self.linkDll % dict({'python': self.Python, 'MSVC': self.MSVC, 'PSDK': self.PSDK, 'suffix64': self.suffix64, 'pythonIPath': self.PythonIPath, 'pythonVersion': self.PythonVersion, 'arch': self.arch, 'filename': filename, 'basename': basename, 'dllext': self.dllext}, **sysconf.get_config_vars())
        link += ' ' + ' '.join(extraLink)
        sys.stderr.write(link + '\n')
        if os.system(link) != 0:
            raise Exception('failed to link %s.' % basename)
frozenMainCode = '\n/* Python interpreter main program for frozen scripts */\n\n#include <Python.h>\n\n#if PY_MAJOR_VERSION >= 3\n#include <locale.h>\n\n#if PY_MINOR_VERSION < 5\n#define Py_DecodeLocale _Py_char2wchar\n#endif\n#endif\n\n#ifdef MS_WINDOWS\nextern void PyWinFreeze_ExeInit(void);\nextern void PyWinFreeze_ExeTerm(void);\n\nextern PyAPI_FUNC(int) PyImport_ExtendInittab(struct _inittab *newtab);\n#endif\n\n/* Main program */\n\nEXTRA_INIT_FUNC_DECLS\n\nint\nPy_FrozenMain(int argc, char **argv)\n{\n    char *p;\n    int n, sts = 1;\n    int inspect = 0;\n    int unbuffered = 0;\n\n#if PY_MAJOR_VERSION >= 3\n    int i;\n    char *oldloc;\n    wchar_t **argv_copy = NULL;\n    /* We need a second copies, as Python might modify the first one. */\n    wchar_t **argv_copy2 = NULL;\n\n    if (argc > 0) {\n        argv_copy = (wchar_t **)alloca(sizeof(wchar_t *) * argc);\n        argv_copy2 = (wchar_t **)alloca(sizeof(wchar_t *) * argc);\n    }\n#endif\n\n    Py_FrozenFlag = 1; /* Suppress errors from getpath.c */\n    Py_NoSiteFlag = 1;\n    Py_NoUserSiteDirectory = 1;\n\n    if ((p = Py_GETENV("PYTHONINSPECT")) && *p != \'\\0\')\n        inspect = 1;\n    if ((p = Py_GETENV("PYTHONUNBUFFERED")) && *p != \'\\0\')\n        unbuffered = 1;\n\n    if (unbuffered) {\n        setbuf(stdin, (char *)NULL);\n        setbuf(stdout, (char *)NULL);\n        setbuf(stderr, (char *)NULL);\n    }\n\n#if PY_MAJOR_VERSION >= 3\n    oldloc = setlocale(LC_ALL, NULL);\n    setlocale(LC_ALL, "");\n    for (i = 0; i < argc; i++) {\n        argv_copy[i] = Py_DecodeLocale(argv[i], NULL);\n        argv_copy2[i] = argv_copy[i];\n        if (!argv_copy[i]) {\n            fprintf(stderr, "Unable to decode the command line argument #%i\\n",\n                            i + 1);\n            argc = i;\n            goto error;\n        }\n    }\n    setlocale(LC_ALL, oldloc);\n#endif\n\n#ifdef MS_WINDOWS\n    PyImport_ExtendInittab(extensions);\n#endif /* MS_WINDOWS */\n\n    if (argc >= 1) {\n#if PY_MAJOR_VERSION >= 3\n        Py_SetProgramName(argv_copy[0]);\n#else\n        Py_SetProgramName(argv[0]);\n#endif\n    }\n\n    Py_Initialize();\n#ifdef MS_WINDOWS\n    PyWinFreeze_ExeInit();\n#endif\n\n    if (Py_VerboseFlag)\n        fprintf(stderr, "Python %s\\n%s\\n",\n            Py_GetVersion(), Py_GetCopyright());\n\n#if PY_MAJOR_VERSION >= 3\n    PySys_SetArgv(argc, argv_copy);\n#else\n    PySys_SetArgv(argc, argv);\n#endif\n\nEXTRA_INIT_FUNC_CALLS\n\n    n = PyImport_ImportFrozenModule("__main__");\n    if (n == 0)\n        Py_FatalError("__main__ not frozen");\n    if (n < 0) {\n        PyErr_Print();\n        sts = 1;\n    }\n    else\n        sts = 0;\n\n    if (inspect && isatty((int)fileno(stdin)))\n        sts = PyRun_AnyFile(stdin, "<stdin>") != 0;\n\n#ifdef MS_WINDOWS\n    PyWinFreeze_ExeTerm();\n#endif\n    Py_Finalize();\n\n#if PY_MAJOR_VERSION >= 3\nerror:\n    if (argv_copy2) {\n        for (i = 0; i < argc; i++) {\n#if PY_MINOR_VERSION >= 4\n            PyMem_RawFree(argv_copy2[i]);\n#else\n            PyMem_Free(argv_copy2[i]);\n#endif\n        }\n    }\n#endif\n    return sts;\n}\n'
frozenDllMainCode = '\n#include <windows.h>\n\nstatic char *possibleModules[] = {\n    "pywintypes",\n    "pythoncom",\n    "win32ui",\n    NULL,\n};\n\nBOOL CallModuleDllMain(char *modName, DWORD dwReason);\n\n\n/*\n  Called by a frozen .EXE only, so that built-in extension\n  modules are initialized correctly\n*/\nvoid PyWinFreeze_ExeInit(void)\n{\n    char **modName;\n    for (modName = possibleModules;*modName;*modName++) {\n/*      printf("Initialising \'%s\'\\n", *modName); */\n        CallModuleDllMain(*modName, DLL_PROCESS_ATTACH);\n    }\n}\n\n/*\n  Called by a frozen .EXE only, so that built-in extension\n  modules are cleaned up\n*/\nvoid PyWinFreeze_ExeTerm(void)\n{\n    // Must go backwards\n    char **modName;\n    for (modName = possibleModules+(sizeof(possibleModules) / sizeof(char *))-2;\n         modName >= possibleModules;\n         *modName--) {\n/*      printf("Terminating \'%s\'\\n", *modName);*/\n        CallModuleDllMain(*modName, DLL_PROCESS_DETACH);\n    }\n}\n\nBOOL WINAPI DllMain(HINSTANCE hInstance, DWORD dwReason, LPVOID lpReserved)\n{\n    BOOL ret = TRUE;\n    switch (dwReason) {\n        case DLL_PROCESS_ATTACH:\n        {\n            char **modName;\n            for (modName = possibleModules;*modName;*modName++) {\n                BOOL ok = CallModuleDllMain(*modName, dwReason);\n                if (!ok)\n                    ret = FALSE;\n            }\n            break;\n        }\n        case DLL_PROCESS_DETACH:\n        {\n            // Must go backwards\n            char **modName;\n            for (modName = possibleModules+(sizeof(possibleModules) / sizeof(char *))-2;\n                 modName >= possibleModules;\n                 *modName--)\n                CallModuleDllMain(*modName, DLL_PROCESS_DETACH);\n            break;\n        }\n    }\n    return ret;\n}\n\nBOOL CallModuleDllMain(char *modName, DWORD dwReason)\n{\n    BOOL (WINAPI * pfndllmain)(HINSTANCE, DWORD, LPVOID);\n\n    char funcName[255];\n    HMODULE hmod = GetModuleHandle(NULL);\n    strcpy(funcName, "_DllMain");\n    strcat(funcName, modName);\n    strcat(funcName, "@12"); // stdcall convention.\n    pfndllmain = (BOOL (WINAPI *)(HINSTANCE, DWORD, LPVOID))GetProcAddress(hmod, funcName);\n    if (pfndllmain==NULL) {\n        /* No function by that name exported - then that module does\n           not appear in our frozen program - return OK\n                */\n        return TRUE;\n    }\n    return (*pfndllmain)(hmod, dwReason, NULL);\n}\n'
mainInitCode = '\n%(frozenMainCode)s\n\nint\nmain(int argc, char *argv[]) {\n  PyImport_FrozenModules = _PyImport_FrozenModules;\n  return Py_FrozenMain(argc, argv);\n}\n'
dllInitCode = '\n/*\n * Call this function to extend the frozen modules array with a new\n * array of frozen modules, provided in a C-style array, at runtime.\n * Returns the total number of frozen modules.\n */\nstatic int\nextend_frozen_modules(const struct _frozen *new_modules, int new_count) {\n  int orig_count;\n  struct _frozen *realloc_FrozenModules;\n\n  /* First, count the number of frozen modules we had originally. */\n  orig_count = 0;\n  while (PyImport_FrozenModules[orig_count].name != NULL) {\n    ++orig_count;\n  }\n\n  if (new_count == 0) {\n    /* Trivial no-op. */\n    return orig_count;\n  }\n\n  /* Reallocate the PyImport_FrozenModules array bigger to make room\n     for the additional frozen modules.  We just leak the original\n     array; it\'s too risky to try to free it. */\n  realloc_FrozenModules = (struct _frozen *)malloc((orig_count + new_count + 1) * sizeof(struct _frozen));\n\n  /* The new frozen modules go at the front of the list. */\n  memcpy(realloc_FrozenModules, new_modules, new_count * sizeof(struct _frozen));\n\n  /* Then the original set of frozen modules. */\n  memcpy(realloc_FrozenModules + new_count, PyImport_FrozenModules, orig_count * sizeof(struct _frozen));\n\n  /* Finally, a single 0-valued entry marks the end of the array. */\n  memset(realloc_FrozenModules + orig_count + new_count, 0, sizeof(struct _frozen));\n\n  /* Assign the new pointer. */\n  PyImport_FrozenModules = realloc_FrozenModules;\n\n  return orig_count + new_count;\n}\n\n#if PY_MAJOR_VERSION >= 3\nstatic PyModuleDef mdef = {\n  PyModuleDef_HEAD_INIT,\n  "%(moduleName)s",\n  "",\n  -1,\n  NULL, NULL, NULL, NULL, NULL\n};\n\n%(dllexport)sPyObject *PyInit_%(moduleName)s(void) {\n  extend_frozen_modules(_PyImport_FrozenModules, sizeof(_PyImport_FrozenModules) / sizeof(struct _frozen));\n  return PyModule_Create(&mdef);\n}\n#else\nstatic PyMethodDef nullMethods[] = {\n  {NULL, NULL}\n};\n\n%(dllexport)svoid init%(moduleName)s(void) {\n  extend_frozen_modules(_PyImport_FrozenModules, sizeof(_PyImport_FrozenModules) / sizeof(struct _frozen));\n  Py_InitModule("%(moduleName)s", nullMethods);\n}\n#endif\n'
programFile = '\n#include <Python.h>\n#ifdef _WIN32\n#include <malloc.h>\n#endif\n\n%(moduleDefs)s\n\nstruct _frozen _PyImport_FrozenModules[] = {\n%(moduleList)s\n  {NULL, NULL, 0}\n};\n'
okMissing = ['__main__', '_dummy_threading', 'Carbon', 'Carbon.Files', 'Carbon.Folder', 'Carbon.Folders', 'HouseGlobals', 'Carbon.File', 'MacOS', '_emx_link', 'ce', 'mac', 'org.python.core', 'os.path', 'os2', 'posix', 'pwd', 'readline', 'riscos', 'riscosenviron', 'riscospath', 'dbm', 'fcntl', 'win32api', 'win32pipe', 'usercustomize', '_winreg', 'winreg', 'ctypes', 'ctypes.wintypes', 'nt', 'msvcrt', 'EasyDialogs', 'SOCKS', 'ic', 'rourl2path', 'termios', 'vms_lib', 'OverrideFrom23._Res', 'email', 'email.Utils', 'email.Generator', 'email.Iterators', '_subprocess', 'gestalt', 'java.lang', 'direct.extensions_native.extensions_darwin', '_manylinux', 'collections.Iterable', 'collections.Mapping', 'collections.MutableMapping', 'collections.Sequence', 'numpy_distutils', '_winapi']
mach_header_64_layout = '<IIIIIIII'
lc_header_layout = '<II'
section64_header_layout = '<16s16sQQIIIIIIII'
LC_SEGMENT_64 = 25
LC_DYLD_INFO_ONLY = 2147483682
LC_SYMTAB = 2
LC_DYSYMTAB = 11
LC_FUNCTION_STARTS = 38
LC_DATA_IN_CODE = 41
lc_layouts = {LC_SEGMENT_64: '<II16sQQQQIIII', LC_DYLD_INFO_ONLY: '<IIIIIIIIIIII', LC_SYMTAB: '<IIIIII', LC_DYSYMTAB: '<IIIIIIIIIIIIIIIIIIII', LC_FUNCTION_STARTS: '<IIII', LC_DATA_IN_CODE: '<IIII'}
lc_indices_to_slide = {b'__PANDA': [4, 6], b'__LINKEDIT': [3, 5], LC_DYLD_INFO_ONLY: [2, 4, 8, 10], LC_SYMTAB: [2, 4], LC_DYSYMTAB: [14], LC_FUNCTION_STARTS: [2], LC_DATA_IN_CODE: [2]}

class Freezer:

    class ModuleDef:

        def __init__(self, moduleName, filename=None, implicit=False, guess=False, exclude=False, forbid=False, allowChildren=False, fromSource=None, text=None):
            if False:
                return 10
            self.moduleName = moduleName
            self.filename = filename
            if filename is not None and (not isinstance(filename, Filename)):
                self.filename = Filename(filename)
            self.implicit = implicit
            self.guess = guess
            self.exclude = exclude
            self.forbid = forbid
            self.allowChildren = allowChildren
            self.fromSource = fromSource
            self.text = text
            if not self.exclude:
                self.allowChildren = True
            if self.forbid:
                self.exclude = True
                self.allowChildren = False

        def __repr__(self):
            if False:
                while True:
                    i = 10
            args = [repr(self.moduleName), repr(self.filename)]
            if self.implicit:
                args.append('implicit = True')
            if self.guess:
                args.append('guess = True')
            if self.exclude:
                args.append('exclude = True')
            if self.forbid:
                args.append('forbid = True')
            if self.allowChildren:
                args.append('allowChildren = True')
            return 'ModuleDef(%s)' % ', '.join(args)

    def __init__(self, previous=None, debugLevel=0, platform=None, path=None, hiddenImports=None, optimize=None):
        if False:
            for i in range(10):
                print('nop')
        self.platform = platform or PandaSystem.getPlatform()
        self.cenv = None
        self.sourceExtension = '.c'
        self.objectExtension = '.o'
        if self.platform.startswith('win'):
            self.objectExtension = '.obj'
        self.keepTemporaryFiles = False
        self.frozenMainCode = frozenMainCode
        self.frozenDllMainCode = frozenDllMainCode
        self.mainInitCode = mainInitCode
        self.storePythonSource = False
        self.extras = []
        self.extraInitFuncs = []
        self.linkExtensionModules = False
        self.previousModules = {}
        self.modules = {}
        if previous:
            self.previousModules = dict(previous.modules)
            self.modules = dict(previous.modules)
        self.modules['doctest'] = self.ModuleDef('doctest', exclude=True)
        for (moduleName, module) in list(sys.modules.items()):
            if module and getattr(module, '__path__', None) is not None:
                modPath = list(getattr(module, '__path__'))
                if modPath:
                    modulefinder.AddPackagePath(moduleName, modPath[0])
        self.hiddenImports = defaultHiddenImports.copy()
        if hiddenImports is not None:
            self.hiddenImports.update(hiddenImports)
        plyer_platform = None
        if self.platform.startswith('android'):
            plyer_platform = 'android'
        elif self.platform.startswith('linux'):
            plyer_platform = 'linux'
        elif self.platform.startswith('mac'):
            plyer_platform = 'macosx'
        elif self.platform.startswith('win'):
            plyer_platform = 'win'
        if plyer_platform:
            self.hiddenImports['plyer'].append(f'plyer.platforms.{plyer_platform}.*')
        if self.platform == PandaSystem.getPlatform():
            suffixes = [(s, 'rb', _C_EXTENSION) for s in machinery.EXTENSION_SUFFIXES] + [(s, 'rb', _PY_SOURCE) for s in machinery.SOURCE_SUFFIXES] + [(s, 'rb', _PY_COMPILED) for s in machinery.BYTECODE_SUFFIXES]
        else:
            suffixes = [('.py', 'rb', 1), ('.pyc', 'rb', 2)]
            abi_version = '{0}{1}'.format(*sys.version_info)
            abi_flags = ''
            if sys.version_info < (3, 8):
                abi_flags += 'm'
            if 'linux' in self.platform:
                suffixes += [('.cpython-{0}{1}-x86_64-linux-gnu.so'.format(abi_version, abi_flags), 'rb', 3), ('.cpython-{0}{1}-i686-linux-gnu.so'.format(abi_version, abi_flags), 'rb', 3), ('.abi{0}.so'.format(sys.version_info[0]), 'rb', 3), ('.so', 'rb', 3)]
            elif 'win' in self.platform:
                suffixes += [('.cp{0}-win_amd64.pyd'.format(abi_version), 'rb', 3), ('.cp{0}-win32.pyd'.format(abi_version), 'rb', 3), ('.pyd', 'rb', 3)]
            elif 'mac' in self.platform:
                suffixes += [('.cpython-{0}{1}-darwin.so'.format(abi_version, abi_flags), 'rb', 3), ('.abi{0}.so'.format(sys.version_info[0]), 'rb', 3), ('.so', 'rb', 3)]
            else:
                suffixes += [('.cpython-{0}{1}.so'.format(abi_version, abi_flags), 'rb', 3), ('.abi{0}.so'.format(sys.version_info[0]), 'rb', 3), ('.so', 'rb', 3)]
        if optimize is None or optimize < 0:
            self.optimize = sys.flags.optimize
        else:
            self.optimize = optimize
        self.mf = PandaModuleFinder(excludes=['doctest'], suffixes=suffixes, path=path, optimize=self.optimize)

    def excludeFrom(self, freezer):
        if False:
            print('Hello World!')
        " Excludes all modules that have already been processed by\n        the indicated FreezeTool.  This is equivalent to passing the\n        indicated FreezeTool object as previous to this object's\n        constructor, but it may be called at any point during\n        processing. "
        for (key, value) in list(freezer.modules.items()):
            self.previousModules[key] = value
            self.modules[key] = value

    def excludeModule(self, moduleName, forbid=False, allowChildren=False, fromSource=None):
        if False:
            print('Hello World!')
        ' Adds a module to the list of modules not to be exported by\n        this tool.  If forbid is true, the module is furthermore\n        forbidden to be imported, even if it exists on disk.  If\n        allowChildren is true, the children of the indicated module\n        may still be included.'
        self.modules[moduleName] = self.ModuleDef(moduleName, exclude=True, forbid=forbid, allowChildren=allowChildren, fromSource=fromSource)

    def handleCustomPath(self, moduleName):
        if False:
            i = 10
            return i + 15
        ' Indicates a module that may perform runtime manipulation\n        of its __path__ variable, and which must therefore be actually\n        imported at runtime in order to determine the true value of\n        __path__. '
        str = 'import %s' % moduleName
        exec(str)
        module = sys.modules[moduleName]
        for path in module.__path__:
            modulefinder.AddPackagePath(moduleName, path)

    def getModulePath(self, moduleName):
        if False:
            while True:
                i = 10
        ' Looks for the indicated directory module and returns the\n        __path__ member: the list of directories in which its python\n        files can be found.  If the module is a .py file and not a\n        directory, returns None. '
        path = None
        baseName = moduleName
        if '.' in baseName:
            (parentName, baseName) = moduleName.rsplit('.', 1)
            path = self.getModulePath(parentName)
            if path is None:
                return None
        try:
            (file, pathname, description) = self.mf.find_module(baseName, path)
        except ImportError:
            return None
        if not self.mf._dir_exists(pathname):
            return None
        return [pathname]

    def getModuleStar(self, moduleName):
        if False:
            while True:
                i = 10
        ' Looks for the indicated directory module and returns the\n        __all__ member: the list of symbols within the module. '
        path = None
        baseName = moduleName
        if '.' in baseName:
            (parentName, baseName) = moduleName.rsplit('.', 1)
            path = self.getModulePath(parentName)
            if path is None:
                return None
        try:
            (file, pathname, description) = self.mf.find_module(baseName, path)
        except ImportError:
            return None
        if not self.mf._dir_exists(pathname):
            return None
        modules = []
        for basename in sorted(self.mf._listdir(pathname)):
            if basename.endswith('.py') and basename != '__init__.py':
                modules.append(basename[:-3])
        return modules

    def _gatherSubmodules(self, moduleName, implicit=False, newName=None, filename=None, guess=False, fromSource=None, text=None):
        if False:
            i = 10
            return i + 15
        if not newName:
            newName = moduleName
        assert moduleName.endswith('.*')
        assert newName.endswith('.*')
        mdefs = {}
        parentName = moduleName[:-2]
        newParentName = newName[:-2]
        parentNames = [(parentName, newParentName)]
        if parentName.endswith('.*'):
            assert newParentName.endswith('.*')
            topName = parentName[:-2]
            newTopName = newParentName[:-2]
            parentNames = []
            modulePath = self.getModulePath(topName)
            if modulePath:
                for dirname in modulePath:
                    for basename in sorted(self.mf._listdir(dirname)):
                        if self.mf._file_exists(os.path.join(dirname, basename, '__init__.py')):
                            parentName = '%s.%s' % (topName, basename)
                            newParentName = '%s.%s' % (newTopName, basename)
                            if self.getModulePath(parentName):
                                parentNames.append((parentName, newParentName))
        for (parentName, newParentName) in parentNames:
            modules = self.getModuleStar(parentName)
            if modules is None:
                mdefs[newParentName] = self.ModuleDef(parentName, implicit=implicit, guess=guess, fromSource=fromSource, text=text)
            else:
                for basename in modules:
                    moduleName = '%s.%s' % (parentName, basename)
                    newName = '%s.%s' % (newParentName, basename)
                    mdefs[newName] = self.ModuleDef(moduleName, implicit=implicit, guess=True, fromSource=fromSource)
        return mdefs

    def addModule(self, moduleName, implicit=False, newName=None, filename=None, guess=False, fromSource=None, text=None):
        if False:
            return 10
        ' Adds a module to the list of modules to be exported by\n        this tool.  If implicit is true, it is OK if the module does\n        not actually exist.\n\n        newName is the name to call the module when it appears in the\n        output.  The default is the same name it had in the original.\n        Use caution when renaming a module; if another module imports\n        this module by its original name, you will also need to\n        explicitly add the module under its original name, duplicating\n        the module twice in the output.\n\n        The module name may end in ".*", which means to add all of the\n        .py files (other than __init__.py) in a particular directory.\n        It may also end in ".*.*", which means to cycle through all\n        directories within a particular directory.\n        '
        if not newName:
            newName = moduleName
        if moduleName.endswith('.*'):
            self.modules.update(self._gatherSubmodules(moduleName, implicit, newName, filename, guess, fromSource, text))
        else:
            self.modules[newName] = self.ModuleDef(moduleName, filename=filename, implicit=implicit, guess=guess, fromSource=fromSource, text=text)

    def done(self, addStartupModules=False):
        if False:
            for i in range(10):
                print('nop')
        ' Call this method after you have added all modules with\n        addModule().  You may then call generateCode() or\n        writeMultifile() to dump the resulting output.  After a call\n        to done(), you may not add any more modules until you call\n        reset(). '
        if addStartupModules:
            self.modules['_frozen_importlib'] = self.ModuleDef('importlib._bootstrap', implicit=True)
            self.modules['_frozen_importlib_external'] = self.ModuleDef('importlib._bootstrap_external', implicit=True)
            for moduleName in startupModules:
                if moduleName not in self.modules:
                    self.addModule(moduleName, implicit=True)
        excludeDict = {}
        implicitParentDict = {}
        includes = []
        autoIncludes = []
        origToNewName = {}
        for (newName, mdef) in sorted(self.modules.items()):
            moduleName = mdef.moduleName
            origToNewName[moduleName] = newName
            if mdef.implicit and '.' in newName:
                (parentName, baseName) = newName.rsplit('.', 1)
                if parentName in excludeDict:
                    mdef = excludeDict[parentName]
            if mdef.exclude:
                if not mdef.allowChildren:
                    excludeDict[moduleName] = mdef
            elif mdef.implicit or mdef.guess:
                autoIncludes.append(mdef)
            else:
                includes.append(mdef)
        for exclude in excludeDict:
            self.mf.excludes.append(exclude)
        includes.sort(key=self.__sortModuleKey)
        for mdef in includes:
            try:
                self.__loadModule(mdef)
            except ImportError as ex:
                message = 'Unknown module: %s' % mdef.moduleName
                if str(ex) != 'No module named ' + str(mdef.moduleName):
                    message += ' (%s)' % ex
                print(message)
        for mdef in autoIncludes:
            try:
                self.__loadModule(mdef)
                mdef.guess = False
            except Exception:
                pass
        for origName in list(self.mf.modules.keys()):
            hidden = self.hiddenImports.get(origName, [])
            for modname in hidden:
                if modname.endswith('.*'):
                    mdefs = self._gatherSubmodules(modname, implicit=True)
                    for mdef in mdefs.values():
                        try:
                            self.__loadModule(mdef)
                        except ImportError:
                            pass
                else:
                    try:
                        self.__loadModule(self.ModuleDef(modname, implicit=True))
                    except ImportError:
                        pass
        missing = []
        if 'sysconfig' in self.mf.modules and ('linux' in self.platform or 'mac' in self.platform or 'emscripten' in self.platform):
            modname = '_sysconfigdata'
            if sys.version_info >= (3, 6):
                modname += '_'
                if sys.version_info < (3, 8):
                    modname += 'm'
                if 'linux' in self.platform:
                    arch = self.platform.split('_', 1)[1]
                    modname += '_linux_' + arch + '-linux-gnu'
                elif 'mac' in self.platform:
                    modname += '_darwin_darwin'
                elif 'emscripten' in self.platform:
                    if '_' in self.platform:
                        arch = self.platform.split('_', 1)[1]
                    else:
                        arch = 'wasm32'
                    modname += '_emscripten_' + arch + '-emscripten'
            try:
                self.__loadModule(self.ModuleDef(modname, implicit=True))
            except Exception:
                missing.append(modname)
        for origName in list(self.mf.modules.keys()):
            if origName not in origToNewName:
                self.modules[origName] = self.ModuleDef(origName, implicit=True)
        for origName in self.mf.any_missing_maybe()[0]:
            if origName in startupModules:
                continue
            if origName in self.previousModules:
                continue
            if origName in self.modules:
                continue
            self.modules[origName] = self.ModuleDef(origName, exclude=True, implicit=True)
            if origName in okMissing:
                continue
            prefix = origName.split('.')[0]
            if origName not in reportedMissing:
                missing.append(origName)
                reportedMissing[origName] = True
        if missing:
            missing.sort()
            print('There are some missing modules: %r' % missing)

    def __sortModuleKey(self, mdef):
        if False:
            i = 10
            return i + 15
        " A sort key function to sort a list of mdef's into order,\n        primarily to ensure that packages proceed their modules. "
        if mdef.moduleName:
            return ('a', mdef.moduleName.split('.'))
        else:
            return ('b', mdef.filename)

    def __loadModule(self, mdef):
        if False:
            print('Hello World!')
        ' Adds the indicated module to the modulefinder. '
        if mdef.filename:
            tempPath = None
            if '.' not in mdef.moduleName:
                tempPath = Filename(mdef.filename.getDirname()).toOsSpecific()
                self.mf.path.append(tempPath)
            pathname = mdef.filename.toOsSpecific()
            ext = mdef.filename.getExtension()
            if ext == 'pyc' or ext == 'pyo':
                fp = open(pathname, 'rb')
                stuff = ('', 'rb', _PY_COMPILED)
                self.mf.load_module(mdef.moduleName, fp, pathname, stuff)
            else:
                stuff = ('', 'rb', _PY_SOURCE)
                if mdef.text is not None:
                    fp = io.StringIO(mdef.text)
                else:
                    fp = open(pathname, 'rb')
                self.mf.load_module(mdef.moduleName, fp, pathname, stuff)
            if tempPath:
                del self.mf.path[-1]
        else:
            self.mf.import_hook(mdef.moduleName)

    def reset(self):
        if False:
            i = 10
            return i + 15
        ' After a previous call to done(), this resets the\n        FreezeTool object for a new pass.  More modules may be added\n        and dumped to a new target.  Previously-added modules are\n        remembered and will not be dumped again. '
        self.mf = None
        self.previousModules = dict(self.modules)

    def mangleName(self, moduleName):
        if False:
            for i in range(10):
                print('nop')
        return 'M_' + moduleName.replace('.', '__').replace('-', '_')

    def getAllModuleNames(self):
        if False:
            return 10
        ' Return a list of all module names that have been included\n        or forbidden, either in this current pass or in a previous\n        pass.  Module names that have been excluded are not included\n        in this list. '
        moduleNames = []
        for (newName, mdef) in list(self.modules.items()):
            if mdef.guess:
                pass
            elif mdef.exclude and (not mdef.forbid):
                pass
            else:
                moduleNames.append(newName)
        moduleNames.sort()
        return moduleNames

    def getModuleDefs(self):
        if False:
            return 10
        ' Return a list of all of the modules we will be explicitly\n        or implicitly including.  The return value is actually a list\n        of tuples: (moduleName, moduleDef).'
        moduleDefs = []
        for (newName, mdef) in list(self.modules.items()):
            prev = self.previousModules.get(newName, None)
            if not mdef.exclude:
                if prev and (not prev.exclude):
                    pass
                elif mdef.moduleName in self.mf.modules or mdef.moduleName in startupModules or mdef.filename:
                    moduleDefs.append((newName, mdef))
            elif mdef.forbid:
                if not prev or not prev.forbid:
                    moduleDefs.append((newName, mdef))
        moduleDefs.sort()
        return moduleDefs

    def __replacePaths(self):
        if False:
            while True:
                i = 10
        replace_paths = []
        for (moduleName, module) in list(self.mf.modules.items()):
            if module.__code__:
                origPathname = module.__code__.co_filename
                if origPathname:
                    replace_paths.append((origPathname, moduleName))
        self.mf.replace_paths = replace_paths
        for (moduleName, module) in list(self.mf.modules.items()):
            if module.__code__:
                co = self.mf.replace_paths_in_code(module.__code__)
                module.__code__ = co

    def __addPyc(self, multifile, filename, code, compressionLevel):
        if False:
            return 10
        if code:
            data = importlib.util.MAGIC_NUMBER + b'\x00\x00\x00\x00\x00\x00\x00\x00'
            data += marshal.dumps(code)
            stream = StringStream(data)
            multifile.addSubfile(filename, stream, compressionLevel)
            multifile.flush()

    def __addPythonDirs(self, multifile, moduleDirs, dirnames, compressionLevel):
        if False:
            print('Hello World!')
        ' Adds all of the names on dirnames as a module directory. '
        if not dirnames:
            return
        str = '.'.join(dirnames)
        if str not in moduleDirs:
            moduleName = '.'.join(dirnames)
            filename = '/'.join(dirnames) + '/__init__'
            if self.storePythonSource:
                filename += '.py'
                stream = StringStream(b'')
                if multifile.findSubfile(filename) < 0:
                    multifile.addSubfile(filename, stream, 0)
                    multifile.flush()
            else:
                if __debug__:
                    filename += '.pyc'
                else:
                    filename += '.pyo'
                if multifile.findSubfile(filename) < 0:
                    code = compile('', moduleName, 'exec', optimize=self.optimize)
                    self.__addPyc(multifile, filename, code, compressionLevel)
            moduleDirs[str] = True
            self.__addPythonDirs(multifile, moduleDirs, dirnames[:-1], compressionLevel)

    def __addPythonFile(self, multifile, moduleDirs, moduleName, mdef, compressionLevel):
        if False:
            i = 10
            return i + 15
        ' Adds the named module to the multifile as a .pyc file. '
        dirnames = moduleName.split('.')
        if len(dirnames) > 1 and dirnames[-1] == '__init__':
            dirnames = dirnames[:-1]
        self.__addPythonDirs(multifile, moduleDirs, dirnames[:-1], compressionLevel)
        filename = '/'.join(dirnames)
        module = self.mf.modules.get(mdef.moduleName, None)
        if getattr(module, '__path__', None) is not None or (getattr(module, '__file__', None) is not None and getattr(module, '__file__').endswith('/__init__.py')):
            filename += '/__init__'
            moduleDirs[moduleName] = True
            multifile.removeSubfile(filename + '.py')
            if __debug__:
                multifile.removeSubfile(filename + '.pyc')
            else:
                multifile.removeSubfile(filename + '.pyo')
        sourceFilename = None
        if mdef.filename and mdef.filename.getExtension() == 'py':
            sourceFilename = mdef.filename
        elif getattr(module, '__file__', None):
            sourceFilename = Filename.fromOsSpecific(module.__file__)
            sourceFilename.setExtension('py')
            sourceFilename.setText()
        if self.storePythonSource:
            if sourceFilename and sourceFilename.exists():
                filename += '.py'
                multifile.addSubfile(filename, sourceFilename, compressionLevel)
                return
        if __debug__:
            filename += '.pyc'
        else:
            filename += '.pyo'
        code = None
        if module:
            code = getattr(module, '__code__', None)
            if not code:
                extensionFilename = getattr(module, '__file__', None)
                if extensionFilename:
                    self.extras.append((moduleName, extensionFilename))
                else:
                    pass
        elif sourceFilename and sourceFilename.exists():
            source = open(sourceFilename.toOsSpecific(), 'r').read()
            if source and source[-1] != '\n':
                source = source + '\n'
            code = compile(source, str(sourceFilename), 'exec', optimize=self.optimize)
        self.__addPyc(multifile, filename, code, compressionLevel)

    def addToMultifile(self, multifile, compressionLevel=0):
        if False:
            return 10
        ' After a call to done(), this stores all of the accumulated\n        python code into the indicated Multifile.  Additional\n        extension modules are listed in self.extras.  '
        moduleDirs = {}
        for (moduleName, mdef) in self.getModuleDefs():
            if not mdef.exclude:
                self.__addPythonFile(multifile, moduleDirs, moduleName, mdef, compressionLevel)

    def writeMultifile(self, mfname):
        if False:
            while True:
                i = 10
        ' After a call to done(), this stores all of the accumulated\n        python code into a Multifile with the indicated filename,\n        including the extension.  Additional extension modules are\n        listed in self.extras.'
        self.__replacePaths()
        Filename(mfname).unlink()
        multifile = Multifile()
        if not multifile.openReadWrite(mfname):
            raise Exception
        self.addToMultifile(multifile)
        multifile.flush()
        multifile.repack()

    def writeCode(self, filename, initCode=''):
        if False:
            return 10
        ' After a call to done(), this freezes all of the accumulated\n        Python code into a C source file. '
        self.__replacePaths()
        moduleDefs = []
        moduleList = []
        for (moduleName, mdef) in self.getModuleDefs():
            origName = mdef.moduleName
            if mdef.forbid:
                moduleList.append(self.makeForbiddenModuleListEntry(moduleName))
                continue
            assert not mdef.exclude
            module = self.mf.modules.get(origName, None)
            code = getattr(module, '__code__', None)
            if code:
                code = marshal.dumps(code)
                mangledName = self.mangleName(moduleName)
                moduleDefs.append(self.makeModuleDef(mangledName, code))
                moduleList.append(self.makeModuleListEntry(mangledName, code, moduleName, module))
                continue
            extensionFilename = getattr(module, '__file__', None)
            if extensionFilename or self.linkExtensionModules:
                self.extras.append((moduleName, extensionFilename))
            if '.' in moduleName and self.linkExtensionModules:
                code = compile('import sys;del sys.modules["%s"];from importlib._bootstrap import _builtin_from_name;_builtin_from_name("%s")' % (moduleName, moduleName), moduleName, 'exec', optimize=self.optimize)
                code = marshal.dumps(code)
                mangledName = self.mangleName(moduleName)
                moduleDefs.append(self.makeModuleDef(mangledName, code))
                moduleList.append(self.makeModuleListEntry(mangledName, code, moduleName, None))
            elif '.' in moduleName:
                print('WARNING: Python cannot import extension modules under frozen Python packages; %s will be inaccessible.  passing either -l to link in extension modules or use -x %s to exclude the entire package.' % (moduleName, moduleName.split('.')[0]))
        text = programFile % {'moduleDefs': '\n'.join(moduleDefs), 'moduleList': '\n'.join(moduleList)}
        if self.linkExtensionModules and self.extras:
            text += '#if PY_MAJOR_VERSION >= 3\n'
            for (module, fn) in self.extras:
                if sys.platform != 'win32' or fn:
                    libName = module.split('.')[-1]
                    initFunc = builtinInitFuncs.get(module, 'PyInit_' + libName)
                    if initFunc:
                        text += 'extern PyAPI_FUNC(PyObject) *%s(void);\n' % initFunc
            text += '\n'
            if sys.platform == 'win32':
                text += 'static struct _inittab extensions[] = {\n'
            else:
                text += 'struct _inittab _PyImport_Inittab[] = {\n'
            for (module, fn) in self.extras:
                if sys.platform != 'win32' or fn:
                    libName = module.split('.')[-1]
                    initFunc = builtinInitFuncs.get(module, 'PyInit_' + libName) or 'NULL'
                    text += '  {"%s", %s},\n' % (module, initFunc)
            text += '  {0, 0},\n'
            text += '};\n\n'
            text += '#else\n'
            for (module, fn) in self.extras:
                if sys.platform != 'win32' or fn:
                    libName = module.split('.')[-1]
                    initFunc = builtinInitFuncs.get(module, 'init' + libName)
                    if initFunc:
                        text += 'extern PyAPI_FUNC(void) %s(void);\n' % initFunc
            text += '\n'
            if sys.platform == 'win32':
                text += 'static struct _inittab extensions[] = {\n'
            else:
                text += 'struct _inittab _PyImport_Inittab[] = {\n'
            for (module, fn) in self.extras:
                if sys.platform != 'win32' or fn:
                    libName = module.split('.')[-1]
                    initFunc = builtinInitFuncs.get(module, 'init' + libName) or 'NULL'
                    text += '  {"%s", %s},\n' % (module, initFunc)
            text += '  {0, 0},\n'
            text += '};\n'
            text += '#endif\n\n'
        elif sys.platform == 'win32':
            text += 'static struct _inittab extensions[] = {\n'
            text += '  {0, 0},\n'
            text += '};\n\n'
        text += initCode
        if filename is not None:
            file = open(filename, 'w')
            file.write(text)
            file.close()

    def generateCode(self, basename, compileToExe=False):
        if False:
            return 10
        ' After a call to done(), this freezes all of the\n        accumulated python code into either an executable program (if\n        compileToExe is true) or a dynamic library (if compileToExe is\n        false).  The basename is the name of the file to write,\n        without the extension.\n\n        The return value is the newly-generated filename, including\n        the filename extension.  Additional extension modules are\n        listed in self.extras. '
        if compileToExe:
            if not self.__writingModule('__main__'):
                message = "Can't generate an executable without a __main__ module."
                raise Exception(message)
        filename = basename + self.sourceExtension
        dllexport = ''
        dllimport = ''
        if self.platform.startswith('win'):
            dllexport = '__declspec(dllexport) '
            dllimport = '__declspec(dllimport) '
        if not self.cenv:
            self.cenv = CompilationEnvironment(platform=self.platform)
        if compileToExe:
            code = self.frozenMainCode
            decls = ''
            calls = ''
            for func in self.extraInitFuncs:
                if isinstance(func, str):
                    func = ('void', func)
                decls += f'extern {func[0]} {func[1]}();\n'
                calls += f'    {func[1]}();\n'
            code = code.replace('EXTRA_INIT_FUNC_DECLS', decls)
            code = code.replace('EXTRA_INIT_FUNC_CALLS', calls)
            if self.platform.startswith('win'):
                code += self.frozenDllMainCode
            initCode = self.mainInitCode % {'frozenMainCode': code, 'programName': os.path.basename(basename), 'dllexport': dllexport, 'dllimport': dllimport}
            if self.platform.startswith('win'):
                target = basename + '.exe'
            else:
                target = basename
            compileFunc = self.cenv.compileExe
        else:
            if self.platform.startswith('win'):
                target = basename + self.cenv.dllext + '.pyd'
            else:
                target = basename + '.so'
            initCode = dllInitCode % {'moduleName': os.path.basename(basename), 'dllexport': dllexport, 'dllimport': dllimport}
            compileFunc = self.cenv.compileDll
        self.writeCode(filename, initCode=initCode)
        cleanFiles = [filename, basename + self.objectExtension]
        extraLink = []
        if self.linkExtensionModules:
            for (mod, fn) in self.extras:
                if not fn:
                    continue
                if sys.platform == 'win32':
                    libsdir = os.path.join(sys.exec_prefix, 'libs')
                    libfile = os.path.join(libsdir, mod + '.lib')
                    if os.path.isfile(libfile):
                        extraLink.append(mod + '.lib')
                        continue
                    modname = mod.split('.')[-1]
                    libfile = modname + '.lib'
                    symbolName = 'PyInit_' + modname
                    os.system('lib /nologo /def /export:%s /name:%s.pyd /out:%s' % (symbolName, modname, libfile))
                    extraLink.append(libfile)
                    cleanFiles += [libfile, modname + '.exp']
                else:
                    extraLink.append(fn)
        try:
            compileFunc(filename, basename, extraLink=extraLink)
        finally:
            if not self.keepTemporaryFiles:
                for file in cleanFiles:
                    if os.path.exists(file):
                        os.unlink(file)
        return target

    def generateRuntimeFromStub(self, target, stub_file, use_console, fields={}, log_append=False, log_filename_strftime=False):
        if False:
            while True:
                i = 10
        self.__replacePaths()
        if not self.__writingModule('__main__'):
            message = "Can't generate an executable without a __main__ module."
            raise Exception(message)
        if self.platform.startswith('win'):
            modext = '.pyd'
        else:
            modext = '.so'
        pool = b''
        strings = set()
        for (moduleName, mdef) in self.getModuleDefs():
            strings.add(moduleName.encode('ascii'))
        for value in fields.values():
            if value is not None:
                strings.add(value.encode('utf-8'))
        strings = sorted(strings, key=lambda str: -len(str))
        string_offsets = {}
        for string in strings:
            offset = pool.find(string + b'\x00')
            if offset < 0:
                offset = len(pool)
                pool += string + b'\x00'
            string_offsets[string] = offset
        moduleList = []
        for (moduleName, mdef) in self.getModuleDefs():
            origName = mdef.moduleName
            if mdef.forbid:
                moduleList.append((moduleName, 0, 0))
                continue
            if len(pool) & 3 != 0:
                pad = 4 - (len(pool) & 3)
                pool += b'\x00' * pad
            assert not mdef.exclude
            module = self.mf.modules.get(origName, None)
            code = getattr(module, '__code__', None)
            if code:
                code = marshal.dumps(code)
                size = len(code)
                if getattr(module, '__path__', None):
                    size = -size
                moduleList.append((moduleName, len(pool), size))
                pool += code
                continue
            extensionFilename = getattr(module, '__file__', None)
            if extensionFilename:
                self.extras.append((moduleName, extensionFilename))
            if '.' in moduleName and (not self.platform.startswith('android')):
                if self.platform.startswith('macosx') and (not use_console):
                    direxpr = 'sys.path[0]'
                else:
                    direxpr = 'os.path.dirname(sys.executable)'
                code = f'import sys;del sys.modules["{moduleName}"];import sys,os;from importlib.machinery import ExtensionFileLoader,ModuleSpec;from importlib._bootstrap import _load;path=os.path.join({direxpr}, "{moduleName}{modext}");_load(ModuleSpec(name="{moduleName}", loader=ExtensionFileLoader("{moduleName}", path), origin=path))'
                code = compile(code, moduleName, 'exec', optimize=self.optimize)
                code = marshal.dumps(code)
                moduleList.append((moduleName, len(pool), len(code)))
                pool += code
        num_pointers = 12
        stub_data = bytearray(stub_file.read())
        bitnesses = self._get_executable_bitnesses(stub_data)
        header_layouts = {32: '<QQHHHH8x%dII' % num_pointers, 64: '<QQHHHH8x%dQQ' % num_pointers}
        entry_layouts = {32: '<IIi', 64: '<QQixxxx'}
        bitnesses = sorted(bitnesses, reverse=True)
        pool_offset = 0
        for bitness in bitnesses:
            pool_offset += (len(moduleList) + 1) * struct.calcsize(entry_layouts[bitness])
        if self.platform.startswith('win'):
            blob_align = 32
        elif self.platform.endswith('_aarch64') or self.platform.endswith('_arm64'):
            blob_align = 16384
        else:
            blob_align = 4096
        blob_size = pool_offset + len(pool)
        if blob_size & blob_align - 1 != 0:
            pad = blob_align - (blob_size & blob_align - 1)
            blob_size += pad
        append_blob = True
        if self.platform.startswith('macosx') and len(bitnesses) == 1:
            load_commands = self._parse_macho_load_commands(stub_data)
            if b'__PANDA' in load_commands.keys():
                append_blob = False
        if self.platform.startswith('macosx') and (not append_blob):
            blob_offset = self._shift_macho_structures(stub_data, load_commands, blob_size)
        else:
            blob_offset = len(stub_data)
            if blob_offset & blob_align - 1 != 0:
                pad = blob_align - (blob_offset & blob_align - 1)
                stub_data += b'\x00' * pad
                blob_offset += pad
            assert blob_offset % blob_align == 0
            assert blob_offset == len(stub_data)
        field_offsets = {}
        for (key, value) in fields.items():
            if value is not None:
                encoded = value.encode('utf-8')
                field_offsets[key] = pool_offset + string_offsets[encoded]
        blob = b''
        append_offset = False
        for bitness in bitnesses:
            entry_layout = entry_layouts[bitness]
            header_layout = header_layouts[bitness]
            table_offset = len(blob)
            for (moduleName, offset, size) in moduleList:
                encoded = moduleName.encode('ascii')
                string_offset = pool_offset + string_offsets[encoded]
                if size != 0:
                    offset += pool_offset
                blob += struct.pack(entry_layout, string_offset, offset, size)
            blob += struct.pack(entry_layout, 0, 0, 0)
            flags = 0
            if log_append:
                flags |= 1
            if log_filename_strftime:
                flags |= 2
            if self.optimize < 2:
                flags |= 4
            header = struct.pack(header_layout, blob_offset, blob_size, 1, num_pointers, 0, flags, table_offset, field_offsets.get('prc_data', 0), field_offsets.get('default_prc_dir', 0), field_offsets.get('prc_dir_envvars', 0), field_offsets.get('prc_path_envvars', 0), field_offsets.get('prc_patterns', 0), field_offsets.get('prc_encrypted_patterns', 0), field_offsets.get('prc_encryption_key', 0), field_offsets.get('prc_executable_patterns', 0), field_offsets.get('prc_executable_args_envvar', 0), field_offsets.get('main_dir', 0), field_offsets.get('log_filename', 0), 0)
            if not self._replace_symbol(stub_data, b'blobinfo', header, bitness=bitness):
                append_offset = True
        assert len(blob) == pool_offset
        blob += pool
        del pool
        if len(blob) < blob_size:
            blob += b'\x00' * (blob_size - len(blob))
        assert len(blob) == blob_size
        if append_offset:
            warnings.warn('Could not find blob header. Is deploy-stub outdated?')
            blob += struct.pack('<Q', blob_offset)
        with open(target, 'wb') as f:
            if append_blob:
                f.write(stub_data)
                assert f.tell() == blob_offset
                f.write(blob)
            else:
                stub_data[blob_offset:blob_offset + blob_size] = blob
                f.write(stub_data)
        os.chmod(target, 493)
        return target

    def _get_executable_bitnesses(self, data):
        if False:
            for i in range(10):
                print('nop')
        'Returns the bitnesses (32 or 64) of the given executable data.\n        This will contain 1 element for non-fat executables.'
        if data.startswith(b'MZ'):
            (offset,) = struct.unpack_from('<I', data, 60)
            assert data[offset:offset + 4] == b'PE\x00\x00'
            (magic,) = struct.unpack_from('<H', data, offset + 24)
            assert magic in (267, 523)
            if magic == 523:
                return (64,)
            else:
                return (32,)
        elif data.startswith(b'\x7fELF'):
            elfclass = ord(data[4:5])
            assert elfclass in (1, 2)
            return (elfclass * 32,)
        elif data[:4] in (b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe'):
            return (32,)
        elif data[:4] in (b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe'):
            return (64,)
        elif data[:4] in (b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'):
            (num_fat,) = struct.unpack_from('>I', data, 4)
            bitnesses = set()
            ptr = 8
            for i in range(num_fat):
                (cputype, cpusubtype, offset, size, align) = struct.unpack_from('>IIIII', data, ptr)
                ptr += 20
                if cputype & 16777216 != 0:
                    bitnesses.add(64)
                else:
                    bitnesses.add(32)
            return tuple(bitnesses)
        elif data[:4] in (b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'):
            (num_fat,) = struct.unpack_from('>I', data, 4)
            bitnesses = set()
            ptr = 8
            for i in range(num_fat):
                (cputype, cpusubtype, offset, size, align) = struct.unpack_from('>QQQQQ', data, ptr)
                ptr += 40
                if cputype & 16777216 != 0:
                    bitnesses.add(64)
                else:
                    bitnesses.add(32)
            return tuple(bitnesses)

    def _replace_symbol(self, data, symbol_name, replacement, bitness=None):
        if False:
            print('Hello World!')
        'We store a custom section in the binary file containing a header\n        containing offsets to the binary data.\n        If bitness is set, and the binary in question is a macOS universal\n        binary, it only replaces for binaries with the given bitness. '
        if data.startswith(b'MZ'):
            pe = pefile.PEFile()
            pe.read(io.BytesIO(data))
            addr = pe.get_export_address(symbol_name)
            if addr is not None:
                offset = pe.get_address_offset(addr)
                if offset is not None:
                    data[offset:offset + len(replacement)] = replacement
                    return True
        elif data.startswith(b'\x7fELF'):
            return self._replace_symbol_elf(data, symbol_name, replacement)
        elif data[:4] in (b'\xfe\xed\xfa\xce', b'\xce\xfa\xed\xfe', b'\xfe\xed\xfa\xcf', b'\xcf\xfa\xed\xfe'):
            off = self._find_symbol_macho(data, symbol_name)
            if off is not None:
                data[off:off + len(replacement)] = replacement
                return True
            return False
        elif data[:4] in (b'\xca\xfe\xba\xbe', b'\xbe\xba\xfe\xca'):
            (num_fat,) = struct.unpack_from('>I', data, 4)
            replaced = False
            ptr = 8
            for i in range(num_fat):
                (cputype, cpusubtype, offset, size, align) = struct.unpack_from('>IIIII', data, ptr)
                ptr += 20
                if bitness is not None and (cputype & 16777216 != 0) != (bitness == 64):
                    continue
                macho_data = data[offset:offset + size]
                off = self._find_symbol_macho(macho_data, symbol_name)
                if off is not None:
                    off += offset
                    data[off:off + len(replacement)] = replacement
                    replaced = True
            return replaced
        elif data[:4] in (b'\xca\xfe\xba\xbf', b'\xbf\xba\xfe\xca'):
            (num_fat,) = struct.unpack_from('>I', data, 4)
            replaced = False
            ptr = 8
            for i in range(num_fat):
                (cputype, cpusubtype, offset, size, align) = struct.unpack_from('>QQQQQ', data, ptr)
                ptr += 40
                if bitness is not None and (cputype & 16777216 != 0) != (bitness == 64):
                    continue
                macho_data = data[offset:offset + size]
                off = self._find_symbol_macho(macho_data, symbol_name)
                if off is not None:
                    off += offset
                    data[off:off + len(replacement)] = replacement
                    replaced = True
            return replaced
        return False

    def _replace_symbol_elf(self, elf_data, symbol_name, replacement):
        if False:
            while True:
                i = 10
        ' The Linux/FreeBSD implementation of _replace_symbol. '
        replaced = False
        endian = '<>'[ord(elf_data[5:6]) - 1]
        is_64bit = ord(elf_data[4:5]) - 1
        header_struct = endian + ('HHIIIIIHHHHHH', 'HHIQQQIHHHHHH')[is_64bit]
        section_struct = endian + ('4xI4xIIII8xI', '4xI8xQQQI12xQ')[is_64bit]
        symbol_struct = endian + ('IIIBBH', 'IBBHQQ')[is_64bit]
        header_size = struct.calcsize(header_struct)
        (type, machine, version, entry, phoff, shoff, flags, ehsize, phentsize, phnum, shentsize, shnum, shstrndx) = struct.unpack_from(header_struct, elf_data, 16)
        section_offsets = []
        symbol_tables = []
        string_tables = {}
        ptr = shoff
        for i in range(shnum):
            (type, addr, offset, size, link, entsize) = struct.unpack_from(section_struct, elf_data[ptr:ptr + shentsize])
            ptr += shentsize
            section_offsets.append(offset - addr)
            if type == 11 and link != 0:
                symbol_tables.append((offset, size, link, entsize))
                string_tables[link] = None
        for idx in list(string_tables.keys()):
            ptr = shoff + idx * shentsize
            (type, addr, offset, size, link, entsize) = struct.unpack_from(section_struct, elf_data[ptr:ptr + shentsize])
            if type == 3:
                string_tables[idx] = elf_data[offset:offset + size]
        for (offset, size, link, entsize) in symbol_tables:
            entries = size // entsize
            for i in range(entries):
                ptr = offset + i * entsize
                fields = struct.unpack_from(symbol_struct, elf_data[ptr:ptr + entsize])
                if is_64bit:
                    (name, info, other, shndx, value, size) = fields
                else:
                    (name, value, size, info, other, shndx) = fields
                if not name:
                    continue
                name = string_tables[link][name:string_tables[link].find(b'\x00', name)]
                if name == symbol_name:
                    if shndx == 0:
                        continue
                    elif shndx >= 65280 and shndx <= 65535:
                        assert False
                    else:
                        off = section_offsets[shndx] + value
                        elf_data[off:off + len(replacement)] = replacement
                        replaced = True
        return replaced

    def _find_symbol_macho(self, macho_data, symbol_name):
        if False:
            return 10
        ' Returns the offset of the given symbol in the binary file. '
        if macho_data[:4] in (b'\xce\xfa\xed\xfe', b'\xcf\xfa\xed\xfe'):
            endian = '<'
        else:
            endian = '>'
        (cputype, cpusubtype, filetype, ncmds, sizeofcmds, flags) = struct.unpack_from(endian + 'IIIIII', macho_data, 4)
        is_64bit = cputype & 16777216 != 0
        segments = []
        cmd_ptr = 28
        nlist_struct = endian + 'IBBHI'
        if is_64bit:
            nlist_struct = endian + 'IBBHQ'
            cmd_ptr += 4
        nlist_size = struct.calcsize(nlist_struct)
        for i in range(ncmds):
            (cmd, cmd_size) = struct.unpack_from(endian + 'II', macho_data, cmd_ptr)
            cmd_data = macho_data[cmd_ptr + 8:cmd_ptr + cmd_size]
            cmd_ptr += cmd_size
            cmd &= ~2147483648
            if cmd == 1:
                (segname, vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects, flags) = struct.unpack_from(endian + '16sIIIIIIII', cmd_data)
                segments.append((vmaddr, vmsize, fileoff))
            elif cmd == 25:
                (segname, vmaddr, vmsize, fileoff, filesize, maxprot, initprot, nsects, flags) = struct.unpack_from(endian + '16sQQQQIIII', cmd_data)
                segments.append((vmaddr, vmsize, fileoff))
            elif cmd == 2:
                (symoff, nsyms, stroff, strsize) = struct.unpack_from(endian + 'IIII', cmd_data)
                strings = macho_data[stroff:stroff + strsize]
                for j in range(nsyms):
                    (strx, type, sect, desc, value) = struct.unpack_from(nlist_struct, macho_data, symoff)
                    symoff += nlist_size
                    name = strings[strx:strings.find(b'\x00', strx)]
                    if name == b'_' + symbol_name and type & 224 == 0:
                        for (vmaddr, vmsize, fileoff) in segments:
                            rel = value - vmaddr
                            if rel >= 0 and rel < vmsize:
                                return fileoff + rel
                        print('Could not find memory address for symbol %s' % symbol_name)

    def _parse_macho_load_commands(self, macho_data):
        if False:
            for i in range(10):
                print('nop')
        'Returns the list of load commands from macho_data.'
        mach_header_64 = list(struct.unpack_from(mach_header_64_layout, macho_data, 0))
        num_load_commands = mach_header_64[4]
        load_commands = {}
        curr_lc_offset = struct.calcsize(mach_header_64_layout)
        for i in range(num_load_commands):
            lc = struct.unpack_from(lc_header_layout, macho_data, curr_lc_offset)
            layout = lc_layouts.get(lc[0])
            if layout:
                lc = list(struct.unpack_from(layout, macho_data, curr_lc_offset))
                if lc[0] == LC_SEGMENT_64:
                    stripped_name = lc[2].rstrip(b'\x00')
                    if stripped_name in [b'__PANDA', b'__LINKEDIT']:
                        load_commands[stripped_name] = (curr_lc_offset, lc)
                else:
                    load_commands[lc[0]] = (curr_lc_offset, lc)
            curr_lc_offset += lc[1]
        return load_commands

    def _shift_macho_structures(self, macho_data, load_commands, blob_size):
        if False:
            print('Hello World!')
        'Given the stub and the size of our blob, make room for it and edit\n        all of the necessary structures to keep the binary valid. Returns the\n        offset where the blob should be placed.'
        for lc_key in load_commands.keys():
            for index in lc_indices_to_slide[lc_key]:
                load_commands[lc_key][1][index] += blob_size
            if lc_key == b'__PANDA':
                section_header_offset = load_commands[lc_key][0] + struct.calcsize(lc_layouts[LC_SEGMENT_64])
                section_header = list(struct.unpack_from(section64_header_layout, macho_data, section_header_offset))
                section_header[3] = blob_size
                struct.pack_into(section64_header_layout, macho_data, section_header_offset, *section_header)
            layout = LC_SEGMENT_64 if lc_key in [b'__PANDA', b'__LINKEDIT'] else lc_key
            struct.pack_into(lc_layouts[layout], macho_data, load_commands[lc_key][0], *load_commands[lc_key][1])
        blob_offset = load_commands[b'__PANDA'][1][5]
        macho_data[blob_offset:blob_offset] = b'\x00' * blob_size
        return blob_offset

    def makeModuleDef(self, mangledName, code):
        if False:
            print('Hello World!')
        lines = ',\n  '.join((','.join(map(str, code[i:i + 16])) for i in range(0, len(code), 16)))
        return f'static unsigned char {mangledName}[] = {{\n  {lines}\n}};\n'

    def makeModuleListEntry(self, mangledName, code, moduleName, module):
        if False:
            return 10
        size = len(code)
        if getattr(module, '__path__', None):
            size = -size
        return '  {"%s", %s, %s},' % (moduleName, mangledName, size)

    def makeForbiddenModuleListEntry(self, moduleName):
        if False:
            i = 10
            return i + 15
        return '  {"%s", NULL, 0},' % moduleName

    def __writingModule(self, moduleName):
        if False:
            print('Hello World!')
        ' Returns true if we are outputting the named module in this\n        pass, false if we have already output in a previous pass, or\n        if it is not yet on the output table. '
        mdef = self.modules.get(moduleName, (None, None))
        if mdef.exclude:
            return False
        if moduleName in self.previousModules:
            return False
        return True

class PandaModuleFinder(modulefinder.ModuleFinder):

    def __init__(self, *args, **kw):
        if False:
            while True:
                i = 10
        '\n        :param path: search path to look on, defaults to sys.path\n        :param suffixes: defaults to imp.get_suffixes()\n        :param excludes: a list of modules to exclude\n        :param debug: an integer indicating the level of verbosity\n        '
        self.builtin_module_names = kw.pop('builtin_module_names', sys.builtin_module_names)
        self.suffixes = kw.pop('suffixes', [(s, 'rb', _C_EXTENSION) for s in machinery.EXTENSION_SUFFIXES] + [(s, 'r', _PY_SOURCE) for s in machinery.SOURCE_SUFFIXES] + [(s, 'rb', _PY_COMPILED) for s in machinery.BYTECODE_SUFFIXES])
        self.optimize = kw.pop('optimize', -1)
        modulefinder.ModuleFinder.__init__(self, *args, **kw)
        self._zip_files = {}

    def _open_file(self, path, mode):
        if False:
            print('Hello World!')
        ' Opens a module at the given path, which may contain a zip file.\n        Returns None if the module could not be found. '
        if os.path.isfile(path):
            if 'b' not in mode:
                return io.open(path, mode, encoding='utf8')
            else:
                return open(path, mode)
        (dir, dirname) = os.path.split(path)
        fn = dirname
        while dirname:
            if os.path.isfile(dir):
                if dir in self._zip_files:
                    zip = self._zip_files[dir]
                elif zipfile.is_zipfile(dir):
                    zip = zipfile.ZipFile(dir)
                    self._zip_files[dir] = zip
                else:
                    return None
                try:
                    zip_fn = fn.replace(os.path.sep, '/')
                    if zip_fn.startswith('deploy_libs/_tkinter.'):
                        if any((entry.endswith('.whl') and os.path.basename(entry).startswith('tkinter-') for entry in self.path)):
                            return None
                    fp = zip.open(zip_fn, 'r')
                except KeyError:
                    return None
                if 'b' not in mode:
                    return io.TextIOWrapper(fp, encoding='utf8')
                return fp
            (dir, dirname) = os.path.split(dir)
            fn = os.path.join(dirname, fn)
        return None

    def _file_exists(self, path):
        if False:
            while True:
                i = 10
        if os.path.exists(path):
            return os.path.isfile(path)
        fh = self._open_file(path, 'rb')
        if fh:
            fh.close()
            return True
        return False

    def _dir_exists(self, path):
        if False:
            for i in range(10):
                print('nop')
        'Returns True if the given directory exists, either on disk or inside\n        a wheel.'
        if os.path.isdir(path):
            return True
        (dir, dirname) = os.path.split(path.rstrip(os.path.sep + '/'))
        fn = dirname
        while dirname:
            if os.path.isfile(dir):
                if dir in self._zip_files:
                    zip = self._zip_files[dir]
                elif zipfile.is_zipfile(dir):
                    zip = zipfile.ZipFile(dir)
                    self._zip_files[dir] = zip
                else:
                    return None
                prefix = fn.replace(os.path.sep, '/') + '/'
                for name in zip.namelist():
                    if name.startswith(prefix):
                        return True
                return False
            (dir, dirname) = os.path.split(dir)
            fn = os.path.join(dirname, fn)
        return False

    def _listdir(self, path):
        if False:
            i = 10
            return i + 15
        'Lists files in the given directory if it exists.'
        if os.path.isdir(path):
            return os.listdir(path)
        (dir, dirname) = os.path.split(path.rstrip(os.path.sep + '/'))
        fn = dirname
        while dirname:
            if os.path.isfile(dir):
                if dir in self._zip_files:
                    zip = self._zip_files[dir]
                elif zipfile.is_zipfile(dir):
                    zip = zipfile.ZipFile(dir)
                    self._zip_files[dir] = zip
                else:
                    return []
                prefix = fn.replace(os.path.sep, '/') + '/'
                result = []
                for name in zip.namelist():
                    if name.startswith(prefix) and '/' not in name[len(prefix):]:
                        result.append(name[len(prefix):])
                return result
            (dir, dirname) = os.path.split(dir)
            fn = os.path.join(dirname, fn)
        return []

    def load_module(self, fqname, fp, pathname, file_info):
        if False:
            while True:
                i = 10
        'Copied from ModuleFinder.load_module with fixes to handle sending bytes\n        to compile() for PY_SOURCE types. Sending bytes to compile allows it to\n        handle file encodings.'
        (suffix, mode, type) = file_info
        self.msgin(2, 'load_module', fqname, fp and 'fp', pathname)
        if type == _PKG_DIRECTORY:
            m = self.load_package(fqname, pathname)
            self.msgout(2, 'load_module ->', m)
            return m
        if type is _PKG_NAMESPACE_DIRECTORY:
            m = self.add_module(fqname)
            m.__code__ = compile('', '', 'exec', optimize=self.optimize)
            m.__path__ = pathname
            return m
        if type == _PY_SOURCE:
            if fqname in overrideModules:
                code = overrideModules[fqname]
            else:
                code = fp.read()
            if isinstance(code, bytes):
                start_marker = b'# start delvewheel patch'
                end_marker = b'# end delvewheel patch'
            else:
                start_marker = '# start delvewheel patch'
                end_marker = '# end delvewheel patch'
            start = code.find(start_marker)
            while start >= 0:
                end = code.find(end_marker, start) + len(end_marker)
                code = code[:start] + code[end:]
                start = code.find(start_marker)
            code += b'\n' if isinstance(code, bytes) else '\n'
            co = compile(code, pathname, 'exec', optimize=self.optimize)
        elif type == _PY_COMPILED:
            if sys.version_info >= (3, 7):
                try:
                    data = fp.read()
                    importlib._bootstrap_external._classify_pyc(data, fqname, {})
                except ImportError as exc:
                    self.msgout(2, 'raise ImportError: ' + str(exc), pathname)
                    raise
                co = marshal.loads(memoryview(data)[16:])
            else:
                try:
                    marshal_data = importlib._bootstrap_external._validate_bytecode_header(fp.read())
                except ImportError as exc:
                    self.msgout(2, 'raise ImportError: ' + str(exc), pathname)
                    raise
                co = marshal.loads(marshal_data)
        else:
            co = None
        m = self.add_module(fqname)
        m.__file__ = pathname
        if co:
            if self.replace_paths:
                co = self.replace_paths_in_code(co)
            m.__code__ = co
            self.scan_code(co, m)
        self.msgout(2, 'load_module ->', m)
        return m

    def _safe_import_hook(self, name, caller, fromlist, level=-1):
        if False:
            return 10
        if name in self.badmodules:
            self._add_badmodule(name, caller)
            return
        if level <= 0 and caller and (caller.__name__ in ignoreImports):
            if name in ignoreImports[caller.__name__]:
                return
        try:
            self.import_hook(name, caller, level=level)
        except ImportError as msg:
            self.msg(2, 'ImportError:', str(msg))
            self._add_badmodule(name, caller)
        except SyntaxError as msg:
            self.msg(2, 'SyntaxError:', str(msg))
            self._add_badmodule(name, caller)
        else:
            if fromlist:
                for sub in fromlist:
                    fullname = name + '.' + sub
                    if fullname in self.badmodules:
                        self._add_badmodule(fullname, caller)
                        continue
                    try:
                        self.import_hook(name, caller, [sub], level=level)
                    except ImportError as msg:
                        self.msg(2, 'ImportError:', str(msg))
                        self._add_badmodule(fullname, caller)

    def scan_code(self, co, m):
        if False:
            while True:
                i = 10
        code = co.co_code
        if hasattr(self, 'scan_opcodes_25'):
            scanner = self.scan_opcodes_25
        else:
            scanner = self.scan_opcodes
        for (what, args) in scanner(co):
            if what == 'store':
                (name,) = args
                m.globalnames[name] = 1
            elif what in ('import', 'absolute_import'):
                (fromlist, name) = args
                have_star = 0
                if fromlist is not None:
                    if '*' in fromlist:
                        have_star = 1
                    fromlist = [f for f in fromlist if f != '*']
                if what == 'absolute_import':
                    level = 0
                else:
                    level = -1
                self._safe_import_hook(name, m, fromlist, level=level)
                if have_star:
                    mm = None
                    if m.__path__:
                        mm = self.modules.get(m.__name__ + '.' + name)
                    if mm is None:
                        mm = self.modules.get(name)
                    if mm is not None:
                        m.globalnames.update(mm.globalnames)
                        m.starimports.update(mm.starimports)
                        if mm.__code__ is None:
                            m.starimports[name] = 1
                    else:
                        m.starimports[name] = 1
            elif what == 'relative_import':
                (level, fromlist, name) = args
                parent = self.determine_parent(m, level=level)
                if name:
                    self._safe_import_hook(name, m, fromlist, level=level)
                else:
                    self._safe_import_hook(parent.__name__, None, fromlist, level=0)
                if fromlist and '*' in fromlist:
                    if name:
                        mm = self.modules.get(parent.__name__ + '.' + name)
                    else:
                        mm = self.modules.get(parent.__name__)
                    if mm is not None:
                        m.globalnames.update(mm.globalnames)
                        m.starimports.update(mm.starimports)
                        if mm.__code__ is None:
                            m.starimports[name] = 1
                    else:
                        m.starimports[name] = 1
            else:
                raise RuntimeError(what)
        for c in co.co_consts:
            if isinstance(c, type(co)):
                self.scan_code(c, m)

    def find_module(self, name, path=None, parent=None):
        if False:
            while True:
                i = 10
        ' Finds a module with the indicated name on the given search path\n        (or self.path if None).  Returns a tuple like (fp, path, stuff), where\n        stuff is a tuple like (suffix, mode, type). '
        if parent is not None:
            fullname = parent.__name__ + '.' + name
        else:
            fullname = name
        if fullname in self.excludes:
            raise ImportError(name)
        if fullname in overrideModules:
            return (None, '', ('.py', 'r', _PY_SOURCE))
        if fullname in self.builtin_module_names:
            return (None, None, ('', '', _C_BUILTIN))
        if path is None:
            path = self.path
            if fullname == 'distutils' and hasattr(sys, 'real_prefix'):
                try:
                    (fp, fn, stuff) = self.find_module('opcode')
                    if fn:
                        path = [os.path.dirname(fn)] + path
                except ImportError:
                    pass
            elif fullname == 'distutils' and 'setuptools' in self.modules and ('_distutils_hack.override' in self.modules):
                setuptools = self.modules['setuptools']
                return self.find_module('_distutils', setuptools.__path__, parent=setuptools)
        elif parent is not None and parent.__name__ in ('setuptools.extern', 'pkg_resources.extern'):
            root = self.modules[parent.__name__.split('.', 1)[0]]
            try:
                (fp, fn, stuff) = self.find_module('_vendor', root.__path__, parent=root)
                vendor = self.load_module(root.__name__ + '._vendor', fp, fn, stuff)
                return self.find_module(name, vendor.__path__, parent=vendor)
            except ImportError:
                pass
        ns_dirs = []
        for dir_path in path:
            basename = os.path.join(dir_path, name.split('.')[-1])
            for stuff in self.suffixes:
                (suffix, mode, _) = stuff
                fp = self._open_file(basename + suffix, mode)
                if fp:
                    return (fp, basename + suffix, stuff)
            for (suffix, mode, _) in self.suffixes:
                init = os.path.join(basename, '__init__' + suffix)
                if self._open_file(init, mode):
                    return (None, basename, ('', '', _PKG_DIRECTORY))
            if self._dir_exists(basename):
                ns_dirs.append(basename)
        if not path:
            if p3extend_frozen and p3extend_frozen.is_frozen_module(name):
                return (None, name, ('', '', _PY_FROZEN))
        if ns_dirs:
            return (None, ns_dirs, ('', '', _PKG_NAMESPACE_DIRECTORY))
        raise ImportError(name)

    def find_all_submodules(self, m):
        if False:
            i = 10
            return i + 15
        if not m.__path__:
            return
        modules = {}
        for dir in m.__path__:
            try:
                names = self._listdir(dir)
            except OSError:
                self.msg(2, "can't list directory", dir)
                continue
            for name in sorted(names):
                mod = None
                for suff in self.suffixes:
                    n = len(suff)
                    if name[-n:] == suff:
                        mod = name[:-n]
                        break
                if mod and mod != '__init__':
                    modules[mod] = mod
        return modules.keys()