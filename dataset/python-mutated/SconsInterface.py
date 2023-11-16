""" Scons interface.

Interaction with scons. Find the binary, and run it with a set of given
options.

"""
import contextlib
import copy
import os
import subprocess
import sys
from nuitka import Options, Tracing
from nuitka.__past__ import unicode
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.Options import getOnefileChildGraceTime, isOnefileMode
from nuitka.plugins.Plugins import Plugins
from nuitka.PythonFlavors import isAnacondaPython, isMSYS2MingwPython, isNuitkaPython
from nuitka.PythonVersions import getSystemPrefixPath, getTargetPythonDLLPath, python_version
from nuitka.utils.Execution import getExecutablePath, withEnvironmentVarsOverridden
from nuitka.utils.FileOperations import deleteFile, getDirectoryRealPath, getExternalUsePath, getWindowsShortPathName, hasFilenameExtension, listDir, withDirectoryChange
from nuitka.utils.InstalledPythons import findInstalledPython
from nuitka.utils.SharedLibraries import detectBinaryMinMacOS
from nuitka.utils.Utils import getArchitecture, isMacOS, isWin32OrPosixWindows, isWin32Windows
from .SconsCaching import checkCachingSuccess
from .SconsUtils import flushSconsReports

def getSconsDataPath():
    if False:
        print('Hello World!')
    'Return path to where data for scons lives, e.g. static C source files.'
    return os.path.dirname(__file__)

def _getSconsInlinePath():
    if False:
        for i in range(10):
            print('nop')
    'Return path to inline copy of scons.'
    return os.path.join(getSconsDataPath(), 'inline_copy')

def _getSconsBinaryCall():
    if False:
        for i in range(10):
            print('nop')
    'Return a way to execute Scons.\n\n    Using potentially in-line copy if no system Scons is available\n    or if we are on Windows, there it is mandatory.\n    '
    inline_path = os.path.join(_getSconsInlinePath(), 'bin', 'scons.py')
    if os.path.exists(inline_path):
        return [_getPythonForSconsExePath(), '-W', 'ignore', getExternalUsePath(inline_path)]
    else:
        scons_path = getExecutablePath('scons')
        if scons_path is not None:
            return [scons_path]
        else:
            Tracing.scons_logger.sysexit('Error, the inline copy of scons is not present, nor a scons binary in the PATH.')

def _getPythonForSconsExePath():
    if False:
        for i in range(10):
            print('nop')
    "Find a way to call any Python that works for Scons.\n\n    Scons needs it as it doesn't support all Python versions.\n    "
    python_exe = Options.getPythonPathForScons()
    if python_exe is not None:
        return python_exe
    scons_supported_pythons = ('3.5', '3.6', '3.7', '3.8', '3.9', '3.10', '3.11')
    if not isWin32Windows():
        scons_supported_pythons += ('2.7', '2.6')
    python_for_scons = findInstalledPython(python_versions=scons_supported_pythons, module_name=None, module_version=None)
    if python_for_scons is None:
        if isWin32Windows():
            scons_python_requirement = 'Python 3.5 or higher'
        else:
            scons_python_requirement = 'Python 2.6, 2.7 or Python >= 3.5'
        Tracing.scons_logger.sysexit('Error, while Nuitka works with older Python, Scons does not, and therefore\nNuitka needs to find a %s executable, so please install\nit.\n\nYou may provide it using option "--python-for-scons=path_to_python.exe"\nin case it is not visible in registry, e.g. due to using uninstalled\nAnaconda Python.\n' % scons_python_requirement)
    return python_for_scons.getPythonExe()

@contextlib.contextmanager
def _setupSconsEnvironment2(scons_filename):
    if False:
        i = 10
        return i + 15
    is_backend = scons_filename == 'Backend.scons'
    if is_backend and isWin32Windows() and (not Options.shallUseStaticLibPython()):
        os.environ['NUITKA_PYTHON_DLL_PATH'] = getTargetPythonDLLPath()
    os.environ['NUITKA_PYTHON_EXE_PATH'] = sys.executable
    old_pythonpath = None
    old_pythonhome = None
    if python_version >= 768:
        if 'PYTHONPATH' in os.environ:
            old_pythonpath = os.environ['PYTHONPATH']
            del os.environ['PYTHONPATH']
        if 'PYTHONHOME' in os.environ:
            old_pythonhome = os.environ['PYTHONHOME']
            del os.environ['PYTHONHOME']
    import nuitka
    os.environ['NUITKA_PACKAGE_DIR'] = os.path.abspath(nuitka.__path__[0])
    yield
    if old_pythonpath is not None:
        os.environ['PYTHONPATH'] = old_pythonpath
    if old_pythonhome is not None:
        os.environ['PYTHONHOME'] = old_pythonhome
    if 'NUITKA_PYTHON_DLL_PATH' in os.environ:
        del os.environ['NUITKA_PYTHON_DLL_PATH']
    del os.environ['NUITKA_PYTHON_EXE_PATH']
    del os.environ['NUITKA_PACKAGE_DIR']

@contextlib.contextmanager
def _setupSconsEnvironment(scons_filename):
    if False:
        i = 10
        return i + 15
    'Setup the scons execution environment.\n\n    For the target Python we provide "NUITKA_PYTHON_DLL_PATH" to see where the\n    Python DLL lives, in case it needs to be copied, and then also the\n    "NUITKA_PYTHON_EXE_PATH" to find the Python binary itself.\n\n    We also need to preserve "PYTHONPATH" and "PYTHONHOME", but remove it\n    potentially as well, so not to confuse the other Python binary used to run\n    scons.\n    '
    if isWin32Windows():
        change_dir = getWindowsShortPathName(os.getcwd())
    else:
        change_dir = None
    with withDirectoryChange(change_dir, allow_none=True):
        with _setupSconsEnvironment2(scons_filename=scons_filename):
            yield

def _buildSconsCommand(options, scons_filename):
    if False:
        for i in range(10):
            print('nop')
    'Build the scons command to run.\n\n    The options are a dictionary to be passed to scons as a command line,\n    and other scons stuff is set.\n    '
    scons_command = _getSconsBinaryCall()
    if not Options.isShowScons():
        scons_command.append('--quiet')
    scons_command += ['-f', getExternalUsePath(os.path.join(getSconsDataPath(), scons_filename)), '--jobs', str(Options.getJobLimit()), '--warn=no-deprecated', '--no-site-dir']
    if Options.isShowScons():
        scons_command.append('--debug=stacktrace')

    def encode(value):
        if False:
            return 10
        if str is bytes and type(value) is unicode:
            return value.encode('utf8')
        else:
            return value
    for (key, value) in options.items():
        if value is None:
            Tracing.scons_logger.sysexit("Error, failure to provide argument for '%s', please report bug." % key)
        scons_command.append(key + '=' + encode(value))
    if str is bytes:
        scons_command.append('arg_encoding=utf8')
    return scons_command

def runScons(options, env_values, scons_filename):
    if False:
        return 10
    with _setupSconsEnvironment(scons_filename):
        env_values['_NUITKA_BUILD_DEFINITIONS_CATALOG'] = ','.join(env_values.keys())
        if 'source_dir' in options and Options.shallCompileWithoutBuildDirectory():
            options = copy.deepcopy(options)
            source_dir = options['source_dir']
            options['source_dir'] = '.'
            options['result_name'] = getExternalUsePath(options['result_name'], only_dirname=True)
            options['nuitka_src'] = getExternalUsePath(options['nuitka_src'])
            if 'result_exe' in options:
                options['result_exe'] = getExternalUsePath(options['result_exe'], only_dirname=True)
        else:
            source_dir = None
        env_values = OrderedDict(env_values)
        env_values['_NUITKA_BUILD_DEFINITIONS_CATALOG'] = ','.join(env_values.keys())
        env_values['NUITKA_QUIET'] = '1' if Tracing.is_quiet else '0'
        scons_command = _buildSconsCommand(options=options, scons_filename=scons_filename)
        if Options.isShowScons():
            Tracing.scons_logger.info('Scons command: %s' % ' '.join(scons_command))
        Tracing.flushStandardOutputs()
        with withEnvironmentVarsOverridden(env_values):
            try:
                result = subprocess.call(scons_command, shell=False, cwd=source_dir)
            except KeyboardInterrupt:
                Tracing.scons_logger.sysexit('User interrupted scons build.')
        flushSconsReports()
        if 'source_dir' in options and result == 0:
            checkCachingSuccess(source_dir or options['source_dir'])
        return result == 0

def asBoolStr(value):
    if False:
        print('Hello World!')
    'Encode booleans for transfer via command line.'
    return 'true' if value else 'false'

def cleanSconsDirectory(source_dir):
    if False:
        while True:
            i = 10
    'Clean scons build directory.'
    extensions = ('.bin', '.c', '.cpp', '.exp', '.h', '.lib', '.manifest', '.o', '.obj', '.os', '.rc', '.res', '.S', '.txt', '.const', '.gcda', '.pgd', '.pgc')

    def check(path):
        if False:
            return 10
        if hasFilenameExtension(path, extensions):
            deleteFile(path, must_exist=True)
    if os.path.isdir(source_dir):
        for (path, _filename) in listDir(source_dir):
            check(path)
        static_dir = os.path.join(source_dir, 'static_src')
        if os.path.exists(static_dir):
            for (path, _filename) in listDir(static_dir):
                check(path)
        plugins_dir = os.path.join(source_dir, 'plugins')
        if os.path.exists(plugins_dir):
            for (path, _filename) in listDir(plugins_dir):
                check(path)

def setCommonSconsOptions(options):
    if False:
        i = 10
        return i + 15
    options['nuitka_src'] = getSconsDataPath()
    options['python_prefix'] = getDirectoryRealPath(getSystemPrefixPath())
    options['experimental'] = ','.join(Options.getExperimentalIndications())
    options['no_deployment'] = ','.join(Options.getNoDeploymentIndications())
    if Options.shallRunInDebugger():
        options['full_names'] = asBoolStr(True)
    if Options.assumeYesForDownloads():
        options['assume_yes_for_downloads'] = asBoolStr(True)
    if not Options.shallUseProgressBar():
        options['progress_bar'] = asBoolStr(False)
    if Options.isClang():
        options['clang_mode'] = asBoolStr(True)
    if Options.isShowScons():
        options['show_scons'] = asBoolStr(True)
    if Options.isMingw64():
        options['mingw_mode'] = asBoolStr(True)
    if Options.getMsvcVersion():
        options['msvc_version'] = Options.getMsvcVersion()
    if Options.shallDisableCCacheUsage():
        options['disable_ccache'] = asBoolStr(True)
    if Options.shallDisableConsoleWindow() and Options.mayDisableConsoleWindow():
        options['disable_console'] = asBoolStr(True)
    if Options.getLtoMode() != 'auto':
        options['lto_mode'] = Options.getLtoMode()
    if isWin32OrPosixWindows() or isMacOS():
        options['noelf_mode'] = asBoolStr(True)
    if Options.isUnstripped():
        options['unstripped_mode'] = asBoolStr(True)
    if isAnacondaPython():
        options['anaconda_python'] = asBoolStr(True)
    if isMSYS2MingwPython():
        options['msys2_mingw_python'] = asBoolStr(True)
    cpp_defines = Plugins.getPreprocessorSymbols()
    if cpp_defines:
        options['cpp_defines'] = ','.join(('%s%s%s' % (key, '=' if value else '', value or '') for (key, value) in cpp_defines.items()))
    cpp_include_dirs = Plugins.getExtraIncludeDirectories()
    if cpp_include_dirs:
        options['cpp_include_dirs'] = ','.join(cpp_include_dirs)
    link_dirs = Plugins.getExtraLinkDirectories()
    if link_dirs:
        options['link_dirs'] = ','.join(link_dirs)
    link_libraries = Plugins.getExtraLinkLibraries()
    if link_libraries:
        options['link_libraries'] = ','.join(link_libraries)
    if isMacOS():
        macos_min_version = detectBinaryMinMacOS(sys.executable)
        if macos_min_version is None:
            Tracing.general.sysexit("Could not detect minimum macOS version for '%s'." % sys.executable)
        options['macos_min_version'] = macos_min_version
        macos_target_arch = Options.getMacOSTargetArch()
        if macos_target_arch == 'universal':
            Tracing.general.sysexit('Cannot create universal macOS binaries (yet), please pick an arch and create two binaries.')
        options['macos_target_arch'] = macos_target_arch
    options['target_arch'] = getArchitecture()
    env_values = OrderedDict()
    string_values = Options.getWindowsVersionInfoStrings()
    if 'CompanyName' in string_values:
        env_values['NUITKA_COMPANY_NAME'] = string_values['CompanyName']
    if 'ProductName' in string_values:
        env_values['NUITKA_PRODUCT_NAME'] = string_values['ProductName']
    product_version = Options.getProductVersion()
    file_version = Options.getFileVersion()
    if product_version is None:
        product_version = file_version
    if product_version is not None:
        product_version = '.'.join((str(d) for d in product_version))
    if file_version is None:
        file_version = product_version
    else:
        file_version = '.'.join((str(d) for d in file_version))
    if product_version != file_version:
        effective_version = '%s-%s' % (product_version, file_version)
    else:
        effective_version = file_version
    if effective_version:
        env_values['NUITKA_VERSION_COMBINED'] = effective_version
    if isNuitkaPython() and (not isWin32OrPosixWindows()):
        import sysconfig
        env_values['CC'] = sysconfig.get_config_var('CC').split()[0]
        env_values['CXX'] = sysconfig.get_config_var('CXX').split()[0]
    if isOnefileMode():
        env_values['_NUITKA_ONEFILE_CHILD_GRACE_TIME_INT'] = str(getOnefileChildGraceTime())
    return env_values