""" Helper functions for the scons file.

"""
from __future__ import print_function
import os
import shutil
import signal
import sys
from nuitka.__past__ import basestring, unicode
from nuitka.containers.OrderedDicts import OrderedDict
from nuitka.Tracing import scons_details_logger, scons_logger
from nuitka.utils.Execution import executeProcess
from nuitka.utils.FileOperations import getFileContentByLine, openTextFile
from nuitka.utils.Utils import isLinux, isMacOS, isPosixWindows

def initScons():
    if False:
        return 10
    os.environ['LANG'] = 'C'

    def no_sync(self):
        if False:
            for i in range(10):
                print('nop')
        pass
    import SCons.dblite
    SCons.dblite.dblite.sync = no_sync

def setupScons(env, source_dir):
    if False:
        return 10
    env['BUILD_DIR'] = source_dir
    sconsign_filename = os.path.abspath(os.path.join(source_dir, '.sconsign-%d%s' % (sys.version_info[0], sys.version_info[1])))
    env.SConsignFile(sconsign_filename)
scons_arguments = {}

def setArguments(arguments):
    if False:
        i = 10
        return i + 15
    'Decode command line arguments.'
    arg_encoding = arguments.get('argument_encoding')
    for (key, value) in arguments.items():
        if arg_encoding is not None:
            value = decodeData(value)
        scons_arguments[key] = value

def getArgumentRequired(name):
    if False:
        print('Hello World!')
    'Helper for string options without default value.'
    return scons_arguments[name]

def getArgumentDefaulted(name, default):
    if False:
        i = 10
        return i + 15
    'Helper for string options with default value.'
    return scons_arguments.get(name, default)

def getArgumentInt(option_name, default=None):
    if False:
        print('Hello World!')
    'Small helper for boolean mode flags.'
    if default is None:
        value = scons_arguments[option_name]
    else:
        value = int(scons_arguments.get(option_name, default))
    return value

def getArgumentBool(option_name, default=None):
    if False:
        while True:
            i = 10
    'Small helper for boolean mode flags.'
    if default is None:
        value = scons_arguments[option_name]
    else:
        value = scons_arguments.get(option_name, 'True' if default else 'False')
    return value.lower() in ('yes', 'true', '1')

def getArgumentList(option_name, default=None):
    if False:
        print('Hello World!')
    'Small helper for list mode options, default should be command separated str.'
    if default is None:
        value = scons_arguments[option_name]
    else:
        value = scons_arguments.get(option_name, default)
    if value:
        return value.split(',')
    else:
        return []

def _enableFlagSettings(env, name, experimental_flags):
    if False:
        while True:
            i = 10
    for flag_name in experimental_flags:
        if not flag_name:
            continue
        flag_name = '%s-%s' % (name, flag_name)
        if '=' in flag_name:
            (flag_name, value) = flag_name.split('=', 1)
        else:
            value = None
        flag_name = flag_name.upper().replace('-', '_').replace('.', '_')
        if value:
            env.Append(CPPDEFINES=[('_NUITKA_%s' % flag_name, value)])
        else:
            env.Append(CPPDEFINES=['_NUITKA_%s' % flag_name])

def prepareEnvironment(mingw_mode):
    if False:
        print('Hello World!')
    if 'CC' in os.environ:
        scons_details_logger.info("CC='%s'" % os.environ['CC'])
        os.environ['CC'] = os.path.normpath(os.path.expanduser(os.environ['CC']))
        if os.path.isdir(os.environ['CC']):
            scons_logger.sysexit("Error, the 'CC' variable must point to file, not directory.")
        if os.path.sep in os.environ['CC']:
            cc_dirname = os.path.dirname(os.environ['CC'])
            if os.path.isdir(cc_dirname):
                addToPATH(None, cc_dirname, prefix=True)
        if os.name == 'nt' and isGccName(os.path.basename(os.environ['CC'])):
            scons_details_logger.info('Environment CC seems to be a gcc, enabling mingw_mode.')
            mingw_mode = True
    else:
        anaconda_python = getArgumentBool('anaconda_python', False)
        if isLinux() and anaconda_python:
            python_prefix = getArgumentRequired('python_prefix')
            addToPATH(None, os.path.join(python_prefix, 'bin'), prefix=True)
    return mingw_mode

def createEnvironment(mingw_mode, msvc_version, target_arch, experimental, no_deployment):
    if False:
        for i in range(10):
            print('nop')
    from SCons.Script import Environment
    args = {}
    if msvc_version == 'list':
        import SCons.Tool.MSCommon.vc
        scons_logger.sysexit('Installed MSVC versions are %s.' % ','.join((repr(v) for v in SCons.Tool.MSCommon.vc.get_installed_vcs())))
    if os.name == 'nt' and (not mingw_mode) and (msvc_version is None) and (msvc_version != 'latest') and (getExecutablePath('cl', env=None) is not None):
        args['MSVC_USE_SCRIPT'] = False
    if mingw_mode or isPosixWindows():
        tools = ['mingw']
        import SCons.Tool.MSCommon.vc
        import SCons.Tool.msvc
        SCons.Tool.MSCommon.vc.msvc_setup_env = lambda *args: None
        SCons.Tool.msvc.msvc_exists = SCons.Tool.MSCommon.vc.msvc_exists = lambda *args: False
    else:
        tools = ['default']
    env = Environment(ENV=os.environ, tools=tools, SHLIBPREFIX='', TARGET_ARCH=target_arch, MSVC_VERSION=msvc_version if msvc_version != 'latest' else None, **args)
    env.nuitka_python = getArgumentBool('nuitka_python', False)
    env.debian_python = getArgumentBool('debian_python', False)
    env.fedora_python = getArgumentBool('fedora_python', False)
    env.msys2_mingw_python = getArgumentBool('msys2_mingw_python', False)
    env.anaconda_python = getArgumentBool('anaconda_python', False)
    env.pyenv_python = getArgumentBool('pyenv_python', False)
    env.apple_python = getArgumentBool('apple_python', False)
    env.noelf_mode = getArgumentBool('noelf_mode', False)
    env.static_libpython = getArgumentDefaulted('static_libpython', '')
    if env.static_libpython:
        assert os.path.exists(env.static_libpython), env.static_libpython
    python_version_str = getArgumentDefaulted('python_version', None)
    if python_version_str is not None:
        env.python_version = tuple((int(d) for d in python_version_str.split('.')))
    else:
        env.python_version = None
    env.module_count = getArgumentInt('module_count', 0)
    env.target_arch = target_arch
    _enableFlagSettings(env, 'no_deployment', no_deployment)
    env.no_deployment_flags = no_deployment
    _enableFlagSettings(env, 'experimental', experimental)
    env.experimental_flags = experimental
    return env

def decodeData(data):
    if False:
        print('Hello World!')
    'Our own decode tries to workaround MSVC misbehavior.'
    try:
        return data.decode(sys.stdout.encoding)
    except UnicodeDecodeError:
        import locale
        try:
            return data.decode(locale.getpreferredencoding())
        except UnicodeDecodeError:
            return data.decode('utf8', 'backslashreplace')

def getExecutablePath(filename, env):
    if False:
        print('Hello World!')
    'Find an execute in either normal PATH, or Scons detected PATH.'
    if os.path.exists(filename):
        return filename
    while filename.startswith('$'):
        filename = env[filename[1:]]
    if os.name == 'nt' and (not filename.lower().endswith('.exe')):
        filename += '.exe'
    if env is None:
        search_path = os.environ['PATH']
    else:
        search_path = env._dict['ENV']['PATH']
    path_elements = search_path.split(os.pathsep)
    for path_element in path_elements:
        path_element = path_element.strip('"')
        full = os.path.normpath(os.path.join(path_element, filename))
        if os.path.exists(full):
            return full
    return None

def changeKeyboardInterruptToErrorExit():
    if False:
        while True:
            i = 10

    def signalHandler(signal, frame):
        if False:
            print('Hello World!')
        sys.exit(2)
    signal.signal(signal.SIGINT, signalHandler)

def setEnvironmentVariable(env, key, value):
    if False:
        i = 10
        return i + 15
    if value is None:
        del os.environ[key]
    elif key in os.environ:
        os.environ[key] = value
    if env is not None:
        if value is None:
            del env._dict['ENV'][key]
        else:
            env._dict['ENV'][key] = value

def addToPATH(env, dirname, prefix):
    if False:
        i = 10
        return i + 15
    if str is bytes and type(dirname) is unicode:
        dirname = dirname.encode('utf8')
    path_value = os.environ['PATH'].split(os.pathsep)
    if prefix:
        path_value.insert(0, dirname)
    else:
        path_value.append(dirname)
    setEnvironmentVariable(env, 'PATH', os.pathsep.join(path_value))

def writeSconsReport(env, source_dir):
    if False:
        i = 10
        return i + 15
    with openTextFile(os.path.join(source_dir, 'scons-report.txt'), 'w', encoding='utf8') as report_file:
        for (key, value) in sorted(env._dict.items()):
            if type(value) is list and all((isinstance(v, basestring) for v in value)):
                value = repr(value)
            if not isinstance(value, basestring):
                continue
            if key.startswith(('_', 'CONFIGURE')):
                continue
            if key in ('MSVSSCONS', 'BUILD_DIR', 'IDLSUFFIXES', 'DSUFFIXES'):
                continue
            print(key + '=' + value, file=report_file)
        print('gcc_mode=%s' % env.gcc_mode, file=report_file)
        print('clang_mode=%s' % env.clang_mode, file=report_file)
        print('msvc_mode=%s' % env.msvc_mode, file=report_file)
        print('mingw_mode=%s' % env.mingw_mode, file=report_file)
        print('clangcl_mode=%s' % env.clangcl_mode, file=report_file)
        print('PATH=%s' % os.environ['PATH'], file=report_file)
_scons_reports = {}

def flushSconsReports():
    if False:
        print('Hello World!')
    _scons_reports.clear()

def readSconsReport(source_dir):
    if False:
        print('Hello World!')
    if source_dir not in _scons_reports:
        scons_report = OrderedDict()
        for line in getFileContentByLine(os.path.join(source_dir, 'scons-report.txt'), encoding='utf8'):
            if '=' not in line:
                continue
            (key, value) = line.strip().split('=', 1)
            scons_report[key] = value
        _scons_reports[source_dir] = scons_report
    return _scons_reports[source_dir]

def getSconsReportValue(source_dir, key):
    if False:
        print('Hello World!')
    return readSconsReport(source_dir).get(key)

def addClangClPathFromMSVC(env):
    if False:
        while True:
            i = 10
    cl_exe = getExecutablePath('cl', env=env)
    if cl_exe is None:
        scons_logger.sysexit('Error, Visual Studio required for using ClangCL on Windows.')
    clang_dir = os.path.join(cl_exe[:cl_exe.lower().rfind('msvc')], 'Llvm')
    if getCompilerArch(mingw_mode=False, msvc_mode=True, the_cc_name='cl.exe', compiler_path=cl_exe) == 'pei-x86-64':
        clang_dir = os.path.join(clang_dir, 'x64', 'bin')
    else:
        clang_dir = os.path.join(clang_dir, 'bin')
    if not os.path.exists(clang_dir):
        scons_details_logger.sysexit("Visual Studio has no Clang component found at '%s'." % clang_dir)
    scons_details_logger.info("Adding Visual Studio directory '%s' for Clang to PATH." % clang_dir)
    addToPATH(env, clang_dir, prefix=True)
    clangcl_path = getExecutablePath('clang-cl', env=env)
    if clangcl_path is None:
        scons_details_logger.sysexit("Visual Studio has no Clang component found at '%s'." % clang_dir)
    env['CC'] = 'clang-cl'
    env['LINK'] = 'lld-link'
    env['CCVERSION'] = None

def isGccName(cc_name):
    if False:
        for i in range(10):
            print('nop')
    return 'gcc' in cc_name or 'g++' in cc_name or 'gnu-cc' in cc_name or ('gnu-gcc' in cc_name)

def isClangName(cc_name):
    if False:
        for i in range(10):
            print('nop')
    return 'clang' in cc_name and '-cl' not in cc_name

def cheapCopyFile(src, dst):
    if False:
        while True:
            i = 10
    dirname = os.path.dirname(dst)
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    if os.name == 'nt':
        if os.path.exists(dst):
            os.unlink(dst)
        shutil.copy(src, dst)
    else:
        src = os.path.abspath(src)
        try:
            link_target = os.readlink(dst)
            if link_target == src:
                return
            os.unlink(dst)
        except OSError as _e:
            try:
                os.unlink(dst)
            except OSError:
                pass
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy(src, dst)

def provideStaticSourceFile(env, sub_path, c11_mode):
    if False:
        i = 10
        return i + 15
    source_filename = os.path.join(env.nuitka_src, 'static_src', sub_path)
    target_filename = os.path.join(env.source_dir, 'static_src', os.path.basename(sub_path))
    if target_filename.endswith('.c') and (not c11_mode):
        target_filename += 'pp'
    cheapCopyFile(source_filename, target_filename)
    return target_filename

def scanSourceDir(env, dirname, plugins):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.exists(dirname):
        return
    added_path = False
    for filename in sorted(os.listdir(dirname)):
        if filename.endswith('.h') and plugins and (not added_path):
            env.Append(CPPPATH=[dirname])
            added_path = True
        if not filename.endswith(('.c', 'cpp')) or not filename.startswith(('module.', '__', 'plugin.')):
            continue
        filename = os.path.join(dirname, filename)
        target_file = filename
        if env.c11_mode:
            yield filename
        else:
            if filename.endswith('.c'):
                target_file += 'pp'
                os.rename(filename, target_file)
            yield target_file

def makeCLiteral(value):
    if False:
        while True:
            i = 10
    value = value.replace('\\', '\\\\')
    value = value.replace('"', '\\"')
    return '"' + value + '"'

def createDefinitionsFile(source_dir, filename, definitions):
    if False:
        for i in range(10):
            print('nop')
    for env_name in os.environ['_NUITKA_BUILD_DEFINITIONS_CATALOG'].split(','):
        definitions[env_name] = os.environ[env_name]
    build_definitions_filename = os.path.join(source_dir, filename)
    with openTextFile(build_definitions_filename, 'w', encoding='utf8') as f:
        for (key, value) in sorted(definitions.items()):
            if key == '_NUITKA_BUILD_DEFINITIONS_CATALOG':
                continue
            if type(value) is int or key.endswith(('_BOOL', '_INT')):
                if type(value) is bool:
                    value = int(value)
                f.write('#define %s %s\n' % (key, value))
            elif type(value) in (str, unicode) and key.endswith('_WIDE_STRING'):
                f.write('#define %s L%s\n' % (key, makeCLiteral(value)))
            else:
                f.write('#define %s %s\n' % (key, makeCLiteral(value)))

def getMsvcVersionString(env):
    if False:
        while True:
            i = 10
    import SCons.Tool.MSCommon.vc
    return SCons.Tool.MSCommon.vc.get_default_version(env)

def getMsvcVersion(env):
    if False:
        while True:
            i = 10
    value = getMsvcVersionString(env)
    if value is None:
        value = os.environ.get('VCToolsVersion', '14.3').rsplit('.', 1)[0]
    value = value.replace('exp', '')
    return tuple((int(d) for d in value.split('.')))

def _getBinaryArch(binary, mingw_mode):
    if False:
        while True:
            i = 10
    if 'linux' in sys.platform or mingw_mode:
        assert os.path.exists(binary), binary
        command = ['objdump', '-f', binary]
        try:
            (data, _err, rv) = executeProcess(command)
        except OSError:
            command[0] = 'llvm-objdump'
            try:
                (data, _err, rv) = executeProcess(command)
            except OSError:
                return None
        if rv != 0:
            return None
        if str is not bytes:
            data = decodeData(data)
        found = None
        for line in data.splitlines():
            if ' file format ' in line:
                found = line.split(' file format ')[-1]
            if '\tfile format ' in line:
                found = line.split('\tfile format ')[-1]
        if os.name == 'nt' and found == 'coff-x86-64':
            found = 'pei-x86-64'
        return found
    else:
        return None
_linker_arch_determined = False
_linker_arch = None

def getLinkerArch(target_arch, mingw_mode):
    if False:
        while True:
            i = 10
    global _linker_arch_determined, _linker_arch
    if not _linker_arch_determined:
        if os.name == 'nt':
            if target_arch == 'x86_64':
                _linker_arch = 'pei-x86-64'
            elif target_arch == 'arm64':
                _linker_arch = 'pei-arm64'
            else:
                _linker_arch = 'pei-i386'
        else:
            _linker_arch = _getBinaryArch(binary=os.environ['NUITKA_PYTHON_EXE_PATH'], mingw_mode=mingw_mode)
        _linker_arch_determined = True
    return _linker_arch
_compiler_arch = {}

def getCompilerArch(mingw_mode, msvc_mode, the_cc_name, compiler_path):
    if False:
        i = 10
        return i + 15
    assert not mingw_mode or not msvc_mode
    if compiler_path not in _compiler_arch:
        if mingw_mode:
            _compiler_arch[compiler_path] = _getBinaryArch(binary=compiler_path, mingw_mode=mingw_mode)
        elif msvc_mode:
            cmdline = [compiler_path]
            if '-cl' in the_cc_name:
                cmdline.append('--version')
            (stdout, stderr, _rv) = executeProcess(command=cmdline)
            if b'x64' in stderr or b'x86_64' in stdout:
                _compiler_arch[compiler_path] = 'pei-x86-64'
            elif b'x86' in stderr or b'i686' in stdout:
                _compiler_arch[compiler_path] = 'pei-i386'
            elif b'ARM64' in stderr:
                _compiler_arch[compiler_path] = 'pei-arm64'
            else:
                assert False, (stdout, stderr)
        else:
            assert False, compiler_path
    return _compiler_arch[compiler_path]

def decideArchMismatch(target_arch, the_cc_name, compiler_path):
    if False:
        for i in range(10):
            print('nop')
    mingw_mode = isGccName(the_cc_name) or isClangName(the_cc_name)
    msvc_mode = not mingw_mode
    linker_arch = getLinkerArch(target_arch=target_arch, mingw_mode=mingw_mode)
    compiler_arch = getCompilerArch(mingw_mode=mingw_mode, msvc_mode=msvc_mode, the_cc_name=the_cc_name, compiler_path=compiler_path)
    return (linker_arch != compiler_arch, linker_arch, compiler_arch)

def raiseNoCompilerFoundErrorExit():
    if False:
        print('Hello World!')
    if os.name == 'nt':
        scons_logger.sysexit('Error, cannot locate suitable C compiler. You have the following options:\n\na) If a suitable Visual Studio version is installed (check above trace\n   outputs for rejection messages), it will be located automatically via\n   registry. But not if you activate the wrong prompt.\n\nb) Using "--mingw64" lets Nuitka download MinGW64 for you. Note: MinGW64\n   is the project name, it does *not* mean 64 bits, just a gcc with better\n   Windows compatibility, it is available for 32 and 64 bits. Cygwin based\n   gcc e.g. do not work.\n')
    else:
        scons_logger.sysexit('Error, cannot locate suitable C compiler.')

def addBinaryBlobSection(env, blob_filename, section_name):
    if False:
        for i in range(10):
            print('nop')
    if isMacOS():
        env.Append(LINKFLAGS=['-Wl,-sectcreate,%(section_name)s,%(section_name)s,%(blob_filename)s' % {'section_name': section_name, 'blob_filename': blob_filename}])
    else:
        assert False