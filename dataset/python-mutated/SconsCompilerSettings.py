""" This contains the tuning of the compilers towards defined goals.

"""
import os
import re
from nuitka.Tracing import scons_details_logger, scons_logger
from nuitka.utils.Download import getCachedDownloadedMinGW64
from nuitka.utils.FileOperations import getReportPath, openTextFile, putTextFileContents
from nuitka.utils.Utils import isFedoraBasedLinux, isMacOS, isPosixWindows, isWin32Windows
from .SconsHacks import myDetectVersion
from .SconsUtils import addBinaryBlobSection, addToPATH, createEnvironment, decideArchMismatch, getExecutablePath, getLinkerArch, getMsvcVersion, getMsvcVersionString, isClangName, isGccName, raiseNoCompilerFoundErrorExit, setEnvironmentVariable

def _detectWindowsSDK(env):
    if False:
        return 10
    if hasattr(env, 'windows_sdk_version'):
        return env.windows_sdk_version
    if 'WindowsSDKVersion' not in env:
        if 'WindowsSDKVersion' in os.environ:
            windows_sdk_version = os.environ['WindowsSDKVersion'].rstrip('\\')
        else:
            windows_sdk_version = None
    else:
        windows_sdk_version = env['WindowsSDKVersion']
    if windows_sdk_version:
        scons_details_logger.info("Using Windows SDK '%s'." % windows_sdk_version)
        env.windows_sdk_version = tuple((int(x) for x in windows_sdk_version.split('.')))
    else:
        scons_logger.warning('Windows SDK must be installed in Visual Studio for it to be usable with Nuitka. Use the Visual Studio installer for adding it.')
        env.windows_sdk_version = None
    return env.windows_sdk_version
_windows_sdk_c11_mode_min_version = (10, 0, 19041, 0)

def _enableC11Settings(env):
    if False:
        while True:
            i = 10
    'Decide if C11 mode can be used and enable the C compile flags for it.\n\n    Args:\n        env - scons environment with compiler information\n\n    Returns:\n        bool - c11_mode flag\n    '
    if env.clangcl_mode:
        c11_mode = True
    elif env.msvc_mode and env.windows_sdk_version >= _windows_sdk_c11_mode_min_version and (getMsvcVersion(env) >= (14, 3)):
        c11_mode = True
    elif env.clang_mode:
        c11_mode = True
    elif env.gcc_mode and env.gcc_version >= (5,):
        c11_mode = True
    else:
        c11_mode = False
    if c11_mode:
        if env.gcc_mode:
            env.Append(CCFLAGS=['-std=c11'])
        elif env.msvc_mode:
            env.Append(CCFLAGS=['/std:c11'])
    if env.msvc_mode and c11_mode:
        env.Append(CCFLAGS=['/wd5105'])
    if not c11_mode:
        env.Append(CPPDEFINES=['_NUITKA_NON_C11_MODE'])
    scons_details_logger.info('Using C11 mode: %s' % c11_mode)
    env.c11_mode = c11_mode

def _enableLtoSettings(env, lto_mode, pgo_mode, job_count):
    if False:
        print('Hello World!')
    orig_lto_mode = lto_mode
    if lto_mode == 'no':
        lto_mode = False
        reason = 'disabled'
    elif lto_mode == 'yes':
        lto_mode = True
        reason = 'enabled'
    elif pgo_mode in ('use', 'generate'):
        lto_mode = True
        reason = 'PGO implies LTO'
    elif env.msvc_mode and getMsvcVersion(env) >= (14,):
        lto_mode = True
        reason = 'known to be supported'
    elif env.nuitka_python:
        lto_mode = True
        reason = 'known to be supported (Nuitka-Python)'
    elif env.fedora_python:
        lto_mode = True
        reason = 'known to be supported (Fedora Python)'
    elif env.debian_python and env.gcc_mode and (not env.clang_mode) and (env.gcc_version >= (6,)):
        lto_mode = True
        reason = 'known to be supported (Debian)'
    elif env.gcc_mode and 'gnu-cc' in env.the_cc_name and env.anaconda_python:
        lto_mode = False
        reason = 'known to be not supported (CondaCC)'
    elif isMacOS() and env.gcc_mode and env.clang_mode:
        lto_mode = True
        reason = 'known to be supported (macOS clang)'
    elif env.mingw_mode and env.clang_mode:
        lto_mode = False
        reason = 'known to not be supported (new MinGW64 Clang)'
    elif env.gcc_mode and env.mingw_mode and (env.gcc_version >= (11, 2)):
        lto_mode = True
        reason = 'known to be supported (new MinGW64)'
    else:
        lto_mode = False
        reason = 'not known to be supported'
    module_count_threshold = 250
    if orig_lto_mode == 'auto' and lto_mode and (env.module_count > module_count_threshold):
        lto_mode = False
        reason = 'might to be too slow %s (>= %d threshold), force with --lto=yes' % (env.module_count, module_count_threshold)
    if lto_mode and env.gcc_mode and (not env.clang_mode) and (env.gcc_version < (4, 6)):
        scons_logger.warning("The gcc compiler %s (version %s) doesn't have the sufficient version for lto mode (>= 4.6). Disabled." % (env['CXX'], env['CXXVERSION']))
        lto_mode = False
        reason = "gcc 4.6 is doesn't have good enough LTO support"
    if env.gcc_mode and lto_mode:
        if env.clang_mode:
            env.Append(CCFLAGS=['-flto'])
            env.Append(LINKFLAGS=['-flto'])
        else:
            env.Append(CCFLAGS=['-flto=%d' % job_count])
            env.Append(LINKFLAGS=['-flto=%d' % job_count])
            env.Append(CCFLAGS=['-fuse-linker-plugin', '-fno-fat-lto-objects'])
            env.Append(LINKFLAGS=['-fuse-linker-plugin'])
            env.Append(LINKFLAGS=['-fpartial-inlining', '-freorder-functions'])
            if env.mingw_mode and 'MAKE' not in os.environ:
                setEnvironmentVariable(env, 'MAKE', 'mingw32-make.exe')
    if env.msvc_mode and lto_mode:
        env.Append(CCFLAGS=['/GL'])
        if not env.clangcl_mode:
            env.Append(LINKFLAGS=['/LTCG'])
            if getMsvcVersion(env) >= (14, 3):
                env.Append(LINKFLAGS=['/CGTHREADS:%d' % job_count])
    if orig_lto_mode == 'auto':
        scons_details_logger.info("LTO mode auto was resolved to mode: '%s' (%s)." % ('yes' if lto_mode else 'no', reason))
    env.lto_mode = lto_mode
    env.orig_lto_mode = orig_lto_mode
    _enablePgoSettings(env, pgo_mode)
_python311_min_msvc_version = (14, 3)

def checkWindowsCompilerFound(env, target_arch, clang_mode, msvc_version, assume_yes_for_downloads):
    if False:
        i = 10
        return i + 15
    'Remove compiler of wrong arch or too old gcc and replace with downloaded winlibs gcc.'
    if os.name == 'nt':
        compiler_path = getExecutablePath(env['CC'], env=env)
        scons_details_logger.info("Checking usability of '%s' from '%s'." % (compiler_path, env['CC']))
        if env.msys2_mingw_python and compiler_path.endswith('/usr/bin/gcc.exe'):
            compiler_path = None
        if compiler_path is not None:
            the_cc_name = os.path.basename(compiler_path)
            if not isGccName(the_cc_name) and (not isClangName(the_cc_name)) and (_detectWindowsSDK(env) is None or (env.python_version is not None and env.python_version >= (3, 11) and (_detectWindowsSDK(env) < _windows_sdk_c11_mode_min_version))):
                compiler_path = None
                env['CC'] = None
        if compiler_path is not None:
            the_cc_name = os.path.basename(compiler_path)
            (decision, linker_arch, compiler_arch) = decideArchMismatch(target_arch=target_arch, the_cc_name=the_cc_name, compiler_path=compiler_path)
            if decision:
                scons_logger.info("Mismatch between Python binary ('%s' -> '%s') and C compiler ('%s' -> '%s') arches, that compiler is ignored!" % (os.environ['NUITKA_PYTHON_EXE_PATH'], linker_arch, compiler_path, compiler_arch))
                compiler_path = None
                env['CC'] = None
        if compiler_path is not None and msvc_version is not None:
            if msvc_version == 'latest':
                scons_logger.info('MSVC version resolved to %s.' % getMsvcVersionString(env))
            elif msvc_version != getMsvcVersionString(env):
                scons_logger.info("Failed to find requested MSVC version ('%s' != '%s')." % (msvc_version, getMsvcVersionString(env)))
                compiler_path = None
                env['CC'] = None
        if compiler_path is not None:
            the_cc_name = os.path.basename(compiler_path)
            if not isGccName(the_cc_name) and None is not env.python_version >= (3, 11) and (getMsvcVersion(env) < _python311_min_msvc_version):
                scons_logger.info('For Python version %s MSVC %s or later is required, not %s which is too old.' % ('.'.join((str(d) for d in env.python_version)), '.'.join((str(d) for d in _python311_min_msvc_version)), getMsvcVersionString(env)))
                compiler_path = None
                env['CC'] = None
        if compiler_path is not None:
            the_cc_name = os.path.basename(compiler_path)
            if isGccName(the_cc_name):
                gcc_version = myDetectVersion(env, compiler_path)
                min_version = (11, 2)
                if gcc_version is not None and (gcc_version < min_version or 'force-winlibs-gcc' in env.experimental_flags):
                    scons_logger.info("Too old gcc '%s' (%r < %r) ignored!" % (compiler_path, gcc_version, min_version))
                    compiler_path = None
                    env['CC'] = None
        if compiler_path is None and msvc_version is None:
            scons_details_logger.info('No usable C compiler, attempt fallback to winlibs gcc.')
            compiler_path = getCachedDownloadedMinGW64(target_arch=target_arch, assume_yes_for_downloads=assume_yes_for_downloads)
            if compiler_path is not None:
                addToPATH(env, os.path.dirname(compiler_path), prefix=True)
                env = createEnvironment(mingw_mode=True, msvc_version=None, target_arch=target_arch, experimental=env.experimental_flags, no_deployment=env.no_deployment_flags)
                if clang_mode:
                    env['CC'] = os.path.join(os.path.dirname(compiler_path), 'clang.exe')
        if env['CC'] is None:
            raiseNoCompilerFoundErrorExit()
    return env

def decideConstantsBlobResourceMode(env, module_mode):
    if False:
        while True:
            i = 10
    if 'NUITKA_RESOURCE_MODE' in os.environ:
        resource_mode = os.environ['NUITKA_RESOURCE_MODE']
        reason = 'user provided'
    elif isWin32Windows():
        resource_mode = 'win_resource'
        reason = 'default for Windows'
    elif isPosixWindows():
        resource_mode = 'linker'
        reason = 'default MSYS2 Posix'
    elif isMacOS():
        resource_mode = 'mac_section'
        reason = 'default for macOS'
    elif env.lto_mode and env.gcc_mode and (not env.clang_mode):
        if module_mode:
            resource_mode = 'code'
        else:
            resource_mode = 'linker'
        reason = 'default for lto gcc with --lto bugs for incbin'
    else:
        resource_mode = 'incbin'
        reason = 'default'
    return (resource_mode, reason)

def addConstantBlobFile(env, blob_filename, resource_desc, target_arch):
    if False:
        print('Hello World!')
    (resource_mode, reason) = resource_desc
    assert blob_filename.endswith('.bin'), blob_filename
    scons_details_logger.info("Using resource mode: '%s' (%s)." % (resource_mode, reason))
    if resource_mode == 'win_resource':
        env.Append(CPPDEFINES=['_NUITKA_CONSTANTS_FROM_RESOURCE'])
    elif resource_mode == 'mac_section':
        env.Append(CPPDEFINES=['_NUITKA_CONSTANTS_FROM_MACOS_SECTION'])
        addBinaryBlobSection(env=env, blob_filename=blob_filename, section_name=os.path.basename(blob_filename)[:-4].lstrip('_'))
    elif resource_mode == 'incbin':
        env.Append(CPPDEFINES=['_NUITKA_CONSTANTS_FROM_INCBIN'])
        constants_generated_filename = os.path.join(env.source_dir, '__constants_data.c')
        putTextFileContents(constants_generated_filename, contents='\n#define INCBIN_PREFIX\n#define INCBIN_STYLE INCBIN_STYLE_SNAKE\n#define INCBIN_LOCAL\n#ifdef _NUITKA_EXPERIMENTAL_WRITEABLE_CONSTANTS\n#define INCBIN_OUTPUT_SECTION ".data"\n#endif\n\n#include "nuitka/incbin.h"\n\nINCBIN(constant_bin, "%(blob_filename)s");\n\nunsigned char const *getConstantsBlobData(void) {\n    return constant_bin_data;\n}\n' % {'blob_filename': blob_filename})
    elif resource_mode == 'linker':
        env.Append(CPPDEFINES=['_NUITKA_CONSTANTS_FROM_LINKER'])
        constant_bin_link_name = 'constant_bin_data'
        if env.mingw_mode:
            constant_bin_link_name = '_' + constant_bin_link_name
        env.Append(LINKFLAGS=['-Wl,-b', '-Wl,binary', '-Wl,%s' % blob_filename, '-Wl,-b', '-Wl,%s' % getLinkerArch(target_arch=target_arch, mingw_mode=env.mingw_mode or isPosixWindows()), '-Wl,-defsym', '-Wl,%s=_binary_%s___constants_bin_start' % (constant_bin_link_name, ''.join((re.sub('[^a-zA-Z0-9_]', '_', c) for c in env.source_dir)))])
    elif resource_mode == 'code':
        env.Append(CPPDEFINES=['_NUITKA_CONSTANTS_FROM_CODE'])
        constants_generated_filename = os.path.join(env.source_dir, '__constants_data.c')

        def writeConstantsDataSource():
            if False:
                while True:
                    i = 10
            with openTextFile(constants_generated_filename, 'w') as output:
                if not env.c11_mode:
                    output.write('extern "C" {')
                output.write('\n// Constant data for the program.\n#if !defined(_NUITKA_EXPERIMENTAL_WRITEABLE_CONSTANTS)\nconst\n#endif\nunsigned char constant_bin_data[] =\n{\n\n')
                with open(blob_filename, 'rb') as f:
                    content = f.read()
                for (count, stream_byte) in enumerate(content):
                    if count % 16 == 0:
                        if count > 0:
                            output.write('\n')
                        output.write('   ')
                    if str is bytes:
                        stream_byte = ord(stream_byte)
                    output.write(' 0x%02x,' % stream_byte)
                output.write('\n};\n')
                if not env.c11_mode:
                    output.write('}')
        writeConstantsDataSource()
    else:
        scons_logger.sysexit("Error, illegal resource mode '%s' specified" % resource_mode)

def enableWindowsStackSize(env, target_arch):
    if False:
        print('Hello World!')
    if target_arch == 'x86_64':
        stack_size = 1024 * 1204 * 8
    else:
        stack_size = 1024 * 1204 * 4
    if env.msvc_mode:
        env.Append(LINKFLAGS=['/STACK:%d' % stack_size])
    if env.mingw_mode:
        env.Append(LINKFLAGS=['-Wl,--stack,%d' % stack_size])

def setupCCompiler(env, lto_mode, pgo_mode, job_count, onefile_compile):
    if False:
        for i in range(10):
            print('nop')
    _enableLtoSettings(env=env, lto_mode=lto_mode, pgo_mode=pgo_mode, job_count=job_count)
    _enableC11Settings(env)
    if env.gcc_mode:
        env.Append(CCFLAGS=['-fvisibility=hidden'])
        if not env.c11_mode:
            env.Append(CXXFLAGS=['-fvisibility-inlines-hidden'])
        if isWin32Windows() and hasattr(env, 'source_dir'):
            env.Append(LINKFLAGS=['-Wl,--exclude-all-symbols'])
            env.Append(LINKFLAGS=['-Wl,--out-implib,%s' % os.path.join(env.source_dir, 'import.lib')])
        env.Append(CCFLAGS=['-fwrapv'])
        if not env.low_memory:
            env.Append(CCFLAGS='-pipe')
    if 'clang' in env.the_cc_name:
        env.Append(CCFLAGS=['-w'])
        env.Append(CPPDEFINES=['_XOPEN_SOURCE'])
        env.Append(CCFLAGS=['-fvisibility=hidden', '-fvisibility-inlines-hidden'])
        if env.debug_mode:
            env.Append(CCFLAGS=['-Wunused-but-set-variable'])
    if isMacOS():
        setEnvironmentVariable(env, 'MACOSX_DEPLOYMENT_TARGET', env.macos_min_version)
        target_flag = '--target=%s-apple-macos%s' % (env.macos_target_arch, env.macos_min_version)
        env.Append(CCFLAGS=[target_flag])
        env.Append(LINKFLAGS=[target_flag])
    if env.mingw_mode:
        env.Append(CPPDEFINES=['_WIN32_WINNT=0x0501'])
    if env.mingw_mode:
        env.Append(LINKFLAGS=['-municode'])
    if env.gcc_version is None and env.gcc_mode and (not env.clang_mode):
        env.gcc_version = myDetectVersion(env, env.the_compiler)
    if env.gcc_mode and (not env.clang_mode) and (env.gcc_version < (4, 5)):
        env.Append(CCFLAGS=['-fno-strict-aliasing'])
    if env.gcc_mode and (not env.clang_mode) and (env.gcc_version >= (4, 6)):
        env.Append(CCFLAGS=['-fpartial-inlining'])
        if env.debug_mode:
            env.Append(CCFLAGS=['-Wunused-but-set-variable'])
    if not env.debug_mode and env.gcc_mode and (not env.clang_mode) and (env.gcc_version >= (5,)):
        env.Append(CCFLAGS=['-ftrack-macro-expansion=0'])
    if env.gcc_mode and (not env.clang_mode):
        env.Append(CCFLAGS=['-Wno-deprecated-declarations'])
    if env.gcc_mode and (not env.clang_mode):
        env.Append(CCFLAGS=['-fno-var-tracking'])
    if env.gcc_mode and (not env.clang_mode) and (env.gcc_version >= (6,)):
        env.Append(CCFLAGS=['-Wno-misleading-indentation'])
    if env.gcc_mode and (not env.clang_mode):
        env.Append(CCFLAGS=['-fcompare-debug-second'])
    if env.gcc_mode and (not env.clang_mode) and env.static_libpython and (not env.lto_mode):
        env.Append(CCFLAGS=['-fno-lto'])
        env.Append(LINKFLAGS=['-fno-lto'])
    if env.gcc_mode and env.lto_mode:
        if env.debug_mode:
            env.Append(LINKFLAGS=['-Og'])
        else:
            env.Append(LINKFLAGS=['-O3' if env.nuitka_python or os.name == 'nt' or (not env.static_libpython) else '-O2'])
    if env.debug_mode:
        if env.clang_mode or (env.gcc_mode and env.gcc_version >= (4, 8)):
            env.Append(CCFLAGS=['-Og'])
        elif env.gcc_mode:
            env.Append(CCFLAGS=['-O1'])
        elif env.msvc_mode:
            env.Append(CCFLAGS=['-O2'])
    else:
        if env.gcc_mode:
            env.Append(CCFLAGS=['-O3' if env.nuitka_python or os.name == 'nt' or (not env.static_libpython) else '-O2'])
        elif env.msvc_mode:
            env.Append(CCFLAGS=['/Ox', '/GF', '/Gy'])
        env.Append(CPPDEFINES=['__NUITKA_NO_ASSERT__'])
    _enableDebugSystemSettings(env, job_count=job_count)
    if env.gcc_mode and (not env.noelf_mode):
        env.Append(LINKFLAGS=['-z', 'noexecstack'])
    if env.mingw_mode:
        if not env.clang_mode:
            env.Append(LINKFLAGS=['-Wl,--enable-auto-import'])
        if env.disable_console:
            env.Append(LINKFLAGS=['-Wl,--subsystem,windows'])
    if env.mingw_mode or env.msvc_mode:
        if env.disable_console:
            env.Append(CPPDEFINES=['_NUITKA_WINMAIN_ENTRY_POINT'])
    if env.mingw_mode and (not env.clang_mode):
        env.Append(LINKFLAGS=['-static-libgcc'])
    if env.mingw_mode and env.target_arch == 'x86_64' and (env.python_version < (3, 12)):
        env.Append(CPPDEFINES=['MS_WIN64'])
    if env.msvc_mode and env.target_arch != 'arm64':
        env.Append(LIBS=['Shell32'])
    if isFedoraBasedLinux():
        env.Append(CCFLAGS=['-fPIC'])
    zlib_inline_copy_dir = os.path.join(env.nuitka_src, 'inline_copy', 'zlib')
    if os.path.exists(os.path.join(zlib_inline_copy_dir, 'crc32.c')):
        env.Append(CPPPATH=[zlib_inline_copy_dir])
    elif onefile_compile:
        env.Append(CPPDEFINES=['_NUITKA_USE_OWN_CRC32'])
    else:
        env.Append(CPPDEFINES=['_NUITKA_USE_SYSTEM_CRC32'])
        env.Append(LIBS='z')

def _enablePgoSettings(env, pgo_mode):
    if False:
        for i in range(10):
            print('nop')
    if pgo_mode == 'no':
        env.progressbar_name = 'Backend'
    elif pgo_mode == 'python':
        env.progressbar_name = 'Python Profile'
        env.Append(CPPDEFINES=['_NUITKA_PGO_PYTHON'])
    elif pgo_mode == 'generate':
        env.progressbar_name = 'Profile'
        env.Append(CPPDEFINES=['_NUITKA_PGO_GENERATE'])
        if env.gcc_mode:
            env.Append(CCFLAGS=['-fprofile-generate'])
            env.Append(LINKFLAGS=['-fprofile-generate'])
        elif env.msvc_mode:
            env.Append(CCFLAGS=['/GL'])
            env.Append(LINKFLAGS=['/GENPROFILE:EXACT'])
            if not env.clangcl_mode:
                env.Append(LINKFLAGS=['/LTCG'])
        else:
            scons_logger.sysexit("Error, PGO not supported for '%s' compiler." % env.the_cc_name)
    elif pgo_mode == 'use':
        env.progressbar_name = 'Backend'
        env.Append(CPPDEFINES=['_NUITKA_PGO_USE'])
        if env.gcc_mode:
            env.Append(CCFLAGS=['-fprofile-use'])
            env.Append(LINKFLAGS=['-fprofile-use'])
        elif env.msvc_mode:
            env.Append(CCFLAGS=['/GL'])
            env.Append(LINKFLAGS=['/USEPROFILE'])
        else:
            scons_logger.sysexit("Error, PGO not supported for '%s' compiler." % env.the_cc_name)
    else:
        assert False, env.pgo_mode
    env.pgo_mode = pgo_mode

def _enableDebugSystemSettings(env, job_count):
    if False:
        for i in range(10):
            print('nop')
    if env.unstripped_mode:
        if env.gcc_mode:
            env.Append(LINKFLAGS=['-g'])
            env.Append(CCFLAGS=['-g'])
            if not env.clang_mode:
                env.Append(CCFLAGS=['-feliminate-unused-debug-types'])
        elif env.msvc_mode:
            env.Append(CCFLAGS=['/Z7'])
            if job_count > 1 and getMsvcVersion(env) >= (11,):
                env.Append(CCFLAGS=['/FS'])
            env.Append(LINKFLAGS=['/DEBUG'])
    elif env.gcc_mode:
        if isMacOS():
            env.Append(LINKFLAGS=['-Wno-deprecated-declarations'])
        elif not env.clang_mode:
            env.Append(LINKFLAGS=['-s'])

def switchFromGccToGpp(env):
    if False:
        while True:
            i = 10
    if not env.gcc_mode or env.clang_mode:
        env.gcc_version = None
        return
    the_compiler = getExecutablePath(env.the_compiler, env)
    if the_compiler is None:
        return
    env.gcc_version = myDetectVersion(env, the_compiler)
    if env.gcc_version is None:
        scons_logger.sysexit("Error, failed to detect gcc version of backend compiler '%s'.\n" % env.the_compiler)
    if '++' in env.the_cc_name:
        scons_logger.sysexit('Error, compiler %s is apparently a C++ compiler, specify a C compiler instead.\n' % env.the_cc_name)
    if env.gcc_version < (4, 4):
        scons_logger.sysexit("The gcc compiler %s (version %s) doesn't have the sufficient version (>= 4.4)." % (env.the_compiler, env.gcc_version))
    if env.mingw_mode and env.gcc_version < (5, 3):
        scons_logger.sysexit("The MinGW64 compiler %s (version %s) doesn't have the sufficient version (>= 5.3)." % (env.the_compiler, env.gcc_version))
    if env.gcc_version < (5,):
        if env.python_version < (3, 11):
            scons_logger.info('The provided gcc is too old, switching to its g++ instead.', mnemonic='too-old-gcc')
            the_gpp_compiler = os.path.join(os.path.dirname(env.the_compiler), os.path.basename(env.the_compiler).replace('gcc', 'g++'))
            if getExecutablePath(the_gpp_compiler, env=env):
                env.the_compiler = the_gpp_compiler
                env.the_cc_name = env.the_cc_name.replace('gcc', 'g++')
            else:
                scons_logger.sysexit('Error, your gcc is too old for C11 support, and no related g++ to workaround that is found.')
        else:
            scons_logger.sysexit('Error, your gcc is too old for C11 support, install a newer one.', mnemonic='too-old-gcc')

def reportCCompiler(env, context, output_func):
    if False:
        for i in range(10):
            print('nop')
    cc_output = env.the_cc_name
    if env.the_cc_name == 'cl':
        cc_output = '%s %s' % (env.the_cc_name, getMsvcVersionString(env))
    elif isGccName(env.the_cc_name):
        cc_output = '%s %s' % (env.the_cc_name, '.'.join((str(d) for d in env.gcc_version)))
    elif isClangName(env.the_cc_name):
        cc_output = '%s %s' % (env.the_cc_name, '.'.join((str(d) for d in myDetectVersion(env, env.the_cc_name))))
    else:
        cc_output = env.the_cc_name
    output_func('%s C compiler: %s (%s).' % (context, getReportPath(env.the_compiler), cc_output))

def importEnvironmentVariableSettings(env):
    if False:
        for i in range(10):
            print('nop')
    'Import typical environment variables that compilation should use.'
    if 'CPPFLAGS' in os.environ:
        scons_logger.info("Scons: Inherited CPPFLAGS='%s' variable." % os.environ['CPPFLAGS'])
        env.Append(CPPFLAGS=os.environ['CPPFLAGS'].split())
    if 'CFLAGS' in os.environ:
        scons_logger.info("Inherited CFLAGS='%s' variable." % os.environ['CFLAGS'])
        env.Append(CCFLAGS=os.environ['CFLAGS'].split())
    if 'CCFLAGS' in os.environ:
        scons_logger.info("Inherited CCFLAGS='%s' variable." % os.environ['CCFLAGS'])
        env.Append(CCFLAGS=os.environ['CCFLAGS'].split())
    if 'CXXFLAGS' in os.environ:
        scons_logger.info("Scons: Inherited CXXFLAGS='%s' variable." % os.environ['CXXFLAGS'])
        env.Append(CXXFLAGS=os.environ['CXXFLAGS'].split())
    if 'LDFLAGS' in os.environ:
        scons_logger.info("Scons: Inherited LDFLAGS='%s' variable." % os.environ['LDFLAGS'])
        env.Append(LINKFLAGS=os.environ['LDFLAGS'].split())