import configparser
import fnmatch
import getpass
import glob
import os
import pickle
import platform
import re
import shutil
import signal
import subprocess
import sys
import threading
import _thread as thread
import time
import locations
SUFFIX_INC = ['.cxx', '.cpp', '.c', '.h', '.I', '.yxx', '.lxx', '.mm', '.rc', '.r']
SUFFIX_DLL = ['.dll', '.dlo', '.dle', '.dli', '.dlm', '.mll', '.exe', '.pyd', '.ocx']
SUFFIX_LIB = ['.lib', '.ilb']
VCS_DIRS = set(['CVS', 'CVSROOT', '.git', '.hg', '__pycache__'])
VCS_FILES = set(['.cvsignore', '.gitignore', '.gitmodules', '.hgignore'])
STARTTIME = time.time()
MAINTHREAD = threading.current_thread()
OUTPUTDIR = 'built'
CUSTOM_OUTPUTDIR = False
THIRDPARTYBASE = None
THIRDPARTYDIR = None
OPTIMIZE = '3'
VERBOSE = False
LINK_ALL_STATIC = False
TARGET = None
TARGET_ARCH = None
HAS_TARGET_ARCH = False
TOOLCHAIN_PREFIX = ''
ANDROID_ABI = None
ANDROID_TRIPLE = None
ANDROID_API = None
SYS_LIB_DIRS = []
SYS_INC_DIRS = []
DEBUG_DEPENDENCIES = False
if sys.platform == 'darwin':
    host_64 = sys.maxsize > 4294967296
else:
    host_64 = platform.architecture()[0] == '64bit'
ANDROID_SYS_LIBS = []
if os.path.exists('/etc/public.libraries.txt'):
    for line in open('/etc/public.libraries.txt', 'r'):
        line = line.strip()
        ANDROID_SYS_LIBS.append(line)
MSVCVERSIONINFO = {(10, 0): {'vsversion': (10, 0), 'vsname': 'Visual Studio 2010'}, (11, 0): {'vsversion': (11, 0), 'vsname': 'Visual Studio 2012'}, (12, 0): {'vsversion': (12, 0), 'vsname': 'Visual Studio 2013'}, (14, 0): {'vsversion': (14, 0), 'vsname': 'Visual Studio 2015'}, (14, 1): {'vsversion': (15, 0), 'vsname': 'Visual Studio 2017'}, (14, 2): {'vsversion': (16, 0), 'vsname': 'Visual Studio 2019'}, (14, 3): {'vsversion': (17, 0), 'vsname': 'Visual Studio 2022'}}
MAYAVERSIONINFO = [('MAYA6', '6.0'), ('MAYA65', '6.5'), ('MAYA7', '7.0'), ('MAYA8', '8.0'), ('MAYA85', '8.5'), ('MAYA2008', '2008'), ('MAYA2009', '2009'), ('MAYA2010', '2010'), ('MAYA2011', '2011'), ('MAYA2012', '2012'), ('MAYA2013', '2013'), ('MAYA20135', '2013.5'), ('MAYA2014', '2014'), ('MAYA2015', '2015'), ('MAYA2016', '2016'), ('MAYA20165', '2016.5'), ('MAYA2017', '2017'), ('MAYA2018', '2018'), ('MAYA2019', '2019'), ('MAYA2020', '2020'), ('MAYA2022', '2022')]
MAXVERSIONINFO = [('MAX6', 'SOFTWARE\\Autodesk\\3DSMAX\\6.0', 'installdir', 'maxsdk\\cssdk\\include'), ('MAX7', 'SOFTWARE\\Autodesk\\3DSMAX\\7.0', 'Installdir', 'maxsdk\\include\\CS'), ('MAX8', 'SOFTWARE\\Autodesk\\3DSMAX\\8.0', 'Installdir', 'maxsdk\\include\\CS'), ('MAX9', 'SOFTWARE\\Autodesk\\3DSMAX\\9.0', 'Installdir', 'maxsdk\\include\\CS'), ('MAX2009', 'SOFTWARE\\Autodesk\\3DSMAX\\11.0\\MAX-1:409', 'Installdir', 'maxsdk\\include\\CS'), ('MAX2010', 'SOFTWARE\\Autodesk\\3DSMAX\\12.0\\MAX-1:409', 'Installdir', 'maxsdk\\include\\CS'), ('MAX2011', 'SOFTWARE\\Autodesk\\3DSMAX\\13.0\\MAX-1:409', 'Installdir', 'maxsdk\\include\\CS'), ('MAX2012', 'SOFTWARE\\Autodesk\\3DSMAX\\14.0\\MAX-1:409', 'Installdir', 'maxsdk\\include\\CS'), ('MAX2013', 'SOFTWARE\\Autodesk\\3DSMAX\\15.0\\MAX-1:409', 'Installdir', 'maxsdk\\include\\CS'), ('MAX2014', 'SOFTWARE\\Autodesk\\3DSMAX\\16.0\\MAX-1:409', 'Installdir', 'maxsdk\\include\\CS')]
MAYAVERSIONS = []
MAXVERSIONS = []
DXVERSIONS = ['DX9']
for (ver, key) in MAYAVERSIONINFO:
    MAYAVERSIONS.append(ver)
for (ver, key1, key2, subdir) in MAXVERSIONINFO:
    MAXVERSIONS.append(ver)
CONFLICTING_FILES = ['dtool/src/dtoolutil/pandaVersion.h', 'dtool/src/dtoolutil/checkPandaVersion.h', 'dtool/src/dtoolutil/checkPandaVersion.cxx', 'dtool/src/prc/prc_parameters.h', 'contrib/src/speedtree/speedtree_parameters.h', 'direct/src/plugin/p3d_plugin_config.h', 'direct/src/plugin_activex/P3DActiveX.rc', 'direct/src/plugin_npapi/nppanda3d.rc', 'direct/src/plugin_standalone/panda3d.rc']

def WarnConflictingFiles(delete=False):
    if False:
        print('Hello World!')
    for cfile in CONFLICTING_FILES:
        if os.path.exists(cfile):
            Warn('file may conflict with build:', cfile)
            if delete:
                os.unlink(cfile)
                print('Deleted.')
WARNINGS = []
THREADS = {}
HAVE_COLORS = False
SETF = ''
try:
    import curses
    curses.setupterm()
    SETF = curses.tigetstr('setf')
    if SETF is None:
        SETF = curses.tigetstr('setaf')
    assert SETF is not None
    HAVE_COLORS = sys.stdout.isatty()
except:
    pass

def DisableColors():
    if False:
        return 10
    global HAVE_COLORS
    HAVE_COLORS = False

def GetColor(color=None):
    if False:
        for i in range(10):
            print('nop')
    if not HAVE_COLORS:
        return ''
    if color is not None:
        color = color.lower()
    if color == 'blue':
        token = curses.tparm(SETF, 1)
    elif color == 'green':
        token = curses.tparm(SETF, 2)
    elif color == 'cyan':
        token = curses.tparm(SETF, 3)
    elif color == 'red':
        token = curses.tparm(SETF, 4)
    elif color == 'magenta':
        token = curses.tparm(SETF, 5)
    elif color == 'yellow':
        token = curses.tparm(SETF, 6)
    else:
        token = curses.tparm(curses.tigetstr('sgr0'))
    return token.decode('ascii')

def ColorText(color, text, reset=True):
    if False:
        return 10
    if reset is True:
        return ''.join((GetColor(color), text, GetColor()))
    else:
        return ''.join((GetColor(color), text))

def PrettyTime(t):
    if False:
        print('Hello World!')
    t = int(t)
    hours = t // 3600
    t -= hours * 3600
    minutes = t // 60
    t -= minutes * 60
    seconds = t
    if hours:
        return '%d hours %d min' % (hours, minutes)
    if minutes:
        return '%d min %d sec' % (minutes, seconds)
    return '%d sec' % seconds

def ProgressOutput(progress, msg, target=None):
    if False:
        while True:
            i = 10
    sys.stdout.flush()
    sys.stderr.flush()
    prefix = ''
    thisthread = threading.current_thread()
    if thisthread is MAINTHREAD:
        if progress is None:
            prefix = ''
        elif progress >= 100.0:
            prefix = '%s[%s%d%%%s] ' % (GetColor('yellow'), GetColor('cyan'), progress, GetColor('yellow'))
        elif progress < 10.0:
            prefix = '%s[%s  %d%%%s] ' % (GetColor('yellow'), GetColor('cyan'), progress, GetColor('yellow'))
        else:
            prefix = '%s[%s %d%%%s] ' % (GetColor('yellow'), GetColor('cyan'), progress, GetColor('yellow'))
    else:
        global THREADS
        ident = thread.get_ident()
        if ident not in THREADS:
            THREADS[ident] = len(THREADS) + 1
        prefix = '%s[%sT%d%s] ' % (GetColor('yellow'), GetColor('cyan'), THREADS[ident], GetColor('yellow'))
    if target is not None:
        suffix = ' ' + ColorText('green', target)
    else:
        suffix = GetColor()
    print(''.join((prefix, msg, suffix)))
    sys.stdout.flush()
    sys.stderr.flush()

def exit(msg=''):
    if False:
        i = 10
        return i + 15
    sys.stdout.flush()
    sys.stderr.flush()
    if threading.current_thread() == MAINTHREAD:
        SaveDependencyCache()
        print('Elapsed Time: ' + PrettyTime(time.time() - STARTTIME))
        print(msg)
        print(ColorText('red', 'Build terminated.'))
        sys.stdout.flush()
        sys.stderr.flush()
        if __name__ != '__main__':
            os._exit(1)
    else:
        print(msg)
        raise 'initiate-exit'

def Warn(msg, extra=None):
    if False:
        while True:
            i = 10
    if extra is not None:
        print('%sWARNING:%s %s %s%s%s' % (GetColor('red'), GetColor(), msg, GetColor('green'), extra, GetColor()))
    else:
        print('%sWARNING:%s %s' % (GetColor('red'), GetColor(), msg))
    sys.stdout.flush()

def Error(msg, extra=None):
    if False:
        return 10
    if extra is not None:
        print('%sERROR:%s %s %s%s%s' % (GetColor('red'), GetColor(), msg, GetColor('green'), extra, GetColor()))
    else:
        print('%sERROR:%s %s' % (GetColor('red'), GetColor(), msg))
    exit()

def GetHost():
    if False:
        i = 10
        return i + 15
    "Returns the host platform, ie. the one we're compiling on."
    if sys.platform == 'win32' or sys.platform == 'cygwin':
        return 'windows'
    elif sys.platform == 'darwin':
        return 'darwin'
    elif sys.platform.startswith('linux'):
        try:
            osname = subprocess.check_output(['uname', '-o'])
            if osname.strip().lower() == b'android':
                return 'android'
            else:
                return 'linux'
        except:
            return 'linux'
    elif sys.platform.startswith('freebsd'):
        return 'freebsd'
    else:
        exit('Unrecognized sys.platform: %s' % sys.platform)

def GetHostArch():
    if False:
        print('Hello World!')
    "Returns the architecture we're compiling on.\n    Its value is also platform-dependent, as different platforms\n    have different architecture naming."
    target = GetTarget()
    if target == 'windows':
        return 'x64' if host_64 else 'x86'
    machine = platform.machine()
    if machine.startswith('armv7'):
        return 'armv7a'
    else:
        return machine

def SetTarget(target, arch=None):
    if False:
        return 10
    "Sets the target platform; the one we're compiling for.  Also\n    sets the target architecture (None for default, if any).  Should\n    be called *before* any calls are made to GetOutputDir, GetCC, etc."
    global TARGET, TARGET_ARCH, HAS_TARGET_ARCH
    global TOOLCHAIN_PREFIX
    host = GetHost()
    host_arch = GetHostArch()
    if target is None:
        target = host
    else:
        target = target.lower()
    if arch is not None:
        HAS_TARGET_ARCH = True
    TOOLCHAIN_PREFIX = ''
    if target == 'windows':
        if arch == 'i386':
            arch = 'x86'
        elif arch == 'amd64':
            arch = 'x64'
        if arch is not None and arch != 'x86' and (arch != 'x64'):
            exit('Windows architecture must be x86 or x64')
    elif target == 'darwin':
        if arch == 'amd64':
            arch = 'x86_64'
        if arch == 'aarch64':
            arch = 'arm64'
        if arch is not None:
            choices = ('i386', 'x86_64', 'ppc', 'ppc64', 'arm64')
            if arch not in choices:
                exit('macOS architecture must be one of %s' % ', '.join(choices))
    elif target == 'android' or target.startswith('android-'):
        if arch is None:
            if host == 'android':
                arch = host_arch
            else:
                arch = 'armv7a'
        if arch == 'aarch64':
            arch = 'arm64'
        global ANDROID_API
        (target, _, api) = target.partition('-')
        if api:
            ANDROID_API = int(api)
        elif arch in ('mips64', 'arm64', 'x86_64'):
            ANDROID_API = 21
        else:
            ANDROID_API = 19
        global ANDROID_ABI, ANDROID_TRIPLE
        if arch == 'armv7a':
            ANDROID_ABI = 'armeabi-v7a'
            ANDROID_TRIPLE = 'armv7a-linux-androideabi'
        elif arch == 'arm':
            ANDROID_ABI = 'armeabi'
            ANDROID_TRIPLE = 'arm-linux-androideabi'
        elif arch == 'arm64':
            ANDROID_ABI = 'arm64-v8a'
            ANDROID_TRIPLE = 'aarch64-linux-android'
        elif arch == 'mips':
            ANDROID_ABI = 'mips'
            ANDROID_TRIPLE = 'mipsel-linux-android'
        elif arch == 'mips64':
            ANDROID_ABI = 'mips64'
            ANDROID_TRIPLE = 'mips64el-linux-android'
        elif arch == 'x86':
            ANDROID_ABI = 'x86'
            ANDROID_TRIPLE = 'i686-linux-android'
        elif arch == 'x86_64':
            ANDROID_ABI = 'x86_64'
            ANDROID_TRIPLE = 'x86_64-linux-android'
        else:
            exit('Android architecture must be arm, armv7a, arm64, mips, mips64, x86 or x86_64, use --arch to specify')
        ANDROID_TRIPLE += str(ANDROID_API)
        TOOLCHAIN_PREFIX = ANDROID_TRIPLE + '-'
    elif target == 'linux':
        if arch is not None:
            TOOLCHAIN_PREFIX = '%s-linux-gnu-' % arch
        elif host != 'linux':
            exit('Should specify an architecture when building for Linux')
    elif target == host:
        if arch is None or arch == host_arch:
            pass
        else:
            exit('Cannot cross-compile for %s-%s from %s-%s' % (target, arch, host, host_arch))
    else:
        exit('Cannot cross-compile for %s from %s' % (target, host))
    if arch is None:
        arch = host_arch
    TARGET = target
    TARGET_ARCH = arch

def GetTarget():
    if False:
        for i in range(10):
            print('nop')
    "Returns the platform we're compiling for.  Defaults to GetHost()."
    global TARGET
    if TARGET is None:
        TARGET = GetHost()
    return TARGET

def HasTargetArch():
    if False:
        print('Hello World!')
    'Returns True if the user specified an architecture to compile for.'
    return HAS_TARGET_ARCH

def GetTargetArch():
    if False:
        while True:
            i = 10
    "Returns the architecture we're compiling for.  Defaults to GetHostArch().\n    Its value is also dependent on that of GetTarget(), as different platforms\n    use a different architecture naming."
    global TARGET_ARCH
    if TARGET_ARCH is None:
        TARGET_ARCH = GetHostArch()
    return TARGET_ARCH

def CrossCompiling():
    if False:
        while True:
            i = 10
    "Returns True if we're cross-compiling."
    return GetTarget() != GetHost()

def GetCC():
    if False:
        while True:
            i = 10
    if TARGET in ('darwin', 'freebsd', 'android'):
        return os.environ.get('CC', TOOLCHAIN_PREFIX + 'clang')
    else:
        return os.environ.get('CC', TOOLCHAIN_PREFIX + 'gcc')

def GetCXX():
    if False:
        for i in range(10):
            print('nop')
    if TARGET in ('darwin', 'freebsd', 'android'):
        return os.environ.get('CXX', TOOLCHAIN_PREFIX + 'clang++')
    else:
        return os.environ.get('CXX', TOOLCHAIN_PREFIX + 'g++')

def GetStrip():
    if False:
        print('Hello World!')
    if TARGET == 'android':
        return 'llvm-strip'
    else:
        return 'strip'

def GetAR():
    if False:
        print('Hello World!')
    if TARGET == 'android':
        return TOOLCHAIN_PREFIX + 'ar'
    else:
        return 'ar'

def GetRanlib():
    if False:
        print('Hello World!')
    if TARGET == 'android':
        return TOOLCHAIN_PREFIX + 'ranlib'
    else:
        return 'ranlib'
BISON = None

def GetBison():
    if False:
        for i in range(10):
            print('nop')
    global BISON
    if BISON is not None:
        return BISON
    win_util_data = os.path.join(GetThirdpartyBase(), 'win-util', 'data')
    if GetHost() == 'windows' and os.path.isdir(win_util_data):
        BISON = os.path.join(GetThirdpartyBase(), 'win-util', 'bison.exe')
    elif LocateBinary('bison'):
        BISON = 'bison'
    else:
        return None
    return BISON
FLEX = None

def GetFlex():
    if False:
        i = 10
        return i + 15
    global FLEX
    if FLEX is not None:
        return FLEX
    win_util = os.path.join(GetThirdpartyBase(), 'win-util')
    if GetHost() == 'windows' and os.path.isdir(win_util):
        FLEX = GetThirdpartyBase() + '/win-util/flex.exe'
    elif LocateBinary('flex'):
        FLEX = 'flex'
    else:
        return None
    return FLEX

def GetFlexVersion():
    if False:
        for i in range(10):
            print('nop')
    flex = GetFlex()
    if not flex:
        return (0, 0, 0)
    try:
        handle = subprocess.Popen(['flex', '--version'], executable=flex, stdout=subprocess.PIPE)
        words = handle.communicate()[0].strip().splitlines()[0].split(b' ')
        if words[1] != 'version':
            version = words[1]
        else:
            version = words[2]
        return tuple(map(int, version.split(b'.')))
    except:
        Warn('Unable to detect flex version')
        return (0, 0, 0)
SEVENZIP = None

def GetSevenZip():
    if False:
        i = 10
        return i + 15
    global SEVENZIP
    if SEVENZIP is not None:
        return SEVENZIP
    win_util = os.path.join(GetThirdpartyBase(), 'win-util')
    if GetHost() == 'windows' and os.path.isdir(win_util):
        SEVENZIP = GetThirdpartyBase() + '/win-util/7za.exe'
    elif LocateBinary('7z'):
        SEVENZIP = '7z'
    else:
        return None
    return SEVENZIP

def HasSevenZip():
    if False:
        while True:
            i = 10
    return GetSevenZip() is not None

def LocateBinary(binary):
    if False:
        print('Hello World!')
    if os.path.isfile(binary):
        return binary
    if 'PATH' not in os.environ or os.environ['PATH'] == '':
        p = os.defpath
    else:
        p = os.environ['PATH']
    pathList = p.split(os.pathsep)
    suffixes = ['']
    if GetHost() == 'windows':
        if not binary.lower().endswith('.exe') and (not binary.lower().endswith('.bat')):
            suffixes = ['.exe', '.bat']
        pathList = ['.'] + pathList
    for path in pathList:
        binpath = os.path.join(os.path.expanduser(path), binary)
        for suffix in suffixes:
            if os.access(binpath + suffix, os.X_OK):
                return os.path.abspath(os.path.realpath(binpath + suffix))
    return None

def oscmd(cmd, ignoreError=False, cwd=None):
    if False:
        while True:
            i = 10
    if VERBOSE:
        print(GetColor('blue') + cmd.split(' ', 1)[0] + ' ' + GetColor('magenta') + cmd.split(' ', 1)[1] + GetColor())
    sys.stdout.flush()
    if sys.platform == 'win32':
        if cmd[0] == '"':
            exe = cmd[1:cmd.index('"', 1)]
        else:
            exe = cmd.split()[0]
        exe_path = LocateBinary(exe)
        if exe_path is None:
            exit('Cannot find ' + exe + ' on search path')
        if cwd is not None:
            pwd = os.getcwd()
            os.chdir(cwd)
        res = os.spawnl(os.P_WAIT, exe_path, cmd)
        if res == -1073741510:
            exit('keyboard interrupt')
        if cwd is not None:
            os.chdir(pwd)
    else:
        cmd = cmd.replace(';', '\\;')
        cmd = cmd.replace('$', '\\$')
        res = subprocess.call(cmd, cwd=cwd, shell=True)
        sig = res & 127
        if GetVerbose() and res != 0:
            print(ColorText('red', 'Process exited with exit status %d and signal code %d' % ((res & 65280) >> 8, sig)))
        if sig == signal.SIGINT:
            exit('keyboard interrupt')
        if sig == signal.SIGSEGV or res == 35584 or res == 34304:
            if LocateBinary('gdb') and GetVerbose() and (GetHost() != 'windows'):
                print(ColorText('red', 'Received SIGSEGV, retrieving traceback...'))
                os.system("gdb -batch -ex 'handle SIG33 pass nostop noprint' -ex 'set pagination 0' -ex 'run' -ex 'bt full' -ex 'info registers' -ex 'thread apply all backtrace' -ex 'quit' --args %s < /dev/null" % cmd)
            else:
                print(ColorText('red', 'Received SIGSEGV'))
            exit('')
    if res != 0 and (not ignoreError):
        if 'interrogate' in cmd.split(' ', 1)[0] and GetVerbose():
            print(ColorText('red', 'Interrogate failed, retrieving debug output...'))
            sys.stdout.flush()
            verbose_cmd = cmd.split(' ', 1)[0] + ' -vv ' + cmd.split(' ', 1)[1]
            if sys.platform == 'win32':
                os.spawnl(os.P_WAIT, exe_path, verbose_cmd)
            else:
                subprocess.call(verbose_cmd, shell=True)
        exit('The following command returned a non-zero value: ' + str(cmd))
    return res

def GetDirectoryContents(dir, filters='*', skip=[]):
    if False:
        print('Hello World!')
    if isinstance(filters, str):
        filters = [filters]
    actual = {}
    files = os.listdir(dir)
    for filter in filters:
        for file in fnmatch.filter(files, filter):
            if skip.count(file) == 0 and os.path.isfile(dir + '/' + file):
                actual[file] = 1
    results = list(actual.keys())
    results.sort()
    return results

def GetDirectorySize(dir):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.isdir(dir):
        return 0
    size = 0
    for (path, dirs, files) in os.walk(dir):
        for file in files:
            try:
                size += os.path.getsize(os.path.join(path, file))
            except:
                pass
    return size
TIMESTAMPCACHE = {}

def GetTimestamp(path):
    if False:
        print('Hello World!')
    if path in TIMESTAMPCACHE:
        return TIMESTAMPCACHE[path]
    try:
        date = int(os.path.getmtime(path))
    except:
        date = 0
    TIMESTAMPCACHE[path] = date
    return date

def ClearTimestamp(path):
    if False:
        return 10
    del TIMESTAMPCACHE[path]
BUILTFROMCACHE = {}

def JustBuilt(files, others):
    if False:
        i = 10
        return i + 15
    dates = {}
    for file in files:
        del TIMESTAMPCACHE[file]
        dates[file] = GetTimestamp(file)
    for file in others:
        dates[file] = GetTimestamp(file)
    key = tuple(files)
    BUILTFROMCACHE[key] = dates

def NeedsBuild(files, others):
    if False:
        while True:
            i = 10
    dates = {}
    for file in files:
        dates[file] = GetTimestamp(file)
        if not os.path.exists(file):
            if DEBUG_DEPENDENCIES:
                print('rebuilding %s because it does not exist' % file)
            return True
    for file in others:
        dates[file] = GetTimestamp(file)
    key = tuple(files)
    if key in BUILTFROMCACHE:
        cached = BUILTFROMCACHE[key]
        if cached == dates:
            return False
        elif DEBUG_DEPENDENCIES:
            print('rebuilding %s because:' % key)
            for key in frozenset(cached.keys()) | frozenset(dates.keys()):
                if key not in cached:
                    print('    new dependency: %s' % key)
                elif key not in dates:
                    print('    removed dependency: %s' % key)
                elif cached[key] != dates[key]:
                    print('    dependency changed: %s' % key)
        if VERBOSE and frozenset(cached) != frozenset(dates):
            Warn('file dependencies changed:', files)
    return True
CXXINCLUDECACHE = {}
CxxIncludeRegex = re.compile('^[ \t]*[#][ \t]*include[ \t]+"([^"]+)"[ \t\r\n]*$')

def CxxGetIncludes(path):
    if False:
        print('Hello World!')
    date = GetTimestamp(path)
    if path in CXXINCLUDECACHE:
        cached = CXXINCLUDECACHE[path]
        if cached[0] == date:
            return cached[1]
    try:
        sfile = open(path, 'r')
    except:
        exit('Cannot open source file "' + path + '" for reading.')
    include = []
    try:
        for line in sfile:
            match = CxxIncludeRegex.match(line, 0)
            if match:
                incname = match.group(1)
                include.append(incname)
    except:
        print('Failed to determine dependencies of "' + path + '".')
        raise
    sfile.close()
    CXXINCLUDECACHE[path] = [date, include]
    return include
JAVAIMPORTCACHE = {}
JavaImportRegex = re.compile('[ \t\r\n;]import[ \t]+([a-zA-Z][^;]+)[ \t\r\n]*;')

def JavaGetImports(path):
    if False:
        print('Hello World!')
    date = GetTimestamp(path)
    if path in JAVAIMPORTCACHE:
        cached = JAVAIMPORTCACHE[path]
        if cached[0] == date:
            return cached[1]
    try:
        source = open(path, 'r').read()
    except:
        exit('Cannot open source file "' + path + '" for reading.')
    imports = []
    try:
        for match in JavaImportRegex.finditer(source, 0):
            impname = match.group(1).strip()
            if not impname.startswith('java.') and (not impname.startswith('dalvik.')) and (not impname.startswith('android.')):
                imports.append(impname.strip())
    except:
        print('Failed to determine dependencies of "' + path + '".')
        raise
    JAVAIMPORTCACHE[path] = [date, imports]
    return imports
DCACHE_VERSION = 3
DCACHE_BACKED_UP = False

def SaveDependencyCache():
    if False:
        for i in range(10):
            print('nop')
    global DCACHE_BACKED_UP
    if not DCACHE_BACKED_UP:
        try:
            if os.path.exists(os.path.join(OUTPUTDIR, 'tmp', 'makepanda-dcache')):
                os.rename(os.path.join(OUTPUTDIR, 'tmp', 'makepanda-dcache'), os.path.join(OUTPUTDIR, 'tmp', 'makepanda-dcache-backup'))
        except:
            pass
        DCACHE_BACKED_UP = True
    try:
        icache = open(os.path.join(OUTPUTDIR, 'tmp', 'makepanda-dcache'), 'wb')
    except:
        icache = None
    if icache is not None:
        print('Storing dependency cache.')
        pickle.dump(DCACHE_VERSION, icache, 0)
        pickle.dump(CXXINCLUDECACHE, icache, 2)
        pickle.dump(BUILTFROMCACHE, icache, 2)
        icache.close()

def LoadDependencyCache():
    if False:
        i = 10
        return i + 15
    global CXXINCLUDECACHE
    global BUILTFROMCACHE
    try:
        icache = open(os.path.join(OUTPUTDIR, 'tmp', 'makepanda-dcache'), 'rb')
    except:
        icache = None
    if icache is not None:
        ver = pickle.load(icache)
        if ver == DCACHE_VERSION:
            CXXINCLUDECACHE = pickle.load(icache)
            BUILTFROMCACHE = pickle.load(icache)
            icache.close()
        else:
            print('Cannot load dependency cache, version is too old!')

def CxxFindSource(name, ipath):
    if False:
        for i in range(10):
            print('nop')
    for dir in ipath:
        if dir == '.':
            full = name
        else:
            full = dir + '/' + name
        if GetTimestamp(full) > 0:
            return full
    exit('Could not find source file: ' + name)

def CxxFindHeader(srcfile, incfile, ipath):
    if False:
        return 10
    if incfile.startswith('.'):
        last = srcfile.rfind('/')
        if last < 0:
            exit('CxxFindHeader cannot handle this case #1')
        srcdir = srcfile[:last + 1]
        while incfile[:1] == '.':
            if incfile[:2] == './':
                incfile = incfile[2:]
            elif incfile[:3] == '../':
                incfile = incfile[3:]
                last = srcdir[:-1].rfind('/')
                if last < 0:
                    exit('CxxFindHeader cannot handle this case #2')
                srcdir = srcdir[:last + 1]
            else:
                exit('CxxFindHeader cannot handle this case #3')
        full = srcdir + incfile
        if GetTimestamp(full) > 0:
            return full
        return 0
    else:
        for dir in ipath:
            full = dir + '/' + incfile
            if GetTimestamp(full) > 0:
                return full
        return 0

def JavaFindClasses(impspec, clspath):
    if False:
        for i in range(10):
            print('nop')
    path = clspath + '/' + impspec.replace('.', '/') + '.class'
    if '*' in path:
        return glob.glob(path)
    else:
        return [path]
CxxIgnoreHeader = {}
CxxDependencyCache = {}

def CxxCalcDependencies(srcfile, ipath, ignore):
    if False:
        print('Hello World!')
    if srcfile in CxxDependencyCache:
        return CxxDependencyCache[srcfile]
    if ignore.count(srcfile):
        return []
    dep = {}
    dep[srcfile] = 1
    includes = CxxGetIncludes(srcfile)
    for include in includes:
        header = CxxFindHeader(srcfile, include, ipath)
        if header != 0:
            if ignore.count(header) == 0:
                hdeps = CxxCalcDependencies(header, ipath, [srcfile] + ignore)
                for x in hdeps:
                    dep[x] = 1
    result = list(dep.keys())
    CxxDependencyCache[srcfile] = result
    return result
global JavaDependencyCache
JavaDependencyCache = {}

def JavaCalcDependencies(srcfile, clspath):
    if False:
        return 10
    if srcfile in JavaDependencyCache:
        return JavaDependencyCache[srcfile]
    deps = set((srcfile,))
    JavaDependencyCache[srcfile] = deps
    imports = JavaGetImports(srcfile)
    for impspec in imports:
        for cls in JavaFindClasses(impspec, clspath):
            deps.add(cls)
    return deps
if sys.platform == 'win32':
    import winreg

def TryRegistryKey(path):
    if False:
        return 10
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_READ)
        return key
    except:
        pass
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, path, 0, winreg.KEY_READ | 256)
        return key
    except:
        pass
    return 0

def ListRegistryKeys(path):
    if False:
        i = 10
        return i + 15
    result = []
    index = 0
    key = TryRegistryKey(path)
    if key != 0:
        try:
            while 1:
                result.append(winreg.EnumKey(key, index))
                index = index + 1
        except:
            pass
        winreg.CloseKey(key)
    return result

def ListRegistryValues(path):
    if False:
        while True:
            i = 10
    result = []
    index = 0
    key = TryRegistryKey(path)
    if key != 0:
        try:
            while 1:
                result.append(winreg.EnumValue(key, index)[0])
                index = index + 1
        except:
            pass
        winreg.CloseKey(key)
    return result

def GetRegistryKey(path, subkey, override64=True):
    if False:
        return 10
    if host_64 and override64:
        path = path.replace('SOFTWARE\\', 'SOFTWARE\\Wow6432Node\\')
    k1 = 0
    key = TryRegistryKey(path)
    if key != 0:
        try:
            (k1, k2) = winreg.QueryValueEx(key, subkey)
        except:
            pass
        winreg.CloseKey(key)
    return k1

def GetProgramFiles():
    if False:
        return 10
    if 'PROGRAMFILES' in os.environ:
        return os.environ['PROGRAMFILES']
    elif os.path.isdir('C:\\Program Files'):
        return 'C:\\Program Files'
    elif os.path.isdir('D:\\Program Files'):
        return 'D:\\Program Files'
    elif os.path.isdir('E:\\Program Files'):
        return 'E:\\Program Files'
    return 0

def GetProgramFiles_x86():
    if False:
        i = 10
        return i + 15
    if 'ProgramFiles(x86)' in os.environ:
        return os.environ['ProgramFiles(x86)']
    elif os.path.isdir('C:\\Program Files (x86)'):
        return 'C:\\Program Files (x86)'
    elif os.path.isdir('D:\\Program Files (x86)'):
        return 'D:\\Program Files (x86)'
    elif os.path.isdir('E:\\Program Files (x86)'):
        return 'E:\\Program Files (x86)'
    return GetProgramFiles()

def GetListOption(opts, prefix):
    if False:
        print('Hello World!')
    res = []
    for x in opts:
        if x.startswith(prefix):
            res.append(x[len(prefix):])
    return res

def GetValueOption(opts, prefix):
    if False:
        i = 10
        return i + 15
    for x in opts:
        if x.startswith(prefix):
            return x[len(prefix):]
    return 0

def GetOptimizeOption(opts):
    if False:
        i = 10
        return i + 15
    val = GetValueOption(opts, 'OPT:')
    if val == 0:
        return OPTIMIZE
    return val

def MakeDirectory(path, mode=None, recursive=False):
    if False:
        return 10
    if os.path.isdir(path):
        return
    if recursive:
        parent = os.path.dirname(path)
        if parent and (not os.path.isdir(parent)):
            MakeDirectory(parent, mode=mode, recursive=True)
    if mode is not None:
        os.mkdir(path, mode)
    else:
        os.mkdir(path)

def ReadFile(wfile):
    if False:
        while True:
            i = 10
    try:
        srchandle = open(wfile, 'r')
        data = srchandle.read()
        srchandle.close()
        return data
    except:
        ex = sys.exc_info()[1]
        exit('Cannot read %s: %s' % (wfile, ex))

def ReadBinaryFile(wfile):
    if False:
        for i in range(10):
            print('nop')
    try:
        srchandle = open(wfile, 'rb')
        data = srchandle.read()
        srchandle.close()
        return data
    except:
        ex = sys.exc_info()[1]
        exit('Cannot read %s: %s' % (wfile, ex))

def WriteFile(wfile, data, newline=None):
    if False:
        i = 10
        return i + 15
    if newline is not None:
        data = data.replace('\r\n', '\n')
        data = data.replace('\r', '\n')
        data = data.replace('\n', newline)
    try:
        dsthandle = open(wfile, 'w', newline='')
        dsthandle.write(data)
        dsthandle.close()
    except:
        ex = sys.exc_info()[1]
        exit('Cannot write to %s: %s' % (wfile, ex))

def WriteBinaryFile(wfile, data):
    if False:
        while True:
            i = 10
    try:
        dsthandle = open(wfile, 'wb')
        dsthandle.write(data)
        dsthandle.close()
    except:
        ex = sys.exc_info()[1]
        exit('Cannot write to %s: %s' % (wfile, ex))

def ConditionalWriteFile(dest, data, newline=None):
    if False:
        i = 10
        return i + 15
    if newline is not None:
        data = data.replace('\r\n', '\n')
        data = data.replace('\r', '\n')
        data = data.replace('\n', newline)
    try:
        rfile = open(dest, 'r')
        contents = rfile.read(-1)
        rfile.close()
    except:
        contents = 0
    if contents != data:
        if VERBOSE:
            print('Writing %s' % dest)
        sys.stdout.flush()
        WriteFile(dest, data)

def DeleteVCS(dir):
    if False:
        return 10
    if dir == '':
        dir = '.'
    for entry in os.listdir(dir):
        subdir = os.path.join(dir, entry)
        if os.path.isdir(subdir):
            if entry in VCS_DIRS:
                shutil.rmtree(subdir)
            else:
                DeleteVCS(subdir)
        elif os.path.isfile(subdir) and (entry in VCS_FILES or entry.startswith('.#')):
            os.remove(subdir)

def DeleteBuildFiles(dir):
    if False:
        while True:
            i = 10
    if dir == '':
        dir = '.'
    for entry in os.listdir(dir):
        subdir = os.path.join(dir, entry)
        if os.path.isfile(subdir) and os.path.splitext(subdir)[-1] in ['.pp', '.moved']:
            os.remove(subdir)
        elif os.path.isdir(subdir):
            if os.path.basename(subdir)[:3] == 'Opt' and os.path.basename(subdir)[4] == '-':
                shutil.rmtree(subdir)
            else:
                DeleteBuildFiles(subdir)

def DeleteEmptyDirs(dir):
    if False:
        return 10
    if dir == '':
        dir = '.'
    entries = os.listdir(dir)
    if not entries:
        os.rmdir(dir)
        return
    for entry in entries:
        subdir = os.path.join(dir, entry)
        if os.path.isdir(subdir):
            if not os.listdir(subdir):
                os.rmdir(subdir)
            else:
                DeleteEmptyDirs(subdir)

def CreateFile(file):
    if False:
        while True:
            i = 10
    if os.path.exists(file) == 0:
        WriteFile(file, '')

def MakeBuildTree():
    if False:
        i = 10
        return i + 15
    MakeDirectory(OUTPUTDIR)
    MakeDirectory(OUTPUTDIR + '/bin')
    MakeDirectory(OUTPUTDIR + '/lib')
    MakeDirectory(OUTPUTDIR + '/tmp')
    MakeDirectory(OUTPUTDIR + '/etc')
    MakeDirectory(OUTPUTDIR + '/plugins')
    MakeDirectory(OUTPUTDIR + '/include')
    MakeDirectory(OUTPUTDIR + '/models')
    MakeDirectory(OUTPUTDIR + '/models/audio')
    MakeDirectory(OUTPUTDIR + '/models/audio/sfx')
    MakeDirectory(OUTPUTDIR + '/models/icons')
    MakeDirectory(OUTPUTDIR + '/models/maps')
    MakeDirectory(OUTPUTDIR + '/models/misc')
    MakeDirectory(OUTPUTDIR + '/models/gui')
    MakeDirectory(OUTPUTDIR + '/pandac')
    MakeDirectory(OUTPUTDIR + '/pandac/input')
    MakeDirectory(OUTPUTDIR + '/panda3d')
    if GetTarget() == 'android':
        MakeDirectory(OUTPUTDIR + '/classes')

def CheckPandaSourceTree():
    if False:
        while True:
            i = 10
    dir = os.getcwd()
    if os.path.exists(os.path.join(dir, 'makepanda/makepanda.py')) == 0 or os.path.exists(os.path.join(dir, 'dtool', 'src', 'dtoolbase', 'dtoolbase.h')) == 0 or os.path.exists(os.path.join(dir, 'panda', 'src', 'pandabase', 'pandabase.h')) == 0:
        exit('Current directory is not the root of the panda tree.')

def GetThirdpartyBase():
    if False:
        while True:
            i = 10
    "Returns the location of the 'thirdparty' directory.\n    Normally, this is simply the thirdparty directory relative\n    to the root of the source root, but if a MAKEPANDA_THIRDPARTY\n    environment variable was set, it is used as the location of the\n    thirdparty directory.  This is useful when wanting to use a single\n    system-wide thirdparty directory, for instance on a build machine."
    global THIRDPARTYBASE
    if THIRDPARTYBASE is not None:
        return THIRDPARTYBASE
    THIRDPARTYBASE = 'thirdparty'
    if 'MAKEPANDA_THIRDPARTY' in os.environ:
        THIRDPARTYBASE = os.environ['MAKEPANDA_THIRDPARTY']
    return THIRDPARTYBASE

def GetThirdpartyDir():
    if False:
        print('Hello World!')
    'Returns the thirdparty directory for the target platform,\n    ie. thirdparty/win-libs-vc10/.  May return None in the future.'
    global THIRDPARTYDIR
    if THIRDPARTYDIR is not None:
        return THIRDPARTYDIR
    base = GetThirdpartyBase()
    target = GetTarget()
    target_arch = GetTargetArch()
    if target == 'windows':
        vc = str(SDK['MSVC_VERSION'][0])
        if target_arch == 'x64':
            THIRDPARTYDIR = base + '/win-libs-vc' + vc + '-x64/'
        else:
            THIRDPARTYDIR = base + '/win-libs-vc' + vc + '/'
    elif target == 'darwin':
        THIRDPARTYDIR = base + '/darwin-libs-a/'
    elif target == 'linux':
        if target_arch in ('aarch64', 'arm64'):
            THIRDPARTYDIR = base + '/linux-libs-arm64/'
        elif target_arch.startswith('arm'):
            THIRDPARTYDIR = base + '/linux-libs-arm/'
        elif target_arch in ('x86_64', 'amd64'):
            THIRDPARTYDIR = base + '/linux-libs-x64/'
        else:
            THIRDPARTYDIR = base + '/linux-libs-a/'
    elif target == 'freebsd':
        if target_arch in ('aarch64', 'arm64'):
            THIRDPARTYDIR = base + '/freebsd-libs-arm64/'
        elif target_arch.startswith('arm'):
            THIRDPARTYDIR = base + '/freebsd-libs-arm/'
        elif target_arch in ('x86_64', 'amd64'):
            THIRDPARTYDIR = base + '/freebsd-libs-x64/'
        else:
            THIRDPARTYDIR = base + '/freebsd-libs-a/'
    elif target == 'android':
        THIRDPARTYDIR = base + '/android-libs-%s/' % target_arch
    else:
        Warn('Unsupported platform:', target)
        return
    if GetVerbose():
        print('Using thirdparty directory: %s' % THIRDPARTYDIR)
    return THIRDPARTYDIR

def GetOutputDir():
    if False:
        i = 10
        return i + 15
    return OUTPUTDIR

def IsCustomOutputDir():
    if False:
        while True:
            i = 10
    return CUSTOM_OUTPUTDIR

def SetOutputDir(outputdir):
    if False:
        for i in range(10):
            print('nop')
    global OUTPUTDIR, CUSTOM_OUTPUTDIR
    OUTPUTDIR = outputdir
    CUSTOM_OUTPUTDIR = True

def GetOptimize():
    if False:
        return 10
    return int(OPTIMIZE)

def SetOptimize(optimize):
    if False:
        for i in range(10):
            print('nop')
    global OPTIMIZE
    OPTIMIZE = optimize

def GetVerbose():
    if False:
        while True:
            i = 10
    return VERBOSE

def SetVerbose(verbose):
    if False:
        while True:
            i = 10
    global VERBOSE
    VERBOSE = verbose

def SetDebugDependencies(dd=True):
    if False:
        return 10
    global DEBUG_DEPENDENCIES
    DEBUG_DEPENDENCIES = dd

def GetLinkAllStatic():
    if False:
        print('Hello World!')
    return LINK_ALL_STATIC

def SetLinkAllStatic(val=True):
    if False:
        while True:
            i = 10
    global LINK_ALL_STATIC
    LINK_ALL_STATIC = val

def UnsetLinkAllStatic():
    if False:
        i = 10
        return i + 15
    global LINK_ALL_STATIC
    LINK_ALL_STATIC = False
PKG_LIST_ALL = []
PKG_LIST_OMIT = {}
PKG_LIST_CUSTOM = set()

def PkgListSet(pkgs):
    if False:
        return 10
    global PKG_LIST_ALL
    global PKG_LIST_OMIT
    PKG_LIST_ALL = pkgs
    PKG_LIST_OMIT = {}
    PkgEnableAll()

def PkgListGet():
    if False:
        i = 10
        return i + 15
    return PKG_LIST_ALL

def PkgEnableAll():
    if False:
        print('Hello World!')
    for x in PKG_LIST_ALL:
        PKG_LIST_OMIT[x] = 0

def PkgDisableAll():
    if False:
        i = 10
        return i + 15
    for x in PKG_LIST_ALL:
        PKG_LIST_OMIT[x] = 1

def PkgEnable(pkg):
    if False:
        while True:
            i = 10
    PKG_LIST_OMIT[pkg] = 0

def PkgDisable(pkg):
    if False:
        for i in range(10):
            print('nop')
    PKG_LIST_OMIT[pkg] = 1

def PkgSetCustomLocation(pkg):
    if False:
        return 10
    PKG_LIST_CUSTOM.add(pkg)

def PkgHasCustomLocation(pkg):
    if False:
        while True:
            i = 10
    return pkg in PKG_LIST_CUSTOM

def PkgSkip(pkg):
    if False:
        print('Hello World!')
    return PKG_LIST_OMIT[pkg]

def PkgSelected(pkglist, pkg):
    if False:
        i = 10
        return i + 15
    if pkglist.count(pkg) == 0:
        return 0
    if PKG_LIST_OMIT[pkg]:
        return 0
    return 1
OVERRIDES_LIST = {}

def AddOverride(spec):
    if False:
        for i in range(10):
            print('nop')
    if spec.find('=') == -1:
        return
    pair = spec.split('=', 1)
    OVERRIDES_LIST[pair[0]] = pair[1]

def OverrideValue(parameter, value):
    if False:
        print('Hello World!')
    if parameter in OVERRIDES_LIST:
        print('Overriding value of key "' + parameter + '" with value "' + OVERRIDES_LIST[parameter] + '"')
        return OVERRIDES_LIST[parameter]
    else:
        return value

def PkgConfigHavePkg(pkgname, tool='pkg-config'):
    if False:
        print('Hello World!')
    'Returns a bool whether the pkg-config package is installed.'
    if sys.platform == 'win32' or CrossCompiling() or (not LocateBinary(tool)):
        return False
    if tool == 'pkg-config':
        handle = os.popen(LocateBinary('pkg-config') + ' --silence-errors --modversion ' + pkgname)
    else:
        return bool(LocateBinary(tool) is not None)
    result = handle.read().strip()
    returnval = handle.close()
    if returnval is not None and returnval != 0:
        return False
    return bool(len(result) > 0)

def PkgConfigGetLibs(pkgname, tool='pkg-config'):
    if False:
        print('Hello World!')
    'Returns a list of libs for the package, prefixed by -l.'
    if sys.platform == 'win32' or CrossCompiling() or (not LocateBinary(tool)):
        return []
    if tool == 'pkg-config':
        handle = os.popen(LocateBinary('pkg-config') + ' --silence-errors --libs-only-l ' + pkgname)
    elif tool == 'fltk-config':
        handle = os.popen(LocateBinary('fltk-config') + ' --ldstaticflags')
    else:
        handle = os.popen(LocateBinary(tool) + ' --libs')
    result = handle.read().strip()
    handle.close()
    libs = []
    r = result.split(' ')
    ri = 0
    while ri < len(r):
        l = r[ri]
        if l.startswith('-l') or l.startswith('/'):
            libs.append(l)
        elif l == '-framework':
            libs.append(l)
            ri += 1
            libs.append(r[ri])
        ri += 1
    return libs

def PkgConfigGetIncDirs(pkgname, tool='pkg-config'):
    if False:
        while True:
            i = 10
    'Returns a list of includes for the package, NOT prefixed by -I.'
    if sys.platform == 'win32' or CrossCompiling() or (not LocateBinary(tool)):
        return []
    if tool == 'pkg-config':
        handle = os.popen(LocateBinary('pkg-config') + ' --silence-errors --cflags-only-I ' + pkgname)
    else:
        handle = os.popen(LocateBinary(tool) + ' --cflags')
    result = handle.read().strip()
    handle.close()
    if len(result) == 0:
        return []
    dirs = []
    for opt in result.split(' '):
        if opt.startswith('-I'):
            inc_dir = opt.replace('-I', '').replace('"', '').strip()
            if inc_dir != '/usr/include' and inc_dir != '/usr/include/':
                dirs.append(inc_dir)
    return dirs

def PkgConfigGetLibDirs(pkgname, tool='pkg-config'):
    if False:
        while True:
            i = 10
    'Returns a list of library paths for the package, NOT prefixed by -L.'
    if sys.platform == 'win32' or CrossCompiling() or (not LocateBinary(tool)):
        return []
    if tool == 'pkg-config':
        handle = os.popen(LocateBinary('pkg-config') + ' --silence-errors --libs-only-L ' + pkgname)
    elif tool == 'wx-config' or tool == 'ode-config':
        return []
    else:
        handle = os.popen(LocateBinary(tool) + ' --ldflags')
    result = handle.read().strip()
    handle.close()
    if len(result) == 0:
        return []
    libs = []
    for l in result.split(' '):
        if l.startswith('-L'):
            libs.append(l.replace('-L', '').replace('"', '').strip())
    return libs

def PkgConfigGetDefSymbols(pkgname, tool='pkg-config'):
    if False:
        print('Hello World!')
    'Returns a dictionary of preprocessor definitions.'
    if sys.platform == 'win32' or CrossCompiling() or (not LocateBinary(tool)):
        return []
    if tool == 'pkg-config':
        handle = os.popen(LocateBinary('pkg-config') + ' --silence-errors --cflags ' + pkgname)
    else:
        handle = os.popen(LocateBinary(tool) + ' --cflags')
    result = handle.read().strip()
    handle.close()
    if len(result) == 0:
        return {}
    defs = {}
    for l in result.split(' '):
        if l.startswith('-D'):
            d = l.replace('-D', '').replace('"', '').strip().split('=')
            if d[0] in ('NDEBUG', '_DEBUG'):
                if GetVerbose():
                    print('Ignoring %s flag provided by %s' % (l, tool))
            elif len(d) == 1:
                defs[d[0]] = ''
            else:
                defs[d[0]] = d[1]
    return defs

def PkgConfigEnable(opt, pkgname, tool='pkg-config'):
    if False:
        print('Hello World!')
    'Adds the libraries and includes to IncDirectory, LibName and LibDirectory.'
    for i in PkgConfigGetIncDirs(pkgname, tool):
        IncDirectory(opt, i)
    for i in PkgConfigGetLibDirs(pkgname, tool):
        LibDirectory(opt, i)
    for i in PkgConfigGetLibs(pkgname, tool):
        LibName(opt, i)
    for (i, j) in PkgConfigGetDefSymbols(pkgname, tool).items():
        DefSymbol(opt, i, j)

def LocateLibrary(lib, lpath=[], prefer_static=False):
    if False:
        for i in range(10):
            print('nop')
    'Searches for the library in the search path, returning its path if found,\n    or None if it was not found.'
    target = GetTarget()
    if prefer_static and target != 'windows':
        for dir in lpath:
            if os.path.isfile(os.path.join(dir, 'lib%s.a' % lib)):
                return os.path.join(dir, 'lib%s.a' % lib)
    for dir in lpath:
        if target == 'windows':
            if os.path.isfile(os.path.join(dir, lib + '.lib')):
                return os.path.join(dir, lib + '.lib')
        elif target == 'darwin' and os.path.isfile(os.path.join(dir, 'lib%s.dylib' % lib)):
            return os.path.join(dir, 'lib%s.dylib' % lib)
        elif target != 'darwin' and os.path.isfile(os.path.join(dir, 'lib%s.so' % lib)):
            return os.path.join(dir, 'lib%s.so' % lib)
        elif os.path.isfile(os.path.join(dir, 'lib%s.a' % lib)):
            return os.path.join(dir, 'lib%s.a' % lib)
    return None

def SystemLibraryExists(lib):
    if False:
        print('Hello World!')
    result = LocateLibrary(lib, SYS_LIB_DIRS)
    if result is not None:
        return True
    if GetHost() == 'android' and GetTarget() == 'android':
        return 'lib%s.so' % lib in ANDROID_SYS_LIBS
    return False

def ChooseLib(libs, thirdparty=None):
    if False:
        print('Hello World!')
    ' Chooses a library from the parameters, in order of preference. Returns the first if none of them were found. '
    lpath = []
    if thirdparty is not None:
        lpath.append(os.path.join(GetThirdpartyDir(), thirdparty.lower(), 'lib'))
    lpath += SYS_LIB_DIRS
    for l in libs:
        libname = l
        if l.startswith('lib'):
            libname = l[3:]
        if LocateLibrary(libname, lpath):
            return libname
    if len(libs) > 0:
        if VERBOSE:
            print(ColorText('cyan', "Couldn't find any of the libraries " + ', '.join(libs)))
        return libs[0]

def SmartPkgEnable(pkg, pkgconfig=None, libs=None, incs=None, defs=None, framework=None, target_pkg=None, tool='pkg-config', thirdparty_dir=None):
    if False:
        i = 10
        return i + 15
    global PKG_LIST_ALL
    if pkg in PkgListGet() and PkgSkip(pkg):
        return
    if target_pkg == '' or target_pkg is None:
        target_pkg = pkg
    if pkgconfig == '':
        pkgconfig = None
    if framework == '':
        framework = None
    if libs is None or libs == '':
        libs = ()
    elif isinstance(libs, str):
        libs = (libs,)
    if incs is None or incs == '':
        incs = ()
    elif isinstance(incs, str):
        incs = (incs,)
    if defs is None or defs == '' or len(defs) == 0:
        defs = {}
    elif isinstance(incs, str):
        defs = {defs: ''}
    elif isinstance(incs, list) or isinstance(incs, tuple) or isinstance(incs, set):
        olddefs = defs
        defs = {}
        for d in olddefs:
            defs[d] = ''
    custom_loc = PkgHasCustomLocation(pkg)
    if not thirdparty_dir:
        thirdparty_dir = pkg.lower()
    pkg_dir = os.path.join(GetThirdpartyDir(), thirdparty_dir)
    if not custom_loc and os.path.isdir(pkg_dir):
        if framework and os.path.isdir(os.path.join(pkg_dir, framework + '.framework')):
            FrameworkDirectory(target_pkg, pkg_dir)
            LibName(target_pkg, '-framework ' + framework)
            return
        inc_dir = os.path.join(pkg_dir, 'include')
        if os.path.isdir(inc_dir):
            IncDirectory(target_pkg, inc_dir)
            for i in incs:
                if os.path.isdir(os.path.join(inc_dir, i)):
                    IncDirectory(target_pkg, os.path.join(inc_dir, i))
        lib_dir = os.path.join(pkg_dir, 'lib')
        lpath = [lib_dir]
        if not PkgSkip('PYTHON'):
            py_lib_dir = os.path.join(lib_dir, SDK['PYTHONVERSION'])
            if os.path.isdir(py_lib_dir):
                lpath.append(py_lib_dir)
        if tool is not None and os.path.isfile(os.path.join(pkg_dir, 'bin', tool)):
            tool = os.path.join(pkg_dir, 'bin', tool)
            for i in PkgConfigGetLibs(None, tool):
                if i.startswith('-l'):
                    libname = i[2:]
                    location = LocateLibrary(libname, lpath, prefer_static=True)
                    if location is not None:
                        LibName(target_pkg, location)
                    else:
                        print(GetColor('cyan') + "Couldn't find library lib" + libname + ' in thirdparty directory ' + pkg.lower() + GetColor())
                        LibName(target_pkg, i)
                else:
                    LibName(target_pkg, i)
            for (i, j) in PkgConfigGetDefSymbols(None, tool).items():
                DefSymbol(target_pkg, i, j)
            return
        for l in libs:
            libname = l
            if l.startswith('lib'):
                libname = l[3:]
            location = LocateLibrary(libname, lpath, prefer_static=True)
            if location is not None:
                if location.endswith('.so') or location.endswith('.dylib'):
                    location = os.path.join(GetOutputDir(), 'lib', os.path.basename(location))
                LibName(target_pkg, location)
            else:
                location = LocateLibrary('panda' + libname, lpath, prefer_static=True)
                if location is not None:
                    if location.endswith('.so') or location.endswith('.dylib'):
                        location = os.path.join(GetOutputDir(), 'lib', os.path.basename(location))
                    LibName(target_pkg, location)
                else:
                    print(GetColor('cyan') + "Couldn't find library lib" + libname + ' in thirdparty directory ' + thirdparty_dir + GetColor())
        for (d, v) in defs.values():
            DefSymbol(target_pkg, d, v)
        return
    elif not custom_loc and GetHost() == 'darwin' and (framework is not None):
        prefix = SDK['MACOSX']
        if os.path.isdir(prefix + '/Library/Frameworks/%s.framework' % framework) or os.path.isdir(prefix + '/System/Library/Frameworks/%s.framework' % framework) or os.path.isdir(prefix + '/Developer/Library/Frameworks/%s.framework' % framework) or os.path.isdir(prefix + '/Users/%s/System/Library/Frameworks/%s.framework' % (getpass.getuser(), framework)):
            LibName(target_pkg, '-framework ' + framework)
            for (d, v) in defs.values():
                DefSymbol(target_pkg, d, v)
            return
        elif VERBOSE:
            print(ColorText('cyan', "Couldn't find the framework %s" % framework))
    elif not custom_loc and LocateBinary(tool) is not None and (tool != 'pkg-config' or pkgconfig is not None):
        if isinstance(pkgconfig, str) or tool != 'pkg-config':
            if PkgConfigHavePkg(pkgconfig, tool):
                return PkgConfigEnable(target_pkg, pkgconfig, tool)
        else:
            have_all_pkgs = True
            for pc in pkgconfig:
                if PkgConfigHavePkg(pc, tool):
                    PkgConfigEnable(target_pkg, pc, tool)
                else:
                    have_all_pkgs = False
            if have_all_pkgs:
                return
    if not custom_loc and pkgconfig is not None and (not libs):
        if pkg in PkgListGet():
            Warn('Could not locate pkg-config package %s, excluding from build' % pkgconfig)
            PkgDisable(pkg)
        else:
            Error('Could not locate pkg-config package %s, aborting build' % pkgconfig)
    else:
        have_pkg = True
        for l in libs:
            libname = l
            if l.startswith('lib'):
                libname = l[3:]
            if custom_loc:
                lpath = [dir for (ppkg, dir) in LIBDIRECTORIES if pkg == ppkg]
                location = LocateLibrary(libname, lpath)
                if location is not None:
                    LibName(target_pkg, location)
                else:
                    have_pkg = False
                    print(GetColor('cyan') + "Couldn't find library lib" + libname + GetColor())
            elif SystemLibraryExists(libname):
                LibName(target_pkg, '-l' + libname)
            else:
                lpath = [dir for (ppkg, dir) in LIBDIRECTORIES if pkg == ppkg or ppkg == 'ALWAYS']
                location = LocateLibrary(libname, lpath)
                if location is not None:
                    LibName(target_pkg, '-l' + libname)
                else:
                    have_pkg = False
                    if VERBOSE or custom_loc:
                        print(GetColor('cyan') + "Couldn't find library lib" + libname + GetColor())
        incdirs = []
        if not custom_loc:
            incdirs += list(SYS_INC_DIRS)
        for (ppkg, pdir) in INCDIRECTORIES[:]:
            if pkg == ppkg or (ppkg == 'ALWAYS' and (not custom_loc)):
                incdirs.append(pdir)
                if custom_loc and pkg != target_pkg:
                    IncDirectory(target_pkg, pdir)
        for i in incs:
            incdir = None
            for dir in incdirs:
                if len(glob.glob(os.path.join(dir, i))) > 0:
                    incdir = sorted(glob.glob(os.path.join(dir, i)))[-1]
            if incdir is None and (i.endswith('/Dense') or i.endswith('.h')):
                have_pkg = False
                if VERBOSE or custom_loc:
                    print(GetColor('cyan') + "Couldn't find header file " + i + GetColor())
            if incdir is not None and os.path.isdir(incdir):
                IncDirectory(target_pkg, incdir)
        if not have_pkg:
            if custom_loc:
                Error('Could not locate thirdparty package %s in specified directory, aborting build' % pkg.lower())
            elif pkg in PkgListGet():
                Warn('Could not locate thirdparty package %s, excluding from build' % pkg.lower())
                PkgDisable(pkg)
            else:
                Error('Could not locate thirdparty package %s, aborting build' % pkg.lower())
SDK = {}

def GetSdkDir(sdkname, sdkkey=None):
    if False:
        for i in range(10):
            print('nop')
    sdkbase = 'sdks'
    if 'MAKEPANDA_SDKS' in os.environ:
        sdkbase = os.environ['MAKEPANDA_SDKS']
    sdir = sdkbase[:]
    target = GetTarget()
    target_arch = GetTargetArch()
    if target == 'windows':
        if target_arch == 'x64':
            sdir += '/win64'
        else:
            sdir += '/win32'
    elif target == 'linux':
        sdir += '/linux'
        sdir += platform.architecture()[0][:2]
    elif target == 'darwin':
        sdir += '/macosx'
    sdir += '/' + sdkname
    if not os.path.isdir(sdir):
        sdir = sdkbase + '/' + sdir
        if target == 'linux':
            sdir += '-linux'
            sdir += platform.architecture()[0][:2]
        elif target == 'darwin':
            sdir += '-osx'
    if sdkkey and os.path.isdir(sdir):
        SDK[sdkkey] = sdir
    return sdir

def SdkLocateDirectX(strMode='default'):
    if False:
        print('Hello World!')
    if GetHost() != 'windows':
        return
    if strMode == 'default':
        GetSdkDir('directx9', 'DX9')
        if 'DX9' not in SDK:
            strMode = 'latest'
    if strMode == 'latest':
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Wow6432Node\\Microsoft\\DirectX\\Microsoft DirectX SDK (June 2010)', 'InstallPath')
            if dir != 0:
                print('Using DirectX SDK June 2010')
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Microsoft\\DirectX\\Microsoft DirectX SDK (June 2010)', 'InstallPath')
            if dir != 0:
                print('Using DirectX SDK June 2010')
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            dir = 'C:/Program Files (x86)/Microsoft DirectX SDK (June 2010)'
            if os.path.isdir(dir):
                print('Using DirectX SDK June 2010')
                SDK['DX9'] = dir
        if 'DX9' not in SDK:
            dir = 'C:/Program Files/Microsoft DirectX SDK (June 2010)'
            if os.path.isdir(dir):
                print('Using DirectX SDK June 2010')
                SDK['DX9'] = dir
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Wow6432Node\\Microsoft\\DirectX\\Microsoft DirectX SDK (August 2009)', 'InstallPath')
            if dir != 0:
                print('Using DirectX SDK Aug 2009')
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Microsoft\\DirectX\\Microsoft DirectX SDK (August 2009)', 'InstallPath')
            if dir != 0:
                print('Using DirectX SDK Aug 2009')
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        archStr = GetTargetArch()
        if 'DX9' not in SDK:
            uninstaller = 'SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall'
            for subdir in ListRegistryKeys(uninstaller):
                if subdir[0] == '{':
                    dir = GetRegistryKey(uninstaller + '\\' + subdir, 'InstallLocation')
                    if dir != 0:
                        if 'DX9' not in SDK and os.path.isfile(dir + '\\Include\\d3d9.h') and os.path.isfile(dir + '\\Include\\d3dx9.h') and os.path.isfile(dir + '\\Include\\dxsdkver.h') and os.path.isfile(dir + '\\Lib\\' + archStr + '\\d3d9.lib') and os.path.isfile(dir + '\\Lib\\' + archStr + '\\d3dx9.lib'):
                            SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            return
    elif strMode == 'jun2010':
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Wow6432Node\\Microsoft\\DirectX\\Microsoft DirectX SDK (June 2010)', 'InstallPath')
            if dir != 0:
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Microsoft\\DirectX\\Microsoft DirectX SDK (June 2010)', 'InstallPath')
            if dir != 0:
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            dir = 'C:/Program Files (x86)/Microsoft DirectX SDK (June 2010)'
            if os.path.isdir(dir):
                SDK['DX9'] = dir
        if 'DX9' not in SDK:
            dir = 'C:/Program Files/Microsoft DirectX SDK (June 2010)'
            if os.path.isdir(dir):
                SDK['DX9'] = dir
        if 'DX9' not in SDK:
            exit("Couldn't find DirectX June2010 SDK")
        else:
            print('Found DirectX SDK June 2010')
    elif strMode == 'aug2009':
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Wow6432Node\\Microsoft\\DirectX\\Microsoft DirectX SDK (August 2009)', 'InstallPath')
            if dir != 0:
                print('Found DirectX SDK Aug 2009')
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            dir = GetRegistryKey('SOFTWARE\\Microsoft\\DirectX\\Microsoft DirectX SDK (August 2009)', 'InstallPath')
            if dir != 0:
                print('Found DirectX SDK Aug 2009')
                SDK['DX9'] = dir.replace('\\', '/').rstrip('/')
        if 'DX9' not in SDK:
            exit("Couldn't find DirectX Aug 2009 SDK")
    if 'DX9' in SDK:
        SDK['DIRECTCAM'] = SDK['DX9']

def SdkLocateMaya():
    if False:
        print('Hello World!')
    for (ver, key) in MAYAVERSIONINFO:
        if PkgSkip(ver) == 0 and ver not in SDK:
            GetSdkDir(ver.lower().replace('x', ''), ver)
            if not ver in SDK:
                if GetHost() == 'windows':
                    for dev in ['Alias|Wavefront', 'Alias', 'Autodesk']:
                        fullkey = 'SOFTWARE\\' + dev + '\\Maya\\' + key + '\\Setup\\InstallPath'
                        res = GetRegistryKey(fullkey, 'MAYA_INSTALL_LOCATION', override64=False)
                        if res != 0:
                            res = res.replace('\\', '/').rstrip('/')
                            SDK[ver] = res
                elif GetHost() == 'darwin':
                    ddir = '/Applications/Autodesk/maya' + key
                    if os.path.isdir(ddir):
                        SDK[ver] = ddir
                else:
                    if GetTargetArch() in ('x86_64', 'amd64'):
                        ddir1 = '/usr/autodesk/maya' + key + '-x64'
                        ddir2 = '/usr/aw/maya' + key + '-x64'
                    else:
                        ddir1 = '/usr/autodesk/maya' + key
                        ddir2 = '/usr/aw/maya' + key
                    if os.path.isdir(ddir1):
                        SDK[ver] = ddir1
                    elif os.path.isdir(ddir2):
                        SDK[ver] = ddir2

def SdkLocateMax():
    if False:
        return 10
    if GetHost() != 'windows':
        return
    for (version, key1, key2, subdir) in MAXVERSIONINFO:
        if PkgSkip(version) == 0:
            if version not in SDK:
                GetSdkDir('maxsdk' + version.lower()[3:], version)
                GetSdkDir('maxsdk' + version.lower()[3:], version + 'CS')
                if not version in SDK:
                    top = GetRegistryKey(key1, key2)
                    if top != 0:
                        SDK[version] = top + 'maxsdk'
                        if os.path.isdir(top + '\\' + subdir) != 0:
                            SDK[version + 'CS'] = top + subdir

def SdkLocatePython(prefer_thirdparty_python=False):
    if False:
        i = 10
        return i + 15
    if PkgSkip('PYTHON'):
        SDK['PYTHONEXEC'] = os.path.realpath(sys.executable)
        return
    abiflags = getattr(sys, 'abiflags', '')
    if GetTarget() == 'windows':
        if PkgHasCustomLocation('PYTHON'):
            sdkdir = FindOptDirectory('PYTHON')
            if sdkdir is None:
                exit('Could not find a Python installation using these command line options.')
        else:
            sdkdir = GetThirdpartyBase() + '/win-python'
            sdkdir += '%d.%d' % sys.version_info[:2]
            if GetOptimize() <= 2:
                sdkdir += '-dbg'
            if GetTargetArch() == 'x64':
                sdkdir += '-x64'
        sdkdir = sdkdir.replace('\\', '/')
        SDK['PYTHON'] = sdkdir
        SDK['PYTHONEXEC'] = SDK['PYTHON'] + '/python'
        if GetOptimize() <= 2:
            SDK['PYTHONEXEC'] += '_d.exe'
        else:
            SDK['PYTHONEXEC'] += '.exe'
        if not os.path.isfile(SDK['PYTHONEXEC']):
            exit('Could not find %s!' % SDK['PYTHONEXEC'])
        if GetOptimize() <= 2:
            py_dlls = glob.glob(SDK['PYTHON'] + '/python[0-9][0-9]_d.dll') + glob.glob(SDK['PYTHON'] + '/python[0-9][0-9][0-9]_d.dll')
        else:
            py_dlls = glob.glob(SDK['PYTHON'] + '/python[0-9][0-9].dll') + glob.glob(SDK['PYTHON'] + '/python[0-9][0-9][0-9].dll')
        if len(py_dlls) == 0:
            exit('Could not find the Python dll in %s.' % SDK['PYTHON'])
        elif len(py_dlls) > 1:
            exit('Found multiple Python dlls in %s.' % SDK['PYTHON'])
        py_dll = os.path.basename(py_dlls[0])
        py_dllver = py_dll.strip('.DHLNOPTY_dhlnopty')
        ver = py_dllver[0] + '.' + py_dllver[1:]
        SDK['PYTHONVERSION'] = 'python' + ver
        os.environ['PYTHONHOME'] = SDK['PYTHON']
        running_ver = '%d.%d' % sys.version_info[:2]
        if ver != running_ver:
            Warn('running makepanda with Python %s, but building Panda3D with Python %s.' % (running_ver, ver))
    elif CrossCompiling() or (prefer_thirdparty_python and os.path.isdir(os.path.join(GetThirdpartyDir(), 'python'))):
        tp_python = os.path.join(GetThirdpartyDir(), 'python')
        if GetTarget() == 'darwin':
            py_libs = glob.glob(tp_python + '/lib/libpython[0-9].[0-9].dylib') + glob.glob(tp_python + '/lib/libpython[0-9].[0-9][0-9].dylib')
        else:
            py_libs = glob.glob(tp_python + '/lib/libpython[0-9].[0-9].so') + glob.glob(tp_python + '/lib/libpython[0-9].[0-9][0-9].so')
        if len(py_libs) == 0:
            py_libs = glob.glob(tp_python + '/lib/libpython[0-9].[0-9].a') + glob.glob(tp_python + '/lib/libpython[0-9].[0-9][0-9].a')
        if len(py_libs) == 0:
            exit('Could not find the Python library in %s.' % tp_python)
        elif len(py_libs) > 1:
            exit('Found multiple Python libraries in %s.' % tp_python)
        py_lib = os.path.basename(py_libs[0])
        py_libver = py_lib.strip('.abdhilnopsty')
        SDK['PYTHONVERSION'] = 'python' + py_libver
        SDK['PYTHONEXEC'] = tp_python + '/bin/' + SDK['PYTHONVERSION']
        SDK['PYTHON'] = tp_python + '/include/' + SDK['PYTHONVERSION']
    elif GetTarget() == 'darwin' and (not PkgHasCustomLocation('PYTHON')):
        sysroot = SDK.get('MACOSX', '')
        version = locations.get_python_version()
        py_fwx = '{0}/System/Library/Frameworks/Python.framework/Versions/{1}'.format(sysroot, version)
        if not os.path.exists(py_fwx):
            py_fwx = '/Library/Frameworks/Python.framework/Versions/' + version
        if not os.path.exists(py_fwx):
            py_fwx = '/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/' + version
        if not os.path.exists(py_fwx):
            exit('Could not locate Python installation at %s' % py_fwx)
        SDK['PYTHON'] = py_fwx + '/Headers'
        SDK['PYTHONVERSION'] = 'python' + version + abiflags
        SDK['PYTHONEXEC'] = py_fwx + '/bin/python' + version
        PkgSetCustomLocation('PYTHON')
        IncDirectory('PYTHON', py_fwx + '/include')
        LibDirectory('PYTHON', py_fwx + '/lib')
    else:
        SDK['PYTHON'] = locations.get_python_inc()
        SDK['PYTHONVERSION'] = 'python' + locations.get_python_version() + abiflags
        SDK['PYTHONEXEC'] = os.path.realpath(sys.executable)
    if CrossCompiling():
        SDK['PYTHONEXEC'] = sys.executable
        host_version = 'python' + locations.get_python_version() + abiflags
        if SDK['PYTHONVERSION'] != host_version:
            exit('Host Python version (%s) must be the same as target Python version (%s)!' % (host_version, SDK['PYTHONVERSION']))
    if GetVerbose():
        print('Using Python %s build located at %s' % (SDK['PYTHONVERSION'][6:], SDK['PYTHON']))
    else:
        print('Using Python %s' % SDK['PYTHONVERSION'][6:])

def SdkLocateVisualStudio(version=(10, 0)):
    if False:
        while True:
            i = 10
    if GetHost() != 'windows':
        return
    try:
        msvcinfo = MSVCVERSIONINFO[version]
    except:
        exit("Couldn't get Visual Studio infomation with MSVC %s.%s version." % version)
    vsversion = msvcinfo['vsversion']
    vsversion_str = '%s.%s' % vsversion
    version_str = '%s.%s' % version
    vswhere_path = LocateBinary('vswhere.exe')
    if not vswhere_path:
        if sys.platform == 'cygwin':
            vswhere_path = '/cygdrive/c/Program Files/Microsoft Visual Studio/Installer/vswhere.exe'
        else:
            vswhere_path = '%s\\Microsoft Visual Studio\\Installer\\vswhere.exe' % GetProgramFiles()
        if not os.path.isfile(vswhere_path):
            vswhere_path = None
    if not vswhere_path:
        if sys.platform == 'cygwin':
            vswhere_path = '/cygdrive/c/Program Files (x86)/Microsoft Visual Studio/Installer/vswhere.exe'
        else:
            vswhere_path = '%s\\Microsoft Visual Studio\\Installer\\vswhere.exe' % GetProgramFiles_x86()
        if not os.path.isfile(vswhere_path):
            vswhere_path = None
    vsdir = 0
    if vswhere_path:
        min_vsversion = vsversion_str
        max_vsversion = '%s.%s' % (vsversion[0] + 1, 0)
        vswhere_cmd = ['vswhere.exe', '-legacy', '-property', 'installationPath', '-version', '[{},{})'.format(min_vsversion, max_vsversion)]
        handle = subprocess.Popen(vswhere_cmd, executable=vswhere_path, stdout=subprocess.PIPE)
        found_paths = handle.communicate()[0].splitlines()
        if found_paths:
            vsdir = found_paths[0].decode('utf-8') + '\\'
    if vsdir == 0:
        vsdir = GetRegistryKey('SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VS7', vsversion_str)
    vcdir = GetRegistryKey('SOFTWARE\\Microsoft\\VisualStudio\\SxS\\VC7', version_str)
    if vsdir != 0:
        SDK['VISUALSTUDIO'] = vsdir
    elif vcdir != 0 and vcdir[-4:] == '\\VC\\':
        vcdir = vcdir[:-3]
        SDK['VISUALSTUDIO'] = vcdir
    elif os.path.isfile('C:\\Program Files\\Microsoft Visual Studio %s\\VC\\bin\\cl.exe' % vsversion_str):
        SDK['VISUALSTUDIO'] = 'C:\\Program Files\\Microsoft Visual Studio %s\\' % vsversion_str
    elif os.path.isfile('C:\\Program Files (x86)\\Microsoft Visual Studio %s\\VC\\bin\\cl.exe' % vsversion_str):
        SDK['VISUALSTUDIO'] = 'C:\\Program Files (x86)\\Microsoft Visual Studio %s\\' % vsversion_str
    elif 'VCINSTALLDIR' in os.environ:
        vcdir = os.environ['VCINSTALLDIR']
        if vcdir[-3:] == '\\VC':
            vcdir = vcdir[:-2]
        elif vcdir[-4:] == '\\VC\\':
            vcdir = vcdir[:-3]
        SDK['VISUALSTUDIO'] = vcdir
    else:
        exit("Couldn't find %s.  To use a different version, use the --msvc-version option." % msvcinfo['vsname'])
    SDK['MSVC_VERSION'] = version
    SDK['VISUALSTUDIO_VERSION'] = vsversion
    if GetVerbose():
        print('Using %s located at %s' % (msvcinfo['vsname'], SDK['VISUALSTUDIO']))
    else:
        print('Using %s' % msvcinfo['vsname'])
    print('Using MSVC %s' % version_str)

def SdkLocateWindows(version=None):
    if False:
        print('Hello World!')
    if GetTarget() != 'windows' or GetHost() != 'windows':
        return
    if version:
        version = version.upper()
    if version == '10':
        version = '10.0'
    if version and version.startswith('10.') and (version.count('.') == 1) or version == '11':
        platsdk = GetRegistryKey('SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots', 'KitsRoot10')
        if not platsdk or not os.path.isdir(platsdk):
            platsdk = 'C:\\Program Files (x86)\\Windows Kits\\10\\'
        if platsdk and os.path.isdir(platsdk):
            min_version = (10, 0, 0)
            if version == '11':
                version = '10.0'
                min_version = (10, 0, 22000)
            incdirs = glob.glob(os.path.join(platsdk, 'Include', version + '.*.*'))
            max_version = ()
            for dir in incdirs:
                verstring = os.path.basename(dir)
                if not os.path.isdir(os.path.join(dir, 'ucrt')):
                    continue
                if not os.path.isdir(os.path.join(dir, 'shared')):
                    continue
                if not os.path.isdir(os.path.join(dir, 'um')):
                    continue
                if not os.path.isdir(os.path.join(platsdk, 'Lib', verstring, 'ucrt')):
                    continue
                if not os.path.isdir(os.path.join(platsdk, 'Lib', verstring, 'um')):
                    continue
                vertuple = tuple(map(int, verstring.split('.')))
                if vertuple > max_version and vertuple > min_version:
                    version = verstring
                    max_version = vertuple
            if not max_version:
                platsdk = None
    elif version and version.startswith('10.'):
        platsdk = GetRegistryKey('SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots', 'KitsRoot10')
        if not platsdk or not os.path.isdir(platsdk):
            platsdk = 'C:\\Program Files (x86)\\Windows Kits\\10\\'
        if version.count('.') == 2:
            version += '.0'
        if platsdk and (not os.path.isdir(os.path.join(platsdk, 'Include', version))):
            platsdk = None
    elif version == '8.1' or not version:
        platsdk = GetRegistryKey('SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots', 'KitsRoot81')
        if not platsdk or not os.path.isdir(platsdk):
            platsdk = 'C:\\Program Files (x86)\\Windows Kits\\8.1\\'
        if not version:
            if not os.path.isdir(platsdk):
                return SdkLocateWindows('7.1')
            version = '8.1'
    elif version == '8.0':
        platsdk = GetRegistryKey('SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots', 'KitsRoot')
    else:
        platsdk = GetRegistryKey('SOFTWARE\\Microsoft\\Microsoft SDKs\\Windows\\v' + version, 'InstallationFolder')
        DefSymbol('ALWAYS', '_USING_V110_SDK71_')
        if not platsdk or not os.path.isdir(platsdk):
            platsdk = GetProgramFiles() + '\\Microsoft SDKs\\Windows\\v' + version
            if not os.path.isdir(platsdk):
                if not version.endswith('A'):
                    return SdkLocateWindows(version + 'A')
                platsdk = None
    if not platsdk or not os.path.isdir(platsdk):
        exit("Couldn't find Windows SDK version %s.  To use a different version, use the --windows-sdk option." % version)
    if not platsdk.endswith('\\'):
        platsdk += '\\'
    SDK['MSPLATFORM'] = platsdk
    SDK['MSPLATFORM_VERSION'] = version
    if GetVerbose():
        print('Using Windows SDK %s located at %s' % (version, platsdk))
    else:
        print('Using Windows SDK %s' % version)

def SdkLocateMacOSX(archs=[]):
    if False:
        i = 10
        return i + 15
    if GetHost() != 'darwin':
        return
    handle = os.popen('xcode-select -print-path')
    xcode_dir = handle.read().strip().rstrip('/')
    handle.close()
    sdk_versions = []
    if 'arm64' not in archs:
        sdk_versions += ['10.13', '10.12']
    sdk_versions += ['14.0', '13.3', '13.1', '13.0', '12.3', '11.3', '11.1', '11.0']
    if 'arm64' not in archs:
        sdk_versions += ['10.15', '10.14']
    for version in sdk_versions:
        sdkname = 'MacOSX' + version
        if os.path.exists('/Library/Developer/CommandLineTools/SDKs/%s.sdk' % sdkname):
            SDK['MACOSX'] = '/Library/Developer/CommandLineTools/SDKs/%s.sdk' % sdkname
            return
        elif os.path.exists('/Developer/SDKs/%s.sdk' % sdkname):
            SDK['MACOSX'] = '/Developer/SDKs/%s.sdk' % sdkname
            return
        elif os.path.exists('/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/%s.sdk' % sdkname):
            SDK['MACOSX'] = '/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/%s.sdk' % sdkname
            return
        elif xcode_dir and os.path.exists('%s/Platforms/MacOSX.platform/Developer/SDKs/%s.sdk' % (xcode_dir, sdkname)):
            SDK['MACOSX'] = '%s/Platforms/MacOSX.platform/Developer/SDKs/%s.sdk' % (xcode_dir, sdkname)
            return
    exit("Couldn't find any suitable MacOSX SDK!")

def SdkLocateSpeedTree():
    if False:
        print('Hello World!')
    dir = GetSdkDir('speedtree')
    if not os.path.exists(dir):
        return
    speedtrees = []
    for dirname in os.listdir(dir):
        if dirname.startswith('SpeedTree SDK v'):
            version = dirname[15:].split()[0]
            version = tuple(map(int, version.split('.')))
            speedtrees.append((version, dirname))
    if not speedtrees:
        return
    speedtrees.sort()
    (version, dirname) = speedtrees[-1]
    SDK['SPEEDTREE'] = os.path.join(dir, dirname)
    SDK['SPEEDTREEAPI'] = 'OpenGL'
    SDK['SPEEDTREEVERSION'] = '%s.%s' % (version[0], version[1])

def SdkLocateAndroid():
    if False:
        for i in range(10):
            print('nop')
    'This actually locates the Android NDK, not the Android SDK.\n    NDK_ROOT must be set to its root directory.'
    global TOOLCHAIN_PREFIX
    if GetTarget() != 'android':
        return
    if ANDROID_API is None:
        SetTarget('android')
    api = ANDROID_API
    SDK['ANDROID_API'] = api
    abi = ANDROID_ABI
    SDK['ANDROID_ABI'] = abi
    SDK['ANDROID_TRIPLE'] = ANDROID_TRIPLE
    if GetHost() == 'android':
        prefix = os.environ.get('PREFIX', '/data/data/com.termux/files/usr')
        SDK['ANDROID_JAR'] = prefix + '/share/aapt/android.jar'
        return
    sdk_root = os.environ.get('ANDROID_HOME')
    if not sdk_root or not os.path.isdir(sdk_root):
        sdk_root = os.environ.get('ANDROID_SDK_ROOT')
        if not sdk_root and GetHost() == 'windows':
            sdk_root = os.path.expanduser(os.path.join('~', 'AppData', 'Local', 'Android', 'Sdk'))
        if not sdk_root:
            exit('ANDROID_SDK_ROOT must be set when compiling for Android!')
        elif not os.path.isdir(sdk_root):
            exit('Cannot find %s.  Please install Android SDK and set ANDROID_SDK_ROOT or ANDROID_HOME.' % sdk_root)
    if os.environ.get('NDK_ROOT') or os.environ.get('ANDROID_NDK_ROOT'):
        ndk_root = os.environ.get('ANDROID_NDK_ROOT')
        if not ndk_root or not os.path.isdir(ndk_root):
            ndk_root = os.environ.get('NDK_ROOT')
            if not ndk_root or not os.path.isdir(ndk_root):
                exit('Cannot find %s.  Please install Android NDK and set ANDROID_NDK_ROOT.' % ndk_root)
    else:
        ndk_root = os.path.join(sdk_root, 'ndk-bundle')
        if not os.path.isdir(os.path.join(ndk_root, 'toolchains')):
            exit('Cannot find the Android NDK.  Install it via the SDK manager or set the ANDROID_NDK_ROOT variable if you have installed it in a different location.')
    SDK['ANDROID_NDK'] = ndk_root
    prebuilt_dir = os.path.join(ndk_root, 'toolchains', 'llvm', 'prebuilt')
    if not os.path.isdir(prebuilt_dir):
        exit('Not found: %s (is the Android NDK installed?)' % prebuilt_dir)
    host_tag = GetHost() + '-x86'
    if host_64:
        host_tag += '_64'
    elif host_tag == 'windows-x86':
        host_tag = 'windows'
    prebuilt_dir = os.path.join(prebuilt_dir, host_tag)
    if host_tag == 'windows-x86_64' and (not os.path.isdir(prebuilt_dir)):
        host_tag = 'windows'
        prebuilt_dir = os.path.join(prebuilt_dir, host_tag)
    SDK['ANDROID_TOOLCHAIN'] = prebuilt_dir
    arch = GetTargetArch()
    for opt in (TOOLCHAIN_PREFIX + '4.9', arch + '-4.9', TOOLCHAIN_PREFIX + '4.8', arch + '-4.8'):
        if os.path.isdir(os.path.join(ndk_root, 'toolchains', opt)):
            SDK['ANDROID_GCC_TOOLCHAIN'] = os.path.join(ndk_root, 'toolchains', opt, 'prebuilt', host_tag)
            break
    TOOLCHAIN_PREFIX = ''
    if arch == 'armv7a':
        arch_dir = 'arch-arm'
    elif arch == 'aarch64':
        arch_dir = 'arch-arm64'
    else:
        arch_dir = 'arch-' + arch
    SDK['SYSROOT'] = os.path.join(ndk_root, 'platforms', 'android-%s' % api, arch_dir).replace('\\', '/')
    stdlibc = os.path.join(ndk_root, 'sources', 'cxx-stl', 'llvm-libc++')
    stl_lib = os.path.join(stdlibc, 'libs', abi, 'libc++_shared.so')
    CopyFile(os.path.join(GetOutputDir(), 'lib', 'libc++_shared.so'), stl_lib)
    if api < 21:
        LibName('ALWAYS', '-landroid_support')
    SDK['ANDROID_JAR'] = os.path.join(sdk_root, 'platforms', 'android-%s' % api, 'android.jar')
    if not os.path.isfile(SDK['ANDROID_JAR']):
        exit('Cannot find %s.  Install platform API level %s via the SDK manager or change the targeted API level with --target=android-#' % (SDK['ANDROID_JAR'], api))
    versions = []
    for version in os.listdir(os.path.join(sdk_root, 'build-tools')):
        match = re.match('([0-9]+)\\.([0-9]+)\\.([0-9]+)', version)
        if match:
            version_tuple = (int(match.group(1)), int(match.group(2)), int(match.group(3)))
            versions.append(version_tuple)
    versions.sort()
    if versions:
        version = versions[-1]
        SDK['ANDROID_BUILD_TOOLS'] = os.path.join(sdk_root, 'build-tools', '{0}.{1}.{2}'.format(*version))
    if GetHost() == 'windows':
        jdk_home = os.environ.get('JDK_HOME') or os.environ.get('JAVA_HOME')
        if not jdk_home:
            studio_path = GetRegistryKey('SOFTWARE\\Android Studio', 'Path', override64=False)
            if studio_path and os.path.isdir(studio_path):
                jdk_home = os.path.join(studio_path, 'jre')
        if not jdk_home or not os.path.isdir(jdk_home):
            exit('Cannot find JDK.  Please set JDK_HOME or JAVA_HOME.')
        javac = os.path.join(jdk_home, 'bin', 'javac.exe')
        if not os.path.isfile(javac):
            exit('Cannot find %s.  Install the JDK and set JDK_HOME or JAVA_HOME.' % javac)
        SDK['JDK'] = jdk_home

def SdkAutoDisableDirectX():
    if False:
        return 10
    for ver in DXVERSIONS + ['DIRECTCAM']:
        if PkgSkip(ver) == 0:
            if ver not in SDK:
                if GetHost() == 'windows':
                    WARNINGS.append('I cannot locate SDK for ' + ver)
                    WARNINGS.append('I have automatically added this command-line option: --no-' + ver.lower())
                PkgDisable(ver)
            else:
                WARNINGS.append('Using ' + ver + ' sdk: ' + SDK[ver])

def SdkAutoDisableMaya():
    if False:
        for i in range(10):
            print('nop')
    for (ver, key) in MAYAVERSIONINFO:
        if ver not in SDK and PkgSkip(ver) == 0:
            if GetHost() == 'windows':
                WARNINGS.append('The registry does not appear to contain a pointer to the ' + ver + ' SDK.')
            else:
                WARNINGS.append('I cannot locate SDK for ' + ver)
            WARNINGS.append('I have automatically added this command-line option: --no-' + ver.lower())
            PkgDisable(ver)

def SdkAutoDisableMax():
    if False:
        print('Hello World!')
    for (version, key1, key2, subdir) in MAXVERSIONINFO:
        if PkgSkip(version) == 0 and (version not in SDK or version + 'CS' not in SDK):
            if GetHost() == 'windows':
                if version in SDK:
                    WARNINGS.append('Your copy of ' + version + ' does not include the character studio SDK')
                else:
                    WARNINGS.append('The registry does not appear to contain a pointer to ' + version)
                WARNINGS.append('I have automatically added this command-line option: --no-' + version.lower())
            PkgDisable(version)

def SdkAutoDisableSpeedTree():
    if False:
        for i in range(10):
            print('nop')
    if 'SPEEDTREE' not in SDK and PkgSkip('SPEEDTREE') == 0:
        PkgDisable('SPEEDTREE')
        WARNINGS.append('I cannot locate SDK for SpeedTree')
        WARNINGS.append('I have automatically added this command-line option: --no-speedtree')

def AddToPathEnv(path, add):
    if False:
        for i in range(10):
            print('nop')
    if path in os.environ:
        if sys.platform == 'cygwin' and path != 'PATH':
            os.environ[path] = add + ';' + os.environ[path]
        else:
            os.environ[path] = add + os.pathsep + os.environ[path]
    else:
        os.environ[path] = add

def SetupVisualStudioEnviron():
    if False:
        return 10
    if 'VISUALSTUDIO' not in SDK:
        exit('Could not find Visual Studio install directory')
    if 'MSPLATFORM' not in SDK:
        exit('Could not find the Microsoft Platform SDK')
    if SDK['VISUALSTUDIO_VERSION'] >= (15, 0):
        try:
            vsver_file = open(os.path.join(SDK['VISUALSTUDIO'], 'VC\\Auxiliary\\Build\\Microsoft.VCToolsVersion.default.txt'), 'r')
            SDK['VCTOOLSVERSION'] = vsver_file.readline().strip()
            vcdir_suffix = 'VC\\Tools\\MSVC\\%s\\' % SDK['VCTOOLSVERSION']
        except:
            exit("Couldn't find tool version of %s." % MSVCVERSIONINFO[SDK['MSVC_VERSION']]['vsname'])
    else:
        vcdir_suffix = 'VC\\'
    os.environ['VCINSTALLDIR'] = SDK['VISUALSTUDIO'] + vcdir_suffix
    os.environ['WindowsSdkDir'] = SDK['MSPLATFORM']
    winsdk_ver = SDK['MSPLATFORM_VERSION']
    arch = GetTargetArch()
    bindir = ''
    libdir = ''
    if 'VCTOOLSVERSION' in SDK:
        bindir = 'Host' + GetHostArch().upper() + '\\' + arch
        libdir = arch
    else:
        if arch == 'x64':
            bindir = 'amd64'
            libdir = 'amd64'
        elif arch != 'x86':
            bindir = arch
            libdir = arch
        if arch != 'x86' and GetHostArch() == 'x86':
            bindir = 'x86_' + bindir
    vc_binpath = SDK['VISUALSTUDIO'] + vcdir_suffix + 'bin'
    binpath = os.path.join(vc_binpath, bindir)
    if not os.path.isfile(binpath + '\\cl.exe'):
        if arch == 'x64' and os.path.isfile(vc_binpath + '\\x86_amd64\\cl.exe'):
            binpath = '{0}\\x86_amd64;{0}'.format(vc_binpath)
        elif winsdk_ver.startswith('10.'):
            exit("Couldn't find compilers in %s.  You may need to install the Windows SDK 7.1 and the Visual C++ 2010 SP1 Compiler Update for Windows SDK 7.1." % binpath)
        else:
            exit("Couldn't find compilers in %s." % binpath)
    AddToPathEnv('PATH', binpath)
    AddToPathEnv('PATH', SDK['VISUALSTUDIO'] + 'Common7\\IDE')
    AddToPathEnv('INCLUDE', os.environ['VCINSTALLDIR'] + 'include')
    AddToPathEnv('INCLUDE', os.environ['VCINSTALLDIR'] + 'atlmfc\\include')
    AddToPathEnv('LIB', os.environ['VCINSTALLDIR'] + 'lib\\' + libdir)
    AddToPathEnv('LIB', os.environ['VCINSTALLDIR'] + 'atlmfc\\lib\\' + libdir)
    winsdk_ver = SDK['MSPLATFORM_VERSION']
    if winsdk_ver.startswith('10.'):
        AddToPathEnv('PATH', SDK['MSPLATFORM'] + 'bin\\' + arch)
        AddToPathEnv('PATH', SDK['MSPLATFORM'] + 'bin\\' + winsdk_ver + '\\' + arch)
        inc_dir = SDK['MSPLATFORM'] + 'Include\\' + winsdk_ver + '\\'
        lib_dir = SDK['MSPLATFORM'] + 'Lib\\' + winsdk_ver + '\\'
        AddToPathEnv('INCLUDE', inc_dir + 'shared')
        AddToPathEnv('INCLUDE', inc_dir + 'ucrt')
        AddToPathEnv('INCLUDE', inc_dir + 'um')
        AddToPathEnv('LIB', lib_dir + 'ucrt\\' + arch)
        AddToPathEnv('LIB', lib_dir + 'um\\' + arch)
    elif winsdk_ver == '8.1':
        AddToPathEnv('PATH', SDK['MSPLATFORM'] + 'bin\\' + arch)
        inc_dir = SDK['MSPLATFORM'] + 'Include\\'
        lib_dir = SDK['MSPLATFORM'] + 'Lib\\winv6.3\\'
        AddToPathEnv('INCLUDE', inc_dir + 'shared')
        AddToPathEnv('INCLUDE', inc_dir + 'ucrt')
        AddToPathEnv('INCLUDE', inc_dir + 'um')
        AddToPathEnv('LIB', lib_dir + 'ucrt\\' + arch)
        AddToPathEnv('LIB', lib_dir + 'um\\' + arch)
    else:
        AddToPathEnv('PATH', SDK['MSPLATFORM'] + 'bin')
        AddToPathEnv('INCLUDE', SDK['MSPLATFORM'] + 'include')
        AddToPathEnv('INCLUDE', SDK['MSPLATFORM'] + 'include\\atl')
        AddToPathEnv('INCLUDE', SDK['MSPLATFORM'] + 'include\\mfc')
        if arch != 'x64':
            AddToPathEnv('LIB', SDK['MSPLATFORM'] + 'lib')
            AddToPathEnv('PATH', SDK['VISUALSTUDIO'] + 'VC\\redist\\x86\\Microsoft.VC100.CRT')
            AddToPathEnv('PATH', SDK['VISUALSTUDIO'] + 'VC\\redist\\x86\\Microsoft.VC100.MFC')
        elif os.path.isdir(SDK['MSPLATFORM'] + 'lib\\x64'):
            AddToPathEnv('LIB', SDK['MSPLATFORM'] + 'lib\\x64')
        elif os.path.isdir(SDK['MSPLATFORM'] + 'lib\\amd64'):
            AddToPathEnv('LIB', SDK['MSPLATFORM'] + 'lib\\amd64')
        else:
            exit('Could not locate 64-bits libraries in Windows SDK directory!\nUsing directory: %s' % SDK['MSPLATFORM'])
    if winsdk_ver in ('7.1', '7.1A', '8.0', '8.1') and SDK['VISUALSTUDIO_VERSION'] >= (14, 0):
        win_kit = GetRegistryKey('SOFTWARE\\Microsoft\\Windows Kits\\Installed Roots', 'KitsRoot10')
        if not win_kit or not os.path.isdir(win_kit):
            win_kit = 'C:\\Program Files (x86)\\Windows Kits\\10\\'
        elif not win_kit.endswith('\\'):
            win_kit += '\\'
        for vnum in (10150, 10240, 10586, 14393, 15063, 16299, 17134, 17763, 18362, 19041, 20348, 22000):
            version = '10.0.{0}.0'.format(vnum)
            if os.path.isfile(win_kit + 'Include\\' + version + '\\ucrt\\assert.h'):
                print('Using Universal CRT %s' % version)
                break
        AddToPathEnv('LIB', '%s\\Lib\\%s\\ucrt\\%s' % (win_kit, version, arch))
        AddToPathEnv('INCLUDE', '%s\\Include\\%s\\ucrt' % (win_kit, version))
        CopyAllFiles(GetOutputDir() + '/bin/', win_kit + 'Redist\\ucrt\\DLLs\\' + arch + '\\')
INCDIRECTORIES = []
LIBDIRECTORIES = []
FRAMEWORKDIRECTORIES = []
LIBNAMES = []
DEFSYMBOLS = []

def IncDirectory(opt, dir):
    if False:
        i = 10
        return i + 15
    INCDIRECTORIES.append((opt, dir))

def LibDirectory(opt, dir):
    if False:
        i = 10
        return i + 15
    LIBDIRECTORIES.append((opt, dir))

def FrameworkDirectory(opt, dir):
    if False:
        print('Hello World!')
    FRAMEWORKDIRECTORIES.append((opt, dir))

def FindIncDirectory(opt):
    if False:
        return 10
    for (mod, dir) in INCDIRECTORIES:
        if mod == opt:
            return os.path.abspath(dir)

def FindLibDirectory(opt):
    if False:
        return 10
    for (mod, dir) in LIBDIRECTORIES:
        if mod == opt:
            return os.path.abspath(dir)

def FindOptDirectory(opt):
    if False:
        i = 10
        return i + 15
    include_dir = FindIncDirectory(opt)
    lib_dir = FindLibDirectory(opt)
    if include_dir and lib_dir:
        common_dir = os.path.commonprefix([include_dir, lib_dir])
        if common_dir:
            return os.path.abspath(common_dir)
    elif include_dir:
        return os.path.abspath(os.path.join(include_dir, os.pardir))
    elif lib_dir:
        return os.path.abspath(os.path.join(lib_dir, os.pardir))

def LibName(opt, name):
    if False:
        return 10
    if name.startswith(GetThirdpartyDir()):
        if not os.path.exists(name):
            WARNINGS.append(name + ' not found.  Skipping Package ' + opt)
            if opt in PkgListGet():
                if not PkgSkip(opt):
                    Warn('Could not locate thirdparty package %s, excluding from build' % opt.lower())
                    PkgDisable(opt)
                return
            else:
                Error('Could not locate thirdparty package %s, aborting build' % opt.lower())
    LIBNAMES.append((opt, name))

def DefSymbol(opt, sym, val=''):
    if False:
        for i in range(10):
            print('nop')
    DEFSYMBOLS.append((opt, sym, val))

def SetupBuildEnvironment(compiler):
    if False:
        return 10
    if GetVerbose():
        print('Using compiler: %s' % compiler)
        print('Host OS: %s' % GetHost())
        print('Host arch: %s' % GetHostArch())
    target = GetTarget()
    if target != 'android':
        print('Target OS: %s' % GetTarget())
    else:
        print('Target OS: %s (API level %d)' % (GetTarget(), ANDROID_API))
    print('Target arch: %s' % GetTargetArch())
    os.environ['LC_ALL'] = 'en_US.UTF-8'
    os.environ['LANGUAGE'] = 'en'
    if GetTarget() == 'android' and GetHost() != 'android':
        AddToPathEnv('PATH', os.path.join(SDK['ANDROID_TOOLCHAIN'], 'bin'))
        if 'ANDROID_BUILD_TOOLS' in SDK:
            AddToPathEnv('PATH', SDK['ANDROID_BUILD_TOOLS'])
        if 'JDK' in SDK:
            AddToPathEnv('PATH', os.path.join(SDK['JDK'], 'bin'))
            os.environ['JAVA_HOME'] = SDK['JDK']
    if compiler == 'MSVC':
        SetupVisualStudioEnviron()
    if compiler == 'GCC':
        global SYS_LIB_DIRS, SYS_INC_DIRS
        if sys.platform == 'darwin':
            SYS_LIB_DIRS.append(SDK['MACOSX'] + '/usr/lib')
        if not SDK.get('MACOSX'):
            local_lib = SDK.get('SYSROOT', '') + '/usr/local/lib'
            if os.path.isdir(local_lib):
                SYS_LIB_DIRS.append(local_lib)
        sysroot_flag = ''
        if SDK.get('MACOSX'):
            sysroot_flag = ' -isysroot ' + SDK['MACOSX']
        if GetTarget() == 'android':
            sysroot_flag = ' -target ' + ANDROID_TRIPLE
        cmd = GetCXX() + ' -print-search-dirs' + sysroot_flag
        handle = os.popen(cmd)
        for line in handle:
            if not line.startswith('libraries: ='):
                continue
            line = line[12:].strip()
            libdirs = line.split(':')
            while libdirs:
                libdir = os.path.normpath(libdirs.pop(0))
                if os.path.isdir(libdir):
                    if libdir not in SYS_LIB_DIRS:
                        SYS_LIB_DIRS.append(libdir)
                elif len(libdir) == 1:
                    libdirs[0] = libdir + ':' + libdirs[0]
                elif GetVerbose():
                    print('Ignoring non-existent library directory %s' % libdir)
        returnval = handle.close()
        if returnval is not None and returnval != 0:
            Warn('%s failed' % cmd)
            SYS_LIB_DIRS += [SDK.get('SYSROOT', '') + '/usr/lib']
        if target == 'android' and GetHost() == 'windows':
            libdir = SDK.get('SYSROOT', '') + '/usr/lib'
            if GetTargetArch() == 'x86_64':
                libdir += '64'
            SYS_LIB_DIRS += [libdir]
        cmd = GetCXX() + ' -x c++ -v -E ' + os.devnull
        cmd += sysroot_flag
        null = open(os.devnull, 'w')
        handle = subprocess.Popen(cmd, stdout=null, stderr=subprocess.PIPE, shell=True)
        scanning = False
        for line in handle.communicate()[1].splitlines():
            line = line.decode('utf-8', 'replace')
            if not scanning:
                if line.startswith('#include'):
                    scanning = True
                continue
            if sys.platform == 'win32':
                if not line.startswith(' '):
                    continue
            elif not line.startswith(' /'):
                continue
            line = line.strip()
            if line.endswith(' (framework directory)'):
                pass
            elif os.path.isdir(line):
                SYS_INC_DIRS.append(os.path.normpath(line))
            elif GetVerbose():
                print('Ignoring non-existent include directory %s' % line)
        if handle.returncode != 0 or not SYS_INC_DIRS:
            Warn('%s failed or did not produce the expected result' % cmd)
            sysroot = SDK.get('SYSROOT', '')
            SYS_INC_DIRS = [sysroot + '/usr/include', sysroot + '/usr/local/include']
            pcbsd_inc = sysroot + '/usr/PCBSD/local/include'
            if os.path.isdir(pcbsd_inc):
                SYS_INC_DIRS.append(pcbsd_inc)
        null.close()
        if GetVerbose():
            print('System library search path:')
            for dir in SYS_LIB_DIRS:
                print('  ' + dir)
            print('System include search path:')
            for dir in SYS_INC_DIRS:
                print('  ' + dir)
    if CrossCompiling():
        return
    builtdir = GetOutputDir()
    AddToPathEnv('PYTHONPATH', builtdir)
    AddToPathEnv('PANDA_PRC_DIR', os.path.join(builtdir, 'etc'))
    AddToPathEnv('PATH', os.path.join(builtdir, 'bin'))
    if GetHost() == 'windows':
        AddToPathEnv('PYTHONPATH', os.path.join(builtdir, 'bin'))
        AddToPathEnv('PATH', os.path.join(builtdir, 'plugins'))
    if GetHost() != 'windows':
        ldpath = os.environ.get('LD_LIBRARY_PATH', '').split(os.pathsep)
        if GetHost() == 'darwin':
            dyldpath = os.environ.get('DYLD_LIBRARY_PATH', '').split(os.pathsep)
        for i in ldpath[:]:
            if i.startswith('/usr/lib/panda3d') or i.startswith('/usr/local/panda'):
                ldpath.remove(i)
        if GetHost() == 'darwin':
            for i in dyldpath[:]:
                if i.startswith('/Applications/Panda3D') or i.startswith('/Developer/Panda3D'):
                    dyldpath.remove(i)
        ldpath.insert(0, os.path.join(builtdir, 'lib'))
        os.environ['LD_LIBRARY_PATH'] = os.pathsep.join(ldpath)
        if GetHost() == 'darwin':
            dyldpath.insert(0, os.path.join(builtdir, 'lib'))
            os.environ['DYLD_LIBRARY_PATH'] = os.pathsep.join(dyldpath)
            os.environ['PATH'] = os.path.join(builtdir, 'lib') + ':' + os.environ.get('PATH', '')
        if os.path.exists('/usr/PCBSD'):
            os.environ['LD_LIBRARY_PATH'] += os.pathsep + '/usr/PCBSD/local/lib'

def CopyFile(dstfile, srcfile):
    if False:
        i = 10
        return i + 15
    if dstfile[-1] == '/':
        dstfile += os.path.basename(srcfile)
    if NeedsBuild([dstfile], [srcfile]):
        if os.path.islink(srcfile):
            if os.path.isfile(dstfile) or os.path.islink(dstfile):
                print('Removing file %s' % dstfile)
                os.unlink(dstfile)
            elif os.path.isdir(dstfile):
                print('Removing directory %s' % dstfile)
                shutil.rmtree(dstfile)
            os.symlink(os.readlink(srcfile), dstfile)
        else:
            WriteBinaryFile(dstfile, ReadBinaryFile(srcfile))
        if sys.platform == 'cygwin' and os.path.splitext(dstfile)[1].lower() in ('.dll', '.exe'):
            os.chmod(dstfile, 493)
        JustBuilt([dstfile], [srcfile])

def CopyAllFiles(dstdir, srcdir, suffix=''):
    if False:
        print('Hello World!')
    for x in GetDirectoryContents(srcdir, ['*' + suffix]):
        CopyFile(dstdir + x, srcdir + x)

def CopyAllHeaders(dir, skip=[]):
    if False:
        return 10
    for filename in GetDirectoryContents(dir, ['*.h', '*.I', '*.T'], skip):
        srcfile = dir + '/' + filename
        dstfile = OUTPUTDIR + '/include/' + filename
        if NeedsBuild([dstfile], [srcfile]):
            WriteBinaryFile(dstfile, ReadBinaryFile(srcfile))
            JustBuilt([dstfile], [srcfile])

def CopyTree(dstdir, srcdir, omitVCS=True, exclude=()):
    if False:
        return 10
    if os.path.isdir(dstdir):
        source_entries = os.listdir(srcdir)
        for entry in source_entries:
            srcpth = os.path.join(srcdir, entry)
            dstpth = os.path.join(dstdir, entry)
            if entry in exclude:
                continue
            if os.path.islink(srcpth) or os.path.isfile(srcpth):
                if not omitVCS or entry not in VCS_FILES:
                    CopyFile(dstpth, srcpth)
            elif not omitVCS or entry not in VCS_DIRS:
                CopyTree(dstpth, srcpth)
        for entry in os.listdir(dstdir):
            if entry not in source_entries or entry in exclude:
                path = os.path.join(dstdir, entry)
                if os.path.islink(path) or os.path.isfile(path):
                    os.remove(path)
                elif os.path.isdir(path):
                    shutil.rmtree(path)
    else:
        if GetHost() == 'windows':
            srcdir = srcdir.replace('/', '\\')
            dstdir = dstdir.replace('/', '\\')
            cmd = 'xcopy /I/Y/E/Q "' + srcdir + '" "' + dstdir + '"'
            oscmd(cmd)
        elif subprocess.call(['cp', '-R', '-f', srcdir, dstdir]) != 0:
            exit('Copy failed.')
        for entry in exclude:
            path = os.path.join(dstdir, entry)
            if os.path.islink(path) or os.path.isfile(path):
                os.remove(path)
            elif os.path.isdir(path):
                shutil.rmtree(path)
        if omitVCS:
            DeleteVCS(dstdir)

def CopyPythonTree(dstdir, srcdir, threads=0):
    if False:
        for i in range(10):
            print('nop')
    if not os.path.isdir(dstdir):
        os.mkdir(dstdir)
    exclude_files = set(VCS_FILES)
    exclude_files.add('panda3d.py')
    for entry in os.listdir(srcdir):
        srcpth = os.path.join(srcdir, entry)
        dstpth = os.path.join(dstdir, entry)
        if os.path.isfile(srcpth):
            (base, ext) = os.path.splitext(entry)
            if entry not in exclude_files and ext not in SUFFIX_INC + ['.pyc', '.pyo']:
                if NeedsBuild([dstpth], [srcpth]):
                    WriteBinaryFile(dstpth, ReadBinaryFile(srcpth))
                    JustBuilt([dstpth], [srcpth])
        elif entry not in VCS_DIRS:
            CopyPythonTree(dstpth, srcpth, threads=threads)
cfg_parser = None

def GetMetadataValue(key):
    if False:
        while True:
            i = 10
    global cfg_parser
    if not cfg_parser:
        cfg_parser = configparser.ConfigParser()
        path = os.path.join(os.path.dirname(__file__), '..', 'setup.cfg')
        assert cfg_parser.read(path), 'Could not read setup.cfg file.'
    value = cfg_parser.get('metadata', key)
    if key == 'classifiers':
        value = value.strip().split('\n')
    return value

def ParsePandaVersion(fn):
    if False:
        print('Hello World!')
    try:
        f = open(fn, 'r')
        pattern = re.compile('^[ \t]*[#][ \t]*define[ \t]+PANDA_VERSION[ \t]+([0-9]+)[ \t]+([0-9]+)[ \t]+([0-9]+)')
        for line in f:
            match = pattern.match(line, 0)
            if match:
                f.close()
                return match.group(1) + '.' + match.group(2) + '.' + match.group(3)
        f.close()
    except:
        pass
    return '0.0.0'
RESOURCE_FILE_TEMPLATE = 'VS_VERSION_INFO VERSIONINFO\n FILEVERSION %(commaversion)s\n PRODUCTVERSION %(commaversion)s\n FILEFLAGSMASK 0x3fL\n FILEFLAGS %(debugflag)s\n FILEOS 0x40004L\n FILETYPE 0x2L\n FILESUBTYPE 0x0L\nBEGIN\n    BLOCK "StringFileInfo"\n    BEGIN\n        BLOCK "040904e4"\n        BEGIN\n            VALUE "FileDescription", "%(description)s\\0"\n            VALUE "FileVersion", "%(dotversion)s"\n            VALUE "LegalTrademarks", "\\0"\n            VALUE "MIMEType", "%(mimetype)s\\0"\n            VALUE "FileExtents", "%(extension)s\\0"\n            VALUE "FileOpenName", "%(filedesc)s\\0"\n            VALUE "OLESelfRegister", "\\0"\n            VALUE "OriginalFilename", "%(filename)s\\0"\n            VALUE "ProductName", "%(name)s %(version)s\\0"\n            VALUE "ProductVersion", "%(dotversion)s"\n        END\n    END\n    BLOCK "VarFileInfo"\n    BEGIN\n        VALUE "Translation", 0x409, 1252\n    END\nEND\n'

def GenerateResourceFile(**kwargs):
    if False:
        return 10
    if 'debugflag' not in kwargs:
        if GetOptimize() <= 2:
            kwargs['debugflag'] = '0x1L'
        else:
            kwargs['debugflag'] = '0x0L'
    kwargs['dotversion'] = kwargs['version']
    if len(kwargs['dotversion'].split('.')) == 3:
        kwargs['dotversion'] += '.0'
    if 'commaversion' not in kwargs:
        kwargs['commaversion'] = kwargs['dotversion'].replace('.', ',')
    rcdata = ''
    if 'noinclude' not in kwargs:
        rcdata += '#define APSTUDIO_READONLY_SYMBOLS\n'
        rcdata += '#include "winresrc.h"\n'
        rcdata += '#undef APSTUDIO_READONLY_SYMBOLS\n'
    rcdata += RESOURCE_FILE_TEMPLATE % kwargs
    if 'icon' in kwargs:
        rcdata += '\nICON_FILE       ICON    "%s"\n' % kwargs['icon']
    return rcdata

def WriteResourceFile(basename, **kwargs):
    if False:
        return 10
    if not basename.endswith('.rc'):
        basename += '.rc'
    basename = GetOutputDir() + '/include/' + basename
    ConditionalWriteFile(basename, GenerateResourceFile(**kwargs))
    return basename

def GenerateEmbeddedStringFile(string_name, data):
    if False:
        for i in range(10):
            print('nop')
    yield ('extern const char %s[] = {\n' % string_name)
    i = 0
    for byte in data:
        if i == 0:
            yield ' '
        yield (' 0x%02x,' % byte)
        i += 1
        if i >= 12:
            yield '\n'
            i = 0
    yield '\n};\n'

def WriteEmbeddedStringFile(basename, inputs, string_name=None):
    if False:
        while True:
            i = 10
    if os.path.splitext(basename)[1] not in SUFFIX_INC:
        basename += '.cxx'
    target = GetOutputDir() + '/tmp/' + basename
    if string_name is None:
        string_name = os.path.basename(os.path.splitext(target)[0])
        string_name = string_name.replace('-', '_')
    data = bytearray()
    for input in inputs:
        fp = open(input, 'rb')
        if os.path.splitext(input)[1] in SUFFIX_INC:
            line = '#line 1 "%s"\n' % input
            data += bytearray(line.encode('ascii', 'replace'))
        data += bytearray(fp.read())
        fp.close()
    data.append(0)
    output = ''.join(GenerateEmbeddedStringFile(string_name, data))
    ConditionalWriteFile(target, output)
    return target
ORIG_EXT = {}
PYABI_SPECIFIC = set()
WARNED_FILES = set()

def GetOrigExt(x):
    if False:
        for i in range(10):
            print('nop')
    return ORIG_EXT[x]

def SetOrigExt(x, v):
    if False:
        return 10
    ORIG_EXT[x] = v

def GetExtensionSuffix():
    if False:
        print('Hello World!')
    if GetTarget() == 'windows':
        if GetTargetArch() == 'x64':
            return '.cp%d%d-win_amd64.pyd' % sys.version_info[:2]
        else:
            return '.cp%d%d-win32.pyd' % sys.version_info[:2]
    elif CrossCompiling():
        return '.{0}.so'.format(GetPythonABI())
    else:
        import _imp
        return _imp.extension_suffixes()[0]

def GetPythonABI():
    if False:
        return 10
    if not CrossCompiling():
        soabi = locations.get_config_var('SOABI')
        if soabi:
            return soabi
    return 'cpython-%d%d' % sys.version_info[:2]

def CalcLocation(fn, ipath):
    if False:
        return 10
    if fn.startswith('panda3d/') and fn.endswith('.py'):
        return OUTPUTDIR + '/' + fn
    if fn.endswith('.class'):
        return OUTPUTDIR + '/classes/' + fn
    if fn.count('/'):
        return fn
    dllext = ''
    target = GetTarget()
    if GetOptimize() <= 2 and target == 'windows':
        dllext = '_d'
    if fn == 'AndroidManifest.xml':
        return OUTPUTDIR + '/' + fn
    if fn == 'classes.dex':
        return OUTPUTDIR + '/' + fn
    if fn.endswith('.cxx'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.I'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.h'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.c'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.py'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.yxx'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.lxx'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.xml'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.java'):
        return CxxFindSource(fn, ipath)
    if fn.endswith('.egg'):
        return OUTPUTDIR + '/models/' + fn
    if fn.endswith('.egg.pz'):
        return OUTPUTDIR + '/models/' + fn
    if fn.endswith('.pyd'):
        return OUTPUTDIR + '/panda3d/' + fn[:-4] + GetExtensionSuffix()
    if target == 'windows':
        if fn.endswith('.def'):
            return CxxFindSource(fn, ipath)
        if fn.endswith('.rc'):
            return CxxFindSource(fn, ipath)
        if fn.endswith('.idl'):
            return CxxFindSource(fn, ipath)
        if fn.endswith('.obj'):
            return OUTPUTDIR + '/tmp/' + fn
        if fn.endswith('.res'):
            return OUTPUTDIR + '/tmp/' + fn
        if fn.endswith('.tlb'):
            return OUTPUTDIR + '/tmp/' + fn
        if fn.endswith('.dll'):
            return OUTPUTDIR + '/bin/' + fn[:-4] + dllext + '.dll'
        if fn.endswith('.ocx'):
            return OUTPUTDIR + '/plugins/' + fn[:-4] + dllext + '.ocx'
        if fn.endswith('.mll'):
            return OUTPUTDIR + '/plugins/' + fn[:-4] + dllext + '.mll'
        if fn.endswith('.dlo'):
            return OUTPUTDIR + '/plugins/' + fn[:-4] + dllext + '.dlo'
        if fn.endswith('.dli'):
            return OUTPUTDIR + '/plugins/' + fn[:-4] + dllext + '.dli'
        if fn.endswith('.dle'):
            return OUTPUTDIR + '/plugins/' + fn[:-4] + dllext + '.dle'
        if fn.endswith('.plugin'):
            return OUTPUTDIR + '/plugins/' + fn[:-7] + dllext + '.dll'
        if fn.endswith('.exe'):
            return OUTPUTDIR + '/bin/' + fn
        if fn.endswith('.p3d'):
            return OUTPUTDIR + '/bin/' + fn
        if fn.endswith('.lib'):
            return OUTPUTDIR + '/lib/' + fn[:-4] + dllext + '.lib'
        if fn.endswith('.ilb'):
            return OUTPUTDIR + '/tmp/' + fn[:-4] + dllext + '.lib'
    elif target == 'darwin':
        if fn.endswith('.mm'):
            return CxxFindSource(fn, ipath)
        if fn.endswith('.r'):
            return CxxFindSource(fn, ipath)
        if fn.endswith('.plist'):
            return CxxFindSource(fn, ipath)
        if fn.endswith('.obj'):
            return OUTPUTDIR + '/tmp/' + fn[:-4] + '.o'
        if fn.endswith('.dll'):
            return OUTPUTDIR + '/lib/' + fn[:-4] + '.dylib'
        if fn.endswith('.mll'):
            return OUTPUTDIR + '/plugins/' + fn
        if fn.endswith('.exe'):
            return OUTPUTDIR + '/bin/' + fn[:-4]
        if fn.endswith('.p3d'):
            return OUTPUTDIR + '/bin/' + fn[:-4]
        if fn.endswith('.lib'):
            return OUTPUTDIR + '/lib/' + fn[:-4] + '.a'
        if fn.endswith('.ilb'):
            return OUTPUTDIR + '/tmp/' + fn[:-4] + '.a'
        if fn.endswith('.rsrc'):
            return OUTPUTDIR + '/tmp/' + fn
        if fn.endswith('.plugin'):
            return OUTPUTDIR + '/plugins/' + fn
        if fn.endswith('.app'):
            return OUTPUTDIR + '/bin/' + fn
    else:
        if fn.endswith('.obj'):
            return OUTPUTDIR + '/tmp/' + fn[:-4] + '.o'
        if fn.endswith('.dll'):
            return OUTPUTDIR + '/lib/' + fn[:-4] + '.so'
        if fn.endswith('.mll'):
            return OUTPUTDIR + '/plugins/' + fn
        if fn.endswith('.plugin'):
            return OUTPUTDIR + '/plugins/' + fn[:-7] + dllext + '.so'
        if fn.endswith('.exe'):
            return OUTPUTDIR + '/bin/' + fn[:-4]
        if fn.endswith('.p3d'):
            return OUTPUTDIR + '/bin/' + fn[:-4]
        if fn.endswith('.lib'):
            return OUTPUTDIR + '/lib/' + fn[:-4] + '.a'
        if fn.endswith('.ilb'):
            return OUTPUTDIR + '/tmp/' + fn[:-4] + '.a'
    if fn.endswith('.dat'):
        return OUTPUTDIR + '/tmp/' + fn
    if fn.endswith('.in'):
        return OUTPUTDIR + '/pandac/input/' + fn
    return fn

def FindLocation(fn, ipath, pyabi=None):
    if False:
        return 10
    if GetLinkAllStatic():
        if fn.endswith('.dll'):
            fn = fn[:-4] + '.lib'
        elif fn.endswith('.pyd'):
            fn = 'libpy.panda3d.' + os.path.splitext(fn[:-4] + GetExtensionSuffix())[0] + '.lib'
    loc = CalcLocation(fn, ipath)
    (base, ext) = os.path.splitext(fn)
    if loc in PYABI_SPECIFIC:
        if loc.startswith(OUTPUTDIR + '/tmp'):
            if pyabi is not None:
                loc = OUTPUTDIR + '/tmp/' + pyabi + loc[len(OUTPUTDIR) + 4:]
            else:
                raise RuntimeError('%s is a Python-specific target, use PyTargetAdd instead of TargetAdd' % fn)
        elif ext != '.pyd' and loc not in WARNED_FILES:
            WARNED_FILES.add(loc)
            Warn('file depends on Python but is not in an ABI-specific directory:', loc)
    ORIG_EXT[loc] = ext
    return loc

def GetCurrentPythonVersionInfo():
    if False:
        while True:
            i = 10
    if PkgSkip('PYTHON'):
        return
    return {'version': SDK['PYTHONVERSION'][6:].rstrip('dmu'), 'soabi': GetPythonABI(), 'ext_suffix': GetExtensionSuffix(), 'executable': sys.executable, 'purelib': locations.get_python_lib(False), 'platlib': locations.get_python_lib(True)}

def UpdatePythonVersionInfoFile(new_info):
    if False:
        i = 10
        return i + 15
    import json
    json_file = os.path.join(GetOutputDir(), 'tmp', 'python_versions.json')
    json_data = []
    if os.path.isfile(json_file) and (not PkgSkip('PYTHON')):
        try:
            with open(json_file, 'r') as fh:
                json_data = json.load(fh)
        except:
            json_data = []
        for version_info in json_data[:]:
            core_pyd = os.path.join(GetOutputDir(), 'panda3d', 'core' + version_info['ext_suffix'])
            if version_info['ext_suffix'] == new_info['ext_suffix'] or version_info['soabi'] == new_info['soabi'] or (not os.path.isfile(core_pyd)) or (version_info['version'].split('.', 1)[0] == '2') or (version_info['version'] in ('3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7')):
                json_data.remove(version_info)
    if not PkgSkip('PYTHON'):
        json_data.append(new_info)
    if VERBOSE:
        print('Writing %s' % json_file)
    with open(json_file, 'w') as fh:
        json.dump(json_data, fh, indent=4)

def ReadPythonVersionInfoFile():
    if False:
        return 10
    import json
    json_file = os.path.join(GetOutputDir(), 'tmp', 'python_versions.json')
    if os.path.isfile(json_file):
        try:
            json_data = json.load(open(json_file, 'r'))
        except:
            pass
        for version_info in json_data[:]:
            if version_info['version'] in ('2.6', '2.7', '3.0', '3.1', '3.2', '3.3', '3.4', '3.5', '3.6', '3.7'):
                json_data.remove(version_info)
        return json_data
    return []

class Target:
    pass
TARGET_LIST = []
TARGET_TABLE = {}

def TargetAdd(target, dummy=0, opts=[], input=[], dep=[], ipath=None, winrc=None, pyabi=None):
    if False:
        print('Hello World!')
    if dummy != 0:
        exit('Syntax error in TargetAdd ' + target)
    if ipath is None:
        ipath = opts
    if not ipath:
        ipath = []
    if isinstance(input, str):
        input = [input]
    if isinstance(dep, str):
        dep = [dep]
    if target.endswith('.pyd') and (not pyabi):
        raise RuntimeError('Use PyTargetAdd to build .pyd targets')
    full = FindLocation(target, [OUTPUTDIR + '/include'], pyabi=pyabi)
    if full not in TARGET_TABLE:
        t = Target()
        t.name = full
        t.inputs = []
        t.deps = {}
        t.opts = []
        TARGET_TABLE[full] = t
        TARGET_LIST.append(t)
    else:
        t = TARGET_TABLE[full]
    for x in opts:
        if x not in t.opts:
            t.opts.append(x)
    ipath = [OUTPUTDIR + '/tmp'] + GetListOption(ipath, 'DIR:') + [OUTPUTDIR + '/include']
    for x in input:
        fullinput = FindLocation(x, ipath, pyabi=pyabi)
        t.inputs.append(fullinput)
        if os.path.splitext(x)[-1] not in SUFFIX_DLL:
            t.deps[fullinput] = 1
            (base, suffix) = os.path.splitext(x)
            if SUFFIX_INC.count(suffix):
                for d in CxxCalcDependencies(fullinput, ipath, []):
                    t.deps[d] = 1
            elif suffix == '.java':
                for d in JavaCalcDependencies(fullinput, OUTPUTDIR + '/classes'):
                    t.deps[d] = 1
        if GetLinkAllStatic() and ORIG_EXT[fullinput] == '.lib' and (fullinput in TARGET_TABLE):
            tdep = TARGET_TABLE[fullinput]
            for y in tdep.inputs:
                if ORIG_EXT[y] == '.lib':
                    t.inputs.append(y)
            for (opt, _) in LIBNAMES + LIBDIRECTORIES + FRAMEWORKDIRECTORIES:
                if opt in tdep.opts and opt not in t.opts:
                    t.opts.append(opt)
        if x.endswith('.in'):
            outbase = os.path.basename(x)[:-3]
            woutc = GetOutputDir() + '/tmp/' + outbase + '_igate.cxx'
            t.deps[woutc] = 1
        if target.endswith('.in'):
            (base, ext) = os.path.splitext(fullinput)
            fulln = base + '.N'
            if os.path.isfile(fulln):
                t.deps[fulln] = 1
    for x in dep:
        fulldep = FindLocation(x, ipath, pyabi=pyabi)
        t.deps[fulldep] = 1
    if winrc and GetTarget() == 'windows':
        TargetAdd(target, input=WriteResourceFile(target.split('/')[-1].split('.')[0], **winrc))
    ext = os.path.splitext(target)[1]
    if ext == '.in':
        if not CrossCompiling():
            t.deps[FindLocation('interrogate.exe', [])] = 1
        t.deps[FindLocation('dtool_have_python.dat', [])] = 1
    if ext in ('.obj', '.tlb', '.res', '.plugin', '.app') or ext in SUFFIX_DLL or ext in SUFFIX_LIB:
        t.deps[FindLocation('platform.dat', [])] = 1
    if target.endswith('.obj') and any((x.endswith('.in') for x in input)):
        if not CrossCompiling():
            t.deps[FindLocation('interrogate_module.exe', [])] = 1
    if target.endswith('.pz') and (not CrossCompiling()):
        t.deps[FindLocation('pzip.exe', [])] = 1
    if target.endswith('.in'):
        outbase = os.path.basename(target)[:-3]
        woutc = OUTPUTDIR + '/tmp/' + outbase + '_igate.cxx'
        CxxDependencyCache[woutc] = []
        PyTargetAdd(outbase + '_igate.obj', opts=opts + ['PYTHON', 'BIGOBJ'], input=woutc, dep=target)

def PyTargetAdd(target, opts=[], **kwargs):
    if False:
        return 10
    if PkgSkip('PYTHON'):
        return
    if 'PYTHON' not in opts:
        opts = opts + ['PYTHON']
    abi = GetPythonABI()
    MakeDirectory(OUTPUTDIR + '/tmp/' + abi)
    orig = CalcLocation(target, [OUTPUTDIR + '/include'])
    PYABI_SPECIFIC.add(orig)
    if orig.startswith(OUTPUTDIR + '/tmp/') and os.path.exists(orig):
        print('Removing file %s' % orig)
        os.unlink(orig)
    TargetAdd(target, opts=opts, pyabi=abi, **kwargs)