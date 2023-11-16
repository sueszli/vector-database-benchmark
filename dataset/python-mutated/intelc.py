"""SCons.Tool.icl

Tool-specific initialization for the Intel C/C++ compiler.
Supports Linux and Windows compilers, v7 and up.

There normally shouldn't be any need to import this module directly.
It will usually be imported through the generic SCons.Tool.Tool()
selection method.

"""
from __future__ import division, print_function
__revision__ = 'src/engine/SCons/Tool/intelc.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
import math, sys, os.path, glob, string, re
is_windows = sys.platform == 'win32'
is_win64 = is_windows and (os.environ['PROCESSOR_ARCHITECTURE'] == 'AMD64' or ('PROCESSOR_ARCHITEW6432' in os.environ and os.environ['PROCESSOR_ARCHITEW6432'] == 'AMD64'))
is_linux = sys.platform.startswith('linux')
is_mac = sys.platform == 'darwin'
if is_windows:
    import SCons.Tool.msvc
elif is_linux:
    import SCons.Tool.gcc
elif is_mac:
    import SCons.Tool.gcc
import SCons.Util
import SCons.Warnings

class IntelCError(SCons.Errors.InternalError):
    pass

class MissingRegistryError(IntelCError):
    pass

class MissingDirError(IntelCError):
    pass

class NoRegistryModuleError(IntelCError):
    pass

def linux_ver_normalize(vstr):
    if False:
        i = 10
        return i + 15
    'Normalize a Linux compiler version number.\n    Intel changed from "80" to "9.0" in 2005, so we assume if the number\n    is greater than 60 it\'s an old-style number and otherwise new-style.\n    Always returns an old-style float like 80 or 90 for compatibility with Windows.\n    Shades of Y2K!'
    m = re.match('([0-9]+)\\.([0-9]+)\\.([0-9]+)', vstr)
    if m:
        (vmaj, vmin, build) = m.groups()
        return float(vmaj) * 10.0 + float(vmin) + float(build) / 1000.0
    else:
        f = float(vstr)
        if is_windows:
            return f
        elif f < 60:
            return f * 10.0
        else:
            return f

def check_abi(abi):
    if False:
        for i in range(10):
            print('nop')
    'Check for valid ABI (application binary interface) name,\n    and map into canonical one'
    if not abi:
        return None
    abi = abi.lower()
    if is_windows:
        valid_abis = {'ia32': 'ia32', 'x86': 'ia32', 'ia64': 'ia64', 'em64t': 'em64t', 'amd64': 'em64t'}
    if is_linux:
        valid_abis = {'ia32': 'ia32', 'x86': 'ia32', 'x86_64': 'x86_64', 'em64t': 'x86_64', 'amd64': 'x86_64'}
    if is_mac:
        valid_abis = {'ia32': 'ia32', 'x86': 'ia32', 'x86_64': 'x86_64', 'em64t': 'x86_64'}
    try:
        abi = valid_abis[abi]
    except KeyError:
        raise SCons.Errors.UserError('Intel compiler: Invalid ABI %s, valid values are %s' % (abi, list(valid_abis.keys())))
    return abi

def get_version_from_list(v, vlist):
    if False:
        i = 10
        return i + 15
    'See if we can match v (string) in vlist (list of strings)\n    Linux has to match in a fuzzy way.'
    if is_windows:
        if v in vlist:
            return v
        else:
            return None
    else:
        fuzz = 0.001
        for vi in vlist:
            if math.fabs(linux_ver_normalize(vi) - linux_ver_normalize(v)) < fuzz:
                return vi
        return None

def get_intel_registry_value(valuename, version=None, abi=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Return a value from the Intel compiler registry tree. (Windows only)\n    '
    if is_win64:
        K = 'Software\\Wow6432Node\\Intel\\Compilers\\C++\\' + version + '\\' + abi.upper()
    else:
        K = 'Software\\Intel\\Compilers\\C++\\' + version + '\\' + abi.upper()
    try:
        k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, K)
    except SCons.Util.RegError:
        if is_win64:
            K = 'Software\\Wow6432Node\\Intel\\Suites\\' + version + '\\Defaults\\C++\\' + abi.upper()
        else:
            K = 'Software\\Intel\\Suites\\' + version + '\\Defaults\\C++\\' + abi.upper()
        try:
            k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, K)
            uuid = SCons.Util.RegQueryValueEx(k, 'SubKey')[0]
            if is_win64:
                K = 'Software\\Wow6432Node\\Intel\\Suites\\' + version + '\\' + uuid + '\\C++'
            else:
                K = 'Software\\Intel\\Suites\\' + version + '\\' + uuid + '\\C++'
            k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, K)
            try:
                v = SCons.Util.RegQueryValueEx(k, valuename)[0]
                return v
            except SCons.Util.RegError:
                if abi.upper() == 'EM64T':
                    abi = 'em64t_native'
                if is_win64:
                    K = 'Software\\Wow6432Node\\Intel\\Suites\\' + version + '\\' + uuid + '\\C++\\' + abi.upper()
                else:
                    K = 'Software\\Intel\\Suites\\' + version + '\\' + uuid + '\\C++\\' + abi.upper()
                k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, K)
            try:
                v = SCons.Util.RegQueryValueEx(k, valuename)[0]
                return v
            except SCons.Util.RegError:
                raise MissingRegistryError("%s was not found in the registry, for Intel compiler version %s, abi='%s'" % (K, version, abi))
        except SCons.Util.RegError:
            raise MissingRegistryError("%s was not found in the registry, for Intel compiler version %s, abi='%s'" % (K, version, abi))
        except SCons.Util.WinError:
            raise MissingRegistryError("%s was not found in the registry, for Intel compiler version %s, abi='%s'" % (K, version, abi))
    try:
        v = SCons.Util.RegQueryValueEx(k, valuename)[0]
        return v
    except SCons.Util.RegError:
        raise MissingRegistryError('%s\\%s was not found in the registry.' % (K, valuename))

def get_all_compiler_versions():
    if False:
        for i in range(10):
            print('nop')
    'Returns a sorted list of strings, like "70" or "80" or "9.0"\n    with most recent compiler version first.\n    '
    versions = []
    if is_windows:
        if is_win64:
            keyname = 'Software\\WoW6432Node\\Intel\\Compilers\\C++'
        else:
            keyname = 'Software\\Intel\\Compilers\\C++'
        try:
            k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, keyname)
        except SCons.Util.WinError:
            if is_win64:
                keyname = 'Software\\WoW6432Node\\Intel\\Suites'
            else:
                keyname = 'Software\\Intel\\Suites'
            try:
                k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, keyname)
            except SCons.Util.WinError:
                return []
        i = 0
        versions = []
        try:
            while i < 100:
                subkey = SCons.Util.RegEnumKey(k, i)
                if subkey == 'Defaults':
                    i = i + 1
                    continue
                ok = False
                for try_abi in ('IA32', 'IA32e', 'IA64', 'EM64T'):
                    try:
                        d = get_intel_registry_value('ProductDir', subkey, try_abi)
                    except MissingRegistryError:
                        continue
                    if os.path.exists(d):
                        ok = True
                if ok:
                    versions.append(subkey)
                else:
                    try:
                        value = get_intel_registry_value('ProductDir', subkey, 'IA32')
                    except MissingRegistryError as e:
                        print('scons: *** Ignoring the registry key for the Intel compiler version %s.\nscons: *** It seems that the compiler was uninstalled and that the registry\nscons: *** was not cleaned up properly.\n' % subkey)
                    else:
                        print('scons: *** Ignoring ' + str(value))
                i = i + 1
        except EnvironmentError:
            pass
    elif is_linux or is_mac:
        for d in glob.glob('/opt/intel_cc_*'):
            m = re.search('cc_(.*)$', d)
            if m:
                versions.append(m.group(1))
        for d in glob.glob('/opt/intel/cc*/*'):
            m = re.search('([0-9][0-9.]*)$', d)
            if m:
                versions.append(m.group(1))
        for d in glob.glob('/opt/intel/Compiler/*'):
            m = re.search('([0-9][0-9.]*)$', d)
            if m:
                versions.append(m.group(1))
        for d in glob.glob('/opt/intel/composerxe-*'):
            m = re.search('([0-9][0-9.]*)$', d)
            if m:
                versions.append(m.group(1))
        for d in glob.glob('/opt/intel/composer_xe_*'):
            m = re.search('([0-9]{0,4})(?:_sp\\d*)?\\.([0-9][0-9.]*)$', d)
            if m:
                versions.append('%s.%s' % (m.group(1), m.group(2)))
        for d in glob.glob('/opt/intel/compilers_and_libraries_*'):
            m = re.search('([0-9]{0,4})(?:_sp\\d*)?\\.([0-9][0-9.]*)$', d)
            if m:
                versions.append('%s.%s' % (m.group(1), m.group(2)))

    def keyfunc(str):
        if False:
            for i in range(10):
                print('nop')
        'Given a dot-separated version string, return a tuple of ints representing it.'
        return [int(x) for x in str.split('.')]
    return sorted(SCons.Util.unique(versions), key=keyfunc, reverse=True)

def get_intel_compiler_top(version, abi):
    if False:
        while True:
            i = 10
    '\n    Return the main path to the top-level dir of the Intel compiler,\n    using the given version.\n    The compiler will be in <top>/bin/icl.exe (icc on linux),\n    the include dir is <top>/include, etc.\n    '
    if is_windows:
        if not SCons.Util.can_read_reg:
            raise NoRegistryModuleError('No Windows registry module was found')
        top = get_intel_registry_value('ProductDir', version, abi)
        archdir = {'x86_64': 'intel64', 'amd64': 'intel64', 'em64t': 'intel64', 'x86': 'ia32', 'i386': 'ia32', 'ia32': 'ia32'}[abi]
        if not os.path.exists(os.path.join(top, 'Bin', 'icl.exe')) and (not os.path.exists(os.path.join(top, 'Bin', abi, 'icl.exe'))) and (not os.path.exists(os.path.join(top, 'Bin', archdir, 'icl.exe'))):
            raise MissingDirError("Can't find Intel compiler in %s" % top)
    elif is_mac or is_linux:

        def find_in_2008style_dir(version):
            if False:
                i = 10
                return i + 15
            dirs = ('/opt/intel/cc/%s', '/opt/intel_cc_%s')
            if abi == 'x86_64':
                dirs = ('/opt/intel/cce/%s',)
            top = None
            for d in dirs:
                if os.path.exists(os.path.join(d % version, 'bin', 'icc')):
                    top = d % version
                    break
            return top

        def find_in_2010style_dir(version):
            if False:
                i = 10
                return i + 15
            dirs = '/opt/intel/Compiler/%s/*' % version
            dirs = glob.glob(dirs)
            dirs.sort()
            dirs.reverse()
            top = None
            for d in dirs:
                if os.path.exists(os.path.join(d, 'bin', 'ia32', 'icc')) or os.path.exists(os.path.join(d, 'bin', 'intel64', 'icc')):
                    top = d
                    break
            return top

        def find_in_2011style_dir(version):
            if False:
                i = 10
                return i + 15
            top = None
            for d in glob.glob('/opt/intel/composer_xe_*'):
                m = re.search('([0-9]{0,4})(?:_sp\\d*)?\\.([0-9][0-9.]*)$', d)
                if m:
                    cur_ver = '%s.%s' % (m.group(1), m.group(2))
                    if cur_ver == version and (os.path.exists(os.path.join(d, 'bin', 'ia32', 'icc')) or os.path.exists(os.path.join(d, 'bin', 'intel64', 'icc'))):
                        top = d
                        break
            if not top:
                for d in glob.glob('/opt/intel/composerxe-*'):
                    m = re.search('([0-9][0-9.]*)$', d)
                    if m and m.group(1) == version and (os.path.exists(os.path.join(d, 'bin', 'ia32', 'icc')) or os.path.exists(os.path.join(d, 'bin', 'intel64', 'icc'))):
                        top = d
                        break
            return top

        def find_in_2016style_dir(version):
            if False:
                return 10
            top = None
            for d in glob.glob('/opt/intel/compilers_and_libraries_%s/linux' % version):
                if os.path.exists(os.path.join(d, 'bin', 'ia32', 'icc')) or os.path.exists(os.path.join(d, 'bin', 'intel64', 'icc')):
                    top = d
                    break
            return top
        top = find_in_2016style_dir(version) or find_in_2011style_dir(version) or find_in_2010style_dir(version) or find_in_2008style_dir(version)
        if not top:
            raise MissingDirError("Can't find version %s Intel compiler in %s (abi='%s')" % (version, top, abi))
    return top

def generate(env, version=None, abi=None, topdir=None, verbose=0):
    if False:
        while True:
            i = 10
    'Add Builders and construction variables for Intel C/C++ compiler\n    to an Environment.\n    args:\n      version: (string) compiler version to use, like "80"\n      abi:     (string) \'win32\' or whatever Itanium version wants\n      topdir:  (string) compiler top dir, like\n                         "c:\\Program Files\\Intel\\Compiler70"\n                        If topdir is used, version and abi are ignored.\n      verbose: (int)    if >0, prints compiler version used.\n    '
    if not (is_mac or is_linux or is_windows):
        return
    if is_windows:
        SCons.Tool.msvc.generate(env)
    elif is_linux:
        SCons.Tool.gcc.generate(env)
    elif is_mac:
        SCons.Tool.gcc.generate(env)
    vlist = get_all_compiler_versions()
    if not version:
        if vlist:
            version = vlist[0]
    else:
        v = get_version_from_list(version, vlist)
        if not v:
            raise SCons.Errors.UserError('Invalid Intel compiler version %s: ' % version + 'installed versions are %s' % ', '.join(vlist))
        version = v
    abi = check_abi(abi)
    if abi is None:
        if is_mac or is_linux:
            uname_m = os.uname()[4]
            if uname_m == 'x86_64':
                abi = 'x86_64'
            else:
                abi = 'ia32'
        elif is_win64:
            abi = 'em64t'
        else:
            abi = 'ia32'
    if version and (not topdir):
        try:
            topdir = get_intel_compiler_top(version, abi)
        except (SCons.Util.RegError, IntelCError):
            topdir = None
    if not topdir:

        class ICLTopDirWarning(SCons.Warnings.Warning):
            pass
        if (is_mac or is_linux) and (not env.Detect('icc')) or (is_windows and (not env.Detect('icl'))):
            SCons.Warnings.enableWarningClass(ICLTopDirWarning)
            SCons.Warnings.warn(ICLTopDirWarning, "Failed to find Intel compiler for version='%s', abi='%s'" % (str(version), str(abi)))
        else:
            SCons.Warnings.enableWarningClass(ICLTopDirWarning)
            SCons.Warnings.warn(ICLTopDirWarning, "Can't find Intel compiler top dir for version='%s', abi='%s'" % (str(version), str(abi)))
    if topdir:
        archdir = {'x86_64': 'intel64', 'amd64': 'intel64', 'em64t': 'intel64', 'x86': 'ia32', 'i386': 'ia32', 'ia32': 'ia32'}[abi]
        if os.path.exists(os.path.join(topdir, 'bin', archdir)):
            bindir = 'bin/%s' % archdir
            libdir = 'lib/%s' % archdir
        else:
            bindir = 'bin'
            libdir = 'lib'
        if verbose:
            print("Intel C compiler: using version %s (%g), abi %s, in '%s/%s'" % (repr(version), linux_ver_normalize(version), abi, topdir, bindir))
            if is_linux:
                os.system('%s/%s/icc --version' % (topdir, bindir))
            if is_mac:
                os.system('%s/%s/icc --version' % (topdir, bindir))
        env['INTEL_C_COMPILER_TOP'] = topdir
        if is_linux:
            paths = {'INCLUDE': 'include', 'LIB': libdir, 'PATH': bindir, 'LD_LIBRARY_PATH': libdir}
            for p in list(paths.keys()):
                env.PrependENVPath(p, os.path.join(topdir, paths[p]))
        if is_mac:
            paths = {'INCLUDE': 'include', 'LIB': libdir, 'PATH': bindir, 'LD_LIBRARY_PATH': libdir}
            for p in list(paths.keys()):
                env.PrependENVPath(p, os.path.join(topdir, paths[p]))
        if is_windows:
            paths = (('INCLUDE', 'IncludeDir', 'Include'), ('LIB', 'LibDir', 'Lib'), ('PATH', 'BinDir', 'Bin'))
            if version is None:
                version = ''
            for p in paths:
                try:
                    path = get_intel_registry_value(p[1], version, abi)
                    path = path.replace('$(ICInstallDir)', topdir + os.sep)
                except IntelCError:
                    env.PrependENVPath(p[0], os.path.join(topdir, p[2]))
                else:
                    env.PrependENVPath(p[0], path.split(os.pathsep))
    if is_windows:
        env['CC'] = 'icl'
        env['CXX'] = 'icl'
        env['LINK'] = 'xilink'
    else:
        env['CC'] = 'icc'
        env['CXX'] = 'icpc'
        env['AR'] = 'xiar'
        env['LD'] = 'xild'
    if version:
        env['INTEL_C_COMPILER_VERSION'] = linux_ver_normalize(version)
    if is_windows:
        envlicdir = os.environ.get('INTEL_LICENSE_FILE', '')
        K = 'SOFTWARE\\Intel\\Licenses'
        try:
            k = SCons.Util.RegOpenKeyEx(SCons.Util.HKEY_LOCAL_MACHINE, K)
            reglicdir = SCons.Util.RegQueryValueEx(k, 'w_cpp')[0]
        except (AttributeError, SCons.Util.RegError):
            reglicdir = ''
        defaultlicdir = 'C:\\Program Files\\Common Files\\Intel\\Licenses'
        licdir = None
        for ld in [envlicdir, reglicdir]:
            if ld and (ld.find('@') != -1 or os.path.exists(ld)):
                licdir = ld
                break
        if not licdir:
            licdir = defaultlicdir
            if not os.path.exists(licdir):

                class ICLLicenseDirWarning(SCons.Warnings.Warning):
                    pass
                SCons.Warnings.enableWarningClass(ICLLicenseDirWarning)
                SCons.Warnings.warn(ICLLicenseDirWarning, 'Intel license dir was not found.  Tried using the INTEL_LICENSE_FILE environment variable (%s), the registry (%s) and the default path (%s).  Using the default path as a last resort.' % (envlicdir, reglicdir, defaultlicdir))
        env['ENV']['INTEL_LICENSE_FILE'] = licdir

def exists(env):
    if False:
        while True:
            i = 10
    if not (is_mac or is_linux or is_windows):
        return 0
    try:
        versions = get_all_compiler_versions()
    except (SCons.Util.RegError, IntelCError):
        versions = None
    detected = versions is not None and len(versions) > 0
    if not detected:
        if is_windows:
            return env.Detect('icl')
        elif is_linux:
            return env.Detect('icc')
        elif is_mac:
            return env.Detect('icc')
    return detected