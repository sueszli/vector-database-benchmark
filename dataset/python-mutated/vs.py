__revision__ = 'src/engine/SCons/Tool/MSCommon/vs.py bee7caf9defd6e108fc2998a2520ddb36a967691 2019-12-17 02:07:09 bdeegan'
__doc__ = 'Module to detect Visual Studio and/or Visual C/C++\n'
import os
import SCons.Errors
import SCons.Util
from .common import debug, get_output, is_win64, normalize_env, parse_output, read_reg
import SCons.Tool.MSCommon.vc

class VisualStudio(object):
    """
    An abstract base class for trying to find installed versions of
    Visual Studio.
    """

    def __init__(self, version, **kw):
        if False:
            i = 10
            return i + 15
        self.version = version
        kw['vc_version'] = kw.get('vc_version', version)
        kw['sdk_version'] = kw.get('sdk_version', version)
        self.__dict__.update(kw)
        self._cache = {}

    def find_batch_file(self):
        if False:
            print('Hello World!')
        vs_dir = self.get_vs_dir()
        if not vs_dir:
            debug('find_executable():  no vs_dir')
            return None
        batch_file = os.path.join(vs_dir, self.batch_file_path)
        batch_file = os.path.normpath(batch_file)
        if not os.path.isfile(batch_file):
            debug('find_batch_file():  %s not on file system' % batch_file)
            return None
        return batch_file

    def find_vs_dir_by_vc(self):
        if False:
            i = 10
            return i + 15
        SCons.Tool.MSCommon.vc.get_installed_vcs()
        dir = SCons.Tool.MSCommon.vc.find_vc_pdir(self.vc_version)
        if not dir:
            debug('find_vs_dir_by_vc():  no installed VC %s' % self.vc_version)
            return None
        return os.path.abspath(os.path.join(dir, os.pardir))

    def find_vs_dir_by_reg(self):
        if False:
            while True:
                i = 10
        root = 'Software\\'
        if is_win64():
            root = root + 'Wow6432Node\\'
        for key in self.hkeys:
            if key == 'use_dir':
                return self.find_vs_dir_by_vc()
            key = root + key
            try:
                comps = read_reg(key)
            except SCons.Util.WinError as e:
                debug('find_vs_dir_by_reg(): no VS registry key {}'.format(repr(key)))
            else:
                debug('find_vs_dir_by_reg(): found VS in registry: {}'.format(comps))
                return comps
        return None

    def find_vs_dir(self):
        if False:
            i = 10
            return i + 15
        ' Can use registry or location of VC to find vs dir\n        First try to find by registry, and if that fails find via VC dir\n        '
        vs_dir = self.find_vs_dir_by_reg()
        if not vs_dir:
            vs_dir = self.find_vs_dir_by_vc()
        debug('find_vs_dir(): found VS in ' + str(vs_dir))
        return vs_dir

    def find_executable(self):
        if False:
            i = 10
            return i + 15
        vs_dir = self.get_vs_dir()
        if not vs_dir:
            debug('find_executable():  no vs_dir ({})'.format(vs_dir))
            return None
        executable = os.path.join(vs_dir, self.executable_path)
        executable = os.path.normpath(executable)
        if not os.path.isfile(executable):
            debug('find_executable():  {} not on file system'.format(executable))
            return None
        return executable

    def get_batch_file(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._cache['batch_file']
        except KeyError:
            batch_file = self.find_batch_file()
            self._cache['batch_file'] = batch_file
            return batch_file

    def get_executable(self):
        if False:
            i = 10
            return i + 15
        try:
            debug('get_executable using cache:%s' % self._cache['executable'])
            return self._cache['executable']
        except KeyError:
            executable = self.find_executable()
            self._cache['executable'] = executable
            debug('get_executable not in cache:%s' % executable)
            return executable

    def get_vs_dir(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            return self._cache['vs_dir']
        except KeyError:
            vs_dir = self.find_vs_dir()
            self._cache['vs_dir'] = vs_dir
            return vs_dir

    def get_supported_arch(self):
        if False:
            i = 10
            return i + 15
        try:
            return self._cache['supported_arch']
        except KeyError:
            self._cache['supported_arch'] = self.supported_arch
            return self.supported_arch

    def reset(self):
        if False:
            print('Hello World!')
        self._cache = {}
SupportedVSList = [VisualStudio('14.2', vc_version='14.2', sdk_version='10.0A', hkeys=[], common_tools_var='VS160COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='VC\\Auxiliary\\Build\\vsvars32.bat', supported_arch=['x86', 'amd64', 'arm']), VisualStudio('14.1', vc_version='14.1', sdk_version='10.0A', hkeys=[], common_tools_var='VS150COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='VC\\Auxiliary\\Build\\vsvars32.bat', supported_arch=['x86', 'amd64', 'arm']), VisualStudio('14.0', vc_version='14.0', sdk_version='10.0', hkeys=['Microsoft\\VisualStudio\\14.0\\Setup\\VS\\ProductDir'], common_tools_var='VS140COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64', 'arm']), VisualStudio('14.0Exp', vc_version='14.0', sdk_version='10.0A', hkeys=['Microsoft\\VisualStudio\\14.0\\Setup\\VS\\ProductDir'], common_tools_var='VS140COMNTOOLS', executable_path='Common7\\IDE\\WDExpress.exe', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64', 'arm']), VisualStudio('12.0', vc_version='12.0', sdk_version='8.1A', hkeys=['Microsoft\\VisualStudio\\12.0\\Setup\\VS\\ProductDir'], common_tools_var='VS120COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64']), VisualStudio('12.0Exp', vc_version='12.0', sdk_version='8.1A', hkeys=['Microsoft\\VisualStudio\\12.0\\Setup\\VS\\ProductDir'], common_tools_var='VS120COMNTOOLS', executable_path='Common7\\IDE\\WDExpress.exe', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64']), VisualStudio('11.0', sdk_version='8.0A', hkeys=['Microsoft\\VisualStudio\\11.0\\Setup\\VS\\ProductDir'], common_tools_var='VS110COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64']), VisualStudio('11.0Exp', vc_version='11.0', sdk_version='8.0A', hkeys=['Microsoft\\VisualStudio\\11.0\\Setup\\VS\\ProductDir'], common_tools_var='VS110COMNTOOLS', executable_path='Common7\\IDE\\WDExpress.exe', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64']), VisualStudio('10.0', sdk_version='7.0A', hkeys=['Microsoft\\VisualStudio\\10.0\\Setup\\VS\\ProductDir'], common_tools_var='VS100COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64']), VisualStudio('10.0Exp', vc_version='10.0', sdk_version='7.0A', hkeys=['Microsoft\\VCExpress\\10.0\\Setup\\VS\\ProductDir'], common_tools_var='VS100COMNTOOLS', executable_path='Common7\\IDE\\VCExpress.exe', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86']), VisualStudio('9.0', sdk_version='6.0A', hkeys=['Microsoft\\VisualStudio\\9.0\\Setup\\VS\\ProductDir'], common_tools_var='VS90COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86', 'amd64']), VisualStudio('9.0Exp', vc_version='9.0', sdk_version='6.0A', hkeys=['Microsoft\\VCExpress\\9.0\\Setup\\VS\\ProductDir'], common_tools_var='VS90COMNTOOLS', executable_path='Common7\\IDE\\VCExpress.exe', batch_file_path='Common7\\Tools\\vsvars32.bat', supported_arch=['x86']), VisualStudio('8.0', sdk_version='6.0A', hkeys=['Microsoft\\VisualStudio\\8.0\\Setup\\VS\\ProductDir'], common_tools_var='VS80COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', default_dirname='Microsoft Visual Studio 8', supported_arch=['x86', 'amd64']), VisualStudio('8.0Exp', vc_version='8.0Exp', sdk_version='6.0A', hkeys=['Microsoft\\VCExpress\\8.0\\Setup\\VS\\ProductDir'], common_tools_var='VS80COMNTOOLS', executable_path='Common7\\IDE\\VCExpress.exe', batch_file_path='Common7\\Tools\\vsvars32.bat', default_dirname='Microsoft Visual Studio 8', supported_arch=['x86']), VisualStudio('7.1', sdk_version='6.0', hkeys=['Microsoft\\VisualStudio\\7.1\\Setup\\VS\\ProductDir'], common_tools_var='VS71COMNTOOLS', executable_path='Common7\\IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', default_dirname='Microsoft Visual Studio .NET 2003', supported_arch=['x86']), VisualStudio('7.0', sdk_version='2003R2', hkeys=['Microsoft\\VisualStudio\\7.0\\Setup\\VS\\ProductDir'], common_tools_var='VS70COMNTOOLS', executable_path='IDE\\devenv.com', batch_file_path='Common7\\Tools\\vsvars32.bat', default_dirname='Microsoft Visual Studio .NET', supported_arch=['x86']), VisualStudio('6.0', sdk_version='2003R1', hkeys=['Microsoft\\VisualStudio\\6.0\\Setup\\Microsoft Visual Studio\\ProductDir', 'use_dir'], common_tools_var='VS60COMNTOOLS', executable_path='Common\\MSDev98\\Bin\\MSDEV.COM', batch_file_path='Common7\\Tools\\vsvars32.bat', default_dirname='Microsoft Visual Studio', supported_arch=['x86'])]
SupportedVSMap = {}
for vs in SupportedVSList:
    SupportedVSMap[vs.version] = vs
InstalledVSList = None
InstalledVSMap = None

def get_installed_visual_studios():
    if False:
        while True:
            i = 10
    global InstalledVSList
    global InstalledVSMap
    if InstalledVSList is None:
        InstalledVSList = []
        InstalledVSMap = {}
        for vs in SupportedVSList:
            debug('trying to find VS %s' % vs.version)
            if vs.get_executable():
                debug('found VS %s' % vs.version)
                InstalledVSList.append(vs)
                InstalledVSMap[vs.version] = vs
    return InstalledVSList

def reset_installed_visual_studios():
    if False:
        print('Hello World!')
    global InstalledVSList
    global InstalledVSMap
    InstalledVSList = None
    InstalledVSMap = None
    for vs in SupportedVSList:
        vs.reset()
    SCons.Tool.MSCommon.vc.reset_installed_vcs()

def msvs_exists():
    if False:
        while True:
            i = 10
    return len(get_installed_visual_studios()) > 0

def get_vs_by_version(msvs):
    if False:
        i = 10
        return i + 15
    global InstalledVSMap
    global SupportedVSMap
    debug('get_vs_by_version()')
    if msvs not in SupportedVSMap:
        msg = 'Visual Studio version %s is not supported' % repr(msvs)
        raise SCons.Errors.UserError(msg)
    get_installed_visual_studios()
    vs = InstalledVSMap.get(msvs)
    debug('InstalledVSMap:%s' % InstalledVSMap)
    debug('get_vs_by_version: found vs:%s' % vs)
    return vs

def get_default_version(env):
    if False:
        for i in range(10):
            print('nop')
    'Returns the default version string to use for MSVS.\n\n    If no version was requested by the user through the MSVS environment\n    variable, query all the available visual studios through\n    get_installed_visual_studios, and take the highest one.\n\n    Return\n    ------\n    version: str\n        the default version.\n    '
    if 'MSVS' not in env or not SCons.Util.is_Dict(env['MSVS']):
        versions = [vs.version for vs in get_installed_visual_studios()]
        env['MSVS'] = {'VERSIONS': versions}
    else:
        versions = env['MSVS'].get('VERSIONS', [])
    if 'MSVS_VERSION' not in env:
        if versions:
            env['MSVS_VERSION'] = versions[0]
        else:
            debug('get_default_version: WARNING: no installed versions found, using first in SupportedVSList (%s)' % SupportedVSList[0].version)
            env['MSVS_VERSION'] = SupportedVSList[0].version
    env['MSVS']['VERSION'] = env['MSVS_VERSION']
    return env['MSVS_VERSION']

def get_default_arch(env):
    if False:
        while True:
            i = 10
    'Return the default arch to use for MSVS\n\n    if no version was requested by the user through the MSVS_ARCH environment\n    variable, select x86\n\n    Return\n    ------\n    arch: str\n    '
    arch = env.get('MSVS_ARCH', 'x86')
    msvs = InstalledVSMap.get(env['MSVS_VERSION'])
    if not msvs:
        arch = 'x86'
    elif arch not in msvs.get_supported_arch():
        fmt = 'Visual Studio version %s does not support architecture %s'
        raise SCons.Errors.UserError(fmt % (env['MSVS_VERSION'], arch))
    return arch

def merge_default_version(env):
    if False:
        i = 10
        return i + 15
    version = get_default_version(env)
    arch = get_default_arch(env)

def msvs_setup_env(env):
    if False:
        while True:
            i = 10
    batfilename = msvs.get_batch_file()
    msvs = get_vs_by_version(version)
    if msvs is None:
        return
    if batfilename is not None:
        vars = ('LIB', 'LIBPATH', 'PATH', 'INCLUDE')
        msvs_list = get_installed_visual_studios()
        vscommonvarnames = [vs.common_tools_var for vs in msvs_list]
        save_ENV = env['ENV']
        nenv = normalize_env(env['ENV'], ['COMSPEC'] + vscommonvarnames, force=True)
        try:
            output = get_output(batfilename, arch, env=nenv)
        finally:
            env['ENV'] = save_ENV
        vars = parse_output(output, vars)
        for (k, v) in vars.items():
            env.PrependENVPath(k, v, delete_existing=1)

def query_versions():
    if False:
        while True:
            i = 10
    'Query the system to get available versions of VS. A version is\n    considered when a batfile is found.'
    msvs_list = get_installed_visual_studios()
    versions = [msvs.version for msvs in msvs_list]
    return versions