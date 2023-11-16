import re
from twisted.internet import defer
from buildbot import config
from buildbot.process import buildstep
from buildbot.process import results
from buildbot.process.logobserver import LogLineObserver

class MSLogLineObserver(LogLineObserver):
    stdoutDelimiter = '\r\n'
    stderrDelimiter = '\r\n'
    _re_delimiter = re.compile('^(\\d+>)?-{5}.+-{5}$')
    _re_file = re.compile('^(\\d+>)?[^ ]+\\.(cpp|c)$')
    _re_warning = re.compile(' ?: warning [A-Z]+[0-9]+:')
    _re_error = re.compile(' ?error ([A-Z]+[0-9]+)?\\s?: ')
    nbFiles = 0
    nbProjects = 0
    nbWarnings = 0
    nbErrors = 0
    logwarnings = None
    logerrors = None

    def __init__(self, logwarnings, logerrors, **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(**kwargs)
        self.logwarnings = logwarnings
        self.logerrors = logerrors

    def outLineReceived(self, line):
        if False:
            for i in range(10):
                print('nop')
        if self._re_delimiter.search(line):
            self.nbProjects += 1
            self.logwarnings.addStdout(f'{line}\n')
            self.logerrors.addStdout(f'{line}\n')
            self.step.setProgress('projects', self.nbProjects)
        elif self._re_file.search(line):
            self.nbFiles += 1
            self.step.setProgress('files', self.nbFiles)
        elif self._re_warning.search(line):
            self.nbWarnings += 1
            self.logwarnings.addStdout(f'{line}\n')
            self.step.setProgress('warnings', self.nbWarnings)
        elif self._re_error.search(f'{line}\n'):
            self.nbErrors += 1
            self.logerrors.addStderr(f'{line}\n')

class VisualStudio(buildstep.ShellMixin, buildstep.BuildStep):
    name = 'compile'
    description = 'compiling'
    descriptionDone = 'compile'
    progressMetrics = buildstep.BuildStep.progressMetrics + ('projects', 'files', 'warnings')
    logobserver = None
    installdir = None
    default_installdir = None
    mode = 'rebuild'
    projectfile = None
    config = None
    useenv = False
    project = None
    PATH = []
    INCLUDE = []
    LIB = []
    renderables = ['projectfile', 'config', 'project', 'mode']

    def __init__(self, installdir=None, mode='rebuild', projectfile=None, config='release', useenv=False, project=None, INCLUDE=None, LIB=None, PATH=None, **kwargs):
        if False:
            print('Hello World!')
        if INCLUDE is None:
            INCLUDE = []
        if LIB is None:
            LIB = []
        if PATH is None:
            PATH = []
        self.installdir = installdir
        self.mode = mode
        self.projectfile = projectfile
        self.config = config
        self.useenv = useenv
        self.project = project
        if INCLUDE:
            self.INCLUDE = INCLUDE
            self.useenv = True
        if LIB:
            self.LIB = LIB
            self.useenv = True
        if PATH:
            self.PATH = PATH
        kwargs = self.setupShellMixin(kwargs, prohibitArgs=['command'])
        super().__init__(**kwargs)

    def add_env_path(self, name, value):
        if False:
            i = 10
            return i + 15
        ' concat a path for this name '
        try:
            oldval = self.env[name]
            if not oldval.endswith(';'):
                oldval = oldval + ';'
        except KeyError:
            oldval = ''
        if not value.endswith(';'):
            value = value + ';'
        self.env[name] = oldval + value

    @defer.inlineCallbacks
    def setup_log_files(self):
        if False:
            for i in range(10):
                print('nop')
        logwarnings = (yield self.addLog('warnings'))
        logerrors = (yield self.addLog('errors'))
        self.logobserver = MSLogLineObserver(logwarnings, logerrors)
        yield self.addLogObserver('stdio', self.logobserver)

    def setupEnvironment(self):
        if False:
            i = 10
            return i + 15
        if self.env is None:
            self.env = {}
        for path in self.PATH:
            self.add_env_path('PATH', path)
        for path in self.INCLUDE:
            self.add_env_path('INCLUDE', path)
        for path in self.LIB:
            self.add_env_path('LIB', path)
        if not self.installdir:
            self.installdir = self.default_installdir

    def evaluate_result(self, cmd):
        if False:
            i = 10
            return i + 15
        self.setStatistic('projects', self.logobserver.nbProjects)
        self.setStatistic('files', self.logobserver.nbFiles)
        self.setStatistic('warnings', self.logobserver.nbWarnings)
        self.setStatistic('errors', self.logobserver.nbErrors)
        if cmd.didFail():
            return results.FAILURE
        if self.logobserver.nbErrors > 0:
            return results.FAILURE
        if self.logobserver.nbWarnings > 0:
            return results.WARNINGS
        return results.SUCCESS

    @defer.inlineCallbacks
    def run(self):
        if False:
            while True:
                i = 10
        self.setupEnvironment()
        yield self.setup_log_files()
        cmd = (yield self.makeRemoteShellCommand())
        yield self.runCommand(cmd)
        yield self.finish_logs()
        self.results = self.evaluate_result(cmd)
        return self.results

    def getResultSummary(self):
        if False:
            while True:
                i = 10
        if self.logobserver is None:
            return {'step': results.statusToString(self.results)}
        description = f'compile {self.logobserver.nbProjects} projects {self.logobserver.nbFiles} files'
        if self.logobserver.nbWarnings > 0:
            description += f' {self.logobserver.nbWarnings} warnings'
        if self.logobserver.nbErrors > 0:
            description += f' {self.logobserver.nbErrors} errors'
        if self.results != results.SUCCESS:
            description += f' ({results.statusToString(self.results)})'
            if self.timed_out:
                description += ' (timed out)'
        return {'step': description}

    @defer.inlineCallbacks
    def finish_logs(self):
        if False:
            for i in range(10):
                print('nop')
        log = (yield self.getLog('warnings'))
        yield log.finish()
        log = (yield self.getLog('errors'))
        yield log.finish()

class VC6(VisualStudio):
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio'

    def setupEnvironment(self):
        if False:
            for i in range(10):
                print('nop')
        super().setupEnvironment()
        VSCommonDir = self.installdir + '\\Common'
        MSVCDir = self.installdir + '\\VC98'
        MSDevDir = VSCommonDir + '\\msdev98'
        self.add_env_path('PATH', MSDevDir + '\\BIN')
        self.add_env_path('PATH', MSVCDir + '\\BIN')
        self.add_env_path('PATH', VSCommonDir + '\\TOOLS\\WINNT')
        self.add_env_path('PATH', VSCommonDir + '\\TOOLS')
        self.add_env_path('INCLUDE', MSVCDir + '\\INCLUDE')
        self.add_env_path('INCLUDE', MSVCDir + '\\ATL\\INCLUDE')
        self.add_env_path('INCLUDE', MSVCDir + '\\MFC\\INCLUDE')
        self.add_env_path('LIB', MSVCDir + '\\LIB')
        self.add_env_path('LIB', MSVCDir + '\\MFC\\LIB')

    @defer.inlineCallbacks
    def run(self):
        if False:
            while True:
                i = 10
        command = ['msdev', self.projectfile, '/MAKE']
        if self.project is not None:
            command.append(self.project + ' - ' + self.config)
        else:
            command.append('ALL - ' + self.config)
        if self.mode == 'rebuild':
            command.append('/REBUILD')
        elif self.mode == 'clean':
            command.append('/CLEAN')
        else:
            command.append('/BUILD')
        if self.useenv:
            command.append('/USEENV')
        self.command = command
        res = (yield super().run())
        return res

class VC7(VisualStudio):
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio .NET 2003'

    def setupEnvironment(self):
        if False:
            i = 10
            return i + 15
        super().setupEnvironment()
        VSInstallDir = self.installdir + '\\Common7\\IDE'
        VCInstallDir = self.installdir
        MSVCDir = self.installdir + '\\VC7'
        self.add_env_path('PATH', VSInstallDir)
        self.add_env_path('PATH', MSVCDir + '\\BIN')
        self.add_env_path('PATH', VCInstallDir + '\\Common7\\Tools')
        self.add_env_path('PATH', VCInstallDir + '\\Common7\\Tools\\bin')
        self.add_env_path('INCLUDE', MSVCDir + '\\INCLUDE')
        self.add_env_path('INCLUDE', MSVCDir + '\\ATLMFC\\INCLUDE')
        self.add_env_path('INCLUDE', MSVCDir + '\\PlatformSDK\\include')
        self.add_env_path('INCLUDE', VCInstallDir + '\\SDK\\v1.1\\include')
        self.add_env_path('LIB', MSVCDir + '\\LIB')
        self.add_env_path('LIB', MSVCDir + '\\ATLMFC\\LIB')
        self.add_env_path('LIB', MSVCDir + '\\PlatformSDK\\lib')
        self.add_env_path('LIB', VCInstallDir + '\\SDK\\v1.1\\lib')

    @defer.inlineCallbacks
    def run(self):
        if False:
            return 10
        command = ['devenv.com', self.projectfile]
        if self.mode == 'rebuild':
            command.append('/Rebuild')
        elif self.mode == 'clean':
            command.append('/Clean')
        else:
            command.append('/Build')
        command.append(self.config)
        if self.useenv:
            command.append('/UseEnv')
        if self.project is not None:
            command.append('/Project')
            command.append(self.project)
        self.command = command
        res = (yield super().run())
        return res
VS2003 = VC7

class VC8(VC7):
    arch = None
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio 8'
    renderables = ['arch']

    def __init__(self, arch='x86', **kwargs):
        if False:
            while True:
                i = 10
        self.arch = arch
        super().__init__(**kwargs)

    def setupEnvironment(self):
        if False:
            print('Hello World!')
        VisualStudio.setupEnvironment(self)
        VSInstallDir = self.installdir
        VCInstallDir = self.installdir + '\\VC'
        self.add_env_path('PATH', VSInstallDir + '\\Common7\\IDE')
        if self.arch == 'x64':
            self.add_env_path('PATH', VCInstallDir + '\\BIN\\x86_amd64')
        self.add_env_path('PATH', VCInstallDir + '\\BIN')
        self.add_env_path('PATH', VSInstallDir + '\\Common7\\Tools')
        self.add_env_path('PATH', VSInstallDir + '\\Common7\\Tools\\bin')
        self.add_env_path('PATH', VCInstallDir + '\\PlatformSDK\\bin')
        self.add_env_path('PATH', VSInstallDir + '\\SDK\\v2.0\\bin')
        self.add_env_path('PATH', VCInstallDir + '\\VCPackages')
        self.add_env_path('PATH', '${PATH}')
        self.add_env_path('INCLUDE', VCInstallDir + '\\INCLUDE')
        self.add_env_path('INCLUDE', VCInstallDir + '\\ATLMFC\\include')
        self.add_env_path('INCLUDE', VCInstallDir + '\\PlatformSDK\\include')
        archsuffix = ''
        if self.arch == 'x64':
            archsuffix = '\\amd64'
        self.add_env_path('LIB', VCInstallDir + '\\LIB' + archsuffix)
        self.add_env_path('LIB', VCInstallDir + '\\ATLMFC\\LIB' + archsuffix)
        self.add_env_path('LIB', VCInstallDir + '\\PlatformSDK\\lib' + archsuffix)
        self.add_env_path('LIB', VSInstallDir + '\\SDK\\v2.0\\lib' + archsuffix)
VS2005 = VC8

class VCExpress9(VC8):

    @defer.inlineCallbacks
    def run(self):
        if False:
            print('Hello World!')
        command = ['vcexpress', self.projectfile]
        if self.mode == 'rebuild':
            command.append('/Rebuild')
        elif self.mode == 'clean':
            command.append('/Clean')
        else:
            command.append('/Build')
        command.append(self.config)
        if self.useenv:
            command.append('/UseEnv')
        if self.project is not None:
            command.append('/Project')
            command.append(self.project)
        self.command = command
        res = (yield VisualStudio.run(self))
        return res

class VC9(VC8):
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio 9.0'
VS2008 = VC9

class VC10(VC9):
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio 10.0'
VS2010 = VC10

class VC11(VC10):
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio 11.0'
VS2012 = VC11

class VC12(VC11):
    default_installdir = 'C:\\Program Files\\Microsoft Visual Studio 12.0'
VS2013 = VC12

class VC14(VC12):
    default_installdir = 'C:\\Program Files (x86)\\Microsoft Visual Studio 14.0'
VS2015 = VC14

class VC141(VC14):
    default_installdir = 'C:\\\\Program Files (x86)\\\\Microsoft Visual Studio\\\\2017\\\\Community'
VS2017 = VC141

class VS2019(VS2017):
    default_installdir = 'C:\\\\Program Files (x86)\\\\Microsoft Visual Studio\\\\2019\\\\Community'

class VS2022(VS2017):
    default_installdir = 'C:\\\\Program Files (x86)\\\\Microsoft Visual Studio\\\\2022\\\\Community'

def _msbuild_format_defines_parameter(defines):
    if False:
        print('Hello World!')
    if defines is None or len(defines) == 0:
        return ''
    return f''' /p:DefineConstants="{';'.join(defines)}"'''

def _msbuild_format_target_parameter(mode, project):
    if False:
        return 10
    modestring = None
    if mode == 'clean':
        modestring = 'Clean'
    elif mode == 'build':
        modestring = 'Build'
    elif mode == 'rebuild':
        modestring = 'Rebuild'
    parameter = ''
    if project is not None:
        if modestring == 'Rebuild' or modestring is None:
            parameter = f' /t:"{project}"'
        else:
            parameter = f' /t:"{project}:{modestring}"'
    elif modestring is not None:
        parameter = f' /t:{modestring}'
    return parameter

class MsBuild4(VisualStudio):
    platform = None
    defines = None
    vcenv_bat = '${VS110COMNTOOLS}..\\..\\VC\\vcvarsall.bat'
    renderables = ['platform']
    description = 'building'

    def __init__(self, platform, defines=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.platform = platform
        self.defines = defines
        super().__init__(**kwargs)

    def setupEnvironment(self):
        if False:
            for i in range(10):
                print('nop')
        super().setupEnvironment()
        self.env['VCENV_BAT'] = self.vcenv_bat

    def describe_project(self, done=False):
        if False:
            for i in range(10):
                print('nop')
        project = self.project
        if project is None:
            project = 'solution'
        return f'{project} for {self.config}|{self.platform}'

    def getCurrentSummary(self):
        if False:
            print('Hello World!')
        return {'step': 'building ' + self.describe_project()}

    def getResultSummary(self):
        if False:
            for i in range(10):
                print('nop')
        return {'step': 'built ' + self.describe_project()}

    @defer.inlineCallbacks
    def run(self):
        if False:
            i = 10
            return i + 15
        if self.platform is None:
            config.error('platform is mandatory. Please specify a string such as "Win32"')
        yield self.updateSummary()
        command = f'"%VCENV_BAT%" x86 && msbuild "{self.projectfile}" /p:Configuration="{self.config}" /p:Platform="{self.platform}" /maxcpucount'
        command += _msbuild_format_target_parameter(self.mode, self.project)
        command += _msbuild_format_defines_parameter(self.defines)
        self.command = command
        res = (yield super().run())
        return res
MsBuild = MsBuild4

class MsBuild12(MsBuild4):
    vcenv_bat = '${VS120COMNTOOLS}..\\..\\VC\\vcvarsall.bat'

class MsBuild14(MsBuild4):
    vcenv_bat = '${VS140COMNTOOLS}..\\..\\VC\\vcvarsall.bat'

class MsBuild141(VisualStudio):
    platform = None
    defines = None
    vcenv_bat = '\\VC\\Auxiliary\\Build\\vcvarsall.bat'
    renderables = ['platform']
    version_range = '[15.0,16.0)'

    def __init__(self, platform, defines=None, **kwargs):
        if False:
            print('Hello World!')
        self.platform = platform
        self.defines = defines
        super().__init__(**kwargs)

    def setupEnvironment(self):
        if False:
            i = 10
            return i + 15
        super().setupEnvironment()
        self.env['VCENV_BAT'] = self.vcenv_bat
        self.add_env_path('PATH', 'C:\\Program Files (x86)\\Microsoft Visual Studio\\Installer\\')
        self.add_env_path('PATH', 'C:\\Program Files\\Microsoft Visual Studio\\Installer\\')
        self.add_env_path('PATH', '${PATH}')

    def describe_project(self, done=False):
        if False:
            return 10
        project = self.project
        if project is None:
            project = 'solution'
        return f'{project} for {self.config}|{self.platform}'

    @defer.inlineCallbacks
    def run(self):
        if False:
            i = 10
            return i + 15
        if self.platform is None:
            config.error('platform is mandatory. Please specify a string such as "Win32"')
        self.description = 'building ' + self.describe_project()
        self.descriptionDone = 'built ' + self.describe_project()
        yield self.updateSummary()
        command = f'''FOR /F "tokens=*" %%I in ('vswhere.exe -version "{self.version_range}" -products * -property installationPath')  do "%%I\\%VCENV_BAT%" x86 && msbuild "{self.projectfile}" /p:Configuration="{self.config}" /p:Platform="{self.platform}" /maxcpucount'''
        command += _msbuild_format_target_parameter(self.mode, self.project)
        command += _msbuild_format_defines_parameter(self.defines)
        self.command = command
        res = (yield super().run())
        return res
MsBuild15 = MsBuild141

class MsBuild16(MsBuild141):
    version_range = '[16.0,17.0)'

class MsBuild17(MsBuild141):
    version_range = '[17.0,18.0)'