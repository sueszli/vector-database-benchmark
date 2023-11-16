import os
import logging
import io
from edk2toolext.environment import shell_environment
from edk2toolext.environment.uefi_build import UefiBuilder
from edk2toolext.invocables.edk2_platform_build import BuildSettingsManager
from edk2toolext.invocables.edk2_setup import SetupSettingsManager, RequiredSubmodule
from edk2toolext.invocables.edk2_update import UpdateSettingsManager
from edk2toolext.invocables.edk2_pr_eval import PrEvalSettingsManager
from edk2toollib.utility_functions import RunCmd
from edk2toollib.utility_functions import GetHostInfo

class CommonPlatform:
    """ Common settings for this platform.  Define static data here and use
        for the different parts of stuart
    """
    PackagesSupported = ('EmulatorPkg',)
    ArchSupported = ('X64', 'IA32')
    TargetsSupported = ('DEBUG', 'RELEASE', 'NOOPT')
    Scopes = ('emulatorpkg', 'edk2-build')
    WorkspaceRoot = os.path.realpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

class SettingsManager(UpdateSettingsManager, SetupSettingsManager, PrEvalSettingsManager):

    def GetPackagesSupported(self):
        if False:
            for i in range(10):
                print('nop')
        ' return iterable of edk2 packages supported by this build.\n        These should be edk2 workspace relative paths '
        return CommonPlatform.PackagesSupported

    def GetArchitecturesSupported(self):
        if False:
            print('Hello World!')
        ' return iterable of edk2 architectures supported by this build '
        return CommonPlatform.ArchSupported

    def GetTargetsSupported(self):
        if False:
            print('Hello World!')
        ' return iterable of edk2 target tags supported by this build '
        return CommonPlatform.TargetsSupported

    def GetRequiredSubmodules(self):
        if False:
            for i in range(10):
                print('nop')
        ' return iterable containing RequiredSubmodule objects.\n        If no RequiredSubmodules return an empty iterable\n        '
        rs = []
        rs.append(RequiredSubmodule('CryptoPkg/Library/OpensslLib/openssl', False))
        result = io.StringIO()
        ret = RunCmd('git', 'config --file .gitmodules --get-regexp path', workingdir=self.GetWorkspaceRoot(), outstream=result)
        if ret == 0:
            for line in result.getvalue().splitlines():
                (_, _, path) = line.partition(' ')
                if path is not None:
                    if path not in [x.path for x in rs]:
                        rs.append(RequiredSubmodule(path, True))
        return rs

    def SetArchitectures(self, list_of_requested_architectures):
        if False:
            while True:
                i = 10
        ' Confirm the requests architecture list is valid and configure SettingsManager\n        to run only the requested architectures.\n\n        Raise Exception if a list_of_requested_architectures is not supported\n        '
        unsupported = set(list_of_requested_architectures) - set(self.GetArchitecturesSupported())
        if len(unsupported) > 0:
            errorString = 'Unsupported Architecture Requested: ' + ' '.join(unsupported)
            logging.critical(errorString)
            raise Exception(errorString)
        self.ActualArchitectures = list_of_requested_architectures

    def GetWorkspaceRoot(self):
        if False:
            for i in range(10):
                print('nop')
        ' get WorkspacePath '
        return CommonPlatform.WorkspaceRoot

    def GetActiveScopes(self):
        if False:
            for i in range(10):
                print('nop')
        ' return tuple containing scopes that should be active for this process '
        return CommonPlatform.Scopes

    def FilterPackagesToTest(self, changedFilesList: list, potentialPackagesList: list) -> list:
        if False:
            while True:
                i = 10
        " Filter other cases that this package should be built\n        based on changed files. This should cover things that can't\n        be detected as dependencies. "
        build_these_packages = []
        possible_packages = potentialPackagesList.copy()
        for f in changedFilesList:
            if 'BaseTools' in f:
                if os.path.splitext(f) not in ['.txt', '.md']:
                    build_these_packages = possible_packages
                    break
            if 'platform-build-run-steps.yml' in f:
                build_these_packages = possible_packages
                break
        return build_these_packages

    def GetPlatformDscAndConfig(self) -> tuple:
        if False:
            return 10
        ' If a platform desires to provide its DSC then Policy 4 will evaluate if\n        any of the changes will be built in the dsc.\n\n        The tuple should be (<workspace relative path to dsc file>, <input dictionary of dsc key value pairs>)\n        '
        return (os.path.join('EmulatorPkg', 'EmulatorPkg.dsc'), {})

class PlatformBuilder(UefiBuilder, BuildSettingsManager):

    def __init__(self):
        if False:
            return 10
        UefiBuilder.__init__(self)

    def AddCommandLineOptions(self, parserObj):
        if False:
            return 10
        ' Add command line options to the argparser '
        parserObj.add_argument('-a', '--arch', dest='build_arch', type=str, default='X64', help='Optional - architecture to build.  IA32 will use IA32 for Pei & Dxe. X64 will use X64 for both PEI and DXE.')

    def RetrieveCommandLineOptions(self, args):
        if False:
            print('Hello World!')
        '  Retrieve command line options from the argparser '
        shell_environment.GetBuildVars().SetValue('TARGET_ARCH', args.build_arch.upper(), 'From CmdLine')
        shell_environment.GetBuildVars().SetValue('ACTIVE_PLATFORM', 'EmulatorPkg/EmulatorPkg.dsc', 'From CmdLine')

    def GetWorkspaceRoot(self):
        if False:
            print('Hello World!')
        ' get WorkspacePath '
        return CommonPlatform.WorkspaceRoot

    def GetPackagesPath(self):
        if False:
            return 10
        ' Return a list of workspace relative paths that should be mapped as edk2 PackagesPath '
        return ()

    def GetActiveScopes(self):
        if False:
            while True:
                i = 10
        ' return tuple containing scopes that should be active for this process '
        return CommonPlatform.Scopes

    def GetName(self):
        if False:
            i = 10
            return i + 15
        ' Get the name of the repo, platform, or product being build '
        ' Used for naming the log file, among others '
        if shell_environment.GetBuildVars().GetValue('MAKE_STARTUP_NSH', 'FALSE') == 'TRUE':
            return 'EmulatorPkg_With_Run'
        return 'EmulatorPkg'

    def GetLoggingLevel(self, loggerType):
        if False:
            while True:
                i = 10
        ' Get the logging level for a given type\n        base == lowest logging level supported\n        con  == Screen logging\n        txt  == plain text file logging\n        md   == markdown file logging\n        '
        return logging.DEBUG

    def SetPlatformEnv(self):
        if False:
            return 10
        logging.debug('PlatformBuilder SetPlatformEnv')
        self.env.SetValue('PRODUCT_NAME', 'EmulatorPkg', 'Platform Hardcoded')
        self.env.SetValue('TOOL_CHAIN_TAG', 'VS2019', 'Default Toolchain')
        if self.env.GetValue('TOOL_CHAIN_TAG') == 'VS2019' or self.env.GetValue('TOOL_CHAIN_TAG') == 'VS2017':
            key = self.env.GetValue('TOOL_CHAIN_TAG') + '_HOST'
            if self.env.GetValue('TARGET_ARCH') == 'IA32':
                shell_environment.ShellEnvironment().set_shell_var(key, 'x86')
            elif self.env.GetValue('TARGET_ARCH') == 'X64':
                shell_environment.ShellEnvironment().set_shell_var(key, 'x64')
        if GetHostInfo().os.upper() == 'LINUX':
            self.ConfigureLinuxDLinkPath()
        if GetHostInfo().os.upper() == 'WINDOWS':
            self.env.SetValue('BLD_*_WIN_HOST_BUILD', 'TRUE', 'Trigger Windows host build')
        self.env.SetValue('MAKE_STARTUP_NSH', 'FALSE', 'Default to false')
        key = 'BLD_*_BUILD_' + self.env.GetValue('TARGET_ARCH')
        self.env.SetValue(key, 'TRUE', 'match script in build.sh')
        return 0

    def PlatformPreBuild(self):
        if False:
            for i in range(10):
                print('nop')
        return 0

    def PlatformPostBuild(self):
        if False:
            return 10
        return 0

    def FlashRomImage(self):
        if False:
            for i in range(10):
                print('nop')
        ' Use the FlashRom Function to run the emulator.  This gives an easy stuart command line to\n        activate the emulator. '
        OutputPath = os.path.join(self.env.GetValue('BUILD_OUTPUT_BASE'), self.env.GetValue('TARGET_ARCH'))
        if self.env.GetValue('MAKE_STARTUP_NSH') == 'TRUE':
            f = open(os.path.join(OutputPath, 'startup.nsh'), 'w')
            f.write('BOOT SUCCESS !!! \n')
            f.write('reset\n')
            f.close()
        if GetHostInfo().os.upper() == 'WINDOWS':
            cmd = 'WinHost.exe'
        elif GetHostInfo().os.upper() == 'LINUX':
            cmd = './Host'
        else:
            logging.critical('Unsupported Host')
            return -1
        return RunCmd(cmd, '', workingdir=OutputPath)

    def ConfigureLinuxDLinkPath(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        logic copied from build.sh to setup the correct libraries\n        '
        if self.env.GetValue('TARGET_ARCH') == 'IA32':
            LIB_NAMES = ['ld-linux.so.2', 'libdl.so.2 crt1.o', 'crti.o crtn.o']
            LIB_SEARCH_PATHS = ['/usr/lib/i386-linux-gnu', '/usr/lib32', '/lib32', '/usr/lib', '/lib']
        elif self.env.GetValue('TARGET_ARCH') == 'X64':
            LIB_NAMES = ['ld-linux-x86-64.so.2', 'libdl.so.2', 'crt1.o', 'crti.o', 'crtn.o']
            LIB_SEARCH_PATHS = ['/usr/lib/x86_64-linux-gnu', '/usr/lib64', '/lib64', '/usr/lib', '/lib']
        HOST_DLINK_PATHS = ''
        for lname in LIB_NAMES:
            logging.debug(f'Looking for {lname}')
            for dname in LIB_SEARCH_PATHS:
                logging.debug(f'In {dname}')
                if os.path.isfile(os.path.join(dname, lname)):
                    logging.debug(f'Found {lname} in {dname}')
                    HOST_DLINK_PATHS += os.path.join(os.path.join(dname, lname)) + os.pathsep
                    break
        HOST_DLINK_PATHS = HOST_DLINK_PATHS.rstrip(os.pathsep)
        logging.critical(f'Setting HOST_DLINK_PATHS to {HOST_DLINK_PATHS}')
        shell_environment.ShellEnvironment().set_shell_var('HOST_DLINK_PATHS', HOST_DLINK_PATHS)