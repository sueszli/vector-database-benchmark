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

class SettingsManager(UpdateSettingsManager, SetupSettingsManager, PrEvalSettingsManager):

    def GetPackagesSupported(self):
        if False:
            return 10
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
            print('Hello World!')
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
            return 10
        ' get WorkspacePath '
        return CommonPlatform.WorkspaceRoot

    def GetActiveScopes(self):
        if False:
            return 10
        ' return tuple containing scopes that should be active for this process '
        return CommonPlatform.Scopes

    def FilterPackagesToTest(self, changedFilesList: list, potentialPackagesList: list) -> list:
        if False:
            for i in range(10):
                print('nop')
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
            while True:
                i = 10
        ' If a platform desires to provide its DSC then Policy 4 will evaluate if\n        any of the changes will be built in the dsc.\n\n        The tuple should be (<workspace relative path to dsc file>, <input dictionary of dsc key value pairs>)\n        '
        dsc = CommonPlatform.GetDscName(','.join(self.ActualArchitectures))
        return (f'OvmfPkg/{dsc}', {})

class PlatformBuilder(UefiBuilder, BuildSettingsManager):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        UefiBuilder.__init__(self)

    def AddCommandLineOptions(self, parserObj):
        if False:
            for i in range(10):
                print('nop')
        ' Add command line options to the argparser '
        parserObj.add_argument('-a', '--arch', dest='build_arch', type=str, default='IA32,X64', help='Optional - CSV of architecture to build.  IA32 will use IA32 for Pei & Dxe. X64 will use X64 for both PEI and DXE.  IA32,X64 will use IA32 for PEI and X64 for DXE. default is IA32,X64')

    def RetrieveCommandLineOptions(self, args):
        if False:
            return 10
        '  Retrieve command line options from the argparser '
        shell_environment.GetBuildVars().SetValue('TARGET_ARCH', ' '.join(args.build_arch.upper().split(',')), 'From CmdLine')
        dsc = CommonPlatform.GetDscName(args.build_arch)
        shell_environment.GetBuildVars().SetValue('ACTIVE_PLATFORM', f'OvmfPkg/{dsc}', 'From CmdLine')

    def GetWorkspaceRoot(self):
        if False:
            print('Hello World!')
        ' get WorkspacePath '
        return CommonPlatform.WorkspaceRoot

    def GetPackagesPath(self):
        if False:
            print('Hello World!')
        ' Return a list of workspace relative paths that should be mapped as edk2 PackagesPath '
        return ()

    def GetActiveScopes(self):
        if False:
            return 10
        ' return tuple containing scopes that should be active for this process '
        return CommonPlatform.Scopes

    def GetName(self):
        if False:
            i = 10
            return i + 15
        ' Get the name of the repo, platform, or product being build '
        ' Used for naming the log file, among others '
        if shell_environment.GetBuildVars().GetValue('MAKE_STARTUP_NSH', 'FALSE') == 'TRUE':
            return 'OvmfPkg_With_Run'
        return 'OvmfPkg'

    def GetLoggingLevel(self, loggerType):
        if False:
            i = 10
            return i + 15
        ' Get the logging level for a given type\n        base == lowest logging level supported\n        con  == Screen logging\n        txt  == plain text file logging\n        md   == markdown file logging\n        '
        return logging.DEBUG

    def SetPlatformEnv(self):
        if False:
            while True:
                i = 10
        logging.debug('PlatformBuilder SetPlatformEnv')
        self.env.SetValue('PRODUCT_NAME', 'OVMF', 'Platform Hardcoded')
        self.env.SetValue('MAKE_STARTUP_NSH', 'FALSE', 'Default to false')
        self.env.SetValue('QEMU_HEADLESS', 'FALSE', 'Default to false')
        self.env.SetValue('DISABLE_DEBUG_MACRO_CHECK', 'TRUE', 'Disable by default')
        return 0

    def PlatformPreBuild(self):
        if False:
            i = 10
            return i + 15
        return 0

    def PlatformPostBuild(self):
        if False:
            while True:
                i = 10
        return 0

    def FlashRomImage(self):
        if False:
            while True:
                i = 10
        VirtualDrive = os.path.join(self.env.GetValue('BUILD_OUTPUT_BASE'), 'VirtualDrive')
        os.makedirs(VirtualDrive, exist_ok=True)
        OutputPath_FV = os.path.join(self.env.GetValue('BUILD_OUTPUT_BASE'), 'FV')
        if self.env.GetValue('QEMU_SKIP') and self.env.GetValue('QEMU_SKIP').upper() == 'TRUE':
            logging.info('skipping qemu boot test')
            return 0
        cmd = 'qemu-system-x86_64'
        args = '-debugcon stdio'
        args += ' -global isa-debugcon.iobase=0x402'
        args += ' -net none'
        args += ' -smp 4'
        args += f' -drive file=fat:rw:{VirtualDrive},format=raw,media=disk'
        if self.env.GetValue('QEMU_HEADLESS').upper() == 'TRUE':
            args += ' -display none'
        if self.env.GetBuildValue('SMM_REQUIRE') == '1':
            args += ' -machine q35,smm=on'
            args += ' --accel tcg,thread=single'
            args += ' -global driver=cfi.pflash01,property=secure,value=on'
            args += ' -drive if=pflash,format=raw,unit=0,file=' + os.path.join(OutputPath_FV, 'OVMF_CODE.fd') + ',readonly=on'
            args += ' -drive if=pflash,format=raw,unit=1,file=' + os.path.join(OutputPath_FV, 'OVMF_VARS.fd')
        else:
            args += ' -pflash ' + os.path.join(OutputPath_FV, 'OVMF.fd')
        if self.env.GetValue('MAKE_STARTUP_NSH').upper() == 'TRUE':
            f = open(os.path.join(VirtualDrive, 'startup.nsh'), 'w')
            f.write('BOOT SUCCESS !!! \n')
            f.write('reset -s\n')
            f.close()
        ret = RunCmd(cmd, args)
        if ret == 3221225477:
            return 0
        return ret