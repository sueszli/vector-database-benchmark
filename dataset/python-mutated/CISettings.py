import os
import logging
import sys
from edk2toolext.environment import shell_environment
from edk2toolext.invocables.edk2_ci_build import CiBuildSettingsManager
from edk2toolext.invocables.edk2_setup import SetupSettingsManager, RequiredSubmodule
from edk2toolext.invocables.edk2_update import UpdateSettingsManager
from edk2toolext.invocables.edk2_pr_eval import PrEvalSettingsManager
from edk2toollib.utility_functions import GetHostInfo
from pathlib import Path
try:
    root = Path(__file__).parent.parent.resolve()
    sys.path.append(str(root / 'BaseTools' / 'Plugin' / 'CodeQL' / 'integration'))
    import stuart_codeql as codeql_helpers
except ImportError:
    pass

class Settings(CiBuildSettingsManager, UpdateSettingsManager, SetupSettingsManager, PrEvalSettingsManager):

    def __init__(self):
        if False:
            print('Hello World!')
        self.ActualPackages = []
        self.ActualTargets = []
        self.ActualArchitectures = []
        self.ActualToolChainTag = ''
        self.UseBuiltInBaseTools = None
        self.ActualScopes = None

    def AddCommandLineOptions(self, parserObj):
        if False:
            while True:
                i = 10
        group = parserObj.add_mutually_exclusive_group()
        group.add_argument('-force_piptools', '--fpt', dest='force_piptools', action='store_true', default=False, help='Force the system to use pip tools')
        group.add_argument('-no_piptools', '--npt', dest='no_piptools', action='store_true', default=False, help='Force the system to not use pip tools')
        try:
            codeql_helpers.add_command_line_option(parserObj)
        except NameError:
            pass

    def RetrieveCommandLineOptions(self, args):
        if False:
            print('Hello World!')
        super().RetrieveCommandLineOptions(args)
        if args.force_piptools:
            self.UseBuiltInBaseTools = True
        if args.no_piptools:
            self.UseBuiltInBaseTools = False
        try:
            self.codeql = codeql_helpers.is_codeql_enabled_on_command_line(args)
        except NameError:
            pass

    def GetPackagesSupported(self):
        if False:
            return 10
        ' return iterable of edk2 packages supported by this build.\n        These should be edk2 workspace relative paths '
        return ('ArmPkg', 'ArmPlatformPkg', 'ArmVirtPkg', 'DynamicTablesPkg', 'EmbeddedPkg', 'EmulatorPkg', 'IntelFsp2Pkg', 'IntelFsp2WrapperPkg', 'MdePkg', 'MdeModulePkg', 'NetworkPkg', 'PcAtChipsetPkg', 'SecurityPkg', 'UefiCpuPkg', 'FmpDevicePkg', 'ShellPkg', 'SignedCapsulePkg', 'StandaloneMmPkg', 'FatPkg', 'CryptoPkg', 'PrmPkg', 'UnitTestFrameworkPkg', 'OvmfPkg', 'RedfishPkg', 'SourceLevelDebugPkg', 'UefiPayloadPkg')

    def GetArchitecturesSupported(self):
        if False:
            for i in range(10):
                print('nop')
        ' return iterable of edk2 architectures supported by this build '
        return ('IA32', 'X64', 'ARM', 'AARCH64', 'RISCV64', 'LOONGARCH64')

    def GetTargetsSupported(self):
        if False:
            for i in range(10):
                print('nop')
        ' return iterable of edk2 target tags supported by this build '
        return ('DEBUG', 'RELEASE', 'NO-TARGET', 'NOOPT')

    def SetPackages(self, list_of_requested_packages):
        if False:
            while True:
                i = 10
        ' Confirm the requested package list is valid and configure SettingsManager\n        to build the requested packages.\n\n        Raise UnsupportedException if a requested_package is not supported\n        '
        unsupported = set(list_of_requested_packages) - set(self.GetPackagesSupported())
        if len(unsupported) > 0:
            logging.critical('Unsupported Package Requested: ' + ' '.join(unsupported))
            raise Exception('Unsupported Package Requested: ' + ' '.join(unsupported))
        self.ActualPackages = list_of_requested_packages

    def SetArchitectures(self, list_of_requested_architectures):
        if False:
            return 10
        ' Confirm the requests architecture list is valid and configure SettingsManager\n        to run only the requested architectures.\n\n        Raise Exception if a list_of_requested_architectures is not supported\n        '
        unsupported = set(list_of_requested_architectures) - set(self.GetArchitecturesSupported())
        if len(unsupported) > 0:
            logging.critical('Unsupported Architecture Requested: ' + ' '.join(unsupported))
            raise Exception('Unsupported Architecture Requested: ' + ' '.join(unsupported))
        self.ActualArchitectures = list_of_requested_architectures

    def SetTargets(self, list_of_requested_target):
        if False:
            while True:
                i = 10
        ' Confirm the request target list is valid and configure SettingsManager\n        to run only the requested targets.\n\n        Raise UnsupportedException if a requested_target is not supported\n        '
        unsupported = set(list_of_requested_target) - set(self.GetTargetsSupported())
        if len(unsupported) > 0:
            logging.critical('Unsupported Targets Requested: ' + ' '.join(unsupported))
            raise Exception('Unsupported Targets Requested: ' + ' '.join(unsupported))
        self.ActualTargets = list_of_requested_target

    def GetActiveScopes(self):
        if False:
            while True:
                i = 10
        ' return tuple containing scopes that should be active for this process '
        if self.ActualScopes is None:
            scopes = ('cibuild', 'edk2-build', 'host-based-test')
            self.ActualToolChainTag = shell_environment.GetBuildVars().GetValue('TOOL_CHAIN_TAG', '')
            is_linux = GetHostInfo().os.upper() == 'LINUX'
            if self.UseBuiltInBaseTools is None:
                is_linux = GetHostInfo().os.upper() == 'LINUX'
                try:
                    import edk2basetools
                    self.UseBuiltInBaseTools = True
                except ImportError:
                    self.UseBuiltInBaseTools = False
                    pass
            if self.UseBuiltInBaseTools == True:
                scopes += ('pipbuild-unix',) if is_linux else ('pipbuild-win',)
                logging.warning('Using Pip Tools based BaseTools')
            else:
                logging.warning('Falling back to using in-tree BaseTools')
            try:
                scopes += codeql_helpers.get_scopes(self.codeql)
                if self.codeql:
                    shell_environment.GetBuildVars().SetValue('STUART_CODEQL_AUDIT_ONLY', 'TRUE', 'Set in CISettings.py')
            except NameError:
                pass
            self.ActualScopes = scopes
        return self.ActualScopes

    def GetRequiredSubmodules(self):
        if False:
            return 10
        ' return iterable containing RequiredSubmodule objects.\n        If no RequiredSubmodules return an empty iterable\n        '
        rs = []
        rs.append(RequiredSubmodule('ArmPkg/Library/ArmSoftFloatLib/berkeley-softfloat-3', False))
        rs.append(RequiredSubmodule('CryptoPkg/Library/OpensslLib/openssl', False))
        rs.append(RequiredSubmodule('UnitTestFrameworkPkg/Library/CmockaLib/cmocka', False))
        rs.append(RequiredSubmodule('UnitTestFrameworkPkg/Library/GoogleTestLib/googletest', False))
        rs.append(RequiredSubmodule('MdeModulePkg/Universal/RegularExpressionDxe/oniguruma', False))
        rs.append(RequiredSubmodule('MdeModulePkg/Library/BrotliCustomDecompressLib/brotli', False))
        rs.append(RequiredSubmodule('BaseTools/Source/C/BrotliCompress/brotli', False))
        rs.append(RequiredSubmodule('RedfishPkg/Library/JsonLib/jansson', False))
        rs.append(RequiredSubmodule('UnitTestFrameworkPkg/Library/SubhookLib/subhook', False))
        rs.append(RequiredSubmodule('MdePkg/Library/BaseFdtLib/libfdt', False))
        rs.append(RequiredSubmodule('MdePkg/Library/MipiSysTLib/mipisyst', False))
        rs.append(RequiredSubmodule('CryptoPkg/Library/MbedTlsLib/mbedtls', False))
        return rs

    def GetName(self):
        if False:
            i = 10
            return i + 15
        return 'Edk2'

    def GetDependencies(self):
        if False:
            print('Hello World!')
        return []

    def GetPackagesPath(self):
        if False:
            i = 10
            return i + 15
        return ()

    def GetWorkspaceRoot(self):
        if False:
            print('Hello World!')
        ' get WorkspacePath '
        return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def FilterPackagesToTest(self, changedFilesList: list, potentialPackagesList: list) -> list:
        if False:
            for i in range(10):
                print('nop')
        ' Filter potential packages to test based on changed files. '
        build_these_packages = []
        possible_packages = potentialPackagesList.copy()
        for f in changedFilesList:
            nodes = f.split('/')
            if f.endswith('.py') and '.pytool' in nodes:
                build_these_packages = possible_packages
                break
            if 'BaseTools' in nodes:
                if os.path.splitext(f) not in ['.txt', '.md']:
                    build_these_packages = possible_packages
                    break
        return build_these_packages