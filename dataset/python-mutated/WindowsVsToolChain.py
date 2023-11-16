import os
import logging
from edk2toolext.environment.plugintypes.uefi_build_plugin import IUefiBuildPlugin
import edk2toollib.windows.locate_tools as locate_tools
from edk2toollib.windows.locate_tools import FindWithVsWhere
from edk2toolext.environment import shell_environment
from edk2toolext.environment import version_aggregator
from edk2toollib.utility_functions import GetHostInfo

class WindowsVsToolChain(IUefiBuildPlugin):

    def do_post_build(self, thebuilder):
        if False:
            return 10
        return 0

    def do_pre_build(self, thebuilder):
        if False:
            i = 10
            return i + 15
        self.Logger = logging.getLogger('WindowsVsToolChain')
        interesting_keys = ['ExtensionSdkDir', 'INCLUDE', 'LIB', 'LIBPATH', 'UniversalCRTSdkDir', 'UCRTVersion', 'WindowsLibPath', 'WindowsSdkBinPath', 'WindowsSdkDir', 'WindowsSdkVerBinPath', 'WindowsSDKVersion', 'VCToolsInstallDir', 'Path']
        if thebuilder.env.GetValue('TOOL_CHAIN_TAG') == 'VS2017':
            HostType = shell_environment.GetEnvironment().get_shell_var('VS2017_HOST')
            if HostType is not None:
                HostType = HostType.lower()
                self.Logger.info(f'HOST TYPE defined by environment.  Host Type is {HostType}')
            else:
                HostInfo = GetHostInfo()
                if HostInfo.arch == 'x86':
                    if HostInfo.bit == '32':
                        HostType = 'x86'
                    elif HostInfo.bit == '64':
                        HostType = 'x64'
                else:
                    raise NotImplementedError()
            VC_HOST_ARCH_TRANSLATOR = {'x86': 'x86', 'x64': 'AMD64', 'arm': 'not supported', 'arm64': 'not supported'}
            if shell_environment.GetEnvironment().get_shell_var('VS2017_PREFIX') != None:
                self.Logger.info('VS2017_PREFIX is already set.')
            else:
                install_path = self._get_vs_install_path('VS2017'.lower(), 'VS150INSTALLPATH')
                vc_ver = self._get_vc_version(install_path, 'VS150TOOLVER')
                if install_path is None or vc_ver is None:
                    self.Logger.error('Failed to configure environment for VS2017')
                    return -1
                version_aggregator.GetVersionAggregator().ReportVersion('Visual Studio Install Path', install_path, version_aggregator.VersionTypes.INFO)
                version_aggregator.GetVersionAggregator().ReportVersion('VC Version', vc_ver, version_aggregator.VersionTypes.TOOL)
                prefix = os.path.join(install_path, 'VC', 'Tools', 'MSVC', vc_ver)
                prefix = prefix + os.path.sep
                shell_environment.GetEnvironment().set_shell_var('VS2017_PREFIX', prefix)
                shell_environment.GetEnvironment().set_shell_var('VS2017_HOST', HostType)
                shell_env = shell_environment.GetEnvironment()
                vs_vars = locate_tools.QueryVcVariables(interesting_keys, VC_HOST_ARCH_TRANSLATOR[HostType], vs_version='vs2017')
                for (k, v) in vs_vars.items():
                    shell_env.set_shell_var(k, v)
            if not os.path.exists(shell_environment.GetEnvironment().get_shell_var('VS2017_PREFIX')):
                self.Logger.error('Path for VS2017 toolchain is invalid')
                return -2
        elif thebuilder.env.GetValue('TOOL_CHAIN_TAG') == 'VS2019':
            HostType = shell_environment.GetEnvironment().get_shell_var('VS2019_HOST')
            if HostType is not None:
                HostType = HostType.lower()
                self.Logger.info(f'HOST TYPE defined by environment.  Host Type is {HostType}')
            else:
                HostInfo = GetHostInfo()
                if HostInfo.arch == 'x86':
                    if HostInfo.bit == '32':
                        HostType = 'x86'
                    elif HostInfo.bit == '64':
                        HostType = 'x64'
                else:
                    raise NotImplementedError()
            VC_HOST_ARCH_TRANSLATOR = {'x86': 'x86', 'x64': 'AMD64', 'arm': 'not supported', 'arm64': 'not supported'}
            if shell_environment.GetEnvironment().get_shell_var('VS2019_PREFIX') != None:
                self.Logger.info('VS2019_PREFIX is already set.')
            else:
                install_path = self._get_vs_install_path('VS2019'.lower(), 'VS160INSTALLPATH')
                vc_ver = self._get_vc_version(install_path, 'VS160TOOLVER')
                if install_path is None or vc_ver is None:
                    self.Logger.error('Failed to configure environment for VS2019')
                    return -1
                version_aggregator.GetVersionAggregator().ReportVersion('Visual Studio Install Path', install_path, version_aggregator.VersionTypes.INFO)
                version_aggregator.GetVersionAggregator().ReportVersion('VC Version', vc_ver, version_aggregator.VersionTypes.TOOL)
                prefix = os.path.join(install_path, 'VC', 'Tools', 'MSVC', vc_ver)
                prefix = prefix + os.path.sep
                shell_environment.GetEnvironment().set_shell_var('VS2019_PREFIX', prefix)
                shell_environment.GetEnvironment().set_shell_var('VS2019_HOST', HostType)
                shell_env = shell_environment.GetEnvironment()
                vs_vars = locate_tools.QueryVcVariables(interesting_keys, VC_HOST_ARCH_TRANSLATOR[HostType], vs_version='vs2019')
                for (k, v) in vs_vars.items():
                    shell_env.set_shell_var(k, v)
            if not os.path.exists(shell_environment.GetEnvironment().get_shell_var('VS2019_PREFIX')):
                self.Logger.error('Path for VS2019 toolchain is invalid')
                return -2
        return 0

    def _get_vs_install_path(self, vs_version, varname):
        if False:
            print('Hello World!')
        path = None
        if varname is not None:
            path = shell_environment.GetEnvironment().get_shell_var(varname)
        if path is None:
            try:
                path = FindWithVsWhere(vs_version=vs_version)
            except (EnvironmentError, ValueError, RuntimeError) as e:
                self.Logger.error(str(e))
                return None
            if path is not None and os.path.exists(path):
                self.Logger.debug('Found VS instance for %s', vs_version)
            else:
                self.Logger.error(f'VsWhere successfully executed, but could not find VS instance for {vs_version}.')
        return path

    def _get_vc_version(self, path, varname):
        if False:
            while True:
                i = 10
        vc_ver = shell_environment.GetEnvironment().get_shell_var(varname)
        if path is None:
            self.Logger.critical('Failed to find Visual Studio tools.  Might need to check for VS install')
            return vc_ver
        if vc_ver is None:
            p2 = os.path.join(path, 'VC', 'Tools', 'MSVC')
            if not os.path.isdir(p2):
                self.Logger.critical('Failed to find VC tools.  Might need to check for VS install')
                return vc_ver
            vc_ver = os.listdir(p2)[-1].strip()
            self.Logger.debug('Found VC Tool version is %s' % vc_ver)
        return vc_ver