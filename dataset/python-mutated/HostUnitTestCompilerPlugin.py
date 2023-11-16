import logging
import os
import re
from edk2toollib.uefi.edk2.parsers.dsc_parser import DscParser
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toolext.environment.uefi_build import UefiBuilder
from edk2toolext import edk2_logging
from edk2toolext.environment.var_dict import VarDict
from edk2toollib.utility_functions import GetHostInfo

class HostUnitTestCompilerPlugin(ICiBuildPlugin):
    """
    A CiBuildPlugin that compiles the dsc for host based unit test apps.
    An IUefiBuildPlugin may be attached to this plugin that will run the
    unit tests and collect the results after successful compilation.

    Configuration options:
    "HostUnitTestCompilerPlugin": {
        "DscPath": "<path to dsc from root of pkg>"
    }
    """

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            i = 10
            return i + 15
        ' Provide the testcase name and classname for use in reporting\n            testclassname: a descriptive string for the testcase can include whitespace\n            classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n        '
        (num, types) = self.__GetHostUnitTestArch(environment)
        types = types.replace(' ', '_')
        return ('Compile and Run Host-Based UnitTests for ' + packagename + ' on arch ' + types, packagename + '.HostUnitTestCompiler.' + types)

    def RunsOnTargetList(self):
        if False:
            while True:
                i = 10
        return ['NOOPT']

    def __GetHostUnitTestArch(self, environment):
        if False:
            print('Hello World!')
        requested = environment.GetValue('TARGET_ARCH').split(' ')
        host = []
        if GetHostInfo().arch == 'x86':
            if GetHostInfo().bit == '64':
                host.append('X64')
        elif GetHostInfo().arch == 'ARM':
            if GetHostInfo().bit == '64':
                host.append('AARCH64')
            elif GetHostInfo().bit == '32':
                host.append('ARM')
        willrun = set(requested) & set(host)
        return (len(willrun), ' '.join(willrun))

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            i = 10
            return i + 15
        self._env = environment
        environment.SetValue('CI_BUILD_TYPE', 'host_unit_test', 'Set in HostUnitTestCompilerPlugin')
        if 'DscPath' not in pkgconfig:
            tc.SetSkipped()
            tc.LogStdError('DscPath not found in config file.  Nothing to compile for HostBasedUnitTests.')
            return -1
        AP = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        APDSC = os.path.join(AP, pkgconfig['DscPath'].strip())
        AP_Path = Edk2pathObj.GetEdk2RelativePathFromAbsolutePath(APDSC)
        if AP is None or AP_Path is None or (not os.path.isfile(APDSC)):
            tc.SetSkipped()
            tc.LogStdError('Package HostBasedUnitTest Dsc not found.')
            return -1
        logging.info('Building {0}'.format(AP_Path))
        self._env.SetValue('ACTIVE_PLATFORM', AP_Path, 'Set in Compiler Plugin')
        (num, RUNNABLE_ARCHITECTURES) = self.__GetHostUnitTestArch(environment)
        if num == 0:
            tc.SetSkipped()
            tc.LogStdError('No host architecture compatibility')
            return -1
        if not environment.SetValue('TARGET_ARCH', RUNNABLE_ARCHITECTURES, 'Update Target Arch based on Host Support'):
            environment.AllowOverride('TARGET_ARCH')
            if not environment.SetValue('TARGET_ARCH', RUNNABLE_ARCHITECTURES, 'Update Target Arch based on Host Support'):
                raise RuntimeError("Can't Change TARGET_ARCH as required")
        dp = DscParser()
        dp.SetBaseAbsPath(Edk2pathObj.WorkspacePath)
        dp.SetPackagePaths(Edk2pathObj.PackagePathList)
        dp.ParseFile(AP_Path)
        if 'SUPPORTED_ARCHITECTURES' in dp.LocalVars:
            SUPPORTED_ARCHITECTURES = dp.LocalVars['SUPPORTED_ARCHITECTURES'].split('|')
            TARGET_ARCHITECTURES = environment.GetValue('TARGET_ARCH').split(' ')
            if len(set(SUPPORTED_ARCHITECTURES) & set(TARGET_ARCHITECTURES)) == 0:
                tc.SetSkipped()
                tc.LogStdError('No supported architecutres to build for host unit tests')
                return -1
        uefiBuilder = UefiBuilder()
        ret = uefiBuilder.Go(Edk2pathObj.WorkspacePath, os.pathsep.join(Edk2pathObj.PackagePathList), PLMHelper, PLM)
        if ret != 0:
            tc.SetFailed('Compile failed for {0}'.format(packagename), 'Compile_FAILED')
            tc.LogStdError('{0} Compile failed with error code {1} '.format(AP_Path, ret))
            return 1
        else:
            tc.SetSuccess()
            return 0