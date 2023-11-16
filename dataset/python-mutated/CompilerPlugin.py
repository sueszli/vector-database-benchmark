import logging
import os
import re
from edk2toollib.uefi.edk2.parsers.dsc_parser import DscParser
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toolext.environment.uefi_build import UefiBuilder
from edk2toolext import edk2_logging
from edk2toolext.environment.var_dict import VarDict

class CompilerPlugin(ICiBuildPlugin):
    """
    A CiBuildPlugin that compiles the package dsc
    from the package being tested.

    Configuration options:
    "CompilerPlugin": {
        "DscPath": "<path to dsc from root of pkg>"
    }
    """

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            for i in range(10):
                print('nop')
        ' Provide the testcase name and classname for use in reporting\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n        '
        target = environment.GetValue('TARGET')
        return ('Compile ' + packagename + ' ' + target, packagename + '.Compiler.' + target)

    def RunsOnTargetList(self):
        if False:
            i = 10
            return i + 15
        return ['DEBUG', 'RELEASE']

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            i = 10
            return i + 15
        self._env = environment
        if 'DscPath' not in pkgconfig:
            tc.SetSkipped()
            tc.LogStdError('DscPath not found in config file.  Nothing to compile.')
            return -1
        AP = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        APDSC = os.path.join(AP, pkgconfig['DscPath'].strip())
        AP_Path = Edk2pathObj.GetEdk2RelativePathFromAbsolutePath(APDSC)
        if AP is None or AP_Path is None or (not os.path.isfile(APDSC)):
            tc.SetSkipped()
            tc.LogStdError('Package Dsc not found.')
            return -1
        logging.info('Building {0}'.format(AP_Path))
        self._env.SetValue('ACTIVE_PLATFORM', AP_Path, 'Set in Compiler Plugin')
        dp = DscParser()
        dp.SetBaseAbsPath(Edk2pathObj.WorkspacePath)
        dp.SetPackagePaths(Edk2pathObj.PackagePathList)
        dp.ParseFile(AP_Path)
        if 'SUPPORTED_ARCHITECTURES' in dp.LocalVars:
            SUPPORTED_ARCHITECTURES = dp.LocalVars['SUPPORTED_ARCHITECTURES'].split('|')
            TARGET_ARCHITECTURES = environment.GetValue('TARGET_ARCH').split(' ')
            if len(set(SUPPORTED_ARCHITECTURES) & set(TARGET_ARCHITECTURES)) == 0:
                tc.SetSkipped()
                tc.LogStdError('No supported architecutres to build')
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