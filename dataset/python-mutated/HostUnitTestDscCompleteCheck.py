import logging
import os
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toollib.uefi.edk2.parsers.dsc_parser import DscParser
from edk2toollib.uefi.edk2.parsers.inf_parser import InfParser, AllPhases
from edk2toolext.environment.var_dict import VarDict

class HostUnitTestDscCompleteCheck(ICiBuildPlugin):
    """
    A CiBuildPlugin that scans the package Host Unit Test dsc file and confirms all Host application modules (inf files) are
    listed in the components sections.

    Configuration options:
    "HostUnitTestDscCompleteCheck": {
        "DscPath": "", # Path to Host based unit test DSC file
        "IgnoreInf": []  # Ignore INF if found in filesystem but not dsc
    }
    """

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            print('Hello World!')
        ' Provide the testcase name and classname for use in reporting\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n                testclassname: a descriptive string for the testcase can include whitespace\n                classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n        '
        return ('Check the ' + packagename + ' Host Unit Test DSC for a being complete', packagename + '.HostUnitTestDscCompleteCheck')

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            while True:
                i = 10
        overall_status = 0
        if 'DscPath' not in pkgconfig:
            tc.SetSkipped()
            tc.LogStdError('DscPath not found in config file.  Nothing to check.')
            return -1
        abs_pkg_path = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        abs_dsc_path = os.path.join(abs_pkg_path, pkgconfig['DscPath'].strip())
        wsr_dsc_path = Edk2pathObj.GetEdk2RelativePathFromAbsolutePath(abs_dsc_path)
        if abs_dsc_path is None or wsr_dsc_path == '' or (not os.path.isfile(abs_dsc_path)):
            tc.SetSkipped()
            tc.LogStdError('Package Host Unit Test Dsc not found')
            return 0
        INFFiles = self.WalkDirectoryForExtension(['.inf'], abs_pkg_path)
        INFFiles = [Edk2pathObj.GetEdk2RelativePathFromAbsolutePath(x) for x in INFFiles]
        if 'IgnoreInf' in pkgconfig:
            for a in pkgconfig['IgnoreInf']:
                a = a.replace(os.sep, '/')
                try:
                    tc.LogStdOut('Ignoring INF {0}'.format(a))
                    INFFiles.remove(a)
                except:
                    tc.LogStdError('HostUnitTestDscCompleteCheck.IgnoreInf -> {0} not found in filesystem.  Invalid ignore file'.format(a))
                    logging.info('HostUnitTestDscCompleteCheck.IgnoreInf -> {0} not found in filesystem.  Invalid ignore file'.format(a))
        dp = DscParser()
        dp.SetBaseAbsPath(Edk2pathObj.WorkspacePath)
        dp.SetPackagePaths(Edk2pathObj.PackagePathList)
        dp.SetInputVars(environment.GetAllBuildKeyValues())
        dp.ParseFile(wsr_dsc_path)
        for INF in INFFiles:
            if not any((INF.strip() in x for x in dp.ThreeMods)) and (not any((INF.strip() in x for x in dp.SixMods))) and (not any((INF.strip() in x for x in dp.OtherMods))):
                infp = InfParser().SetBaseAbsPath(Edk2pathObj.WorkspacePath)
                infp.SetPackagePaths(Edk2pathObj.PackagePathList)
                infp.ParseFile(INF)
                if 'MODULE_TYPE' not in infp.Dict:
                    tc.LogStdOut('Ignoring INF. Missing key for MODULE_TYPE {0}'.format(INF))
                    continue
                if infp.Dict['MODULE_TYPE'] == 'HOST_APPLICATION':
                    pass
                elif len(infp.SupportedPhases) > 0 and 'HOST_APPLICATION' in infp.SupportedPhases and (infp.SupportedPhases != AllPhases):
                    pass
                else:
                    tc.LogStdOut('Ignoring INF. MODULE_TYPE or suppored phases not HOST_APPLICATION {0}'.format(INF))
                    continue
                logging.critical(INF + ' not in ' + wsr_dsc_path)
                tc.LogStdError('{0} not in {1}'.format(INF, wsr_dsc_path))
                overall_status = overall_status + 1
        if overall_status != 0:
            tc.SetFailed('HostUnitTestDscCompleteCheck {0} Failed.  Errors {1}'.format(wsr_dsc_path, overall_status), 'CHECK_FAILED')
        else:
            tc.SetSuccess()
        return overall_status