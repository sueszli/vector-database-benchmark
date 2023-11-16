import logging
import os
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toollib.uefi.edk2.parsers.inf_parser import InfParser
from edk2toolext.environment.var_dict import VarDict

class DependencyCheck(ICiBuildPlugin):
    """
    A CiBuildPlugin that finds all modules (inf files) in a package and reviews the packages used
    to confirm they are acceptable.  This is to help enforce layering and identify improper
    dependencies between packages.

    Configuration options:
    "DependencyCheck": {
        "AcceptableDependencies": [], # Package dec files that are allowed in all INFs.  Example: MdePkg/MdePkg.dec
        "AcceptableDependencies-<MODULE_TYPE>": [], # OPTIONAL Package dependencies for INFs that are HOST_APPLICATION
        "AcceptableDependencies-HOST_APPLICATION": [], # EXAMPLE Package dependencies for INFs that are HOST_APPLICATION
        "IgnoreInf": []  # Ignore INF if found in filesystem
    }
    """

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            while True:
                i = 10
        ' Provide the testcase name and classname for use in reporting\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n                testclassname: a descriptive string for the testcase can include whitespace\n                classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n        '
        return ('Test Package Dependencies for modules in ' + packagename, packagename + '.DependencyCheck')

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            while True:
                i = 10
        overall_status = 0
        abs_pkg_path = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        INFFiles = self.WalkDirectoryForExtension(['.inf'], abs_pkg_path)
        INFFiles = [Edk2pathObj.GetEdk2RelativePathFromAbsolutePath(x) for x in INFFiles]
        if 'IgnoreInf' in pkgconfig:
            for a in pkgconfig['IgnoreInf']:
                a = a.replace(os.sep, '/')
                try:
                    INFFiles.remove(a)
                    tc.LogStdOut('IgnoreInf {0}'.format(a))
                except:
                    logging.info('DependencyConfig.IgnoreInf -> {0} not found in filesystem.  Invalid ignore file'.format(a))
                    tc.LogStdError('DependencyConfig.IgnoreInf -> {0} not found in filesystem.  Invalid ignore file'.format(a))
        if 'AcceptableDependencies' not in pkgconfig:
            logging.info('DependencyCheck Skipped.  No Acceptable Dependencies defined.')
            tc.LogStdOut('DependencyCheck Skipped.  No Acceptable Dependencies defined.')
            tc.SetSkipped()
            return -1
        for k in pkgconfig.keys():
            if k.startswith('AcceptableDependencies'):
                pkgstring = '\n'.join(pkgconfig[k])
                if '-' in k:
                    (_, _, mod_type) = k.partition('-')
                    tc.LogStdOut(f'Additional dependencies for MODULE_TYPE {mod_type}:\n {pkgstring}')
                else:
                    tc.LogStdOut(f'Acceptable Dependencies:\n {pkgstring}')
        for file in INFFiles:
            ip = InfParser()
            logging.debug('Parsing ' + file)
            ip.SetBaseAbsPath(Edk2pathObj.WorkspacePath).SetPackagePaths(Edk2pathObj.PackagePathList).ParseFile(file)
            if 'MODULE_TYPE' not in ip.Dict:
                tc.LogStdOut('Ignoring INF. Missing key for MODULE_TYPE {0}'.format(file))
                continue
            mod_type = ip.Dict['MODULE_TYPE'].upper()
            for p in ip.PackagesUsed:
                if p not in pkgconfig['AcceptableDependencies']:
                    mod_specific_key = 'AcceptableDependencies-' + mod_type
                    if mod_specific_key in pkgconfig and p in pkgconfig[mod_specific_key]:
                        continue
                    logging.error('Dependency Check: Invalid Dependency INF: {0} depends on pkg {1}'.format(file, p))
                    tc.LogStdError('Dependency Check: Invalid Dependency INF: {0} depends on pkg {1}'.format(file, p))
                    overall_status += 1
        if overall_status != 0:
            tc.SetFailed('Failed with {0} errors'.format(overall_status), 'DEPENDENCYCHECK_FAILED')
        else:
            tc.SetSuccess()
        return overall_status