import logging
import os
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toollib.uefi.edk2.parsers.dec_parser import DecParser
from edk2toollib.uefi.edk2.parsers.inf_parser import InfParser
from edk2toolext.environment.var_dict import VarDict

class LibraryClassCheck(ICiBuildPlugin):
    """
    A CiBuildPlugin that scans the code tree and library classes for undeclared
    files

    Configuration options:
    "LibraryClassCheck": {
        IgnoreHeaderFile: [],  # Ignore a file found on disk
        IgnoreLibraryClass: [] # Ignore a declaration found in dec file
    }
    """

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            return 10
        ' Provide the testcase name and classname for use in reporting\n            testclassname: a descriptive string for the testcase can include whitespace\n            classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n        '
        return ('Check library class declarations in ' + packagename, packagename + '.LibraryClassCheck')

    def __GetPkgDec(self, rootpath):
        if False:
            while True:
                i = 10
        try:
            allEntries = os.listdir(rootpath)
            for entry in allEntries:
                if entry.lower().endswith('.dec'):
                    return os.path.join(rootpath, entry)
        except Exception:
            logging.error('Unable to find DEC for package:{0}'.format(rootpath))
        return None

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            i = 10
            return i + 15
        overall_status = 0
        LibraryClassIgnore = []
        abs_pkg_path = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        abs_dec_path = self.__GetPkgDec(abs_pkg_path)
        wsr_dec_path = Edk2pathObj.GetEdk2RelativePathFromAbsolutePath(abs_dec_path)
        if abs_dec_path is None or wsr_dec_path == '' or (not os.path.isfile(abs_dec_path)):
            tc.SetSkipped()
            tc.LogStdError('No DEC file {0} in package {1}'.format(abs_dec_path, abs_pkg_path))
            return -1
        dec = DecParser()
        dec.SetBaseAbsPath(Edk2pathObj.WorkspacePath).SetPackagePaths(Edk2pathObj.PackagePathList)
        dec.ParseFile(wsr_dec_path)
        AllHeaderFiles = []
        for includepath in dec.IncludePaths:
            AbsLibraryIncludePath = os.path.join(abs_pkg_path, includepath, 'Library')
            if not os.path.isdir(AbsLibraryIncludePath):
                continue
            hfiles = self.WalkDirectoryForExtension(['.h'], AbsLibraryIncludePath)
            hfiles = [os.path.relpath(x, abs_pkg_path) for x in hfiles]
            hfiles = [x.replace('\\', '/') for x in hfiles]
            AllHeaderFiles.extend(hfiles)
        if len(AllHeaderFiles) == 0:
            tc.SetSkipped()
            tc.LogStdError(f'No Library include folder in any Include path')
            return -1
        if 'IgnoreHeaderFile' in pkgconfig:
            for a in pkgconfig['IgnoreHeaderFile']:
                try:
                    tc.LogStdOut('Ignoring Library Header File {0}'.format(a))
                    AllHeaderFiles.remove(a)
                except:
                    tc.LogStdError('LibraryClassCheck.IgnoreHeaderFile -> {0} not found.  Invalid Header File'.format(a))
                    logging.info('LibraryClassCheck.IgnoreHeaderFile -> {0} not found.  Invalid Header File'.format(a))
        if 'IgnoreLibraryClass' in pkgconfig:
            LibraryClassIgnore = pkgconfig['IgnoreLibraryClass']
        for lcd in dec.LibraryClasses:
            if '\\' in lcd.path:
                tc.LogStdError('LibraryClassCheck.DecFilePathSeparator -> {0} invalid.'.format(lcd.path))
                logging.error('LibraryClassCheck.DecFilePathSeparator -> {0} invalid.'.format(lcd.path))
                overall_status += 1
                continue
            if lcd.name in LibraryClassIgnore:
                tc.LogStdOut('Ignoring Library Class Name {0}'.format(lcd.name))
                LibraryClassIgnore.remove(lcd.name)
                continue
            logging.debug(f'Looking for Library Class {lcd.path}')
            try:
                AllHeaderFiles.remove(lcd.path)
            except ValueError:
                tc.LogStdError(f'Library {lcd.name} with path {lcd.path} not found in package filesystem')
                logging.error(f'Library {lcd.name} with path {lcd.path} not found in package filesystem')
                overall_status += 1
        for h in AllHeaderFiles:
            tc.LogStdError(f'Library Header File {h} not declared in package DEC {wsr_dec_path}')
            logging.error(f'Library Header File {h} not declared in package DEC {wsr_dec_path}')
            overall_status += 1
        for r in LibraryClassIgnore:
            tc.LogStdError('LibraryClassCheck.IgnoreLibraryClass -> {0} not found.  Library Class not found'.format(r))
            logging.info('LibraryClassCheck.IgnoreLibraryClass -> {0} not found.  Library Class not found'.format(r))
        if overall_status != 0:
            tc.SetFailed('LibraryClassCheck {0} Failed.  Errors {1}'.format(wsr_dec_path, overall_status), 'CHECK_FAILED')
        else:
            tc.SetSuccess()
        return overall_status