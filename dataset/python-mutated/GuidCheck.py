import logging
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toollib.uefi.edk2.guid_list import GuidList
from edk2toolext.environment.var_dict import VarDict

class GuidCheck(ICiBuildPlugin):
    """
    A CiBuildPlugin that scans the code tree and looks for duplicate guids
    from the package being tested.

    Configuration options:
    "GuidCheck": {
        "IgnoreGuidName": [], # provide in format guidname=guidvalue or just guidname
        "IgnoreGuidValue": [],
        "IgnoreFoldersAndFiles": [],
        "IgnoreDuplicates": [] # Provide in format guidname=guidname=guidname...
    }
    """

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            return 10
        ' Provide the testcase name and classname for use in reporting\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n                testclassname: a descriptive string for the testcase can include whitespace\n                classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n        '
        return ('Confirm GUIDs are unique in ' + packagename, packagename + '.GuidCheck')

    def _FindConflictingGuidValues(self, guidlist: list) -> list:
        if False:
            print('Hello World!')
        ' Find all duplicate guids by guid value and report them as errors\n        '
        guidsorted = sorted(guidlist, key=lambda x: x.guid.upper(), reverse=True)
        previous = None
        error = None
        errors = []
        for index in range(len(guidsorted)):
            i = guidsorted[index]
            if previous is not None:
                if i.guid == previous.guid:
                    if error is None:
                        error = ErrorEntry('guid')
                        error.entries.append(previous)
                        errors.append(error)
                    error.entries.append(i)
                else:
                    error = None
            previous = i
        return errors

    def _FindConflictingGuidNames(self, guidlist: list) -> list:
        if False:
            while True:
                i = 10
        ' Find all duplicate guids by name and if they are not all\n        from inf files report them as errors.  It is ok to have\n        BASE_NAME duplication.\n\n        Is this useful?  It would catch two same named guids in dec file\n        that resolve to different values.\n        '
        namesorted = sorted(guidlist, key=lambda x: x.name.upper())
        previous = None
        error = None
        errors = []
        for index in range(len(namesorted)):
            i = namesorted[index]
            if previous is not None:
                if i.name == previous.name:
                    if error is None:
                        error = ErrorEntry('name')
                        error.entries.append(previous)
                        errors.append(error)
                    error.entries.append(i)
                else:
                    error = None
            previous = i
            for e in errors[:]:
                if len([en for en in e.entries if not en.absfilepath.lower().endswith('.inf')]) == 0:
                    errors.remove(e)
        return errors

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            return 10
        Errors = []
        abs_pkg_path = Edk2pathObj.GetAbsolutePathOnThisSystemFromEdk2RelativePath(packagename)
        if abs_pkg_path is None:
            tc.SetSkipped()
            tc.LogStdError('No package {0}'.format(packagename))
            return -1
        All_Ignores = ['/Build', '/Conf']
        if 'IgnoreFoldersAndFiles' in pkgconfig:
            All_Ignores.extend(pkgconfig['IgnoreFoldersAndFiles'])
        gs = GuidList.guidlist_from_filesystem(Edk2pathObj.WorkspacePath, ignore_lines=All_Ignores)
        if 'IgnoreGuidValue' in pkgconfig:
            for a in pkgconfig['IgnoreGuidValue']:
                try:
                    tc.LogStdOut('Ignoring Guid {0}'.format(a.upper()))
                    for b in gs[:]:
                        if b.guid == a.upper():
                            gs.remove(b)
                except:
                    tc.LogStdError('GuidCheck.IgnoreGuid -> {0} not found.  Invalid ignore guid'.format(a.upper()))
                    logging.info('GuidCheck.IgnoreGuid -> {0} not found.  Invalid ignore guid'.format(a.upper()))
        if 'IgnoreGuidName' in pkgconfig:
            for a in pkgconfig['IgnoreGuidName']:
                entry = a.split('=')
                if len(entry) > 2:
                    tc.LogStdError('GuidCheck.IgnoreGuidName -> {0} Invalid Format.'.format(a))
                    logging.info('GuidCheck.IgnoreGuidName -> {0} Invalid Format.'.format(a))
                    continue
                try:
                    tc.LogStdOut('Ignoring Guid {0}'.format(a))
                    for b in gs[:]:
                        if b.name == entry[0]:
                            if len(entry) == 1:
                                gs.remove(b)
                            elif len(entry) == 2 and b.guid.upper() == entry[1].upper():
                                gs.remove(b)
                            else:
                                c.LogStdError('GuidCheck.IgnoreGuidName -> {0} incomplete match.  Invalid ignore guid'.format(a))
                except:
                    tc.LogStdError('GuidCheck.IgnoreGuidName -> {0} not found.  Invalid ignore name'.format(a))
                    logging.info('GuidCheck.IgnoreGuidName -> {0} not found.  Invalid ignore name'.format(a))
        Errors.extend(self._FindConflictingGuidValues(gs))
        if 'IgnoreDuplicates' in pkgconfig:
            for a in pkgconfig['IgnoreDuplicates']:
                names = a.split('=')
                if len(names) < 2:
                    tc.LogStdError('GuidCheck.IgnoreDuplicates -> {0} invalid format'.format(a))
                    logging.info('GuidCheck.IgnoreDuplicates -> {0} invalid format'.format(a))
                    continue
                for b in Errors[:]:
                    if b.type != 'guid':
                        continue
                    t = [x for x in b.entries if x.name not in names]
                    if len(t) == len(b.entries):
                        continue
                    elif len(t) == 0:
                        tc.LogStdOut('GuidCheck.IgnoreDuplicates -> {0}'.format(a))
                        Errors.remove(b)
                    elif len(t) < len(b.entries):
                        tc.LogStdOut('GuidCheck.IgnoreDuplicates -> {0} incomplete match'.format(a))
                        logging.info('GuidCheck.IgnoreDuplicates -> {0} incomplete match'.format(a))
                    else:
                        tc.LogStdOut('GuidCheck.IgnoreDuplicates -> {0} unknown error.'.format(a))
                        logging.info('GuidCheck.IgnoreDuplicates -> {0} unknown error'.format(a))
        Errors.extend(self._FindConflictingGuidNames(gs))
        for er in Errors[:]:
            InMyPackage = False
            for a in er.entries:
                if abs_pkg_path in a.absfilepath:
                    InMyPackage = True
                    break
            if not InMyPackage:
                Errors.remove(er)
            else:
                logging.error(str(er))
                tc.LogStdError(str(er))
        overall_status = len(Errors)
        if overall_status != 0:
            tc.SetFailed('GuidCheck {0} Failed.  Errors {1}'.format(packagename, overall_status), 'CHECK_FAILED')
        else:
            tc.SetSuccess()
        return overall_status

class ErrorEntry:
    """ Custom/private class for reporting errors in the GuidList
    """

    def __init__(self, errortype):
        if False:
            return 10
        self.type = errortype
        self.entries = []

    def __str__(self):
        if False:
            return 10
        a = f'Error Duplicate {self.type}: '
        if self.type == 'guid':
            a += f' {self.entries[0].guid}'
        elif self.type == 'name':
            a += f' {self.entries[0].name}'
        a += f' ({len(self.entries)})\n'
        for e in self.entries:
            a += '\t' + str(e) + '\n'
        return a