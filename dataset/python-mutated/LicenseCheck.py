import os
import shutil
import logging
import re
from io import StringIO
from typing import List, Tuple
from edk2toolext.environment.plugintypes.ci_build_plugin import ICiBuildPlugin
from edk2toolext.environment.var_dict import VarDict
from edk2toollib.utility_functions import RunCmd

class LicenseCheck(ICiBuildPlugin):
    """
    A CiBuildPlugin to check the license for new added files.

    Configuration options:
    "LicenseCheck": {
        "IgnoreFiles": []
    },
    """
    license_format_preflix = 'SPDX-License-Identifier'
    bsd2_patent = 'BSD-2-Clause-Patent'
    Readdedfileformat = re.compile('\\+\\+\\+ b\\/(.*)')
    file_extension_list = ['.c', '.h', '.inf', '.dsc', '.dec', '.py', '.bat', '.sh', '.uni', '.yaml', '.fdf', '.inc', 'yml', '.asm', '.asm16', '.asl', '.vfr', '.s', '.S', '.aslc', '.nasm', '.nasmb', '.idf', '.Vfr', '.H']

    def GetTestName(self, packagename: str, environment: VarDict) -> tuple:
        if False:
            i = 10
            return i + 15
        ' Provide the testcase name and classname for use in reporting\n            testclassname: a descriptive string for the testcase can include whitespace\n            classname: should be patterned <packagename>.<plugin>.<optionally any unique condition>\n\n            Args:\n              packagename: string containing name of package to build\n              environment: The VarDict for the test to run in\n            Returns:\n                a tuple containing the testcase name and the classname\n                (testcasename, classname)\n        '
        return ('Check for license for ' + packagename, packagename + '.LicenseCheck')

    def RunBuildPlugin(self, packagename, Edk2pathObj, pkgconfig, environment, PLM, PLMHelper, tc, output_stream=None):
        if False:
            while True:
                i = 10
        temp_path = os.path.join(Edk2pathObj.WorkspacePath, 'Build', '.pytool', 'Plugin', 'LicenseCheck')
        if not os.path.exists(temp_path):
            os.makedirs(temp_path)
        temp_diff_output = os.path.join(temp_path, 'diff.txt')
        params = 'diff --output={} --unified=0 origin/master HEAD'.format(temp_diff_output)
        RunCmd('git', params)
        with open(temp_diff_output) as file:
            patch = file.read().strip().split('\n')
        if os.path.exists(temp_path):
            shutil.rmtree(temp_path)
        ignore_files = []
        if 'IgnoreFiles' in pkgconfig:
            ignore_files = pkgconfig['IgnoreFiles']
        self.ok = True
        self.startcheck = False
        self.license = True
        self.all_file_pass = True
        count = len(patch)
        line_index = 0
        for line in patch:
            if line.startswith('--- /dev/null'):
                nextline = patch[line_index + 1]
                added_file = self.Readdedfileformat.search(nextline).group(1)
                added_file_extension = os.path.splitext(added_file)[1]
                if added_file_extension in self.file_extension_list and packagename in added_file:
                    if self.IsIgnoreFile(added_file, ignore_files):
                        line_index = line_index + 1
                        continue
                    self.startcheck = True
                    self.license = False
            if self.startcheck and self.license_format_preflix in line:
                if self.bsd2_patent in line:
                    self.license = True
            if line_index + 1 == count or (patch[line_index + 1].startswith('diff --') and self.startcheck):
                if not self.license:
                    self.all_file_pass = False
                    error_message = 'Invalid license in: ' + added_file + ' Hint: Only BSD-2-Clause-Patent is accepted.'
                    logging.error(error_message)
                self.startcheck = False
                self.license = True
            line_index = line_index + 1
        if self.all_file_pass:
            tc.SetSuccess()
            return 0
        else:
            tc.SetFailed('License Check {0} Failed. '.format(packagename), 'LICENSE_CHECK_FAILED')
            return 1

    def IsIgnoreFile(self, file: str, ignore_files: List[str]) -> bool:
        if False:
            i = 10
            return i + 15
        for f in ignore_files:
            if f in file:
                return True
        return False