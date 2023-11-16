import os
import re
import sys
from pathlib import Path
'\nRun this file with the Cura project root as the working directory\nChecks for invalid imports. When importing from plugins, there will be no problems when running from source, \nbut for some build types the plugins dir is not on the path, so relative imports should be used instead. eg:\nfrom ..UltimakerCloudScope import UltimakerCloudScope  <-- OK\nimport plugins.Marketplace.src ...  <-- NOT OK\n'

class InvalidImportsChecker:
    REGEX = re.compile('^\\s*(from plugins|import plugins)')

    def check(self):
        if False:
            i = 10
            return i + 15
        ' Checks for invalid imports\n\n        :return: True if checks passed, False when the test fails\n        '
        cwd = os.getcwd()
        cura_result = checker.check_dir(os.path.join(cwd, 'cura'))
        plugins_result = checker.check_dir(os.path.join(cwd, 'plugins'))
        result = cura_result and plugins_result
        if not result:
            print('error: sources contain invalid imports. Use relative imports when referencing plugin source files')
        return result

    def check_dir(self, root_dir: str) -> bool:
        if False:
            i = 10
            return i + 15
        ' Checks a directory for invalid imports\n\n        :return: True if checks passed, False when the test fails\n        '
        passed = True
        for path_like in Path(root_dir).rglob('*.py'):
            if not self.check_file(str(path_like)):
                passed = False
        return passed

    def check_file(self, file_path):
        if False:
            return 10
        ' Checks a file for invalid imports\n\n        :return: True if checks passed, False when the test fails\n        '
        passed = True
        with open(file_path, 'r', encoding='utf-8') as inputFile:
            for (line_i, line) in enumerate(inputFile, 1):
                match = self.REGEX.search(line)
                if match:
                    path = os.path.relpath(file_path)
                    print('{path}:{line_i}:{match}'.format(path=path, line_i=line_i, match=match.group(1)))
                    passed = False
        return passed
if __name__ == '__main__':
    checker = InvalidImportsChecker()
    sys.exit(0 if checker.check() else 1)