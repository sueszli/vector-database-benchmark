import argparse
import os
import re
import sys

class StaticChecker:
    """Run simple checks on the entire document or specific lines."""

    def __init__(self, replace: bool) -> None:
        if False:
            while True:
                i = 10
        'Initialize a :class:`.StaticChecker` instance.\n\n        :param replace: Whether or not to make replacements.\n\n        '
        self.full_file_checks = [self.check_for_double_syntax]
        self.line_checks = [self.check_for_noreturn]
        self.replace = replace

    def check_for_double_syntax(self, filename: str, content: str) -> bool:
        if False:
            return 10
        'Checks a file for double-slash statements (``/r/`` and ``/u/``).\n\n        :param filename: The name of the file to check & replace.\n        :param content: The content of the file.\n\n        :returns: A boolean with the status of the check.\n\n        '
        if os.path.join('praw', 'const.py') in filename:
            return True
        new_content = re.sub('(^|\\s)/(u|r)/', '\\1\\2/', content)
        if content == new_content:
            return True
        if self.replace:
            with open(filename, 'w') as fp:
                fp.write(new_content)
            print(f"{filename}: Replaced all instances of '/r/' and/or '/u/' to 'r/' and/or 'u/'.")
            return True
        print(f"{filename}: This file contains instances of '/r/' and/or '/u/'. Please change them to 'r/' and/or 'u/'.")
        return False

    def check_for_noreturn(self, filename: str, line_number: int, content: str) -> bool:
        if False:
            return 10
        'Checks a line for ``NoReturn`` statements.\n\n        :param filename: The name of the file to check & replace.\n        :param line_number: The line number.\n        :param content: The content of the line.\n\n        :returns: A boolean with the status of the check.\n\n        '
        if 'noreturn' in content.lower():
            print(f"{filename}: Line {line_number} has phrase 'noreturn', please edit and remove this.")
            return False
        return True

    def run_checks(self) -> bool:
        if False:
            print('Hello World!')
        'Scan a directory and run the checks.\n\n        The directory is assumed to be the praw directory located in the parent\n        directory of the file, so if this file exists in\n        ``~/praw/tools/static_word_checks.py``, it will check ``~/praw/praw``.\n\n        It runs the checks located in the ``self.full_file_checks`` and\n        ``self.line_checks`` lists, with full file checks being run first.\n\n        Full-file checks are checks that can also fix the errors they find, while the\n        line checks can just warn about found errors.\n\n        - Full file checks:\n\n          - :meth:`.check_for_double_syntax`\n\n        - Line checks:\n\n          - :meth:`.check_for_noreturn`\n\n        '
        status = True
        directory = os.path.abspath(os.path.join(__file__, '..', '..', 'praw'))
        for (current_directory, _directories, filenames) in os.walk(directory):
            for filename in filenames:
                if not filename.endswith('.py'):
                    continue
                filename = os.path.join(current_directory, filename)
                for check in self.full_file_checks:
                    with open(filename) as fp:
                        full_content = fp.read()
                    status &= check(filename, full_content)
                for check in self.line_checks:
                    with open(filename) as fp:
                        lines = fp.readlines()
                    for (line_number, line) in enumerate(lines, 1):
                        status &= check(filename, line_number, line)
        return status

def main():
    if False:
        for i in range(10):
            print('nop')
    'The main function.'
    parser = argparse.ArgumentParser(description='Run static line checks and optionally replace values that should not be used.')
    parser.add_argument('-r', '--replace', action='store_true', default=False, help='If it is possible, tries to reformat values. Not all checks can reformat values, and those will have to be edited manually.')
    args = parser.parse_args()
    check = StaticChecker(args.replace)
    return int(not check.run_checks())
if __name__ == '__main__':
    sys.exit(main())