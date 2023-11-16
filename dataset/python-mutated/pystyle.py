"""
Checks PEP8 compliance, with some exceptions.
"""
try:
    from pycodestyle import StyleGuide
except ImportError:
    from pep8 import StyleGuide
IGNORE_ERRORS = ('E221', 'E241', 'E251', 'E501')

def filter_file_list(check_files, dirnames):
    if False:
        for i in range(10):
            print('nop')
    "\n    Yields all those files in check_files that are in one of the directories\n    and end in '.py'.\n    "
    for filename in check_files:
        if not filename.endswith('.py'):
            continue
        if any((filename.startswith(dirname) for dirname in dirnames)):
            yield filename

def find_issues(check_files, dirnames):
    if False:
        return 10
    '\n    Finds all issues in the given directories (filtered by check_files).\n    '
    checker = StyleGuide()
    checker.options.ignore = IGNORE_ERRORS
    filenames = dirnames
    if check_files is not None:
        filenames = filter_file_list(check_files, dirnames)
    report = checker.check_files(filenames)
    if report.messages:
        yield ('style issue', 'python code violates pep8', None)