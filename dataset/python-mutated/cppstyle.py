"""
Checks some code style rules for cpp files.
"""
import re
from .util import findfiles, readfile, issue_str_line
MISSING_SPACES_RE = re.compile('(?:(?:\\)(\\{))|(?:\\s+(?:if|for|while)(\\()))')
EXTRA_SPACES_RE = re.compile('(?:(?:(\\s+);))')
INDENT_FAIL_RE = re.compile('(?:(\\n\\t*[ ]+\\t+))')
INDENT_FAIL_LINE_RE = re.compile('(?:(^\\t*[ ]+\\t+))')

def filter_file_list(check_files, dirnames):
    if False:
        return 10
    "\n    Yields all those files in check_files that are in one of the directories\n    and end in '.cpp' or '.h' and some other conditions.\n    "
    for filename in check_files:
        if not (filename.endswith('.cpp') or filename.endswith('.h')):
            continue
        if filename.endswith('.gen.h') or filename.endswith('.gen.cpp'):
            continue
        if any((filename.startswith(dirname) for dirname in dirnames)):
            yield filename

def find_issues(check_files, dirnames):
    if False:
        return 10
    '\n    Finds all issues in the given directories (filtered by check_files).\n    '
    if check_files is not None:
        filenames = filter_file_list(check_files, dirnames)
    else:
        filenames = filter_file_list(findfiles(dirnames), dirnames)
    for filename in filenames:
        data = readfile(filename)
        analyse_each_line = False
        if MISSING_SPACES_RE.search(data) or EXTRA_SPACES_RE.search(data) or INDENT_FAIL_RE.search(data):
            analyse_each_line = True
        if analyse_each_line:
            yield from find_issues_with_lines(data, filename)

def find_issues_with_lines(data, filename):
    if False:
        for i in range(10):
            print('nop')
    '\n    Checks a file for issues per line\n    '
    for (num, line) in enumerate(data.splitlines(True), start=1):
        match = MISSING_SPACES_RE.search(line)
        if match:
            start = match.start(1) + match.start(2)
            end = start + 1
            yield issue_str_line('Missing space', filename, line, num, (start, end))
        match = EXTRA_SPACES_RE.search(line)
        if match:
            yield issue_str_line('Extra space', filename, line, num, (match.start(1), match.end(1)))
        match = INDENT_FAIL_LINE_RE.search(line)
        if match:
            yield issue_str_line('Wrong indentation', filename, line, num, (match.start(1), match.end(1)))