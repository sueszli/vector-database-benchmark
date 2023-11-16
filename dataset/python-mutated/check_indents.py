import re
from typing import Tuple

def find_whitespace(string: str) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    groups = re.match('(\\s*)([^\\s].*$)?', string, flags=re.S).groups()
    return (groups[0], groups[1] or '')

def check_file(filename: str) -> bool:
    if False:
        return 10
    '\n    Checks a single file, returning True and writing a cleaned file if a mix\n    of tabs and spaces was found.\n    '
    problem_found = False
    found_spaces = False
    found_tabs = False
    with open(filename, 'r') as f:
        contents = list(f.readlines())
    for line in contents:
        (whitespace, rest) = find_whitespace(line)
        found_spaces = found_spaces or ' ' in whitespace
        found_tabs = found_tabs or '\t' in whitespace
    problem_found = found_spaces and found_tabs
    if problem_found:
        print(f'Found mixed spaces and tabs in {filename}.')
        with open(filename, 'w') as f:
            f.writelines((whitespace.replace('\t', '    ') + rest for (whitespace, rest) in map(find_whitespace, contents)))
    return problem_found
if __name__ == '__main__':
    import sys
    filenames = sys.argv[1:]
    exit_code = 0
    for filename in filenames:
        if check_file(filename):
            exit_code = -1
    sys.exit(exit_code)