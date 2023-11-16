"""Command line tool which changes the branch to be
ready to build and test the given Elastic stack version.
"""
import re
import sys
from pathlib import Path
SOURCE_DIR = Path(__file__).absolute().parent.parent

def find_and_replace(path, pattern, replace):
    if False:
        i = 10
        return i + 15
    with open(path, 'r') as f:
        old_data = f.read()
    if re.search(pattern, old_data, flags=re.MULTILINE) is None:
        print(f"Didn't find the pattern {pattern!r} in {path!s}")
        exit(1)
    new_data = re.sub(pattern, replace, old_data, flags=re.MULTILINE)
    with open(path, 'w') as f:
        f.truncate()
        f.write(new_data)

def main():
    if False:
        return 10
    if len(sys.argv) != 2:
        print('usage: utils/bump-version.py [stack version]')
        exit(1)
    stack_version = sys.argv[1]
    try:
        python_version = re.search('^([0-9][0-9\\.]*[0-9]+)', stack_version).group(1)
    except AttributeError:
        print(f"Couldn't match the given stack version {stack_version!r}")
        exit(1)
    for _ in range(3):
        if len(python_version.split('.')) >= 3:
            break
        python_version += '.0'
    find_and_replace(path=SOURCE_DIR / 'elasticsearch/_version.py', pattern='__versionstr__ = \\"[0-9]+[0-9\\.]*[0-9](?:\\+dev)?\\"', replace=f'__versionstr__ = "{python_version}"')
    major_minor_version = '.'.join(python_version.split('.')[:2])
    find_and_replace(path=SOURCE_DIR / '.ci/test-matrix.yml', pattern='STACK_VERSION:\\s+\\- "[0-9]+[0-9\\.]*[0-9](?:\\-SNAPSHOT)?"', replace=f'STACK_VERSION:\n  - "{major_minor_version}.0-SNAPSHOT"')
if __name__ == '__main__':
    main()