"""
Verify that new news entries have valid filenames. Usage:

.. code-block:: bash

    ./scripts/verify-news-fragments.py

"""
import re
import sys
from pathlib import Path
CHANGELOG_GUIDE = 'https://github.com/pyinstaller/pyinstaller/blob/develop/doc/development/changelog-entries.rst#changelog-entries'
CHANGE_TYPES = {'bootloader', 'breaking', 'bugfix', 'build', 'core', 'doc', 'feature', 'hooks', 'moduleloader', 'process', 'tests', 'deprecation'}
NEWS_PATTERN = re.compile('(\\d+)\\.(\\w+)\\.(?:(\\d+)\\.)?rst')
NEWS_DIR = Path(__file__).absolute().parent.parent / 'news'

def validate_name(name):
    if False:
        while True:
            i = 10
    "\n    Check a filename/filepath matches the required format.\n\n    :param name: Name of news fragment file.\n    :type: str, os.Pathlike\n\n    :raises: ``SystemExit`` if above checks don't pass.\n    "
    match = NEWS_PATTERN.fullmatch(Path(name).name)
    if match is None:
        raise SystemExit(f"'{name}' does not match the '(pr-number).(type).rst' or '(pr-number).(type).(enumeration).rst' changelog entries formats. See:\n{CHANGELOG_GUIDE}")
    if match.group(2) not in CHANGE_TYPES:
        sys.exit("'{}' of of invalid type '{}'. Valid types are:\n{}".format(name, match.group(2), CHANGE_TYPES))
    print(name, 'is ok')

def main():
    if False:
        i = 10
        return i + 15
    for file in NEWS_DIR.iterdir():
        if file.name in ['README.txt', '_template.rst', '.gitignore']:
            continue
        validate_name(file)
if __name__ == '__main__':
    main()