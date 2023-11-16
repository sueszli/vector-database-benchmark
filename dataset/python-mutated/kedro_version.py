"""
Get version of Kedro
"""
import os.path
import re
import sys
from pathlib import Path
VERSION_MATCHSTR = '\\s*__version__\\s*=\\s*"(\\d+\\.\\d+\\.\\d+)"'

def get_kedro_version(init_file_path):
    if False:
        for i in range(10):
            print('nop')
    match_obj = re.search(VERSION_MATCHSTR, Path(init_file_path).read_text())
    return match_obj.group(1)

def main(argv):
    if False:
        for i in range(10):
            print('nop')
    kedro_path = argv[1]
    init_file_path = os.path.join(kedro_path, '__init__.py')
    print(get_kedro_version(init_file_path))
if __name__ == '__main__':
    main(sys.argv)