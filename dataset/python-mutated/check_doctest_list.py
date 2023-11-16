"""
This script is responsible for cleaning the list of doctests by making sure the entries all exist and are in
alphabetical order.

Usage (from the root of the repo):

Check that the doctest list is properly sorted and all files exist (used in `make repo-consistency`):

```bash
python utils/check_doctest_list.py
```

Auto-sort the doctest list if it is not properly sorted (used in `make fix-copies`):

```bash
python utils/check_doctest_list.py --fix_and_overwrite
```
"""
import argparse
import os
REPO_PATH = '.'
DOCTEST_FILE_PATHS = ['not_doctested.txt', 'slow_documentation_tests.txt']

def clean_doctest_list(doctest_file: str, overwrite: bool=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Cleans the doctest in a given file.\n\n    Args:\n        doctest_file (`str`):\n            The path to the doctest file to check or clean.\n        overwrite (`bool`, *optional*, defaults to `False`):\n            Whether or not to fix problems. If `False`, will error when the file is not clean.\n    '
    non_existent_paths = []
    all_paths = []
    with open(doctest_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip().split(' ')[0]
            path = os.path.join(REPO_PATH, line)
            if not (os.path.isfile(path) or os.path.isdir(path)):
                non_existent_paths.append(line)
            all_paths.append(line)
    if len(non_existent_paths) > 0:
        non_existent_paths = '\n'.join([f'- {f}' for f in non_existent_paths])
        raise ValueError(f'`{doctest_file}` contains non-existent paths:\n{non_existent_paths}')
    sorted_paths = sorted(all_paths)
    if all_paths != sorted_paths:
        if not overwrite:
            raise ValueError(f'Files in `{doctest_file}` are not in alphabetical order, run `make fix-copies` to fix this automatically.')
        with open(doctest_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted_paths) + '\n')
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--fix_and_overwrite', action='store_true', help='Whether to fix inconsistencies.')
    args = parser.parse_args()
    for doctest_file in DOCTEST_FILE_PATHS:
        doctest_file = os.path.join(REPO_PATH, 'utils', doctest_file)
        clean_doctest_list(doctest_file, args.fix_and_overwrite)