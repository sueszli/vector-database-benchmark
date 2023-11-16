from __future__ import annotations
import os
import sys
from glob import glob
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir, os.pardir))

def check_dir_init_file(provider_files: list[str]) -> None:
    if False:
        i = 10
        return i + 15
    missing_init_dirs = []
    for path in provider_files:
        if path.endswith('/__pycache__'):
            continue
        if os.path.isdir(path) and (not os.path.exists(os.path.join(path, '__init__.py'))):
            missing_init_dirs.append(path)
    if missing_init_dirs:
        with open(os.path.join(ROOT_DIR, 'scripts/ci/license-templates/LICENSE.txt')) as license:
            license_txt = license.readlines()
        prefixed_licensed_txt = [f'# {line}' if line != '\n' else '#\n' for line in license_txt]
        for missing_init_dir in missing_init_dirs:
            with open(os.path.join(missing_init_dir, '__init__.py'), 'w') as init_file:
                init_file.write(''.join(prefixed_licensed_txt))
        print('No __init__.py file was found in the following provider directories:')
        print('\n'.join(missing_init_dirs))
        print('\nThe missing __init__.py files have been created. Please add these new files to a commit.')
        sys.exit(1)
if __name__ == '__main__':
    all_provider_subpackage_dirs = sorted(glob(f'{ROOT_DIR}/airflow/providers/**/*', recursive=True))
    check_dir_init_file(all_provider_subpackage_dirs)
    all_test_provider_subpackage_dirs = sorted(glob(f'{ROOT_DIR}/tests/providers/**/*', recursive=True))
    check_dir_init_file(all_test_provider_subpackage_dirs)