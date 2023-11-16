#!/usr/bin/env python
"""Return target Python options for use with ansible-test."""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

from ansible import release


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--only-versions', action='store_true')

    options = parser.parse_args()

    ansible_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(release.__file__))))
    source_root = os.path.join(ansible_root, 'test', 'lib')

    sys.path.insert(0, source_root)

    from ansible_test._internal import constants

    args = []

    for python_version in constants.SUPPORTED_PYTHON_VERSIONS:
        executable = shutil.which(f'python{python_version}')

        if executable:
            if python_version.startswith('2.'):
                cmd = [executable, '-m', 'virtualenv', '--version']
            else:
                cmd = [executable, '-m', 'venv', '--help']

            process = subprocess.run(cmd, capture_output=True, check=False)

            print(f'{executable} - {"fail" if process.returncode else "pass"}', file=sys.stderr)

            if not process.returncode:
                if options.only_versions:
                    args.append(python_version)
                    continue

                args.extend(['--target-python', f'venv/{python_version}'])

    print(' '.join(args))


if __name__ == '__main__':
    main()
