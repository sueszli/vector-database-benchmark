"""Utility script to verify qiskit copyright file headers"""
import argparse
import multiprocessing
import subprocess
import sys
import re
reno = re.compile('releasenotes\\/notes')
exact_reno = re.compile('^releasenotes\\/notes')

def discover_files():
    if False:
        print('Hello World!')
    'Find all .py, .pyx, .pxd files in a list of trees'
    cmd = ['git', 'ls-tree', '-r', '--name-only', 'HEAD']
    res = subprocess.run(cmd, capture_output=True, check=True, encoding='UTF8')
    files = res.stdout.split('\n')
    return files

def validate_path(file_path):
    if False:
        return 10
    'Validate a path in the git tree.'
    if reno.search(file_path) and (not exact_reno.search(file_path)):
        return file_path
    return None

def _main():
    if False:
        while True:
            i = 10
    parser = argparse.ArgumentParser(description='Find any stray release notes.')
    _args = parser.parse_args()
    files = discover_files()
    with multiprocessing.Pool() as pool:
        res = pool.map(validate_path, files)
    failed_files = [x for x in res if x is not None]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write('%s is not in the correct location.\n' % failed_file)
        sys.exit(1)
    sys.exit(0)
if __name__ == '__main__':
    _main()