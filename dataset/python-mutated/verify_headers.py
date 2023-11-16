"""Utility script to verify qiskit copyright file headers"""
import argparse
import multiprocessing
import os
import sys
import re
pep263 = re.compile('^[ \\t\\f]*#.*?coding[:=][ \\t]*([-_.a-zA-Z0-9]+)')

def discover_files(code_paths):
    if False:
        for i in range(10):
            print('nop')
    'Find all .py, .pyx, .pxd files in a list of trees'
    out_paths = []
    for path in code_paths:
        if os.path.isfile(path):
            out_paths.append(path)
        else:
            for directory in os.walk(path):
                dir_path = directory[0]
                for subfile in directory[2]:
                    if subfile.endswith('.py') or subfile.endswith('.pyx') or subfile.endswith('.pxd'):
                        out_paths.append(os.path.join(dir_path, subfile))
    return out_paths

def validate_header(file_path):
    if False:
        i = 10
        return i + 15
    'Validate the header for a single file'
    header = '# This code is part of Qiskit.\n#\n'
    apache_text = '#\n# This code is licensed under the Apache License, Version 2.0. You may\n# obtain a copy of this license in the LICENSE.txt file in the root directory\n# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.\n#\n# Any modifications or derivative works of this code must retain this\n# copyright notice, and modified files need to carry a notice indicating\n# that they have been altered from the originals.\n'
    count = 0
    with open(file_path, encoding='utf8') as fd:
        lines = fd.readlines()
    start = 0
    for (index, line) in enumerate(lines):
        count += 1
        if count > 5:
            return (file_path, False, 'Header not found in first 5 lines')
        if count <= 2 and pep263.match(line):
            return (file_path, False, 'Unnecessary encoding specification (PEP 263, 3120)')
        if line == '# This code is part of Qiskit.\n':
            start = index
            break
    if ''.join(lines[start:start + 2]) != header:
        return (file_path, False, 'Header up to copyright line does not match: %s' % header)
    if not lines[start + 2].startswith('# (C) Copyright IBM 20'):
        return (file_path, False, 'Header copyright line not found')
    if ''.join(lines[start + 3:start + 11]) != apache_text:
        return (file_path, False, "Header apache text string doesn't match:\n %s" % apache_text)
    return (file_path, True, None)

def _main():
    if False:
        for i in range(10):
            print('nop')
    default_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qiskit')
    parser = argparse.ArgumentParser(description='Check file headers.')
    parser.add_argument('paths', type=str, nargs='*', default=[default_path], help='Paths to scan by default uses ../qiskit from the script')
    args = parser.parse_args()
    files = discover_files(args.paths)
    with multiprocessing.Pool() as pool:
        res = pool.map(validate_header, files)
    failed_files = [x for x in res if x[1] is False]
    if len(failed_files) > 0:
        for failed_file in failed_files:
            sys.stderr.write('%s failed header check because:\n' % failed_file[0])
            sys.stderr.write('%s\n\n' % failed_file[2])
        sys.exit(1)
    sys.exit(0)
if __name__ == '__main__':
    _main()