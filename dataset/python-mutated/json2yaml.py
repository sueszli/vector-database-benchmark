"""
A utility script which sends test messages to a queue.
"""
from __future__ import absolute_import
import argparse
import fnmatch
try:
    import simplejson as json
except ImportError:
    import json
import os
import pprint
import subprocess
import traceback
import yaml
PRINT = pprint.pprint
YAML_HEADER = '---'

def get_files_matching_pattern(dir_, pattern):
    if False:
        for i in range(10):
            print('nop')
    files = []
    for (root, _, filenames) in os.walk(dir_):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(os.path.join(root, filename))
    return files

def json_2_yaml_convert(filename):
    if False:
        i = 10
        return i + 15
    data = None
    try:
        with open(filename, 'r') as json_file:
            data = json.load(json_file)
    except:
        PRINT('Failed on {}'.format(filename))
        traceback.print_exc()
        return (filename, '')
    new_filename = os.path.splitext(filename)[0] + '.yaml'
    with open(new_filename, 'w') as yaml_file:
        yaml_file.write(YAML_HEADER + '\n')
        yaml_file.write(yaml.safe_dump(data, default_flow_style=False))
    return (filename, new_filename)

def git_rm(filename):
    if False:
        print('Hello World!')
    try:
        subprocess.check_call(['git', 'rm', filename])
    except subprocess.CalledProcessError:
        PRINT('Failed to git rm {}'.format(filename))
        traceback.print_exc()
        return (False, filename)
    return (True, filename)

def main(dir_, skip_convert):
    if False:
        return 10
    files = get_files_matching_pattern(dir_, '*.json')
    if skip_convert:
        PRINT(files)
        return
    results = [json_2_yaml_convert(filename) for filename in files]
    PRINT('*** conversion done ***')
    PRINT(['converted {} to {}'.format(result[0], result[1]) for result in results])
    results = [git_rm(filename) for (filename, new_filename) in results if new_filename]
    PRINT('*** git rm done ***')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='json2yaml converter.')
    parser.add_argument('--dir', '-d', required=True, help='The dir to look for json.')
    parser.add_argument('--skipconvert', '-s', action='store_true', help='Skip conversion')
    args = parser.parse_args()
    main(dir_=args.dir, skip_convert=args.skipconvert)