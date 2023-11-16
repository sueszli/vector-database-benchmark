from __future__ import print_function
import argparse
import subprocess
import sys

def print_failure(msg):
    if False:
        print('Hello World!')
    '\n    Print a failure message.\n\n    Parameters\n    ----------\n    msg : str\n        The failure message to print.\n    '
    print('\x1b[91m' + msg + '\x1b[0m')

def update_hosts_file(*flags):
    if False:
        i = 10
        return i + 15
    '\n    Wrapper around running updateHostsFile.py\n\n    Parameters\n    ----------\n    flags : varargs\n        Commandline flags to pass into updateHostsFile.py. For more info, run\n        the following command in the terminal or command prompt:\n\n        ```\n        python updateHostsFile.py -h\n        ```\n    '
    if subprocess.call([sys.executable, 'updateHostsFile.py'] + list(flags)):
        print_failure('Failed to update hosts file')

def update_readme_file():
    if False:
        print('Hello World!')
    '\n    Wrapper around running updateReadme.py\n    '
    if subprocess.call([sys.executable, 'updateReadme.py']):
        print_failure('Failed to update readme file')

def recursively_loop_extensions(extension, extensions, current_extensions):
    if False:
        return 10
    '\n    Helper function that recursively calls itself to prevent manually creating\n    all possible combinations of extensions.\n\n    Will call update_hosts_file for all combinations of extensions\n    '
    c_extensions = extensions.copy()
    c_current_extensions = current_extensions.copy()
    c_current_extensions.append(extension)
    name = '-'.join(c_current_extensions)
    params = ('-a', '-n', '-o', 'alternates/' + name, '-e') + tuple(c_current_extensions)
    update_hosts_file(*params)
    params = ('-a', '-n', '-s', '--nounifiedhosts', '-o', 'alternates/' + name + '-only', '-e') + tuple(c_current_extensions)
    update_hosts_file(*params)
    while len(c_extensions) > 0:
        recursively_loop_extensions(c_extensions.pop(0), c_extensions, c_current_extensions)

def main():
    if False:
        return 10
    parser = argparse.ArgumentParser(description='Creates custom hosts file from hosts stored in data subfolders.')
    parser.parse_args()
    update_hosts_file('-a')
    extensions = ['fakenews', 'gambling', 'porn', 'social']
    while len(extensions) > 0:
        recursively_loop_extensions(extensions.pop(0), extensions, [])
    update_readme_file()
if __name__ == '__main__':
    main()