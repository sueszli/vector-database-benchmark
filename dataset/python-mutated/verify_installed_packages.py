import argparse
import os
import sys
import logging
from os import path
root_dir = path.abspath(path.join(path.abspath(__file__), '..', '..', '..'))
common_task_path = path.abspath(path.join(root_dir, 'scripts', 'devops_tasks'))
sys.path.append(common_task_path)
from common_tasks import get_installed_packages

def verify_packages(package_file_path):
    if False:
        i = 10
        return i + 15
    packages = []
    with open(package_file_path, 'r') as packages_file:
        packages = packages_file.readlines()
    packages = [p.replace('\n', '') for p in packages]
    invalid_lines = [p for p in packages if '==' not in p]
    if invalid_lines:
        logging.error('packages.txt has package details in invalid format. Expected format is <package-name>==<version>')
        sys.exit(1)
    installed = {}
    for p in get_installed_packages():
        if '==' in p:
            [package, version] = p.split('==')
            installed[package.upper()] = version
    expected = {}
    for p in packages:
        [package, version] = p.split('==')
        expected[package.upper()] = version
    missing_packages = [pkg for pkg in expected.keys() if installed.get(pkg) != expected.get(pkg)]
    if missing_packages:
        logging.error('Version is incorrect for following package[s]')
        for package in missing_packages:
            logging.error('%s, Expected[%s], Installed[%s]', package, expected[package], installed[package])
        sys.exit(1)
    else:
        logging.info('Verified package version')
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Install either latest or minimum version of dependent packages.')
    parser.add_argument('-f', '--packages-file', dest='packages_file', help='Path to a file that has list of packages and version to verify', required=True)
    args = parser.parse_args()
    if os.path.exists(args.packages_file):
        verify_packages(args.packages_file)