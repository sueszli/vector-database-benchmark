import sys
import argparse
import logging
from os import path
from subprocess import check_call
root_dir = path.abspath(path.join(path.abspath(__file__), '..', '..', '..'))
common_task_path = path.abspath(path.join(root_dir, 'scripts', 'devops_tasks'))
sys.path.append(common_task_path)
from common_tasks import get_installed_packages
from ci_tools.functions import discover_targeted_packages
from ci_tools.parsing import ParsedSetup
EXCLUDED_PKGS = ['azure-common']
DEV_INDEX_URL = 'https://pkgs.dev.azure.com/azure-sdk/public/_packaging/azure-sdk-for-python/pypi/simple'
logging.getLogger().setLevel(logging.INFO)

def get_installed_azure_packages(pkg_name_to_exclude):
    if False:
        for i in range(10):
            print('nop')
    installed_pkgs = [p.split('==')[0] for p in get_installed_packages() if p.startswith('azure-')]
    pkgs = discover_targeted_packages('', root_dir)
    valid_azure_packages = [path.basename(p) for p in pkgs if 'mgmt' not in p and '-nspkg' not in p]
    pkg_names = [p for p in installed_pkgs if p in valid_azure_packages and p != pkg_name_to_exclude and (p not in EXCLUDED_PKGS)]
    logging.info('Installed azure sdk packages: %s', pkg_names)
    return pkg_names

def uninstall_packages(packages):
    if False:
        while True:
            i = 10
    commands = [sys.executable, '-m', 'pip', 'uninstall']
    logging.info('Uninstalling packages: %s', packages)
    commands.extend(packages)
    commands.append('--yes')
    check_call(commands)
    logging.info('Uninstalled packages')

def install_packages(packages):
    if False:
        return 10
    commands = [sys.executable, '-m', 'pip', 'install']
    logging.info('Installing dev build version for packages: %s', packages)
    commands.extend(packages)
    commands.extend(['--index-url', DEV_INDEX_URL])
    check_call(commands)

def install_dev_build_packages(pkg_name_to_exclude):
    if False:
        while True:
            i = 10
    azure_pkgs = get_installed_azure_packages(pkg_name_to_exclude)
    uninstall_packages(azure_pkgs)
    install_packages(azure_pkgs)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Install dev build version of dependent packages for current package')
    parser.add_argument('-t', '--target', dest='target_package', help='The target package directory on disk.', required=True)
    args = parser.parse_args()
    if args.target_package:
        pkg_dir = path.abspath(args.target_package)
        pkg_details = ParsedSetup.from_path(pkg_dir)
        install_dev_build_packages(pkg_details.name)