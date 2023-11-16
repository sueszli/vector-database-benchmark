import argparse
import os
import sys
import logging
import re
from subprocess import check_call
from typing import TYPE_CHECKING
from pkg_resources import parse_version
from pypi_tools.pypi import PyPIClient
from packaging.specifiers import SpecifierSet
from packaging.version import Version, parse
import pdb
from ci_tools.parsing import ParsedSetup, parse_require
from ci_tools.functions import compare_python_version
from typing import List
DEV_REQ_FILE = 'dev_requirements.txt'
NEW_DEV_REQ_FILE = 'new_dev_requirements.txt'
PKGS_TXT_FILE = 'packages.txt'
logging.getLogger().setLevel(logging.INFO)
MINIMUM_VERSION_GENERIC_OVERRIDES = {'azure-common': '1.1.10', 'msrest': '0.6.10', 'typing-extensions': '4.6.0', 'opentelemetry-api': '1.3.0', 'opentelemetry-sdk': '1.3.0', 'azure-core': '1.11.0', 'requests': '2.19.0', 'six': '1.12.0', 'cryptography': '3.3.2', 'msal': '1.23.0'}
MAXIMUM_VERSION_GENERIC_OVERRIDES = {}
MINIMUM_VERSION_SPECIFIC_OVERRIDES = {'azure-eventhub': {'azure-core': '1.25.0'}, 'azure-eventhub-checkpointstoreblob-aio': {'azure-core': '1.25.0', 'azure-eventhub': '5.11.0'}, 'azure-eventhub-checkpointstoreblob': {'azure-core': '1.25.0', 'azure-eventhub': '5.11.0'}, 'azure-eventhub-checkpointstoretable': {'azure-core': '1.25.0', 'azure-eventhub': '5.11.0'}, 'azure-identity': {'msal': '1.23.0'}}
MAXIMUM_VERSION_SPECIFIC_OVERRIDES = {}
PLATFORM_SPECIFIC_MINIMUM_OVERRIDES = {'>=3.12.0': {'azure-core': '1.23.1', 'aiohttp': '3.8.6', 'six': '1.16.0', 'requests': '2.30.0'}}
PLATFORM_SPECIFIC_MAXIMUM_OVERRIDES = {}
SPECIAL_CASE_OVERRIDES = {'azure-core': {'<1.24.0': ['msrest<0.7.0']}}

def install_dependent_packages(setup_py_file_path, dependency_type, temp_dir):
    if False:
        print('Hello World!')
    released_packages = find_released_packages(setup_py_file_path, dependency_type)
    override_added_packages = []
    for pkg_spec in released_packages:
        override_added_packages.extend(check_pkg_against_overrides(pkg_spec))
    logging.info('%s released packages: %s', dependency_type, released_packages)
    dev_req_file_path = filter_dev_requirements(setup_py_file_path, released_packages, temp_dir)
    if override_added_packages:
        logging.info(f'Expanding the requirement set by the packages {override_added_packages}.')
    install_set = released_packages + list(set(override_added_packages))
    if released_packages or dev_req_file_path:
        install_packages(install_set, dev_req_file_path)
    if released_packages:
        pkgs_file_path = os.path.join(temp_dir, PKGS_TXT_FILE)
        with open(pkgs_file_path, 'w') as pkgs_file:
            for package in released_packages:
                pkgs_file.write(package + '\n')
        logging.info('Created file %s to track azure packages found on PyPI', pkgs_file_path)

def check_pkg_against_overrides(pkg_specifier: str) -> List[str]:
    if False:
        print('Hello World!')
    '\n    Checks a set of package specifiers of form "[A==1.0.0, B=2.0.0]". Used to inject additional package installations\n    as indicated by the SPECIAL_CASE_OVERRIDES dictionary.\n\n    :param str pkg_specifier: A specifically targeted package that is about to be passed to install_packages.\n    '
    additional_installs = []
    (target_package, target_version) = pkg_specifier.split('==')
    target_version = Version(target_version)
    if target_package in SPECIAL_CASE_OVERRIDES:
        special_case_specifiers = SPECIAL_CASE_OVERRIDES[target_package]
        for specifier_set in special_case_specifiers.keys():
            spec = SpecifierSet(specifier_set)
            if target_version in spec:
                additional_installs.extend(special_case_specifiers[specifier_set])
    return additional_installs

def find_released_packages(setup_py_path, dependency_type):
    if False:
        while True:
            i = 10
    pkg_info = ParsedSetup.from_path(setup_py_path)
    requires = [r for r in pkg_info.requires if '-nspkg' not in r]
    avlble_packages = [x for x in map(lambda x: process_requirement(x, dependency_type, pkg_info.name), requires) if x]
    return avlble_packages

def process_bounded_versions(originating_pkg_name: str, pkg_name: str, versions: List[str]) -> List[str]:
    if False:
        while True:
            i = 10
    '\n    Processes a target package based on an originating package (target is a dep of originating) and the versions available from pypi for the target package.\n\n    Returns the set of versions AFTER general, platform, and package-specific overrides have been applied.\n\n    :param str originating_pkg_name: The name of the package whos requirements are being processed.\n    :param str pkg_name: A specific requirement of the originating package being processed.\n    :param List[str] versions: All the versions available on pypi for pkg_name.\n    '
    if pkg_name in MINIMUM_VERSION_GENERIC_OVERRIDES:
        versions = [v for v in versions if parse_version(v) >= parse_version(MINIMUM_VERSION_GENERIC_OVERRIDES[pkg_name])]
    for platform_bound in PLATFORM_SPECIFIC_MINIMUM_OVERRIDES.keys():
        if compare_python_version(platform_bound):
            restrictions = PLATFORM_SPECIFIC_MINIMUM_OVERRIDES[platform_bound]
            if pkg_name in restrictions:
                versions = [v for v in versions if parse_version(v) >= parse_version(restrictions[pkg_name])]
    if originating_pkg_name in MINIMUM_VERSION_SPECIFIC_OVERRIDES and pkg_name in MINIMUM_VERSION_SPECIFIC_OVERRIDES[originating_pkg_name]:
        versions = [v for v in versions if parse_version(v) >= parse_version(MINIMUM_VERSION_SPECIFIC_OVERRIDES[originating_pkg_name][pkg_name])]
    if pkg_name in MAXIMUM_VERSION_GENERIC_OVERRIDES:
        versions = [v for v in versions if parse_version(v) <= parse_version(MAXIMUM_VERSION_GENERIC_OVERRIDES[pkg_name])]
    for platform_bound in PLATFORM_SPECIFIC_MAXIMUM_OVERRIDES.keys():
        if compare_python_version(platform_bound):
            restrictions = PLATFORM_SPECIFIC_MAXIMUM_OVERRIDES[platform_bound]
            if pkg_name in restrictions:
                versions = [v for v in versions if parse_version(v) <= parse_version(restrictions[pkg_name])]
    if originating_pkg_name in MAXIMUM_VERSION_SPECIFIC_OVERRIDES and pkg_name in MAXIMUM_VERSION_SPECIFIC_OVERRIDES[originating_pkg_name]:
        versions = [v for v in versions if parse_version(v) <= parse_version(MAXIMUM_VERSION_SPECIFIC_OVERRIDES[originating_pkg_name][pkg_name])]
    return versions

def process_requirement(req, dependency_type, orig_pkg_name):
    if False:
        while True:
            i = 10
    (pkg_name, spec) = parse_require(req)
    client = PyPIClient()
    versions = [str(v) for v in client.get_ordered_versions(pkg_name, True)]
    logging.info('Versions available on PyPI for %s: %s', pkg_name, versions)
    versions = process_bounded_versions(orig_pkg_name, pkg_name, versions)
    if dependency_type == 'Latest':
        versions.reverse()
    for version in versions:
        if spec is None:
            return pkg_name + '==' + version
        if version in spec:
            logging.info('Found %s version %s that matches specifier %s', dependency_type, version, spec)
            return pkg_name + '==' + version
    logging.error('No version is found on PyPI for package %s that matches specifier %s', pkg_name, spec)
    return ''

def check_req_against_exclusion(req, req_to_exclude):
    if False:
        return 10
    '\n    This function evaluates a requirement from a dev_requirements file against a file name. Returns True\n    if the requirement is for the same package listed in "req_to_exclude". False otherwise.\n\n    :param req: An incoming "req" looks like a requirement that appears in a dev_requirements file. EG: [ "../../../tools/azure-devtools",\n        "https://docsupport.blob.core.windows.net/repackaged/cffi-1.14.6-cp310-cp310-win_amd64.whl; sys_platform==\'win32\' and python_version >= \'3.10\'",\n        "msrestazure>=0.4.11", "pytest" ]\n\n    :param req_to_exclude: A valid and complete python package name. No specifiers.\n    '
    req_id = ''
    for c in req:
        if re.match('[A-Za-z0-9_-]', c):
            req_id += c
        else:
            break
    return req_id == req_to_exclude

def filter_dev_requirements(setup_py_path, released_packages, temp_dir):
    if False:
        i = 10
        return i + 15
    dev_req_path = os.path.join(os.path.dirname(setup_py_path), DEV_REQ_FILE)
    requirements = []
    with open(dev_req_path, 'r') as dev_req_file:
        requirements = dev_req_file.readlines()
    released_packages = [p.split('==')[0] for p in released_packages]
    prebuilt_dev_reqs = [os.path.basename(req.replace('\n', '')) for req in requirements if os.path.sep in req]
    req_to_exclude = [req for req in prebuilt_dev_reqs if req.split('-')[0].replace('_', '-') in released_packages]
    req_to_exclude.extend(released_packages)
    filtered_req = [req for req in requirements if os.path.basename(req.replace('\n', '')) not in req_to_exclude and (not any([check_req_against_exclusion(req, i) for i in req_to_exclude]))]
    logging.info('Filtered dev requirements: %s', filtered_req)
    new_dev_req_path = ''
    if filtered_req:
        new_dev_req_path = os.path.join(temp_dir, NEW_DEV_REQ_FILE)
        with open(new_dev_req_path, 'w') as dev_req_file:
            dev_req_file.writelines(filtered_req)
    return new_dev_req_path

def install_packages(packages, req_file):
    if False:
        return 10
    commands = [sys.executable, '-m', 'pip', 'install']
    if packages:
        commands.extend(packages)
    if req_file:
        commands.extend(['-r', req_file])
    logging.info('Installing packages. Command: %s', commands)
    check_call(commands)
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Install either latest or minimum version of dependent packages.')
    parser.add_argument('-t', '--target', dest='target_package', help='The target package directory on disk.', required=True)
    parser.add_argument('-d', '--dependency-type', dest='dependency_type', choices=['Latest', 'Minimum'], help="Dependency type to install. Dependency type is either 'Latest' or 'Minimum'", required=True)
    parser.add_argument('-w', '--work-dir', dest='work_dir', help='Temporary working directory to create new dev requirements file', required=True)
    args = parser.parse_args()
    setup_path = os.path.join(os.path.abspath(args.target_package), 'setup.py')
    if not (os.path.exists(setup_path) and os.path.exists(args.work_dir)):
        logging.error('Invalid arguments. Please make sure target directory and working directory are valid path')
        sys.exit(1)
    install_dependent_packages(setup_path, args.dependency_type, args.work_dir)