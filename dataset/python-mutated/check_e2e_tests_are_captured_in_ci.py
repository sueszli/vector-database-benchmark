"""A script to check that the CI config files & wdio.conf.js have
the same e2e test suites.
"""
from __future__ import annotations
import os
import re
from scripts import common
from core import utils
from typing import List
import yaml
TEST_SUITES_NOT_RUN_IN_CI = ['full']
WEBDRIVERIO_CONF_FILE_PATH = os.path.join(os.getcwd(), 'core', 'tests', 'wdio.conf.js')
SAMPLE_TEST_SUITE_THAT_IS_KNOWN_TO_EXIST = 'publication'
CI_PATH = os.path.join(os.getcwd(), '.github', 'workflows')

def get_e2e_suite_names_from_ci_config_file() -> List[str]:
    if False:
        i = 10
        return i + 15
    'Extracts the script section from the CI config files.\n\n    Returns:\n        list(str). An alphabetically-sorted list of names of test suites\n        from the script section in the CI config files.\n    '
    suites = []
    file_contents = read_and_parse_ci_config_files()
    for file_content in file_contents:
        workflow_dict = yaml.load(file_content, Loader=yaml.Loader)
        suites += workflow_dict['jobs']['e2e_test']['strategy']['matrix']['suite']
    return sorted(suites)

def get_e2e_suite_names_from_webdriverio_file() -> List[str]:
    if False:
        print('Hello World!')
    'Extracts the test suites section from the wdio.conf.js file.\n\n    Returns:\n        list(str). An alphabetically-sorted list of names of test suites\n        from the wdio.conf.js file.\n    '
    webdriverio_config_file_content = read_webdriverio_conf_file()
    suite_object_string = re.compile('suites = {([^}]+)}').findall(webdriverio_config_file_content)[0]
    key_regex = re.compile('\\b([a-zA-Z_-]*):')
    webdriverio_suites = key_regex.findall(suite_object_string)
    return sorted(webdriverio_suites)

def read_webdriverio_conf_file() -> str:
    if False:
        return 10
    'Returns the contents of core/tests/wdio.conf.js file.\n\n    Returns:\n        str. The contents of wdio.conf.js, as a string.\n    '
    webdriverio_config_file_content = utils.open_file(WEBDRIVERIO_CONF_FILE_PATH, 'r').read()
    return webdriverio_config_file_content

def read_and_parse_ci_config_files() -> List[str]:
    if False:
        print('Hello World!')
    'Returns the contents of CI config files.\n\n    Returns:\n        list(str). Contents of the CI config files.\n    '
    ci_dicts = []
    for filepath in os.listdir(CI_PATH):
        if re.fullmatch('e2e_.*\\.yml', filepath):
            ci_file_content = utils.open_file(os.path.join(CI_PATH, filepath), 'r').read()
            ci_dicts.append(ci_file_content)
    return ci_dicts

def get_e2e_test_filenames_from_webdriverio_dir() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Extracts the names of the all test files in core/tests/webdriverio\n    and core/tests/webdriverio_desktop directory.\n\n    Returns:\n        list(str). An alphabetically-sorted list of of the all test files\n        in core/tests/webdriverio and core/tests/webdriverio_desktop directory.\n    '
    webdriverio_test_suite_files = []
    webdriverio_files = os.path.join(os.getcwd(), 'core', 'tests', 'webdriverio')
    webdriverio_desktop_files = os.path.join(os.getcwd(), 'core', 'tests', 'webdriverio_desktop')
    for file_name in os.listdir(webdriverio_files):
        webdriverio_test_suite_files.append(file_name)
    for file_name in os.listdir(webdriverio_desktop_files):
        webdriverio_test_suite_files.append(file_name)
    return sorted(webdriverio_test_suite_files)

def get_e2e_test_filenames_from_webdriverio_conf_file() -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Extracts the filenames from the suites object of\n    wdio.conf.js file.\n\n    Returns:\n        list(str). An alphabetically-sorted list of filenames extracted\n        from the wdio.conf.js file.\n    '
    webdriverio_config_file_content = read_webdriverio_conf_file()
    suite_object_string = re.compile('suites = {([^}]+)}').findall(webdriverio_config_file_content)[0]
    test_files_regex = re.compile('/([a-zA-Z]*.js)')
    e2e_test_files = test_files_regex.findall(suite_object_string)
    return sorted(e2e_test_files)

def main() -> None:
    if False:
        return 10
    'Check that the CI config files and wdio.conf.js have the same\n    e2e test suites.\n    '
    print('Checking all e2e test files are captured in wdio.conf.js...')
    webdriverio_test_suite_files = get_e2e_test_filenames_from_webdriverio_dir()
    webdriverio_conf_test_suites = get_e2e_test_filenames_from_webdriverio_conf_file()
    if not webdriverio_test_suite_files == webdriverio_conf_test_suites:
        raise Exception('One or more test file from webdriverio or webdriverio_desktop directory is missing from wdio.conf.js')
    print('Done!')
    print('Checking e2e tests are captured in CI config files...')
    webdriverio_test_suites = get_e2e_suite_names_from_webdriverio_file()
    ci_suite_names = get_e2e_suite_names_from_ci_config_file()
    for excluded_test in TEST_SUITES_NOT_RUN_IN_CI:
        webdriverio_test_suites.remove(excluded_test)
    if not ci_suite_names:
        raise Exception('The e2e test suites that have been extracted from script section from CI config files are empty.')
    if not webdriverio_test_suites:
        raise Exception('The e2e test suites that have been extracted from wdio.conf.js are empty.')
    if set(webdriverio_test_suites) != set(ci_suite_names):
        raise Exception('WebdriverIO test suites and CI test suites are not in sync. Following suites are not in sync: {}'.format(utils.compute_list_difference(webdriverio_test_suites, ci_suite_names)))
    print('Done!')
if __name__ == '__main__':
    main()