"""Helper script used for updating feconf.

ONLY RELEASE COORDINATORS SHOULD USE THIS SCRIPT.

Usage: Run this script from your oppia root folder:

    python -m scripts.release_scripts.update_configs
"""
from __future__ import annotations
import argparse
import os
import re
from core import utils
from scripts import common
from typing import Final, List, Optional
FECONF_REGEX: Final = '^([A-Z_]+ = ).*$'
CONSTANTS_REGEX: Final = '^(  "[A-Z_]+": ).*$'
_PARSER: Final = argparse.ArgumentParser(description='Updates configs.')
_PARSER.add_argument('--release_dir_path', dest='release_dir_path', help='Path of directory where all files are copied for release.', required=True)
_PARSER.add_argument('--deploy_data_path', dest='deploy_data_path', help='Path for deploy data directory.', required=True)
_PARSER.add_argument('--personal_access_token', dest='personal_access_token', help='The personal access token for the GitHub id of user.', default=None)

def apply_changes_based_on_config(local_filepath: str, config_filepath: str, expected_config_line_regex: str) -> None:
    if False:
        while True:
            i = 10
    'Updates the local file based on the deployment configuration specified\n    in the config file.\n\n    Each line of the config file should match the expected config line regex.\n\n    Args:\n        local_filepath: str. Absolute path of the local file to be modified.\n        config_filepath: str. Absolute path of the config file to use.\n        expected_config_line_regex: str. The regex to use to verify each line\n            of the config file. It should have a single group, which\n            corresponds to the prefix to extract.\n\n    Raises:\n        Exception. Line(s) in config file are not matching with the regex.\n    '
    with utils.open_file(config_filepath, 'r') as config_file:
        config_lines = config_file.read().splitlines()
    with utils.open_file(local_filepath, 'r') as local_file:
        local_lines = local_file.read().splitlines()
    local_filename = os.path.basename(local_filepath)
    config_filename = os.path.basename(config_filepath)
    local_line_numbers = []
    for config_line in config_lines:
        match_result = re.match(expected_config_line_regex, config_line)
        if match_result is None:
            raise Exception('Invalid line in %s config file: %s' % (config_filename, config_line))
        matching_local_line_numbers = [line_number for (line_number, line) in enumerate(local_lines) if line.startswith(match_result.group(1))]
        assert len(matching_local_line_numbers) == 1, 'Could not find correct number of lines in %s matching: %s, %s' % (local_filename, config_line, matching_local_line_numbers)
        local_line_numbers.append(matching_local_line_numbers[0])
    for (index, config_line) in enumerate(config_lines):
        local_lines[local_line_numbers[index]] = config_line
    with utils.open_file(local_filepath, 'w') as writable_local_file:
        writable_local_file.write('\n'.join(local_lines) + '\n')

def update_app_yaml(release_app_dev_yaml_path: str, feconf_config_path: str) -> None:
    if False:
        return 10
    'Updates app.yaml file with more strict CORS HTTP header.\n\n    Args:\n        release_app_dev_yaml_path: str. Absolute path of the app_dev.yaml file.\n        feconf_config_path: str. Absolute path of the feconf config file.\n\n    Raises:\n        Exception. No OPPIA_SITE_URL key found.\n    '
    with utils.open_file(feconf_config_path, 'r') as feconf_config_file:
        feconf_config_contents = feconf_config_file.read()
    with utils.open_file(release_app_dev_yaml_path, 'r') as app_yaml_file:
        app_yaml_contents = app_yaml_file.read()
    oppia_site_url_searched_key = re.search("OPPIA_SITE_URL = \\'(.*)\\'", feconf_config_contents)
    if oppia_site_url_searched_key is None:
        raise Exception('Error: No OPPIA_SITE_URL key found.')
    project_origin = oppia_site_url_searched_key.group(1)
    access_control_allow_origin_header = 'Access-Control-Allow-Origin: %s' % project_origin
    (edited_app_yaml_contents, _) = re.subn('Access-Control-Allow-Origin: \\"\\*\\"', access_control_allow_origin_header, app_yaml_contents)
    with utils.open_file(release_app_dev_yaml_path, 'w') as app_yaml_file:
        app_yaml_file.write(edited_app_yaml_contents)

def verify_config_files(release_feconf_path: str, release_app_dev_yaml_path: str) -> None:
    if False:
        i = 10
        return i + 15
    'Verifies that feconf is updated correctly to include\n    redishost and app.yaml to include correct headers.\n\n    Args:\n        release_feconf_path: str. The path to feconf file in release\n            directory.\n        release_app_dev_yaml_path: str. The path to app_dev.yaml file in release\n            directory.\n\n    Raises:\n        Exception. REDISHOST not updated before deployment.\n        Exception. Access-Control-Allow-Origin not updated to specific origin\n            before deployment.\n    '
    feconf_contents = utils.open_file(release_feconf_path, 'r').read()
    if 'REDISHOST' not in feconf_contents or "REDISHOST = 'localhost'" in feconf_contents:
        raise Exception('REDISHOST must be updated before deployment.')
    with utils.open_file(release_app_dev_yaml_path, 'r') as app_yaml_file:
        app_yaml_contents = app_yaml_file.read()
    if 'Access-Control-Allow-Origin: "*"' in app_yaml_contents:
        raise Exception('\'Access-Control-Allow-Origin: "*"\' must be updated to a specific origin before deployment.')

def update_analytics_constants_based_on_config(release_analytics_constants_path: str, analytics_constants_config_path: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the GA4 and UA IDs in the analytics constants JSON file.\n\n    Args:\n        release_analytics_constants_path: str. The path to constants file.\n        analytics_constants_config_path: str. The path to constants config file.\n\n    Raises:\n        Exception. No GA_ANALYTICS_ID key found.\n        Exception. No SITE_NAME_FOR_ANALYTICS key found.\n        Exception. No CAN_SEND_ANALYTICS_EVENTS key found.\n    '
    with utils.open_file(analytics_constants_config_path, 'r') as config_file:
        config_file_contents = config_file.read()
    ga_analytics_searched_key = re.search('"GA_ANALYTICS_ID": "(.*)"', config_file_contents)
    if ga_analytics_searched_key is None:
        raise Exception('Error: No GA_ANALYTICS_ID key found.')
    ga_analytics_id = ga_analytics_searched_key.group(1)
    site_name_for_analytics_searched_key = re.search('"SITE_NAME_FOR_ANALYTICS": "(.*)"', config_file_contents)
    if site_name_for_analytics_searched_key is None:
        raise Exception('Error: No SITE_NAME_FOR_ANALYTICS key found.')
    site_name_for_analytics = site_name_for_analytics_searched_key.group(1)
    can_send_analytics_events_searched_key = re.search('"CAN_SEND_ANALYTICS_EVENTS": (true|false)', config_file_contents)
    if can_send_analytics_events_searched_key is None:
        raise Exception('Error: No CAN_SEND_ANALYTICS_EVENTS key found.')
    can_send_analytics_events = can_send_analytics_events_searched_key.group(1)
    common.inplace_replace_file(release_analytics_constants_path, '"GA_ANALYTICS_ID": ""', '"GA_ANALYTICS_ID": "%s"' % ga_analytics_id)
    common.inplace_replace_file(release_analytics_constants_path, '"SITE_NAME_FOR_ANALYTICS": ""', '"SITE_NAME_FOR_ANALYTICS": "%s"' % site_name_for_analytics)
    common.inplace_replace_file(release_analytics_constants_path, '"CAN_SEND_ANALYTICS_EVENTS": false', '"CAN_SEND_ANALYTICS_EVENTS": %s' % can_send_analytics_events)

def main(args: Optional[List[str]]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Updates the files corresponding to LOCAL_FECONF_PATH and\n    LOCAL_CONSTANTS_PATH after doing the prerequisite checks.\n    '
    options = _PARSER.parse_args(args=args)
    feconf_config_path = os.path.join(options.deploy_data_path, 'feconf_updates.config')
    constants_config_path = os.path.join(options.deploy_data_path, 'constants_updates.config')
    analytics_constants_config_path = os.path.join(options.deploy_data_path, 'analytics_constants_updates.config')
    release_feconf_path = os.path.join(options.release_dir_path, common.FECONF_PATH)
    release_constants_path = os.path.join(options.release_dir_path, common.CONSTANTS_FILE_PATH)
    release_app_dev_yaml_path = os.path.join(options.release_dir_path, common.APP_DEV_YAML_PATH)
    release_analytics_constants_path = os.path.join(options.release_dir_path, common.ANALYTICS_CONSTANTS_FILE_PATH)
    apply_changes_based_on_config(release_feconf_path, feconf_config_path, FECONF_REGEX)
    apply_changes_based_on_config(release_constants_path, constants_config_path, CONSTANTS_REGEX)
    update_app_yaml(release_app_dev_yaml_path, feconf_config_path)
    update_analytics_constants_based_on_config(release_analytics_constants_path, analytics_constants_config_path)
    verify_config_files(release_feconf_path, release_app_dev_yaml_path)
if __name__ == '__main__':
    main()