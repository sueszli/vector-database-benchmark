"""Script that provides changes specific to oppia repo to be written
to release summary file.
"""
from __future__ import annotations
import argparse
import os
import re
from typing import Dict, Final, List, Optional
from scripts import common
from core import utils
GIT_CMD_DIFF_NAMES_ONLY_FORMAT_STRING: Final = 'git diff --name-only %s %s'
GIT_CMD_SHOW_FORMAT_STRING: Final = 'git show %s:core/feconf.py'
VERSION_RE_FORMAT_STRING: Final = '%s\\s*=\\s*(\\d+|\\.)+'
FECONF_SCHEMA_VERSION_CONSTANT_NAMES: Final = ['CURRENT_STATE_SCHEMA_VERSION', 'CURRENT_COLLECTION_SCHEMA_VERSION']
FECONF_FILEPATH: Final = os.path.join('core', 'feconf.py')
_PARSER: Final = argparse.ArgumentParser()
_PARSER.add_argument('--release_tag', required=True, type=str, help='The release tag from which to fetch the changes.')

def get_changed_schema_version_constant_names(release_tag_to_diff_against: str) -> List[str]:
    if False:
        return 10
    'Returns a list of schema version constant names in feconf that have\n    changed since the release against which diff is being checked.\n\n    Args:\n        release_tag_to_diff_against: str. The release tag to diff against.\n\n    Returns:\n        list(str). List of version constant names in feconf that changed.\n    '
    changed_version_constants_in_feconf = []
    git_show_cmd = GIT_CMD_SHOW_FORMAT_STRING % release_tag_to_diff_against
    old_feconf = common.run_cmd(git_show_cmd.split(' '))
    with utils.open_file(FECONF_FILEPATH, 'r') as feconf_file:
        new_feconf = feconf_file.read()
    for version_constant in FECONF_SCHEMA_VERSION_CONSTANT_NAMES:
        old_version = re.findall(VERSION_RE_FORMAT_STRING % version_constant, old_feconf)[0]
        new_version = re.findall(VERSION_RE_FORMAT_STRING % version_constant, new_feconf)[0]
        if old_version != new_version:
            changed_version_constants_in_feconf.append(version_constant)
    return changed_version_constants_in_feconf

def _get_changed_filenames_since_tag(release_tag_to_diff_against: str) -> List[str]:
    if False:
        print('Hello World!')
    'Get names of changed files from git since a given release.\n\n    Args:\n        release_tag_to_diff_against: str. The release tag to diff against.\n\n    Returns:\n        list(str). List of filenames for files that have been modified since\n        the release against which diff is being checked.\n    '
    diff_cmd = GIT_CMD_DIFF_NAMES_ONLY_FORMAT_STRING % (release_tag_to_diff_against, 'HEAD')
    return common.run_cmd(diff_cmd.split(' ')).splitlines()

def get_setup_scripts_changes_status(release_tag_to_diff_against: str) -> Dict[str, bool]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a dict of setup script filepaths with a status of whether\n    they have changed or not since the release against which diff is\n    being checked.\n\n    Args:\n        release_tag_to_diff_against: str. The release tag to diff against.\n\n    Returns:\n        dict. Dict consisting of key as script name and value as boolean\n        indicating whether or not the script is modified since the release\n        against which diff is being checked.\n    '
    setup_script_filepaths = ['scripts/%s' % item for item in ['setup.py', 'setup_gae.py', 'install_third_party_libs.py', 'install_third_party.py']]
    changed_filenames = _get_changed_filenames_since_tag(release_tag_to_diff_against)
    changes_dict = {script_filepath: script_filepath in changed_filenames for script_filepath in setup_script_filepaths}
    return changes_dict

def get_changed_storage_models_filenames(release_tag_to_diff_against: str) -> List[str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns a list of filepaths in core/storage whose contents have\n    changed since the release against which diff is being checked.\n\n    Args:\n        release_tag_to_diff_against: str. The release tag to diff against.\n\n    Returns:\n        list(str). The changed filenames in core/storage (if any).\n    '
    changed_model_filenames = _get_changed_filenames_since_tag(release_tag_to_diff_against)
    return [model_filename for model_filename in changed_model_filenames if model_filename.startswith('core/storage')]

def get_changes(release_tag_to_diff_against: str) -> List[str]:
    if False:
        return 10
    'Collects changes in storage models, setup scripts and feconf\n    since the release tag passed in arguments.\n\n    Args:\n        release_tag_to_diff_against: str. The release tag to diff against.\n\n    Returns:\n        list(str). A list of lines to be written to the release summary file.\n        These lines describe the changed storage model names, setup script names\n        and feconf schema version names since the release against which diff is\n        being checked.\n    '
    changes = []
    feconf_version_changes = get_changed_schema_version_constant_names(release_tag_to_diff_against)
    if feconf_version_changes:
        changes.append('\n### Feconf version changes:\nThis indicates that a migration may be needed\n\n')
        for var in feconf_version_changes:
            changes.append('* %s\n' % var)
    setup_changes = get_setup_scripts_changes_status(release_tag_to_diff_against)
    if setup_changes:
        changes.append('\n### Changed setup scripts:\n')
        for var in setup_changes.keys():
            changes.append('* %s\n' % var)
    storage_changes = get_setup_scripts_changes_status(release_tag_to_diff_against)
    if storage_changes:
        changes.append('\n### Changed storage models:\n')
        for item in storage_changes:
            changes.append('* %s\n' % item)
    return changes

def main(args: Optional[List[str]]=None) -> None:
    if False:
        while True:
            i = 10
    'Main method for fetching repo specific changes.'
    options = _PARSER.parse_args(args=args)
    changes = get_changes(options.release_tag)
    print('\n'.join(changes))
if __name__ == '__main__':
    main()