"""Helper script used for creating a new release or hotfix branch on GitHub.

ONLY RELEASE COORDINATORS SHOULD USE THIS SCRIPT.

Usage: Run this script from your oppia root folder:

For release branch:

    python -m scripts.release_scripts.cut_release_or_hotfix_branch
    --release_version="x.y.z"

where x.y.z is the new version of Oppia, e.g. 2.5.3. The generated branch
name will be release-x.y.z, e.g. release-2.5.3.

For hotfix branch:

    python -m scripts.release_scripts.cut_release_or_hotfix_branch
    --release_version="x.y.z" --hotfix_number=d

where x.y.z is the new version of Oppia, e.g. 2.5.3,
d is number of the hotfix being created, e.g. 1. The generated branch
name will be release-x.y.z-hotfix-d, e.g. release-2.5.3-hotfix-1.
"""
from __future__ import annotations
import argparse
import json
import re
import subprocess
from core import constants
from core import utils
from scripts import common
from typing import Final, List, Optional, Pattern, Tuple
BRANCH_TYPE_HOTFIX = 'hotfix'
BRANCH_TYPE_RELEASE = 'release'

def require_release_version_to_have_correct_format(arg: str, pattern: Pattern[str]=re.compile('\\d\\.\\d\\.\\d')) -> str:
    if False:
        return 10
    'Checks that the release version name matches the expected pattern.\n\n    Args:\n        arg: str. The release version name.\n        pattern: RegularExpression. The pattern that release version should\n            match.\n\n    Raises:\n        argparse.ArgumentTypeError. The release version name does not match\n            the pattern.\n\n    Returns:\n        str. The release version name with correct pattern.\n    '
    if not pattern.match(arg):
        raise argparse.ArgumentTypeError('The format of "release_version" should be: x.x.x')
    return arg
_PARSER: Final = argparse.ArgumentParser()
_PARSER.add_argument('--release_version', help='version of the release for which the branch cut is being made or the hotfix is being created', type=require_release_version_to_have_correct_format)
_PARSER.add_argument('--hotfix_number', default=0)

def verify_target_branch_does_not_already_exist(remote_alias: str, new_branch_name: str) -> None:
    if False:
        while True:
            i = 10
    "Checks that the new release branch doesn't already exist locally or\n    remotely.\n\n    Args:\n        remote_alias: str. The alias that points to the remote oppia\n            repository. Example: When calling git remote -v, you get:\n            upstream    https://github.com/oppia/oppia.git (fetch),\n            where 'upstream' is the alias that points to the remote oppia\n            repository.\n        new_branch_name: str. The name of the new branch to cut.\n\n    Raises:\n        Exception. The target branch name already exists locally.\n        Exception. The target branch name already exists on the remote\n            oppia repository.\n    "
    git_branch_output = subprocess.check_output(['git', 'branch'], encoding='utf-8').split('\n')
    if new_branch_name in git_branch_output:
        raise Exception('ERROR: The target branch name already exists locally. Run "git branch -D %s" to delete it.' % new_branch_name)
    git_ls_remote_output = subprocess.check_output(['git', 'ls-remote', '--heads', remote_alias], encoding='utf-8').split('\n')
    remote_branch_ref = 'refs/heads/%s' % new_branch_name
    if remote_branch_ref in git_ls_remote_output:
        raise Exception('ERROR: The target branch name already exists on the remote repo.')

def verify_target_version_compatible_with_latest_release(target_version: str) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Checks that the target version is consistent with the latest released\n    version on GitHub.\n\n    Args:\n        target_version: str. The release version.\n\n    Raises:\n        Exception. Failed to fetch latest release info from GitHub.\n        Exception. Could not parse version number of latest GitHub release.\n        AssertionError. The previous and the current major version are not the\n            same.\n        AssertionError. The current patch version is not equal to previous patch\n            version plus one.\n        AssertionError. The current patch version is greater or equal to 10.\n        AssertionError. The current minor version is not equal to previous\n            minor version plus one.\n        AssertionError. The current patch version is different than 0.\n    '
    response = utils.url_open('https://api.github.com/repos/oppia/oppia/releases/latest')
    if response.getcode() != 200:
        raise Exception('ERROR: Failed to fetch latest release info from GitHub.')
    data = json.load(response)
    latest_release_tag_name = data['tag_name']
    match_result = re.match('v(\\d)\\.(\\d)\\.(\\d)', latest_release_tag_name)
    if match_result is None:
        raise Exception('ERROR: Could not parse version number of latest GitHub release.')
    (prev_major, prev_minor, prev_patch) = match_result.group(1, 2, 3)
    match_result = re.match('(\\d)\\.(\\d)\\.(\\d)', target_version)
    if match_result is None:
        raise Exception('ERROR: Could not parse target version.')
    (curr_major, curr_minor, curr_patch) = match_result.group(1, 2, 3)
    assert prev_major == curr_major, 'Unexpected major version change.'
    if prev_minor == curr_minor:
        assert int(curr_patch) == int(prev_patch) + 1, 'The current patch version is not equal to previous patch version plus one.'
    else:
        assert int(curr_minor) == int(prev_minor) + 1, 'The current minor version is not equal to previous minor version plus one.'
        assert int(curr_patch) == 0, 'The current patch version is different than 0.'

def verify_hotfix_number_is_one_ahead_of_previous_hotfix_number(remote_alias: str, target_version: str, hotfix_number: int) -> None:
    if False:
        while True:
            i = 10
    "Checks that the hotfix number is one ahead of previous hotfix\n    number.\n\n    Args:\n        remote_alias: str. The alias that points to the remote oppia\n            repository. Example: When calling git remote -v, you get:\n            upstream    https://github.com/oppia/oppia.git (fetch),\n            where 'upstream' is the alias that points to the remote oppia\n            repository.\n        target_version: str. The release version.\n        hotfix_number: int. The number for the hotfix branch.\n\n    Raises:\n        Exception. The difference between two continuous hotfix numbers\n            is not one.\n    "
    all_branches = subprocess.check_output(['git', 'branch', '-a'], encoding='utf-8').split('\n')
    last_hotfix_number = 0
    release_branch_exists = False
    hotfix_branch_name_regex = '^remotes/%s/release-%s-hotfix-\\d*$' % (remote_alias, target_version)
    for branch_name in all_branches:
        branch_name = branch_name.lstrip().rstrip()
        if branch_name == 'remotes/%s/release-%s' % (remote_alias, target_version):
            release_branch_exists = True
        if re.match(hotfix_branch_name_regex, branch_name):
            branch_hotfix_number = int(branch_name[branch_name.rfind('-') + 1:])
            if branch_hotfix_number > last_hotfix_number:
                last_hotfix_number = branch_hotfix_number
    assert release_branch_exists, 'Release branch is missing.'
    assert hotfix_number == last_hotfix_number + 1, 'The difference between two continuous hotfix numbers is not one.'

def _get_release_branch_type_and_name(target_version: str) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns type and name of release branch for a target version.\n\n    Args:\n        target_version: str. The release version.\n\n    Returns:\n        tuple(str, str). The type and name of release branch.\n    '
    return (BRANCH_TYPE_RELEASE, 'release-%s' % target_version)

def _get_hotfix_branch_type_and_name(target_version: str, hotfix_number: int) -> Tuple[str, str]:
    if False:
        for i in range(10):
            print('nop')
    'Returns type and name of hotfix branch for a target version.\n\n    Args:\n        target_version: str. The release version.\n        hotfix_number: int. The number for the hotfix branch.\n\n    Returns:\n        tuple(str, str). The type and name of hotfix branch.\n    '
    return (BRANCH_TYPE_HOTFIX, 'release-%s-hotfix-%s' % (target_version, hotfix_number))

def execute_branch_cut(target_version: str, hotfix_number: int) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Creates & pushes the new release branch to Github.\n\n    Args:\n        target_version: str. The release version.\n        hotfix_number: int. The number for the hotfix branch.\n\n    Raises:\n        Exception. Actions tests are failing on the branch from which\n            the new branch is cut.\n    '
    if not hotfix_number:
        (new_branch_type, new_branch_name) = _get_release_branch_type_and_name(target_version)
    else:
        (new_branch_type, new_branch_name) = _get_hotfix_branch_type_and_name(target_version, hotfix_number)
    common.require_cwd_to_be_oppia()
    common.verify_local_repo_is_clean()
    common.verify_current_branch_name('develop')
    remote_alias = common.get_remote_alias(constants.release_constants.REMOTE_URLS)
    subprocess.check_call(['git', 'pull', remote_alias, 'develop'])
    verify_target_branch_does_not_already_exist(remote_alias, new_branch_name)
    if not hotfix_number:
        branch_to_check = 'develop'
    elif hotfix_number == 1:
        branch_to_check = 'release-%s' % target_version
    else:
        branch_to_check = 'release-%s-hotfix-%s' % (target_version, hotfix_number - 1)
    common.open_new_tab_in_browser_if_possible('https://github.com/oppia/oppia/actions?query=branch:%s' % branch_to_check)
    print('Please confirm: are Actions checks passing on %s? (y/n) ' % branch_to_check)
    answer = input().lower()
    if answer not in common.AFFIRMATIVE_CONFIRMATIONS:
        raise Exception('Tests should pass on %s before this script is run.' % branch_to_check)
    if new_branch_type == BRANCH_TYPE_HOTFIX:
        verify_hotfix_number_is_one_ahead_of_previous_hotfix_number(remote_alias, target_version, hotfix_number)
        if hotfix_number == 1:
            branch_to_cut_from = 'release-%s' % target_version
        else:
            branch_to_cut_from = 'release-%s-hotfix-%s' % (target_version, hotfix_number - 1)
        print('Cutting a new hotfix branch: %s' % new_branch_name)
        subprocess.check_call(['git', 'checkout', branch_to_cut_from])
        common.update_branch_with_upstream()
        subprocess.check_call(['git', 'checkout', '-b', new_branch_name, branch_to_cut_from])
    else:
        verify_target_version_compatible_with_latest_release(target_version)
        print('Cutting a new release branch: %s' % new_branch_name)
        subprocess.check_call(['git', 'checkout', '-b', new_branch_name])
    if new_branch_type == BRANCH_TYPE_RELEASE:
        print('Pushing new %s branch to GitHub.' % new_branch_type)
        subprocess.check_call(['git', 'push', remote_alias, new_branch_name])
    else:
        print('Please cherrypick the required PRs and push the branch to Github once this script is done.\nNote: It is fine to push the branch only after creating the branch protection rule and doing all the cherrypicks.')
    print('')
    print('New %s branch successfully cut. You are now on branch %s' % (new_branch_type, new_branch_name))
    print('Done!')

def main(args: Optional[List[str]]=None) -> None:
    if False:
        for i in range(10):
            print('nop')
    'Main method for creating a release or hotfix branch.'
    parsed_args = _PARSER.parse_args(args=args)
    if parsed_args.release_version:
        target_version = parsed_args.release_version
    else:
        raise Exception('ERROR: A "release_version" arg must be specified.')
    hotfix_number = int(parsed_args.hotfix_number)
    execute_branch_cut(target_version, hotfix_number)
if __name__ == '__main__':
    main()