"""Pre-push hook that executes the Python/JS linters on all files that
deviate from develop.
(By providing the list of files to `scripts.linters.pre_commit_linter`)
To install the hook manually simply execute this script from the oppia root dir
with the `--install` flag.
To bypass the validation upon `git push` use the following command:
`git push REMOTE BRANCH --no-verify`

This hook works only on Unix like systems as of now.
On Vagrant under Windows it will still copy the hook to the .git/hooks dir
but it will have no effect.
"""
from __future__ import annotations
import argparse
import collections
import os
import pprint
import re
import shutil
import subprocess
import sys
from types import TracebackType
from typing import Dict, Final, List, Optional, Tuple, Type
sys.path.append(os.getcwd())
from scripts import common
from scripts import install_python_prod_dependencies
GitRef = collections.namedtuple('GitRef', ['local_ref', 'local_sha1', 'remote_ref', 'remote_sha1'])
FileDiff = collections.namedtuple('FileDiff', ['status', 'name'])
GIT_NULL_COMMIT: Final = '4b825dc642cb6eb9a060e54bf8d69288fbee4904'
LINTER_MODULE: Final = 'scripts.linters.pre_commit_linter'
MYPY_TYPE_CHECK_MODULE: Final = 'scripts.run_mypy_checks'
FILE_DIR: Final = os.path.abspath(os.path.dirname(__file__))
OPPIA_DIR: Final = os.path.join(FILE_DIR, os.pardir, os.pardir)
LINTER_FILE_FLAG: Final = '--files'
PYTHON_CMD: Final = 'python'
OPPIA_PARENT_DIR: Final = os.path.join(FILE_DIR, os.pardir, os.pardir, os.pardir)
FRONTEND_TEST_CMDS: Final = [PYTHON_CMD, '-m', 'scripts.run_frontend_tests', '--check_coverage']
BACKEND_ASSOCIATED_TEST_FILE_CHECK_CMD: Final = [PYTHON_CMD, '-m', 'scripts.check_backend_associated_test_file']
CI_PROTRACTOR_CHECK_CMDS: Final = [PYTHON_CMD, '-m', 'scripts.check_e2e_tests_are_captured_in_ci']
TYPESCRIPT_CHECKS_CMDS: Final = [PYTHON_CMD, '-m', 'scripts.typescript_checks']
STRICT_TYPESCRIPT_CHECKS_CMDS: Final = [PYTHON_CMD, '-m', 'scripts.typescript_checks', '--strict_checks']
GIT_IS_DIRTY_CMD: Final = 'git status --porcelain --untracked-files=no'

class ChangedBranch:
    """Context manager class that changes branch when there are modified files
    that need to be linted. It does not change branch when modified files are
    not committed.
    """

    def __init__(self, new_branch: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        get_branch_cmd = 'git symbolic-ref -q --short HEAD'.split()
        self.old_branch = subprocess.check_output(get_branch_cmd, encoding='utf-8').strip()
        self.new_branch = new_branch
        self.is_same_branch = self.old_branch == self.new_branch

    def __enter__(self) -> None:
        if False:
            return 10
        if not self.is_same_branch:
            try:
                subprocess.check_output(['git', 'checkout', self.new_branch, '--'], encoding='utf-8')
            except subprocess.CalledProcessError:
                print('\nCould not change branch to %s. This is most probably because you are in a dirty state. Change manually to the branch that is being linted or stash your changes.' % self.new_branch)
                sys.exit(1)

    def __exit__(self, exc_type: Optional[Type[BaseException]], exc_val: Optional[BaseException], exc_tb: Optional[TracebackType]) -> None:
        if False:
            i = 10
            return i + 15
        if not self.is_same_branch:
            subprocess.check_output(['git', 'checkout', self.old_branch, '--'], encoding='utf-8')

def start_subprocess_for_result(cmd: List[str]) -> Tuple[bytes, bytes]:
    if False:
        print('Hello World!')
    'Starts subprocess and returns (stdout, stderr).'
    task = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = task.communicate()
    return (out, err)

def get_remote_name() -> Optional[bytes]:
    if False:
        return 10
    'Get the remote name of the local repository.\n\n    Returns:\n        Optional[bytes]. The remote name of the local repository.\n\n    Raises:\n        ValueError. Subprocess failed to start.\n        Exception. Upstream not set.\n    '
    remote_name = b''
    remote_num = 0
    get_remotes_name_cmd = 'git remote'.split()
    task = subprocess.Popen(get_remotes_name_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (out, err) = task.communicate()
    remotes = out[:-1].split(b'\n')
    if not err:
        for remote in remotes:
            get_remotes_url_cmd = (b'git config --get remote.%s.url' % remote).split()
            task = subprocess.Popen(get_remotes_url_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            (remote_url, err) = task.communicate()
            if not err:
                if remote_url.endswith(b'oppia/oppia.git\n'):
                    remote_num += 1
                    remote_name = remote
            else:
                raise ValueError(err)
    else:
        raise ValueError(err)
    if not remote_num:
        raise Exception("Error: Please set upstream for the lint checks to run efficiently. To do that follow these steps:\n1. Run the command 'git remote -v'\n2a. If upstream is listed in the command output, then run the command 'git remote set-url upstream https://github.com/oppia/oppia.git'\n2b. If upstream is not listed in the command output, then run the command 'git remote add upstream https://github.com/oppia/oppia.git'\n")
    if remote_num > 1:
        print('Warning: Please keep only one remote branch for oppia:develop to run the lint checks efficiently.\n')
        return None
    return remote_name

def git_diff_name_status(left: str, right: str, diff_filter: str='') -> List[FileDiff]:
    if False:
        i = 10
        return i + 15
    'Compare two branches/commits with git.\n\n    Parameter:\n        left: str. The name of the lefthand branch.\n        right: str. The name of the righthand branch.\n        diff_filter: str. Arguments given to --diff-filter (ACMRTD...).\n\n    Returns:\n        list. List of FileDiffs (tuple with name/status).\n\n    Raises:\n        ValueError. Raise ValueError if git command fails.\n    '
    git_cmd = ['git', 'diff', '--name-status']
    if diff_filter:
        git_cmd.append('--diff-filter={}'.format(diff_filter))
    git_cmd.extend([left, right])
    git_cmd.append('--')
    (out, err) = start_subprocess_for_result(git_cmd)
    if not err:
        file_list = []
        for line in out.splitlines():
            file_list.append(FileDiff(bytes([line[0]]), line[line.rfind(b'\t') + 1:]))
        return file_list
    else:
        raise ValueError(err)

def get_merge_base(branch: str, other_branch: str) -> str:
    if False:
        print('Hello World!')
    "Returns the most-recent commit shared by both branches. Order doesn't\n    matter.\n\n    The commit returned is the same one used on GitHub's UI for comparing pull\n    requests.\n\n    Args:\n        branch: str. A branch name or commit hash.\n        other_branch: str. A branch name or commit hash.\n\n    Returns:\n        str. The common commit hash shared by both branches.\n\n    Raises:\n        ValueError. An error occurred in the git command.\n    "
    (merge_base, err) = start_subprocess_for_result(['git', 'merge-base', branch, other_branch])
    if err:
        raise ValueError(err)
    return merge_base.decode('utf-8').strip()

def compare_to_remote(remote: str, local_branch: str, remote_branch: Optional[str]=None) -> List[FileDiff]:
    if False:
        i = 10
        return i + 15
    'Compare local with remote branch with git diff.\n\n    Parameter:\n        remote: str. Name of the git remote being pushed to.\n        local_branch: str. Name of the git branch being pushed to.\n        remote_branch: str|None. The name of the branch on the remote\n            to test against. If None, the remote branch is considered\n            to be the same as the local branch.\n\n    Returns:\n        list(FileDiff). List of FileDiffs that are modified, changed,\n        renamed or added but not deleted.\n\n    Raises:\n        ValueError. Raise ValueError if git command fails.\n    '
    remote_branch = remote_branch if remote_branch else local_branch
    git_remote = '%s/%s' % (remote, remote_branch)
    start_subprocess_for_result(['git', 'pull', remote])
    return git_diff_name_status(get_merge_base(git_remote, local_branch), local_branch)

def extract_files_to_lint(file_diffs: List[FileDiff]) -> List[bytes]:
    if False:
        while True:
            i = 10
    'Grab only files out of a list of FileDiffs that have a ACMRT status.'
    if not file_diffs:
        return []
    lint_files = [f.name for f in file_diffs if f.status in b'ACMRT']
    return lint_files

def get_parent_branch_name_for_diff() -> str:
    if False:
        print('Hello World!')
    'Returns remote branch name against which the diff has to be checked.\n\n    Returns:\n        str. The name of the remote branch.\n    '
    if common.is_current_branch_a_hotfix_branch():
        return 'release-%s' % common.get_current_release_version_number(common.get_current_branch_name())
    return 'develop'

def collect_files_being_pushed(ref_list: List[GitRef], remote: str) -> Dict[str, Tuple[List[FileDiff], List[bytes]]]:
    if False:
        return 10
    'Collect modified files and filter those that need linting.\n\n    Parameter:\n        ref_list: list of references to parse (provided by git in stdin)\n        remote: str. The name of the remote being pushed to.\n\n    Returns:\n        dict. Dict mapping branch names to 2-tuples of the form (list of\n        changed files, list of files to lint).\n    '
    if not ref_list:
        return {}
    ref_heads_only = [ref for ref in ref_list if ref.local_ref.startswith('refs/heads/') or ref.local_ref == 'HEAD']
    branches = [ref.local_ref.split('/')[-1] for ref in ref_heads_only]
    hashes = [ref.local_sha1 for ref in ref_heads_only]
    collected_files = {}
    for (branch, _) in zip(branches, hashes):
        modified_files = compare_to_remote(remote, branch, remote_branch=get_parent_branch_name_for_diff())
        files_to_lint = extract_files_to_lint(modified_files)
        collected_files[branch] = (modified_files, files_to_lint)
    for (branch, (modified_files, files_to_lint)) in collected_files.items():
        if modified_files:
            print('\nModified files in %s:' % branch)
            pprint.pprint(modified_files)
            print('\nFiles to lint in %s:' % branch)
            pprint.pprint(files_to_lint)
            print('\n')
    return collected_files

def get_refs() -> List[GitRef]:
    if False:
        while True:
            i = 10
    'Returns the ref list taken from STDIN.'
    ref_list = [GitRef(*ref_str.split()) for ref_str in sys.stdin]
    if ref_list:
        print('ref_list:')
        pprint.pprint(ref_list)
    return ref_list

def start_linter(files: List[bytes]) -> int:
    if False:
        print('Hello World!')
    'Starts the lint checks and returns the returncode of the task.'
    cmd_list: List[str] = [PYTHON_CMD, '-m', LINTER_MODULE, LINTER_FILE_FLAG]
    for file in files:
        cmd_list.append(file.decode('utf-8'))
    task = subprocess.Popen(cmd_list)
    task.communicate()
    return task.returncode

def execute_mypy_checks() -> int:
    if False:
        print('Hello World!')
    'Executes the mypy type checks.\n\n    Returns:\n        int. The return code from mypy checks.\n    '
    task = subprocess.Popen([PYTHON_CMD, '-m', MYPY_TYPE_CHECK_MODULE, '--skip-install'])
    task.communicate()
    return task.returncode

def run_script_and_get_returncode(cmd_list: List[str]) -> int:
    if False:
        print('Hello World!')
    'Runs script and returns the returncode of the task.\n\n    Args:\n        cmd_list: list(str). The cmd list containing the command to be run.\n\n    Returns:\n        int. The return code from the task executed.\n    '
    task = subprocess.Popen(cmd_list)
    task.communicate()
    task.wait()
    return task.returncode

def has_uncommitted_files() -> bool:
    if False:
        return 10
    'Returns true if the repo contains modified files that are uncommitted.\n    Ignores untracked files.\n    '
    uncommitted_files = subprocess.check_output(GIT_IS_DIRTY_CMD.split(' '), encoding='utf-8')
    return bool(len(uncommitted_files))

def install_hook() -> None:
    if False:
        for i in range(10):
            print('nop')
    'Installs the pre_push_hook script and makes it executable.\n    It ensures that oppia/ is the root folder.\n\n    Raises:\n        ValueError. Raise ValueError if chmod command fails.\n    '
    oppia_dir = os.getcwd()
    hooks_dir = os.path.join(oppia_dir, '.git', 'hooks')
    pre_push_file = os.path.join(hooks_dir, 'pre-push')
    chmod_cmd = ['chmod', '+x', pre_push_file]
    file_is_symlink = os.path.islink(pre_push_file)
    file_exists = os.path.exists(pre_push_file)
    if file_is_symlink and file_exists:
        print('Symlink already exists')
    else:
        if file_is_symlink and (not file_exists):
            os.unlink(pre_push_file)
            print('Removing broken symlink')
        try:
            os.symlink(os.path.abspath(__file__), pre_push_file)
            print('Created symlink in .git/hooks directory')
        except (OSError, AttributeError):
            shutil.copy(__file__, pre_push_file)
            print('Copied file to .git/hooks directory')
    print('Making pre-push hook file executable ...')
    (_, err_chmod_cmd) = start_subprocess_for_result(chmod_cmd)
    if not err_chmod_cmd:
        print('pre-push hook file is now executable!')
    else:
        raise ValueError(err_chmod_cmd)

def does_diff_include_js_or_ts_files(diff_files: List[bytes]) -> bool:
    if False:
        return 10
    'Returns true if diff includes JavaScript or TypeScript files.\n\n    Args:\n        diff_files: list(bytes). List of files changed.\n\n    Returns:\n        bool. Whether the diff contains changes in any JavaScript or TypeScript\n        files.\n    '
    for file_path in diff_files:
        if file_path.endswith(b'.ts') or file_path.endswith(b'.js'):
            return True
    return False

def does_diff_include_ts_files(diff_files: List[bytes]) -> bool:
    if False:
        print('Hello World!')
    'Returns true if diff includes TypeScript files.\n\n    Args:\n        diff_files: list(bytes). List of files changed.\n\n    Returns:\n        bool. Whether the diff contains changes in any TypeScript files.\n    '
    for file_path in diff_files:
        if file_path.endswith(b'.ts'):
            return True
    return False

def does_diff_include_ci_config_or_js_files(diff_files: List[bytes]) -> bool:
    if False:
        return 10
    'Returns true if diff includes CI config or Javascript files.\n\n    Args:\n        diff_files: list(bytes). List of files changed.\n\n    Returns:\n        bool. Whether the diff contains changes in CI config or\n        Javascript files.\n    '
    for file_path in diff_files:
        if file_path.endswith(b'.js') or re.search(b'e2e_.*\\.yml', file_path):
            return True
    return False

def check_for_backend_python_library_inconsistencies() -> None:
    if False:
        return 10
    "Checks the state of the 'third_party/python_libs' folder and compares it\n    to the required libraries specified in 'requirements.txt'.\n    If any inconsistencies are found, the script displays the inconsistencies\n    and exits.\n    "
    mismatches = install_python_prod_dependencies.get_mismatches()
    if mismatches:
        print('Your currently installed python libraries do not match the\nlibraries listed in your "requirements.txt" file. Here is a\nfull list of library/version discrepancies:\n')
        print('{:<35} |{:<25}|{:<25}'.format('Library', 'Requirements Version', 'Currently Installed Version'))
        for (library_name, version_strings) in mismatches.items():
            print('{!s:<35} |{!s:<25}|{!s:<25}'.format(library_name, version_strings[0], version_strings[1]))
        print('\n')
        common.print_each_string_after_two_new_lines(['Please fix these discrepancies by editing the `requirements.in`\nfile or running `scripts.install_third_party` to regenerate\nthe `third_party/python_libs` directory.\n'])
        sys.exit(1)
    else:
        print('Python dependencies consistency check succeeded.')

def main(args: Optional[List[str]]=None) -> None:
    if False:
        print('Hello World!')
    'Main method for pre-push hook that executes the Python/JS linters on all\n    files that deviate from develop.\n    '
    parser = argparse.ArgumentParser()
    parser.add_argument('remote', nargs='?', help='provided by git before push')
    parser.add_argument('url', nargs='?', help='provided by git before push')
    parser.add_argument('--install', action='store_true', default=False, help='Install pre_push_hook to the .git/hooks dir')
    parsed_args = parser.parse_args(args=args)
    if parsed_args.install:
        install_hook()
        return
    remote = get_remote_name()
    remote = remote if remote else parsed_args.remote
    refs = get_refs()
    collected_files = collect_files_being_pushed(refs, remote.decode('utf-8'))
    if collected_files and has_uncommitted_files():
        print('Your repo is in a dirty state which prevents the linting from working.\nStash your changes or commit them.\n')
        sys.exit(1)
    check_for_backend_python_library_inconsistencies()
    for (branch, (modified_files, files_to_lint)) in collected_files.items():
        with ChangedBranch(branch):
            if not modified_files and (not files_to_lint):
                continue
            if files_to_lint:
                lint_status = start_linter(files_to_lint)
                if lint_status != 0:
                    print('Push failed, please correct the linting issues above.')
                    sys.exit(1)
            mypy_check_status = execute_mypy_checks()
            if mypy_check_status != 0:
                print('Push failed, please correct the mypy type annotation issues above.')
                sys.exit(mypy_check_status)
            backend_associated_test_file_check_status = run_script_and_get_returncode(BACKEND_ASSOCIATED_TEST_FILE_CHECK_CMD)
            if backend_associated_test_file_check_status != 0:
                print('Push failed due to some backend files lacking an associated test file.')
                sys.exit(1)
            typescript_checks_status = 0
            if does_diff_include_ts_files(files_to_lint):
                typescript_checks_status = run_script_and_get_returncode(TYPESCRIPT_CHECKS_CMDS)
            if typescript_checks_status != 0:
                print('Push aborted due to failing typescript checks.')
                sys.exit(1)
            strict_typescript_checks_status = 0
            if does_diff_include_ts_files(files_to_lint):
                strict_typescript_checks_status = run_script_and_get_returncode(STRICT_TYPESCRIPT_CHECKS_CMDS)
            if strict_typescript_checks_status != 0:
                print('Push aborted due to failing typescript checks in strict mode.')
                sys.exit(1)
            frontend_status = 0
            ci_check_status = 0
            if does_diff_include_js_or_ts_files(files_to_lint):
                frontend_status = run_script_and_get_returncode(FRONTEND_TEST_CMDS)
            if frontend_status != 0:
                print('Push aborted due to failing frontend tests.')
                sys.exit(1)
            if does_diff_include_ci_config_or_js_files(files_to_lint):
                ci_check_status = run_script_and_get_returncode(CI_PROTRACTOR_CHECK_CMDS)
            if ci_check_status != 0:
                print('Push aborted due to failing e2e test configuration check.')
                sys.exit(1)
    return
if __name__ == '__main__':
    main()