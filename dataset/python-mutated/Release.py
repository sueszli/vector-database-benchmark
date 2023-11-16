""" Release related common functionality.

"""
import os
from nuitka.tools.Basics import getHomePath
from nuitka.utils.Execution import NuitkaCalledProcessError, check_output
from nuitka.utils.FileOperations import getFileContents, getFileFirstLine, openTextFile, withDirectoryChange
from nuitka.Version import getNuitkaVersion

def checkAtHome(expected='Nuitka Staging'):
    if False:
        while True:
            i = 10
    assert os.path.isfile('setup.py')
    if os.path.isdir('.git'):
        git_dir = '.git'
    else:
        line = getFileFirstLine('.git', 'r').strip()
        git_dir = line[8:]
    git_description_filename = os.path.join(git_dir, 'description')
    description = getFileContents(git_description_filename).strip()
    assert description == expected, (expected, description)

def _getGitCommandOutput(command):
    if False:
        i = 10
        return i + 15
    if type(command) is str:
        command = command.split()
    home_path = getHomePath()
    with withDirectoryChange(home_path):
        output = check_output(command).strip()
    if str is not bytes:
        output = output.decode()
    return output

def getBranchName():
    if False:
        while True:
            i = 10
    'Get the git branch name currently running from.'
    try:
        return _getGitCommandOutput('git branch --show-current')
    except NuitkaCalledProcessError:
        return _getGitCommandOutput('git symbolic-ref --short HEAD')

def getBranchRemoteName():
    if False:
        while True:
            i = 10
    'Get the git remote name of the branch currently running from.'
    return _getGitCommandOutput('git config branch.%s.remote' % getBranchName())

def getBranchRemoteUrl():
    if False:
        for i in range(10):
            print('nop')
    'Get the git remote url of the branch currently running from.'
    return _getGitCommandOutput('git config remote.%s.url' % getBranchRemoteName())

def getBranchRemoteIdentifier():
    if False:
        i = 10
        return i + 15
    'Get the git remote identifier of the branch currently running from.\n\n    This identifier is used to classify git origins, they might be github,\n    private git, or unknown.\n    '
    branch_remote_url = getBranchRemoteUrl()
    branch_remote_host = branch_remote_url.split(':', 1)[0].split('@')[-1]
    if branch_remote_host.endswith('.home'):
        branch_remote_host = branch_remote_host.rsplit('.', 1)[0]
    if branch_remote_host == 'mastermind':
        return 'private'
    elif branch_remote_host.endswith('nuitka.net'):
        return 'private'
    elif branch_remote_host == 'github':
        return 'public'
    else:
        return 'unknown'

def checkBranchName():
    if False:
        return 10
    branch_name = getBranchName()
    nuitka_version = getNuitkaVersion()
    assert branch_name in ('main', 'develop', 'factory', 'release/' + nuitka_version, 'hotfix/' + nuitka_version), branch_name
    return branch_name

def getBranchCategory(branch_name):
    if False:
        print('Hello World!')
    'There are 3 categories of releases. Map branch name on them.'
    if branch_name.startswith('release') or branch_name == 'main' or branch_name.startswith('hotfix/'):
        category = 'stable'
    elif branch_name == 'factory':
        category = 'factory'
    elif branch_name == 'develop':
        category = 'develop'
    else:
        assert False
    return category

def checkNuitkaChangelog():
    if False:
        for i in range(10):
            print('nop')
    with openTextFile('Changelog.rst', 'r') as f:
        while True:
            line = f.readline().strip()
            if line.startswith('***') and line.endswith('***'):
                break
        line = f.readline()
    if '(Draft)' in line:
        return 'draft'
    else:
        return 'final'