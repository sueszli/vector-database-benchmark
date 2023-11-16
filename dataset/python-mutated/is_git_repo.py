import os
import sys
import pytest
from llnl.util.filesystem import mkdirp, working_dir
import spack
from spack.version import ver

@pytest.fixture(scope='function')
def git_tmp_worktree(git, tmpdir, mock_git_version_info):
    if False:
        return 10
    'Create new worktree in a temporary folder and monkeypatch\n    spack.paths.prefix to point to it.\n    '
    git_version = spack.fetch_strategy.GitFetchStrategy.version_from_git(git)
    if git_version < ver('2.17.0'):
        pytest.skip('git_tmp_worktree requires git v2.17.0')
    with working_dir(mock_git_version_info[0]):
        if sys.platform == 'win32':
            long_pth = str(tmpdir).split(os.path.sep)
            tmp_worktree = os.path.sep.join(long_pth[:-1])
        else:
            tmp_worktree = str(tmpdir)
        worktree_root = os.path.sep.join([tmp_worktree, 'wrktree'])
        mkdirp(worktree_root)
        git('worktree', 'add', '--detach', worktree_root, 'HEAD')
        yield worktree_root
        git('worktree', 'remove', '--force', worktree_root)

def test_is_git_repo_in_worktree(git_tmp_worktree):
    if False:
        for i in range(10):
            print('nop')
    'Verify that spack.cmd.spack_is_git_repo() can identify a git repository\n    in a worktree.\n    '
    assert spack.cmd.is_git_repo(git_tmp_worktree)

def test_spack_is_git_repo_nongit(tmpdir, monkeypatch):
    if False:
        print('Hello World!')
    'Verify that spack.cmd.spack_is_git_repo() correctly returns False if we\n    are in a non-git directory.\n    '
    assert not spack.cmd.is_git_repo(str(tmpdir))