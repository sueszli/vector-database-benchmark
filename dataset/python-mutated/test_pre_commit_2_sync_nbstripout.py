import pytest
from git.exc import HookExecutionError
from pre_commit.main import main as pre_commit
from jupytext import read, write
from jupytext.cli import jupytext
from .utils import skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_sync_nbstripout(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev, notebook_with_outputs):
    if False:
        print('Hello World!')
    'Here we sync the ipynb notebook with a Markdown file and also apply nbstripout.'
    pre_commit_config_yaml = f'\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--sync]\n\n- repo: https://github.com/kynan/nbstripout\n  rev: 0.5.0\n  hooks:\n  - id: nbstripout\n'
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks', '-f'])
    write(notebook_with_outputs, 'test.ipynb')
    jupytext(['--set-formats', 'ipynb,md', 'test.ipynb'])
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='files were modified by this hook'):
        tmp_repo.index.commit('failing')
    tmp_repo.git.add('test.ipynb')
    tmp_repo.git.add('test.md')
    tmp_repo.index.commit('passing')
    assert 'test.ipynb' in tmp_repo.tree()
    assert 'test.md' in tmp_repo.tree()
    nb = read('test.ipynb')
    assert not nb.cells[0].outputs