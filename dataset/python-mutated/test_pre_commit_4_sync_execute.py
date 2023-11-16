import pytest
from git.exc import HookExecutionError
from nbformat.v4.nbbase import new_code_cell
from pre_commit.main import main as pre_commit
from jupytext import read, write
from .utils import requires_user_kernel_python3, skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@requires_user_kernel_python3
@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_sync_execute(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev, notebook_with_outputs):
    if False:
        return 10
    'Here we sync the ipynb notebook with a py:percent file and execute it (this is probably not a very\n    recommendable hook!)'
    pre_commit_config_yaml = f'\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--sync, --execute, --show-changes]\n    additional_dependencies:\n    - nbconvert\n'
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks', '-f'])
    nb = notebook_with_outputs
    nb.cells = [new_code_cell('3+4')]
    nb.metadata['jupytext'] = {'formats': 'ipynb,py:percent'}
    write(nb, 'test.ipynb')
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='files were modified by this hook'):
        tmp_repo.index.commit('failing')
    tmp_repo.git.add('test.ipynb')
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('passing')
    assert 'test.ipynb' in tmp_repo.tree()
    assert 'test.py' in tmp_repo.tree()
    nb = read('test.ipynb')
    assert nb.cells[0].outputs[0]['data']['text/plain'] == '7'