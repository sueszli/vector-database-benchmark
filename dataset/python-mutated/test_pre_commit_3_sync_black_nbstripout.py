import pytest
from git.exc import HookExecutionError
from pre_commit.main import main as pre_commit
from jupytext import read, write
from .utils import skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_sync_black_nbstripout(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev, notebook_with_outputs):
    if False:
        for i in range(10):
            print('nop')
    'Here we sync the ipynb notebook with a py:percent file and also apply black and nbstripout.'
    pre_commit_config_yaml = f'\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--sync, --pipe, black]\n    additional_dependencies:\n    - black==22.3.0  # Matches hook\n\n- repo: https://github.com/psf/black\n  rev: 22.3.0\n  hooks:\n  - id: black\n\n- repo: https://github.com/kynan/nbstripout\n  rev: 0.5.0\n  hooks:\n  - id: nbstripout\n'
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks', '-f'])
    tmpdir.join('jupytext.toml').write('formats = "ipynb,py:percent"')
    tmp_repo.git.add('jupytext.toml')
    tmp_repo.index.commit('pair notebooks')
    write(notebook_with_outputs, 'test.ipynb')
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='files were modified by this hook'):
        tmp_repo.index.commit('failing')
    tmp_repo.git.add('test.ipynb')
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('passing')
    assert 'test.ipynb' in tmp_repo.tree()
    assert 'test.py' in tmp_repo.tree()
    nb = read('test.ipynb')
    assert nb.cells[0].source == '1 + 1'
    assert not nb.cells[0].outputs