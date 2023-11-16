import shutil
import pytest
from git.exc import HookExecutionError
from nbformat.v4.nbbase import new_markdown_cell
from pre_commit.main import main as pre_commit
from jupytext import read, write
from jupytext.cli import jupytext
from .utils import skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_sync(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev, python_notebook):
    if False:
        for i in range(10):
            print('nop')
    pre_commit_config_yaml = f'\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--sync]\n'
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks', '-f'])
    nb = python_notebook
    write(nb, 'test.ipynb')
    jupytext(['--set-formats', 'ipynb,py:percent', 'test.ipynb'])
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='git add test.py'):
        tmp_repo.index.commit('failing')
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('passing')
    assert 'test.ipynb' in tmp_repo.tree()
    assert 'test.py' in tmp_repo.tree()
    nb = read('test.ipynb')
    nb.cells.append(new_markdown_cell('A new cell'))
    write(nb, 'test.ipynb')
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='files were modified by this hook'):
        tmp_repo.index.commit('failing')
    assert 'A new cell' in tmpdir.join('test.py').read()
    with pytest.raises(HookExecutionError, match='git add test.py'):
        tmp_repo.index.commit('still failing')
    nb = read('test.ipynb')
    assert len(nb.cells) == 2
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('passing')
    nb.cells.append(new_markdown_cell('A third cell'))
    write(nb, 'test.py', fmt='py:percent')
    tmp_repo.git.add('test.py')
    with pytest.raises(HookExecutionError, match='git add test.ipynb'):
        tmp_repo.index.commit('failing')
    tmp_repo.git.add('test.ipynb')
    tmp_repo.index.commit('passing')
    nb = read('test.ipynb')
    assert len(nb.cells) == 3
    tmpdir.mkdir('subfolder')
    shutil.move('test.py', 'subfolder')
    shutil.move('test.ipynb', 'subfolder')
    tmp_repo.git.add('subfolder/test.ipynb')
    tmp_repo.git.add('subfolder/test.py')
    tmp_repo.index.commit('passing')
    status = pre_commit(['run', '--all'])
    assert status == 0