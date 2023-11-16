import pytest
from git.exc import HookExecutionError
from nbformat.v4.nbbase import new_markdown_cell
from pre_commit.main import main as pre_commit
from jupytext import TextFileContentsManager, read
from .utils import skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_sync_with_config(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev, python_notebook):
    if False:
        print('Hello World!')
    pre_commit_config_yaml = f'\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--sync]\n'
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks', '-f'])
    tmpdir.join('jupytext.toml').write('formats = "ipynb,py:percent"\n')
    nb = python_notebook
    cm = TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.save(dict(type='notebook', content=nb), 'test.ipynb')
    assert 'text_representation' in tmpdir.join('test.py').read()
    assert 'text_representation' not in tmpdir.join('test.ipynb').read()
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='git add test.py'):
        tmp_repo.index.commit('failing')
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('passing')
    assert 'test.ipynb' in tmp_repo.tree()
    assert 'test.py' in tmp_repo.tree()
    assert 'text_representation' in tmpdir.join('test.py').read()
    assert 'text_representation' not in tmpdir.join('test.ipynb').read()
    nb = cm.get('test.ipynb')['content']
    nb.cells.append(new_markdown_cell('A new cell'))
    cm.save(dict(type='notebook', content=nb), 'test.ipynb')
    assert 'text_representation' in tmpdir.join('test.py').read()
    assert 'text_representation' not in tmpdir.join('test.ipynb').read()
    assert 'A new cell' in tmpdir.join('test.py').read()
    nb = read('test.ipynb')
    assert len(nb.cells) == 2
    tmp_repo.git.add('test.ipynb')
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('passing')