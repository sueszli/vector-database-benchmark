import pytest
from git.exc import HookExecutionError
from nbformat.v4.nbbase import new_markdown_cell, new_notebook
from pre_commit.main import main as pre_commit
from jupytext import read, write
from jupytext.cli import jupytext
from jupytext.compare import compare_cells
from .utils import skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_ipynb_to_py(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev):
    if False:
        print('Hello World!')
    'Here we document and test the expected behavior of the pre-commit hook in the\n    directional (--to) mode. Note that here, the ipynb file is always the source for\n    updates - i.e. changes on the .py file will not trigger the hook.\n    '
    pre_commit_config_yaml = f'\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--from, ipynb, --to, "py:percent"]\n'
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks'])
    nb = new_notebook(cells=[new_markdown_cell('A short notebook')])
    write(nb, 'test.ipynb')
    jupytext(['--from', 'ipynb', '--to', 'py:percent', 'test.ipynb'])
    tmp_repo.git.add('.')
    tmp_repo.index.commit('test')
    nb = new_notebook(cells=[new_markdown_cell('Some other text')])
    write(nb, 'test.ipynb')
    tmp_repo.git.add('test.ipynb')
    with pytest.raises(HookExecutionError, match='files were modified by this hook'):
        tmp_repo.index.commit('fails')
    with pytest.raises(HookExecutionError, match='git add test.py'):
        tmp_repo.index.commit('fails again')
    tmp_repo.git.add('test.py')
    tmp_repo.index.commit('succeeds')
    assert 'test.ipynb' in tmp_repo.tree()
    assert 'test.py' in tmp_repo.tree()
    nb = new_notebook(cells=[new_markdown_cell('Some updated text')])
    write(nb, 'test.py', fmt='py:percent')
    tmp_repo.index.commit('update py version')
    nb = read('test.ipynb')
    compare_cells(nb.cells, [new_markdown_cell('Some other text')], compare_ids=False)