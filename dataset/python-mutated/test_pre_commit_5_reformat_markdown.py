import pytest
from git.exc import HookExecutionError
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook
from pre_commit.main import main as pre_commit
from jupytext import read, write
from .utils import requires_pandoc, skip_pre_commit_tests_on_windows, skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo

@requires_pandoc
@skip_pre_commit_tests_on_windows
@skip_pre_commit_tests_when_jupytext_folder_is_not_a_git_repo
def test_pre_commit_hook_sync_reformat_code_and_markdown(tmpdir, cwd_tmpdir, tmp_repo, jupytext_repo_root, jupytext_repo_rev, notebook_with_outputs):
    if False:
        while True:
            i = 10
    'Here we sync the ipynb notebook with a py:percent file and also apply black and pandoc to reformat both\n    code and markdown cells.\n\n    Note: the new cell ids introduced in nbformat 5.1.0 are not yet supported by all the programs that treat\n    ipynb files. Consequently we pin the version of nbformat to 5.0.8 in all the environments below (and you\n    will have to do the same on the environment in which you edit the notebooks).\n    '
    pre_commit_config_yaml = f"\nrepos:\n- repo: {jupytext_repo_root}\n  rev: {jupytext_repo_rev}\n  hooks:\n  - id: jupytext\n    args: [--sync, --pipe-fmt, ipynb, --pipe, 'pandoc --from ipynb --to ipynb --markdown-headings=atx', --show-changes]\n    additional_dependencies:\n    - nbformat==5.0.8  # because pandoc 2.11.4 does not preserve yet the new cell ids\n  - id: jupytext\n    args: [--sync, --pipe, black, --show-changes]\n    additional_dependencies:\n    - black==22.3.0  # Matches black hook below\n    - nbformat==5.0.8  # for compatibility with the pandoc hook above\n\n- repo: https://github.com/psf/black\n  rev: 22.3.0\n  hooks:\n  - id: black\n"
    tmpdir.join('.pre-commit-config.yaml').write(pre_commit_config_yaml)
    tmp_repo.git.add('.pre-commit-config.yaml')
    pre_commit(['install', '--install-hooks', '-f'])
    tmpdir.join('jupytext.toml').write('formats = "ipynb,py:percent"')
    tmp_repo.git.add('jupytext.toml')
    tmp_repo.index.commit('pair notebooks')
    notebook = new_notebook(cells=[new_code_cell('1+1'), new_markdown_cell('This is a complex markdown cell\n\n# With a h1 header\n## And a h2 header\n\n| And       | A  | Table |\n| --------- | ---| ----- |\n| 0         | 1  | 2     |\n\n!!!WARNING!!! This hook does not seem compatible with\nexplicit paragraph breaks (two spaces at the end of a line).\n\nAnd a VERY long line.\n'.replace('VERY ', 'very ' * 51))], metadata=notebook_with_outputs.metadata)
    notebook.nbformat = 4
    notebook.nbformat_minor = 4
    write(notebook, 'test.ipynb')
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
    print(nb.cells[1].source)
    assert nb.cells[1].source == 'This is a complex markdown cell\n\n# With a h1 header\n\n## And a h2 header\n\n| And | A   | Table |\n|-----|-----|-------|\n| 0   | 1   | 2     |\n\n!!!WARNING!!! This hook does not seem compatible with explicit paragraph\nbreaks (two spaces at the end of a line).\n\nAnd a very very very very very very very very very very very very very\nvery very very very very very very very very very very very very very\nvery very very very very very very very very very very very very very\nvery very very very very very very very very very long line.'