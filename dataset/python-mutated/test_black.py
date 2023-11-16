import os
from copy import deepcopy
from shutil import copyfile
import pytest
from nbformat.v4.nbbase import new_code_cell, new_notebook
from jupytext import read, write
from jupytext.cli import jupytext, pipe_notebook, system
from jupytext.combine import black_invariant
from jupytext.compare import compare, compare_cells, compare_notebooks
from jupytext.header import _DEFAULT_NOTEBOOK_METADATA
from .utils import list_notebooks, requires_autopep8, requires_black, requires_flake8

@requires_black
@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_apply_black_on_python_notebooks(tmpdir, cwd_tmpdir, nb_file):
    if False:
        print('Hello World!')
    copyfile(nb_file, 'notebook.ipynb')
    jupytext(args=['notebook.ipynb', '--to', 'py:percent'])
    system('black', 'notebook.py')
    jupytext(args=['notebook.py', '--to', 'ipynb', '--update'])
    nb1 = read(nb_file)
    nb2 = read('notebook.ipynb')
    nb3 = read('notebook.py')
    assert len(nb1.cells) == len(nb2.cells)
    assert len(nb1.cells) == len(nb3.cells)
    for (c1, c2) in zip(nb1.cells, nb2.cells):
        assert black_invariant(c1.source) == black_invariant(c2.source)
        assert 'lines_to_next_cell' not in c2.metadata
        assert c1.cell_type == c2.cell_type
        if c1.cell_type == 'code':
            compare(c1.outputs, c2.outputs)
    compare(nb1.metadata, nb2.metadata)

def test_black_invariant():
    if False:
        print('Hello World!')
    text_org = 'long_string = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa" \\\n              "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"\n'
    text_black = 'long_string = (\n    "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"\n    "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"\n)\n'
    assert black_invariant(text_org) == black_invariant(text_black)

@requires_black
def test_pipe_into_black():
    if False:
        print('Hello World!')
    nb_org = new_notebook(cells=[new_code_cell('1        +1', id='cell-id')])
    nb_dest = new_notebook(cells=[new_code_cell('1 + 1', id='cell-id')])
    nb_pipe = pipe_notebook(nb_org, 'black')
    compare_notebooks(nb_pipe, nb_dest, allow_expected_differences=False, compare_ids=True)

@requires_autopep8
def test_pipe_into_autopep8():
    if False:
        for i in range(10):
            print('nop')
    nb_org = new_notebook(cells=[new_code_cell('1        +1', id='cell-id')])
    nb_dest = new_notebook(cells=[new_code_cell('1 + 1', id='cell-id')])
    nb_pipe = pipe_notebook(nb_org, 'autopep8 -')
    compare_notebooks(nb_pipe, nb_dest, allow_expected_differences=False, compare_ids=True)

@requires_flake8
def test_pipe_into_flake8():
    if False:
        i = 10
        return i + 15
    nb = new_notebook(cells=[new_code_cell('# correct code\n1 + 1')])
    pipe_notebook(nb, 'flake8', update=False)
    nb = new_notebook(cells=[new_code_cell('incorrect code')])
    with pytest.raises(SystemExit):
        pipe_notebook(nb, 'flake8', update=False)

@requires_black
@requires_flake8
@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_apply_black_through_jupytext(tmpdir, nb_file):
    if False:
        print('Hello World!')
    metadata = read(nb_file).metadata
    nb_org = new_notebook(cells=[new_code_cell('1        +1', id='cell-id')], metadata=metadata)
    nb_black = new_notebook(cells=[new_code_cell('1 + 1', id='cell-id')], metadata=metadata)
    tmp_ipynb = str(tmpdir.mkdir('notebook_folder').join('notebook.ipynb'))
    tmp_py = str(tmpdir.mkdir('script_folder').join('notebook.py'))
    write(nb_org, tmp_ipynb)
    jupytext([tmp_ipynb, '--pipe', 'black'])
    nb_now = read(tmp_ipynb)
    compare_notebooks(nb_now, nb_black, compare_ids=True)
    write(nb_org, tmp_ipynb)
    jupytext([tmp_ipynb, '--to', '../script_folder//py:percent', '--pipe', 'black'])
    assert os.path.isfile(tmp_py)
    nb_now = read(tmp_py)
    nb_now.metadata = metadata
    compare_notebooks(nb_now, nb_black)
    os.remove(tmp_py)
    write(nb_org, tmp_ipynb)
    jupytext([tmp_ipynb, '--from', 'notebook_folder//ipynb', '--to', 'script_folder//py:percent', '--pipe', 'black', '--check', 'flake8'])
    assert os.path.isfile(tmp_py)
    nb_now = read(tmp_py)
    nb_now.metadata = metadata
    compare_notebooks(nb_now, nb_black)

@requires_black
@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_apply_black_and_sync_on_paired_notebook(tmpdir, cwd_tmpdir, nb_file):
    if False:
        i = 10
        return i + 15
    metadata = read(nb_file).metadata
    metadata['jupytext'] = {'formats': 'ipynb,py'}
    assert 'language_info' in metadata
    nb_org = new_notebook(cells=[new_code_cell('1        +1', id='cell-id')], metadata=metadata)
    nb_black = new_notebook(cells=[new_code_cell('1 + 1', id='cell-id')], metadata=metadata)
    write(nb_org, 'notebook.ipynb')
    jupytext(['notebook.ipynb', '--pipe', 'black', '--sync'])
    nb_now = read('notebook.ipynb')
    compare_notebooks(nb_now, nb_black, compare_ids=True)
    assert 'language_info' in nb_now.metadata
    nb_now = read('notebook.py')
    nb_now.metadata['jupytext'].pop('text_representation')
    nb_black.metadata = {key: nb_black.metadata[key] for key in nb_black.metadata if key in _DEFAULT_NOTEBOOK_METADATA.split(',')}
    compare_notebooks(nb_now, nb_black)

@requires_black
def test_apply_black_on_markdown_notebook(tmpdir):
    if False:
        print('Hello World!')
    text = '---\njupyter:\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n  language_info:\n    codemirror_mode:\n      name: ipython\n      version: 3\n    file_extension: .py\n    mimetype: text/x-python\n    name: python\n    nbconvert_exporter: python\n    pygments_lexer: ipython3\n    version: 3.7.4\n---\n\n```python\n1    +     2+3+4\n```\n'
    tmp_md = str(tmpdir.join('test.md'))
    with open(tmp_md, 'w') as fp:
        fp.write(text)
    jupytext([tmp_md, '--pipe', 'black'])
    nb = read(tmp_md)
    compare_cells(nb.cells, [new_code_cell('1 + 2 + 3 + 4')], compare_ids=False)

@requires_black
def test_black_through_tempfile(tmpdir, text='```python\n1 +    2 + 3\n```\n', black='```python\n1 + 2 + 3\n```\n'):
    if False:
        for i in range(10):
            print('nop')
    tmp_md = str(tmpdir.join('notebook.md'))
    with open(tmp_md, 'w') as fp:
        fp.write(text)
    jupytext([tmp_md, '--pipe', 'black {}'])
    with open(tmp_md) as fp:
        compare(fp.read(), black)

@requires_black
def test_pipe_black_removes_lines_to_next_cell_metadata(tmpdir, cwd_tmpdir, text='# %%\ndef func():\n    return 42\n# %%\nfunc()'):
    if False:
        while True:
            i = 10
    tmpdir.join('notebook.py').write(text)
    jupytext(['--set-formats', 'ipynb,py:percent', 'notebook.py'])
    nb = read(tmpdir.join('notebook.ipynb'))
    assert nb.cells[0].metadata['lines_to_next_cell'] == 0
    jupytext(['--sync', 'notebook.py', '--pipe', 'black'])
    nb = read(tmpdir.join('notebook.ipynb'))
    assert 'lines_to_next_cell' not in nb.cells[0].metadata
    new_text = tmpdir.join('notebook.py').read()
    assert '\n\n# %%\nfunc()' in new_text

@requires_black
@pytest.mark.parametrize('code,black_should_fail', [('myvar = %dont_format_me', False), ('incomplete_instruction = (...', True)])
def test_pipe_black_uses_warn_only_781(tmpdir, cwd_tmpdir, code, black_should_fail, python_notebook, capsys):
    if False:
        while True:
            i = 10
    nb = python_notebook
    nb.cells.append(new_code_cell(code))
    write(nb, 'notebook.ipynb')
    if not black_should_fail:
        jupytext(['--pipe', 'black', 'notebook.ipynb'])
        return
    with pytest.raises(SystemExit):
        jupytext(['--pipe', 'black', 'notebook.ipynb'])
    (out, err) = capsys.readouterr()
    assert "Error: The command 'black -' exited with code" in err
    assert '--warn-only' in err
    jupytext(['--pipe', 'black', 'notebook.ipynb', '--warn-only'])
    (out, err) = capsys.readouterr()
    assert "Warning: The command 'black -' exited with code" in err
    actual = read('notebook.ipynb')
    compare_notebooks(actual, nb)

@requires_black
def test_pipe_black_preserve_outputs(notebook_with_outputs, tmpdir, cwd_tmpdir, capsys):
    if False:
        i = 10
        return i + 15
    write(notebook_with_outputs, 'test.ipynb')
    jupytext(['--pipe', 'black', 'test.ipynb'])
    nb = read('test.ipynb')
    expected = deepcopy(notebook_with_outputs)
    expected.cells[0].source = '1 + 1'
    compare_notebooks(nb, expected)
    (out, err) = capsys.readouterr()
    assert not err
    assert 'replaced' not in out
    assert '--update' not in out