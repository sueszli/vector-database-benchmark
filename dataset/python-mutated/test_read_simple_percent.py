import os
import pytest
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook, new_raw_cell
import jupytext
from jupytext.compare import compare, compare_notebooks
from .utils import notebook_model

def test_read_simple_file(script='# ---\n# title: Simple file\n# ---\n\n# %% [markdown]\n# This is a markdown cell\n\n# %% [md]\n# This is also a markdown cell\n\n# %% [raw]\n# This is a raw cell\n\n# %%% sub-cell title\n# This is a sub-cell\n\n# %%%% sub-sub-cell title\n# This is a sub-sub-cell\n\n# %% And now a code cell\n1 + 2 + 3 + 4\n5\n6\n# %%magic # this is a commented magic, not a cell\n\n7\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py:percent')
    compare_notebooks(new_notebook(cells=[new_raw_cell('---\ntitle: Simple file\n---'), new_markdown_cell('This is a markdown cell'), new_markdown_cell('This is also a markdown cell', metadata={'region_name': 'md'}), new_raw_cell('This is a raw cell'), new_code_cell('# This is a sub-cell', metadata={'title': 'sub-cell title', 'cell_depth': 1}), new_code_cell('# This is a sub-sub-cell', metadata={'title': 'sub-sub-cell title', 'cell_depth': 2}), new_code_cell('1 + 2 + 3 + 4\n5\n6\n%%magic # this is a commented magic, not a cell\n\n7', metadata={'title': 'And now a code cell'})]), nb)
    script2 = jupytext.writes(nb, 'py:percent')
    compare(script2, script)

def test_read_cell_with_metadata(script='# %% a code cell with parameters {"tags": ["parameters"]}\na = 3\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(script, 'py:percent')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == 'a = 3'
    assert nb.cells[0].metadata == {'title': 'a code cell with parameters', 'tags': ['parameters']}
    script2 = jupytext.writes(nb, 'py:percent')
    compare(script2, script)

def test_read_nbconvert_script(script='\n# coding: utf-8\n\n# A markdown cell\n\n# In[1]:\n\n\nimport pandas as pd\n\npd.options.display.max_rows = 6\npd.options.display.max_columns = 20\n\n\n# Another markdown cell\n\n# In[2]:\n\n\n1 + 1\n\n\n# Again, a markdown cell\n\n# In[33]:\n\n\n2 + 2\n\n\n# <codecell>\n\n\n3 + 3\n'):
    if False:
        print('Hello World!')
    assert jupytext.formats.guess_format(script, '.py')[0] == 'percent'
    nb = jupytext.reads(script, '.py')
    assert len(nb.cells) == 5

def test_read_remove_blank_lines(script='# %%\nimport pandas as pd\n\n# %% Display a data frame\ndf = pd.DataFrame({\'A\': [1, 2], \'B\': [3, 4]},\n                  index=pd.Index([\'x0\', \'x1\'], name=\'x\'))\ndf\n\n# %% Pandas plot {"tags": ["parameters"]}\ndf.plot(kind=\'bar\')\n\n\n# %% sample class\nclass MyClass:\n    pass\n\n\n# %% a function\ndef f(x):\n    return 42 * x\n\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py')
    assert len(nb.cells) == 5
    for i in range(5):
        assert nb.cells[i].cell_type == 'code'
        assert not nb.cells[i].source.startswith('\n')
        assert not nb.cells[i].source.endswith('\n')
    script2 = jupytext.writes(nb, 'py:percent')
    compare(script2, script)

def test_no_crash_on_square_bracket(script="# %% In [2]\nprint('Hello')\n"):
    if False:
        print('Hello World!')
    nb = jupytext.reads(script, 'py')
    script2 = jupytext.writes(nb, 'py:percent')
    compare(script2, script)

def test_nbconvert_cell(script="# In[2]:\nprint('Hello')\n"):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(script, 'py')
    script2 = jupytext.writes(nb, 'py:percent')
    expected = "# %%\nprint('Hello')\n"
    compare(script2, expected)

def test_nbformat_v3_nbpy_cell(script="# <codecell>\nprint('Hello')\n"):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py')
    script2 = jupytext.writes(nb, 'py:percent')
    expected = "# %%\nprint('Hello')\n"
    compare(script2, expected)

def test_multiple_empty_cells():
    if False:
        i = 10
        return i + 15
    nb = new_notebook(cells=[new_code_cell(), new_code_cell(), new_code_cell()], metadata={'jupytext': {'notebook_metadata_filter': '-all'}})
    text = jupytext.writes(nb, 'py:percent')
    expected = '# %%\n\n# %%\n\n# %%\n'
    compare(text, expected)
    nb2 = jupytext.reads(text, 'py:percent')
    nb2.metadata = nb.metadata
    compare_notebooks(nb2, nb)

def test_first_cell_markdown_191():
    if False:
        print('Hello World!')
    text = '# %% [markdown]\n# Docstring\n\n# %%\nfrom math import pi\n\n# %% [markdown]\n# Another markdown cell\n'
    nb = jupytext.reads(text, 'py')
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[2].cell_type == 'markdown'

def test_multiline_comments_in_markdown_1():
    if False:
        while True:
            i = 10
    text = "# %% [markdown]\n'''\na\nlong\ncell\n'''\n"
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'a\nlong\ncell'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_multiline_comments_in_markdown_2():
    if False:
        return 10
    text = '# %% [markdown]\n"""\na\nlong\ncell\n"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'a\nlong\ncell'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_multiline_comments_format_option():
    if False:
        for i in range(10):
            print('nop')
    text = '# %% [markdown]\n"""\na\nlong\ncell\n"""\n'
    nb = new_notebook(cells=[new_markdown_cell('a\nlong\ncell')], metadata={'jupytext': {'cell_markers': '"""', 'notebook_metadata_filter': '-all'}})
    py = jupytext.writes(nb, 'py:percent')
    compare(py, text)

def test_multiline_comments_in_raw_cell():
    if False:
        i = 10
        return i + 15
    text = '# %% [raw]\n"""\nsome\ntext\n"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == 'some\ntext'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_multiline_comments_in_markdown_cell_no_line_return():
    if False:
        i = 10
        return i + 15
    text = '# %% [markdown]\n"""a\nlong\ncell"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'a\nlong\ncell'

def test_multiline_comments_in_markdown_cell_is_robust_to_additional_cell_marker():
    if False:
        return 10
    text = '# %% [markdown]\n"""\nsome text, and a fake cell marker\n# %% [raw]\n"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'some text, and a fake cell marker\n# %% [raw]'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_cell_markers_option_in_contents_manager(tmpdir):
    if False:
        while True:
            i = 10
    tmp_ipynb = tmpdir / 'notebook.ipynb'
    tmp_py = tmpdir / 'notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = new_notebook(cells=[new_code_cell('1 + 1'), new_markdown_cell('a\nlong\ncell')], metadata={'jupytext': {'formats': 'ipynb,py:percent', 'notebook_metadata_filter': '-all', 'cell_markers': "'''"}})
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    with open(tmp_py) as fp:
        text = fp.read()
    compare(text, "# %%\n1 + 1\n\n# %% [markdown]\n'''\na\nlong\ncell\n'''\n")
    nb2 = jupytext.read(tmp_py)
    compare_notebooks(nb, nb2)

def test_cell_markers_in_config(tmpdir, python_notebook):
    if False:
        while True:
            i = 10
    (tmpdir / 'jupytext.toml').write('cell_markers = \'"""\'\n')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = python_notebook
    nb.metadata['jupytext'] = {'formats': 'ipynb,py:percent'}
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    text = (tmpdir / 'notebook.py').read()
    assert '# %% [markdown]\n"""\nA short notebook\n"""\n' in text
    nb2 = jupytext.read(tmpdir / 'notebook.py')
    compare_notebooks(nb, nb2)

def test_cell_markers_in_contents_manager(tmpdir):
    if False:
        print('Hello World!')
    tmp_ipynb = tmpdir / 'notebook.ipynb'
    tmp_py = tmpdir / 'notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.cell_markers = "'''"
    nb = new_notebook(cells=[new_code_cell('1 + 1'), new_markdown_cell('a\nlong\ncell')], metadata={'jupytext': {'formats': 'ipynb,py:percent', 'notebook_metadata_filter': '-all'}})
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    with open(tmp_py) as fp:
        text = fp.read()
    compare(text, "# %%\n1 + 1\n\n# %% [markdown]\n'''\na\nlong\ncell\n'''\n")
    nb2 = jupytext.read(tmp_py)
    compare_notebooks(nb, nb2)

def test_cell_markers_in_contents_manager_does_not_impact_light_format(tmpdir):
    if False:
        print('Hello World!')
    tmp_ipynb = tmpdir / 'notebook.ipynb'
    tmp_py = tmpdir / 'notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.cell_markers = "'''"
    nb = new_notebook(cells=[new_code_cell('1 + 1'), new_markdown_cell('a\nlong\ncell')], metadata={'jupytext': {'formats': 'ipynb,py', 'notebook_metadata_filter': '-all'}})
    with pytest.warns(UserWarning, match='Ignored cell markers'):
        cm.save(model=notebook_model(nb), path='notebook.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    with open(tmp_py) as fp:
        text = fp.read()
    compare(text, '1 + 1\n\n# a\n# long\n# cell\n')
    nb2 = jupytext.read(tmp_py)
    compare_notebooks(nb, nb2)

def test_single_triple_quote_works(no_jupytext_version_number, text='# ---\n# jupyter:\n#   jupytext:\n#     cell_markers: \'"""\'\n#     formats: ipynb,py:percent\n#     text_representation:\n#       extension: .py\n#       format_name: percent\n# ---\n\n# %%\nprint("hello")\n', notebook=new_notebook(cells=[new_code_cell('print("hello")')])):
    if False:
        print('Hello World!')
    compare_notebooks(jupytext.reads(text, 'py'), notebook)

def test_docstring_with_quadruple_quote(nb=new_notebook(cells=[new_code_cell('def fun_1(df):\n  """"\n  docstring starting with 4 double quotes and ending with 3\n  """\n  return df'), new_code_cell('def fun_2(df):\n  """\n  docstring\n  """\n  return df')])):
    if False:
        i = 10
        return i + 15
    'Reproduces https://github.com/mwouts/jupytext/issues/460'
    py = jupytext.writes(nb, 'py:percent')
    nb2 = jupytext.reads(py, 'py')
    compare_notebooks(nb2, nb)

def test_cell_marker_has_same_indentation_as_code(text="# %%\nif __name__ == '__main__':\n    print(1)\n\n    # %%\n    # INDENTED COMMENT\n    print(2)\n", nb_expected=new_notebook(cells=[new_code_cell("if __name__ == '__main__':\n    print(1)"), new_code_cell('    # INDENTED COMMENT\n    print(2)')])):
    if False:
        i = 10
        return i + 15
    'The cell marker should have the same indentation as the first code line. See issue #562'
    nb_actual = jupytext.reads(text, fmt='py:percent')
    compare_notebooks(nb_actual, nb_expected)
    text_actual = jupytext.writes(nb_actual, fmt='py:percent')
    compare(text_actual, text)