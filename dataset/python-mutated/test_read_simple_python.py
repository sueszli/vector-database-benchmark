import pytest
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook, new_raw_cell
import jupytext
from jupytext.compare import compare, compare_notebooks

def test_read_simple_file(pynb='# ---\n# title: Simple file\n# ---\n\n# Here we have some text\n# And below we have some python code\n\ndef f(x):\n    return x+1\n\n\ndef h(y):\n    return y-1\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 4
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: Simple file\n---'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[1].source == 'Here we have some text\nAnd below we have some python code'
    assert nb.cells[2].cell_type == 'code'
    compare(nb.cells[2].source, 'def f(x):\n    return x+1')
    assert nb.cells[3].cell_type == 'code'
    compare(nb.cells[3].source, 'def h(y):\n    return y-1')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_less_simple_file(pynb='# ---\n# title: Less simple file\n# ---\n\n# Here we have some text\n# And below we have some python code\n\n# This is a comment about function f\ndef f(x):\n    return x+1\n\n\n# And a comment on h\ndef h(y):\n    return y-1\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 4
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: Less simple file\n---'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[1].source == 'Here we have some text\nAnd below we have some python code'
    assert nb.cells[2].cell_type == 'code'
    compare(nb.cells[2].source, '# This is a comment about function f\ndef f(x):\n    return x+1')
    assert nb.cells[3].cell_type == 'code'
    compare(nb.cells[3].source, '# And a comment on h\ndef h(y):\n    return y-1')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_indented_comment(text='def f():\n    return 1\n\n    # f returns 1\n\n\ndef g():\n    return 2\n\n\n# h returns 3\ndef h():\n    return 3\n', ref=new_notebook(cells=[new_code_cell('def f():\n    return 1\n\n    # f returns 1'), new_code_cell('def g():\n    return 2'), new_code_cell('# h returns 3\ndef h():\n    return 3')])):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(text, 'py')
    compare_notebooks(nb, ref)
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_non_pep8(text='def f():\n    return 1\ndef g():\n    return 2\n\ndef h():\n    return 3\n', ref=new_notebook(cells=[new_code_cell('def f():\n    return 1\ndef g():\n    return 2', metadata={'lines_to_next_cell': 1}), new_code_cell('def h():\n    return 3')])):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(text, 'py')
    compare_notebooks(nb, ref)
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_read_non_pep8_file(pynb='# ---\n# title: Non-pep8 file\n# ---\n\n# This file is non-pep8 as the function below has\n# two consecutive blank lines in its body\n\ndef f(x):\n\n\n    return x+1\n'):
    if False:
        return 10
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: Non-pep8 file\n---'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[1].source == 'This file is non-pep8 as the function below has\ntwo consecutive blank lines in its body'
    assert nb.cells[2].cell_type == 'code'
    compare(nb.cells[2].source, 'def f(x):\n\n\n    return x+1')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_cell_two_blank_lines(pynb='# ---\n# title: cell with two consecutive blank lines\n# ---\n\n# +\na = 1\n\n\na + 2\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: cell with two consecutive blank lines\n---'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'a = 1\n\n\na + 2'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_cell_explicit_start(pynb="\nimport pandas as pd\n# +\ndef data():\n    return pd.DataFrame({'A': [0, 1]})\n\n\ndata()\n"):
    if False:
        print('Hello World!')
    nb = jupytext.reads(pynb, 'py')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_complex_cells(pynb='import pandas as pd\n\n# +\ndef data():\n    return pd.DataFrame({\'A\': [0, 1]})\n\n\ndata()\n\n# +\ndef data2():\n    return pd.DataFrame({\'B\': [0, 1]})\n\n\ndata2()\n\n# +\n# Finally we have a cell with only comments\n# This cell should remain a code cell and not get converted\n# to markdown\n\n# + {"endofcell": "--"}\n# This cell has an enumeration in it that should not\n# match the endofcell marker!\n# - item 1\n# - item 2\n# -\n# --\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 5
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[3].cell_type == 'code'
    assert nb.cells[4].cell_type == 'code'
    assert nb.cells[3].source == '# Finally we have a cell with only comments\n# This cell should remain a code cell and not get converted\n# to markdown'
    assert nb.cells[4].source == '# This cell has an enumeration in it that should not\n# match the endofcell marker!\n# - item 1\n# - item 2\n# -'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_prev_function(pynb="def test_read_cell_explicit_start_end(pynb='''\nimport pandas as pd\n# +\ndef data():\n    return pd.DataFrame({'A': [0, 1]})\n\n\ndata()\n'''):\n    nb = jupytext.reads(pynb, 'py')\n    pynb2 = jupytext.writes(nb, 'py')\n    compare(pynb2, pynb)\n"):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(pynb, 'py')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_cell_with_one_blank_line_end(pynb='import pandas\n\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 1
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_code_cell_fully_commented(pynb='# +\n# This is a code cell that\n# only contains comments\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == '# This is a code cell that\n# only contains comments'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_file_with_two_blank_line_end(pynb='import pandas\n\n\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(pynb, 'py')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_one_blank_lines_after_endofcell(pynb='# +\n# This is a code cell with explicit end of cell\n1 + 1\n\n2 + 2\n# -\n\n# This cell is a cell with implicit start\n1 + 1\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == '# This is a code cell with explicit end of cell\n1 + 1\n\n2 + 2'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == '# This cell is a cell with implicit start\n1 + 1'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_two_cells_with_explicit_start(pynb='# +\n# Cell one\n1 + 1\n\n1 + 1\n\n# +\n# Cell two\n2 + 2\n\n2 + 2\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == '# Cell one\n1 + 1\n\n1 + 1'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == '# Cell two\n2 + 2\n\n2 + 2'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_escape_start_pattern(pynb='# The code start pattern \'# +\' can\n# appear in code and markdown cells.\n\n# In markdown cells it is escaped like here:\n# # + {"sample_metadata": "value"}\n\n# In code cells like this one, it is also escaped\n# # + {"sample_metadata": "value"}\n1 + 1\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[1].source == 'In markdown cells it is escaped like here:\n# + {"sample_metadata": "value"}'
    assert nb.cells[2].source == '# In code cells like this one, it is also escaped\n# + {"sample_metadata": "value"}\n1 + 1'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_dictionary_with_blank_lines_not_broken(pynb="# This is a markdown cell, and below\n# we have a long dictionary with blank lines\n# inside it\n\ndictionary = {\n    'a': 'A',\n    'b': 'B',\n\n    # and the end\n    'z': 'Z'}\n"):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[0].source == 'This is a markdown cell, and below\nwe have a long dictionary with blank lines\ninside it'
    assert nb.cells[1].source == "dictionary = {\n    'a': 'A',\n    'b': 'B',\n\n    # and the end\n    'z': 'Z'}"
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_isolated_cell_with_magic(pynb='# ---\n# title: cell with isolated jupyter magic\n# ---\n\n# A magic command included in a markdown\n# paragraph is code\n#\n# %matplotlib inline\n\n# a code block may start with\n# a magic command, like this one:\n\n# %matplotlib inline\n\n# or that one\n\n# %matplotlib inline\n1 + 1\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 6
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: cell with isolated jupyter magic\n---'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[2].cell_type == 'markdown'
    assert nb.cells[3].cell_type == 'code'
    assert nb.cells[3].source == '%matplotlib inline'
    assert nb.cells[4].cell_type == 'markdown'
    assert nb.cells[5].cell_type == 'code'
    assert nb.cells[5].source == '%matplotlib inline\n1 + 1'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_ipython_help_are_commented_297(text='# This is a markdown cell\n# that ends with a question: float?\n\n# The next cell is also a markdown cell,\n# because it has no code marker:\n\n# float?\n\n# +\n# float?\n\n# +\n# float??\n\n# +\n# Finally a question in a code\n# # cell?\n', nb=new_notebook(cells=[new_markdown_cell('This is a markdown cell\nthat ends with a question: float?'), new_markdown_cell('The next cell is also a markdown cell,\nbecause it has no code marker:'), new_markdown_cell('float?'), new_code_cell('float?'), new_code_cell('float??'), new_code_cell('# Finally a question in a code\n# cell?')])):
    if False:
        return 10
    nb2 = jupytext.reads(text, 'py')
    compare_notebooks(nb2, nb)
    text2 = jupytext.writes(nb2, 'py')
    compare(text2, text)

def test_questions_in_unmarked_cells_are_not_uncommented_297(text='# This cell has no explicit marker\n# question?\n1 + 2\n', nb=new_notebook(cells=[new_code_cell('# This cell has no explicit marker\n# question?\n1 + 2', metadata={'comment_questions': False})])):
    if False:
        for i in range(10):
            print('nop')
    nb2 = jupytext.reads(text, 'py')
    compare_notebooks(nb2, nb)
    text2 = jupytext.writes(nb2, 'py')
    compare(text2, text)

def test_read_multiline_comment(pynb='\'\'\'This is a multiline\ncomment with "quotes", \'single quotes\'\n# and comments\nand line breaks\n\n\nand it ends here\'\'\'\n\n\n1 + 1\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == '\'\'\'This is a multiline\ncomment with "quotes", \'single quotes\'\n# and comments\nand line breaks\n\n\nand it ends here\'\'\''
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == '1 + 1'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_no_space_after_code(pynb='# -*- coding: utf-8 -*-\n# Markdown cell\n\ndef f(x):\n    return x+1\n\n# And a new cell, and non ascii contênt\n'):
    if False:
        return 10
    nb = jupytext.reads(pynb, 'py')
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'Markdown cell'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'def f(x):\n    return x+1'
    assert nb.cells[2].cell_type == 'markdown'
    assert nb.cells[2].source == 'And a new cell, and non ascii contênt'
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_write_script(pynb="#!/usr/bin/env python\n# coding=utf-8\nprint('Hello world')\n"):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(pynb, 'py')
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_read_write_script_with_metadata_241(no_jupytext_version_number, pynb='#!/usr/bin/env python3\n# -*- coding: utf-8 -*-\n# ---\n# jupyter:\n#   jupytext:\n#     text_representation:\n#       extension: .py\n#       format_name: light\n#   kernelspec:\n#     display_name: Python 3\n#     language: python\n#     name: python3\n# ---\n\na = 2\n\na + 1\n'):
    if False:
        return 10
    nb = jupytext.reads(pynb, 'py')
    assert 'executable' in nb.metadata['jupytext']
    assert 'encoding' in nb.metadata['jupytext']
    pynb2 = jupytext.writes(nb, 'py')
    compare(pynb2, pynb)

def test_notebook_blank_lines(script='# +\n# This is a comment\n# followed by two variables\na = 3\n\nb = 4\n# -\n\n# New cell is a variable\nc = 5\n\n\n# +\n# Now we have two functions\ndef f(x):\n    return x + x\n\n\ndef g(x):\n    return x + x + x\n\n\n# -\n\n\n# A commented block that is two lines away\n# from previous cell\n\n# A function again\ndef h(x):\n    return x + 1\n\n\n# variable\nd = 6\n'):
    if False:
        print('Hello World!')
    notebook = jupytext.reads(script, 'py')
    assert len(notebook.cells) >= 6
    for cell in notebook.cells:
        lines = cell.source.splitlines()
        if len(lines) != 1:
            assert lines[0], cell.source
            assert lines[-1], cell.source
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)

def test_notebook_two_blank_lines_before_next_cell(script='# +\n# This is cell with a function\n\ndef f(x):\n    return 4\n\n\n# +\n# Another cell\nc = 5\n\n\ndef g(x):\n    return 6\n\n\n# +\n# Final cell\n\n1 + 1\n'):
    if False:
        i = 10
        return i + 15
    notebook = jupytext.reads(script, 'py')
    assert len(notebook.cells) == 3
    for cell in notebook.cells:
        lines = cell.source.splitlines()
        if len(lines) != 1:
            assert lines[0]
            assert lines[-1]
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)

def test_notebook_one_blank_line_between_cells(script='# +\n1 + 1\n\n2 + 2\n\n# +\n3 + 3\n\n4 + 4\n\n# +\n5 + 5\n\n\ndef g(x):\n    return 6\n\n\n# +\n7 + 7\n\n\ndef h(x):\n    return 8\n\n\n# +\ndef i(x):\n    return 9\n\n\n10 + 10\n\n\n# +\ndef j(x):\n    return 11\n\n\n12 + 12\n'):
    if False:
        for i in range(10):
            print('nop')
    notebook = jupytext.reads(script, 'py')
    for cell in notebook.cells:
        lines = cell.source.splitlines()
        assert lines[0]
        assert lines[-1]
        assert not cell.metadata, cell.source
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)

def test_notebook_with_magic_and_bash_cells(script='# This is a test for issue #181\n\n# %load_ext line_profiler\n\n# !head -4 data/president_heights.csv\n'):
    if False:
        for i in range(10):
            print('nop')
    notebook = jupytext.reads(script, 'py')
    for cell in notebook.cells:
        lines = cell.source.splitlines()
        assert lines[0]
        assert lines[-1]
        assert not cell.metadata, cell.source
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)

def test_notebook_no_line_to_next_cell(nb=new_notebook(cells=[new_markdown_cell('Markdown cell #1'), new_code_cell('%load_ext line_profiler'), new_markdown_cell('Markdown cell #2'), new_code_cell('%lprun -f ...'), new_markdown_cell('Markdown cell #3'), new_code_cell('# And a function!\ndef f(x):\n    return 5')])):
    if False:
        i = 10
        return i + 15
    script = jupytext.writes(nb, 'py')
    nb2 = jupytext.reads(script, 'py')
    nb2.metadata.pop('jupytext')
    compare_notebooks(nb2, nb)

def test_notebook_one_blank_line_before_first_markdown_cell(script='\n# This is a markdown cell\n\n1 + 1\n'):
    if False:
        print('Hello World!')
    notebook = jupytext.reads(script, 'py')
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)
    assert len(notebook.cells) == 3
    for cell in notebook.cells:
        lines = cell.source.splitlines()
        if len(lines):
            assert lines[0]
            assert lines[-1]

def test_read_markdown_cell_with_triple_quote_307(script="# This script test that commented triple quotes '''\n# do not impede the correct identification of Markdown cells\n\n# Here is Markdown cell number 2 '''\n"):
    if False:
        return 10
    notebook = jupytext.reads(script, 'py')
    assert len(notebook.cells) == 2
    assert notebook.cells[0].cell_type == 'markdown'
    assert notebook.cells[0].source == "This script test that commented triple quotes '''\ndo not impede the correct identification of Markdown cells"
    assert notebook.cells[1].cell_type == 'markdown'
    assert notebook.cells[1].source == "Here is Markdown cell number 2 '''"
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)

def test_read_explicit_markdown_cell_with_triple_quote_307(script='# {{{ [md] {"special": "metadata"}\n# some text \'\'\'\n# }}}\n\nprint(\'hello world\')\n\n# {{{ [md] {"special": "metadata"}\n# more text \'\'\'\n# }}}\n'):
    if False:
        return 10
    notebook = jupytext.reads(script, 'py')
    assert len(notebook.cells) == 3
    assert notebook.cells[0].cell_type == 'markdown'
    assert notebook.cells[0].source == "some text '''"
    assert notebook.cells[1].cell_type == 'code'
    assert notebook.cells[1].source == "print('hello world')"
    assert notebook.cells[2].cell_type == 'markdown'
    assert notebook.cells[2].source == "more text '''"
    script2 = jupytext.writes(notebook, 'py')
    compare(script2, script)

def test_round_trip_markdown_cell_with_magic():
    if False:
        print('Hello World!')
    notebook = new_notebook(cells=[new_markdown_cell('IPython has magic commands like\n%quickref')], metadata={'jupytext': {'main_language': 'python'}})
    text = jupytext.writes(notebook, 'py')
    notebook2 = jupytext.reads(text, 'py')
    compare_notebooks(notebook2, notebook)

def test_round_trip_python_with_js_cell():
    if False:
        return 10
    notebook = new_notebook(cells=[new_code_cell("import notebook.nbextensions\nnotebook.nbextensions.install_nbextension('index.js', user=True)"), new_code_cell("%%javascript\nJupyter.utils.load_extensions('jupytext')")])
    text = jupytext.writes(notebook, 'py')
    notebook2 = jupytext.reads(text, 'py')
    compare_notebooks(notebook2, notebook)

def test_round_trip_python_with_js_cell_no_cell_metadata():
    if False:
        i = 10
        return i + 15
    notebook = new_notebook(cells=[new_code_cell("import notebook.nbextensions\nnotebook.nbextensions.install_nbextension('index.js', user=True)"), new_code_cell("%%javascript\nJupyter.utils.load_extensions('jupytext')")], metadata={'jupytext': {'notebook_metadata_filter': '-all', 'cell_metadata_filter': '-all'}})
    text = jupytext.writes(notebook, 'py')
    notebook2 = jupytext.reads(text, 'py')
    compare_notebooks(notebook2, notebook)

def test_raw_with_metadata(no_jupytext_version_number, text='# + key="value" active=""\n# Raw cell\n# # Commented line\n', notebook=new_notebook(cells=[new_raw_cell('Raw cell\n# Commented line', metadata={'key': 'value'})])):
    if False:
        return 10
    nb2 = jupytext.reads(text, 'py')
    compare_notebooks(nb2, notebook)
    text2 = jupytext.writes(notebook, 'py')
    compare(text2, text)

def test_raw_with_metadata_2(no_jupytext_version_number, text='# + [raw] key="value"\n# Raw cell\n# # Commented line\n', notebook=new_notebook(cells=[new_raw_cell('Raw cell\n# Commented line', metadata={'key': 'value'})])):
    if False:
        i = 10
        return i + 15
    nb2 = jupytext.reads(text, 'py')
    compare_notebooks(nb2, notebook)

def test_markdown_with_metadata(no_jupytext_version_number, text='# + [markdown] key="value"\n# Markdown cell\n', notebook=new_notebook(cells=[new_markdown_cell('Markdown cell', metadata={'key': 'value'})])):
    if False:
        print('Hello World!')
    nb2 = jupytext.reads(text, 'py')
    compare_notebooks(nb2, notebook)
    text2 = jupytext.writes(notebook, 'py')
    compare(text2, text)

def test_multiline_comments_in_markdown_1():
    if False:
        return 10
    text = "# + [markdown]\n'''\na\nlong\ncell\n'''\n"
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'a\nlong\ncell'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_multiline_comments_in_markdown_2():
    if False:
        for i in range(10):
            print('nop')
    text = '# + [markdown]\n"""\na\nlong\ncell\n"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'a\nlong\ncell'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_multiline_comments_in_raw_cell():
    if False:
        return 10
    text = '# + active=""\n"""\nsome\ntext\n"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == 'some\ntext'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_multiline_comments_in_markdown_cell_no_line_return():
    if False:
        return 10
    text = '# + [md]\n"""a\nlong\ncell"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'a\nlong\ncell'

def test_multiline_comments_in_markdown_cell_is_robust_to_additional_cell_marker():
    if False:
        i = 10
        return i + 15
    text = '# + [md]\n"""\nsome text, and a fake cell marker\n# + [raw]\n"""\n'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'some text, and a fake cell marker\n# + [raw]'
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_active_tag(text='# + tags=["active-py"]\ninterpreter = \'python\'\n\n# + tags=["active-ipynb"]\n# interpreter = \'ipython\'\n', ref=new_notebook(cells=[new_raw_cell("interpreter = 'python'", metadata={'tags': ['active-py']}), new_code_cell("interpreter = 'ipython'", metadata={'tags': ['active-ipynb']})])):
    if False:
        print('Hello World!')
    nb = jupytext.reads(text, 'py')
    compare_notebooks(nb, ref)
    py = jupytext.writes(nb, 'py')
    compare(py, text)

def test_indented_bash_command(no_jupytext_version_number, nb=new_notebook(cells=[new_code_cell('try:\n    !echo jo\n    pass\nexcept:\n    pass')]), text='try:\n    # !echo jo\n    pass\nexcept:\n    pass\n'):
    if False:
        i = 10
        return i + 15
    'Reproduces https://github.com/mwouts/jupytext/issues/437'
    py = jupytext.writes(nb, 'py')
    compare(py, text)
    nb2 = jupytext.reads(py, 'py')
    compare_notebooks(nb2, nb)

def test_two_raw_cells_are_preserved(nb=new_notebook(cells=[new_raw_cell('---\nX\n---'), new_raw_cell('Y')])):
    if False:
        while True:
            i = 10
    'Test the pattern described at https://github.com/mwouts/jupytext/issues/466'
    py = jupytext.writes(nb, 'py')
    nb2 = jupytext.reads(py, 'py')
    compare_notebooks(nb2, nb)

def test_no_metadata_on_multiline_decorator(text='import pytest\n\n\n@pytest.mark.parametrize(\n    "arg",\n    [\n        \'a\',\n        \'b\',\n        \'c\'\n    ],\n)\ndef test_arg(arg):\n    assert isinstance(arg, str)\n'):
    if False:
        print('Hello World!')
    'Applying black on the code of jupytext 1.4.2 turns some pytest parameters into multi-lines ones, and\n    causes a few failures in test_pep8.py:test_no_metadata_when_py_is_pep8'
    nb = jupytext.reads(text, 'py')
    assert len(nb.cells) == 2
    for cell in nb.cells:
        assert cell.cell_type == 'code'
    assert nb.cells[0].source == 'import pytest'
    assert nb.cells[0].metadata == {}

@pytest.mark.parametrize('script,cell', [('if True:\n    # # !rm file 1\n    # !rm file 2\n', 'if True:\n    # !rm file 1\n    !rm file 2'), ('# +\nif True:\n    # help?\n    # ?help\n    # # ?help\n    # # help?\n', 'if True:\n    help?\n    ?help\n    # ?help\n    # help?')])
def test_indented_magic_commands(script, cell):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'code'
    compare(nb.cells[0].source, cell)
    assert nb.cells[0].metadata == {}
    compare(jupytext.writes(nb, 'py'), script)