import pytest
from nbformat.v4.nbbase import new_code_cell, new_notebook
from jupytext import read, reads, writes
from jupytext.compare import compare
from jupytext.pep8 import cell_ends_with_code, cell_ends_with_function_or_class, cell_has_code, next_instruction_is_function_or_class, pep8_lines_between_cells
from .utils import list_notebooks

def test_next_instruction_is_function_or_class():
    if False:
        for i in range(10):
            print('nop')
    text = "@pytest.mark.parametrize('py_file',\n    [py_file for py_file in list_notebooks('../src/jupytext') + list_notebooks('.') if\n                                     py_file.endswith('.py')])\ndef test_no_metadata_when_py_is_pep8(py_file):\n    pass\n"
    assert next_instruction_is_function_or_class(text.splitlines())

def test_cell_ends_with_code():
    if False:
        return 10
    assert not cell_ends_with_code([])

def test_cell_ends_with_function_or_class():
    if False:
        print('Hello World!')
    text = "class A:\n    __init__():\n    '''A docstring\nwith two lines or more'''\n        self.a = 0\n"
    assert cell_ends_with_function_or_class(text.splitlines())
    lines = ['#', '#']
    assert not cell_ends_with_function_or_class(lines)
    text = '# two blank line after this class\nclass A:\n    pass\n\n\n# so we do not need to insert two blank lines below this cell\n    '
    assert not cell_ends_with_function_or_class(text.splitlines())
    text = '# All lines\n# are commented'
    assert not cell_ends_with_function_or_class(text.splitlines())
    text = '# Two blank lines after function\ndef f(x):\n    return x\n\n\n# And a comment here'
    assert not cell_ends_with_function_or_class(text.splitlines())
    assert not cell_ends_with_function_or_class(['', '#'])

def test_pep8_lines_between_cells():
    if False:
        while True:
            i = 10
    prev_lines = 'a = a_long_instruction(\n    over_two_lines=True)'.splitlines()
    next_lines = 'def f(x):\n    return x'.splitlines()
    assert cell_ends_with_code(prev_lines)
    assert next_instruction_is_function_or_class(next_lines)
    assert pep8_lines_between_cells(prev_lines, next_lines, '.py') == 2

def test_pep8_lines_between_cells_bis():
    if False:
        return 10
    prev_lines = 'def f(x):\n    return x'.splitlines()
    next_lines = '# A markdown cell\n\n# An instruction\na = 5\n'.splitlines()
    assert cell_ends_with_function_or_class(prev_lines)
    assert cell_has_code(next_lines)
    assert pep8_lines_between_cells(prev_lines, next_lines, '.py') == 2
    next_lines = '# A markdown cell\n\n# Only markdown here\n# And here\n'.splitlines()
    assert cell_ends_with_function_or_class(prev_lines)
    assert not cell_has_code(next_lines)
    assert pep8_lines_between_cells(prev_lines, next_lines, '.py') == 1

def test_pep8_lines_between_cells_ter():
    if False:
        while True:
            i = 10
    prev_lines = ['from jupytext.cell_to_text import RMarkdownCellExporter']
    next_lines = '@pytest.mark.parametrize(\n    "lines",\n    [\n        "# text",\n        """# # %%R\n# # comment\n# 1 + 1\n# 2 + 2\n""",\n    ],\n)\ndef test_paragraph_is_fully_commented(lines):\n    assert paragraph_is_fully_commented(\n        lines.splitlines(), comment="#", main_language="python"\n    )'.splitlines()
    assert cell_ends_with_code(prev_lines)
    assert next_instruction_is_function_or_class(next_lines)
    assert pep8_lines_between_cells(prev_lines, next_lines, '.py') == 2

def test_pep8():
    if False:
        return 10
    text = 'import os\n\npath = os.path\n\n\n# code cell #1, with a comment on f\ndef f(x):\n    return x + 1\n\n\n# markdown cell #1\n\n# code cell #2 - an instruction\na = 4\n\n\n# markdown cell #2\n\n# code cell #3 with a comment on g\ndef g(x):\n    return x + 1\n\n\n# markdown cell #3\n\n# the two lines are:\n# - right below the function/class\n# - below the last python paragraph (i.e. NOT ABOVE g)\n\n# code cell #4\nx = 4\n'
    nb = reads(text, 'py')
    for cell in nb.cells:
        assert not cell.metadata
    text2 = writes(nb, 'py')
    compare(text2, text)

def test_pep8_bis():
    if False:
        i = 10
        return i + 15
    text = '# This is a markdown cell\n\n# a code cell\ndef f(x):\n    return x + 1\n\n# And another markdown cell\n# Separated from f by just one line\n# As there is no code here\n'
    nb = reads(text, 'py')
    for cell in nb.cells:
        assert not cell.metadata
    text2 = writes(nb, 'py')
    compare(text2, text)

@pytest.mark.parametrize('py_file', [py_file for py_file in list_notebooks('../src/jupytext') + list_notebooks('.') if py_file.endswith('.py') and 'folding_markers' not in py_file])
def test_no_metadata_when_py_is_pep8(py_file):
    if False:
        return 10
    'This test assumes that all Python files in the jupytext folder follow PEP8 rules'
    nb = read(py_file)
    for (i, cell) in enumerate(nb.cells):
        if 'title' in cell.metadata:
            cell.metadata.pop('title')
        if i == 0 and (not cell.source):
            assert cell.metadata == {'lines_to_next_cell': 0}, py_file
        else:
            assert not cell.metadata, (py_file, cell.source)

def test_notebook_ends_with_exactly_one_empty_line_682():
    if False:
        while True:
            i = 10
    '(Issue #682)\n    Steps to reproduce:\n\n        Have a notebook that ends in a python code cell (with no empty lines at the end of the cell).\n        run jupytext --to py:percent notebookWithCodeCell.ipynb.\n        See that the generated python code file has two empty lines at the end.\n\n    I would expect there to just be one new line.'
    nb = new_notebook(cells=[new_code_cell('1+1')], metadata={'jupytext': {'main_language': 'python'}})
    py = writes(nb, 'py:percent')
    assert py.endswith('1+1\n')