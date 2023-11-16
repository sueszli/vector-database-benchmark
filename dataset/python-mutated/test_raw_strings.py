import pytest
from nbformat.v4.nbbase import new_markdown_cell, new_notebook
import jupytext
from jupytext.compare import compare

def test_markdown_cell_with_backslash_is_encoded_with_raw_string(nb=new_notebook(cells=[new_markdown_cell('A $\\LaTeX$ expression')]), py='# %% [markdown]\nr"""\nA $\\LaTeX$ expression\n"""\n'):
    if False:
        for i in range(10):
            print('nop')
    nb.metadata['jupytext'] = {'cell_markers': '"""', 'notebook_metadata_filter': '-all'}
    py2 = jupytext.writes(nb, 'py:percent')
    compare(py2, py)

@pytest.mark.parametrize('r', ['r', 'R'])
@pytest.mark.parametrize('triple_quote', ['"""', "'''"])
@pytest.mark.parametrize('expr', ['$\\LaTeX$', 'common'])
def test_raw_string_is_stable_over_round_trip(r, triple_quote, expr):
    if False:
        for i in range(10):
            print('nop')
    py = f'# %% [markdown]\n{r}{triple_quote}\nA {expr} expression\n{triple_quote}\n'
    nb = jupytext.reads(py, 'py:percent')
    (cell,) = nb.cells
    assert cell.cell_type == 'markdown'
    assert cell.source == f'A {expr} expression'
    assert cell.metadata['cell_marker'] == f'{r}{triple_quote}'
    py2 = jupytext.writes(nb, 'py:percent')
    compare(py2, py)