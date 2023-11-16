import nbformat
import pytest
import jupytext
from jupytext.compare import compare_notebooks
from .utils import list_notebooks

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_identity_source_write_read(nb_file):
    if False:
        i = 10
        return i + 15
    'Test that writing the notebook with jupytext, and read again,\n    is the same as removing outputs'
    with open(nb_file) as fp:
        nb1 = nbformat.read(fp, as_version=4)
    py = jupytext.writes(nb1, 'py')
    nb2 = jupytext.reads(py, 'py')
    compare_notebooks(nb2, nb1)