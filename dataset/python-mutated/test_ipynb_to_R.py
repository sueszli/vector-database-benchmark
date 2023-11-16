import itertools
import nbformat
import pytest
import jupytext
from jupytext.compare import compare_notebooks
from .utils import list_notebooks

@pytest.mark.parametrize('nb_file,ext', itertools.product(list_notebooks('ipynb_R'), ['.r', '.R']))
def test_identity_source_write_read(nb_file, ext):
    if False:
        while True:
            i = 10
    '\n    Test that writing the notebook with R, and read again,\n    is the same as removing outputs\n    '
    with open(nb_file) as fp:
        nb1 = nbformat.read(fp, as_version=4)
    R = jupytext.writes(nb1, ext)
    nb2 = jupytext.reads(R, ext)
    compare_notebooks(nb2, nb1)