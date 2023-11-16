import pytest
from nbformat.v4.nbbase import new_notebook
from tornado.web import HTTPError
import jupytext

def test_combine_same_version_ok(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = 'notebook.ipynb'
    tmp_nbpy = 'notebook.py'
    with open(str(tmpdir.join(tmp_nbpy)), 'w') as fp:
        fp.write("# ---\n# jupyter:\n#   jupytext_formats: ipynb,py\n#   jupytext_format_version: '1.2'\n# ---\n\n# New cell\n")
    nb = new_notebook(metadata={'jupytext_formats': 'ipynb,py'})
    jupytext.write(nb, str(tmpdir.join(tmp_ipynb)))
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,py'
    cm.root_dir = str(tmpdir)
    nb = cm.get(tmp_ipynb)
    cells = nb['content']['cells']
    assert len(cells) == 1
    assert cells[0].cell_type == 'markdown'
    assert cells[0].source == 'New cell'

def test_combine_lower_version_raises(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = 'notebook.ipynb'
    tmp_nbpy = 'notebook.py'
    with open(str(tmpdir.join(tmp_nbpy)), 'w') as fp:
        fp.write("# ---\n# jupyter:\n#   jupytext_formats: ipynb,py\n#   jupytext_format_version: '0.0'\n# ---\n\n# New cell\n")
    nb = new_notebook(metadata={'jupytext_formats': 'ipynb,py'})
    jupytext.write(nb, str(tmpdir.join(tmp_ipynb)))
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,py'
    cm.root_dir = str(tmpdir)
    with pytest.raises(HTTPError):
        cm.get(tmp_ipynb)