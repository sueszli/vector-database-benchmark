import pytest
from nbformat.v4.nbbase import new_markdown_cell, new_notebook
import jupytext
from jupytext.compare import compare
from .utils import list_notebooks

@pytest.mark.parametrize('nb_file', list_notebooks() + list_notebooks('Rmd'))
def test_notebook_contents_is_unicode(nb_file):
    if False:
        i = 10
        return i + 15
    nb = jupytext.read(nb_file)
    for cell in nb.cells:
        assert isinstance(cell.source, str)

def test_write_non_ascii(tmpdir):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads('Non-ascii contênt', 'Rmd')
    jupytext.write(nb, str(tmpdir.join('notebook.Rmd')))
    jupytext.write(nb, str(tmpdir.join('notebook.ipynb')))

def test_no_encoding_in_python_scripts(no_jupytext_version_number):
    if False:
        while True:
            i = 10
    'No UTF encoding should not be added to Python scripts'
    nb = new_notebook(cells=[new_markdown_cell('α')], metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    py_light = jupytext.writes(nb, 'py:light')
    compare(py_light, '# ---\n# jupyter:\n#   kernelspec:\n#     display_name: Python 3\n#     language: python\n#     name: python3\n# ---\n\n# α\n')

def test_encoding_in_scripts_only(no_jupytext_version_number):
    if False:
        i = 10
        return i + 15
    'UTF encoding should not be added to markdown files'
    nb = new_notebook(cells=[new_markdown_cell('α')], metadata={'encoding': '# -*- coding: utf-8 -*-', 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}})
    py_light = jupytext.writes(nb, 'py:light')
    compare(py_light, '# -*- coding: utf-8 -*-\n# ---\n# jupyter:\n#   kernelspec:\n#     display_name: Python 3\n#     language: python\n#     name: python3\n# ---\n\n# α\n')
    nb = jupytext.reads(py_light, 'py:light')
    assert 'encoding' in nb.metadata['jupytext']
    py_percent = jupytext.writes(nb, 'py:percent')
    compare(py_percent, '# -*- coding: utf-8 -*-\n# ---\n# jupyter:\n#   kernelspec:\n#     display_name: Python 3\n#     language: python\n#     name: python3\n# ---\n\n# %% [markdown]\n# α\n')
    md = jupytext.writes(nb, 'md')
    compare(md, "---\njupyter:\n  jupytext:\n    encoding: '# -*- coding: utf-8 -*-'\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n\nα\n")
    rmd = jupytext.writes(nb, 'Rmd')
    compare(rmd, "---\njupyter:\n  jupytext:\n    encoding: '# -*- coding: utf-8 -*-'\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n\nα\n")