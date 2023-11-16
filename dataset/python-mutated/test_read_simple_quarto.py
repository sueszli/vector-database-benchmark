from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook
import jupytext
from jupytext.compare import compare, compare_notebooks
from .utils import requires_quarto

@requires_quarto
def test_qmd_to_ipynb(qmd='Some text\n\n```{python}\n1 + 1\n```\n', nb=new_notebook(cells=[new_markdown_cell('Some text'), new_code_cell('1 + 1')], metadata={'kernelspec': {'display_name': 'python_kernel', 'language': 'python', 'name': 'python_kernel'}})):
    if False:
        while True:
            i = 10
    nb2 = jupytext.reads(qmd, 'qmd')
    compare_notebooks(nb2, nb)
    qmd2 = jupytext.writes(nb, 'qmd')
    qmd2_without_header = qmd2.rsplit('---\n\n', 1)[1]
    compare(qmd2_without_header, qmd)