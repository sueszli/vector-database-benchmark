from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook
import jupytext
from jupytext.compare import compare, compare_notebooks

def test_read_simple_code(no_jupytext_version_number, ml='(* %% *)\nlet sum x y = x + y\n', nb=new_notebook(cells=[new_code_cell('let sum x y = x + y')])):
    if False:
        print('Hello World!')
    nb2 = jupytext.reads(ml, 'ml:percent')
    compare_notebooks(nb2, nb)
    ml2 = jupytext.writes(nb, 'ml:percent')
    compare(ml2, ml)

def test_read_simple_markdown(no_jupytext_version_number, ml='(* %% [markdown] *)\n(* # Example of an OCaml notebook *)\n', nb=new_notebook(cells=[new_markdown_cell('# Example of an OCaml notebook')])):
    if False:
        while True:
            i = 10
    nb2 = jupytext.reads(ml, 'ml:percent')
    compare_notebooks(nb2, nb)
    ml2 = jupytext.writes(nb, 'ml:percent')
    compare(ml2, ml)