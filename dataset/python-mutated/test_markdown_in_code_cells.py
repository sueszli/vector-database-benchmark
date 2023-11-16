"""Issue #712"""
from nbformat.v4.nbbase import new_code_cell, new_notebook
from jupytext import reads, writes
from jupytext.cell_to_text import three_backticks_or_more
from jupytext.compare import compare, compare_notebooks
from .utils import requires_myst

def test_three_backticks_or_more():
    if False:
        while True:
            i = 10
    assert three_backticks_or_more(['']) == '```'
    assert three_backticks_or_more(['``']) == '```'
    assert three_backticks_or_more(['```python']) == '````'
    assert three_backticks_or_more(['```']) == '````'
    assert three_backticks_or_more(['`````python']) == '``````'
    assert three_backticks_or_more(['`````']) == '``````'

def test_triple_backticks_in_code_cell(no_jupytext_version_number, nb=new_notebook(metadata={'main_language': 'python'}, cells=[new_code_cell('a = """\n```\nfoo\n```\n"""')]), text='---\njupyter:\n  jupytext:\n    main_language: python\n---\n\n````python\na = """\n```\nfoo\n```\n"""\n````\n'):
    if False:
        i = 10
        return i + 15
    actual_text = writes(nb, fmt='md')
    compare(actual_text, text)
    actual_nb = reads(text, fmt='md')
    compare_notebooks(actual_nb, nb)

@requires_myst
def test_triple_backticks_in_code_cell_myst(no_jupytext_version_number, nb=new_notebook(metadata={'main_language': 'python'}, cells=[new_code_cell('a = """\n```\nfoo\n```\n"""')]), text='---\njupytext:\n  main_language: python\n---\n\n````{code-cell}\na = """\n```\nfoo\n```\n"""\n````\n'):
    if False:
        return 10
    actual_text = writes(nb, fmt='md:myst')
    compare(actual_text, text)
    actual_nb = reads(text, fmt='md:myst')
    compare_notebooks(actual_nb, nb)

def test_alternate_tree_four_five_backticks(no_jupytext_version_number, nb=new_notebook(metadata={'main_language': 'python'}, cells=[new_code_cell('a = """\n```\n"""'), new_code_cell('b = 2'), new_code_cell('c = """\n````\n"""')]), text='---\njupyter:\n  jupytext:\n    main_language: python\n---\n\n````python\na = """\n```\n"""\n````\n\n```python\nb = 2\n```\n\n`````python\nc = """\n````\n"""\n`````\n'):
    if False:
        print('Hello World!')
    actual_text = writes(nb, fmt='md')
    compare(actual_text, text)
    actual_nb = reads(text, fmt='md')
    compare_notebooks(actual_nb, nb)