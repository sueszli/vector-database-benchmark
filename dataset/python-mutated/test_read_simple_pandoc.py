import pytest
from nbformat.v4.nbbase import new_markdown_cell, new_notebook
import jupytext
from jupytext.cli import jupytext as jupytext_cli
from jupytext.compare import compare, compare_notebooks
from jupytext.pandoc import PandocError
from .utils import requires_no_pandoc, requires_pandoc

@requires_pandoc
def test_pandoc_implicit(markdown='# Lorem ipsum\n\n**Lorem ipsum** dolor sit amet, consectetur adipiscing elit. Nunc luctus\nbibendum felis dictum sodales.\n\n``` code\nprint("hello")\n```\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(markdown, 'md:pandoc')
    markdown2 = jupytext.writes(nb, 'md')
    nb2 = jupytext.reads(markdown2, 'md')
    compare_notebooks(nb2, nb)
    markdown3 = jupytext.writes(nb2, 'md')
    compare(markdown3, markdown2)

@requires_pandoc
def test_pandoc_explicit(markdown='::: {.cell .markdown}\n# Lorem\n\n**Lorem ipsum** dolor sit amet, consectetur adipiscing elit. Nunc luctus\nbibendum felis dictum sodales.\n:::'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(markdown, 'md')
    markdown2 = jupytext.writes(nb, 'md')
    compare('\n'.join(markdown2.splitlines()[12:]), markdown)

@requires_pandoc
def test_pandoc_utf8_in_md(markdown='::: {.cell .markdown}\n# Utf-8 support\n\nThis is the greek letter $\\pi$: π\n:::'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(markdown, 'md')
    markdown2 = jupytext.writes(nb, 'md')
    compare('\n'.join(markdown2.splitlines()[12:]), markdown)

@requires_pandoc
def test_pandoc_utf8_in_nb(nb=new_notebook(cells=[new_markdown_cell('# Utf-8 support\n\nThis is the greek letter $\\pi$: π')])):
    if False:
        print('Hello World!')
    markdown = jupytext.writes(nb, 'md:pandoc')
    nb2 = jupytext.reads(markdown, 'md:pandoc')
    nb2.metadata.pop('jupytext')
    compare_notebooks(nb, nb2, 'md:pandoc')

@requires_no_pandoc
def test_meaningfull_error_when_pandoc_is_missing(tmpdir):
    if False:
        while True:
            i = 10
    nb_file = tmpdir.join('notebook.ipynb')
    jupytext.write(new_notebook(), str(nb_file))
    with pytest.raises(PandocError, match="The Pandoc Markdown format requires 'pandoc>=2.7.2'"):
        jupytext_cli([str(nb_file), '--to', 'md:pandoc'])