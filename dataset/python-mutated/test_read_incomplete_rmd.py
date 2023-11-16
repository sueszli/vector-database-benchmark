import jupytext

def test_incomplete_header(rmd='---\ntitle: Incomplete header\n\n```{python}\n1+1\n```\n'):
    if False:
        return 10
    nb = jupytext.reads(rmd, 'Rmd')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == '---\ntitle: Incomplete header'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == '1+1'

def test_code_in_markdown_block(rmd="```{python}\na = 1\nb = 2\na + b\n```\n\n```python\n'''Code here goes to a Markdown cell'''\n\n\n'''even if we have two blank lines above'''\n```\n\n```{bash}\nls -l\n```\n"):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(rmd, 'Rmd')
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == 'a = 1\nb = 2\na + b'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[1].source == "```python\n'''Code here goes to a Markdown cell'''\n\n\n'''even if we have two blank lines above'''\n```"
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[2].source == '%%bash\nls -l'

def test_unterminated_header(rmd='---\ntitle: Unterminated header\n\n```{python}\n1+3\n```\n\nsome text\n\n```{r}\n1+4\n```\n\n```{python not_terminated}\n1+5\n'):
    if False:
        return 10
    nb = jupytext.reads(rmd, 'Rmd')
    assert len(nb.cells) == 5
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == '---\ntitle: Unterminated header'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == '1+3'
    assert nb.cells[2].cell_type == 'markdown'
    assert nb.cells[2].source == 'some text'
    assert nb.cells[3].cell_type == 'code'
    assert nb.cells[3].source == '%%R\n1+4'
    assert nb.cells[4].cell_type == 'code'
    assert nb.cells[4].metadata == {'name': 'not_terminated'}
    assert nb.cells[4].source == '1+5'