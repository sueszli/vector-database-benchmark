import pytest
from nbformat.v4.nbbase import new_markdown_cell
from jupytext.cell_reader import LightScriptCellReader, RMarkdownCellReader, paragraph_is_fully_commented, uncomment
from jupytext.cell_to_text import RMarkdownCellExporter

@pytest.mark.parametrize('lines', ['# text', '# # %%R\n# # comment\n# 1 + 1\n# 2 + 2\n'])
def test_paragraph_is_fully_commented(lines):
    if False:
        for i in range(10):
            print('nop')
    assert paragraph_is_fully_commented(lines.splitlines(), comment='#', main_language='python')

def test_paragraph_is_not_fully_commented(lines='# text\nnot fully commented out'):
    if False:
        return 10
    assert not paragraph_is_fully_commented(lines.splitlines(), comment='#', main_language='python')

def test_uncomment():
    if False:
        return 10
    assert uncomment(['# line one', '#line two', 'line three'], '#') == ['line one', 'line two', 'line three']
    assert uncomment(['# line one', '#line two', 'line three'], '') == ['# line one', '#line two', 'line three']

def test_text_to_code_cell():
    if False:
        for i in range(10):
            print('nop')
    text = '```{python}\n1+2+3\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'code'
    assert cell.source == '1+2+3'
    assert cell.metadata == {'language': 'python'}
    assert lines[pos:] == []

def test_text_to_code_cell_empty_code():
    if False:
        while True:
            i = 10
    text = '```{python}\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'code'
    assert cell.source == ''
    assert cell.metadata == {'language': 'python'}
    assert lines[pos:] == []

def test_text_to_code_cell_empty_code_no_blank_line():
    if False:
        while True:
            i = 10
    text = '```{python}\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'code'
    assert cell.source == ''
    assert cell.metadata == {'language': 'python'}
    assert lines[pos:] == []

def test_text_to_markdown_cell():
    if False:
        print('Hello World!')
    text = 'This is\na markdown cell\n\n```{python}\n1+2+3\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'markdown'
    assert cell.source == 'This is\na markdown cell'
    assert cell.metadata == {}
    assert pos == 3

def test_text_to_markdown_no_blank_line():
    if False:
        i = 10
        return i + 15
    text = 'This is\na markdown cell\n```{python}\n1+2+3\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'markdown'
    assert cell.source == 'This is\na markdown cell'
    assert cell.metadata == {'lines_to_next_cell': 0}
    assert pos == 2

def test_text_to_markdown_two_blank_line():
    if False:
        for i in range(10):
            print('nop')
    text = '\n\n```{python}\n1+2+3\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'markdown'
    assert cell.source == ''
    assert cell.metadata == {}
    assert pos == 2

def test_text_to_markdown_one_blank_line():
    if False:
        print('Hello World!')
    text = '\n```{python}\n1+2+3\n```\n'
    lines = text.splitlines()
    (cell, pos) = RMarkdownCellReader().read(lines)
    assert cell.cell_type == 'markdown'
    assert cell.source == ''
    assert cell.metadata == {'lines_to_next_cell': 0}
    assert pos == 1

def test_empty_markdown_to_text():
    if False:
        while True:
            i = 10
    cell = new_markdown_cell(source='')
    text = RMarkdownCellExporter(cell, 'python').cell_to_text()
    assert text == ['']

def test_text_to_cell_py():
    if False:
        while True:
            i = 10
    text = '1+1\n'
    lines = text.splitlines()
    (cell, pos) = LightScriptCellReader().read(lines)
    assert cell.cell_type == 'code'
    assert cell.source == '1+1'
    assert cell.metadata == {}
    assert pos == 1

def test_text_to_cell_py2():
    if False:
        for i in range(10):
            print('nop')
    text = 'def f(x):\n    return x+1'
    lines = text.splitlines()
    (cell, pos) = LightScriptCellReader().read(lines)
    assert cell.cell_type == 'code'
    assert cell.source == 'def f(x):\n    return x+1'
    assert cell.metadata == {}
    assert pos == 2

def test_code_to_cell():
    if False:
        for i in range(10):
            print('nop')
    text = 'def f(x):\n    return x+1'
    lines = text.splitlines()
    (cell, pos) = LightScriptCellReader().read(lines)
    assert cell.cell_type == 'code'
    assert cell.source == 'def f(x):\n    return x+1'
    assert cell.metadata == {}
    assert pos == 2

def test_uncomment_ocaml():
    if False:
        i = 10
        return i + 15
    assert uncomment(['(* ## *)'], '(*', '*)') == ['##']
    assert uncomment(['(*##*)'], '(*', '*)') == ['##']