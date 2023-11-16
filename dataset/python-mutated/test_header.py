from nbformat.v4.nbbase import new_markdown_cell, new_notebook, new_raw_cell
import jupytext
from jupytext.compare import compare
from jupytext.formats import get_format_implementation
from jupytext.header import header_to_metadata_and_cell, metadata_and_cell_to_header, uncomment_line

def test_uncomment():
    if False:
        print('Hello World!')
    assert uncomment_line('# line one', '#') == 'line one'
    assert uncomment_line('#line two', '#') == 'line two'
    assert uncomment_line('#line two', '') == '#line two'

def test_header_to_metadata_and_cell_blank_line():
    if False:
        while True:
            i = 10
    text = '---\ntitle: Sample header\n---\n\nHeader is followed by a blank line\n'
    lines = text.splitlines()
    (metadata, _, cell, pos) = header_to_metadata_and_cell(lines, '', '')
    assert metadata == {}
    assert cell.cell_type == 'raw'
    assert cell.source == '---\ntitle: Sample header\n---'
    assert cell.metadata == {}
    assert lines[pos].startswith('Header is')

def test_header_to_metadata_and_cell_no_blank_line():
    if False:
        print('Hello World!')
    text = '---\ntitle: Sample header\n---\nHeader is not followed by a blank line\n'
    lines = text.splitlines()
    (metadata, _, cell, pos) = header_to_metadata_and_cell(lines, '', '')
    assert metadata == {}
    assert cell.cell_type == 'raw'
    assert cell.source == '---\ntitle: Sample header\n---'
    assert cell.metadata == {'lines_to_next_cell': 0}
    assert lines[pos].startswith('Header is')

def test_header_to_metadata_and_cell_metadata():
    if False:
        while True:
            i = 10
    text = '---\ntitle: Sample header\njupyter:\n  mainlanguage: python\n---\n'
    lines = text.splitlines()
    (metadata, _, cell, pos) = header_to_metadata_and_cell(lines, '', '')
    assert metadata == {'mainlanguage': 'python'}
    assert cell.cell_type == 'raw'
    assert cell.source == '---\ntitle: Sample header\n---'
    assert cell.metadata == {'lines_to_next_cell': 0}
    assert pos == len(lines)

def test_metadata_and_cell_to_header(no_jupytext_version_number):
    if False:
        print('Hello World!')
    metadata = {'jupytext': {'mainlanguage': 'python'}}
    nb = new_notebook(metadata=metadata, cells=[new_raw_cell(source='---\ntitle: Sample header\n---')])
    (header, lines_to_next_cell) = metadata_and_cell_to_header(nb, metadata, get_format_implementation('.md'), {'extension': '.md'})
    assert '\n'.join(header) == '---\ntitle: Sample header\njupyter:\n  jupytext:\n    mainlanguage: python\n---'
    assert nb.cells == []
    assert lines_to_next_cell is None

def test_metadata_and_cell_to_header2(no_jupytext_version_number):
    if False:
        return 10
    nb = new_notebook(cells=[new_markdown_cell(source='Some markdown\ntext')])
    (header, lines_to_next_cell) = metadata_and_cell_to_header(nb, {}, get_format_implementation('.md'), {'extension': '.md'})
    assert header == []
    assert len(nb.cells) == 1
    assert lines_to_next_cell is None

def test_notebook_from_plain_script_has_metadata_filter(script='print(\'Hello world")\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(script, '.py')
    assert nb.metadata.get('jupytext', {}).get('notebook_metadata_filter') == '-all'
    assert nb.metadata.get('jupytext', {}).get('cell_metadata_filter') == '-all'
    script2 = jupytext.writes(nb, '.py')
    compare(script2, script)

def test_multiline_metadata(no_jupytext_version_number, notebook=new_notebook(metadata={'multiline': 'A multiline string\n\nwith a blank line', 'jupytext': {'notebook_metadata_filter': 'all'}}), markdown="---\njupyter:\n  jupytext:\n    notebook_metadata_filter: all\n  multiline: 'A multiline string\n\n\n    with a blank line'\n---\n"):
    if False:
        return 10
    actual = jupytext.writes(notebook, '.md')
    compare(actual, markdown)
    nb2 = jupytext.reads(markdown, '.md')
    compare(nb2, notebook)

def test_header_in_html_comment():
    if False:
        i = 10
        return i + 15
    text = '<!--\n\n---\njupyter:\n  title: Sample header\n---\n\n-->\n'
    lines = text.splitlines()
    (metadata, _, cell, _) = header_to_metadata_and_cell(lines, '', '')
    assert metadata == {'title': 'Sample header'}
    assert cell is None

def test_header_to_html_comment(no_jupytext_version_number):
    if False:
        for i in range(10):
            print('nop')
    metadata = {'jupytext': {'mainlanguage': 'python'}}
    nb = new_notebook(metadata=metadata, cells=[])
    (header, lines_to_next_cell) = metadata_and_cell_to_header(nb, metadata, get_format_implementation('.md'), {'extension': '.md', 'hide_notebook_metadata': True})
    compare('\n'.join(header), '<!--\n\n---\njupyter:\n  jupytext:\n    mainlanguage: python\n---\n\n-->')