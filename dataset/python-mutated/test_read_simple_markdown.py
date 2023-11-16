from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook, new_raw_cell
import jupytext
from jupytext.combine import combine_inputs_with_outputs
from jupytext.compare import compare, compare_cells, compare_notebooks

def test_read_mostly_py_markdown_file(markdown="---\ntitle: Simple file\n---\n\n```python\nimport numpy as np\nx = np.arange(0, 2*math.pi, eps)\n```\n\n```python\nx = np.arange(0,1,eps)\ny = np.abs(x)-.5\n```\n\nThis is\na Markdown cell\n\n```\n# followed by a code cell with no language info\n```\n\n```\n# another code cell\n\n\n# with two blank lines\n```\n\nAnd the same markdown cell continues\n\n<!-- #raw -->\nthis is a raw cell\n<!-- #endraw -->\n\n```R\nls()\n```\n\n```R\ncat(stringi::stri_rand_lipsum(3), sep='\n\n')\n```\n"):
    if False:
        print('Hello World!')
    nb = jupytext.reads(markdown, 'md')
    assert nb.metadata['jupytext']['main_language'] == 'python'
    compare_cells(nb.cells, [new_raw_cell('---\ntitle: Simple file\n---'), new_code_cell('import numpy as np\nx = np.arange(0, 2*math.pi, eps)'), new_code_cell('x = np.arange(0,1,eps)\ny = np.abs(x)-.5'), new_markdown_cell('This is\na Markdown cell\n\n```\n# followed by a code cell with no language info\n```\n\n```\n# another code cell\n\n\n# with two blank lines\n```\n\nAnd the same markdown cell continues'), new_raw_cell('this is a raw cell'), new_code_cell('%%R\nls()'), new_code_cell("%%R\ncat(stringi::stri_rand_lipsum(3), sep='\n\n')")], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_read_md_and_markdown_regions(markdown='Some text\n\n<!-- #md -->\nA\n\n\nlong\ncell\n<!-- #endmd -->\n\n<!-- #markdown -->\nAnother\n\n\nlong\ncell\n<!-- #endmarkdown -->\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(markdown, 'md')
    assert nb.metadata['jupytext']['main_language'] == 'python'
    compare_cells(nb.cells, [new_markdown_cell('Some text'), new_markdown_cell('A\n\n\nlong\ncell', metadata={'region_name': 'md'}), new_markdown_cell('Another\n\n\nlong\ncell', metadata={'region_name': 'markdown'})], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_read_mostly_R_markdown_file(markdown="```R\nls()\n```\n\n```R\ncat(stringi::stri_rand_lipsum(3), sep='\n\n')\n```\n"):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(markdown, 'md')
    assert nb.metadata['jupytext']['main_language'] == 'R'
    compare_cells(nb.cells, [new_code_cell('ls()'), new_code_cell("cat(stringi::stri_rand_lipsum(3), sep='\n\n')")], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_read_markdown_file_no_language(markdown="```\nls\n```\n\n```\necho 'Hello World'\n```\n"):
    if False:
        return 10
    nb = jupytext.reads(markdown, 'md')
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_read_julia_notebook(markdown='```julia\n1 + 1\n```\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(markdown, 'md')
    assert nb.metadata['jupytext']['main_language'] == 'julia'
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'code'
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_split_on_header(markdown='A paragraph\n\n# H1 Header\n\n## H2 Header\n\nAnother paragraph\n'):
    if False:
        print('Hello World!')
    fmt = {'extension': '.md', 'split_at_heading': True}
    nb = jupytext.reads(markdown, fmt)
    assert nb.cells[0].source == 'A paragraph'
    assert nb.cells[1].source == '# H1 Header'
    assert nb.cells[2].source == '## H2 Header\n\nAnother paragraph'
    assert len(nb.cells) == 3
    markdown2 = jupytext.writes(nb, fmt)
    compare(markdown2, markdown)

def test_split_on_header_after_two_blank_lines(markdown='A paragraph\n\n\n# H1 Header\n'):
    if False:
        i = 10
        return i + 15
    fmt = {'extension': '.Rmd', 'split_at_heading': True}
    nb = jupytext.reads(markdown, fmt)
    markdown2 = jupytext.writes(nb, fmt)
    compare(markdown2, markdown)

def test_split_at_heading_in_metadata(markdown='---\njupyter:\n  jupytext:\n    split_at_heading: true\n---\n\nA paragraph\n\n# H1 Header\n', nb_expected=new_notebook(cells=[new_markdown_cell('A paragraph'), new_markdown_cell('# H1 Header')])):
    if False:
        return 10
    nb = jupytext.reads(markdown, '.md')
    compare_notebooks(nb, nb_expected)

def test_code_cell_with_metadata(markdown='```python tags=["parameters"]\na = 1\nb = 2\n```\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_code_cell(source='a = 1\nb = 2', metadata={'tags': ['parameters']})], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_raw_cell_with_metadata_json(markdown='<!-- #raw {"key": "value"} -->\nraw content\n<!-- #endraw -->\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_raw_cell(source='raw content', metadata={'key': 'value'})], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_raw_cell_with_metadata(markdown='<!-- #raw key="value" -->\nraw content\n<!-- #endraw -->\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_raw_cell(source='raw content', metadata={'key': 'value'})], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_read_raw_cell_markdown_version_1_1(markdown='---\njupyter:\n  jupytext:\n    text_representation:\n      extension: .md\n      format_name: markdown\n      format_version: \'1.1\'\n      jupytext_version: 1.1.0\n---\n\n```key="value"\nraw content\n```\n'):
    if False:
        return 10
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_raw_cell(source='raw content', metadata={'key': 'value'})], compare_ids=False)
    md2 = jupytext.writes(nb, 'md')
    assert "format_version: '1.1'" not in md2

def test_read_raw_cell_markdown_version_1_1_with_mimetype(header="---\njupyter:\n  jupytext:\n    text_representation:\n      extension: .md\n      format_name: markdown\n      format_version: '1.1'\n      jupytext_version: 1.1.0-rc0\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n", markdown_11='```raw_mimetype="text/restructuredtext"\n.. meta::\n   :description: Topic: Integrated Development Environments, Difficulty: Easy, Category: Tools\n   :keywords: python, introduction, IDE, PyCharm, VSCode, Jupyter, recommendation, tools\n```\n', markdown_12='<!-- #raw raw_mimetype="text/restructuredtext" -->\n.. meta::\n   :description: Topic: Integrated Development Environments, Difficulty: Easy, Category: Tools\n   :keywords: python, introduction, IDE, PyCharm, VSCode, Jupyter, recommendation, tools\n<!-- #endraw -->\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(header + '\n' + markdown_11, 'md')
    compare_cells(nb.cells, [new_raw_cell(source='.. meta::\n   :description: Topic: Integrated Development Environments, Difficulty: Easy, Category: Tools\n   :keywords: python, introduction, IDE, PyCharm, VSCode, Jupyter, recommendation, tools', metadata={'raw_mimetype': 'text/restructuredtext'})], compare_ids=False)
    md2 = jupytext.writes(nb, 'md')
    assert "format_version: '1.1'" not in md2
    nb.metadata['jupytext']['notebook_metadata_filter'] = '-all'
    md2 = jupytext.writes(nb, 'md')
    compare(md2, markdown_12)

def test_markdown_cell_with_metadata_json(markdown='<!-- #region {"key": "value"} -->\nA long\n\n\nmarkdown cell\n<!-- #endregion -->\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_markdown_cell(source='A long\n\n\nmarkdown cell', metadata={'key': 'value'})], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_markdown_cell_with_metadata(markdown='<!-- #region key="value" -->\nA long\n\n\nmarkdown cell\n<!-- #endregion -->\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_markdown_cell(source='A long\n\n\nmarkdown cell', metadata={'key': 'value'})], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_two_markdown_cells(markdown='# A header\n\n<!-- #region -->\nA long\n\n\nmarkdown cell\n<!-- #endregion -->\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(markdown, 'md')
    compare_cells(nb.cells, [new_markdown_cell(source='# A header'), new_markdown_cell(source='A long\n\n\nmarkdown cell')], compare_ids=False)
    markdown2 = jupytext.writes(nb, 'md')
    compare(markdown2, markdown)

def test_combine_md_version_one():
    if False:
        while True:
            i = 10
    markdown = "---\njupyter:\n  jupytext:\n    text_representation:\n      extension: .md\n      format_name: markdown\n      format_version: '1.0'\n      jupytext_version: 1.0.0\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n\nA short markdown cell\n\n```\na raw cell\n```\n\n```python\n1 + 1\n```\n"
    nb_source = jupytext.reads(markdown, 'md')
    nb_meta = jupytext.reads(markdown, 'md')
    for cell in nb_meta.cells:
        cell.metadata = {'key': 'value'}
    nb_source = combine_inputs_with_outputs(nb_source, nb_meta)
    for cell in nb_source.cells:
        assert cell.metadata == {'key': 'value'}, cell.source

def test_jupyter_cell_is_not_split():
    if False:
        return 10
    text = 'Here we have a markdown\nfile with a jupyter code cell\n\n```python\n1 + 1\n\n\n2 + 2\n```\n\nthe code cell should become a Jupyter cell.\n'
    nb = jupytext.reads(text, 'md')
    assert nb.cells[0].cell_type == 'markdown'
    compare(nb.cells[0].source, 'Here we have a markdown\nfile with a jupyter code cell')
    assert nb.cells[1].cell_type == 'code'
    compare(nb.cells[1].source, '1 + 1\n\n\n2 + 2')
    assert nb.cells[2].cell_type == 'markdown'
    compare(nb.cells[2].source, 'the code cell should become a Jupyter cell.')
    assert len(nb.cells) == 3

def test_indented_code_is_not_split():
    if False:
        while True:
            i = 10
    text = 'Here we have a markdown\nfile with an indented code cell\n\n    1 + 1\n\n\n    2 + 2\n\nthe code cell should not become a Jupyter cell,\nnor be split into two pieces.'
    nb = jupytext.reads(text, 'md')
    compare(nb.cells[0].source, text)
    assert nb.cells[0].cell_type == 'markdown'
    assert len(nb.cells) == 1

def test_non_jupyter_code_is_not_split():
    if False:
        for i in range(10):
            print('nop')
    text = 'Here we have a markdown\nfile with a non-jupyter code cell\n\n```{.python}\n1 + 1\n\n\n2 + 2\n```\n\nthe code cell should not become a Jupyter cell,\nnor be split into two pieces.'
    nb = jupytext.reads(text, 'md')
    compare(nb.cells[0].source, text)
    assert nb.cells[0].cell_type == 'markdown'
    assert len(nb.cells) == 1

def test_read_markdown_idl(no_jupytext_version_number, text='---\njupyter:\n  kernelspec:\n    display_name: IDL [conda env:gdl] *\n    language: IDL\n    name: conda-env-gdl-idl\n---\n\n# A sample IDL Markdown notebook\n\n```idl\na = 1\n```\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(text, 'md')
    assert len(nb.cells) == 2
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'a = 1'
    text2 = jupytext.writes(nb, 'md')
    compare(text2, text)

def test_read_markdown_IDL(no_jupytext_version_number, text='---\njupyter:\n  kernelspec:\n    display_name: IDL [conda env:gdl] *\n    language: IDL\n    name: conda-env-gdl-idl\n---\n\n# A sample IDL Markdown notebook\n\n```IDL\na = 1\n```\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(text, 'md')
    assert len(nb.cells) == 2
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'a = 1'
    text2 = jupytext.writes(nb, 'md')
    compare(text2, text)

def test_inactive_cell(text='```python active="md"\n# This becomes a raw cell in Jupyter\n```\n', expected=new_notebook(cells=[new_raw_cell('# This becomes a raw cell in Jupyter', metadata={'active': 'md'})])):
    if False:
        print('Hello World!')
    nb = jupytext.reads(text, 'md')
    compare_notebooks(nb, expected)
    text2 = jupytext.writes(nb, 'md')
    compare(text2, text)

def test_inactive_cell_using_tag(text='```python tags=["active-md"]\n# This becomes a raw cell in Jupyter\n```\n', expected=new_notebook(cells=[new_raw_cell('# This becomes a raw cell in Jupyter', metadata={'tags': ['active-md']})])):
    if False:
        return 10
    nb = jupytext.reads(text, 'md')
    compare_notebooks(nb, expected)
    text2 = jupytext.writes(nb, 'md')
    compare(text2, text)

def test_inactive_cell_using_noeval(text='This is text\n\n```python .noeval\n# This is python code.\n# It should not become a code cell\n```\n'):
    if False:
        print('Hello World!')
    expected = new_notebook(cells=[new_markdown_cell(text[:-1])])
    nb = jupytext.reads(text, 'md')
    compare_notebooks(nb, expected)
    text2 = jupytext.writes(nb, 'md')
    compare(text2, text)

def test_noeval_followed_by_code_works(text='```python .noeval\n# Not a code cell in Jupyter\n```\n\n```python\n1 + 1\n```\n', expected=new_notebook(cells=[new_markdown_cell('```python .noeval\n# Not a code cell in Jupyter\n```'), new_code_cell('1 + 1')])):
    if False:
        return 10
    nb = jupytext.reads(text, 'md')
    compare_notebooks(nb, expected)
    text2 = jupytext.writes(nb, 'md')
    compare(text2, text)

def test_markdown_cell_with_code_works(nb=new_notebook(cells=[new_markdown_cell('```python\n1 + 1\n```')])):
    if False:
        return 10
    text = jupytext.writes(nb, 'md')
    nb2 = jupytext.reads(text, 'md')
    compare_notebooks(nb2, nb)

def test_markdown_cell_with_noeval_code_works(nb=new_notebook(cells=[new_markdown_cell('```python .noeval\n1 + 1\n```')])):
    if False:
        while True:
            i = 10
    text = jupytext.writes(nb, 'md')
    nb2 = jupytext.reads(text, 'md')
    compare_notebooks(nb2, nb)

def test_two_markdown_cell_with_code_works(nb=new_notebook(cells=[new_markdown_cell('```python\n1 + 1\n```'), new_markdown_cell('```python\n2 + 2\n```')])):
    if False:
        print('Hello World!')
    text = jupytext.writes(nb, 'md')
    nb2 = jupytext.reads(text, 'md')
    compare_notebooks(nb2, nb)

def test_two_markdown_cell_with_no_language_code_works(nb=new_notebook(cells=[new_markdown_cell('```\n1 + 1\n```'), new_markdown_cell('```\n2 + 2\n```')])):
    if False:
        for i in range(10):
            print('nop')
    text = jupytext.writes(nb, 'md')
    nb2 = jupytext.reads(text, 'md')
    compare_notebooks(nb2, nb)

def test_markdown_cell_with_code_inside_multiline_string_419(text='```python\nreadme = """\nabove\n\n```python\nx = 2\n```\n\nbelow\n"""\n```\n'):
    if False:
        print('Hello World!')
    'A code cell containing triple backticks is converted to a code cell encapsulated with four backticks'
    nb = jupytext.reads(text, 'md')
    compare(jupytext.writes(nb, 'md'), '`' + text[:-1] + '`\n')
    assert len(nb.cells) == 1

def test_notebook_with_python3_magic(no_jupytext_version_number, nb=new_notebook(metadata={'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}, cells=[new_code_cell('%%python2\na = 1\nprint a'), new_code_cell('%%python3\nb = 2\nprint(b)')]), text='---\njupyter:\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n\n```python2\na = 1\nprint a\n```\n\n```python3\nb = 2\nprint(b)\n```\n'):
    if False:
        print('Hello World!')
    md = jupytext.writes(nb, 'md')
    compare(md, text)
    nb2 = jupytext.reads(md, 'md')
    compare_notebooks(nb2, nb)

def test_update_metadata_filter(no_jupytext_version_number, org='---\njupyter:\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n  extra:\n    key: value\n---\n', target='---\njupyter:\n  extra:\n    key: value\n  jupytext:\n    notebook_metadata_filter: extra\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(org, 'md')
    text = jupytext.writes(nb, 'md')
    compare(text, target)

def test_update_metadata_filter_2(no_jupytext_version_number, org='---\njupyter:\n  jupytext:\n    notebook_metadata_filter: -extra\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n  extra:\n    key: value\n---\n', target='---\njupyter:\n  jupytext:\n    notebook_metadata_filter: -extra\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(org, 'md')
    text = jupytext.writes(nb, 'md')
    compare(text, target)

def test_custom_metadata(no_jupytext_version_number, nb=new_notebook(metadata={'author': 'John Doe', 'title': 'Some serious math', 'jupytext': {'notebook_metadata_filter': 'title,author'}, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}), md='---\njupyter:\n  author: John Doe\n  jupytext:\n    notebook_metadata_filter: title,author\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n  title: Some serious math\n---\n'):
    if False:
        i = 10
        return i + 15
    'Here we test the addition of custom metadata, cf. https://github.com/mwouts/jupytext/issues/469'
    md2 = jupytext.writes(nb, 'md')
    compare(md2, md)
    nb2 = jupytext.reads(md, 'md')
    compare_notebooks(nb2, nb)

def test_hide_notebook_metadata(no_jupytext_version_number, nb=new_notebook(metadata={'jupytext': {'hide_notebook_metadata': True}, 'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}}), md='<!--\n\n---\njupyter:\n  jupytext:\n    hide_notebook_metadata: true\n  kernelspec:\n    display_name: Python 3\n    language: python\n    name: python3\n---\n\n-->\n'):
    if False:
        for i in range(10):
            print('nop')
    'Test the hide_notebook_metadata option'
    md2 = jupytext.writes(nb, 'md')
    compare(md2, md)
    nb2 = jupytext.reads(md, 'md')
    compare_notebooks(nb2, nb)

def test_notebook_with_empty_header_1070(md='---\n\n---\n\nThis file has empty frontmatter.\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(md, fmt='md:markdown')
    md2 = jupytext.writes(nb, 'md')
    compare(md2, md)
    nb2 = jupytext.reads(md, 'md')
    compare_notebooks(nb2, nb)