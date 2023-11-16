import jupytext
from jupytext.compare import compare

def test_read_simple_file(script='# ---\n# title: Simple file\n# ---\n\n# %% [markdown]\n# This is a markdown cell\n\n# %% [raw]\n# This is a raw cell\n\n# %%% sub-cell title\n# This is a sub-cell\n\n# %%%% sub-sub-cell title\n# This is a sub-sub-cell\n\n# %% And now a code cell\n1 + 2 + 3 + 4\n5\n6\n%%pylab inline\n\n7\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py:hydrogen')
    assert len(nb.cells) == 6
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[0].source == '---\ntitle: Simple file\n---'
    assert nb.cells[1].cell_type == 'markdown'
    assert nb.cells[1].source == 'This is a markdown cell'
    assert nb.cells[2].cell_type == 'raw'
    assert nb.cells[2].source == 'This is a raw cell'
    assert nb.cells[3].cell_type == 'code'
    assert nb.cells[3].source == '# This is a sub-cell'
    assert nb.cells[3].metadata['title'] == 'sub-cell title'
    assert nb.cells[4].cell_type == 'code'
    assert nb.cells[4].source == '# This is a sub-sub-cell'
    assert nb.cells[4].metadata['title'] == 'sub-sub-cell title'
    assert nb.cells[5].cell_type == 'code'
    compare(nb.cells[5].source, '1 + 2 + 3 + 4\n5\n6\n%%pylab inline\n\n7')
    assert nb.cells[5].metadata == {'title': 'And now a code cell'}
    script2 = jupytext.writes(nb, 'py:hydrogen')
    compare(script2, script)

def test_read_cell_with_metadata(script='# %% a code cell with parameters {"tags": ["parameters"]}\na = 3\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py:hydrogen')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == 'a = 3'
    assert nb.cells[0].metadata == {'title': 'a code cell with parameters', 'tags': ['parameters']}
    script2 = jupytext.writes(nb, 'py:hydrogen')
    compare(script2, script)

def test_read_nbconvert_script(script='\n# coding: utf-8\n\n# A markdown cell\n\n# In[1]:\n\n\n%pylab inline\nimport pandas as pd\n\npd.options.display.max_rows = 6\npd.options.display.max_columns = 20\n\n\n# Another markdown cell\n\n# In[2]:\n\n\n1 + 1\n\n\n# Again, a markdown cell\n\n# In[33]:\n\n\n2 + 2\n\n\n# <codecell>\n\n\n3 + 3\n'):
    if False:
        return 10
    assert jupytext.formats.guess_format(script, '.py')[0] == 'hydrogen'
    nb = jupytext.reads(script, '.py')
    assert len(nb.cells) == 5

def test_read_remove_blank_lines(script='# %%\nimport pandas as pd\n\n# %% Display a data frame\ndf = pd.DataFrame({\'A\': [1, 2], \'B\': [3, 4]},\n                  index=pd.Index([\'x0\', \'x1\'], name=\'x\'))\ndf\n\n# %% Pandas plot {"tags": ["parameters"]}\ndf.plot(kind=\'bar\')\n\n\n# %% sample class\nclass MyClass:\n    pass\n\n\n# %% a function\ndef f(x):\n    return 42 * x\n\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(script, 'py')
    assert len(nb.cells) == 5
    for i in range(5):
        assert nb.cells[i].cell_type == 'code'
        assert not nb.cells[i].source.startswith('\n')
        assert not nb.cells[i].source.endswith('\n')
    script2 = jupytext.writes(nb, 'py:hydrogen')
    compare(script2, script)

def test_no_crash_on_square_bracket(script="# %% In [2]\nprint('Hello')\n"):
    if False:
        print('Hello World!')
    nb = jupytext.reads(script, 'py')
    script2 = jupytext.writes(nb, 'py:hydrogen')
    compare(script2, script)

def test_nbconvert_cell(script="# In[2]:\nprint('Hello')\n"):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(script, 'py')
    script2 = jupytext.writes(nb, 'py:hydrogen')
    expected = "# %%\nprint('Hello')\n"
    compare(script2, expected)

def test_nbformat_v3_nbpy_cell(script="# <codecell>\nprint('Hello')\n"):
    if False:
        return 10
    nb = jupytext.reads(script, 'py')
    script2 = jupytext.writes(nb, 'py:hydrogen')
    expected = "# %%\nprint('Hello')\n"
    compare(script2, expected)