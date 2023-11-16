import jupytext
from jupytext.compare import compare

def test_mark_cell_with_vim_folding_markers(script='# This is a markdown cell\n\n# {{{ And this is a foldable code region with metadata {"key": "value"}\na = 1\n\nb = 2\n\nc = 3\n# }}}\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(script, 'py')
    assert nb.metadata['jupytext']['cell_markers'] == '{{{,}}}'
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'This is a markdown cell'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'a = 1\n\nb = 2\n\nc = 3'
    assert nb.cells[1].metadata == {'title': 'And this is a foldable code region with metadata', 'key': 'value'}
    script2 = jupytext.writes(nb, 'py')
    compare(script2, script)

def test_mark_cell_with_vscode_pycharm_folding_markers(script='# This is a markdown cell\n\n# region And this is a foldable code region with metadata {"key": "value"}\na = 1\n\nb = 2\n\nc = 3\n# endregion\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(script, 'py')
    assert len(nb.cells) == 2
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'This is a markdown cell'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'a = 1\n\nb = 2\n\nc = 3'
    assert nb.cells[1].metadata == {'title': 'And this is a foldable code region with metadata', 'key': 'value'}
    script2 = jupytext.writes(nb, 'py')
    compare(script2, script)

def test_mark_cell_with_no_title_and_inner_region(script='# This is a markdown cell\n\n# region {"key": "value"}\na = 1\n\n# region An inner region\nb = 2\n# endregion\n\ndef f(x):\n    return x + 1\n\n\n# endregion\n\n\nd = 4\n'):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.reads(script, 'py')
    assert nb.cells[0].cell_type == 'markdown'
    assert nb.cells[0].source == 'This is a markdown cell'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == '# region {"key": "value"}\na = 1'
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[2].metadata['title'] == 'An inner region'
    assert nb.cells[2].source == 'b = 2'
    assert nb.cells[3].cell_type == 'code'
    assert nb.cells[3].source == 'def f(x):\n    return x + 1'
    assert nb.cells[4].cell_type == 'code'
    assert nb.cells[4].source == '# endregion'
    assert nb.cells[5].cell_type == 'code'
    assert nb.cells[5].source == 'd = 4'
    assert len(nb.cells) == 6
    script2 = jupytext.writes(nb, 'py')
    compare(script2, script)

def test_adjacent_regions(script='# region global\n# region innermost\na = 1\n\nb = 2\n# endregion\n# endregion\n'):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(script, 'py')
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == '# region global'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'a = 1\n\nb = 2'
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[2].source == '# endregion'
    script2 = jupytext.writes(nb, 'py')
    compare(script2, script)

def test_indented_markers_are_ignored(script='# region global\n    # region indented\na = 1\n\nb = 2\n    # endregion\n# endregion\n'):
    if False:
        print('Hello World!')
    nb = jupytext.reads(script, 'py')
    assert len(nb.cells) == 1
    assert nb.cells[0].cell_type == 'code'
    script2 = jupytext.writes(nb, 'py')
    compare(script2, script)