import jupytext
from jupytext.compare import compare

def test_read_simple_file(julia='"""\n   cube(x)\n\nCompute the cube of `x`, ``x^3``.\n\n# Examples\n```jldoctest\njulia> cube(2)\n8\n```\n"""\nfunction cube(x)\n   x^3\nend\n\ncube(x)\n\n# And a markdown comment\n'):
    if False:
        while True:
            i = 10
    nb = jupytext.reads(julia, 'jl')
    assert nb.metadata['jupytext']['main_language'] == 'julia'
    assert len(nb.cells) == 3
    assert nb.cells[0].cell_type == 'code'
    assert nb.cells[0].source == '"""\n   cube(x)\n\nCompute the cube of `x`, ``x^3``.\n\n# Examples\n```jldoctest\njulia> cube(2)\n8\n```\n"""\nfunction cube(x)\n   x^3\nend'
    assert nb.cells[1].cell_type == 'code'
    assert nb.cells[1].source == 'cube(x)'
    assert nb.cells[2].cell_type == 'markdown'
    compare(nb.cells[2].source, 'And a markdown comment')
    julia2 = jupytext.writes(nb, 'jl')
    compare(julia2, julia)