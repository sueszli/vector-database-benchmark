from nbformat.v4.nbbase import new_raw_cell
from jupytext import reads, writes
from jupytext.cli import jupytext

def test_set_cell_markers_cli(tmpdir, cwd_tmpdir):
    if False:
        print('Hello World!')
    tmpdir.join('test.py').write('# %% [markdown]\n# A Markdown cell\n')
    jupytext(['--format-options', 'cell_markers="""', 'test.py'])
    py = tmpdir.join('test.py').read()
    assert py.endswith('# %% [markdown]\n"""\nA Markdown cell\n"""\n')

def test_add_cell_to_script_with_cell_markers(no_jupytext_version_number, py='# ---\n# jupyter:\n#   jupytext:\n#     formats: py:percent\n#     cell_markers: \'"""\'\n# ---\n'):
    if False:
        i = 10
        return i + 15
    nb = reads(py, fmt='py:percent')
    nb.cells = [new_raw_cell('A raw cell')]
    py2 = writes(nb, fmt='py:percent')
    assert py2.endswith('# %% [raw]\n"""\nA raw cell\n"""\n')