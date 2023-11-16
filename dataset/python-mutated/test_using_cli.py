import os
import shlex
import pytest
from nbformat.v4.nbbase import new_code_cell, new_notebook
import jupytext
from jupytext.cli import jupytext as jupytext_cli
from .utils import requires_black, requires_myst, requires_user_kernel_python3
doc_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '..', 'docs')

@requires_user_kernel_python3
@requires_black
@requires_myst
@pytest.mark.skipif(not os.path.isdir(doc_path), reason='Documentation folder is missing')
def test_jupytext_commands_in_the_documentation_work(tmpdir):
    if False:
        while True:
            i = 10
    using_cli = os.path.join(doc_path, 'using-cli.md')
    assert os.path.isfile(using_cli)
    using_cli_nb = jupytext.read(using_cli)
    jupytext.write(new_notebook(cells=[new_code_cell('1+1')]), str(tmpdir.join('notebook.ipynb')))
    os.chdir(str(tmpdir))
    cmd_tested = 0
    for cell in using_cli_nb.cells:
        if cell.cell_type != 'code':
            continue
        if not cell.source.startswith('jupytext'):
            continue
        for cmd in cell.source.splitlines():
            if not cmd.startswith('jupytext'):
                continue
            if 'read ipynb from stdin' in cmd:
                continue
            if 'pytest {}' in cmd:
                continue
            if '#' in cmd:
                (left, comment) = cmd.rsplit('#', 1)
                if '"' not in comment:
                    cmd = left
            print(f'Testing: {cmd}')
            args = shlex.split(cmd)[1:]
            assert not jupytext_cli(args), cmd
            cmd_tested += 1
    assert cmd_tested >= 10