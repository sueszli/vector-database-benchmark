import pytest
import jupytext
from jupytext.compare import compare_notebooks

def test_as_version_has_appropriate_type():
    if False:
        i = 10
        return i + 15
    with pytest.raises(TypeError):
        jupytext.read('script.py', 'py:percent')

def test_read_file_with_explicit_fmt(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_py = str(tmpdir.join('notebook.py'))
    with open(tmp_py, 'w') as fp:
        fp.write('# %%\n1 + 1\n')
    nb1 = jupytext.read(tmp_py)
    nb2 = jupytext.read(tmp_py, fmt='py:percent')
    compare_notebooks(nb2, nb1)