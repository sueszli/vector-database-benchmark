import jupytext

def test_remove_encoding_907(tmp_path, python_notebook):
    if False:
        for i in range(10):
            print('nop')
    (tmp_path / 'jupytext.toml').write_text('formats="ipynb,py:percent"')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmp_path)
    cm.save(dict(type='notebook', content=python_notebook), path='nb.ipynb')
    py = (tmp_path / 'nb.py').read_text()
    assert 'coding' not in py
    py = '# -*- coding: utf-8 -*-\n' + py
    (tmp_path / 'nb.py').write_text(py)
    nb = cm.get('nb.ipynb')['content']
    assert 'encoding' in nb.metadata['jupytext']
    cm.save(dict(type='notebook', content=nb), path='nb.ipynb')
    py = (tmp_path / 'nb.py').read_text()
    assert py.startswith('# -*- coding: utf-8 -*-')
    py = '\n'.join(py.splitlines()[1:])
    (tmp_path / 'nb.py').write_text(py)
    nb = cm.get('nb.ipynb')['content']
    assert 'encoding' not in nb.metadata['jupytext']
    py = (tmp_path / 'nb.py').read_text()
    assert 'coding' not in py