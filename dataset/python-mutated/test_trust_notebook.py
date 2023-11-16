import os
import shutil
import pytest
from nbformat.v4.nbbase import new_code_cell, new_output
from jupytext.compare import compare_notebooks
from jupytext.contentsmanager import TextFileContentsManager
from .utils import list_notebooks, requires_myst

@pytest.mark.parametrize('nb_file', list_notebooks('python'))
def test_py_notebooks_are_trusted(nb_file):
    if False:
        for i in range(10):
            print('nop')
    cm = TextFileContentsManager()
    (root, file) = os.path.split(nb_file)
    cm.root_dir = root
    nb = cm.get(file)
    for cell in nb['content'].cells:
        assert cell.metadata.get('trusted', True)

@pytest.mark.parametrize('nb_file', list_notebooks('Rmd'))
def test_rmd_notebooks_are_trusted(nb_file):
    if False:
        for i in range(10):
            print('nop')
    cm = TextFileContentsManager()
    (root, file) = os.path.split(nb_file)
    cm.root_dir = root
    nb = cm.get(file)
    for cell in nb['content'].cells:
        assert cell.metadata.get('trusted', True)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py', skip='hash sign'))
def test_ipynb_notebooks_can_be_trusted(nb_file, tmpdir, no_jupytext_version_number):
    if False:
        print('Hello World!')
    cm = TextFileContentsManager()
    (root, file) = os.path.split(nb_file)
    tmp_ipynb = str(tmpdir.join(file))
    py_file = file.replace('.ipynb', '.py')
    tmp_py = str(tmpdir.join(py_file))
    shutil.copy(nb_file, tmp_ipynb)
    cm.formats = 'ipynb,py'
    cm.root_dir = str(tmpdir)
    model = cm.get(file)
    cm.save(model, py_file)
    nb = model['content']
    for cell in nb.cells:
        cell.metadata.pop('trusted', True)
    cm.notary.unsign(nb)
    model = cm.get(file)
    for cell in model['content'].cells:
        assert 'trusted' not in cell.metadata or not cell.metadata['trusted'] or (not cell.outputs)
    cm.trust_notebook(py_file)
    model = cm.get(file)
    for cell in model['content'].cells:
        assert cell.metadata.get('trusted', True)
    os.remove(tmp_py)
    nb2 = cm.get(file)
    for cell in nb2['content'].cells:
        assert cell.metadata.get('trusted', True)
    compare_notebooks(nb2['content'], model['content'])
    cm.trust_notebook(file)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py', skip='hash sign'))
def test_ipynb_notebooks_can_be_trusted_even_with_metadata_filter(nb_file, tmpdir, no_jupytext_version_number):
    if False:
        for i in range(10):
            print('nop')
    cm = TextFileContentsManager()
    (root, file) = os.path.split(nb_file)
    tmp_ipynb = str(tmpdir.join(file))
    py_file = file.replace('.ipynb', '.py')
    tmp_py = str(tmpdir.join(py_file))
    shutil.copy(nb_file, tmp_ipynb)
    cm.formats = 'ipynb,py'
    cm.notebook_metadata_filter = 'all'
    cm.cell_metadata_filter = '-all'
    cm.root_dir = str(tmpdir)
    model = cm.get(file)
    cm.save(model, py_file)
    nb = model['content']
    for cell in nb.cells:
        cell.metadata.pop('trusted', True)
    cm.notary.unsign(nb)
    cm.trust_notebook(py_file)
    model = cm.get(file)
    for cell in model['content'].cells:
        assert cell.metadata.get('trusted', True)
    os.remove(tmp_py)
    nb2 = cm.get(file)
    for cell in nb2['content'].cells:
        assert cell.metadata.get('trusted', True)
    compare_notebooks(nb2['content'], model['content'])

@pytest.mark.parametrize('nb_file', list_notebooks('percent', skip='hash sign'))
def test_text_notebooks_can_be_trusted(nb_file, tmpdir, no_jupytext_version_number):
    if False:
        i = 10
        return i + 15
    cm = TextFileContentsManager()
    (root, file) = os.path.split(nb_file)
    py_file = str(tmpdir.join(file))
    shutil.copy(nb_file, py_file)
    cm.root_dir = str(tmpdir)
    model = cm.get(file)
    model['type'] == 'notebook'
    cm.save(model, file)
    nb = model['content']
    for cell in nb.cells:
        cell.metadata.pop('trusted', True)
    cm.notary.unsign(nb)
    cm.trust_notebook(file)
    model = cm.get(file)
    for cell in model['content'].cells:
        assert cell.metadata.get('trusted', True)

def test_simple_notebook_is_trusted(tmpdir, python_notebook):
    if False:
        for i in range(10):
            print('nop')
    cm = TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = python_notebook
    cm.notary.unsign(nb)
    assert cm.notary.check_cells(nb)
    assert not cm.notary.check_signature(nb)
    cm.save(dict(type='notebook', content=nb), 'test.ipynb')
    nb = cm.get('test.ipynb')['content']
    assert cm.notary.check_signature(nb)

@requires_myst
def test_myst_notebook_is_trusted_941(tmp_path, myst='---\njupytext:\n  formats: md:myst\n  text_representation:\n    extension: .md\n    format_name: myst\n    format_version: 0.13\n    jupytext_version: 1.11.5\nkernelspec:\n  display_name: itables\n  language: python\n  name: itables\n---\n\n# Downsampling\n\n```{code-cell} ipython3\nfrom itables import init_notebook_mode, show\n\ninit_notebook_mode(all_interactive=True)\n```\n'):
    if False:
        for i in range(10):
            print('nop')
    cm = TextFileContentsManager()
    cm.root_dir = str(tmp_path)
    test_md = tmp_path / 'test.md'
    test_md.write_text(myst)
    nb = cm.get('test.md')['content']
    assert cm.notary.check_cells(nb)

@requires_myst
def test_paired_notebook_with_outputs_is_not_trusted_941(tmp_path, python_notebook):
    if False:
        while True:
            i = 10
    cm = TextFileContentsManager()
    cm.root_dir = str(tmp_path)
    nb = python_notebook
    nb.cells.append(new_code_cell(source='1+1', outputs=[new_output('execute_result')]))
    nb.metadata['jupytext'] = {'formats': 'ipynb,md:myst'}
    cm.notary.unsign(nb)
    cm.save(model=dict(type='notebook', content=nb), path='test.ipynb')
    nb = cm.get('test.md')['content']
    assert not cm.notary.check_cells(nb)
    assert not cm.notary.check_signature(nb)