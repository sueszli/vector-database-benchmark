import itertools
import logging
import os
import re
import shutil
import time
import pytest
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook
from tornado.web import HTTPError
import jupytext
from jupytext.cli import jupytext as jupytext_cli
from jupytext.compare import compare, compare_cells, compare_notebooks
from jupytext.formats import auto_ext_from_metadata, read_format_from_metadata
from jupytext.header import header_to_metadata_and_cell
from jupytext.jupytext import read, write, writes
from jupytext.kernels import kernelspec_from_language
from .utils import list_notebooks, notebook_model, requires_pandoc, requires_quarto, requires_sphinx_gallery, requires_user_kernel_python3

def test_create_contentsmanager():
    if False:
        i = 10
        return i + 15
    jupytext.TextFileContentsManager()

def test_rename(tmpdir):
    if False:
        while True:
            i = 10
    org_file = str(tmpdir.join('notebook.ipynb'))
    new_file = str(tmpdir.join('new.ipynb'))
    jupytext.write(new_notebook(), org_file)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.rename_file('notebook.ipynb', 'new.ipynb')
    assert os.path.isfile(new_file)
    assert not os.path.isfile(org_file)

def test_rename_inconsistent_path(tmpdir):
    if False:
        print('Hello World!')
    org_file = str(tmpdir.join('notebook_suffix.ipynb'))
    new_file = str(tmpdir.join('new.ipynb'))
    jupytext.write(new_notebook(metadata={'jupytext': {'formats': '_suffix.ipynb'}}), org_file)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.get('notebook_suffix.ipynb')
    with pytest.raises(HTTPError):
        cm.rename_file('notebook_suffix.ipynb', 'new.ipynb')
    assert not os.path.isfile(new_file)
    assert os.path.isfile(org_file)

def test_pair_unpair_notebook(tmpdir):
    if False:
        return 10
    tmp_ipynb = 'notebook.ipynb'
    tmp_md = 'notebook.md'
    nb = new_notebook(metadata={'kernelspec': {'display_name': 'Python3', 'language': 'python', 'name': 'python3'}}, cells=[new_code_cell('1 + 1', outputs=[{'data': {'text/plain': ['2']}, 'execution_count': 1, 'metadata': {}, 'output_type': 'execute_result'}])])
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    assert not os.path.isfile(str(tmpdir.join(tmp_md)))
    nb['metadata']['jupytext'] = {'formats': 'ipynb,md'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    assert os.path.isfile(str(tmpdir.join(tmp_md)))
    nb2 = cm.get(tmp_md)['content']
    compare_notebooks(nb, nb2)
    del nb['metadata']['jupytext']
    cm.save(model=notebook_model(nb), path=tmp_md)
    nb2 = cm.get(tmp_md)['content']
    compare_notebooks(nb, nb2, compare_outputs=False)
    assert len(nb2.cells[0]['outputs']) == 0

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb', skip='66'))
def test_load_save_rename(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = 'notebook.ipynb'
    tmp_rmd = 'notebook.Rmd'
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,Rmd'
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    cm.save(model=notebook_model(nb), path=tmp_rmd)
    nb_rmd = cm.get(tmp_rmd)
    compare_notebooks(nb_rmd['content'], nb, 'Rmd')
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.rename(tmp_ipynb, 'new.ipynb')
    assert not os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    assert not os.path.isfile(str(tmpdir.join(tmp_rmd)))
    assert os.path.isfile(str(tmpdir.join('new.ipynb')))
    assert os.path.isfile(str(tmpdir.join('new.Rmd')))
    cm.delete('new.Rmd')
    assert not os.path.isfile(str(tmpdir.join('new.Rmd')))
    model = cm.get('new.ipynb', content=False)
    assert 'last_modified' in model
    cm.save(model=notebook_model(nb), path='new.ipynb')
    assert os.path.isfile(str(tmpdir.join('new.Rmd')))
    cm.delete('new.Rmd')
    cm.rename('new.ipynb', tmp_ipynb)
    assert os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    assert not os.path.isfile(str(tmpdir.join(tmp_rmd)))
    assert not os.path.isfile(str(tmpdir.join('new.ipynb')))
    assert not os.path.isfile(str(tmpdir.join('new.Rmd')))

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb', skip='magic'))
def test_save_load_paired_md_notebook(nb_file, tmpdir):
    if False:
        while True:
            i = 10
    tmp_ipynb = 'notebook.ipynb'
    tmp_md = 'notebook.md'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb.metadata['jupytext'] = {'formats': 'ipynb,md'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    nb_md = cm.get(tmp_md)
    compare_notebooks(nb_md['content'], nb, 'md')
    assert nb_md['content'].metadata['jupytext']['formats'] == 'ipynb,md'

@requires_pandoc
@pytest.mark.parametrize('nb_file', list_notebooks('ipynb', skip='(functional|Notebook with|flavors|invalid|305)'))
def test_save_load_paired_md_pandoc_notebook(nb_file, tmpdir):
    if False:
        i = 10
        return i + 15
    tmp_ipynb = 'notebook.ipynb'
    tmp_md = 'notebook.md'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb.metadata['jupytext'] = {'formats': 'ipynb,md:pandoc'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    nb_md = cm.get(tmp_md)
    compare_notebooks(nb_md['content'], nb, 'md:pandoc')
    assert nb_md['content'].metadata['jupytext']['formats'] == 'ipynb,md:pandoc'

@requires_quarto
@pytest.mark.parametrize('nb_file', list_notebooks('ipynb', skip='(World|functional|Notebook with|plotly_graphs|flavors|complex_metadata|update83|raw_cell|_66|nteract|LaTeX|invalid|305|text_outputs|ir_notebook|jupyter|with_R_magic)'))
def test_save_load_paired_qmd_notebook(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = 'notebook.ipynb'
    tmp_qmd = 'notebook.qmd'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb.metadata['jupytext'] = {'formats': 'ipynb,qmd'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    nb_md = cm.get(tmp_qmd)
    compare_notebooks(nb_md['content'], nb, 'qmd')
    assert nb_md['content'].metadata['jupytext']['formats'] == 'ipynb,qmd'

@pytest.mark.parametrize('py_file', list_notebooks('percent'))
def test_pair_plain_script(py_file, tmpdir, caplog):
    if False:
        return 10
    tmp_py = 'notebook.py'
    tmp_ipynb = 'notebook.ipynb'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(py_file)
    nb.metadata['jupytext']['formats'] = 'ipynb,py:hydrogen'
    cm.save(model=notebook_model(nb), path=tmp_py)
    assert "'Include Metadata' is off" in caplog.text
    assert os.path.isfile(str(tmpdir.join(tmp_py)))
    assert os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    with open(py_file) as fp:
        script = fp.read()
    with open(str(tmpdir.join(tmp_py))) as fp:
        script2 = fp.read()
    compare(script2, script)
    nb2 = cm.get(tmp_py)['content']
    compare_notebooks(nb2, nb)
    assert nb2.metadata['jupytext']['formats'] == 'ipynb,py:hydrogen'
    del nb.metadata['jupytext']['formats']
    cm.save(model=notebook_model(nb), path=tmp_py)
    nb2 = cm.get(tmp_py)['content']
    compare_notebooks(nb2, nb)
    assert 'formats' not in nb2.metadata['jupytext']

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_load_save_rename_nbpy(nb_file, tmpdir):
    if False:
        return 10
    tmp_ipynb = 'notebook.ipynb'
    tmp_nbpy = 'notebook.nb.py'
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,.nb.py'
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    cm.save(model=notebook_model(nb), path=tmp_nbpy)
    nbpy = cm.get(tmp_nbpy)
    compare_notebooks(nbpy['content'], nb)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.rename(tmp_nbpy, 'new.nb.py')
    assert not os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    assert not os.path.isfile(str(tmpdir.join(tmp_nbpy)))
    assert os.path.isfile(str(tmpdir.join('new.ipynb')))
    assert os.path.isfile(str(tmpdir.join('new.nb.py')))
    with pytest.raises(HTTPError):
        cm.rename_file(tmp_nbpy, 'suffix_missing.py')

@pytest.mark.parametrize('script', list_notebooks('python', skip='light'))
def test_load_save_py_freeze_metadata(script, tmpdir):
    if False:
        print('Hello World!')
    tmp_nbpy = 'notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    with open(script) as fp:
        text_py = fp.read()
    with open(str(tmpdir.join(tmp_nbpy)), 'w') as fp:
        fp.write(text_py)
    nb = cm.get(tmp_nbpy)['content']
    cm.save(model=notebook_model(nb), path=tmp_nbpy)
    with open(str(tmpdir.join(tmp_nbpy))) as fp:
        text_py2 = fp.read()
    compare(text_py2, text_py)

def test_load_text_notebook(tmpdir):
    if False:
        while True:
            i = 10
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nbpy = 'text.py'
    with open(str(tmpdir.join(nbpy)), 'w') as fp:
        fp.write('# %%\n1 + 1\n')
    py_model = cm.get(nbpy, content=False)
    assert py_model['type'] == 'notebook'
    assert py_model['content'] is None
    py_model = cm.get(nbpy, content=True)
    assert py_model['type'] == 'notebook'
    assert 'cells' in py_model['content']
    nb_model = dict(type='notebook', content=new_notebook(cells=[new_markdown_cell('A cell')]))
    cm.save(nb_model, 'notebook.ipynb')
    nb_model = cm.get('notebook.ipynb', content=True)
    for key in ['format', 'mimetype', 'type']:
        assert nb_model[key] == py_model[key], key

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_load_save_rename_notebook_with_dot(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = '1.notebook.ipynb'
    tmp_nbpy = '1.notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,py'
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    cm.save(model=notebook_model(nb), path=tmp_nbpy)
    nbpy = cm.get(tmp_nbpy)
    compare_notebooks(nbpy['content'], nb)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.rename(tmp_nbpy, '2.new_notebook.py')
    assert not os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    assert not os.path.isfile(str(tmpdir.join(tmp_nbpy)))
    assert os.path.isfile(str(tmpdir.join('2.new_notebook.ipynb')))
    assert os.path.isfile(str(tmpdir.join('2.new_notebook.py')))

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_load_save_rename_nbpy_default_config(nb_file, tmpdir):
    if False:
        while True:
            i = 10
    tmp_ipynb = 'notebook.ipynb'
    tmp_nbpy = 'notebook.nb.py'
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,.nb.py'
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    cm.save(model=notebook_model(nb), path=tmp_nbpy)
    nbpy = cm.get(tmp_nbpy)
    compare_notebooks(nbpy['content'], nb)
    nbipynb = cm.get(tmp_ipynb)
    compare_notebooks(nbipynb['content'], nb)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.rename(tmp_nbpy, 'new.nb.py')
    assert not os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    assert not os.path.isfile(str(tmpdir.join(tmp_nbpy)))
    assert os.path.isfile(str(tmpdir.join('new.ipynb')))
    assert os.path.isfile(str(tmpdir.join('new.nb.py')))
    cm.rename('new.ipynb', tmp_ipynb)
    assert os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    assert os.path.isfile(str(tmpdir.join(tmp_nbpy)))
    assert not os.path.isfile(str(tmpdir.join('new.ipynb')))
    assert not os.path.isfile(str(tmpdir.join('new.nb.py')))

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_load_save_rename_non_ascii_path(nb_file, tmpdir):
    if False:
        while True:
            i = 10
    tmp_ipynb = 'notebôk.ipynb'
    tmp_nbpy = 'notebôk.nb.py'
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,.nb.py'
    tmpdir = '' + str(tmpdir)
    cm.root_dir = tmpdir
    nb = jupytext.read(nb_file)
    cm.save(model=notebook_model(nb), path=tmp_nbpy)
    nbpy = cm.get(tmp_nbpy)
    compare_notebooks(nbpy['content'], nb)
    nbipynb = cm.get(tmp_ipynb)
    compare_notebooks(nbipynb['content'], nb)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.rename(tmp_nbpy, 'nêw.nb.py')
    assert not os.path.isfile(os.path.join(tmpdir, tmp_ipynb))
    assert not os.path.isfile(os.path.join(tmpdir, tmp_nbpy))
    assert os.path.isfile(os.path.join(tmpdir, 'nêw.ipynb'))
    assert os.path.isfile(os.path.join(tmpdir, 'nêw.nb.py'))
    cm.rename('nêw.ipynb', tmp_ipynb)
    assert os.path.isfile(os.path.join(tmpdir, tmp_ipynb))
    assert os.path.isfile(os.path.join(tmpdir, tmp_nbpy))
    assert not os.path.isfile(os.path.join(tmpdir, 'nêw.ipynb'))
    assert not os.path.isfile(os.path.join(tmpdir, 'nêw.nb.py'))

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_outdated_text_notebook(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'py,ipynb'
    cm.outdated_text_notebook_margin = 0
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    cm.save(model=notebook_model(nb), path='notebook.py')
    model_py = cm.get('notebook.py', load_alternative_format=False)
    model_ipynb = cm.get('notebook.ipynb', load_alternative_format=False)
    assert model_ipynb['last_modified'] <= model_py['last_modified']
    time.sleep(0.5)
    nb.cells.append(new_markdown_cell('New cell'))
    write(nb, str(tmpdir.join('notebook.ipynb')))
    with pytest.raises(HTTPError):
        cm.get('notebook.py')
    cm.outdated_text_notebook_margin = 1.0
    cm.get('notebook.py')
    cm.outdated_text_notebook_margin = float('inf')
    cm.get('notebook.py')

def test_outdated_text_notebook_no_diff_ok(tmpdir, python_notebook):
    if False:
        for i in range(10):
            print('nop')
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'py,ipynb'
    cm.outdated_text_notebook_margin = 0
    cm.root_dir = str(tmpdir)
    nb = python_notebook
    cm.save(model=notebook_model(nb), path='notebook.py')
    model_py = cm.get('notebook.py', load_alternative_format=False)
    model_ipynb = cm.get('notebook.ipynb', load_alternative_format=False)
    assert model_ipynb['last_modified'] <= model_py['last_modified']
    time.sleep(0.5)
    with open(tmpdir / 'notebook.ipynb', 'a'):
        os.utime(tmpdir / 'notebook.ipynb', None)
    cm.get('notebook.py')

def test_outdated_text_notebook_diff_is_shown(tmpdir, python_notebook):
    if False:
        i = 10
        return i + 15
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'py,ipynb'
    cm.outdated_text_notebook_margin = 0
    cm.root_dir = str(tmpdir)
    nb = python_notebook
    nb.cells = [new_markdown_cell('Text version 1.0')]
    cm.save(model=notebook_model(nb), path='notebook.py')
    model_py = cm.get('notebook.py', load_alternative_format=False)
    model_ipynb = cm.get('notebook.ipynb', load_alternative_format=False)
    assert model_ipynb['last_modified'] <= model_py['last_modified']
    time.sleep(0.5)
    nb.cells = [new_markdown_cell('Text version 2.0')]
    jupytext.write(nb, str(tmpdir / 'notebook.ipynb'))
    with pytest.raises(HTTPError) as excinfo:
        cm.get('notebook.py')
    diff = excinfo.value.log_message
    diff = diff[diff.find('Differences'):diff.rfind('Please')]
    compare(diff.replace('\n \n', '\n\n'), 'Differences (jupytext --diff notebook.py notebook.ipynb) are:\n--- notebook.py\n+++ notebook.ipynb\n@@ -12,5 +12,5 @@\n #     name: python_kernel\n # ---\n\n-# Text version 1.0\n+# Text version 2.0\n\n')

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_reload_notebook_after_jupytext_cli(nb_file, tmpdir):
    if False:
        print('Hello World!')
    tmp_ipynb = str(tmpdir.join('notebook.ipynb'))
    tmp_nbpy = str(tmpdir.join('notebook.py'))
    cm = jupytext.TextFileContentsManager()
    cm.outdated_text_notebook_margin = 0
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb.metadata.setdefault('jupytext', {})['formats'] = 'py,ipynb'
    cm.save(model=notebook_model(nb), path='notebook.py')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_nbpy)
    jupytext_cli([tmp_nbpy, '--to', 'ipynb', '--update'])
    nb1 = cm.get('notebook.py')['content']
    nb2 = cm.get('notebook.ipynb')['content']
    compare_notebooks(nb, nb1)
    compare_notebooks(nb, nb2)

@pytest.mark.parametrize('nb_file', list_notebooks('percent'))
def test_load_save_percent_format(nb_file, tmpdir):
    if False:
        while True:
            i = 10
    tmp_py = 'notebook.py'
    with open(nb_file) as stream:
        text_py = stream.read()
    with open(str(tmpdir.join(tmp_py)), 'w') as stream:
        stream.write(text_py)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = cm.get(tmp_py)['content']
    del nb.metadata['jupytext']['notebook_metadata_filter']
    cm.save(model=notebook_model(nb), path=tmp_py)
    with open(str(tmpdir.join(tmp_py))) as stream:
        text_py2 = stream.read()
    header = text_py2[:-len(text_py)]
    assert any(['percent' in line for line in header.splitlines()])
    text_py2 = text_py2[-len(text_py):]
    compare(text_py2, text_py)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_julia'))
def test_save_to_percent_format(nb_file, tmpdir):
    if False:
        return 10
    tmp_ipynb = 'notebook.ipynb'
    tmp_jl = 'notebook.jl'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.preferred_jupytext_formats_save = 'jl:percent'
    nb = jupytext.read(nb_file)
    nb['metadata']['jupytext'] = {'formats': 'ipynb,jl'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    with open(str(tmpdir.join(tmp_jl))) as stream:
        text_jl = stream.read()
    (metadata, _, _, _) = header_to_metadata_and_cell(text_jl.splitlines(), '#', '')
    assert metadata['jupytext']['formats'] == 'ipynb,jl:percent'

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_save_using_preferred_and_default_format_170(nb_file, tmpdir):
    if False:
        while True:
            i = 10
    nb = read(nb_file)
    tmp_py = str(tmpdir.join('python/notebook.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.preferred_jupytext_formats_save = 'py:percent'
    cm.formats = 'ipynb,python//py'
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    nb_py = read(tmp_py)
    assert nb_py.metadata['jupytext']['text_representation']['format_name'] == 'percent'
    tmp_py = str(tmpdir.join('python/notebook.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.preferred_jupytext_formats_save = 'python//py:percent'
    cm.formats = 'ipynb,python//py'
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    nb_py = read(tmp_py)
    assert nb_py.metadata['jupytext']['text_representation']['format_name'] == 'percent'
    tmp_py = str(tmpdir.join('python/notebook.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.formats = 'ipynb,python//py:percent'
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    nb_py = read(tmp_py)
    assert nb_py.metadata['jupytext']['text_representation']['format_name'] == 'percent'

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_open_using_preferred_and_default_format_174(nb_file, tmpdir):
    if False:
        return 10
    tmp_ipynb = str(tmpdir.join('notebook.ipynb'))
    tmp_py = str(tmpdir.join('python/notebook.py'))
    tmp_py2 = str(tmpdir.join('other/notebook.py'))
    os.makedirs(str(tmpdir.join('other')))
    shutil.copyfile(nb_file, tmp_ipynb)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.formats = 'ipynb,python//py:percent'
    cm.notebook_metadata_filter = 'all'
    cm.cell_metadata_filter = 'all'
    model = cm.get('notebook.ipynb')
    cm.save(model=model, path='notebook.ipynb')
    assert os.path.isfile(tmp_py)
    os.remove(tmp_ipynb)
    model2 = cm.get('python/notebook.py')
    compare_notebooks(model2['content'], model['content'])
    shutil.move(tmp_py, tmp_py2)
    model2 = cm.get('other/notebook.py')
    compare_notebooks(model2['content'], model['content'])
    cm.save(model=model, path='other/notebook.py')
    assert not os.path.isfile(tmp_ipynb)
    assert not os.path.isfile(str(tmpdir.join('other/notebook.ipynb')))

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py', skip='many hash'))
def test_kernelspec_are_preserved(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = str(tmpdir.join('notebook.ipynb'))
    tmp_py = str(tmpdir.join('notebook.py'))
    shutil.copyfile(nb_file, tmp_ipynb)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.formats = 'ipynb,py'
    cm.notebook_metadata_filter = '-all'
    model = cm.get('notebook.ipynb')
    model['content'].metadata['kernelspec'] = {'display_name': 'Kernel name', 'language': 'python', 'name': 'custom'}
    cm.save(model=model, path='notebook.ipynb')
    assert os.path.isfile(tmp_py)
    model2 = cm.get('notebook.ipynb')
    compare_notebooks(model2['content'], model['content'])

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_save_to_light_percent_sphinx_format(nb_file, tmpdir):
    if False:
        print('Hello World!')
    tmp_ipynb = 'notebook.ipynb'
    tmp_lgt_py = 'notebook.lgt.py'
    tmp_pct_py = 'notebook.pct.py'
    tmp_spx_py = 'notebook.spx.py'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb['metadata']['jupytext'] = {'formats': 'ipynb,.pct.py:percent,.lgt.py:light,.spx.py:sphinx'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    with open(str(tmpdir.join(tmp_pct_py))) as stream:
        assert read_format_from_metadata(stream.read(), '.py') == 'percent'
    with open(str(tmpdir.join(tmp_lgt_py))) as stream:
        assert read_format_from_metadata(stream.read(), '.py') == 'light'
    with open(str(tmpdir.join(tmp_spx_py))) as stream:
        assert read_format_from_metadata(stream.read(), '.py') == 'sphinx'
    model = cm.get(path=tmp_pct_py)
    compare_notebooks(model['content'], nb)
    model = cm.get(path=tmp_lgt_py)
    compare_notebooks(model['content'], nb)
    model = cm.get(path=tmp_spx_py)
    model = cm.get(path=tmp_ipynb)
    compare_notebooks(model['content'], nb)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_pair_notebook_with_dot(nb_file, tmpdir):
    if False:
        i = 10
        return i + 15
    tmp_py = 'file.5.1.py'
    tmp_ipynb = 'file.5.1.ipynb'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb['metadata']['jupytext'] = {'formats': 'ipynb,py:percent'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    assert os.path.isfile(str(tmpdir.join(tmp_ipynb)))
    with open(str(tmpdir.join(tmp_py))) as stream:
        assert read_format_from_metadata(stream.read(), '.py') == 'percent'
    model = cm.get(path=tmp_py)
    assert model['name'] == 'file.5.1.py'
    compare_notebooks(model['content'], nb)
    model = cm.get(path=tmp_ipynb)
    assert model['name'] == 'file.5.1.ipynb'
    compare_notebooks(model['content'], nb)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_preferred_format_allows_to_read_others_format(nb_file, tmpdir):
    if False:
        print('Hello World!')
    tmp_ipynb = 'notebook.ipynb'
    tmp_nbpy = 'notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.preferred_jupytext_formats_save = 'py:light'
    cm.root_dir = str(tmpdir)
    nb = jupytext.read(nb_file)
    nb['metadata']['jupytext'] = {'formats': 'ipynb,py'}
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.preferred_jupytext_formats_read = 'py:percent'
    model = cm.get(tmp_nbpy)
    assert model['content']['metadata']['jupytext']['formats'] == 'ipynb,py:light'
    compare_notebooks(model['content'], nb)
    model['content']['metadata']['jupytext']['formats'] == 'ipynb,py'
    cm.preferred_jupytext_formats_save = 'py:percent'
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    model = cm.get(tmp_nbpy)
    compare_notebooks(model['content'], nb)
    assert model['content']['metadata']['jupytext']['formats'] == 'ipynb,py:percent'

def test_preferred_formats_read_auto(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_py = 'notebook.py'
    with open(str(tmpdir.join(tmp_py)), 'w') as script:
        script.write('# cell one\n1 + 1\n')
    cm = jupytext.TextFileContentsManager()
    cm.preferred_jupytext_formats_read = 'auto:percent'
    cm.root_dir = str(tmpdir)
    model = cm.get(tmp_py)
    assert 'percent' == model['content']['metadata']['jupytext']['text_representation']['format_name']

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb'))
def test_save_in_auto_extension_global(nb_file, tmpdir):
    if False:
        i = 10
        return i + 15
    nb = jupytext.read(nb_file)
    auto_ext = auto_ext_from_metadata(nb.metadata)
    tmp_ipynb = 'notebook.ipynb'
    tmp_script = 'notebook' + auto_ext
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,auto'
    cm.preferred_jupytext_formats_save = 'auto:percent'
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    with open(str(tmpdir.join(tmp_script))) as stream:
        assert read_format_from_metadata(stream.read(), auto_ext) == 'percent'
    model = cm.get(path=tmp_script)
    assert 'formats' not in model['content'].metadata.get('jupytext', {})
    compare_notebooks(model['content'], nb)

def test_global_auto_pairing_works_with_empty_notebook(tmpdir):
    if False:
        return 10
    nb = new_notebook()
    tmp_ipynb = str(tmpdir.join('notebook.ipynb'))
    tmp_py = str(tmpdir.join('notebook.py'))
    tmp_auto = str(tmpdir.join('notebook.auto'))
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,auto'
    cm.preferred_jupytext_formats_save = 'auto:percent'
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert not os.path.isfile(tmp_py)
    assert not os.path.isfile(tmp_auto)
    assert 'notebook.ipynb' not in cm.paired_notebooks
    model = cm.get(path='notebook.ipynb')
    compare_notebooks(model['content'], nb)
    nb.metadata['language_info'] = {'codemirror_mode': {'name': 'ipython', 'version': 3}, 'file_extension': '.py', 'mimetype': 'text/x-python', 'name': 'python', 'nbconvert_exporter': 'python', 'pygments_lexer': 'ipython3', 'version': '3.7.3'}
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    assert not os.path.isfile(tmp_auto)
    assert len(cm.paired_notebooks['notebook.ipynb']) == 2
    with open(tmp_py, 'a') as fp:
        fp.write('# %%\n2+2\n')
    nb2 = cm.get(path='notebook.ipynb')['content']
    assert len(nb2.cells) == 1
    assert nb2.cells[0].source == '2+2'

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb'))
def test_save_in_auto_extension_global_with_format(nb_file, tmpdir):
    if False:
        print('Hello World!')
    nb = jupytext.read(nb_file)
    auto_ext = auto_ext_from_metadata(nb.metadata)
    tmp_ipynb = 'notebook.ipynb'
    tmp_script = 'notebook' + auto_ext
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,auto:percent'
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    with open(str(tmpdir.join(tmp_script))) as stream:
        assert read_format_from_metadata(stream.read(), auto_ext) == 'percent'
    model = cm.get(path=tmp_script)
    assert 'formats' not in model['content'].metadata.get('jupytext', {})
    compare_notebooks(model['content'], nb)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb'))
def test_save_in_auto_extension_local(nb_file, tmpdir):
    if False:
        i = 10
        return i + 15
    nb = jupytext.read(nb_file)
    nb.metadata.setdefault('jupytext', {})['formats'] = 'ipynb,auto:percent'
    auto_ext = auto_ext_from_metadata(nb.metadata)
    tmp_ipynb = 'notebook.ipynb'
    tmp_script = 'notebook' + auto_ext
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    with open(str(tmpdir.join(tmp_script))) as stream:
        assert read_format_from_metadata(stream.read(), auto_ext) == 'percent'
    model = cm.get(path=tmp_script)
    compare_notebooks(model['content'], nb)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb'))
def test_save_in_pct_and_lgt_auto_extensions(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    nb = jupytext.read(nb_file)
    auto_ext = auto_ext_from_metadata(nb.metadata)
    tmp_ipynb = 'notebook.ipynb'
    tmp_pct_script = 'notebook.pct' + auto_ext
    tmp_lgt_script = 'notebook.lgt' + auto_ext
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,.pct.auto,.lgt.auto'
    cm.preferred_jupytext_formats_save = '.pct.auto:percent,.lgt.auto:light'
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    with open(str(tmpdir.join(tmp_pct_script))) as stream:
        assert read_format_from_metadata(stream.read(), auto_ext) == 'percent'
    with open(str(tmpdir.join(tmp_lgt_script))) as stream:
        assert read_format_from_metadata(stream.read(), auto_ext) == 'light'

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb', skip='(magic|305)'))
def test_metadata_filter_is_effective(nb_file, tmpdir):
    if False:
        i = 10
        return i + 15
    nb = jupytext.read(nb_file)
    tmp_ipynb = 'notebook.ipynb'
    tmp_script = 'notebook.py'
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    cm.formats = 'ipynb,py'
    cm.notebook_metadata_filter = 'jupytext,-all'
    cm.cell_metadata_filter = '-all'
    nb = cm.get(tmp_ipynb)['content']
    assert nb.metadata['jupytext']['cell_metadata_filter'] == '-all'
    assert nb.metadata['jupytext']['notebook_metadata_filter'] == 'jupytext,-all'
    cm.save(model=notebook_model(nb), path=tmp_ipynb)
    nb2 = jupytext.read(str(tmpdir.join(tmp_script)))
    assert set(nb2.metadata.keys()) <= {'jupytext', 'kernelspec'}
    for cell in nb2.cells:
        assert not cell.metadata
    nb3 = cm.get(tmp_script)['content']
    compare_notebooks(nb3, nb)

def test_no_metadata_added_to_scripts_139(tmpdir):
    if False:
        while True:
            i = 10
    tmp_script = str(tmpdir.join('script.py'))
    text = "import os\n\n\nprint('hello1')\n\n\n\nprint('hello2')\n"
    with open(tmp_script, 'w') as fp:
        fp.write(text)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.freeze_metadata = True
    cm.notebook_metadata_filter = '-all'
    cm.cell_metadata_filter = '-lines_to_next_cell'
    model = cm.get('script.py')
    for cell in model['content'].cells:
        cell.metadata.update({'ExecuteTime': {'start_time': '2019-02-06T11:53:21.208644Z', 'end_time': '2019-02-06T11:53:21.213071Z'}})
    cm.save(model=model, path='script.py')
    with open(tmp_script) as fp:
        compare(fp.read(), text)

@pytest.mark.parametrize('nb_file,ext', itertools.product(list_notebooks('ipynb_py'), ['.py', '.ipynb']))
def test_local_format_can_deactivate_pairing(nb_file, ext, tmpdir):
    if False:
        while True:
            i = 10
    'This is a test for #157: local format can be used to deactivate the global pairing'
    nb = jupytext.read(nb_file)
    nb.metadata['jupytext_formats'] = ext[1:]
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,py'
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path='notebook' + ext)
    assert os.path.isfile(str(tmpdir.join('notebook.py'))) == (ext == '.py')
    assert os.path.isfile(str(tmpdir.join('notebook.ipynb'))) == (ext == '.ipynb')
    nb2 = cm.get('notebook' + ext)['content']
    compare_notebooks(nb2, nb)
    cm.save(model=notebook_model(nb2), path='notebook' + ext)
    assert os.path.isfile(str(tmpdir.join('notebook.py'))) == (ext == '.py')
    assert os.path.isfile(str(tmpdir.join('notebook.ipynb'))) == (ext == '.ipynb')
    nb3 = cm.get('notebook' + ext)['content']
    compare_notebooks(nb3, nb)

@pytest.mark.parametrize('nb_file', list_notebooks('Rmd'))
def test_global_pairing_allows_to_save_other_file_types(nb_file, tmpdir):
    if False:
        while True:
            i = 10
    'This is a another test for #157: local format can be used to deactivate the global pairing'
    nb = jupytext.read(nb_file)
    cm = jupytext.TextFileContentsManager()
    cm.formats = 'ipynb,py'
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path='notebook.Rmd')
    assert os.path.isfile(str(tmpdir.join('notebook.Rmd')))
    assert not os.path.isfile(str(tmpdir.join('notebook.py')))
    assert not os.path.isfile(str(tmpdir.join('notebook.ipynb')))
    nb2 = cm.get('notebook.Rmd')['content']
    compare_notebooks(nb2, nb)

@requires_user_kernel_python3
@pytest.mark.parametrize('nb_file', list_notebooks('R'))
def test_python_kernel_preserves_R_files(nb_file, tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Opening a R file with a Jupyter server that has no R kernel should not modify the file'
    tmp_r_file = str(tmpdir.join('script.R'))
    with open(nb_file) as fp:
        script = fp.read()
    with open(tmp_r_file, 'w') as fp:
        fp.write(script)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    model = cm.get('script.R')
    model['content'].metadata['kernelspec'] = kernelspec_from_language('python')
    cm.save(model=model, path='script.R')
    with open(tmp_r_file) as fp:
        script2 = fp.read()
    compare(script2, script)

def test_pair_notebook_in_another_folder(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    os.makedirs(str(tmpdir.join('notebooks')))
    tmp_ipynb = str(tmpdir.join('notebooks/notebook_name.ipynb'))
    tmp_py = str(tmpdir.join('scripts/notebook_name.py'))
    cm.save(model=notebook_model(new_notebook(metadata={'jupytext': {'formats': 'notebooks//ipynb,scripts//py'}})), path='notebooks/notebook_name.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    cm.get('notebooks/notebook_name.ipynb')
    cm.get('scripts/notebook_name.py')

def test_pair_notebook_in_dotdot_folder(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    os.makedirs(str(tmpdir.join('notebooks')))
    tmp_ipynb = str(tmpdir.join('notebooks/notebook_name.ipynb'))
    tmp_py = str(tmpdir.join('scripts/notebook_name.py'))
    cm.save(model=notebook_model(new_notebook(metadata={'jupytext': {'formats': 'ipynb,../scripts//py'}})), path='notebooks/notebook_name.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    cm.get('notebooks/notebook_name.ipynb')
    cm.get('scripts/notebook_name.py')

@requires_sphinx_gallery
def test_rst2md_option(tmpdir):
    if False:
        i = 10
        return i + 15
    tmp_py = str(tmpdir.join('notebook.py'))
    nb = new_notebook(cells=[new_markdown_cell('A short sphinx notebook'), new_markdown_cell(':math:`1+1`')])
    write(nb, tmp_py, fmt='py:sphinx')
    cm = jupytext.TextFileContentsManager()
    cm.sphinx_convert_rst2md = True
    cm.root_dir = str(tmpdir)
    nb2 = cm.get('notebook.py')['content']
    assert nb2.cells[2].source == '$1+1$'
    assert nb2.metadata['jupytext']['rst2md'] is False

def test_split_at_heading_option(tmpdir):
    if False:
        i = 10
        return i + 15
    text = 'Markdown text\n\n# Header one\n\n## Header two\n'
    tmp_md = str(tmpdir.join('notebook.md'))
    with open(tmp_md, 'w') as fp:
        fp.write(text)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.split_at_heading = True
    nb = cm.get('notebook.md')['content']
    assert nb.cells[0].source == 'Markdown text'
    assert nb.cells[1].source == '# Header one'
    assert nb.cells[2].source == '## Header two'
    nb.metadata['jupytext']['notebook_metadata_filter'] = '-all'
    text2 = writes(nb, 'md')
    compare(text2, text)

def test_load_then_change_formats(tmpdir):
    if False:
        while True:
            i = 10
    tmp_ipynb = str(tmpdir.join('nb.ipynb'))
    tmp_py = str(tmpdir.join('nb.py'))
    nb = new_notebook(metadata={'jupytext': {'formats': 'ipynb,py:light'}})
    write(nb, tmp_ipynb)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    model = cm.get('nb.ipynb')
    assert model['content'].metadata['jupytext']['formats'] == 'ipynb,py:light'
    cm.save(model, path='nb.ipynb')
    assert os.path.isfile(tmp_py)
    assert read(tmp_py).metadata['jupytext']['formats'] == 'ipynb,py:light'
    time.sleep(0.5)
    del model['content'].metadata['jupytext']['formats']
    cm.save(model, path='nb.ipynb')
    cm.get('nb.ipynb')
    os.remove(tmp_py)
    model['content'].metadata.setdefault('jupytext', {})['formats'] = 'ipynb,py:percent'
    cm.save(model, path='nb.ipynb')
    assert os.path.isfile(tmp_py)
    assert read(tmp_py).metadata['jupytext']['formats'] == 'ipynb,py:percent'
    os.remove(tmp_py)
    del model['content'].metadata['jupytext']['formats']
    cm.save(model, path='nb.ipynb')
    assert not os.path.isfile(tmp_py)

def test_set_then_change_formats(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_py = str(tmpdir.join('nb.py'))
    nb = new_notebook(metadata={'jupytext': {'formats': 'ipynb,py:light'}})
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert os.path.isfile(tmp_py)
    assert read(tmp_py).metadata['jupytext']['formats'] == 'ipynb,py:light'
    os.remove(tmp_py)
    nb.metadata['jupytext']['formats'] = 'ipynb,py:percent'
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert os.path.isfile(tmp_py)
    assert read(tmp_py).metadata['jupytext']['formats'] == 'ipynb,py:percent'
    os.remove(tmp_py)
    del nb.metadata['jupytext']['formats']
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert not os.path.isfile(tmp_py)

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py')[:1])
def test_set_then_change_auto_formats(tmpdir, nb_file):
    if False:
        return 10
    tmp_ipynb = str(tmpdir.join('nb.ipynb'))
    tmp_py = str(tmpdir.join('nb.py'))
    tmp_rmd = str(tmpdir.join('nb.Rmd'))
    nb = new_notebook(metadata=read(nb_file).metadata)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb.metadata['jupytext'] = {'formats': 'ipynb,auto:light'}
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert 'nb.py' in cm.paired_notebooks
    assert 'nb.auto' not in cm.paired_notebooks
    assert os.path.isfile(tmp_py)
    assert read(tmp_ipynb).metadata['jupytext']['formats'] == 'ipynb,py:light'
    time.sleep(0.5)
    nb.metadata['jupytext'] = {'formats': 'ipynb,Rmd'}
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert 'nb.Rmd' in cm.paired_notebooks
    assert 'nb.py' not in cm.paired_notebooks
    assert 'nb.auto' not in cm.paired_notebooks
    assert os.path.isfile(tmp_rmd)
    assert read(tmp_ipynb).metadata['jupytext']['formats'] == 'ipynb,Rmd'
    cm.get('nb.ipynb')
    time.sleep(0.5)
    del nb.metadata['jupytext']
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert 'nb.Rmd' not in cm.paired_notebooks
    assert 'nb.py' not in cm.paired_notebooks
    assert 'nb.auto' not in cm.paired_notebooks
    cm.get('nb.ipynb')

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_py'))
def test_share_py_recreate_ipynb(tmpdir, nb_file):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = str(tmpdir.join('nb.ipynb'))
    tmp_py = str(tmpdir.join('nb.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.preferred_jupytext_formats_save = 'py:percent'
    cm.formats = 'ipynb,py'
    cm.notebook_metadata_filter = '-all'
    cm.cell_metadata_filter = '-all'
    nb = read(nb_file)
    model_ipynb = cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    os.remove(tmp_ipynb)
    model = cm.get('nb.py')
    cm.save(model=model, path='nb.py')
    assert os.path.isfile(tmp_ipynb)
    assert model_ipynb['last_modified'] == model['last_modified']

def test_vim_folding_markers(tmpdir):
    if False:
        i = 10
        return i + 15
    tmp_ipynb = str(tmpdir.join('nb.ipynb'))
    tmp_py = str(tmpdir.join('nb.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.cell_markers = '{{{,}}}'
    cm.formats = 'ipynb,py'
    nb = new_notebook(cells=[new_code_cell("# region\n'''Sample cell with region markers'''\n'''End of the cell'''\n# end region"), new_code_cell('a = 1\n\n\nb = 1')])
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    nb2 = cm.get('nb.ipynb')['content']
    compare_notebooks(nb2, nb)
    nb3 = read(tmp_py)
    assert nb3.metadata['jupytext']['cell_markers'] == '{{{,}}}'
    with open(tmp_py) as fp:
        text = fp.read()
    text = re.sub(re.compile('# ---.*# ---\\n\\n', re.DOTALL), '', text)
    compare(text, "# region\n'''Sample cell with region markers'''\n'''End of the cell'''\n# end region\n\n# {{{\na = 1\n\n\nb = 1\n# }}}\n")

def test_vscode_pycharm_folding_markers(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    tmp_ipynb = str(tmpdir.join('nb.ipynb'))
    tmp_py = str(tmpdir.join('nb.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.cell_markers = 'region,endregion'
    cm.formats = 'ipynb,py'
    nb = new_notebook(cells=[new_code_cell("# {{{\n'''Sample cell with region markers'''\n'''End of the cell'''\n# }}}"), new_code_cell('a = 1\n\n\nb = 1')])
    cm.save(model=notebook_model(nb), path='nb.ipynb')
    assert os.path.isfile(tmp_ipynb)
    assert os.path.isfile(tmp_py)
    nb2 = cm.get('nb.ipynb')['content']
    compare_notebooks(nb2, nb)
    nb3 = read(tmp_py)
    assert nb3.metadata['jupytext']['cell_markers'] == 'region,endregion'
    with open(tmp_py) as fp:
        text = fp.read()
    text = re.sub(re.compile('# ---.*# ---\\n\\n', re.DOTALL), '', text)
    compare(text, "# {{{\n'''Sample cell with region markers'''\n'''End of the cell'''\n# }}}\n\n# region\na = 1\n\n\nb = 1\n# endregion\n")

def test_open_file_with_cell_markers(tmpdir):
    if False:
        print('Hello World!')
    tmp_py = str(tmpdir.join('nb.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.cell_markers = 'region,endregion'
    text = '# +\n# this is a unique code cell\n1 + 1\n\n2 + 2\n'
    with open(tmp_py, 'w') as fp:
        fp.write(text)
    nb = cm.get('nb.py')['content']
    assert len(nb.cells) == 1
    cm.save(model=notebook_model(nb), path='nb.py')
    with open(tmp_py) as fp:
        text2 = fp.read()
    expected = '# region\n# this is a unique code cell\n1 + 1\n\n2 + 2\n# endregion\n'
    compare(text2, expected)

def test_save_file_with_cell_markers(tmpdir):
    if False:
        print('Hello World!')
    tmp_py = str(tmpdir.join('nb.py'))
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.cell_markers = 'region,endregion'
    text = '# +\n# this is a unique code cell\n1 + 1\n\n2 + 2\n'
    with open(tmp_py, 'w') as fp:
        fp.write(text)
    nb = cm.get('nb.py')['content']
    assert len(nb.cells) == 1
    cm.save(model=notebook_model(nb), path='nb.py')
    with open(tmp_py) as fp:
        text2 = fp.read()
    compare(text2, '# region\n# this is a unique code cell\n1 + 1\n\n2 + 2\n# endregion\n')
    nb2 = cm.get('nb.py')['content']
    compare_notebooks(nb2, nb)
    assert nb2.metadata['jupytext']['cell_markers'] == 'region,endregion'

def test_notebook_extensions(tmpdir, cwd_tmpdir):
    if False:
        i = 10
        return i + 15
    nb = new_notebook()
    write(nb, 'script.py')
    write(nb, 'notebook.Rmd')
    write(nb, 'notebook.ipynb')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.notebook_extensions = 'ipynb,Rmd'
    model = cm.get('notebook.ipynb')
    assert model['type'] == 'notebook'
    model = cm.get('notebook.Rmd')
    assert model['type'] == 'notebook'
    model = cm.get('script.py')
    assert model['type'] == 'file'

def test_notebook_extensions_in_config(tmpdir, cwd_tmpdir):
    if False:
        i = 10
        return i + 15
    nb = new_notebook()
    write(nb, 'script.py')
    write(nb, 'notebook.Rmd')
    write(nb, 'notebook.ipynb')
    tmpdir.join('jupytext.toml').write('notebook_extensions = ["ipynb", "Rmd"]')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    model = cm.get('notebook.ipynb')
    assert model['type'] == 'notebook'
    model = cm.get('notebook.Rmd')
    assert model['type'] == 'notebook'
    model = cm.get('script.py')
    assert model['type'] == 'file'

def test_invalid_config_in_cm(tmpdir, cwd_tmpdir):
    if False:
        print('Hello World!')
    nb = new_notebook()
    write(nb, 'notebook.ipynb')
    tmpdir.join('pyproject.toml').write('[tool.jupysql.SqlMagic]\nautopandas = False\ndisplaylimit = 1')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.get('')
    model = cm.get('notebook.ipynb')
    assert model['type'] == 'notebook'

def test_download_file_318(tmpdir):
    if False:
        print('Hello World!')
    tmp_ipynb = str(tmpdir.join('notebook.ipynb'))
    tmp_py = str(tmpdir.join('notebook.py'))
    nb = new_notebook()
    nb.metadata['jupytext'] = {'formats': 'ipynb,py'}
    write(nb, tmp_ipynb)
    write(nb, tmp_py)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.notebook_extensions = 'ipynb'
    model = cm.get('notebook.ipynb', content=True, type=None, format=None)
    assert model['type'] == 'notebook'

def test_markdown_and_r_extensions(tmpdir):
    if False:
        while True:
            i = 10
    tmp_r = str(tmpdir.join('script.r'))
    tmp_markdown = str(tmpdir.join('notebook.markdown'))
    nb = new_notebook()
    write(nb, tmp_r)
    write(nb, tmp_markdown)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    model = cm.get('script.r')
    assert model['type'] == 'notebook'
    model = cm.get('notebook.markdown')
    assert model['type'] == 'notebook'

def test_server_extension_issubclass():
    if False:
        i = 10
        return i + 15

    class SubClassTextFileContentsManager(jupytext.TextFileContentsManager):
        pass
    assert not isinstance(SubClassTextFileContentsManager, jupytext.TextFileContentsManager)
    assert issubclass(SubClassTextFileContentsManager, jupytext.TextFileContentsManager)

def test_multiple_pairing(tmpdir):
    if False:
        while True:
            i = 10
    'Test that multiple pairing works. Input cells are loaded from the most recent text representation among\n    the paired ones'
    tmp_ipynb = str(tmpdir.join('notebook.ipynb'))
    tmp_md = str(tmpdir.join('notebook.md'))
    tmp_py = str(tmpdir.join('notebook.py'))

    def nb(text):
        if False:
            return 10
        return new_notebook(cells=[new_markdown_cell(text)], metadata={'jupytext': {'formats': 'ipynb,md,py'}})
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.save(model=notebook_model(nb('saved from cm')), path='notebook.ipynb')
    compare_notebooks(jupytext.read(tmp_ipynb), nb('saved from cm'))
    compare_notebooks(jupytext.read(tmp_md), nb('saved from cm'))
    compare_notebooks(jupytext.read(tmp_py), nb('saved from cm'))
    jupytext.write(nb('md edited'), tmp_md)
    model = cm.get('notebook.ipynb')
    compare_notebooks(model['content'], nb('md edited'))
    cm.save(model=model, path='notebook.ipynb')
    compare_notebooks(jupytext.read(tmp_ipynb), nb('md edited'))
    compare_notebooks(jupytext.read(tmp_md), nb('md edited'))
    compare_notebooks(jupytext.read(tmp_py), nb('md edited'))
    jupytext.write(nb('py edited'), tmp_py)
    model = cm.get('notebook.md')
    compare_notebooks(model['content'], nb('md edited'))
    model = cm.get('notebook.ipynb')
    compare_notebooks(model['content'], nb('py edited'))
    cm.save(model=model, path='notebook.ipynb')
    compare_notebooks(jupytext.read(tmp_ipynb), nb('py edited'))
    compare_notebooks(jupytext.read(tmp_md), nb('py edited'))
    compare_notebooks(jupytext.read(tmp_py), nb('py edited'))
    model_ipynb = cm.get('notebook.ipynb', content=False, load_alternative_format=False)
    model_md = cm.get('notebook.md', content=False, load_alternative_format=False)
    model_py = cm.get('notebook.py', content=False, load_alternative_format=False)
    assert model_ipynb['last_modified'] <= model_py['last_modified']
    assert model_py['last_modified'] <= model_md['last_modified']

def test_filter_jupytext_version_information_416(python_notebook, tmpdir, cwd_tmpdir):
    if False:
        return 10
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    cm.notebook_metadata_filter = '-jupytext.text_representation.jupytext_version'
    notebook = python_notebook
    notebook.metadata['jupytext_formats'] = 'ipynb,py'
    model = notebook_model(notebook)
    cm.save(model=model, path='notebook.ipynb')
    assert os.path.isfile('notebook.py')
    with open('notebook.py') as fp:
        text = fp.read()
    assert '---' in text
    assert 'jupytext:' in text
    assert 'kernelspec:' in text
    assert 'jupytext_version:' not in text

def test_new_untitled(tmpdir):
    if False:
        while True:
            i = 10
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    (untitled, ext) = cm.new_untitled(type='notebook')['path'].split('.')
    assert untitled
    assert ext == 'ipynb'
    assert cm.new_untitled(type='notebook', ext='.md')['path'] == untitled + '1.md'
    assert cm.new_untitled(type='notebook', ext='.py')['path'] == untitled + '2.py'
    assert cm.new_untitled(type='notebook', ext='.md:myst')['path'] == untitled + '3.md'
    assert cm.new_untitled(type='notebook', ext='.py:percent')['path'] == untitled + '4.py'
    assert cm.new_untitled(type='notebook', ext='.Rmd')['path'] == untitled + '5.Rmd'
    for ext in ['.py', '.md']:
        model = cm.new_untitled(type='file', ext=ext)
        assert model['content'] is None
        assert model['path'] == f'untitled{ext}'
    assert cm.new_untitled(type='directory')['path'] == 'Untitled Folder'

def test_nested_prefix(tmpdir):
    if False:
        while True:
            i = 10
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    nb = new_notebook(cells=[new_code_cell('1+1'), new_markdown_cell('Some text')], metadata={'jupytext': {'formats': 'ipynb,nested/prefix//.py'}})
    cm.save(model=notebook_model(nb), path='notebook.ipynb')
    assert tmpdir.join('nested').join('prefix').join('notebook.py').isfile()

def fs_meta_manager(tmpdir):
    if False:
        while True:
            i = 10
    try:
        from jupyterfs.metamanager import MetaManager
    except ImportError:
        pytest.skip('jupyterfs is not available')
    cm_class = jupytext.build_jupytext_contents_manager_class(MetaManager)
    logger = logging.getLogger('jupyter-fs')
    cm = cm_class(parent=None, log=logger)
    cm.initResource({'url': f'osfs://{tmpdir}'})
    return cm

def test_jupytext_jupyter_fs_metamanager(tmpdir):
    if False:
        return 10
    'Test the basic get/save functions of Jupytext with a fs manager\n    https://github.com/mwouts/jupytext/issues/618'
    cm = fs_meta_manager(tmpdir)
    osfs = [h for h in cm._managers if h != ''][0]
    text = 'some text\n'
    cm.save(dict(type='file', content=text, format='text'), path=osfs + ':text.md')
    nb = new_notebook(cells=[new_markdown_cell('A markdown cell'), new_code_cell('1 + 1')])
    cm.save(notebook_model(nb), osfs + ':notebook.ipynb')
    cm.save(notebook_model(nb), osfs + ':text_notebook.md')
    directory = cm.get(osfs + ':/')
    assert {file['name'] for file in directory['content']} == {'text.md', 'text_notebook.md', 'notebook.ipynb'}
    model = cm.get(osfs + ':/text.md', type='file')
    assert model['type'] == 'file'
    assert model['content'] == text
    model = cm.get(osfs + ':text.md', type='notebook')
    assert model['type'] == 'notebook'
    compare_cells(model['content'].cells, [new_markdown_cell(text.strip())], compare_ids=False)
    for nb_file in ['notebook.ipynb', 'text_notebook.md']:
        model = cm.get(osfs + ':' + nb_file)
        assert model['type'] == 'notebook'
        actual_cells = model['content'].cells
        for cell in actual_cells:
            cell.metadata = {}
        compare_cells(actual_cells, nb.cells, compare_ids=False)

def test_config_jupytext_jupyter_fs_meta_manager(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Test the configuration of Jupytext with a fs manager'
    tmpdir.join('jupytext.toml').write('formats = "ipynb,py"')
    cm = fs_meta_manager(tmpdir)
    osfs = [h for h in cm._managers if h != ''][0]
    nb = new_notebook()
    cm.save(dict(type='file', content='text', format='text'), path=osfs + ':text.md')
    cm.save(notebook_model(nb), osfs + ':script.py')
    cm.save(notebook_model(nb), osfs + ':text_notebook.md')
    cm.save(notebook_model(nb), osfs + ':notebook.ipynb')
    directory = cm.get(osfs + ':/')
    assert {file['name'] for file in directory['content']} == {'jupytext.toml', 'text.md', 'text_notebook.md', 'notebook.ipynb', 'notebook.py', 'script.py', 'script.ipynb'}

def test_timestamp_is_correct_after_reload_978(tmp_path, python_notebook):
    if False:
        print('Hello World!')
    'Here we reproduce the conditions in Issue #978 and make sure no\n    warning is generated'
    nb = python_notebook
    nb.metadata['jupytext'] = {'formats': 'ipynb,py:percent'}
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmp_path)
    ipynb_file = tmp_path / 'nb.ipynb'
    py_file = tmp_path / 'nb.py'
    cm.save(notebook_model(nb), path='nb.ipynb')
    assert ipynb_file.exists()
    assert py_file.exists()
    org_model = cm.get('nb.ipynb')
    time.sleep(0.5)
    text = py_file.read_text()
    text = text + '\n\n# %%\n# A new cell\n2 + 2\n'
    py_file.write_text(text)
    model = cm.get('nb.ipynb')
    nb = model['content']
    assert 'A new cell' in nb.cells[-1].source
    assert model['last_modified'] > org_model['last_modified']

def test_move_paired_notebook_to_subdir_1059(tmp_path, python_notebook):
    if False:
        while True:
            i = 10
    (tmp_path / 'jupytext.toml').write_text('formats = "notebooks///ipynb,scripts///py:percent"\n')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmp_path)
    (tmp_path / 'notebooks').mkdir()
    cm.save(notebook_model(python_notebook), path='notebooks/my_notebook.ipynb')
    assert (tmp_path / 'notebooks' / 'my_notebook.ipynb').exists()
    assert (tmp_path / 'scripts' / 'my_notebook.py').exists()
    (tmp_path / 'notebooks' / 'subdir').mkdir()
    cm.rename_file('notebooks/my_notebook.ipynb', 'notebooks/subdir/my_notebook.ipynb')
    assert (tmp_path / 'notebooks' / 'subdir' / 'my_notebook.ipynb').exists()
    assert (tmp_path / 'scripts' / 'subdir' / 'my_notebook.py').exists()
    assert not (tmp_path / 'notebooks' / 'my_notebook.ipynb').exists()
    assert not (tmp_path / 'scripts' / 'my_notebook.py').exists()
    model = cm.get('scripts/subdir/my_notebook.py')
    nb = model['content']
    compare_notebooks(nb, python_notebook, fmt='py:percent')