import pytest
from nbformat.v4.nbbase import new_notebook
import jupytext
from jupytext.compare import compare
from jupytext.formats import JupytextFormatError, divine_format, get_format_implementation, guess_format, long_form_multiple_formats, read_format_from_metadata, rearrange_jupytext_metadata, short_form_multiple_formats, update_jupytext_formats_metadata, validate_one_format
from .utils import list_notebooks, requires_myst, requires_pandoc

@pytest.mark.parametrize('nb_file', list_notebooks('python'))
def test_guess_format_light(nb_file):
    if False:
        print('Hello World!')
    with open(nb_file) as stream:
        assert guess_format(stream.read(), ext='.py')[0] == 'light'

@pytest.mark.parametrize('nb_file', list_notebooks('percent'))
def test_guess_format_percent(nb_file):
    if False:
        return 10
    with open(nb_file) as stream:
        assert guess_format(stream.read(), ext='.py')[0] == 'percent'

def test_guess_format_simple_percent(nb='# %%\nprint("hello world!")\n'):
    if False:
        while True:
            i = 10
    assert guess_format(nb, ext='.py')[0] == 'percent'

def test_guess_format_simple_percent_with_magic(nb='# %%\n# %time\nprint("hello world!")\n'):
    if False:
        i = 10
        return i + 15
    assert guess_format(nb, ext='.py')[0] == 'percent'

def test_guess_format_simple_hydrogen_with_magic(nb='# %%\n%time\nprint("hello world!")\n'):
    if False:
        i = 10
        return i + 15
    assert guess_format(nb, ext='.py')[0] == 'hydrogen'

@pytest.mark.parametrize('nb_file', list_notebooks('sphinx'))
def test_guess_format_sphinx(nb_file):
    if False:
        i = 10
        return i + 15
    with open(nb_file) as stream:
        assert guess_format(stream.read(), ext='.py')[0] == 'sphinx'

def test_guess_format_hydrogen():
    if False:
        i = 10
        return i + 15
    text = '# %%\ncat hello.txt\n'
    assert guess_format(text, ext='.py')[0] == 'hydrogen'

def test_divine_format():
    if False:
        i = 10
        return i + 15
    assert divine_format('{"cells":[]}') == 'ipynb'
    assert divine_format('def f(x):\n    x + 1') == 'py:light'
    assert divine_format('# %%\ndef f(x):\n    x + 1\n\n# %%\ndef g(x):\n    x + 2\n') == 'py:percent'
    assert divine_format('This is a markdown file\nwith one code block\n\n```\n1 + 1\n```\n') == 'md'
    assert divine_format(';; ---\n;; jupyter:\n;;   jupytext:\n;;     text_representation:\n;;       extension: .ss\n;;       format_name: percent\n;; ---') == 'ss:percent'

def test_get_format_implementation():
    if False:
        for i in range(10):
            print('nop')
    assert get_format_implementation('.py').format_name == 'light'
    assert get_format_implementation('.py', 'percent').format_name == 'percent'
    with pytest.raises(JupytextFormatError):
        get_format_implementation('.py', 'wrong_format')

def test_script_with_magics_not_percent(script='# %%time\n1 + 2'):
    if False:
        i = 10
        return i + 15
    assert guess_format(script, '.py')[0] == 'light'

def test_script_with_spyder_cell_is_percent(script='#%%\n1 + 2'):
    if False:
        i = 10
        return i + 15
    assert guess_format(script, '.py')[0] == 'percent'

def test_script_with_percent_cell_and_magic_is_hydrogen(script='#%%\n%matplotlib inline\n'):
    if False:
        return 10
    assert guess_format(script, '.py')[0] == 'hydrogen'

def test_script_with_percent_cell_and_kernelspec(script='# ---\n# jupyter:\n#   kernelspec:\n#     display_name: Python3\n#     language: python\n#     name: python3\n# ---\n\n# %%\na = 1\n'):
    if False:
        print('Hello World!')
    assert guess_format(script, '.py')[0] == 'percent'

def test_script_with_spyder_cell_with_name_is_percent(script='#%% cell name\n1 + 2'):
    if False:
        for i in range(10):
            print('nop')
    assert guess_format(script, '.py')[0] == 'percent'

def test_read_format_from_metadata(script="---\njupyter:\n  jupytext:\n    formats: ipynb,pct.py:percent,lgt.py:light,spx.py:sphinx,md,Rmd\n    text_representation:\n      extension: .pct.py\n      format_name: percent\n      format_version: '1.1'\n      jupytext_version: 0.8.0\n---"):
    if False:
        i = 10
        return i + 15
    assert read_format_from_metadata(script, '.Rmd') is None

def test_update_jupytext_formats_metadata():
    if False:
        i = 10
        return i + 15
    nb = new_notebook(metadata={'jupytext': {'formats': 'py'}})
    update_jupytext_formats_metadata(nb.metadata, 'py:light')
    assert nb.metadata['jupytext']['formats'] == 'py:light'
    nb = new_notebook(metadata={'jupytext': {'formats': 'ipynb,py'}})
    update_jupytext_formats_metadata(nb.metadata, 'py:light')
    assert nb.metadata['jupytext']['formats'] == 'ipynb,py:light'

def test_decompress_formats():
    if False:
        return 10
    assert long_form_multiple_formats('ipynb') == [{'extension': '.ipynb'}]
    assert long_form_multiple_formats('ipynb,md') == [{'extension': '.ipynb'}, {'extension': '.md'}]
    assert long_form_multiple_formats('ipynb,py:light') == [{'extension': '.ipynb'}, {'extension': '.py', 'format_name': 'light'}]
    assert long_form_multiple_formats(['ipynb', '.py:light']) == [{'extension': '.ipynb'}, {'extension': '.py', 'format_name': 'light'}]
    assert long_form_multiple_formats('.pct.py:percent') == [{'extension': '.py', 'suffix': '.pct', 'format_name': 'percent'}]

def test_compress_formats():
    if False:
        while True:
            i = 10
    assert short_form_multiple_formats([{'extension': '.ipynb'}]) == 'ipynb'
    assert short_form_multiple_formats('ipynb') == 'ipynb'
    assert short_form_multiple_formats([{'extension': '.ipynb'}, {'extension': '.md'}]) == 'ipynb,md'
    assert short_form_multiple_formats([{'extension': '.ipynb'}, {'extension': '.py', 'format_name': 'light'}]) == 'ipynb,py:light'
    assert short_form_multiple_formats([{'extension': '.ipynb'}, {'extension': '.py', 'format_name': 'light'}, {'extension': '.md', 'comment_magics': True}]) == 'ipynb,py:light,md'
    assert short_form_multiple_formats([{'extension': '.py', 'suffix': '.pct', 'format_name': 'percent'}]) == '.pct.py:percent'

def test_rearrange_jupytext_metadata():
    if False:
        for i in range(10):
            print('nop')
    metadata = {'nbrmd_formats': 'ipynb,py'}
    rearrange_jupytext_metadata(metadata)
    compare(metadata, {'jupytext': {'formats': 'ipynb,py'}})
    metadata = {'jupytext_formats': 'ipynb,py'}
    rearrange_jupytext_metadata(metadata)
    compare(metadata, {'jupytext': {'formats': 'ipynb,py'}})
    metadata = {'executable': '#!/bin/bash'}
    rearrange_jupytext_metadata(metadata)
    compare(metadata, {'jupytext': {'executable': '#!/bin/bash'}})

def test_rearrange_jupytext_metadata_metadata_filter():
    if False:
        return 10
    metadata = {'jupytext': {'metadata_filter': {'notebook': {'additional': ['one', 'two'], 'excluded': 'all'}, 'cells': {'additional': 'all', 'excluded': ['three', 'four']}}}}
    rearrange_jupytext_metadata(metadata)
    compare(metadata, {'jupytext': {'notebook_metadata_filter': 'one,two,-all', 'cell_metadata_filter': 'all,-three,-four'}})

def test_rearrange_jupytext_metadata_add_dot_in_suffix():
    if False:
        print('Hello World!')
    metadata = {'jupytext': {'text_representation': {'jupytext_version': '0.8.6'}, 'formats': 'ipynb,pct.py,lgt.py'}}
    rearrange_jupytext_metadata(metadata)
    compare(metadata, {'jupytext': {'text_representation': {'jupytext_version': '0.8.6'}, 'formats': 'ipynb,.pct.py,.lgt.py'}})

def test_fix_139():
    if False:
        return 10
    text = '# ---\n# jupyter:\n#   jupytext:\n#     metadata_filter:\n#       cells:\n#         additional:\n#           - "lines_to_next_cell"\n#         excluded:\n#           - "all"\n# ---\n\n# + {"lines_to_next_cell": 2}\n1 + 1\n# -\n\n\n1 + 1\n'
    nb = jupytext.reads(text, 'py:light')
    text2 = jupytext.writes(nb, 'py:light')
    assert 'cell_metadata_filter: -all' in text2
    assert 'lines_to_next_cell' not in text2

def test_validate_one_format():
    if False:
        while True:
            i = 10
    with pytest.raises(JupytextFormatError):
        validate_one_format('py:percent')
    with pytest.raises(JupytextFormatError):
        validate_one_format({'extension': 'py', 'format_name': 'invalid'})
    with pytest.raises(JupytextFormatError):
        validate_one_format({})
    with pytest.raises(JupytextFormatError):
        validate_one_format({'extension': '.py', 'unknown_option': True})
    with pytest.raises(JupytextFormatError):
        validate_one_format({'extension': '.py', 'comment_magics': 'TRUE'})

def test_set_auto_ext():
    if False:
        return 10
    with pytest.raises(ValueError):
        long_form_multiple_formats('ipynb,auto:percent', {})

@requires_pandoc
def test_pandoc_format_is_preserved():
    if False:
        return 10
    formats_org = 'ipynb,md,.pandoc.md:pandoc,py:light'
    long = long_form_multiple_formats(formats_org)
    formats_new = short_form_multiple_formats(long)
    compare(formats_new, formats_org)

@requires_myst
def test_write_as_myst(tmpdir):
    if False:
        while True:
            i = 10
    'Inspired by https://github.com/mwouts/jupytext/issues/462'
    nb = new_notebook()
    tmp_md = str(tmpdir.join('notebook.md'))
    jupytext.write(nb, tmp_md, fmt='myst')
    with open(tmp_md) as fp:
        md = fp.read()
    assert 'myst' in md

def test_write_raises_when_fmt_does_not_exists(tmpdir):
    if False:
        for i in range(10):
            print('nop')
    'Inspired by https://github.com/mwouts/jupytext/issues/462'
    nb = new_notebook()
    tmp_md = str(tmpdir.join('notebook.md'))
    with pytest.raises(JupytextFormatError):
        jupytext.write(nb, tmp_md, fmt='unknown_format')

@pytest.mark.parametrize('config_file,config_contents', [('jupytext.toml', '# Always pair ipynb notebooks to md files\nformats = "ipynb,md"\n'), ('jupytext.toml', '# Always pair ipynb notebooks to py:percent files\nformats = "ipynb,py:percent"\n'), ('jupytext.toml', '# Always pair ipynb notebooks to py:percent files\nformats = ["ipynb", "py:percent"]\n'), ('pyproject.toml', '[tool.jupytext]\nformats = "ipynb,py:percent"\n'), ('jupytext.toml', '# Pair notebooks in subfolders of \'notebooks\' to scripts in subfolders of \'scripts\'\nformats = "notebooks///ipynb,scripts///py:percent"\n'), ('jupytext.toml', '[formats]\n"notebooks/" = "ipynb"\n"scripts/" = "py:percent"\n')])
def test_configuration_examples_from_documentation(config_file, config_contents, python_notebook, tmp_path):
    if False:
        print('Hello World!')
    'Here we make sure that the config examples from\n    https://jupytext.readthedocs.io/en/latest/config.html#configuring-paired-notebooks-globally\n    just work\n    '
    (tmp_path / config_file).write_text(config_contents)
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmp_path)
    (tmp_path / 'notebooks').mkdir()
    cm.save(dict(type='notebook', content=python_notebook), 'notebooks/nb.ipynb')
    assert (tmp_path / 'notebooks' / 'nb.ipynb').is_file()
    assert (tmp_path / 'notebooks' / 'nb.py').is_file() or (tmp_path / 'notebooks' / 'nb.md').is_file() or (tmp_path / 'scripts' / 'nb.py').is_file()