import pytest
from nbformat import NotebookNode
import jupytext
from jupytext.compare import compare, compare_cells
HEADER = {'.py': '# ---\n# jupyter:\n#   jupytext:\n#     main_language: python\n# ---\n\n', '.R': '# ---\n# jupyter:\n#   jupytext:\n#     main_language: python\n# ---\n\n', '.md': '---\njupyter:\n  jupytext:\n    main_language: python\n---\n\n', '.Rmd': '---\njupyter:\n  jupytext:\n    main_language: python\n---\n\n'}
ACTIVE_ALL = {'.py': '# + active="ipynb,py,R,Rmd"\n# This cell is active in all extensions\n', '.Rmd': '```{python active="ipynb,py,R,Rmd"}\n# This cell is active in all extensions\n```\n', '.md': '```python active="ipynb,py,R,Rmd"\n# This cell is active in all extensions\n```\n', '.R': '# + active="ipynb,py,R,Rmd"\n# This cell is active in all extensions\n', '.ipynb': {'cell_type': 'code', 'source': '# This cell is active in all extensions', 'metadata': {'active': 'ipynb,py,R,Rmd'}, 'execution_count': None, 'outputs': []}}

def check_active_cell(ext, active_dict):
    if False:
        while True:
            i = 10
    text = ('' if ext == '.py' else HEADER[ext]) + active_dict[ext]
    nb = jupytext.reads(text, ext)
    assert len(nb.cells) == 1
    compare(jupytext.writes(nb, ext), text)
    cell = NotebookNode(active_dict['.ipynb'])
    compare_cells(nb.cells, [cell], compare_ids=False)

@pytest.mark.parametrize('ext', ['.Rmd', '.md', '.py', '.R'])
def test_active_all(ext, no_jupytext_version_number):
    if False:
        return 10
    check_active_cell(ext, ACTIVE_ALL)
ACTIVE_IPYNB = {'.py': '# + active="ipynb"\n# # This cell is active only in ipynb\n# %matplotlib inline\n', '.Rmd': '```{python active="ipynb", eval=FALSE}\n# This cell is active only in ipynb\n%matplotlib inline\n```\n', '.md': '```python active="ipynb"\n# This cell is active only in ipynb\n%matplotlib inline\n```\n', '.R': '# + active="ipynb"\n# # This cell is active only in ipynb\n# %matplotlib inline\n', '.ipynb': {'cell_type': 'code', 'source': '# This cell is active only in ipynb\n%matplotlib inline', 'metadata': {'active': 'ipynb'}, 'execution_count': None, 'outputs': []}}

@pytest.mark.parametrize('ext', ['.Rmd', '.md', '.py', '.R'])
def test_active_ipynb(ext, no_jupytext_version_number):
    if False:
        while True:
            i = 10
    check_active_cell(ext, ACTIVE_IPYNB)
ACTIVE_IPYNB_RMD_USING_TAG = {'.py': '# + tags=["active-ipynb-Rmd"]\n# # This cell is active only in ipynb and Rmd\n# %matplotlib inline\n', '.Rmd': '```{python tags=c("active-ipynb-Rmd")}\n# This cell is active only in ipynb and Rmd\n# %matplotlib inline\n```\n', '.md': '```python tags=["active-ipynb-Rmd"]\n# This cell is active only in ipynb and Rmd\n%matplotlib inline\n```\n', '.R': '# + tags=["active-ipynb-Rmd"]\n# # This cell is active only in ipynb and Rmd\n# %matplotlib inline\n', '.ipynb': {'cell_type': 'code', 'source': '# This cell is active only in ipynb and Rmd\n%matplotlib inline', 'metadata': {'tags': ['active-ipynb-Rmd']}, 'execution_count': None, 'outputs': []}}

@pytest.mark.parametrize('ext', ['.Rmd', '.md', '.py', '.R'])
def test_active_ipynb_rmd_using_tags(ext, no_jupytext_version_number):
    if False:
        return 10
    check_active_cell(ext, ACTIVE_IPYNB_RMD_USING_TAG)
ACTIVE_IPYNB_RSPIN = {'.R': '#+ active="ipynb", eval=FALSE\n# # This cell is active only in ipynb\n# 1 + 1\n', '.ipynb': {'cell_type': 'code', 'source': '# This cell is active only in ipynb\n1 + 1', 'metadata': {'active': 'ipynb'}, 'execution_count': None, 'outputs': []}}

def test_active_ipynb_rspin(no_jupytext_version_number):
    if False:
        i = 10
        return i + 15
    nb = jupytext.reads(ACTIVE_IPYNB_RSPIN['.R'], 'R:spin')
    assert len(nb.cells) == 1
    compare(jupytext.writes(nb, 'R:spin'), ACTIVE_IPYNB_RSPIN['.R'])
    cell = NotebookNode(ACTIVE_IPYNB_RSPIN['.ipynb'])
    compare_cells(nb.cells, [cell], compare_ids=False)
ACTIVE_PY_IPYNB = {'.py': '# + active="ipynb,py"\n# This cell is active in py and ipynb extensions\n', '.Rmd': '```{python active="ipynb,py", eval=FALSE}\n# This cell is active in py and ipynb extensions\n```\n', '.md': '```python active="ipynb,py"\n# This cell is active in py and ipynb extensions\n```\n', '.R': '# + active="ipynb,py"\n# # This cell is active in py and ipynb extensions\n', '.ipynb': {'cell_type': 'code', 'source': '# This cell is active in py and ipynb extensions', 'metadata': {'active': 'ipynb,py'}, 'execution_count': None, 'outputs': []}}

@pytest.mark.parametrize('ext', ['.Rmd', '.md', '.py', '.R'])
def test_active_py_ipynb(ext, no_jupytext_version_number):
    if False:
        i = 10
        return i + 15
    check_active_cell(ext, ACTIVE_PY_IPYNB)
ACTIVE_PY_R_IPYNB = {'.py': '# + active="ipynb,py,R"\n# This cell is active in py, R and ipynb extensions\n', '.Rmd': '```{python active="ipynb,py,R", eval=FALSE}\n# This cell is active in py, R and ipynb extensions\n```\n', '.R': '# + active="ipynb,py,R"\n# This cell is active in py, R and ipynb extensions\n', '.ipynb': {'cell_type': 'code', 'source': '# This cell is active in py, R and ipynb extensions', 'metadata': {'active': 'ipynb,py,R'}, 'execution_count': None, 'outputs': []}}

@pytest.mark.parametrize('ext', ['.Rmd', '.py', '.R'])
def test_active_py_r_ipynb(ext, no_jupytext_version_number):
    if False:
        while True:
            i = 10
    check_active_cell(ext, ACTIVE_PY_R_IPYNB)
ACTIVE_RMD = {'.py': '# + active="Rmd"\n# # This cell is active in Rmd only\n', '.Rmd': '```{python active="Rmd"}\n# This cell is active in Rmd only\n```\n', '.R': '# + active="Rmd"\n# # This cell is active in Rmd only\n', '.ipynb': {'cell_type': 'raw', 'source': '# This cell is active in Rmd only', 'metadata': {'active': 'Rmd'}}}

@pytest.mark.parametrize('ext', ['.Rmd', '.py', '.R'])
def test_active_rmd(ext, no_jupytext_version_number):
    if False:
        i = 10
        return i + 15
    check_active_cell(ext, ACTIVE_RMD)
ACTIVE_NOT_INCLUDE_RMD = {'.py': '# + tags=["remove_cell"] active="Rmd"\n# # This cell is active in Rmd only\n', '.Rmd': '```{python include=FALSE, active="Rmd"}\n# This cell is active in Rmd only\n```\n', '.R': '# + tags=["remove_cell"] active="Rmd"\n# # This cell is active in Rmd only\n', '.ipynb': {'cell_type': 'raw', 'source': '# This cell is active in Rmd only', 'metadata': {'active': 'Rmd', 'tags': ['remove_cell']}}}

@pytest.mark.parametrize('ext', ['.Rmd', '.py', '.R'])
def test_active_not_include_rmd(ext, no_jupytext_version_number):
    if False:
        while True:
            i = 10
    check_active_cell(ext, ACTIVE_NOT_INCLUDE_RMD)

def test_active_cells_from_py_percent(text='# %% active="py"\nprint(\'should only be displayed in py file\')\n\n# %% tags=["active-py"]\nprint(\'should only be displayed in py file\')\n\n# %% active="ipynb"\n# print(\'only in jupyter\')\n'):
    if False:
        return 10
    'Example taken from https://github.com/mwouts/jupytext/issues/477'
    nb = jupytext.reads(text, 'py:percent')
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[1].cell_type == 'raw'
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[2].source == "print('only in jupyter')"
    text2 = jupytext.writes(nb, 'py:percent')
    compare(text2, text)

def test_active_cells_from_py_light(text='# + active="py"\nprint(\'should only be displayed in py file\')\n\n# + tags=["active-py"]\nprint(\'should only be displayed in py file\')\n\n# + active="ipynb"\n# print(\'only in jupyter\')\n'):
    if False:
        return 10
    'Example adapted from https://github.com/mwouts/jupytext/issues/477'
    nb = jupytext.reads(text, 'py')
    assert nb.cells[0].cell_type == 'raw'
    assert nb.cells[1].cell_type == 'raw'
    assert nb.cells[2].cell_type == 'code'
    assert nb.cells[2].source == "print('only in jupyter')"
    text2 = jupytext.writes(nb, 'py')
    compare(text2, text)