from copy import deepcopy
import pytest
from nbformat.v4.nbbase import new_code_cell, new_markdown_cell, new_notebook
import jupytext
from jupytext.combine import combine_inputs_with_outputs
from jupytext.compare import compare, compare_notebooks
from .utils import list_notebooks

def test_combine():
    if False:
        return 10
    nb_source = new_notebook(cells=[new_markdown_cell('Markdown text'), new_code_cell('a=3'), new_code_cell('a+1'), new_code_cell('a+1'), new_markdown_cell('Markdown text'), new_code_cell('a+2')])
    nb_outputs = new_notebook(cells=[new_markdown_cell('Markdown text'), new_code_cell('a=3'), new_code_cell('a+1'), new_code_cell('a+2'), new_markdown_cell('Markdown text')])
    nb_outputs.cells[2].outputs = ['4']
    nb_outputs.cells[3].outputs = ['5']
    nb_source = combine_inputs_with_outputs(nb_source, nb_outputs)
    assert nb_source.cells[2].outputs == ['4']
    assert nb_source.cells[3].outputs == []
    assert nb_source.cells[5].outputs == ['5']

def test_read_text_and_combine_with_outputs(tmpdir):
    if False:
        return 10
    tmp_ipynb = 'notebook.ipynb'
    tmp_script = 'notebook.py'
    with open(str(tmpdir.join(tmp_script)), 'w') as fp:
        fp.write('# ---\n# jupyter:\n#   jupytext_formats: ipynb,py:light\n# ---\n\n1+1\n\n2+2\n\n3+3\n')
    with open(str(tmpdir.join(tmp_ipynb)), 'w') as fp:
        fp.write('{\n "cells": [\n  {\n   "cell_type": "code",\n   "execution_count": 1,\n   "metadata": {},\n   "outputs": [\n    {\n     "data": {\n      "text/plain": [\n       "2"\n      ]\n     },\n     "execution_count": 1,\n     "metadata": {},\n     "output_type": "execute_result"\n    }\n   ],\n   "source": [\n    "1+1"\n   ]\n  },\n  {\n   "cell_type": "code",\n   "execution_count": 3,\n   "metadata": {},\n   "outputs": [\n    {\n     "data": {\n      "text/plain": [\n       "6"\n      ]\n     },\n     "execution_count": 3,\n     "metadata": {},\n     "output_type": "execute_result"\n    }\n   ],\n   "source": [\n    "3+3"\n   ]\n  }\n ],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 2\n}\n')
    cm = jupytext.TextFileContentsManager()
    cm.root_dir = str(tmpdir)
    model = cm.get(tmp_script)
    nb = model['content']
    assert nb.cells[0]['source'] == '1+1'
    assert nb.cells[1]['source'] == '2+2'
    assert nb.cells[2]['source'] == '3+3'
    assert nb.cells[0]['outputs']
    assert not nb.cells[1]['outputs']
    assert nb.cells[2]['outputs']
    assert len(nb.cells) == 3

@pytest.mark.parametrize('nb_file', list_notebooks('ipynb_all'))
def test_combine_stable(nb_file):
    if False:
        i = 10
        return i + 15
    nb_org = jupytext.read(nb_file)
    nb_source = deepcopy(nb_org)
    nb_outputs = deepcopy(nb_org)
    for cell in nb_source.cells:
        cell.outputs = []
    nb_source = combine_inputs_with_outputs(nb_source, nb_outputs)
    compare_notebooks(nb_source, nb_org)

def test_combine_reorder():
    if False:
        for i in range(10):
            print('nop')
    nb_source = new_notebook(cells=[new_markdown_cell('Markdown text'), new_code_cell('1+1'), new_code_cell('2+2'), new_code_cell('3+3'), new_markdown_cell('Markdown text'), new_code_cell('4+4')])
    nb_outputs = new_notebook(cells=[new_markdown_cell('Markdown text'), new_code_cell('2+2'), new_code_cell('4+4'), new_code_cell('1+1'), new_code_cell('3+3'), new_markdown_cell('Markdown text')])
    nb_outputs.cells[1].outputs = ['4']
    nb_outputs.cells[2].outputs = ['8']
    nb_outputs.cells[3].outputs = ['2']
    nb_outputs.cells[4].outputs = ['6']
    nb_source = combine_inputs_with_outputs(nb_source, nb_outputs)
    assert nb_source.cells[1].outputs == ['2']
    assert nb_source.cells[2].outputs == ['4']
    assert nb_source.cells[3].outputs == ['6']
    assert nb_source.cells[5].outputs == ['8']

def test_combine_split():
    if False:
        print('Hello World!')
    nb_source = new_notebook(cells=[new_code_cell('1+1'), new_code_cell('2+2')])
    nb_outputs = new_notebook(cells=[new_code_cell('1+1\n2+2')])
    nb_outputs.cells[0].outputs = ['4']
    nb_source = combine_inputs_with_outputs(nb_source, nb_outputs)
    assert nb_source.cells[0].outputs == []
    assert nb_source.cells[1].outputs == ['4']

def test_combine_refactor():
    if False:
        for i in range(10):
            print('nop')
    nb_source = new_notebook(cells=[new_code_cell('a=1'), new_code_cell('a+1'), new_code_cell('a+2')])
    nb_outputs = new_notebook(cells=[new_code_cell('b=1'), new_code_cell('b+1'), new_code_cell('b+2')])
    nb_outputs.cells[1].outputs = ['2']
    nb_outputs.cells[2].outputs = ['3']
    nb_source = combine_inputs_with_outputs(nb_source, nb_outputs)
    assert nb_source.cells[0].outputs == []
    assert nb_source.cells[1].outputs == ['2']
    assert nb_source.cells[2].outputs == ['3']

def test_combine_attachments():
    if False:
        i = 10
        return i + 15
    nb_source = new_notebook(cells=[new_markdown_cell('![image.png](attachment:image.png)')])
    nb_outputs = new_notebook(cells=[new_markdown_cell('![image.png](attachment:image.png)', attachments={'image.png': {'image/png': 'SOME_LONG_IMAGE_CODE...=='}})])
    nb_source = combine_inputs_with_outputs(nb_source, nb_outputs)
    compare(nb_source, nb_outputs)