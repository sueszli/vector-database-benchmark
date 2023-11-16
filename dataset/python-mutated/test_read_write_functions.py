from io import StringIO
from pathlib import Path
import nbformat
from nbformat.v4.nbbase import new_markdown_cell, new_notebook
import jupytext
from jupytext.compare import compare, compare_notebooks

def test_simple_hook(tmpdir):
    if False:
        i = 10
        return i + 15
    nb_file = str(tmpdir.join('notebook.ipynb'))
    md_file = str(tmpdir.join('notebook.md'))
    nbformat.write(new_notebook(cells=[new_markdown_cell('Some text')]), nb_file)
    nb = jupytext.read(nb_file)
    jupytext.write(nb, md_file)
    with open(md_file) as fp:
        text = fp.read()
    assert 'Some text' in text.splitlines()

def test_simple_hook_with_explicit_format(tmpdir):
    if False:
        while True:
            i = 10
    nb_file = str(tmpdir.join('notebook.ipynb'))
    py_file = str(tmpdir.join('notebook.py'))
    nbformat.write(new_notebook(cells=[new_markdown_cell('Some text')]), nb_file)
    nb = jupytext.read(nb_file)
    jupytext.write(nb, py_file, fmt='py:percent')
    with open(py_file) as fp:
        text = fp.read()
    assert '# %% [markdown]' in text.splitlines()
    assert '# Some text' in text.splitlines()

def test_no_error_on_path_object(tmpdir):
    if False:
        i = 10
        return i + 15
    nb_file = Path(str(tmpdir.join('notebook.ipynb')))
    md_file = nb_file.with_suffix('.md')
    nbformat.write(new_notebook(cells=[new_markdown_cell('Some text')]), str(nb_file))
    nb = jupytext.read(nb_file)
    jupytext.write(nb, md_file)

def test_read_ipynb_from_stream():
    if False:
        for i in range(10):
            print('nop')

    def stream():
        if False:
            i = 10
            return i + 15
        return StringIO('{\n "cells": [\n  {\n   "cell_type": "code",\n   "metadata": {},\n   "source": [\n    "1 + 1"\n   ]\n  }\n ],\n "metadata": {},\n "nbformat": 4,\n "nbformat_minor": 4\n}\n')
    nb = jupytext.read(stream())
    nb2 = jupytext.read(stream(), fmt='ipynb')
    compare(nb2, nb)

def test_read_py_percent_from_stream():
    if False:
        while True:
            i = 10

    def stream():
        if False:
            return 10
        return StringIO('# %%\n1 + 1\n')
    nb = jupytext.read(stream())
    nb2 = jupytext.read(stream(), fmt='py:percent')
    compare_notebooks(nb2, nb)