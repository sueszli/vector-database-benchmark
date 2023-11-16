"""Turn the examples section of a function docstring into a notebook."""
import re
import sys
import pydoc
import seaborn
from seaborn.external.docscrape import NumpyDocString
import nbformat

def line_type(line):
    if False:
        for i in range(10):
            print('nop')
    if line.startswith('    '):
        return 'code'
    else:
        return 'markdown'

def add_cell(nb, lines, cell_type):
    if False:
        print('Hello World!')
    cell_objs = {'code': nbformat.v4.new_code_cell, 'markdown': nbformat.v4.new_markdown_cell}
    text = '\n'.join(lines)
    cell = cell_objs[cell_type](text)
    nb['cells'].append(cell)
if __name__ == '__main__':
    (_, name) = sys.argv
    obj = getattr(seaborn, name)
    if obj.__class__.__name__ != 'function':
        obj = obj.__init__
    lines = NumpyDocString(pydoc.getdoc(obj))['Examples']
    pat = re.compile('\\s{4}[>\\.]{3} (ax = ){0,1}(g = ){0,1}')
    nb = nbformat.v4.new_notebook()
    cell_type = 'markdown'
    cell = []
    for line in lines:
        if '.. plot' in line or ':context:' in line:
            continue
        if not line:
            continue
        if line_type(line) != cell_type:
            add_cell(nb, cell, cell_type)
            cell_type = line_type(line)
            cell = []
        if line_type(line) == 'code':
            line = re.sub(pat, '', line)
        cell.append(line)
    add_cell(nb, cell, cell_type)
    nbformat.write(nb, f'docstrings/{name}.ipynb')