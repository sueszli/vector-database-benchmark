import os.path
import textwrap
from pyqtgraph.parametertree.Parameter import PARAM_TYPES, _PARAM_ITEM_TYPES

def mkDocs(typeList):
    if False:
        print('Hello World!')
    typeNames = sorted([typ.__name__ for typ in typeList])
    typDocs = [f'    .. autoclass:: {name}\n       :members:\n    ' for name in typeNames]
    indented = '\n'.join(typDocs)
    return textwrap.dedent(indented)[:-1]
types = set(PARAM_TYPES.values())
items = [typ.itemClass for typ in PARAM_TYPES.values() if typ.itemClass is not None] + [item for item in _PARAM_ITEM_TYPES.values()]
items = set(items)
doc = f'..\n  This file is auto-generated from pyqtgraph/tools/rebuildPtreeRst.py. Do not modify by hand! Instead, rerun the\n  generation script with `python pyqtgraph/tools/rebuildPtreeRst.py`.\n\nBuilt-in Parameter Types\n========================\n\n.. currentmodule:: pyqtgraph.parametertree.parameterTypes\n\nParameters\n----------\n\n{mkDocs(types)}\n\nParameterItems\n--------------\n\n{mkDocs(items)}\n'
here = os.path.dirname(__file__)
rstFilename = os.path.join(here, '..', 'doc', 'source', 'parametertree', 'parametertypes.rst')
with open(rstFilename, 'w') as ofile:
    ofile.write(doc)