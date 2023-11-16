"""Dynamically compile classes for all HTML elements and output them to the
elements directory.

This script generates the element classes in the reflex.el.elements module.
Run as follows:

    python -m reflex.el.precompile

Make sure to delete the __init__.py file in the elements directory before
running this script.
"""
import os
from reflex.utils import path_ops
from .constants import ELEMENT_TO_PROPS, ELEMENTS
FILE_DIR = os.path.dirname(os.path.realpath(__file__))
ELEMENTS_DIR = os.path.join(FILE_DIR, 'elements')
INIT_PY_PATH = os.path.join(ELEMENTS_DIR, '__init__.py')

def element_path(element: str) -> str:
    if False:
        return 10
    'Get the name of the Python file for the given element.\n\n    Args:\n        element: The name of the element. For example, `a` or `div`.\n\n    Returns:\n        The name of the Python file for the given element.\n    '
    return os.path.join(ELEMENTS_DIR, f'{element}.py')
PROP = '    {prop}: Var_[Union[str, int, bool]]'.format

def compile_pyclass_props(element: str) -> str:
    if False:
        return 10
    'Compile props for an element.\n\n    Args:\n        element: The name of the element. For example, `a` or `div`.\n\n    Returns:\n        A string containing compiled props for the element.\n    '
    return path_ops.join((PROP(prop=prop) for prop in ELEMENT_TO_PROPS[element]))
PYCLASS = path_ops.join(['', 'class {name}(Element):  # noqa: E742', '    """Display the {element} element."""', '', '    tag = "{element}"', '', '{props}', '', '', '{call_name} = {name}.create', '']).format

def compile_pyclass(element: str) -> str:
    if False:
        for i in range(10):
            print('nop')
    'Compile a Python class for an element.\n\n    Args:\n        element: The name of the element. For example, `a` or `div`.\n\n    Returns:\n        A string containing a Python class for the element.\n    '
    name = element.capitalize()
    props = compile_pyclass_props(element)
    call_name = 'del_' if element == 'del' else element
    return PYCLASS(name=name, element=element, props=props, call_name=call_name)
INIT_PY = ['"""Element classes. This is an auto-generated file. Do not edit. See ../generate.py."""', 'from typing import Union', '', 'from reflex.el.element import Element', 'from reflex.vars import Var as Var_', '']
for element in sorted(ELEMENTS):
    INIT_PY.append(compile_pyclass(element))
os.makedirs(ELEMENTS_DIR, exist_ok=True)
with open(INIT_PY_PATH, 'w+') as f:
    f.write(path_ops.join(INIT_PY))