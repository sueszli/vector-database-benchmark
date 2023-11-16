"""Helper functions to add documentation to type aliases."""
from typing import Dict
_EXTRA_DOCS: Dict[int, str] = {}

def document(obj, doc):
    if False:
        i = 10
        return i + 15
    'Adds a docstring to typealias by overriding the `__doc__` attribute.\n\n  Note: Overriding `__doc__` is only possible after python 3.7.\n\n  Args:\n    obj: Typealias object that needs to be documented.\n    doc: Docstring of the typealias. It should follow the standard pystyle\n      docstring rules.\n  '
    try:
        obj.__doc__ = doc
    except AttributeError:
        _EXTRA_DOCS[id(obj)] = doc