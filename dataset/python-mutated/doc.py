"""

"""
from __future__ import annotations
import logging
log = logging.getLogger(__name__)
import weakref
from contextlib import contextmanager
from typing import TYPE_CHECKING, Iterator, cast
from ..document import Document
from .state import curstate
if TYPE_CHECKING:
    from ..document.locking import UnlockedDocumentProxy
__all__ = ('curdoc', 'patch_curdoc', 'set_curdoc')

def curdoc() -> Document:
    if False:
        while True:
            i = 10
    ' Return the document for the current default state.\n\n    Returns:\n        Document : the current default document object.\n\n    '
    if len(_PATCHED_CURDOCS) > 0:
        doc = _PATCHED_CURDOCS[-1]()
        if doc is None:
            raise RuntimeError('Patched curdoc has been previously destroyed')
        return cast(Document, doc)
    return curstate().document

@contextmanager
def patch_curdoc(doc: Document | UnlockedDocumentProxy) -> Iterator[None]:
    if False:
        while True:
            i = 10
    ' Temporarily override the value of ``curdoc()`` and then return it to\n    its original state.\n\n    This context manager is useful for controlling the value of ``curdoc()``\n    while invoking functions (e.g. callbacks). The cont\n\n    Args:\n        doc (Document) : new Document to use for ``curdoc()``\n\n    '
    global _PATCHED_CURDOCS
    _PATCHED_CURDOCS.append(weakref.ref(doc))
    del doc
    yield
    _PATCHED_CURDOCS.pop()

def set_curdoc(doc: Document) -> None:
    if False:
        while True:
            i = 10
    ' Configure the current document (returned by curdoc()).\n\n    Args:\n        doc (Document) : new Document to use for curdoc()\n\n    Returns:\n        None\n\n    .. warning::\n        Calling this function will replace any existing document.\n\n    '
    curstate().document = doc
_PATCHED_CURDOCS: list[weakref.ReferenceType[Document | UnlockedDocumentProxy]] = []