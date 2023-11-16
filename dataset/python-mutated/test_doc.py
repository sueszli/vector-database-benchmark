from __future__ import annotations
import pytest
pytest
import weakref
from bokeh.document import Document
from bokeh.io.state import curstate
import bokeh.io.doc as bid

def test_curdoc_from_curstate() -> None:
    if False:
        return 10
    assert bid.curdoc() is curstate().document

def test_set_curdoc_sets_curstate() -> None:
    if False:
        while True:
            i = 10
    d = Document()
    bid.set_curdoc(d)
    assert curstate().document is d

def test_patch_curdoc() -> None:
    if False:
        print('Hello World!')
    d1 = Document()
    d2 = Document()
    orig_doc = bid.curdoc()
    assert bid._PATCHED_CURDOCS == []
    with bid.patch_curdoc(d1):
        assert len(bid._PATCHED_CURDOCS) == 1
        assert isinstance(bid._PATCHED_CURDOCS[0], weakref.ReferenceType)
        assert bid.curdoc() is d1
        with bid.patch_curdoc(d2):
            assert len(bid._PATCHED_CURDOCS) == 2
            assert isinstance(bid._PATCHED_CURDOCS[1], weakref.ReferenceType)
            assert bid.curdoc() is d2
        assert len(bid._PATCHED_CURDOCS) == 1
        assert isinstance(bid._PATCHED_CURDOCS[0], weakref.ReferenceType)
        assert bid.curdoc() is d1
    assert bid.curdoc() is orig_doc

def _doc():
    if False:
        while True:
            i = 10
    return Document()

def test_patch_curdoc_weakref_raises() -> None:
    if False:
        for i in range(10):
            print('nop')
    with bid.patch_curdoc(_doc()):
        with pytest.raises(RuntimeError) as e:
            bid.curdoc()
            assert str(e) == 'Patched curdoc has been previously destroyed'