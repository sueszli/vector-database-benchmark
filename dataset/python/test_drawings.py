"""
Extract drawings of a PDF page and compare with stored expected result.
"""
import io
import os
import sys
import pprint

import fitz

scriptdir = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(scriptdir, "resources", "symbol-list.pdf")
symbols = os.path.join(scriptdir, "resources", "symbols.txt")


def test_drawings1():
    symbols_text = open(symbols).read()  # expected result
    doc = fitz.open(filename)
    page = doc[0]
    paths = page.get_cdrawings()
    out = io.StringIO()  # pprint output goes here
    pprint.pprint(paths, stream=out)
    assert symbols_text == out.getvalue()


def test_drawings2():
    delta = (0, 20, 0, 20)
    doc = fitz.open()
    page = doc.new_page()

    r = fitz.Rect(100, 100, 200, 200)
    page.draw_circle(r.br, 2, color=0)
    r += delta

    page.draw_line(r.tl, r.br, color=0)
    r += delta

    page.draw_oval(r, color=0)
    r += delta

    page.draw_rect(r, color=0)
    r += delta

    page.draw_quad(r.quad, color=0)
    r += delta

    page.draw_polyline((r.tl, r.tr, r.br), color=0)
    r += delta

    page.draw_bezier(r.tl, r.tr, r.br, r.bl, color=0)
    r += delta

    page.draw_curve(r.tl, r.tr, r.br, color=0)
    r += delta

    page.draw_squiggle(r.tl, r.br, color=0)
    r += delta

    rects = [p["rect"] for p in page.get_cdrawings()]
    bboxes = [b[1] for b in page.get_bboxlog()]
    for i, r in enumerate(rects):
        assert fitz.Rect(r) in fitz.Rect(bboxes[i])


def _dict_difference(a, b):
    '''
    Returns `(keys_a, keys_b, key_values)`, information about differences
    between dicts `a` and `b`.
    
    `keys_a` is the set of keys that are in `a` but not in `b`.

    `keys_b` is the set of keys that are in `b` but not in `a`.

    `key_values` is a dict with keys that are in both `a` and `b` but where the
    values differ; the values in this dict are `(value_a, value_b)`.
    '''
    keys_a = set()
    keys_b = set()
    key_values = dict()
    for key in a:
        if key not in b:
            keys_a.add( key)
    for key in b:
        if key not in a:
            keys_b.add( key)
    for key, va in a.items():
        if key in b:
            vb = b[key]
            if va != vb:
                key_values[key] = (va, vb)
    return keys_a, keys_b, key_values


def test_drawings3():
    doc = fitz.open()

    page1 = doc.new_page()
    shape1 = page1.new_shape()
    shape1.draw_line((10, 10), (10, 50))
    shape1.draw_line((10, 50), (100, 100))
    shape1.finish(closePath=False, color=(0,0,0), width=5)
    shape1.commit()
    drawings1 = list(page1.get_drawings())

    page2 = doc.new_page()
    shape2 = page2.new_shape()
    shape2.draw_line((10, 10), (10, 50))
    shape2.draw_line((10, 50), (100, 100))
    shape2.finish(closePath=True, color=(0,0,0), width=5)
    shape2.commit()
    drawings2 = list(page2.get_drawings())

    page3 = doc.new_page()
    shape3 = page3.new_shape()
    shape3.draw_line((10, 10), (10, 50))
    shape3.draw_line((10, 50), (100, 100))
    shape3.draw_line((100, 100), (50, 70))
    shape3.finish(closePath=False, color=(0,0,0), width=5)
    shape3.commit()
    drawings3 = list(page3.get_drawings())

    page4 = doc.new_page()
    shape4 = page4.new_shape()
    shape4.draw_line((10, 10), (10, 50))
    shape4.draw_line((10, 50), (100, 100))
    shape4.draw_line((100, 100), (50, 70))
    shape4.finish(closePath=True, color=(0,0,0), width=5)
    shape4.commit()
    drawings4 = list(page4.get_drawings())

    assert len(drawings1) == len(drawings2) == 1
    drawings1 = drawings1[0]
    drawings2 = drawings2[0]
    diff = _dict_difference( drawings1, drawings2)
    assert diff == (set(), set(), {'closePath': (False, True)})
    
    assert len(drawings3) == len(drawings4) == 1
    drawings3 = drawings3[0]
    drawings4 = drawings4[0]
    diff = _dict_difference( drawings3, drawings4)
    assert diff == (set(), set(), {'closePath': (False, True)})
    
def test_2365():
    """Draw a filled rectangle on a new page.

    Then extract the page's vector graphics and confirm that only one path
    was generated which has all the right properties."""
    doc = fitz.open()
    page = doc.new_page()
    rect = fitz.Rect(100, 100, 200, 200)
    page.draw_rect(
        rect, color=fitz.pdfcolor["black"], fill=fitz.pdfcolor["yellow"], width=3
    )
    paths = page.get_drawings()
    assert len(paths) == 1
    path = paths[0]
    assert path["type"] == "fs"
    assert path["fill"] == fitz.pdfcolor["yellow"]
    assert path["fill_opacity"] == 1
    assert path["color"] == fitz.pdfcolor["black"]
    assert path["stroke_opacity"] == 1
    assert path["width"] == 3
    assert path["rect"] == rect

def test_2462():
    """
    Assertion happens, if this code does NOT bring down the interpreter.

    Background:
    We previously ignored clips for non-vector-graphics. However, ending
    a clip does not refer back the object(s) that have been clipped.
    In order to correctly compute the "scissor" rectangle, we now keep track
    of the clipped object type.
    """
    doc = fitz.open(f"{scriptdir}/resources/test-2462.pdf")
    page = doc[0]
    vg = page.get_drawings(extended=True)

def test_2556():
    """Ensure that incomplete clip paths will be properly ignored."""
    doc = fitz.open()  # new empty PDF
    page = doc.new_page()  # new page
    # following contains an incomplete clip
    c = b"q 50 697.6 400 100.0 re W n q 0 0 m W n Q "
    xref = doc.get_new_xref()  # prepare /Contents object for page
    doc.update_object(xref,"<<>>")  # new xref now is a dictionary
    doc.update_stream(xref, c)  # store drawing commands
    page.set_contents(xref)  # give the page this xref as /Contents
    # following will bring down interpreter if fix not installed
    assert page.get_drawings(extended=True)