"""
Check some low-level PDF object manipulations:
1. Set page rotation and compare with string in object definition.
2. Set page rotation via string manipulation and compare with result of
   proper page property.
3. Read the PDF trailer and verify it has the keys "/Root", "/ID", etc.
"""
import fitz
import os
scriptdir = os.path.abspath(os.path.dirname(__file__))
resources = os.path.join(scriptdir, 'resources')
filename = os.path.join(resources, '001003ED.pdf')

def test_rotation1():
    if False:
        while True:
            i = 10
    doc = fitz.open()
    page = doc.new_page()
    page.set_rotation(270)
    assert doc.xref_get_key(page.xref, 'Rotate') == ('int', '270')

def test_rotation2():
    if False:
        print('Hello World!')
    doc = fitz.open()
    page = doc.new_page()
    doc.xref_set_key(page.xref, 'Rotate', '270')
    assert page.rotation == 270

def test_trailer():
    if False:
        print('Hello World!')
    'Access PDF trailer information.'
    doc = fitz.open(filename)
    xreflen = doc.xref_length()
    (_, xreflen_str) = doc.xref_get_key(-1, 'Size')
    assert xreflen == int(xreflen_str)
    trailer_keys = doc.xref_get_keys(-1)
    assert 'ID' in trailer_keys
    assert 'Root' in trailer_keys

def test_valid_name():
    if False:
        i = 10
        return i + 15
    'Verify correct PDF names in method xref_set_key.'
    doc = fitz.open()
    page = doc.new_page()
    doc.xref_set_key(page.xref, 'Rotate', '90')
    assert page.rotation == 90
    error_generated = False
    try:
        doc.xref_set_key(page.xref, 'my rotate', '90')
    except ValueError as e:
        assert str(e) == "bad 'key'"
        error_generated = True
    assert error_generated
    doc.xref_set_key(page.xref, 'my_rotate/something', '90')
    assert doc.xref_get_key(page.xref, 'my_rotate/something') == ('int', '90')
    doc.xref_set_key(page.xref, 'my_rotate', '/90')
    assert doc.xref_get_key(page.xref, 'my_rotate') == ('name', '/90')
    error_generated = False
    try:
        doc.xref_set_key(page.xref, 'my_rotate', '/9/0')
    except ValueError as e:
        assert str(e) == "bad 'value'"
        error_generated = True
    assert error_generated