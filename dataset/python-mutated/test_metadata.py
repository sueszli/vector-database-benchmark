"""
1. Read metadata and compare with stored expected result.
2. Erase metadata and assert object has indeed been deleted.
"""
import json
import os
import fitz
scriptdir = os.path.abspath(os.path.dirname(__file__))
filename = os.path.join(scriptdir, 'resources', '001003ED.pdf')
metafile = os.path.join(scriptdir, 'resources', 'metadata.txt')
doc = fitz.open(filename)

def test_metadata():
    if False:
        print('Hello World!')
    assert json.dumps(doc.metadata) == open(metafile).read()

def test_erase_meta():
    if False:
        i = 10
        return i + 15
    doc.set_metadata({})
    statement1 = doc.xref_get_key(-1, 'Info')[1] == 'null'
    statement2 = 'Info' not in doc.xref_get_keys(-1)
    assert statement2 or statement1