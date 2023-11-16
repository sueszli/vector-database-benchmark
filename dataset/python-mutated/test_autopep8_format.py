import pytest
from pylsp import uris
from pylsp.plugins.autopep8_format import pylsp_format_document, pylsp_format_range
from pylsp.workspace import Document
DOC_URI = uris.from_fs_path(__file__)
DOC = 'a =    123\n\n\n\n\ndef func():\n    pass\n'
GOOD_DOC = "A = ['hello', 'world']\n"
INDENTED_DOC = "def foo():\n    print('asdf',\n    file=None\n    )\n\nbar = { 'foo': foo\n}\n"
CORRECT_INDENTED_DOC = "def foo():\n    print('asdf',\n          file=None\n          )\n\n\nbar = {'foo': foo\n       }\n"

def test_format(config, workspace):
    if False:
        return 10
    doc = Document(DOC_URI, workspace, DOC)
    res = pylsp_format_document(config, workspace, doc, options=None)
    assert len(res) == 1
    assert res[0]['newText'] == 'a = 123\n\n\ndef func():\n    pass\n'

def test_range_format(config, workspace):
    if False:
        return 10
    doc = Document(DOC_URI, workspace, DOC)
    def_range = {'start': {'line': 0, 'character': 0}, 'end': {'line': 2, 'character': 0}}
    res = pylsp_format_range(config, workspace, doc, def_range, options=None)
    assert len(res) == 1
    assert res[0]['newText'] == 'a = 123\n\n\n\n\ndef func():\n    pass\n'

def test_no_change(config, workspace):
    if False:
        i = 10
        return i + 15
    doc = Document(DOC_URI, workspace, GOOD_DOC)
    assert not pylsp_format_document(config, workspace, doc, options=None)

def test_hanging_indentation(config, workspace):
    if False:
        while True:
            i = 10
    doc = Document(DOC_URI, workspace, INDENTED_DOC)
    res = pylsp_format_document(config, workspace, doc, options=None)
    assert len(res) == 1
    assert res[0]['newText'] == CORRECT_INDENTED_DOC

@pytest.mark.parametrize('newline', ['\r\n', '\r'])
def test_line_endings(config, workspace, newline):
    if False:
        return 10
    doc = Document(DOC_URI, workspace, f'import os;import sys{2 * newline}dict(a=1)')
    res = pylsp_format_document(config, workspace, doc, options=None)
    assert res[0]['newText'] == f'import os{newline}import sys{2 * newline}dict(a=1){newline}'