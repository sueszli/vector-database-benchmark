import os
from pylsp import lsp, uris
from pylsp.workspace import Document
from pylsp.plugins import pydocstyle_lint
DOC_URI = uris.from_fs_path(os.path.join(os.path.dirname(__file__), 'pydocstyle.py'))
TEST_DOC_URI = uris.from_fs_path(__file__)
DOC = 'import sys\n\ndef hello():\n\tpass\n\nimport json\n'

def test_pydocstyle(config, workspace):
    if False:
        for i in range(10):
            print('nop')
    doc = Document(DOC_URI, workspace, DOC)
    diags = pydocstyle_lint.pylsp_lint(config, workspace, doc)
    assert all((d['source'] == 'pydocstyle' for d in diags))
    assert diags[0] == {'code': 'D100', 'message': 'D100: Missing docstring in public module', 'severity': lsp.DiagnosticSeverity.Warning, 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 11}}, 'source': 'pydocstyle'}

def test_pydocstyle_test_document(config, workspace):
    if False:
        while True:
            i = 10
    doc = Document(TEST_DOC_URI, workspace, '')
    diags = pydocstyle_lint.pylsp_lint(config, workspace, doc)
    assert not diags

def test_pydocstyle_empty_source(config, workspace):
    if False:
        while True:
            i = 10
    doc = Document(DOC_URI, workspace, '')
    diags = pydocstyle_lint.pylsp_lint(config, workspace, doc)
    assert diags[0]['message'] == 'D100: Missing docstring in public module'
    assert len(diags) == 1

def test_pydocstyle_invalid_source(config, workspace):
    if False:
        return 10
    doc = Document(DOC_URI, workspace, 'bad syntax')
    diags = pydocstyle_lint.pylsp_lint(config, workspace, doc)
    assert not diags