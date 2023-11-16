import contextlib
from pathlib import Path
import os
import tempfile
from pylsp import lsp, uris
from pylsp.workspace import Document, Workspace
from pylsp.plugins import pylint_lint
DOC_URI = uris.from_fs_path(__file__)
DOC = 'import sys\n\ndef hello():\n\tpass\n\nimport json\n'
DOC_SYNTAX_ERR = 'def hello()\n    pass\n'

@contextlib.contextmanager
def temp_document(doc_text, workspace):
    if False:
        while True:
            i = 10
    try:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
            name = temp_file.name
            temp_file.write(doc_text)
        yield Document(uris.from_fs_path(name), workspace)
    finally:
        os.remove(name)

def write_temp_doc(document, contents):
    if False:
        i = 10
        return i + 15
    with open(document.path, 'w', encoding='utf-8') as temp_file:
        temp_file.write(contents)

def test_pylint(config, workspace):
    if False:
        while True:
            i = 10
    with temp_document(DOC, workspace) as doc:
        diags = pylint_lint.pylsp_lint(config, workspace, doc, True)
        msg = '[unused-import] Unused import sys'
        unused_import = [d for d in diags if d['message'] == msg][0]
        assert unused_import['range']['start'] == {'line': 0, 'character': 0}
        assert unused_import['severity'] == lsp.DiagnosticSeverity.Warning
        assert unused_import['tags'] == [lsp.DiagnosticTag.Unnecessary]
        config.plugin_settings('pylint')['executable'] = 'pylint'
        diags = pylint_lint.pylsp_lint(config, workspace, doc, True)
        msg = 'Unused import sys (unused-import)'
        unused_import = [d for d in diags if d['message'] == msg][0]
        assert unused_import['range']['start'] == {'line': 0, 'character': 0}
        assert unused_import['severity'] == lsp.DiagnosticSeverity.Warning

def test_syntax_error_pylint(config, workspace):
    if False:
        return 10
    with temp_document(DOC_SYNTAX_ERR, workspace) as doc:
        diag = pylint_lint.pylsp_lint(config, workspace, doc, True)[0]
        assert diag['message'].startswith('[syntax-error]')
        assert diag['message'].count("expected ':'") or diag['message'].count('invalid syntax')
        assert diag['range']['start'] == {'line': 0, 'character': 12}
        assert diag['severity'] == lsp.DiagnosticSeverity.Error
        assert 'tags' not in diag
        config.plugin_settings('pylint')['executable'] = 'pylint'
        diag = pylint_lint.pylsp_lint(config, workspace, doc, True)[0]
        assert diag['message'].count("expected ':'") or diag['message'].count('invalid syntax')
        assert diag['range']['start'] == {'line': 0, 'character': 12}
        assert diag['severity'] == lsp.DiagnosticSeverity.Error

def test_lint_free_pylint(config, workspace):
    if False:
        while True:
            i = 10
    ws = Workspace(str(Path(__file__).absolute().parents[2]), workspace._endpoint)
    assert not pylint_lint.pylsp_lint(config, ws, Document(uris.from_fs_path(__file__), ws), True)

def test_lint_caching(workspace):
    if False:
        print('Hello World!')
    flags = '--disable=invalid-name'
    with temp_document(DOC, workspace) as doc:
        diags = pylint_lint.PylintLinter.lint(doc, True, flags)
        assert diags
        write_temp_doc(doc, '')
        assert pylint_lint.PylintLinter.lint(doc, False, flags) == diags
        assert not pylint_lint.PylintLinter.lint(doc, True, flags)
        assert not pylint_lint.PylintLinter.lint(doc, False, flags)

def test_per_file_caching(config, workspace):
    if False:
        for i in range(10):
            print('nop')
    with temp_document(DOC, workspace) as doc:
        assert pylint_lint.pylsp_lint(config, workspace, doc, True)
    assert not pylint_lint.pylsp_lint(config, workspace, Document(uris.from_fs_path(__file__), workspace), False)