import os
import pytest
from pylsp import lsp, uris
from pylsp.workspace import Document
from pylsp.plugins import pycodestyle_lint
DOC_URI = uris.from_fs_path(__file__)
DOC = 'import sys\n\ndef hello( ):\n\tpass\nprint("hello"\n ,"world"\n)\n\nimport json\n\n\n'

def test_pycodestyle(workspace):
    if False:
        print('Hello World!')
    doc = Document(DOC_URI, workspace, DOC)
    diags = pycodestyle_lint.pylsp_lint(workspace, doc)
    assert all((d['source'] == 'pycodestyle' for d in diags))
    msg = 'W191 indentation contains tabs'
    mod_import = [d for d in diags if d['message'] == msg][0]
    assert mod_import['code'] == 'W191'
    assert mod_import['severity'] == lsp.DiagnosticSeverity.Warning
    assert mod_import['range']['start'] == {'line': 3, 'character': 0}
    assert mod_import['range']['end'] == {'line': 3, 'character': 6}
    msg = 'W391 blank line at end of file'
    mod_import = [d for d in diags if d['message'] == msg][0]
    assert mod_import['code'] == 'W391'
    assert mod_import['severity'] == lsp.DiagnosticSeverity.Warning
    assert mod_import['range']['start'] == {'line': 10, 'character': 0}
    assert mod_import['range']['end'] == {'line': 10, 'character': 1}
    msg = "E201 whitespace after '('"
    mod_import = [d for d in diags if d['message'] == msg][0]
    assert mod_import['code'] == 'E201'
    assert mod_import['severity'] == lsp.DiagnosticSeverity.Warning
    assert mod_import['range']['start'] == {'line': 2, 'character': 10}
    assert mod_import['range']['end'] == {'line': 2, 'character': 14}
    msg = 'E128 continuation line under-indented for visual indent'
    mod_import = [d for d in diags if d['message'] == msg][0]
    assert mod_import['code'] == 'E128'
    assert mod_import['severity'] == lsp.DiagnosticSeverity.Warning
    assert mod_import['range']['start'] == {'line': 5, 'character': 1}
    assert mod_import['range']['end'] == {'line': 5, 'character': 10}

def test_pycodestyle_config(workspace):
    if False:
        return 10
    "Test that we load config files properly.\n\n    Config files are loaded in the following order:\n        tox.ini pep8.cfg setup.cfg pycodestyle.cfg\n\n    Each overriding the values in the last.\n\n    These files are first looked for in the current document's\n    directory and then each parent directory until any one is found\n    terminating at the workspace root.\n\n    If any section called 'pycodestyle' exists that will be solely used\n    and any config in a 'pep8' section will be ignored\n    "
    doc_uri = uris.from_fs_path(os.path.join(workspace.root_path, 'test.py'))
    workspace.put_document(doc_uri, DOC)
    doc = workspace.get_document(doc_uri)
    diags = pycodestyle_lint.pylsp_lint(workspace, doc)
    assert [d for d in diags if d['code'] == 'W191']
    content = {'setup.cfg': ('[pycodestyle]\nignore = W191, E201, E128', True), 'tox.ini': ('', False)}
    for (conf_file, (content, working)) in list(content.items()):
        with open(os.path.join(workspace.root_path, conf_file), 'w+', encoding='utf-8') as f:
            f.write(content)
        workspace._config.settings.cache_clear()
        diags = pycodestyle_lint.pylsp_lint(workspace, doc)
        assert len([d for d in diags if d['code'] == 'W191']) == (0 if working else 1)
        assert len([d for d in diags if d['code'] == 'E201']) == (0 if working else 1)
        assert [d for d in diags if d['code'] == 'W391']
        os.unlink(os.path.join(workspace.root_path, conf_file))
    workspace._config.update({'plugins': {'pycodestyle': {'ignore': ['W191', 'E201']}}})
    diags = pycodestyle_lint.pylsp_lint(workspace, doc)
    assert not [d for d in diags if d['code'] == 'W191']
    assert not [d for d in diags if d['code'] == 'E201']
    assert [d for d in diags if d['code'] == 'W391']

@pytest.mark.parametrize('newline', ['\r\n', '\r'])
def test_line_endings(workspace, newline):
    if False:
        i = 10
        return i + 15
    "\n    Check that Pycodestyle doesn't generate false positives with line endings\n    other than LF.\n    "
    source = f'try:{newline}    1/0{newline}except Exception:{newline}    pass{newline}'
    doc = Document(DOC_URI, workspace, source)
    diags = pycodestyle_lint.pylsp_lint(workspace, doc)
    assert len(diags) == 0