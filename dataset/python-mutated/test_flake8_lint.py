import tempfile
import os
from textwrap import dedent
from unittest.mock import patch
from pylsp import lsp, uris
from pylsp.plugins import flake8_lint
from pylsp.workspace import Document
DOC_URI = uris.from_fs_path(__file__)
DOC = 'import pylsp\n\nt = "TEST"\n\ndef using_const():\n\ta = 8 + 9\n\treturn t\n'

def temp_document(doc_text, workspace):
    if False:
        return 10
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        name = temp_file.name
        temp_file.write(doc_text)
    doc = Document(uris.from_fs_path(name), workspace)
    return (name, doc)

def test_flake8_unsaved(workspace):
    if False:
        return 10
    doc = Document('', workspace, DOC)
    diags = flake8_lint.pylsp_lint(workspace, doc)
    msg = "F841 local variable 'a' is assigned to but never used"
    unused_var = [d for d in diags if d['message'] == msg][0]
    assert unused_var['source'] == 'flake8'
    assert unused_var['code'] == 'F841'
    assert unused_var['range']['start'] == {'line': 5, 'character': 1}
    assert unused_var['range']['end'] == {'line': 5, 'character': 11}
    assert unused_var['severity'] == lsp.DiagnosticSeverity.Error
    assert unused_var['tags'] == [lsp.DiagnosticTag.Unnecessary]

def test_flake8_lint(workspace):
    if False:
        return 10
    (name, doc) = temp_document(DOC, workspace)
    try:
        diags = flake8_lint.pylsp_lint(workspace, doc)
        msg = "F841 local variable 'a' is assigned to but never used"
        unused_var = [d for d in diags if d['message'] == msg][0]
        assert unused_var['source'] == 'flake8'
        assert unused_var['code'] == 'F841'
        assert unused_var['range']['start'] == {'line': 5, 'character': 1}
        assert unused_var['range']['end'] == {'line': 5, 'character': 11}
        assert unused_var['severity'] == lsp.DiagnosticSeverity.Error
    finally:
        os.remove(name)

def test_flake8_respecting_configuration(workspace):
    if False:
        print('Hello World!')
    docs = [('src/__init__.py', ''), ('src/a.py', DOC), ('src/b.py', 'import os'), ('setup.cfg', dedent('\n        [flake8]\n        ignore = E302,W191\n        per-file-ignores =\n            src/a.py:F401\n            src/b.py:W292\n        '))]
    made = {}
    for (rel, contents) in docs:
        location = os.path.join(workspace.root_path, rel)
        made[rel] = {'uri': uris.from_fs_path(location)}
        os.makedirs(os.path.dirname(location), exist_ok=True)
        with open(location, 'w', encoding='utf-8') as fle:
            fle.write(contents)
        workspace.put_document(made[rel]['uri'], contents)
        made[rel]['document'] = workspace._docs[made[rel]['uri']]
    diags = flake8_lint.pylsp_lint(workspace, made['src/a.py']['document'])
    assert diags == [{'source': 'flake8', 'code': 'F841', 'range': {'start': {'line': 5, 'character': 1}, 'end': {'line': 5, 'character': 11}}, 'message': "F841 local variable 'a' is assigned to but never used", 'severity': 1, 'tags': [1]}]
    diags = flake8_lint.pylsp_lint(workspace, made['src/b.py']['document'])
    assert diags == [{'source': 'flake8', 'code': 'F401', 'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 9}}, 'message': "F401 'os' imported but unused", 'severity': 1, 'tags': [1]}]

def test_flake8_config_param(workspace):
    if False:
        print('Hello World!')
    with patch('pylsp.plugins.flake8_lint.Popen') as popen_mock:
        mock_instance = popen_mock.return_value
        mock_instance.communicate.return_value = [bytes(), bytes()]
        flake8_conf = '/tmp/some.cfg'
        workspace._config.update({'plugins': {'flake8': {'config': flake8_conf}}})
        (_name, doc) = temp_document(DOC, workspace)
        flake8_lint.pylsp_lint(workspace, doc)
        (call_args,) = popen_mock.call_args[0]
        assert 'flake8' in call_args
        assert '--config={}'.format(flake8_conf) in call_args

def test_flake8_executable_param(workspace):
    if False:
        print('Hello World!')
    with patch('pylsp.plugins.flake8_lint.Popen') as popen_mock:
        mock_instance = popen_mock.return_value
        mock_instance.communicate.return_value = [bytes(), bytes()]
        flake8_executable = '/tmp/flake8'
        workspace._config.update({'plugins': {'flake8': {'executable': flake8_executable}}})
        (_name, doc) = temp_document(DOC, workspace)
        flake8_lint.pylsp_lint(workspace, doc)
        (call_args,) = popen_mock.call_args[0]
        assert flake8_executable in call_args

def get_flake8_cfg_settings(workspace, config_str):
    if False:
        print('Hello World!')
    "Write a ``setup.cfg``, load it in the workspace, and return the flake8 settings.\n\n    This function creates a ``setup.cfg``; you'll have to delete it yourself.\n    "
    with open(os.path.join(workspace.root_path, 'setup.cfg'), 'w+', encoding='utf-8') as f:
        f.write(config_str)
    workspace.update_config({'pylsp': {'configurationSources': ['flake8']}})
    return workspace._config.plugin_settings('flake8')

def test_flake8_multiline(workspace):
    if False:
        i = 10
        return i + 15
    config_str = '[flake8]\nexclude =\n    blah/,\n    file_2.py\n    '
    doc_str = "print('hi')\nimport os\n"
    doc_uri = uris.from_fs_path(os.path.join(workspace.root_path, 'blah/__init__.py'))
    workspace.put_document(doc_uri, doc_str)
    flake8_settings = get_flake8_cfg_settings(workspace, config_str)
    assert 'exclude' in flake8_settings
    assert len(flake8_settings['exclude']) == 2
    with patch('pylsp.plugins.flake8_lint.Popen') as popen_mock:
        mock_instance = popen_mock.return_value
        mock_instance.communicate.return_value = [bytes(), bytes()]
        doc = workspace.get_document(doc_uri)
        flake8_lint.pylsp_lint(workspace, doc)
    call_args = popen_mock.call_args[0][0]
    init_file = os.path.join('blah', '__init__.py')
    assert call_args == ['flake8', '-', '--exclude=blah/,file_2.py', '--stdin-display-name', init_file]
    os.unlink(os.path.join(workspace.root_path, 'setup.cfg'))

def test_flake8_per_file_ignores(workspace):
    if False:
        print('Hello World!')
    config_str = '[flake8]\nignores = F403\nper-file-ignores =\n    **/__init__.py:F401,E402\n    test_something.py:E402,\nexclude =\n    file_1.py\n    file_2.py\n    '
    doc_str = "print('hi')\nimport os\n"
    doc_uri = uris.from_fs_path(os.path.join(workspace.root_path, 'blah/__init__.py'))
    workspace.put_document(doc_uri, doc_str)
    flake8_settings = get_flake8_cfg_settings(workspace, config_str)
    assert 'perFileIgnores' in flake8_settings
    assert len(flake8_settings['perFileIgnores']) == 2
    assert 'exclude' in flake8_settings
    assert len(flake8_settings['exclude']) == 2
    doc = workspace.get_document(doc_uri)
    res = flake8_lint.pylsp_lint(workspace, doc)
    assert not res
    os.unlink(os.path.join(workspace.root_path, 'setup.cfg'))

def test_per_file_ignores_alternative_syntax(workspace):
    if False:
        return 10
    config_str = '[flake8]\nper-file-ignores = **/__init__.py:F401,E402\n    '
    doc_str = "print('hi')\nimport os\n"
    doc_uri = uris.from_fs_path(os.path.join(workspace.root_path, 'blah/__init__.py'))
    workspace.put_document(doc_uri, doc_str)
    flake8_settings = get_flake8_cfg_settings(workspace, config_str)
    assert 'perFileIgnores' in flake8_settings
    assert len(flake8_settings['perFileIgnores']) == 2
    doc = workspace.get_document(doc_uri)
    res = flake8_lint.pylsp_lint(workspace, doc)
    assert not res
    os.unlink(os.path.join(workspace.root_path, 'setup.cfg'))