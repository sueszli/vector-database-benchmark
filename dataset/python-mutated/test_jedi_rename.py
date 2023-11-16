import os
import pytest
from pylsp import uris
from pylsp.plugins.jedi_rename import pylsp_rename
from pylsp.workspace import Document
DOC_NAME = 'test1.py'
DOC = 'class Test1():\n    pass\n\nclass Test2(Test1):\n    pass\n'
DOC_NAME_EXTRA = 'test2.py'
DOC_EXTRA = 'from test1 import Test1\nx = Test1()\n'
DOC_NAME_SIMPLE = 'test3.py'
DOC_SIMPLE = 'foo = 12'

@pytest.fixture
def tmp_workspace(temp_workspace_factory):
    if False:
        for i in range(10):
            print('nop')
    return temp_workspace_factory({DOC_NAME: DOC, DOC_NAME_EXTRA: DOC_EXTRA, DOC_NAME_SIMPLE: DOC_SIMPLE})

def test_jedi_rename(tmp_workspace, config):
    if False:
        i = 10
        return i + 15
    position = {'line': 0, 'character': 6}
    DOC_URI = uris.from_fs_path(os.path.join(tmp_workspace.root_path, DOC_NAME))
    doc = Document(DOC_URI, tmp_workspace)
    result = pylsp_rename(config, tmp_workspace, doc, position, 'ShouldBeRenamed')
    assert len(result.keys()) == 1
    changes = result.get('documentChanges')
    assert len(changes) == 2
    assert changes[0]['textDocument']['uri'] == doc.uri
    assert changes[0]['textDocument']['version'] == doc.version
    assert changes[0].get('edits') == [{'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 5, 'character': 0}}, 'newText': 'class ShouldBeRenamed():\n    pass\n\nclass Test2(ShouldBeRenamed):\n    pass\n'}]
    path = os.path.join(tmp_workspace.root_path, DOC_NAME_EXTRA)
    uri_extra = uris.from_fs_path(path)
    assert changes[1]['textDocument']['uri'] == uri_extra
    assert changes[1]['textDocument']['version'] is None
    expected = 'from test1 import ShouldBeRenamed\nx = ShouldBeRenamed()\n'
    if os.name == 'nt':
        expected = expected.replace('\n', '\r\n')
    assert changes[1].get('edits') == [{'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 2, 'character': 0}}, 'newText': expected}]
    position = {'line': 0, 'character': 0}
    DOC_URI = uris.from_fs_path(os.path.join(tmp_workspace.root_path, DOC_NAME_SIMPLE))
    doc = Document(DOC_URI, tmp_workspace)
    result = pylsp_rename(config, tmp_workspace, doc, position, 'bar')
    assert len(result.keys()) == 1
    changes = result.get('documentChanges')
    assert len(changes) == 1
    assert changes[0]['textDocument']['uri'] == doc.uri
    assert changes[0]['textDocument']['version'] == doc.version
    assert changes[0].get('edits') == [{'range': {'start': {'line': 0, 'character': 0}, 'end': {'line': 0, 'character': 0}}, 'newText': 'bar = 12'}]