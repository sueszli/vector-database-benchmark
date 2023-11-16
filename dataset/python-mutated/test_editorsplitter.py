"""
Tests for EditorSplitter class in splitter.py
"""
from unittest.mock import Mock
import pathlib
import os
import os.path as osp
from functools import partial
import pytest
from qtpy.QtCore import Qt
from spyder.plugins.editor.widgets.editorstack import EditorStack
from spyder.plugins.editor.widgets.splitter import EditorSplitter

def editor_stack():
    if False:
        i = 10
        return i + 15
    editor_stack = EditorStack(None, [], False)
    editor_stack.set_find_widget(Mock())
    editor_stack.set_io_actions(Mock(), Mock(), Mock(), Mock())
    return editor_stack

@pytest.fixture
def editor_splitter_bot(qtbot):
    if False:
        i = 10
        return i + 15
    'Create editor splitter.'
    es = EditorSplitter(None, Mock(), [], first=True)
    qtbot.addWidget(es)
    es.resize(640, 480)
    es.show()
    return es

@pytest.fixture
def editor_splitter_lsp(qtbot_module, completion_plugin_all_started, request):
    if False:
        while True:
            i = 10
    text = '\n    import sys\n    '
    (completions, capabilities) = completion_plugin_all_started

    def report_file_open(options):
        if False:
            i = 10
            return i + 15
        filename = options['filename']
        language = options['language']
        callback = options['codeeditor']
        completions.register_file(language.lower(), filename, callback)
        callback.register_completion_capabilities(capabilities)
        with qtbot_module.waitSignal(callback.completions_response_signal, timeout=30000):
            callback.start_completion_services()

    def register_editorstack(editorstack):
        if False:
            print('Hello World!')
        editorstack.sig_perform_completion_request.connect(completions.send_request)
        editorstack.sig_open_file.connect(report_file_open)
        editorstack.register_completion_capabilities(capabilities, 'python')

    def clone(editorstack, template=None):
        if False:
            print('Hello World!')
        editorstack.set_find_widget(Mock())
        editorstack.set_io_actions(Mock(), Mock(), Mock(), Mock())
        editorstack.new('test.py', 'utf-8', text)
    mock_plugin = Mock()
    editorsplitter = EditorSplitter(None, mock_plugin, [], register_editorstack_cb=register_editorstack)
    editorsplitter.editorstack.set_find_widget(Mock())
    editorsplitter.editorstack.set_io_actions(Mock(), Mock(), Mock(), Mock())
    editorsplitter.editorstack.new('test.py', 'utf-8', text)
    mock_plugin.clone_editorstack.side_effect = partial(clone, template=editorsplitter.editorstack)
    qtbot_module.addWidget(editorsplitter)
    editorsplitter.resize(640, 480)
    editorsplitter.show()

    def teardown():
        if False:
            i = 10
            return i + 15
        editorsplitter.hide()
        editorsplitter.close()
    request.addfinalizer(teardown)
    lsp = completions.get_provider('lsp')
    return (editorsplitter, lsp)

@pytest.fixture
def editor_splitter_layout_bot(editor_splitter_bot):
    if False:
        while True:
            i = 10
    'Create editor splitter for testing layouts.'
    es = editor_splitter_bot

    def clone(editorstack):
        if False:
            print('Hello World!')
        editorstack.close_action.setEnabled(False)
        editorstack.set_find_widget(Mock())
        editorstack.set_io_actions(Mock(), Mock(), Mock(), Mock())
        editorstack.new('foo.py', 'utf-8', 'a = 1\nprint(a)\n\nx = 2')
        editorstack.new('layout_test.py', 'utf-8', 'print(spam)')
        with open(__file__) as f:
            text = f.read()
        editorstack.new(__file__, 'utf-8', text)
    es.plugin.clone_editorstack.side_effect = clone
    clone(es.editorstack)
    return es

def test_init(editor_splitter_bot):
    if False:
        return 10
    '"Test __init__.'
    es = editor_splitter_bot
    assert es.orientation() == Qt.Horizontal
    assert es.testAttribute(Qt.WA_DeleteOnClose)
    assert not es.childrenCollapsible()
    assert not es.toolbar_list
    assert not es.menu_list
    assert es.register_editorstack_cb == es.plugin.register_editorstack
    assert es.unregister_editorstack_cb == es.plugin.unregister_editorstack
    assert not es.menu_actions
    assert es.editorstack.menu_actions != []
    assert isinstance(es.editorstack, EditorStack)
    es.plugin.register_editorstack.assert_called_with(es.editorstack)
    es.plugin.unregister_editorstack.assert_not_called()
    es.plugin.clone_editorstack.assert_not_called()
    assert es.count() == 1
    assert es.widget(0) == es.editorstack

def test_close(editor_splitter_bot, qtbot):
    if False:
        return 10
    'Test the interface for closing the editor splitters.'
    es = editor_splitter_bot
    es.split()
    esw1 = es.widget(1)
    esw1.editorstack.set_closable(True)
    assert es.count() == 2
    assert esw1.count() == 1
    esw1.split()
    esw1w1 = esw1.widget(1)
    esw1w1.editorstack.set_closable(True)
    assert es.count() == 2
    assert esw1.count() == 2
    assert esw1w1.count() == 1
    esw1.split()
    esw1w2 = esw1.widget(2)
    esw1w2.editorstack.set_closable(True)
    assert es.count() == 2
    assert esw1.count() == 3
    assert esw1w1.count() == esw1w2.count() == 1
    assert es.isVisible()
    assert esw1.isVisible()
    assert esw1w1.isVisible()
    assert esw1w2.isVisible()
    with qtbot.waitSignal(esw1.editorstack.destroyed, timeout=1000):
        esw1.editorstack.close_split()
    assert es.count() == 2
    assert esw1.count() == 2
    assert esw1.editorstack is None
    assert es.isVisible()
    assert esw1.isVisible()
    assert esw1w1.isVisible()
    assert esw1w2.isVisible()
    with qtbot.waitSignal(esw1w1.destroyed, timeout=1000):
        esw1w1.editorstack.close_split()
    with pytest.raises(RuntimeError):
        esw1w1.count()
    assert es.count() == 2
    assert esw1.count() == 1
    assert es.isVisible()
    assert esw1.isVisible()
    assert esw1w2.isVisible()
    with qtbot.waitSignal(esw1.destroyed, timeout=1000):
        esw1w2.editorstack.close_split()
    with pytest.raises(RuntimeError):
        esw1.count()
    with pytest.raises(RuntimeError):
        esw1w2.count()
    assert es.isVisible()
    assert es.count() == 1
    es.editorstack.close_split()
    assert es.isVisible()
    assert es.count() == 1

def test_split(editor_splitter_layout_bot):
    if False:
        print('Hello World!')
    'Test split() that adds new splitters to this instance.'
    es = editor_splitter_layout_bot
    es.split()
    assert es.orientation() == Qt.Vertical
    assert not es.editorstack.horsplit_action.isEnabled()
    assert es.editorstack.versplit_action.isEnabled()
    assert es.count() == 2
    assert isinstance(es.widget(1), EditorSplitter)
    assert es.widget(1).count() == 1
    assert es.widget(1).editorstack == es.widget(1).widget(0)
    es.widget(1).plugin.clone_editorstack.assert_called_with(editorstack=es.widget(1).editorstack)
    es.editorstack.sig_split_horizontally.emit()
    assert es.orientation() == Qt.Horizontal
    assert es.editorstack.horsplit_action.isEnabled()
    assert not es.editorstack.versplit_action.isEnabled()
    assert es.count() == 3
    assert isinstance(es.widget(2), EditorSplitter)
    assert es.widget(1).count() == 1
    assert es.widget(2).count() == 1
    es1 = es.widget(1)
    es1.editorstack.sig_split_vertically.emit()
    assert es.orientation() == Qt.Horizontal
    assert es1.orientation() == Qt.Vertical
    assert not es1.editorstack.horsplit_action.isEnabled()
    assert es1.editorstack.versplit_action.isEnabled()
    assert es1.count() == 2
    assert isinstance(es1.widget(0), EditorStack)
    assert isinstance(es1.widget(1), EditorSplitter)
    assert not es1.widget(1).isHidden()

def test_iter_editorstacks(editor_splitter_bot):
    if False:
        i = 10
        return i + 15
    'Test iter_editorstacks.'
    es = editor_splitter_bot
    es_iter = es.iter_editorstacks
    assert es_iter() == [(es.editorstack, es.orientation())]
    es.split(Qt.Vertical)
    esw1 = es.widget(1)
    assert es_iter() == [(es.editorstack, es.orientation()), (esw1.editorstack, esw1.orientation())]
    es.split(Qt.Horizontal)
    assert es_iter() == [(es.editorstack, es.orientation()), (esw1.editorstack, esw1.orientation())]
    esw1.split(Qt.Vertical)
    esw1w1 = es.widget(1).widget(1)
    assert es_iter() == [(es.editorstack, es.orientation()), (esw1.editorstack, esw1.orientation()), (esw1w1.editorstack, esw1w1.orientation())]

def test_get_layout_settings(editor_splitter_bot, qtbot, mocker):
    if False:
        while True:
            i = 10
    'Test get_layout_settings().'
    es = editor_splitter_bot
    setting = es.get_layout_settings()
    assert setting['splitsettings'] == [(False, None, [])]
    stack1 = editor_stack()
    stack1.new('foo.py', 'utf-8', 'a = 1\nprint(a)\n\nx = 2')
    stack1.new('layout_test.py', 'utf-8', 'spam egg\n')
    stack2 = editor_stack()
    stack2.new('test.py', 'utf-8', 'test text')
    mocker.patch.object(EditorSplitter, 'iter_editorstacks')
    EditorSplitter.iter_editorstacks.return_value = [(stack1, Qt.Vertical), (stack2, Qt.Horizontal)]
    setting = es.get_layout_settings()
    assert setting['hexstate']
    assert setting['sizes'] == es.sizes()
    assert setting['splitsettings'] == [(False, 'foo.py', [5, 3]), (False, 'test.py', [2])]

def test_set_layout_settings_dont_goto(editor_splitter_layout_bot):
    if False:
        for i in range(10):
            print('nop')
    'Test set_layout_settings().'
    es = editor_splitter_layout_bot
    linecount = es.editorstack.data[2].editor.get_cursor_line_number()
    state = '000000ff000000010000000200000231000001ff00ffffffff010000000200'
    sizes = [561, 511]
    splitsettings = [(False, 'layout_test.py', [2, 1, 52]), (False, 'foo.py', [3, 2, 125]), (False, __file__, [1, 1, 1])]
    new_settings = {'hexstate': state, 'sizes': sizes, 'splitsettings': splitsettings}
    get_settings = es.get_layout_settings()
    assert es.count() == 1
    assert get_settings['hexstate'] != state
    assert get_settings['splitsettings'] != splitsettings
    assert es.set_layout_settings({'spam': 'test'}) is None
    es.set_layout_settings(new_settings, dont_goto=True)
    get_settings = es.get_layout_settings()
    assert es.count() == 2
    assert es.widget(1).count() == 2
    assert es.widget(1).widget(1).count() == 1
    assert get_settings['hexstate'] == state
    assert get_settings['splitsettings'] == [(False, 'foo.py', [5, 2, linecount]), (False, 'foo.py', [5, 2, linecount]), (False, 'foo.py', [5, 2, linecount])]

def test_set_layout_settings_goto(editor_splitter_layout_bot):
    if False:
        while True:
            i = 10
    'Test set_layout_settings().'
    es = editor_splitter_layout_bot
    state = '000000ff000000010000000200000231000001ff00ffffffff010000000200'
    sizes = [561, 511]
    splitsettings = [(False, 'layout_test.py', [2, 1, 52]), (False, 'foo.py', [3, 2, 125]), (False, __file__, [1, 1, 1])]
    new_settings = {'hexstate': state, 'sizes': sizes, 'splitsettings': splitsettings}
    es.set_layout_settings(new_settings, dont_goto=None)
    get_settings = es.get_layout_settings()
    assert get_settings['splitsettings'] == [(False, 'foo.py', [2, 1, 52]), (False, 'foo.py', [3, 2, 125]), (False, 'foo.py', [1, 1, 1])]

@pytest.mark.order(1)
@pytest.mark.skipif(os.name == 'nt', reason='Makes other tests fail on Windows')
def test_lsp_splitter_close(editor_splitter_lsp):
    if False:
        for i in range(10):
            print('nop')
    'Test for spyder-ide/spyder#9341.'
    (editorsplitter, lsp_manager) = editor_splitter_lsp
    editorsplitter.split()
    lsp_files = lsp_manager.clients['python']['instance'].watched_files
    editor = editorsplitter.editorstack.get_current_editor()
    path = pathlib.Path(osp.abspath(editor.filename)).as_uri()
    assert len(lsp_files[path]) == 2
    editorstacks = editorsplitter.iter_editorstacks()
    assert len(editorstacks) == 2
    last_editorstack = editorstacks[0][0]
    last_editorstack.close()
    lsp_files = lsp_manager.clients['python']['instance'].watched_files
    assert len(lsp_files[path]) == 1
if __name__ == '__main__':
    import os.path as osp
    pytest.main(['-x', osp.basename(__file__), '-v', '-rw'])