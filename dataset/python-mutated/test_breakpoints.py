"""
Tests for breakpoints.
"""
from unittest.mock import Mock
import pytest
from qtpy.QtGui import QTextCursor
from spyder import version_info
from spyder.py3compat import to_text_string
import spyder.plugins.editor.widgets.codeeditor as codeeditor
from spyder.plugins.debugger.utils import breakpointsmanager
from spyder.plugins.debugger.utils.breakpointsmanager import BreakpointsManager

def reset_emits(editor):
    if False:
        for i in range(10):
            print('nop')
    'Reset signal mocks.'
    if version_info > (4,):
        editor.sig_flags_changed.reset_mock()
    editor.sig_breakpoints_changed_called = False

def editor_assert_helper(editor, block=None, bp=False, bpc=None, emits=True):
    if False:
        return 10
    'Run the tests for call to add_remove_breakpoint.\n\n    Args:\n        editor: CodeEditor instance.\n        block: Block of text.\n        bp: Is breakpoint active?\n        bpc: Condition set for breakpoint.\n        emits: Boolean to test if signals were emitted?\n    '
    data = block.userData()
    assert data.breakpoint == bp
    assert data.breakpoint_condition == bpc
    if emits:
        if version_info > (4,):
            editor.sig_flags_changed.emit.assert_called_with()
        assert editor.sig_breakpoints_changed_called
    else:
        if version_info > (4,):
            editor.sig_flags_changed.emit.assert_not_called()
        assert not editor.sig_breakpoints_changed_called

@pytest.fixture
def code_editor_bot(qtbot):
    if False:
        while True:
            i = 10
    'Create code editor with default Python code.'
    editor = codeeditor.CodeEditor(parent=None)
    indent_chars = ' ' * 4
    tab_stop_width_spaces = 4
    editor.setup_editor(language='Python', indent_chars=indent_chars, tab_stop_width_spaces=tab_stop_width_spaces)
    if version_info > (4,):
        editor.sig_flags_changed = Mock()
    else:
        editor.get_linenumberarea_width = Mock(return_value=1)

    def mark_called():
        if False:
            return 10
        editor.sig_breakpoints_changed_called = True
    editor.sig_breakpoints_changed_called = False
    text = 'def f1(a, b):\n"Double quote string."\n\n    c = a * b\n    return c\n'
    editor.set_text(text)
    editor.filename = 'file.py'
    editor.breakpoints_manager = BreakpointsManager(editor)
    editor.breakpoints_manager.sig_repaint_breakpoints.connect(mark_called)
    return (editor, qtbot)

def test_add_remove_breakpoint(code_editor_bot, mocker):
    if False:
        return 10
    'Test CodeEditor.add_remove_breakpoint().'
    (editor, qtbot) = code_editor_bot
    arb = editor.breakpoints_manager.toogle_breakpoint
    mocker.patch.object(breakpointsmanager.QInputDialog, 'getText')
    editor.go_to_line(1)
    block = editor.textCursor().block()
    editor.set_language(None)
    reset_emits(editor)
    arb()
    assert block
    if version_info > (4,):
        editor.sig_flags_changed.emit.assert_not_called()
    assert not editor.sig_breakpoints_changed_called
    editor.set_language('Python')
    reset_emits(editor)
    arb()
    editor_assert_helper(editor, block, bp=True, bpc=None, emits=True)
    reset_emits(editor)
    arb()
    editor_assert_helper(editor, block, bp=False, bpc=None, emits=True)
    reset_emits(editor)
    editor.go_to_line(3)
    block = editor.textCursor().block()
    arb()
    editor.sig_breakpoints_changed_called = True
    editor_assert_helper(editor, block, bp=False, bpc=None, emits=True)
    reset_emits(editor)
    block = editor.document().findBlockByNumber(3)
    arb(line_number=4, condition='a > 50')
    editor_assert_helper(editor, block, bp=True, bpc='a > 50', emits=True)
    reset_emits(editor)
    breakpointsmanager.QInputDialog.getText.return_value = ('a == 42', False)
    arb(line_number=4, edit_condition=True)
    editor_assert_helper(editor, block, bp=True, bpc='a > 50', emits=False)
    breakpointsmanager.QInputDialog.getText.return_value = ('a == 42', True)
    reset_emits(editor)
    arb(line_number=4, edit_condition=True)
    editor_assert_helper(editor, block, bp=True, bpc='a == 42', emits=True)

def test_add_remove_breakpoint_with_edit_condition(code_editor_bot, mocker):
    if False:
        print('Hello World!')
    'Test add/remove breakpoint with edit_condition.'
    (editor, qtbot) = code_editor_bot
    arb = editor.breakpoints_manager.toogle_breakpoint
    mocker.patch.object(breakpointsmanager.QInputDialog, 'getText')
    linenumber = 5
    block = editor.document().findBlockByNumber(linenumber - 1)
    reset_emits(editor)
    breakpointsmanager.QInputDialog.getText.return_value = ('b == 1', False)
    arb(line_number=linenumber, edit_condition=True)
    data = block.userData()
    assert not data
    if version_info > (4,):
        editor.sig_flags_changed.emit.assert_not_called()
    assert not editor.sig_breakpoints_changed_called
    reset_emits(editor)
    breakpointsmanager.QInputDialog.getText.return_value = ('b == 1', True)
    arb(line_number=linenumber, edit_condition=True)
    editor_assert_helper(editor, block, bp=True, bpc='b == 1', emits=True)
    reset_emits(editor)
    breakpointsmanager.QInputDialog.getText.return_value = ('b == 9', False)
    arb(line_number=linenumber, edit_condition=True)
    editor_assert_helper(editor, block, bp=True, bpc='b == 1', emits=False)
    reset_emits(editor)
    arb(line_number=linenumber)
    editor_assert_helper(editor, block, bp=False, bpc=None, emits=True)
    reset_emits(editor)
    breakpointsmanager.QInputDialog.getText.return_value = ('b == 9', False)
    arb(line_number=linenumber, edit_condition=True)
    editor_assert_helper(editor, block, bp=False, bpc=None, emits=False)

def test_get_breakpoints(code_editor_bot):
    if False:
        for i in range(10):
            print('nop')
    'Test CodeEditor.get_breakpoints.'
    (editor, qtbot) = code_editor_bot
    arb = editor.breakpoints_manager.toogle_breakpoint
    gb = editor.breakpoints_manager.get_breakpoints
    assert gb() == []
    bp = [(1, None), (3, None), (4, 'a > 1'), (5, 'c == 10')]
    editor.breakpoints_manager.set_breakpoints(bp)
    assert gb() == [(1, None), (4, 'a > 1'), (5, 'c == 10')]
    arb(line_number=1, condition='a < b')
    arb(line_number=4)
    assert gb() == [(5, 'c == 10')]

def test_clear_breakpoints(code_editor_bot):
    if False:
        print('Hello World!')
    'Test CodeEditor.clear_breakpoints.'
    (editor, qtbot) = code_editor_bot
    assert len(list(editor.blockuserdata_list())) == 1
    bp = [(1, None), (4, None)]
    editor.breakpoints_manager.set_breakpoints(bp)
    assert editor.breakpoints_manager.get_breakpoints() == bp
    assert len(list(editor.blockuserdata_list())) == 2
    editor.breakpoints_manager.clear_breakpoints()
    assert editor.breakpoints_manager.get_breakpoints() == []
    assert len(list(editor.blockuserdata_list())) == 2
    for data in editor.blockuserdata_list():
        assert not data.breakpoint

def test_set_breakpoints(code_editor_bot):
    if False:
        i = 10
        return i + 15
    'Test CodeEditor.set_breakpoints.'
    (editor, qtbot) = code_editor_bot
    editor.breakpoints_manager.set_breakpoints([])
    assert editor.breakpoints_manager.get_breakpoints() == []
    bp = [(1, 'a > b'), (4, None)]
    editor.breakpoints_manager.set_breakpoints(bp)
    assert editor.breakpoints_manager.get_breakpoints() == bp
    assert list(editor.blockuserdata_list())[0].breakpoint
    bp = [(1, None), (5, 'c == 50')]
    editor.breakpoints_manager.set_breakpoints(bp)
    assert editor.breakpoints_manager.get_breakpoints() == bp
    assert list(editor.blockuserdata_list())[0].breakpoint

def test_update_breakpoints(code_editor_bot):
    if False:
        while True:
            i = 10
    'Test CodeEditor.update_breakpoints.'
    (editor, qtbot) = code_editor_bot
    reset_emits(editor)
    assert not editor.sig_breakpoints_changed_called
    editor.breakpoints_manager.toogle_breakpoint(line_number=1)
    editor.textCursor().insertBlock()
    assert editor.sig_breakpoints_changed_called
if __name__ == '__main__':
    pytest.main()