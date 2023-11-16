"""
Tests for pathmanager.py
"""
import os
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QTextCursor
from qtpy.QtWidgets import QVBoxLayout, QWidget
from spyder.plugins.editor.widgets.codeeditor import CodeEditor
from spyder.widgets.findreplace import FindReplace
from spyder.utils.stylesheet import APP_STYLESHEET

@pytest.fixture
def findreplace_editor(qtbot, request):
    if False:
        i = 10
        return i + 15
    'Set up editor with FindReplace widget.'
    widget = QWidget()
    qtbot.addWidget(widget)
    widget.setStyleSheet(str(APP_STYLESHEET))
    layout = QVBoxLayout()
    widget.setLayout(layout)
    editor = CodeEditor(parent=widget)
    editor.setup_editor(color_scheme='spyder/dark', font=QFont('Courier New', 10))
    widget.editor = editor
    layout.addWidget(editor)
    findreplace = FindReplace(editor, enable_replace=True)
    findreplace.set_editor(editor)
    widget.findreplace = findreplace
    layout.addWidget(findreplace)
    widget.resize(900, 360)
    widget.show()
    return widget

def test_findreplace_multiline_replacement(findreplace_editor, qtbot):
    if False:
        print('Hello World!')
    '\n    Test find replace widget for multiline regex replacements\n    See: spyder-ide/spyder#2675\n    '
    expected = '\n\nhello world!\n\n'
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    editor.set_text('\n\nhello\n\n\nworld!\n\n')
    findreplace.show_replace()
    findreplace.re_button.setChecked(True)
    edit = findreplace.search_text.lineEdit()
    edit.clear()
    edit.setText('\\n\\n\\n')
    findreplace.replace_text.setCurrentText(' ')
    qtbot.wait(1000)
    findreplace.replace_find_all()
    qtbot.wait(1000)
    assert editor.toPlainText() == expected

def test_replace_selection(findreplace_editor, qtbot):
    if False:
        i = 10
        return i + 15
    'Test find replace final selection in the editor.\n    For further information see spyder-ide/spyder#12745\n    '
    expected = 'Spyder is greit!\nSpyder is greit!'
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    editor.set_text('Spyder as great!\nSpyder as great!')
    editor.select_lines(0, 2)
    findreplace.show_replace()
    edit = findreplace.search_text.lineEdit()
    edit.clear()
    edit.setText('a')
    findreplace.replace_text.setCurrentText('i')
    findreplace.replace_find_selection()
    qtbot.wait(1000)
    assert editor.get_selected_text() == expected
    assert len(editor.get_selected_text()) == len(expected)

def test_replace_all(findreplace_editor, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test find replace final selection in the editor.\n\n    Regression test for spyder-ide/spyder#20403\n    '
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    findreplace.show_replace()
    findreplace.search_text.setCurrentText('a')
    findreplace.replace_text.setCurrentText('x')
    editor.set_text('a\naa')
    expected = 'x\nxx'
    qtbot.wait(500)
    findreplace.replace_find_all()
    qtbot.wait(500)
    assert editor.toPlainText() == expected
    editor.set_text('a\naa')
    expected = 'x\naa'
    qtbot.wait(500)
    qtbot.mouseClick(findreplace.words_button, Qt.LeftButton)
    findreplace.replace_find_all()
    qtbot.wait(500)
    qtbot.mouseClick(findreplace.words_button, Qt.LeftButton)
    assert editor.toPlainText() == expected
    findreplace.search_text.setCurrentText('a(\\d+)a')
    findreplace.replace_text.setCurrentText('b\\1b')
    editor.set_text('a123a\nabca')
    expected = 'b123b\nabca'
    qtbot.wait(500)
    qtbot.mouseClick(findreplace.re_button, Qt.LeftButton)
    findreplace.replace_find_all()
    qtbot.wait(500)
    qtbot.mouseClick(findreplace.re_button, Qt.LeftButton)
    assert editor.toPlainText() == expected

def test_messages_action(findreplace_editor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that we set the right icons and tooltips on messages_action.\n    '
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    editor.set_text('Spyder as great!')
    assert not findreplace.messages_action.isVisible()
    edit = findreplace.search_text.lineEdit()
    edit.clear()
    qtbot.keyClicks(edit, 'foo')
    assert not findreplace.number_matches_text.isVisible()
    assert findreplace.messages_action.icon().cacheKey() == findreplace.no_matches_icon.cacheKey()
    assert findreplace.messages_action.toolTip() == findreplace.TOOLTIP['no_matches']
    edit.selectAll()
    qtbot.keyClick(edit, Qt.Key_Delete)
    assert not findreplace.messages_action.isVisible()
    msg = ': nothing to repeat at position 0'
    edit.clear()
    findreplace.re_button.setChecked(True)
    qtbot.keyClicks(edit, '?')
    assert not findreplace.number_matches_text.isVisible()
    assert findreplace.messages_action.icon().cacheKey() == findreplace.error_icon.cacheKey()
    assert findreplace.messages_action.toolTip() == findreplace.TOOLTIP['regexp_error'] + msg
    edit.clear()
    qtbot.keyClicks(edit, 'great')
    qtbot.wait(500)
    assert not findreplace.messages_action.isVisible()
    assert findreplace.number_matches_text.isVisible()
    assert findreplace.number_matches_text.text() == '1 of 1'

def test_replace_text_button(findreplace_editor, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test that replace_text_button is checked/unchecked under different\n    scenarios.\n    '
    findreplace = findreplace_editor.findreplace
    findreplace.hide()
    findreplace.show(hide_replace=False)
    qtbot.wait(500)
    assert findreplace.replace_text_button.isChecked()
    qtbot.mouseClick(findreplace.close_button, Qt.LeftButton)
    findreplace.show(hide_replace=True)
    qtbot.wait(500)
    assert not findreplace.replace_text_button.isChecked()
    findreplace.show(hide_replace=False)
    qtbot.wait(500)
    findreplace.show(hide_replace=True)
    assert not findreplace.replace_text_button.isChecked()

def test_update_matches(findreplace_editor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that we update the total number of matches when the editor text has\n    changed.\n    '
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    editor.set_text('foo\nfoo\n')
    edit = findreplace.search_text.lineEdit()
    edit.clear()
    edit.setFocus()
    qtbot.keyClicks(edit, 'foo')
    assert findreplace.number_matches_text.text() == '1 of 2'
    editor.setFocus()
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.End)
    editor.setTextCursor(cursor)
    qtbot.keyClicks(editor, 'foo')
    qtbot.wait(500)
    assert findreplace.number_matches_text.text() == '3 matches'
    assert len(editor.found_results) == 3
    findreplace.hide()
    qtbot.wait(500)
    qtbot.keyClick(editor, Qt.Key_Return)
    qtbot.keyClicks(editor, 'foo')
    qtbot.wait(500)
    assert findreplace.number_matches_text.text() == '3 matches'

def test_clear_action(findreplace_editor, qtbot):
    if False:
        while True:
            i = 10
    '\n    Test that clear_action in the search_text line edit is working as expected.\n    '
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    clear_action = findreplace.search_text.lineEdit().clear_action
    editor.set_text('foo\nfoo\n')
    assert not clear_action.isVisible()
    edit = findreplace.search_text.lineEdit()
    edit.setFocus()
    qtbot.keyClicks(edit, 'foo')
    assert clear_action.isVisible()
    qtbot.wait(500)
    clear_action.triggered.emit()
    assert not clear_action.isVisible()
    assert not findreplace.number_matches_text.isVisible()
    edit.clear()
    edit.setFocus()
    qtbot.keyClicks(edit, 'bar')
    qtbot.wait(500)
    assert findreplace.messages_action.isVisible()
    clear_action.triggered.emit()
    qtbot.wait(500)
    assert not findreplace.messages_action.isVisible()

def test_replace_all_backslash(findreplace_editor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test that we can replace all occurrences of a certain text with an\n    expression that contains backslashes.\n\n    This is a regression test for issue spyder-ide/spyder#21007\n    '
    editor = findreplace_editor.editor
    findreplace = findreplace_editor.findreplace
    editor.set_text('a | b | c')
    edit = findreplace.search_text.lineEdit()
    edit.setFocus()
    qtbot.keyClicks(edit, '|')
    findreplace.replace_text_button.setChecked(True)
    findreplace.replace_text.setCurrentText('\\')
    qtbot.wait(100)
    findreplace.replace_find_all()
    assert editor.toPlainText() == 'a \\ b \\ c'
    editor.selectAll()
    qtbot.keyClick(edit, Qt.Key_Delete)
    edit.clear()
    editor.set_text('\\Psi\n\\alpha\n\\beta\n\\alpha')
    edit.setFocus()
    qtbot.keyClicks(edit, '\\alpha')
    findreplace.replace_text.setCurrentText('\\beta')
    qtbot.wait(100)
    findreplace.replace_find_all()
    assert editor.toPlainText() == '\\Psi\n\\beta\n\\beta\n\\beta'
if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])