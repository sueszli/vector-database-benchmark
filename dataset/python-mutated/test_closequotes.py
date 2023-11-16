"""Tests for close quotes."""
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QTextCursor
from spyder.config.base import running_in_ci
from spyder.plugins.editor.widgets.codeeditor import CodeEditor
from spyder.plugins.editor.utils.editor import TextHelper
from spyder.plugins.editor.extensions.closequotes import CloseQuotesExtension

@pytest.fixture
def editor_close_quotes(qtbot):
    if False:
        return 10
    'Set up Editor with close quotes activated.'
    editor = CodeEditor(parent=None)
    editor.setup_editor(color_scheme='spyder/dark', font=QFont('Courier New', 10), language='Python', close_quotes=True)
    editor.resize(480, 360)
    editor.show()
    qtbot.addWidget(editor)
    return editor

@pytest.mark.parametrize('text, expected_text, cursor_column', [('"', '""', 1), ("'", "''", 1), ('#"', '#"', 2), ("#'", "#'", 2), ('"""', '"""', 3), ("'''", "'''", 3), ('""""', '""""""', 3), ("''''", "''''''", 3), ('"some_string"', '"some_string"', 13), ("'some_string'", "'some_string'", 13), ('"\\""', '"\\""', 4), ("'\\''", "'\\''", 4), ('"\\\\"', '"\\\\"', 4), ("'\\\\'", "'\\\\'", 4)])
def test_close_quotes(qtbot, editor_close_quotes, text, expected_text, cursor_column):
    if False:
        i = 10
        return i + 15
    'Test insertion of extra quotes.'
    editor = editor_close_quotes
    qtbot.keyClicks(editor, text)
    if not running_in_ci():
        qtbot.wait(1000)
    assert editor.toPlainText() == expected_text
    assert cursor_column == TextHelper(editor).current_column_nbr()

@pytest.mark.parametrize('text, expected_text, cursor_column', [('()', '("")', 2), ('{}', '{""}', 2), ('[]', '[""]', 2), (',', '"",', 1), (':', '"":', 1), (';', '"";', 1), ('a', '"a', 1)])
def test_trailing_text(qtbot, editor_close_quotes, text, expected_text, cursor_column):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test insertion of extra quotes inside brackets and before commas,\n    colons and semi-colons.\n    '
    editor = editor_close_quotes
    qtbot.keyClicks(editor, text)
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == expected_text
    assert cursor_column == TextHelper(editor).current_column_nbr()

def test_selected_text(qtbot, editor_close_quotes):
    if False:
        for i in range(10):
            print('nop')
    'Test insert surronding quotes to selected text.'
    editor = editor_close_quotes
    editor.set_text('some text')
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 4)
    editor.setTextCursor(cursor)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '"some" text'
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '""some"" text'
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '"""some""" text'

def test_selected_text_multiple_lines(qtbot, editor_close_quotes):
    if False:
        for i in range(10):
            print('nop')
    'Test insert surronding quotes to multiple lines selected text.'
    editor = editor_close_quotes
    text = 'some text\n\nsome text'
    editor.set_text(text)
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 4)
    cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 2)
    editor.setTextCursor(cursor)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '"some text\n\nsome" text'
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '""some text\n\nsome"" text'
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '"""some text\n\nsome""" text'

def test_close_quotes_in_brackets(qtbot, editor_close_quotes):
    if False:
        for i in range(10):
            print('nop')
    'Test quote completion in nested brackets.'
    editor = editor_close_quotes
    editor.textCursor().insertText('foo()')
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == 'foo("")'
    assert editor.textCursor().columnNumber() == 5
    qtbot.keyPress(editor, Qt.Key_Delete)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == 'foo("")'
    assert editor.textCursor().columnNumber() == 6
    qtbot.keyClicks(editor, ', ,')
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == 'foo("", "",)'
    assert editor.textCursor().columnNumber() == 9
    editor.move_cursor(2)
    qtbot.keyClicks(editor, ' { },')
    editor.move_cursor(-3)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == 'foo("", "", {"" },)'
    assert editor.textCursor().columnNumber() == 14
    editor.move_cursor(4)
    qtbot.keyClicks(editor, ' bar')
    editor.move_cursor(-3)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == 'foo("", "", {"" }, "bar)'
    assert editor.textCursor().columnNumber() == 20

def test_activate_deactivate(qtbot, editor_close_quotes):
    if False:
        print('Hello World!')
    'Test activating/desctivating close quotes editor extension.'
    editor = editor_close_quotes
    quote_extension = editor.editor_extensions.get(CloseQuotesExtension)
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '""'
    editor.set_text('')
    quote_extension.enabled = False
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '"'
    editor.set_text('')
    quote_extension.enabled = True
    qtbot.keyClicks(editor, '"')
    assert editor.toPlainText() == '""'
if __name__ == '__main__':
    pytest.main()