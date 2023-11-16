"""Tests for close brackets."""
import pytest
from qtpy.QtGui import QTextCursor
from spyder.utils.qthelpers import qapplication
from spyder.plugins.editor.widgets.codeeditor import CodeEditor
from spyder.plugins.editor.utils.editor import TextHelper
from spyder.plugins.editor.extensions.closebrackets import CloseBracketsExtension

@pytest.fixture
def editor_close_brackets():
    if False:
        print('Hello World!')
    'Set up Editor with close brackets activated.'
    app = qapplication()
    editor = CodeEditor(parent=None)
    kwargs = {}
    kwargs['language'] = 'Python'
    kwargs['close_parentheses'] = True
    editor.setup_editor(**kwargs)
    return editor

def test_bracket_closing_new_line(qtbot, editor_close_brackets):
    if False:
        return 10
    '\n    Test bracket completion with existing brackets in a new line.\n\n    For spyder-ide/spyder#11217\n    '
    editor = editor_close_brackets
    editor.textCursor().insertText('foo(\nbar)')
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '()')
    assert editor.toPlainText() == 'foo(\nbar())'
    assert editor.textCursor().columnNumber() == 5
    qtbot.keyClicks(editor, ')')
    assert editor.toPlainText() == 'foo(\nbar())'
    assert editor.textCursor().columnNumber() == 6

@pytest.mark.parametrize('text, expected_text, cursor_column', [('(', '()', 1), ('{', '{}', 1), ('[', '[]', 1)])
def test_close_brackets(qtbot, editor_close_brackets, text, expected_text, cursor_column):
    if False:
        return 10
    'Test insertion of brackets.'
    editor = editor_close_brackets
    qtbot.keyClicks(editor, text)
    assert editor.toPlainText() == expected_text
    assert cursor_column == TextHelper(editor).current_column_nbr()

@pytest.mark.parametrize('text, expected_text, cursor_column', [('()', '(())', 2), ('{}', '{()}', 2), ('[]', '[()]', 2), (',', '(),', 1), (':', '():', 1), (';', '();', 1)])
def test_nested_brackets(qtbot, editor_close_brackets, text, expected_text, cursor_column):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test completion of brackets inside brackets and before commas,\n    colons and semi-colons.\n    '
    editor = editor_close_brackets
    qtbot.keyClicks(editor, text)
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '(')
    assert editor.toPlainText() == expected_text
    assert cursor_column == TextHelper(editor).current_column_nbr()

def test_selected_text(qtbot, editor_close_brackets):
    if False:
        print('Hello World!')
    'Test insert surronding brackets to selected text.'
    editor = editor_close_brackets
    editor.set_text('some text')
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 4)
    editor.setTextCursor(cursor)
    qtbot.keyClicks(editor, '(')
    assert editor.toPlainText() == '(some) text'
    qtbot.keyClicks(editor, '}')
    assert editor.toPlainText() == '({some}) text'
    qtbot.keyClicks(editor, '[')
    assert editor.toPlainText() == '({[some]}) text'

def test_selected_text_multiple_lines(qtbot, editor_close_brackets):
    if False:
        i = 10
        return i + 15
    'Test insert surronding brackets to multiple lines selected text.'
    editor = editor_close_brackets
    text = 'some text\n\nsome text'
    editor.set_text(text)
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Right, QTextCursor.KeepAnchor, 4)
    cursor.movePosition(QTextCursor.Down, QTextCursor.KeepAnchor, 2)
    editor.setTextCursor(cursor)
    qtbot.keyClicks(editor, ')')
    assert editor.toPlainText() == '(some text\n\nsome) text'
    qtbot.keyClicks(editor, '{')
    assert editor.toPlainText() == '({some text\n\nsome}) text'
    qtbot.keyClicks(editor, ']')
    assert editor.toPlainText() == '({[some text\n\nsome]}) text'

def test_complex_completion(qtbot, editor_close_brackets):
    if False:
        i = 10
        return i + 15
    'Test bracket completion in nested brackets.'
    editor = editor_close_brackets
    editor.textCursor().insertText('foo(bar)')
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '(')
    assert editor.toPlainText() == 'foo(bar())'
    assert editor.textCursor().columnNumber() == 8
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '[')
    assert editor.toPlainText() == 'foo(bar[())'
    assert editor.textCursor().columnNumber() == 8
    qtbot.keyClicks(editor, ',')
    editor.move_cursor(-1)
    qtbot.keyClicks(editor, '{')
    assert editor.toPlainText() == 'foo(bar[{},())'
    assert editor.textCursor().columnNumber() == 9

def test_bracket_closing(qtbot, editor_close_brackets):
    if False:
        print('Hello World!')
    'Test bracket completion with existing brackets.'
    editor = editor_close_brackets
    editor.textCursor().insertText('foo(bar(x')
    qtbot.keyClicks(editor, ')')
    assert editor.toPlainText() == 'foo(bar(x)'
    assert editor.textCursor().columnNumber() == 10
    qtbot.keyClicks(editor, ')')
    assert editor.toPlainText() == 'foo(bar(x))'
    assert editor.textCursor().columnNumber() == 11
    editor.move_cursor(-2)
    qtbot.keyClicks(editor, ')')
    assert editor.toPlainText() == 'foo(bar(x))'
    assert editor.textCursor().columnNumber() == 10
    qtbot.keyClicks(editor, ')')
    assert editor.toPlainText() == 'foo(bar(x))'
    assert editor.textCursor().columnNumber() == 11

def test_activate_deactivate(qtbot, editor_close_brackets):
    if False:
        return 10
    'Test activating/desctivating close quotes editor extension.'
    editor = editor_close_brackets
    bracket_extension = editor.editor_extensions.get(CloseBracketsExtension)
    qtbot.keyClicks(editor, '(')
    assert editor.toPlainText() == '()'
    editor.set_text('')
    bracket_extension.enabled = False
    qtbot.keyClicks(editor, '(')
    assert editor.toPlainText() == '('
    editor.set_text('')
    bracket_extension.enabled = True
    qtbot.keyClicks(editor, '(')
    assert editor.toPlainText() == '()'
if __name__ == '__main__':
    pytest.main()