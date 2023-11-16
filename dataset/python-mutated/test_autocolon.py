"""
Tests for the automatic insertion of colons in the editor
"""
from qtpy.QtGui import QTextCursor
import pytest
from spyder.utils.qthelpers import qapplication
from spyder.plugins.editor.widgets.codeeditor import CodeEditor

def construct_editor(text):
    if False:
        while True:
            i = 10
    app = qapplication()
    editor = CodeEditor(parent=None)
    editor.setup_editor(language='Python')
    editor.set_text(text)
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.End)
    editor.setTextCursor(cursor)
    return editor

def test_no_auto_colon_after_simple_statement():
    if False:
        while True:
            i = 10
    editor = construct_editor('x = 1')
    assert editor.autoinsert_colons() == False

def test_auto_colon_after_if_statement():
    if False:
        print('Hello World!')
    editor = construct_editor('if x == 1')
    assert editor.autoinsert_colons() == True

def test_no_auto_colon_if_not_at_end_of_line():
    if False:
        for i in range(10):
            print('nop')
    editor = construct_editor('if x == 1')
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Left)
    editor.setTextCursor(cursor)
    assert editor.autoinsert_colons() == False

def test_no_auto_colon_if_unterminated_string():
    if False:
        for i in range(10):
            print('nop')
    editor = construct_editor("if x == '1")
    assert editor.autoinsert_colons() == False

def test_no_auto_colon_in_comment():
    if False:
        return 10
    editor = construct_editor('if x == 1 # comment')
    assert editor.autoinsert_colons() == False

def test_no_auto_colon_if_already_ends_in_colon():
    if False:
        return 10
    editor = construct_editor('if x == 1:')
    assert editor.autoinsert_colons() == False

def test_no_auto_colon_if_ends_in_backslash():
    if False:
        print('Hello World!')
    editor = construct_editor('if x == 1 \\')
    assert editor.autoinsert_colons() == False

def test_no_auto_colon_in_one_line_if_statement():
    if False:
        i = 10
        return i + 15
    editor = construct_editor('if x < 0: x = 0')
    assert editor.autoinsert_colons() == False

def test_auto_colon_even_if_colon_inside_brackets():
    if False:
        i = 10
        return i + 15
    editor = construct_editor("if text[:-1].endswith('bla')")
    assert editor.autoinsert_colons() == True

def test_no_auto_colon_in_listcomp_over_two_lines():
    if False:
        return 10
    editor = construct_editor('ns = [ n for ns in range(10) \n if n < 5 ]')
    assert editor.autoinsert_colons() == False

def test_no_auto_colon_in_listcomp_over_three_lines():
    if False:
        print('Hello World!')
    'Tests spyder-ide/spyder#1354'
    editor = construct_editor('ns = [ n \n for ns in range(10) \n if n < 5 ]')
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Up)
    cursor.movePosition(QTextCursor.EndOfLine)
    editor.setTextCursor(cursor)
    assert not editor.autoinsert_colons()

@pytest.mark.xfail
def test_auto_colon_even_if_colon_inside_quotes():
    if False:
        return 10
    editor = construct_editor("if text == ':'")
    assert editor.autoinsert_colons() == True

@pytest.mark.xfail
def test_auto_colon_in_two_if_statements_on_one_line():
    if False:
        while True:
            i = 10
    editor = construct_editor('if x < 0: x = 0; if x == 0')
    assert editor.autoinsert_colons() == True
if __name__ == '__main__':
    pytest.main()