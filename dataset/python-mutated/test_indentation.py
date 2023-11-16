"""
Tests for the indentation feature
"""
import pytest
from qtpy.QtGui import QTextCursor
from spyder.py3compat import to_text_string
from spyder.plugins.editor.widgets.codeeditor import CodeEditor

def make_indent(editor, single_line=True, start_line=1):
    if False:
        print('Hello World!')
    'Indent and return code.'
    editor.go_to_line(start_line)
    if not single_line:
        editor.moveCursor(QTextCursor.End, mode=QTextCursor.KeepAnchor)
    editor.indent()
    text = editor.toPlainText()
    return to_text_string(text)

def make_unindent(editor, single_line=True, start_line=1):
    if False:
        return 10
    'Unindent and return code.'
    editor.go_to_line(start_line)
    if not single_line:
        editor.moveCursor(QTextCursor.End, mode=QTextCursor.KeepAnchor)
    editor.unindent()
    text = editor.toPlainText()
    return to_text_string(text)

@pytest.fixture
def codeeditor_indent(codeeditor):
    if False:
        while True:
            i = 10
    '\n    Setup CodeEditor with some text useful for folding related tests.\n    '
    editor = codeeditor
    editor.set_indent_chars(' ' * 2)
    return editor

def test_single_line_indent(codeeditor_indent):
    if False:
        return 10
    'Test indentation in a single line.'
    editor = codeeditor_indent
    text = 'class a():\nself.b = 1\nprint(self.b)\n\n'
    expected = 'class a():\n  self.b = 1\nprint(self.b)\n\n'
    editor.set_text(text)
    new_text = make_indent(editor, start_line=2)
    assert new_text == expected

def test_selection_indent(codeeditor_indent):
    if False:
        print('Hello World!')
    'Test indentation with selection of more than one line.'
    editor = codeeditor_indent
    text = 'class a():\nself.b = 1\nprint(self.b)\n\n'
    expected = 'class a():\n  self.b = 1\n  print(self.b)\n  \n'
    editor.set_text(text)
    new_text = make_indent(editor, single_line=False, start_line=2)
    assert new_text == expected

def test_fix_indentation(codeeditor_indent):
    if False:
        for i in range(10):
            print('nop')
    'Test fix_indentation() method.'
    editor = codeeditor_indent
    original = '\t\nclass a():\t\n\tself.b = 1\n\tprint(self.b)\n\n'
    fixed = '  \nclass a():  \n  self.b = 1\n  print(self.b)\n\n'
    editor.set_text(original)
    editor.fix_indentation()
    assert to_text_string(editor.toPlainText()) == fixed
    assert editor.document().isModified()
    editor.undo()
    assert to_text_string(editor.toPlainText()) == original
    assert not editor.document().isModified()
    editor.redo()
    assert to_text_string(editor.toPlainText()) == fixed
    assert editor.document().isModified()

def test_single_line_unindent(codeeditor_indent):
    if False:
        print('Hello World!')
    'Test unindentation in a single line.'
    editor = codeeditor_indent
    text = 'class a():\n  self.b = 1\nprint(self.b)\n\n'
    expected = 'class a():\nself.b = 1\nprint(self.b)\n\n'
    editor.set_text(text)
    new_text = make_unindent(editor, start_line=2)
    assert new_text == expected

def test_selection_unindent(codeeditor_indent):
    if False:
        print('Hello World!')
    'Test unindentation with selection of more than one line.'
    editor = codeeditor_indent
    text = 'class a():\n  self.b = 1\n  print(self.b)\n  \n'
    expected = 'class a():\nself.b = 1\nprint(self.b)\n\n'
    editor.set_text(text)
    new_text = make_unindent(editor, single_line=False, start_line=2)
    assert new_text == expected

def test_single_line_unindent_to_grid(codeeditor_indent):
    if False:
        return 10
    'Test unindentation in a single line.'
    editor = codeeditor_indent
    text = 'class a():\n   self.b = 1\nprint(self.b)\n\n'
    expected = 'class a():\n  self.b = 1\nprint(self.b)\n\n'
    editor.set_text(text)
    new_text = make_unindent(editor, start_line=2)
    assert new_text == expected
    expected2 = 'class a():\nself.b = 1\nprint(self.b)\n\n'
    new_text2 = make_unindent(editor, start_line=2)
    assert new_text2 == expected2

def test_selection_unindent_to_grid(codeeditor_indent):
    if False:
        i = 10
        return i + 15
    'Test unindentation with selection of more than one line.'
    editor = codeeditor_indent
    text = 'class a():\n   self.b = 1\n   print(self.b)\n\n'
    expected = 'class a():\n  self.b = 1\n  print(self.b)\n\n'
    editor.set_text(text)
    new_text = make_unindent(editor, single_line=False, start_line=2)
    assert new_text == expected
    expected2 = 'class a():\nself.b = 1\nprint(self.b)\n\n'
    new_text2 = make_unindent(editor, single_line=False, start_line=2)
    assert new_text2 == expected2
if __name__ == '__main__':
    pytest.main()