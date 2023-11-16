"""
Tests for bookmarks.
"""
import pytest
from qtpy.QtGui import QTextCursor

def test_save_bookmark(editor_plugin_open_files):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test Plugin.save_bookmark.\n\n    Test saving of bookmarks by looking at data in blocks. Reassignment\n    should remove data from old block and put it in new.\n    '
    (editor, _, _) = editor_plugin_open_files(None, None)
    editorstack = editor.get_current_editorstack()
    edtr = editorstack.get_current_editor()
    cursor = edtr.textCursor()
    editor.save_bookmark(1)
    bookmarks = edtr.document().findBlockByNumber(0).userData().bookmarks
    assert bookmarks == [(1, 0)]
    cursor.movePosition(QTextCursor.Down, n=1)
    cursor.movePosition(QTextCursor.Right, n=2)
    edtr.setTextCursor(cursor)
    editor.save_bookmark(1)
    bookmarks = edtr.document().findBlockByNumber(1).userData().bookmarks
    assert bookmarks == [(1, 2)]
    bookmarks = edtr.document().findBlockByNumber(0).userData().bookmarks
    assert bookmarks == []

def test_load_bookmark(editor_plugin_open_files):
    if False:
        print('Hello World!')
    '\n    Test that loading a bookmark works.\n\n    Check this by saving and loading bookmarks and checking for cursor\n    position. Also over multiple files.\n    '
    (editor, _, _) = editor_plugin_open_files(None, None)
    editorstack = editor.get_current_editorstack()
    edtr = editorstack.get_current_editor()
    cursor = edtr.textCursor()
    editor.save_bookmark(1)
    cursor.movePosition(QTextCursor.Down, n=1)
    cursor.movePosition(QTextCursor.Right, n=4)
    edtr.setTextCursor(cursor)
    assert edtr.get_cursor_line_column() != (0, 0)
    editor.load_bookmark(1)
    assert edtr.get_cursor_line_column() == (0, 0)
    cursor.movePosition(QTextCursor.Down, n=1)
    cursor.movePosition(QTextCursor.Right, n=19)
    edtr.setTextCursor(cursor)
    editor.save_bookmark(2)
    edtr.stdkey_backspace()
    edtr.stdkey_backspace()
    editor.load_bookmark(2)
    assert edtr.get_cursor_line_column() == (1, 20)
    editor.save_bookmark(2)
    editorstack.tabs.setCurrentIndex(1)
    editor.load_bookmark(2)
    assert editorstack.tabs.currentIndex() == 0
if __name__ == '__main__':
    pytest.main()