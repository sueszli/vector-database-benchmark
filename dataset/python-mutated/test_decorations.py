"""Tests for editor decorations."""
import os.path as osp
import random
from unittest.mock import patch
from flaky import flaky
import pytest
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont, QTextCursor
from spyder.plugins.editor.widgets.codeeditor import CodeEditor
HERE = osp.dirname(osp.realpath(__file__))
PARENT = osp.dirname(HERE)

def test_decorations(codeeditor, qtbot):
    if False:
        print('Hello World!')
    'Test decorations.'
    editor = codeeditor
    editor.resize(640, random.randint(200, 500))
    base_function = 'def some_function():\n    some_variable = 1\n    some_variable += 2\n    return some_variable\n\n'
    text = ''
    for __ in range(100):
        base_text = base_function * random.randint(2, 8) + '# %%\n'
        text = text + base_text
    editor.set_text(text)
    editor.go_to_line(2)
    cursor = editor.textCursor()
    cursor.movePosition(QTextCursor.Right, n=5)
    editor.setTextCursor(cursor)
    qtbot.wait(3000)
    decorations = editor.decorations._sorted_decorations()
    assert len(decorations) == 2 + text.count('some_variable')
    assert decorations[0].kind == 'current_cell'
    assert decorations[1].kind == 'current_line'
    assert all([d.kind == 'occurrences' for d in decorations[2:5]])
    selected_texts = [d.cursor.selectedText() for d in decorations]
    assert set(selected_texts[2:]) == set(['some_variable'])
    (first, last) = editor.get_buffer_block_numbers()
    max_decorations = last - first
    assert len(editor.extraSelections()) < max_decorations
    editor.decorations.clear()
    editor.decorations._update()
    assert editor.decorations._sorted_decorations() == []
    line_number = random.randint(100, editor.blockCount())
    editor.go_to_line(line_number)
    qtbot.wait(editor.UPDATE_DECORATIONS_TIMEOUT + 100)
    decorations = editor.decorations._sorted_decorations()
    assert decorations[0].kind == 'current_cell'

@flaky(max_runs=10)
def test_update_decorations_when_scrolling(qtbot):
    if False:
        print('Hello World!')
    "\n    Test how many calls we're doing to update decorations when\n    scrolling.\n    "
    patched_object = 'spyder.plugins.editor.utils.decoration.TextDecorationsManager._update'
    with patch(patched_object) as _update:
        editor = CodeEditor(parent=None)
        editor.setup_editor(language='Python', color_scheme='spyder/dark', font=QFont('Monospace', 10))
        editor.resize(640, 480)
        editor.show()
        qtbot.addWidget(editor)
        assert _update.call_count == 0
        with open(osp.join(PARENT, 'codeeditor.py'), 'r', encoding='utf-8') as f:
            text = f.read()
        editor.set_text(text)
        assert _update.call_count == 0
        scrollbar = editor.verticalScrollBar()
        for i in range(6):
            scrollbar.setValue(i * 70)
            qtbot.wait(100)
        assert _update.call_count == 1
        qtbot.wait(editor.UPDATE_DECORATIONS_TIMEOUT + 100)
        assert _update.call_count == 2
        scrollbar = editor.verticalScrollBar()
        value = scrollbar.value()
        for __ in range(400):
            scrollbar.setValue(value + 1)
            value = scrollbar.value()
        assert _update.call_count == 2
        qtbot.wait(editor.UPDATE_DECORATIONS_TIMEOUT + 100)
        assert _update.call_count == 3
        (_, last) = editor.get_visible_block_numbers()
        editor.go_to_line(last)
        for __ in range(200):
            qtbot.keyPress(editor, Qt.Key_Down)
        qtbot.wait(editor.UPDATE_DECORATIONS_TIMEOUT + 100)
        assert _update.call_count == 4
        for __ in range(200):
            qtbot.keyPress(editor, Qt.Key_Up)
        qtbot.wait(editor.UPDATE_DECORATIONS_TIMEOUT + 100)
        assert _update.call_count == 5
if __name__ == '__main__':
    pytest.main()