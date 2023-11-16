"""Tests some cases were completions need to be hidden."""
from flaky import flaky
import pytest
from qtpy.QtCore import Qt

@pytest.mark.order(1)
@flaky(max_runs=10)
def test_automatic_completions_hide_complete(completions_codeeditor, qtbot):
    if False:
        i = 10
        return i + 15
    'Test on-the-fly completion closing when already complete.\n\n    Regression test for issue #11600 and pull requests #11824, #12140\n    and #12710.\n    '
    (code_editor, _) = completions_codeeditor
    completion = code_editor.completion_widget
    delay = 50
    code_editor.toggle_code_snippets(False)
    code_editor.set_text('some = 0\nsomething = 1\n')
    cursor = code_editor.textCursor()
    code_editor.moveCursor(cursor.End)
    with qtbot.waitSignal(completion.sig_show_completions, timeout=10000) as sig:
        qtbot.keyClicks(code_editor, 'some', delay=delay)
    assert 'some' in [x['label'] for x in sig.args[0]]
    assert 'something' in [x['label'] for x in sig.args[0]]
    qtbot.keyClicks(code_editor, 'thing', delay=delay)
    qtbot.wait(500)
    assert completion.isHidden()
    qtbot.keyPress(code_editor, Qt.Key_Enter, delay=300)
    qtbot.keyClicks(code_editor, 'print(something', delay=delay)
    qtbot.wait(500)
    assert completion.isHidden()
    qtbot.keyClicks(code_editor, ')', delay=delay)
    qtbot.keyPress(code_editor, Qt.Key_Enter, delay=300)
    qtbot.keyClicks(code_editor, 'a = {something', delay=delay)
    qtbot.wait(500)
    assert completion.isHidden()
    code_editor.moveCursor(cursor.End)
    qtbot.keyPress(code_editor, Qt.Key_Enter)
    qtbot.keyClicks(code_editor, 'some', delay=delay)
    qtbot.keyPress(code_editor, Qt.Key_Enter)
    qtbot.keyClicks(code_editor, '  None', delay=delay)
    if completion.isVisible():
        qtbot.keyPress(completion, Qt.Key_Enter)
    code_editor.moveCursor(cursor.StartOfWord)
    qtbot.keyPress(code_editor, Qt.Key_Backspace)
    qtbot.wait(2000)
    assert completion.isHidden()
    qtbot.keyPress(code_editor, Qt.Key_Backspace)
    qtbot.wait(2000)
    assert completion.isHidden()
    code_editor.moveCursor(cursor.End)
    qtbot.keyPress(code_editor, Qt.Key_Enter)
    qtbot.keyClicks(code_editor, 'some +  some ', delay=delay)
    qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.keyPress(code_editor, Qt.Key_Backspace)
    qtbot.wait(2000)
    assert completion.isHidden()
    code_editor.toggle_code_snippets(True)

@pytest.mark.order(1)
@flaky(max_runs=5)
def test_automatic_completions_widget_visible(completions_codeeditor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Test on-the-fly completions when the widget is visible and the Backspace\n    key is pressed.\n\n    Regression test for PR #12710\n    '
    (code_editor, _) = completions_codeeditor
    completion = code_editor.completion_widget
    code_editor.toggle_code_snippets(False)
    code_editor.set_text('import math')
    cursor = code_editor.textCursor()
    code_editor.moveCursor(cursor.End)
    qtbot.keyPress(code_editor, Qt.Key_Enter, delay=300)
    with qtbot.waitSignal(completion.sig_show_completions, timeout=10000):
        qtbot.keyClicks(code_editor, 'math.acosh', delay=300)
    assert completion.isVisible()
    qtbot.keyPress(code_editor, Qt.Key_Backspace, delay=300)
    qtbot.wait(500)
    assert completion.isVisible()
    qtbot.keyPress(code_editor, Qt.Key_Backspace, delay=300)
    qtbot.wait(500)
    assert completion.isVisible()
    code_editor.toggle_code_snippets(True)
if __name__ == '__main__':
    pytest.main(['test_completion.py', '--run-slow'])