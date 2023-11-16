"""Tests for editor calltips and hover hints tooltips."""
import os
import sys
from qtpy.QtCore import Qt, QPoint
from qtpy.QtGui import QTextCursor
import pytest
from spyder.plugins.editor.extensions.closebrackets import CloseBracketsExtension
TEST_SIG = 'some_function(foo={}, hello=None)'
TEST_DOCSTRING = 'This is the test docstring.'
TEST_TEXT = "'''Testing something'''\ndef {SIG}:\n    '''{DOC}'''\n\n\nsome_function".format(SIG=TEST_SIG, DOC=TEST_DOCSTRING)

@pytest.mark.order(2)
def test_hide_calltip(completions_codeeditor, qtbot):
    if False:
        for i in range(10):
            print('nop')
    "Test that calltips are hidden when a matching ')' is found."
    (code_editor, _) = completions_codeeditor
    code_editor.show()
    code_editor.raise_()
    code_editor.setFocus()
    text = 'a = "sometext {}"\nprint(a.format'
    code_editor.set_text(text)
    code_editor.go_to_line(2)
    code_editor.move_cursor(14)
    calltip = code_editor.calltip_widget
    assert not calltip.isVisible()
    with qtbot.waitSignal(code_editor.sig_signature_invoked, timeout=30000):
        qtbot.keyClicks(code_editor, '(', delay=3000)
    qtbot.waitUntil(lambda : calltip.isVisible(), timeout=3000)
    qtbot.keyClicks(code_editor, '"hello"')
    qtbot.keyClicks(code_editor, ')', delay=330)
    assert calltip.isVisible()
    qtbot.keyClicks(code_editor, ')', delay=330)
    qtbot.waitUntil(lambda : not calltip.isVisible(), timeout=3000)
    qtbot.keyClick(code_editor, Qt.Key_Enter, delay=330)
    assert not calltip.isVisible()

@pytest.mark.order(2)
@pytest.mark.parametrize('params', [('dict', 'dict'), ('type', 'type'), ('"".format', '-> str'), (TEST_TEXT, TEST_SIG)])
def test_get_calltips(qtbot, completions_codeeditor, params):
    if False:
        print('Hello World!')
    'Test that the editor is returning hints.'
    (code_editor, _) = completions_codeeditor
    (param, expected_output_text) = params
    code_editor.set_text(param)
    code_editor.moveCursor(QTextCursor.End)
    code_editor.calltip_widget.hide()
    bracket_extension = code_editor.editor_extensions.get(CloseBracketsExtension)
    with qtbot.waitSignal(code_editor.sig_signature_invoked, timeout=30000) as blocker:
        qtbot.keyPress(code_editor, Qt.Key_ParenLeft, delay=1000)
        qtbot.wait(2000)
        args = blocker.args
        print('args:', [args])
        output_text = args[0]['signatures']['label']
        assert expected_output_text in output_text
        code_editor.calltip_widget.hide()
    bracket_extension.enable = False
    with qtbot.waitSignal(code_editor.sig_signature_invoked, timeout=30000) as blocker:
        qtbot.keyPress(code_editor, Qt.Key_ParenLeft, delay=1000)
        qtbot.wait(2000)
        args = blocker.args
        print('args:', [args])
        output_text = args[0]['signatures']['label']
        assert expected_output_text in output_text
        code_editor.calltip_widget.hide()
    bracket_extension.enable = True

@pytest.mark.order(2)
@pytest.mark.skipif(not os.name == 'nt', reason='Only works on Windows')
@pytest.mark.parametrize('params', [('"".format', '-> str'), ('import math', 'module'), (TEST_TEXT, TEST_DOCSTRING)])
def test_get_hints(qtbot, completions_codeeditor, params, capsys):
    if False:
        for i in range(10):
            print('nop')
    'Test that the editor is returning hover hints.'
    (code_editor, _) = completions_codeeditor
    (param, expected_output_text) = params
    qtbot.mouseMove(code_editor, QPoint(400, 400))
    code_editor.set_text(param)
    code_editor.moveCursor(QTextCursor.End)
    qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.wait(1000)
    (x, y) = code_editor.get_coordinates('cursor')
    point = code_editor.calculate_real_position(QPoint(x, y))
    with qtbot.waitSignal(code_editor.sig_display_object_info, timeout=30000) as blocker:
        qtbot.mouseMove(code_editor, point)
        qtbot.mouseClick(code_editor, Qt.LeftButton, pos=point)
        qtbot.waitUntil(lambda : code_editor.tooltip_widget.isVisible(), timeout=10000)
        args = blocker.args
        print('args:', [args])
        output_text = args[0]
        assert expected_output_text in output_text
        code_editor.tooltip_widget.hide()
        captured = capsys.readouterr()
        assert captured.err == ''

@pytest.mark.order(2)
@pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on Mac')
@pytest.mark.parametrize('text', ['def test():\n    pass\n\ntest', '# a comment', '"a string"'])
def test_get_hints_not_triggered(qtbot, completions_codeeditor, text):
    if False:
        print('Hello World!')
    'Test that the editor is not returning hover hints for empty docs.'
    (code_editor, _) = completions_codeeditor
    code_editor.set_text(text)
    qtbot.mouseMove(code_editor, QPoint(400, 400))
    code_editor.moveCursor(QTextCursor.End)
    for _ in range(3):
        qtbot.keyPress(code_editor, Qt.Key_Left)
    qtbot.wait(1000)
    (x, y) = code_editor.get_coordinates('cursor')
    point = code_editor.calculate_real_position(QPoint(x, y))
    with qtbot.waitSignal(code_editor.completions_response_signal, timeout=30000):
        qtbot.mouseMove(code_editor, point)
        qtbot.mouseClick(code_editor, Qt.LeftButton, pos=point)
        qtbot.wait(1000)
        assert not code_editor.tooltip_widget.isVisible()