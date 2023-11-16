import os
import sys
import pytest
from qtpy.QtCore import QPoint, Qt
from qtpy.QtGui import QFont
from spyder.config.base import running_in_ci
from spyder.plugins.editor.widgets.codeeditor import CodeEditor
from spyder.plugins.debugger.utils.breakpointsmanager import BreakpointsManager

@pytest.fixture
def editor_bot(qtbot):
    if False:
        return 10
    widget = CodeEditor(None)
    widget.setup_editor(linenumbers=True, markers=True, show_blanks=True, scrollflagarea=True, font=QFont('Courier New', 10), color_scheme='Zenburn', language='Python')
    qtbot.addWidget(widget)
    return widget
long_code = 'Line1\nLine2\nLine3\nLine4\nLine5\nLine6\nLine7\nLine8\nLine9\nLine10\nLine11\nLine12\nLine13\nLine14\nLine15\nLine16\nLine17\nLine18\nLine19\nLine20\n'
short_code = 'line1: Occurences\nline2: Breakpoints\nline3: TODOs\nline4: Code Analysis: warning\nline5: Code Analysis: error\nline6: Found Results\n'

def test_enabled(editor_bot):
    if False:
        return 10
    '"Test that enabling and disabling the srollflagarea panel make\n    it visible or invisible depending on the case.'
    editor = editor_bot
    sfa = editor.scrollflagarea
    editor.show()
    editor.set_text(short_code)
    assert sfa.isVisible()
    sfa.set_enabled(False)
    assert not sfa.isVisible()

@pytest.mark.skipif(sys.platform.startswith('linux'), reason='Fails in Linux')
def test_flag_painting(editor_bot, qtbot):
    if False:
        i = 10
        return i + 15
    '"Test that there is no error when painting all flag types on the\n    scrollbar area when the editor vertical scrollbar is visible and not\n    visible. There is seven different flags: breakpoints, todos, warnings,\n    errors, found_results, and occurences'
    editor = editor_bot
    editor.filename = 'file.py'
    editor.breakpoints_manager = BreakpointsManager(editor)
    sfa = editor.scrollflagarea
    editor.resize(450, 300)
    editor.show()
    editor.set_text(short_code)
    qtbot.waitUntil(lambda : not sfa.slider)
    editor.breakpoints_manager.toogle_breakpoint(line_number=2)
    editor.process_todo([[True, 3]])
    analysis = [{'source': 'pycodestyle', 'range': {'start': {'line': 4, 'character': 0}, 'end': {'line': 4, 'character': 1}}, 'line': 4, 'code': 'E227', 'message': 'E227 warning', 'severity': 2}, {'source': 'pyflakes', 'range': {'start': {'line': 5, 'character': 0}, 'end': {'line': 5, 'character': 1}}, 'message': 'syntax error', 'severity': 1}]
    editor.process_code_analysis(analysis)
    editor.highlight_found_results('line6')
    with qtbot.waitSignal(editor.sig_flags_changed, raising=True, timeout=5000):
        cursor = editor.textCursor()
        cursor.setPosition(2)
        editor.setTextCursor(cursor)
    editor.set_text(long_code)
    editor.breakpoints_manager.toogle_breakpoint(line_number=2)
    editor.process_todo([[True, 3]])
    analysis = [{'source': 'pycodestyle', 'range': {'start': {'line': 4, 'character': 0}, 'end': {'line': 4, 'character': 1}}, 'line': 4, 'code': 'E227', 'message': 'E227 warning', 'severity': 2}, {'source': 'pyflakes', 'range': {'start': {'line': 5, 'character': 0}, 'end': {'line': 5, 'character': 1}}, 'message': 'syntax error', 'severity': 1}]
    editor.process_code_analysis(analysis)
    editor.highlight_found_results('line6')
    with qtbot.waitSignal(editor.sig_flags_changed, raising=True, timeout=5000):
        cursor = editor.textCursor()
        cursor.setPosition(2)
        editor.setTextCursor(cursor)

def test_range_indicator_visible_on_hover_only(editor_bot, qtbot):
    if False:
        i = 10
        return i + 15
    'Test that the slider range indicator is visible only when hovering\n    over the scrollflag area when the editor vertical scrollbar is visible.\n    The scrollflag area should remain hidden at all times when the editor\n    vertical scrollbar is not visible.'
    editor = editor_bot
    sfa = editor.scrollflagarea
    editor.show()
    editor.set_text(short_code)
    editor.resize(450, 150)
    qtbot.waitUntil(lambda : not sfa.slider)
    x = int(sfa.width() / 2)
    y = int(sfa.height() / 2)
    qtbot.mouseMove(sfa, pos=QPoint(x, y), delay=-1)
    assert sfa._range_indicator_is_visible is False
    editor.set_text(long_code)
    editor.resize(450, 150)
    qtbot.waitUntil(lambda : sfa.slider)
    x = int(sfa.width() / 2)
    y = int(sfa.height() / 2)
    qtbot.mouseMove(sfa, pos=QPoint(x, y), delay=-1)
    qtbot.wait(500)
    assert sfa._range_indicator_is_visible is True
    x = int(editor.width() / 2)
    y = int(editor.height() / 2)
    qtbot.mouseMove(editor, pos=QPoint(x, y), delay=-1)
    qtbot.waitUntil(lambda : not sfa._range_indicator_is_visible)

def test_range_indicator_alt_modifier_response(editor_bot, qtbot):
    if False:
        return 10
    'Test that the slider range indicator is visible while the alt key is\n    held down while the cursor is over the editor, but outside of the\n    scrollflag area. In addition, while the alt key is held down, mouse\n    click events in the editor should be forwarded to the scrollfag area and\n    should set the value of the editor vertical scrollbar.'
    editor = editor_bot
    sfa = editor.scrollflagarea
    sfa._unit_testing = True
    vsb = editor.verticalScrollBar()
    editor.show()
    editor.resize(600, 150)
    editor.set_text(long_code)
    qtbot.waitUntil(lambda : sfa.slider)
    qtbot.wait(500)
    w = editor.width()
    h = editor.height()
    qtbot.mousePress(editor, Qt.LeftButton, pos=QPoint(w // 2, h // 2))
    qtbot.keyPress(editor, Qt.Key_Alt)
    editor.resize(600, 150)
    x = int(sfa.width() / 2)
    y = int(sfa.height() / 2)
    qtbot.mouseMove(sfa, pos=QPoint(x, y), delay=-1)
    qtbot.waitUntil(lambda : sfa._range_indicator_is_visible)
    with qtbot.waitSignal(editor.sig_alt_left_mouse_pressed, raising=True):
        qtbot.mousePress(editor.viewport(), Qt.LeftButton, Qt.AltModifier, QPoint(w // 2, h // 2))
    assert vsb.value() == (vsb.minimum() + vsb.maximum()) // 2
    with qtbot.waitSignal(editor.sig_alt_left_mouse_pressed, raising=True):
        qtbot.mousePress(editor.viewport(), Qt.LeftButton, Qt.AltModifier, QPoint(w // 2, 1))
    assert vsb.value() == vsb.minimum()
    with qtbot.waitSignal(editor.sig_alt_left_mouse_pressed, raising=True):
        qtbot.mousePress(editor.viewport(), Qt.LeftButton, Qt.AltModifier, QPoint(w // 2, h - 1))
    assert vsb.value() == vsb.maximum()
    editor.resize(600, 150)
    x = int(sfa.width() / 2)
    y = int(sfa.height() / 2)
    qtbot.mouseMove(sfa, pos=QPoint(x * 100, y), delay=-1)
    qtbot.keyRelease(editor, Qt.Key_Alt)
    qtbot.waitUntil(lambda : not sfa._range_indicator_is_visible, timeout=3000)
if __name__ == '__main__':
    pytest.main([os.path.basename(__file__)])