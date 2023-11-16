import os
import unittest
import sys
from flaky import flaky
import pytest
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtTest import QTest
from qtconsole.console_widget import ConsoleWidget
from qtconsole.qtconsoleapp import JupyterQtConsoleApp
from . import no_display
from IPython.core.inputtransformer2 import TransformerManager
SHELL_TIMEOUT = 20000

@pytest.fixture
def qtconsole(qtbot):
    if False:
        print('Hello World!')
    'Qtconsole fixture.'
    console = JupyterQtConsoleApp()
    console.initialize(argv=[])
    console.window.confirm_exit = False
    console.window.show()
    yield console
    console.window.close()

@flaky(max_runs=3)
@pytest.mark.parametrize('debug', [True, False])
def test_scroll(qtconsole, qtbot, debug):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make sure the scrolling works.\n    '
    window = qtconsole.window
    shell = window.active_frontend
    control = shell._control
    scroll_bar = control.verticalScrollBar()
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    assert scroll_bar.value() == 0
    code = ['import time', 'def print_numbers():', '    for i in range(1000):', '       print(i)', '       time.sleep(.01)']
    for line in code:
        qtbot.keyClicks(control, line)
        qtbot.keyClick(control, QtCore.Qt.Key_Enter)
    with qtbot.waitSignal(shell.executed):
        qtbot.keyClick(control, QtCore.Qt.Key_Enter, modifier=QtCore.Qt.ShiftModifier)

    def run_line(line, block=True):
        if False:
            return 10
        qtbot.keyClicks(control, line)
        if block:
            with qtbot.waitSignal(shell.executed):
                qtbot.keyClick(control, QtCore.Qt.Key_Enter, modifier=QtCore.Qt.ShiftModifier)
        else:
            qtbot.keyClick(control, QtCore.Qt.Key_Enter, modifier=QtCore.Qt.ShiftModifier)
    if debug:
        run_line('%debug print()', block=False)
        qtbot.keyClick(control, QtCore.Qt.Key_Enter)

        def run_line(line, block=True):
            if False:
                return 10
            qtbot.keyClicks(control, '!' + line)
            qtbot.keyClick(control, QtCore.Qt.Key_Enter, modifier=QtCore.Qt.ShiftModifier)
            if block:
                qtbot.waitUntil(lambda : control.toPlainText().strip().split()[-1] == 'ipdb>')
    prev_position = scroll_bar.value()
    for i in range(20):
        run_line('a = 1')
    assert scroll_bar.value() > prev_position
    prev_position = scroll_bar.value() + scroll_bar.pageStep() // 2
    scroll_bar.setValue(prev_position)
    for i in range(2):
        run_line('a')
    assert scroll_bar.value() == prev_position
    for i in range(10):
        run_line('a')
    assert scroll_bar.value() > prev_position
    prev_position = scroll_bar.value()
    run_line('print_numbers()', block=False)
    qtbot.wait(1000)
    assert scroll_bar.value() > prev_position
    prev_position = scroll_bar.value() - scroll_bar.pageStep()
    scroll_bar.setValue(prev_position)
    qtbot.wait(1000)
    assert scroll_bar.value() == prev_position
    prev_position = scroll_bar.maximum() - scroll_bar.pageStep() * 8 // 10
    scroll_bar.setValue(prev_position)
    qtbot.wait(1000)
    assert scroll_bar.value() > prev_position

@flaky(max_runs=3)
def test_input(qtconsole, qtbot):
    if False:
        i = 10
        return i + 15
    '\n    Test input function\n    '
    window = qtconsole.window
    shell = window.active_frontend
    control = shell._control
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    with qtbot.waitSignal(shell.executed):
        shell.execute('import time')
    input_function = 'input'
    shell.execute('print(' + input_function + "('name: ')); time.sleep(3)")
    qtbot.waitUntil(lambda : control.toPlainText().split()[-1] == 'name:')
    qtbot.keyClicks(control, 'test')
    qtbot.keyClick(control, QtCore.Qt.Key_Enter)
    qtbot.waitUntil(lambda : not shell._reading)
    qtbot.keyClick(control, 'z', modifier=QtCore.Qt.ControlModifier)
    for i in range(10):
        qtbot.keyClick(control, QtCore.Qt.Key_Backspace)
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    assert 'name: test\ntest' in control.toPlainText()

@flaky(max_runs=3)
def test_debug(qtconsole, qtbot):
    if False:
        for i in range(10):
            print('nop')
    '\n    Make sure the cursor works while debugging\n\n    It might not because the console is "_executing"\n    '
    window = qtconsole.window
    shell = window.active_frontend
    control = shell._control
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code = '%debug range(1)'
    qtbot.keyClicks(control, code)
    qtbot.keyClick(control, QtCore.Qt.Key_Enter, modifier=QtCore.Qt.ShiftModifier)
    qtbot.waitUntil(lambda : control.toPlainText().strip().split()[-1] == 'ipdb>', timeout=SHELL_TIMEOUT)
    qtbot.keyClicks(control, 'abd')
    qtbot.wait(100)
    qtbot.keyClick(control, QtCore.Qt.Key_Left)
    qtbot.keyClick(control, 'c')
    qtbot.wait(100)
    assert control.toPlainText().strip().split()[-1] == 'abcd'

@flaky(max_runs=15)
def test_input_and_print(qtconsole, qtbot):
    if False:
        return 10
    '\n    Test that we print correctly mixed input and print statements.\n\n    This is a regression test for spyder-ide/spyder#17710.\n    '
    window = qtconsole.window
    shell = window.active_frontend
    control = shell._control

    def wait_for_input():
        if False:
            for i in range(10):
                print('nop')
        qtbot.waitUntil(lambda : control.toPlainText().splitlines()[-1] == 'Write input: ')
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    code = "\nuser_input = None\nwhile user_input != '':\n    user_input = input('Write input: ')\n    print('Input was entered!')\n"
    shell.execute(code)
    wait_for_input()
    repetitions = 3
    for _ in range(repetitions):
        qtbot.keyClicks(control, '1')
        qtbot.keyClick(control, QtCore.Qt.Key_Enter)
        wait_for_input()
    qtbot.keyClick(control, QtCore.Qt.Key_Enter)
    qtbot.waitUntil(lambda : not shell._reading)
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    output = '   ...: \n' + 'Write input: 1\nInput was entered!\n' * repetitions + 'Write input: \nInput was entered!\n'
    assert output in control.toPlainText()

@flaky(max_runs=5)
@pytest.mark.skipif(os.name == 'nt', reason='no SIGTERM on Windows')
def test_restart_after_kill(qtconsole, qtbot):
    if False:
        print('Hello World!')
    '\n    Test that the kernel correctly restarts after a kill.\n    '
    window = qtconsole.window
    shell = window.active_frontend
    control = shell._control

    def wait_for_restart():
        if False:
            i = 10
            return i + 15
        qtbot.waitUntil(lambda : 'Kernel died, restarting' in control.toPlainText())
    qtbot.waitUntil(lambda : shell._prompt_html is not None, timeout=SHELL_TIMEOUT)
    for _ in range(10):
        with qtbot.waitSignal(shell.executed):
            shell.execute('%clear')
        qtbot.wait(500)
        code = 'import os, signal; os.kill(os.getpid(), signal.SIGTERM)'
        shell.execute(code)
        qtbot.waitUntil(lambda : 'Kernel died, restarting' in control.toPlainText())
        qtbot.waitUntil(lambda : control.toPlainText().splitlines()[-1] == 'In [1]: ')
        qtbot.wait(500)

@pytest.mark.skipif(no_display, reason="Doesn't work without a display")
class TestConsoleWidget(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        if False:
            print('Hello World!')
        ' Create the application for the test case.\n        '
        cls._app = QtWidgets.QApplication.instance()
        if cls._app is None:
            cls._app = QtWidgets.QApplication([])
        cls._app.setQuitOnLastWindowClosed(False)

    @classmethod
    def tearDownClass(cls):
        if False:
            return 10
        ' Exit the application.\n        '
        QtWidgets.QApplication.quit()

    def assert_text_equal(self, cursor, text):
        if False:
            return 10
        cursor.select(QtGui.QTextCursor.Document)
        selection = cursor.selectedText()
        self.assertEqual(selection, text)

    def test_special_characters(self):
        if False:
            while True:
                i = 10
        ' Are special characters displayed correctly?\n        '
        w = ConsoleWidget()
        cursor = w._get_prompt_cursor()
        test_inputs = ['xyz\x08\x08=\n', 'foo\x08\nbar\n', 'foo\x08\nbar\r\n', 'abc\rxyz\x08\x08=']
        expected_outputs = ['x=z\u2029', 'foo\u2029bar\u2029', 'foo\u2029bar\u2029', 'x=z']
        for (i, text) in enumerate(test_inputs):
            w._insert_plain_text(cursor, text)
            self.assert_text_equal(cursor, expected_outputs[i])
            cursor.insertText('')

    def test_erase_in_line(self):
        if False:
            for i in range(10):
                print('nop')
        ' Do control sequences for clearing the line work?\n        '
        w = ConsoleWidget()
        cursor = w._get_prompt_cursor()
        test_inputs = ['Hello\x1b[1KBye', 'Hello\x1b[0KBye', 'Hello\r\x1b[0KBye', 'Hello\r\x1b[1KBye', 'Hello\r\x1b[2KBye', 'Hello\x1b[2K\rBye']
        expected_outputs = ['     Bye', 'HelloBye', 'Bye', 'Byelo', 'Bye', 'Bye']
        for (i, text) in enumerate(test_inputs):
            w._insert_plain_text(cursor, text)
            self.assert_text_equal(cursor, expected_outputs[i])
            cursor.insertText('')

    def test_link_handling(self):
        if False:
            for i in range(10):
                print('nop')
        noButton = QtCore.Qt.NoButton
        noButtons = QtCore.Qt.NoButton
        noModifiers = QtCore.Qt.NoModifier
        MouseMove = QtCore.QEvent.MouseMove
        QMouseEvent = QtGui.QMouseEvent
        w = ConsoleWidget()
        cursor = w._get_prompt_cursor()
        w._insert_html(cursor, '<a href="http://python.org">written in</a>')
        obj = w._control
        tip = QtWidgets.QToolTip
        self.assertEqual(tip.text(), '')
        elsewhereEvent = QMouseEvent(MouseMove, QtCore.QPointF(50, 50), noButton, noButtons, noModifiers)
        w.eventFilter(obj, elsewhereEvent)
        self.assertEqual(tip.isVisible(), False)
        self.assertEqual(tip.text(), '')
        overTextEvent = QMouseEvent(MouseMove, QtCore.QPointF(1, 5), noButton, noButtons, noModifiers)
        w.eventFilter(obj, overTextEvent)
        self.assertEqual(tip.isVisible(), True)
        self.assertEqual(tip.text(), 'http://python.org')
        stillOverTextEvent = QMouseEvent(MouseMove, QtCore.QPointF(1, 5), noButton, noButtons, noModifiers)
        w.eventFilter(obj, stillOverTextEvent)
        self.assertEqual(tip.isVisible(), True)
        self.assertEqual(tip.text(), 'http://python.org')

    def test_width_height(self):
        if False:
            print('Hello World!')
        w = ConsoleWidget()
        self.assertEqual(w.width(), QtWidgets.QWidget.width(w))
        self.assertEqual(w.height(), QtWidgets.QWidget.height(w))

    def test_prompt_cursors(self):
        if False:
            for i in range(10):
                print('nop')
        'Test the cursors that keep track of where the prompt begins and\n        ends'
        w = ConsoleWidget()
        w._prompt = 'prompt>'
        doc = w._control.document()
        doc.setMaximumBlockCount(10)
        for _ in range(9):
            w._append_plain_text('line\n')
        w._show_prompt()
        self.assertEqual(doc.blockCount(), 10)
        self.assertEqual(w._prompt_pos, w._get_end_pos())
        self.assertEqual(w._append_before_prompt_pos, w._prompt_pos - len(w._prompt))
        w._append_plain_text('line\n')
        self.assertEqual(w._prompt_pos, w._get_end_pos() - len('line\n'))
        self.assertEqual(w._append_before_prompt_pos, w._prompt_pos - len(w._prompt))
        w._show_prompt()
        self.assertEqual(w._prompt_pos, w._get_end_pos())
        self.assertEqual(w._append_before_prompt_pos, w._prompt_pos - len(w._prompt))
        w._append_plain_text('line', before_prompt=True)
        self.assertEqual(w._prompt_pos, w._get_end_pos())
        self.assertEqual(w._append_before_prompt_pos, w._prompt_pos - len(w._prompt))

    def test_select_all(self):
        if False:
            print('Hello World!')
        w = ConsoleWidget()
        w._append_plain_text('Header\n')
        w._prompt = 'prompt>'
        w._show_prompt()
        control = w._control
        app = QtWidgets.QApplication.instance()
        cursor = w._get_cursor()
        w._insert_plain_text_into_buffer(cursor, 'if:\n    pass')
        cursor.clearSelection()
        control.setTextCursor(cursor)
        w.select_all_smart()
        QTest.keyClick(control, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier)
        copied = app.clipboard().text()
        self.assertEqual(copied, 'if:\n>     pass')
        w.select_all_smart()
        QTest.keyClick(control, QtCore.Qt.Key_C, QtCore.Qt.ControlModifier)
        copied = app.clipboard().text()
        self.assertEqual(copied, 'Header\nprompt>if:\n>     pass')

    @pytest.mark.skipif(sys.platform == 'darwin', reason='Fails on macOS')
    def test_keypresses(self):
        if False:
            return 10
        'Test the event handling code for keypresses.'
        w = ConsoleWidget()
        w._append_plain_text('Header\n')
        w._prompt = 'prompt>'
        w._show_prompt()
        app = QtWidgets.QApplication.instance()
        control = w._control
        w._set_input_buffer('test input')
        self.assertEqual(w._get_input_buffer(), 'test input')
        w._set_input_buffer('test input')
        c = control.textCursor()
        c.setPosition(c.position() - 3)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_K, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), 'test in')
        w._set_input_buffer('test input ')
        app.clipboard().setText('pasted text')
        QTest.keyClick(control, QtCore.Qt.Key_V, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), 'test input pasted text')
        self.assertEqual(control.document().blockCount(), 2)
        w._set_input_buffer('test input ')
        app.clipboard().setText('    pasted text')
        QTest.keyClick(control, QtCore.Qt.Key_V, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), 'test input pasted text')
        self.assertEqual(control.document().blockCount(), 2)
        w._set_input_buffer('test input ')
        app.clipboard().setText('line1\nline2\nline3')
        QTest.keyClick(control, QtCore.Qt.Key_V, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), 'test input line1\nline2\nline3')
        self.assertEqual(control.document().blockCount(), 4)
        self.assertEqual(control.document().findBlockByNumber(1).text(), 'prompt>test input line1')
        self.assertEqual(control.document().findBlockByNumber(2).text(), '> line2')
        self.assertEqual(control.document().findBlockByNumber(3).text(), '> line3')
        w._set_input_buffer('    ')
        app.clipboard().setText('    If 1:\n        pass')
        QTest.keyClick(control, QtCore.Qt.Key_V, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), '    If 1:\n        pass')
        w._set_input_buffer("foo = ['foo', 'foo', 'foo',    \n       'bar', 'bar', 'bar']")
        QTest.keyClick(control, QtCore.Qt.Key_Backspace, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', 'foo',    \n       'bar', 'bar', '")
        QTest.keyClick(control, QtCore.Qt.Key_Backspace, QtCore.Qt.ControlModifier)
        QTest.keyClick(control, QtCore.Qt.Key_Backspace, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', 'foo',    \n       '")
        QTest.keyClick(control, QtCore.Qt.Key_Backspace, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', 'foo',    \n")
        QTest.keyClick(control, QtCore.Qt.Key_Backspace, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', 'foo',")
        w._set_input_buffer("foo = ['foo', 'foo', 'foo',    \n       'bar', 'bar', 'bar']")
        c = control.textCursor()
        c.setPosition(35)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_Delete, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', ',    \n       'bar', 'bar', 'bar']")
        QTest.keyClick(control, QtCore.Qt.Key_Delete, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', \n       'bar', 'bar', 'bar']")
        QTest.keyClick(control, QtCore.Qt.Key_Delete, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', 'bar', 'bar', 'bar']")
        w._set_input_buffer("foo = ['foo', 'foo', 'foo',    \n       'bar', 'bar', 'bar']")
        c = control.textCursor()
        c.setPosition(48)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_Delete, QtCore.Qt.ControlModifier)
        self.assertEqual(w._get_input_buffer(), "foo = ['foo', 'foo', 'foo',    \n'bar', 'bar', 'bar']")
        w._set_input_buffer('line 1\nline 2\nline 3')
        c = control.textCursor()
        c.setPosition(20)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_Right)
        self.assertEqual(control.textCursor().position(), 23)
        QTest.keyClick(control, QtCore.Qt.Key_Left)
        self.assertEqual(control.textCursor().position(), 20)

    def test_indent(self):
        if False:
            while True:
                i = 10
        'Test the event handling code for indent/dedent keypresses .'
        w = ConsoleWidget()
        w._append_plain_text('Header\n')
        w._prompt = 'prompt>'
        w._show_prompt()
        control = w._control
        w._set_input_buffer('')
        c = control.textCursor()
        pos = c.position()
        w._set_input_buffer('If 1:\n    pass')
        c.setPosition(pos, QtGui.QTextCursor.KeepAnchor)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_Tab)
        self.assertEqual(w._get_input_buffer(), '    If 1:\n        pass')
        w._set_input_buffer('')
        c = control.textCursor()
        pos = c.position()
        w._set_input_buffer(' If 2:\n     pass')
        c.setPosition(pos, QtGui.QTextCursor.KeepAnchor)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_Tab)
        self.assertEqual(w._get_input_buffer(), '    If 2:\n        pass')
        w._set_input_buffer('')
        c = control.textCursor()
        pos = c.position()
        w._set_input_buffer('    If 3:\n        pass')
        c.setPosition(pos, QtGui.QTextCursor.KeepAnchor)
        control.setTextCursor(c)
        QTest.keyClick(control, QtCore.Qt.Key_Backtab)
        self.assertEqual(w._get_input_buffer(), 'If 3:\n    pass')

    def test_complete(self):
        if False:
            return 10

        class TestKernelClient(object):

            def is_complete(self, source):
                if False:
                    i = 10
                    return i + 15
                calls.append(source)
                return msg_id
        w = ConsoleWidget()
        cursor = w._get_prompt_cursor()
        w._execute = lambda *args: calls.append(args)
        w.kernel_client = TestKernelClient()
        msg_id = object()
        calls = []
        w.execute('thing', interactive=True)
        self.assertEqual(calls, ['thing'])
        calls = []
        w._handle_is_complete_reply(dict(parent_header=dict(msg_id=msg_id), content=dict(status='incomplete', indent='!!!')))
        self.assert_text_equal(cursor, 'thing\u2029> !!!')
        self.assertEqual(calls, [])
        msg_id = object()
        w.execute('else', interactive=True)
        self.assertEqual(calls, ['else'])
        calls = []
        w._handle_is_complete_reply(dict(parent_header=dict(msg_id=msg_id), content=dict(status='complete', indent='###')))
        self.assertEqual(calls, [('else', False)])
        calls = []
        self.assert_text_equal(cursor, 'thing\u2029> !!!else\u2029')
        msg_id = object()
        w.execute('done', interactive=True)
        self.assertEqual(calls, ['done'])
        calls = []
        self.assert_text_equal(cursor, 'thing\u2029> !!!else\u2029')
        w._trigger_is_complete_callback()
        self.assert_text_equal(cursor, 'thing\u2029> !!!else\u2029\u2029> ')
        w._handle_is_complete_reply(dict(parent_header=dict(msg_id=msg_id), content=dict(status='complete', indent='###')))
        self.assertEqual(calls, [])

    def test_complete_python(self):
        if False:
            print('Hello World!')
        'Test that is_complete is working correctly for Python.'

        class TestIPyKernelClient(object):

            def is_complete(self, source):
                if False:
                    return 10
                tm = TransformerManager()
                check_complete = tm.check_complete(source)
                responses.append(check_complete)
        responses = []
        w = ConsoleWidget()
        w._append_plain_text('Header\n')
        w._prompt = 'prompt>'
        w._show_prompt()
        w.kernel_client = TestIPyKernelClient()
        code = '\n'.join(['if True:', '    a = 1'])
        w._set_input_buffer(code)
        w.execute(interactive=True)
        assert responses == [('incomplete', 4)]
        responses = []
        code = '\n'.join(['if True:', '    a = 1\n\n'])
        w._set_input_buffer(code)
        w.execute(interactive=True)
        assert responses == [('complete', None)]