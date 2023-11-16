"""Debugging console."""
import sys
import code
from typing import MutableSequence
from qutebrowser.qt.core import pyqtSignal, pyqtSlot, Qt
from qutebrowser.qt.widgets import QTextEdit, QWidget, QVBoxLayout, QApplication
from qutebrowser.qt.gui import QTextCursor
from qutebrowser.config import stylesheet
from qutebrowser.misc import cmdhistory, miscwidgets
from qutebrowser.utils import utils, objreg
console_widget = None

class ConsoleLineEdit(miscwidgets.CommandLineEdit):
    """A QLineEdit which executes entered code and provides a history.

    Attributes:
        _history: The command history of executed commands.

    Signals:
        execute: Emitted when a commandline should be executed.
    """
    execute = pyqtSignal(str)

    def __init__(self, _namespace, parent):
        if False:
            return 10
        'Constructor.\n\n        Args:\n            _namespace: The local namespace of the interpreter.\n        '
        super().__init__(parent=parent)
        self._history = cmdhistory.History(parent=self)
        self.returnPressed.connect(self.on_return_pressed)

    @pyqtSlot()
    def on_return_pressed(self):
        if False:
            for i in range(10):
                print('nop')
        'Execute the line of code which was entered.'
        self._history.stop()
        text = self.text()
        if text:
            self._history.append(text)
        self.execute.emit(text)
        self.setText('')

    def history_prev(self):
        if False:
            return 10
        'Go back in the history.'
        try:
            if not self._history.is_browsing():
                item = self._history.start(self.text().strip())
            else:
                item = self._history.previtem()
        except (cmdhistory.HistoryEmptyError, cmdhistory.HistoryEndReachedError):
            return
        self.setText(item)

    def history_next(self):
        if False:
            while True:
                i = 10
        'Go forward in the history.'
        if not self._history.is_browsing():
            return
        try:
            item = self._history.nextitem()
        except cmdhistory.HistoryEndReachedError:
            return
        self.setText(item)

    def keyPressEvent(self, e):
        if False:
            while True:
                i = 10
        'Override keyPressEvent to handle special keypresses.'
        if e.key() == Qt.Key.Key_Up:
            self.history_prev()
            e.accept()
        elif e.key() == Qt.Key.Key_Down:
            self.history_next()
            e.accept()
        elif e.modifiers() & Qt.KeyboardModifier.ControlModifier and e.key() == Qt.Key.Key_C:
            self.setText('')
            e.accept()
        else:
            super().keyPressEvent(e)

class ConsoleTextEdit(QTextEdit):
    """Custom QTextEdit for console output."""

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self.setAcceptRichText(False)
        self.setReadOnly(True)
        self.setFocusPolicy(Qt.FocusPolicy.ClickFocus)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return utils.get_repr(self)

    def append_text(self, text):
        if False:
            print('Hello World!')
        "Append new text and scroll output to bottom.\n\n        We can't use Qt's way to append stuff because that inserts weird\n        newlines.\n        "
        self.moveCursor(QTextCursor.MoveOperation.End)
        self.insertPlainText(text)
        scrollbar = self.verticalScrollBar()
        assert scrollbar is not None
        scrollbar.setValue(scrollbar.maximum())

class ConsoleWidget(QWidget):
    """A widget with an interactive Python console.

    Attributes:
        _lineedit: The line edit in the console.
        _output: The output widget in the console.
        _vbox: The layout which contains everything.
        _more: A flag which is set when more input is expected.
        _buffer: The buffer for multi-line commands.
        _interpreter: The InteractiveInterpreter to execute code with.
    """
    STYLESHEET = '\n        ConsoleWidget > ConsoleTextEdit, ConsoleWidget > ConsoleLineEdit {\n            font: {{ conf.fonts.debug_console }};\n        }\n    '

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        if not hasattr(sys, 'ps1'):
            sys.ps1 = '>>> '
        if not hasattr(sys, 'ps2'):
            sys.ps2 = '... '
        namespace = {'__name__': '__console__', '__doc__': None, 'q_app': QApplication.instance(), 'self': parent, 'objreg': objreg}
        self._more = False
        self._buffer: MutableSequence[str] = []
        self._lineedit = ConsoleLineEdit(namespace, self)
        self._lineedit.execute.connect(self.push)
        self._output = ConsoleTextEdit()
        self.write(self._curprompt())
        self._vbox = QVBoxLayout()
        self._vbox.setSpacing(0)
        self._vbox.addWidget(self._output)
        self._vbox.addWidget(self._lineedit)
        stylesheet.set_register(self)
        self.setLayout(self._vbox)
        self._lineedit.setFocus()
        self._interpreter = code.InteractiveInterpreter(namespace)

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return utils.get_repr(self, visible=self.isVisible())

    def write(self, line):
        if False:
            for i in range(10):
                print('nop')
        'Write a line of text (without added newline) to the output.'
        self._output.append_text(line)

    @pyqtSlot(str)
    def push(self, line):
        if False:
            return 10
        'Push a line to the interpreter.'
        self._buffer.append(line)
        source = '\n'.join(self._buffer)
        self.write(line + '\n')
        with utils.fake_io(self.write), utils.disabled_excepthook():
            self._more = self._interpreter.runsource(source, '<console>')
        self.write(self._curprompt())
        if not self._more:
            self._buffer = []

    def _curprompt(self):
        if False:
            print('Hello World!')
        'Get the prompt which is visible currently.'
        return sys.ps2 if self._more else sys.ps1

def init():
    if False:
        i = 10
        return i + 15
    'Initialize a global console.'
    global console_widget
    console_widget = ConsoleWidget()