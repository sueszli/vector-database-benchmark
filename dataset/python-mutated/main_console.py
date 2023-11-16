import code
import re
import os
from ryven.gui.code_editor.CodeEditorWidget import CodeEditorWidget
from qtpy.QtWidgets import QWidget, QLineEdit, QGridLayout, QPlainTextEdit, QLabel, QPushButton, QGroupBox, QVBoxLayout, QHBoxLayout
from qtpy.QtCore import Signal, QEvent, Qt
from qtpy.QtGui import QTextCharFormat, QBrush, QColor, QFont, QFontMetrics

class MainConsole(QWidget):
    """
    Interpreter with interactive console.
    All console output will be redirected to this command line.
    It provides normal REPL functionality and additional access to editor components
    such as the session object and nodes (by right-click on them).
    The input field below can also expand to a text edit to take whole code blocks.
    """
    instance = None

    def __init__(self, window_theme, history: int=100, blockcount: int=5000):
        if False:
            i = 10
            return i + 15
        super(MainConsole, self).__init__()
        self.session = None
        self.window_theme = window_theme
        self.init_ui(history, blockcount)

    def init_ui(self, history, blockcount):
        if False:
            return 10
        self.content_layout = QGridLayout(self)
        self.content_layout.setContentsMargins(0, 0, 0, 0)
        self.input_layout = QGridLayout()
        self.input_layout.setContentsMargins(0, 0, 0, 0)
        self.input_layout.setSpacing(0)
        self.out_display = ConsoleDisplay(blockcount)
        self.content_layout.addWidget(self.out_display, 1, 0, 1, 2)
        self.inpfmt = self.out_display.currentCharFormat()
        self.inpfmt.setForeground(QBrush(QColor(self.window_theme.colors['primaryColor'])))
        self.outfmt = QTextCharFormat(self.inpfmt)
        self.outfmt.setForeground(QBrush(QColor(self.window_theme.colors['secondaryTextColor'])))
        self.errfmt = QTextCharFormat(self.inpfmt)
        self.errfmt.setForeground(QBrush(QColor(self.window_theme.colors['danger'])))
        self.prompt_label = QLabel('> ', self)
        self.prompt_label.setFixedWidth(15)
        self.input_layout.addWidget(self.prompt_label, 0, 0, 1, 1)
        self.prompt_label.hide()
        self.inptextedit = ConsoleInputTextEdit(self.window_theme)
        self.input_layout.addWidget(self.inptextedit, 1, 0, 2, 2)
        self.inptextedit.hide()
        self.inpedit = ConsoleInputLineEdit(self.inptextedit, max_history=history)
        self.inpedit.returned.connect(self.push)
        self.input_layout.addWidget(self.inpedit, 0, 1, 1, 1)
        self.content_layout.addLayout(self.input_layout, 2, 0, 1, 2)
        self.interp = None
        self.reset_interpreter()
        self.buffer = []
        self.num_added_object_contexts = 0

    def setprompt(self, text: str):
        if False:
            i = 10
            return i + 15
        ...

    def add_obj_context(self, context_obj):
        if False:
            print('Hello World!')
        'adds an object to the current context by initializing a new interpreter with a new context'
        old_context = {} if self.interp is None else self.interp.locals
        name = 'obj' + (str(self.num_added_object_contexts + 1) if self.num_added_object_contexts > 0 else '')
        new_context = {name: context_obj}
        context = {**old_context, **new_context}
        self.interp = code.InteractiveConsole(context)
        print('added as ' + name)
        self.num_added_object_contexts += 1

    def reset_interpreter(self):
        if False:
            print('Hello World!')
        'Initializes a new plain interpreter'
        session = self.session

        def reset():
            if False:
                i = 10
                return i + 15
            self.reset_interpreter()
        context = {**locals()}
        self.num_added_object_contexts = 0
        self.interp = code.InteractiveConsole(context)

    def push(self, commands: str) -> None:
        if False:
            return 10
        'execute entered command which may span multiple lines when code was pasted'
        if commands == 'clear':
            self.out_display.clear()
        else:
            lines = commands.split('\n')
            for line in lines:
                self.writeoutput(line, self.inpfmt)
                self.buffer.append(line)
            source = '\n'.join(self.buffer)
            more = self.interp.runsource(source, '<console>')
            if more:
                if self.prompt_label.isHidden():
                    self.prompt_label.show()
                leading_space = re.match('\\s*', self.buffer[-1]).group()
                self.inpedit.next_line = leading_space
            else:
                self.prompt_label.hide()
                self.buffer = []

    def write(self, line: str) -> None:
        if False:
            i = 10
            return i + 15
        'capture stdout and print to outdisplay'
        if len(line) != 1 or ord(line[0]) != 10:
            self.writeoutput(line.rstrip())

    def errorwrite(self, line: str) -> None:
        if False:
            for i in range(10):
                print('nop')
        'capture stderr and print to outdisplay'
        self.writeoutput(line, self.errfmt)

    def writeoutput(self, line: str, fmt: QTextCharFormat=None) -> None:
        if False:
            while True:
                i = 10
        'prints to outdisplay'
        if fmt:
            self.out_display.setCurrentCharFormat(fmt)
        self.out_display.appendPlainText(line.rstrip())
        self.out_display.setCurrentCharFormat(self.outfmt)

class ConsoleInputLineEdit(QLineEdit):
    """Input line edit with a history buffer for recalling previous lines."""
    returned = Signal(str)

    def __init__(self, code_text_edit, max_history: int=100):
        if False:
            while True:
                i = 10
        super().__init__()
        self.setObjectName('ConsoleInputLineEdit')
        self.code_text_edit = code_text_edit
        self.code_text_edit.returned.connect(self.code_text_edit_returned)
        self.max_hist = max_history
        self.hist_index = 0
        self.hist_list = []
        self.next_line = ''
        self.prompt_pattern = re.compile('^[>\\.]')

    def event(self, ev: QEvent) -> bool:
        if False:
            i = 10
            return i + 15
        if not isinstance(ev, QEvent):
            return False
        if ev.type() == QEvent.KeyPress:
            if ev.key() == Qt.Key_Tab:
                self.insert(' ' * 4)
                return True
            elif ev.key() == Qt.Key_Backtab:
                ccp = self.cursorPosition()
                tl = self.text()[:ccp]
                tr = self.text()[ccp:]
                ends_with_tab = re.match('(.*)\\s\\s\\s\\s$', tl)
                if ends_with_tab:
                    self.setText(tl[:-4] + tr)
                    self.setCursorPosition(ccp - 4)
                    return True
            elif ev.key() == Qt.Key_Up:
                self.recall(self.hist_index - 1)
                return True
            elif ev.key() == Qt.Key_Down:
                self.recall(self.hist_index + 1)
                return True
            elif ev.key() == Qt.Key_Return:
                if len(self.text()) == 0 and ev.modifiers() == Qt.ControlModifier:
                    self.open_text_edit()
                else:
                    self.returnkey()
                return True
        return super().event(ev)

    def open_text_edit(self):
        if False:
            print('Hello World!')
        'Switch to the text edit for easy multi-line input'
        self.hide()
        self.code_text_edit.show()
        self.code_text_edit.setText(self.text())
        self.code_text_edit.setFocus()

    def code_text_edit_returned(self, s):
        if False:
            i = 10
            return i + 15
        'Close text edit and process input'
        self.code_text_edit.hide()
        self.show()
        self.setText(s.replace('\t', '    '))
        self.setFocus()
        self.returnkey()

    def returnkey(self) -> None:
        if False:
            return 10
        text = self.text()
        for line in text.splitlines():
            self.record(line)
        self.returned.emit(text)
        self.setText(self.next_line)
        self.next_line = ''

    def recall(self, index: int) -> None:
        if False:
            i = 10
            return i + 15
        'select a line from the history list'
        if len(self.hist_list) > 0 and 0 <= index < len(self.hist_list):
            self.setText(self.hist_list[index])
            self.hist_index = index

    def record(self, line: str) -> None:
        if False:
            i = 10
            return i + 15
        'store line in history buffer and update hist_index'
        while len(self.hist_list) >= self.max_hist - 1:
            self.hist_list.pop()
        self.hist_list.append(line)
        if self.hist_index == len(self.hist_list) - 1 or line != self.hist_list[self.hist_index]:
            self.hist_index = len(self.hist_list)

class ConsoleInputTextEdit(CodeEditorWidget):
    """A text edit for parsing multi-line code blocks in the input field"""
    returned = Signal(str)

    def keyPressEvent(self, e) -> None:
        if False:
            return 10
        if e.key() == Qt.Key_Return and e.modifiers() == Qt.ControlModifier:
            self.returned.emit(self.toPlainText())
        else:
            return super().keyPressEvent(e)

class ConsoleDisplay(QPlainTextEdit):
    """The console output text field"""

    def __init__(self, max_block_count):
        if False:
            i = 10
            return i + 15
        super(ConsoleDisplay, self).__init__()
        self.setObjectName('ConsoleDisplay')
        self.setMaximumBlockCount(max_block_count)
        self.setReadOnly(True)
        self.setFont(QFont('Source Code Pro', 12))

class RedirectOutput:
    """Redirects 'write()'-calls to a specified method"""

    def __init__(self, func):
        if False:
            while True:
                i = 10
        self.func = func

    def write(self, line):
        if False:
            for i in range(10):
                print('nop')
        self.func(line)

def init_main_console(window_theme):
    if False:
        for i in range(10):
            print('nop')
    MainConsole.instance = MainConsole(window_theme)
    console_stdout_redirect = RedirectOutput(MainConsole.instance.write)
    console_errout_redirect = RedirectOutput(MainConsole.instance.errorwrite)
    return (console_stdout_redirect, console_errout_redirect)