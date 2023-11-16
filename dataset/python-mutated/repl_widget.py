import code, sys, traceback
from ..Qt import QtWidgets, QtGui, QtCore
from ..functions import mkBrush
from .CmdInput import CmdInput

class ReplWidget(QtWidgets.QWidget):
    sigCommandEntered = QtCore.Signal(object, object)
    sigCommandRaisedException = QtCore.Signal(object, object)

    def __init__(self, globals, locals, parent=None):
        if False:
            for i in range(10):
                print('nop')
        self.globals = globals
        self.locals = locals
        self._lastCommandRow = None
        self._commandBuffer = []
        self.stdoutInterceptor = StdoutInterceptor(self.write)
        self.ps1 = '>>> '
        self.ps2 = '... '
        QtWidgets.QWidget.__init__(self, parent=parent)
        self._setupUi()
        isDark = self.output.palette().color(QtGui.QPalette.ColorRole.Base).value() < 128
        outputBlockFormat = QtGui.QTextBlockFormat()
        outputFirstLineBlockFormat = QtGui.QTextBlockFormat(outputBlockFormat)
        outputFirstLineBlockFormat.setTopMargin(5)
        outputCharFormat = QtGui.QTextCharFormat()
        outputCharFormat.setFontWeight(QtGui.QFont.Weight.Normal)
        cmdBlockFormat = QtGui.QTextBlockFormat()
        cmdBlockFormat.setBackground(mkBrush('#335' if isDark else '#CCF'))
        cmdCharFormat = QtGui.QTextCharFormat()
        cmdCharFormat.setFontWeight(QtGui.QFont.Weight.Bold)
        self.textStyles = {'command': (cmdCharFormat, cmdBlockFormat), 'output': (outputCharFormat, outputBlockFormat), 'output_first_line': (outputCharFormat, outputFirstLineBlockFormat)}
        self.input.ps1 = self.ps1
        self.input.ps2 = self.ps2

    def _setupUi(self):
        if False:
            while True:
                i = 10
        self.layout = QtWidgets.QVBoxLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(0)
        self.setLayout(self.layout)
        self.output = QtWidgets.QTextEdit(self)
        font = QtGui.QFont()
        font.setFamily('Courier New')
        font.setStyleStrategy(QtGui.QFont.StyleStrategy.PreferAntialias)
        self.output.setFont(font)
        self.output.setReadOnly(True)
        self.layout.addWidget(self.output)
        self.inputWidget = QtWidgets.QWidget(self)
        self.layout.addWidget(self.inputWidget)
        self.inputLayout = QtWidgets.QHBoxLayout()
        self.inputWidget.setLayout(self.inputLayout)
        self.inputLayout.setContentsMargins(0, 0, 0, 0)
        self.input = CmdInput(parent=self)
        self.inputLayout.addWidget(self.input)
        self.input.sigExecuteCmd.connect(self.runCmd)

    def runCmd(self, cmd):
        if False:
            while True:
                i = 10
        if '\n' in cmd:
            for line in cmd.split('\n'):
                self.runCmd(line)
            return
        if len(self._commandBuffer) == 0:
            self.write(f'{self.ps1}{cmd}\n', style='command')
        else:
            self.write(f'{self.ps2}{cmd}\n', style='command')
        self.sigCommandEntered.emit(self, cmd)
        self._commandBuffer.append(cmd)
        fullcmd = '\n'.join(self._commandBuffer)
        try:
            cmdCode = code.compile_command(fullcmd)
            self.input.setMultiline(False)
        except Exception:
            self._commandBuffer = []
            self.displayException()
            self.input.setMultiline(False)
        else:
            if cmdCode is None:
                self.input.setMultiline(True)
                return
            self._commandBuffer = []
            try:
                with self.stdoutInterceptor:
                    exec(cmdCode, self.globals(), self.locals())
            except Exception as exc:
                self.displayException()
                self.sigCommandRaisedException.emit(self, exc)
            cursor = self.output.textCursor()
            if cursor.columnNumber() > 0:
                self.write('\n')

    def write(self, strn, style='output', scrollToBottom='auto'):
        if False:
            while True:
                i = 10
        "Write a string into the console.\n\n        If scrollToBottom is 'auto', then the console is automatically scrolled\n        to fit the new text only if it was already at the bottom.\n        "
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            sys.__stdout__.write(strn)
            return
        cursor = self.output.textCursor()
        cursor.movePosition(QtGui.QTextCursor.MoveOperation.End)
        self.output.setTextCursor(cursor)
        sb = self.output.verticalScrollBar()
        scroll = sb.value()
        if scrollToBottom == 'auto':
            atBottom = scroll == sb.maximum()
            scrollToBottom = atBottom
        row = cursor.blockNumber()
        if style == 'command':
            self._lastCommandRow = row
        if style == 'output' and row == self._lastCommandRow + 1:
            (firstLine, endl, strn) = strn.partition('\n')
            self._setTextStyle('output_first_line')
            self.output.insertPlainText(firstLine + endl)
        if len(strn) > 0:
            self._setTextStyle(style)
            self.output.insertPlainText(strn)
            if style != 'output':
                self._setTextStyle('output')
        if scrollToBottom:
            sb.setValue(sb.maximum())
        else:
            sb.setValue(scroll)

    def displayException(self):
        if False:
            i = 10
            return i + 15
        '\n        Display the current exception and stack.\n        '
        tb = traceback.format_exc()
        lines = []
        indent = 4
        prefix = ''
        for l in tb.split('\n'):
            lines.append(' ' * indent + prefix + l)
        self.write('\n'.join(lines))

    def _setTextStyle(self, style):
        if False:
            return 10
        (charFormat, blockFormat) = self.textStyles[style]
        cursor = self.output.textCursor()
        cursor.setBlockFormat(blockFormat)
        self.output.setCurrentCharFormat(charFormat)

class StdoutInterceptor:
    """Used to temporarily redirect writes meant for sys.stdout and sys.stderr to a new location
    """

    def __init__(self, writeFn):
        if False:
            print('Hello World!')
        self._orig_stdout = None
        self._orig_stderr = None
        self.writeFn = writeFn

    def realOutputFiles(self):
        if False:
            for i in range(10):
                print('nop')
        'Return the real sys.stdout and stderr (which are sometimes masked while running commands)\n        '
        return (self._orig_stdout or sys.stdout, self._orig_stderr or sys.stderr)

    def print(self, *args):
        if False:
            for i in range(10):
                print('nop')
        'Print to real stdout (for debugging)\n        '
        self.realOutputFiles()[0].write(' '.join(map(str, args)) + '\n')

    def flush(self):
        if False:
            print('Hello World!')
        pass

    def fileno(self):
        if False:
            i = 10
            return i + 15
        return 1

    def write(self, strn):
        if False:
            for i in range(10):
                print('nop')
        self.writeFn(strn)

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        self._orig_stdout = sys.stdout
        self._orig_stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            print('Hello World!')
        sys.stdout = self._orig_stdout
        sys.stderr = self._orig_stderr
        self._orig_stdout = None
        self._orig_stderr = None