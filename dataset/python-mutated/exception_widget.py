import sys, re, traceback, threading
from .. import exceptionHandling as exceptionHandling
from ..Qt import QtWidgets, QtCore
from ..functions import SignalBlock
from .stackwidget import StackWidget

class ExceptionHandlerWidget(QtWidgets.QGroupBox):
    sigStackItemClicked = QtCore.Signal(object, object)
    sigStackItemDblClicked = QtCore.Signal(object, object)
    _threadException = QtCore.Signal(object)

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._setupUi()
        self.filterString = ''
        self._inSystrace = False
        self._threadException.connect(self._threadExceptionHandler)

    def _setupUi(self):
        if False:
            for i in range(10):
                print('nop')
        self.setTitle('Exception Handling')
        self.layout = QtWidgets.QGridLayout(self)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setHorizontalSpacing(2)
        self.layout.setVerticalSpacing(0)
        self.clearExceptionBtn = QtWidgets.QPushButton('Clear Stack', self)
        self.clearExceptionBtn.setEnabled(False)
        self.layout.addWidget(self.clearExceptionBtn, 0, 6, 1, 1)
        self.catchAllExceptionsBtn = QtWidgets.QPushButton('Show All Exceptions', self)
        self.catchAllExceptionsBtn.setCheckable(True)
        self.layout.addWidget(self.catchAllExceptionsBtn, 0, 1, 1, 1)
        self.catchNextExceptionBtn = QtWidgets.QPushButton('Show Next Exception', self)
        self.catchNextExceptionBtn.setCheckable(True)
        self.layout.addWidget(self.catchNextExceptionBtn, 0, 0, 1, 1)
        self.onlyUncaughtCheck = QtWidgets.QCheckBox('Only Uncaught Exceptions', self)
        self.onlyUncaughtCheck.setChecked(True)
        self.layout.addWidget(self.onlyUncaughtCheck, 0, 4, 1, 1)
        self.stackTree = StackWidget(self)
        self.layout.addWidget(self.stackTree, 2, 0, 1, 7)
        self.runSelectedFrameCheck = QtWidgets.QCheckBox('Run commands in selected stack frame', self)
        self.runSelectedFrameCheck.setChecked(True)
        self.layout.addWidget(self.runSelectedFrameCheck, 3, 0, 1, 7)
        spacerItem = QtWidgets.QSpacerItem(40, 20, QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Minimum)
        self.layout.addItem(spacerItem, 0, 5, 1, 1)
        self.filterLabel = QtWidgets.QLabel('Filter (regex):', self)
        self.layout.addWidget(self.filterLabel, 0, 2, 1, 1)
        self.filterText = QtWidgets.QLineEdit(self)
        self.layout.addWidget(self.filterText, 0, 3, 1, 1)
        self.catchAllExceptionsBtn.toggled.connect(self.catchAllExceptions)
        self.catchNextExceptionBtn.toggled.connect(self.catchNextException)
        self.clearExceptionBtn.clicked.connect(self.clearExceptionClicked)
        self.stackTree.itemClicked.connect(self.stackItemClicked)
        self.stackTree.itemDoubleClicked.connect(self.stackItemDblClicked)
        self.onlyUncaughtCheck.toggled.connect(self.updateSysTrace)
        self.filterText.textChanged.connect(self._filterTextChanged)

    def setStack(self, frame=None):
        if False:
            return 10
        self.clearExceptionBtn.setEnabled(True)
        self.stackTree.setStack(frame)

    def setException(self, exc=None, lastFrame=None):
        if False:
            i = 10
            return i + 15
        self.clearExceptionBtn.setEnabled(True)
        self.stackTree.setException(exc, lastFrame=lastFrame)

    def selectedFrame(self):
        if False:
            i = 10
            return i + 15
        return self.stackTree.selectedFrame()

    def catchAllExceptions(self, catch=True):
        if False:
            return 10
        '\n        If True, the console will catch all unhandled exceptions and display the stack\n        trace. Each exception caught clears the last.\n        '
        with SignalBlock(self.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
            self.catchAllExceptionsBtn.setChecked(catch)
        if catch:
            with SignalBlock(self.catchNextExceptionBtn.toggled, self.catchNextException):
                self.catchNextExceptionBtn.setChecked(False)
            self.enableExceptionHandling()
        else:
            self.disableExceptionHandling()

    def catchNextException(self, catch=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        If True, the console will catch the next unhandled exception and display the stack\n        trace.\n        '
        with SignalBlock(self.catchNextExceptionBtn.toggled, self.catchNextException):
            self.catchNextExceptionBtn.setChecked(catch)
        if catch:
            with SignalBlock(self.catchAllExceptionsBtn.toggled, self.catchAllExceptions):
                self.catchAllExceptionsBtn.setChecked(False)
            self.enableExceptionHandling()
        else:
            self.disableExceptionHandling()

    def enableExceptionHandling(self):
        if False:
            for i in range(10):
                print('nop')
        exceptionHandling.registerCallback(self.exceptionHandler)
        self.updateSysTrace()

    def disableExceptionHandling(self):
        if False:
            i = 10
            return i + 15
        exceptionHandling.unregisterCallback(self.exceptionHandler)
        self.updateSysTrace()

    def clearExceptionClicked(self):
        if False:
            print('Hello World!')
        self.stackTree.clear()
        self.clearExceptionBtn.setEnabled(False)

    def updateSysTrace(self):
        if False:
            while True:
                i = 10
        if not self.catchNextExceptionBtn.isChecked() and (not self.catchAllExceptionsBtn.isChecked()):
            if sys.gettrace() == self.systrace:
                self._disableSysTrace()
            return
        if self.onlyUncaughtCheck.isChecked():
            if sys.gettrace() == self.systrace:
                self._disableSysTrace()
        elif sys.gettrace() not in (None, self.systrace):
            self.onlyUncaughtCheck.setChecked(False)
            raise Exception('sys.settrace is in use (are you using another debugger?); cannot monitor for caught exceptions.')
        else:
            self._enableSysTrace()

    def _enableSysTrace(self):
        if False:
            while True:
                i = 10
        sys.settrace(self.systrace)
        threading.settrace(self.systrace)
        if hasattr(threading, 'settrace_all_threads'):
            threading.settrace_all_threads(self.systrace)

    def _disableSysTrace(self):
        if False:
            i = 10
            return i + 15
        sys.settrace(None)
        threading.settrace(None)
        if hasattr(threading, 'settrace_all_threads'):
            threading.settrace_all_threads(None)

    def exceptionHandler(self, excInfo, lastFrame=None):
        if False:
            return 10
        if isinstance(excInfo, Exception):
            exc = excInfo
        else:
            exc = excInfo.exc_value
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if not isGuiThread:
            self._threadException.emit((excInfo, lastFrame))
            return
        if self.catchNextExceptionBtn.isChecked():
            self.catchNextExceptionBtn.setChecked(False)
        elif not self.catchAllExceptionsBtn.isChecked():
            return
        self.setException(exc, lastFrame=lastFrame)

    def _threadExceptionHandler(self, args):
        if False:
            print('Hello World!')
        self.exceptionHandler(*args)

    def systrace(self, frame, event, arg):
        if False:
            while True:
                i = 10
        if event != 'exception':
            return self.systrace
        if self._inSystrace:
            return self.systrace
        self._inSystrace = True
        try:
            if self.checkException(*arg):
                self.exceptionHandler(arg[1], lastFrame=frame)
        except Exception as exc:
            print('Exception in systrace:')
            traceback.print_exc()
        finally:
            self.inSystrace = False
        return self.systrace

    def checkException(self, excType, exc, tb):
        if False:
            return 10
        filename = tb.tb_frame.f_code.co_filename
        function = tb.tb_frame.f_code.co_name
        filterStr = self.filterString
        if filterStr != '':
            if isinstance(exc, Exception):
                msg = traceback.format_exception_only(type(exc), exc)
            elif isinstance(exc, str):
                msg = exc
            else:
                msg = repr(exc)
            match = re.search(filterStr, '%s:%s:%s' % (filename, function, msg))
            return match is not None
        if excType is GeneratorExit or excType is StopIteration:
            return False
        if excType is AttributeError:
            if filename.endswith('numpy/core/fromnumeric.py') and function in ('all', '_wrapit', 'transpose', 'sum'):
                return False
            if filename.endswith('numpy/core/arrayprint.py') and function in '_array2string':
                return False
            if filename.endswith('MetaArray.py') and function == '__getattr__':
                for name in ('__array_interface__', '__array_struct__', '__array__'):
                    if name in exc:
                        return False
            if filename.endswith('flowchart/eq.py'):
                return False
        if excType is TypeError:
            if filename.endswith('numpy/lib/function_base.py') and function == 'iterable':
                return False
        return True

    def stackItemClicked(self, item):
        if False:
            return 10
        self.sigStackItemClicked.emit(self, item)

    def stackItemDblClicked(self, item):
        if False:
            while True:
                i = 10
        self.sigStackItemDblClicked.emit(self, item)

    def _filterTextChanged(self, value):
        if False:
            while True:
                i = 10
        self.filterString = str(value)