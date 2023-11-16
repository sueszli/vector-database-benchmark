"""
Using ConsoleWidget to interactively inspect exception backtraces


TODO
 - fix uncaught exceptions in threads (python 3.12)
 - allow using qtconsole
 - provide thread info for stacks
 - add thread browser?
 - add object browser?
    - clicking on a stack frame populates list of locals?
 - optional merged exception stacks

"""
import sys
import queue
import functools
import threading
import pyqtgraph as pg
import pyqtgraph.console
from pyqtgraph.Qt import QtWidgets
from pyqtgraph.debug import threadName

def raiseException():
    if False:
        while True:
            i = 10
    'Raise an exception\n    '
    x = 'inside raiseException()'
    raise Exception(f'Raised an exception {x} in {threadName()}')

def raiseNested():
    if False:
        while True:
            i = 10
    'Raise an exception while handling another\n    '
    x = 'inside raiseNested()'
    try:
        raiseException()
    except Exception:
        raise Exception(f'Raised during exception handling {x} in {threadName()}')

def raiseFrom():
    if False:
        return 10
    'Raise an exception from another\n    '
    x = 'inside raiseFrom()'
    try:
        raiseException()
    except Exception as exc:
        raise Exception(f'Raised-from during exception handling {x} in {threadName()}') from exc

def raiseCaughtException():
    if False:
        return 10
    'Raise and catch an exception\n    '
    x = 'inside raiseCaughtException()'
    try:
        raise Exception(f'Raised an exception {x} in {threadName()}')
    except Exception:
        print(f'Raised and caught exception {x} in {threadName()}  trace: {sys._getframe().f_trace}')

def captureStack():
    if False:
        i = 10
        return i + 15
    'Inspect the curent call stack\n    '
    x = 'inside captureStack()'
    global console
    console.setStack()
    return x
threadRunQueue = queue.Queue()

def threadRunner():
    if False:
        for i in range(10):
            print('nop')
    global threadRunQueue
    sys.settrace(lambda *args: None)
    while True:
        (func, args) = threadRunQueue.get()
        try:
            print(f'running {func} from thread, trace: {sys._getframe().f_trace}')
            func(*args)
        except Exception:
            sys.excepthook(*sys.exc_info())
thread = threading.Thread(target=threadRunner, name='background_thread', daemon=True)
thread.start()

def runInStack(func):
    if False:
        print('Hello World!')
    x = 'inside runInStack(func)'
    runInStack2(func)
    return x

def runInStack2(func):
    if False:
        print('Hello World!')
    x = 'inside runInStack2(func)'
    runInStack3(func)
    return x

def runInStack3(func):
    if False:
        while True:
            i = 10
    x = 'inside runInStack3(func)'
    runInStack4(func)
    return x

def runInStack4(func):
    if False:
        for i in range(10):
            print('nop')
    x = 'inside runInStack4(func)'
    func()
    return x

class SignalEmitter(pg.QtCore.QObject):
    signal = pg.QtCore.Signal(object, object)

    def __init__(self, queued):
        if False:
            while True:
                i = 10
        pg.QtCore.QObject.__init__(self)
        if queued:
            self.signal.connect(self.run, pg.QtCore.Qt.ConnectionType.QueuedConnection)
        else:
            self.signal.connect(self.run)

    def run(self, func, args):
        if False:
            while True:
                i = 10
        func(*args)
signalEmitter = SignalEmitter(queued=False)
queuedSignalEmitter = SignalEmitter(queued=True)

def runFunc(func):
    if False:
        while True:
            i = 10
    if signalCheck.isChecked():
        if queuedSignalCheck.isChecked():
            func = functools.partial(queuedSignalEmitter.signal.emit, runInStack, (func,))
        else:
            func = functools.partial(signalEmitter.signal.emit, runInStack, (func,))
    if threadCheck.isChecked():
        threadRunQueue.put((runInStack, (func,)))
    else:
        runInStack(func)
funcs = [raiseException, raiseNested, raiseFrom, raiseCaughtException, captureStack]
app = pg.mkQApp()
win = pg.QtWidgets.QSplitter(pg.QtCore.Qt.Orientation.Horizontal)
ctrl = QtWidgets.QWidget()
ctrlLayout = QtWidgets.QVBoxLayout()
ctrl.setLayout(ctrlLayout)
win.addWidget(ctrl)
btns = []
for func in funcs:
    btn = QtWidgets.QPushButton(func.__doc__)
    btn.clicked.connect(functools.partial(runFunc, func))
    btns.append(btn)
    ctrlLayout.addWidget(btn)
threadCheck = QtWidgets.QCheckBox('Run in thread')
ctrlLayout.addWidget(threadCheck)
signalCheck = QtWidgets.QCheckBox('Run from Qt signal')
ctrlLayout.addWidget(signalCheck)
queuedSignalCheck = QtWidgets.QCheckBox('Use queued Qt signal')
ctrlLayout.addWidget(queuedSignalCheck)
ctrlLayout.addStretch()
console = pyqtgraph.console.ConsoleWidget(text='\nUse ConsoleWidget to interactively inspect exception tracebacks and call stacks!\n\n- Enable "Show next exception" and the next unhandled exception will be displayed below.\n- Click any of the buttons to the left to generate an exception.\n- When an exception traceback is shown, you can select any of the stack frames and then run commands from that context,\n  allowing you to inspect variables along the stack. (hint: most of the functions called by the buttons to the left \n  have a variable named "x" in their local scope)\n- Note that this is not like a typical debugger--the program is not paused when an exception is caught; we simply keep\n  a reference to the stack frames and continue on.\n- By default, we only catch unhandled exceptions. If you need to inspect a handled exception (one that is caught by\n  a try:except block), then uncheck the "Only handled exceptions" box. Note, however that this incurs a performance \n  penalty and will interfere with other debuggers.\n\n\n')
console.catchNextException()
win.addWidget(console)
win.resize(1400, 800)
win.setSizes([300, 1100])
win.show()
if __name__ == '__main__':
    pg.exec()