from .Qt import QtCore
__all__ = ['ThreadsafeTimer']

class ThreadsafeTimer(QtCore.QObject):
    """
    Thread-safe replacement for QTimer.
    """
    timeout = QtCore.Signal()
    sigTimerStopRequested = QtCore.Signal()
    sigTimerStartRequested = QtCore.Signal(object)

    def __init__(self):
        if False:
            while True:
                i = 10
        QtCore.QObject.__init__(self)
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerFinished)
        self.timer.moveToThread(QtCore.QCoreApplication.instance().thread())
        self.moveToThread(QtCore.QCoreApplication.instance().thread())
        self.sigTimerStopRequested.connect(self.stop, QtCore.Qt.ConnectionType.QueuedConnection)
        self.sigTimerStartRequested.connect(self.start, QtCore.Qt.ConnectionType.QueuedConnection)

    def start(self, timeout):
        if False:
            return 10
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.timer.start(int(timeout))
        else:
            self.sigTimerStartRequested.emit(timeout)

    def stop(self):
        if False:
            i = 10
            return i + 15
        isGuiThread = QtCore.QThread.currentThread() == QtCore.QCoreApplication.instance().thread()
        if isGuiThread:
            self.timer.stop()
        else:
            self.sigTimerStopRequested.emit()

    def timerFinished(self):
        if False:
            for i in range(10):
                print('nop')
        self.timeout.emit()