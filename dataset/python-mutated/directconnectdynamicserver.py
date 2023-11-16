import sys
from PyQt5.QtCore import pyqtProperty, pyqtSignal, pyqtSlot, QCoreApplication, QObject, QTimer, QUrl
from PyQt5.QtRemoteObjects import QRemoteObjectHost

class SimpleSwitch(QObject):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._currState = False
        self._stateChangeTimer = QTimer(self)
        self._stateChangeTimer.timeout.connect(self._timeout)
        self._stateChangeTimer.start(2000)
        print('Source node started')

    @pyqtSlot()
    def pushCurrState(self, currState):
        if False:
            return 10
        pass

    def _get_currState(self):
        if False:
            print('Hello World!')
        return self._currState

    def _set_currState(self, value):
        if False:
            for i in range(10):
                print('nop')
        if self._currState != value:
            self._currState = value
            self.currStateChanged.emit(value)
    currStateChanged = pyqtSignal(bool)
    currState = pyqtProperty(bool, fget=_get_currState, fset=_set_currState, notify=currStateChanged)

    @pyqtSlot(bool)
    def server_slot(self, clientState):
        if False:
            for i in range(10):
                print('nop')
        print('Replica state is', clientState)

    def _timeout(self):
        if False:
            print('Hello World!')
        self.currState = not self.currState
        print('Source state is', self.currState)
if __name__ == '__main__':
    app = QCoreApplication(sys.argv)
    srcSwitch = SimpleSwitch()
    srcNode = QRemoteObjectHost(QUrl('local:replica'))
    srcNode.enableRemoting(srcSwitch, 'SimpleSwitch')
    sys.exit(app.exec_())