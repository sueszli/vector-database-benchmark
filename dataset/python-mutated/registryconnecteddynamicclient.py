import sys
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QCoreApplication, QObject, QUrl
from PyQt5.QtRemoteObjects import QRemoteObjectNode

class DynamicClient(QObject):
    echoSwitchState = pyqtSignal(bool)

    def __init__(self, replica, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._replica = replica
        self._clientSwitchState = False
        replica.initialized.connect(self.initConnection)

    @pyqtSlot(bool)
    def recSwitchState(self, value):
        if False:
            print('Hello World!')
        self._clientSwitchState = self._replica.property('currState')
        print('Received source state', value, self._clientSwitchState)
        self.echoSwitchState.emit(self._clientSwitchState)

    @pyqtSlot()
    def initConnection(self):
        if False:
            print('Hello World!')
        self._replica.currStateChanged.connect(self.recSwitchState)
        self.echoSwitchState.connect(self._replica.server_slot)
if __name__ == '__main__':
    app = QCoreApplication(sys.argv)
    repNode = QRemoteObjectNode(QUrl('local:registry'))
    replica = repNode.acquireDynamic('SimpleSwitch')
    rswitch = DynamicClient(replica)
    sys.exit(app.exec_())