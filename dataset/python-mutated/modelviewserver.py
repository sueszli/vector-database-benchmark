import sys
try:
    from PyQt5.QtCore import pyqtSlot, QLoggingCategory, QModelIndex, QObject, Qt, QTimer, QUrl
    from PyQt5.QtGui import QColor, QStandardItem, QStandardItemModel
    from PyQt5.QtRemoteObjects import QRemoteObjectHost, QRemoteObjectRegistryHost
    from PyQt5.QtWidgets import QApplication, QTreeView
except ImportError:
    from PySide2.QtCore import Slot as pyqtSlot, QModelIndex, QObject, Qt, QTimer, QUrl
    from PySide2.QtGui import QColor, QStandardItem, QStandardItemModel
    from PySide2.QtRemoteObjects import QRemoteObjectHost, QRemoteObjectRegistryHost
    from PySide2.QtWidgets import QApplication, QTreeView

class TimerHandler(QObject):

    def __init__(self, model, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self._model = model

    @pyqtSlot()
    def changeData(self):
        if False:
            print('Hello World!')
        for i in range(10, 50):
            self._model.setData(self._model.index(i, 1), QColor(Qt.blue), Qt.BackgroundRole)

    @pyqtSlot()
    def insertData(self):
        if False:
            return 10
        self._model.insertRows(2, 9)
        for i in range(2, 11):
            self._model.setData(self._model.index(i, 1), QColor(Qt.green), Qt.BackgroundRole)
            self._model.setData(self._model.index(i, 1), 'InsertedRow', Qt.DisplayRole)

    @pyqtSlot()
    def removeData(self):
        if False:
            return 10
        self._model.removeRows(2, 4)

    @pyqtSlot()
    def changeFlags(self):
        if False:
            return 10
        item = self._model.item(0, 0)
        item.setEnabled(False)
        item = item.child(0, 0)
        item.setFlags(item.flags() & Qt.ItemIsSelectable)

    @pyqtSlot()
    def moveData(self):
        if False:
            i = 10
            return i + 15
        self._model.moveRows(QModelIndex(), 2, 4, QModelIndex(), 10)

def addChild(numChildren, nestingLevel):
    if False:
        print('Hello World!')
    result = []
    if nestingLevel == 0:
        return result
    for i in range(numChildren):
        child = QStandardItem('Child num {}, nesting level {}'.format(i + 1, nestingLevel))
        if i == 0:
            child.appendRow(addChild(numChildren, nestingLevel - 1))
        result.append(child)
    return result
if __name__ == '__main__':
    try:
        QLoggingCategory.setFilterRules('qt.remoteobjects.debug=false\nqt.remoteobjects.warning=false')
    except NameError:
        pass
    app = QApplication(sys.argv)
    sourceModel = QStandardItemModel()
    sourceModel.setHorizontalHeaderLabels(['First Column with spacing', 'Second Column with spacing'])
    for i in range(10000):
        firstItem = QStandardItem('FancyTextNumber {}'.format(i))
        if i == 0:
            firstItem.appendRow(addChild(2, 2))
        secondItem = QStandardItem('FancyRow2TextNumber {}'.format(i))
        if i % 2 == 0:
            firstItem.setBackground(Qt.red)
        sourceModel.invisibleRootItem().appendRow([firstItem, secondItem])
    roleNames = {Qt.DisplayRole: b'_text', Qt.BackgroundRole: b'_color'}
    sourceModel.setItemRoleNames(roleNames)
    roles = [Qt.DisplayRole, Qt.BackgroundRole]
    node = QRemoteObjectRegistryHost(QUrl('local:registry'))
    node2 = QRemoteObjectHost(QUrl('local:replica'), QUrl('local:registry'))
    node2.enableRemoting(sourceModel, 'RemoteModel', roles)
    view = QTreeView()
    view.setWindowTitle('SourceView')
    view.setModel(sourceModel)
    view.show()
    handler = TimerHandler(sourceModel)
    QTimer.singleShot(5000, handler.changeData)
    QTimer.singleShot(10000, handler.insertData)
    QTimer.singleShot(11000, handler.changeFlags)
    QTimer.singleShot(12000, handler.removeData)
    QTimer.singleShot(13000, handler.moveData)
    sys.exit(app.exec_())