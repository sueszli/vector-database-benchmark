from ..Qt import QtCore, QtWidgets
if not hasattr(QtCore, 'Signal'):
    QtCore.Signal = QtCore.pyqtSignal
import weakref

class CanvasManager(QtCore.QObject):
    SINGLETON = None
    sigCanvasListChanged = QtCore.Signal()

    def __init__(self):
        if False:
            return 10
        if CanvasManager.SINGLETON is not None:
            raise Exception('Can only create one canvas manager.')
        CanvasManager.SINGLETON = self
        QtCore.QObject.__init__(self)
        self.canvases = weakref.WeakValueDictionary()

    @classmethod
    def instance(cls):
        if False:
            return 10
        return CanvasManager.SINGLETON

    def registerCanvas(self, canvas, name):
        if False:
            print('Hello World!')
        n2 = name
        i = 0
        while n2 in self.canvases:
            n2 = '%s_%03d' % (name, i)
            i += 1
        self.canvases[n2] = canvas
        self.sigCanvasListChanged.emit()
        return n2

    def unregisterCanvas(self, name):
        if False:
            for i in range(10):
                print('nop')
        c = self.canvases[name]
        del self.canvases[name]
        self.sigCanvasListChanged.emit()

    def listCanvases(self):
        if False:
            i = 10
            return i + 15
        return list(self.canvases.keys())

    def getCanvas(self, name):
        if False:
            return 10
        return self.canvases[name]
manager = CanvasManager()

class CanvasCombo(QtWidgets.QComboBox):

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QtWidgets.QComboBox.__init__(self, parent)
        man = CanvasManager.instance()
        man.sigCanvasListChanged.connect(self.updateCanvasList)
        self.hostName = None
        self.updateCanvasList()

    def updateCanvasList(self):
        if False:
            for i in range(10):
                print('nop')
        canvases = CanvasManager.instance().listCanvases()
        canvases.insert(0, '')
        if self.hostName in canvases:
            canvases.remove(self.hostName)
        sel = self.currentText()
        if sel in canvases:
            self.blockSignals(True)
        self.clear()
        for i in canvases:
            self.addItem(i)
            if i == sel:
                self.setCurrentIndex(self.count())
        self.blockSignals(False)

    def setHostName(self, name):
        if False:
            return 10
        self.hostName = name
        self.updateCanvasList()