import os
import re
from ..GraphicsScene import GraphicsScene
from ..Qt import QtCore, QtWidgets
from ..widgets.FileDialog import FileDialog
LastExportDirectory = None

class Exporter(object):
    """
    Abstract class used for exporting graphics to file / printer / whatever.
    """
    allowCopy = False
    Exporters = []

    @classmethod
    def register(cls):
        if False:
            print('Hello World!')
        '\n        Used to register Exporter classes to appear in the export dialog.\n        '
        Exporter.Exporters.append(cls)

    def __init__(self, item):
        if False:
            print('Hello World!')
        '\n        Initialize with the item to be exported.\n        Can be an individual graphics item or a scene.\n        '
        object.__init__(self)
        self.item = item

    def parameters(self):
        if False:
            i = 10
            return i + 15
        'Return the parameters used to configure this exporter.'
        raise Exception('Abstract method must be overridden in subclass.')

    def export(self, fileName=None, toBytes=False, copy=False):
        if False:
            while True:
                i = 10
        '\n        If *fileName* is None, pop-up a file dialog.\n        If *toBytes* is True, return a bytes object rather than writing to file.\n        If *copy* is True, export to the copy buffer rather than writing to file.\n        '
        raise Exception('Abstract method must be overridden in subclass.')

    def fileSaveDialog(self, filter=None, opts=None):
        if False:
            i = 10
            return i + 15
        if opts is None:
            opts = {}
        self.fileDialog = FileDialog()
        self.fileDialog.setFileMode(QtWidgets.QFileDialog.FileMode.AnyFile)
        self.fileDialog.setAcceptMode(QtWidgets.QFileDialog.AcceptMode.AcceptSave)
        if filter is not None:
            if isinstance(filter, str):
                self.fileDialog.setNameFilter(filter)
            elif isinstance(filter, list):
                self.fileDialog.setNameFilters(filter)
        global LastExportDirectory
        exportDir = LastExportDirectory
        if exportDir is not None:
            self.fileDialog.setDirectory(exportDir)
        self.fileDialog.show()
        self.fileDialog.opts = opts
        self.fileDialog.fileSelected.connect(self.fileSaveFinished)
        return

    def fileSaveFinished(self, fileName):
        if False:
            print('Hello World!')
        global LastExportDirectory
        LastExportDirectory = os.path.split(fileName)[0]
        ext = os.path.splitext(fileName)[1].lower().lstrip('.')
        selectedExt = re.search('\\*\\.(\\w+)\\b', self.fileDialog.selectedNameFilter())
        if selectedExt is not None:
            selectedExt = selectedExt.groups()[0].lower()
            if ext != selectedExt:
                fileName = fileName + '.' + selectedExt.lstrip('.')
        self.export(fileName=fileName, **self.fileDialog.opts)

    def getScene(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.item, GraphicsScene):
            return self.item
        else:
            return self.item.scene()

    def getSourceRect(self):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(self.item, GraphicsScene):
            w = self.item.getViewWidget()
            return w.viewportTransform().inverted()[0].mapRect(w.rect())
        else:
            return self.item.sceneBoundingRect()

    def getTargetRect(self):
        if False:
            print('Hello World!')
        if isinstance(self.item, GraphicsScene):
            return self.item.getViewWidget().rect()
        else:
            return self.item.mapRectToDevice(self.item.boundingRect())

    def setExportMode(self, export, opts=None):
        if False:
            return 10
        "\n        Call setExportMode(export, opts) on all items that will \n        be painted during the export. This informs the item\n        that it is about to be painted for export, allowing it to \n        alter its appearance temporarily\n        \n        \n        *export*  - bool; must be True before exporting and False afterward\n        *opts*    - dict; common parameters are 'antialias' and 'background'\n        "
        if opts is None:
            opts = {}
        for item in self.getPaintItems():
            if hasattr(item, 'setExportMode'):
                item.setExportMode(export, opts)

    def getPaintItems(self, root=None):
        if False:
            print('Hello World!')
        'Return a list of all items that should be painted in the correct order.'
        if root is None:
            root = self.item
        preItems = []
        postItems = []
        if isinstance(root, QtWidgets.QGraphicsScene):
            childs = [i for i in root.items() if i.parentItem() is None]
            rootItem = []
        else:
            childs = root.childItems()
            rootItem = [root]
        childs.sort(key=lambda a: a.zValue())
        while len(childs) > 0:
            ch = childs.pop(0)
            tree = self.getPaintItems(ch)
            if ch.flags() & ch.GraphicsItemFlag.ItemStacksBehindParent or (ch.zValue() < 0 and ch.flags() & ch.GraphicsItemFlag.ItemNegativeZStacksBehindParent):
                preItems.extend(tree)
            else:
                postItems.extend(tree)
        return preItems + rootItem + postItems

    def render(self, painter, targetRect, sourceRect, item=None):
        if False:
            print('Hello World!')
        self.getScene().render(painter, QtCore.QRectF(targetRect), QtCore.QRectF(sourceRect))