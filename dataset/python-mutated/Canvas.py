__all__ = ['Canvas']
import gc
import importlib
import weakref
import warnings
from ..graphicsItems.GridItem import GridItem
from ..graphicsItems.ROI import ROI
from ..graphicsItems.ViewBox import ViewBox
from ..Qt import QT_LIB, QtCore, QtGui, QtWidgets
from . import CanvasTemplate_generic as ui_template
from .CanvasItem import CanvasItem, GroupCanvasItem
from .CanvasManager import CanvasManager
translate = QtCore.QCoreApplication.translate

class Canvas(QtWidgets.QWidget):
    sigSelectionChanged = QtCore.Signal(object, object)
    sigItemTransformChanged = QtCore.Signal(object, object)
    sigItemTransformChangeFinished = QtCore.Signal(object, object)

    def __init__(self, parent=None, allowTransforms=True, hideCtrl=False, name=None):
        if False:
            i = 10
            return i + 15
        QtWidgets.QWidget.__init__(self, parent)
        warnings.warn('pyqtgrapoh.cavas will be deprecated in pyqtgraph and migrate to acq4.  Removal will occur after September, 2023.', DeprecationWarning, stacklevel=2)
        self.ui = ui_template.Ui_Form()
        self.ui.setupUi(self)
        self.view = ViewBox()
        self.ui.view.setCentralItem(self.view)
        self.itemList = self.ui.itemList
        self.itemList.setSelectionMode(self.itemList.SelectionMode.ExtendedSelection)
        self.allowTransforms = allowTransforms
        self.multiSelectBox = SelectBox()
        self.view.addItem(self.multiSelectBox)
        self.multiSelectBox.hide()
        self.multiSelectBox.setZValue(1000000.0)
        self.ui.mirrorSelectionBtn.hide()
        self.ui.reflectSelectionBtn.hide()
        self.ui.resetTransformsBtn.hide()
        self.redirect = None
        self.items = []
        self.view.setAspectLocked(True)
        grid = GridItem()
        self.grid = CanvasItem(grid, name='Grid', movable=False)
        self.addItem(self.grid)
        self.hideBtn = QtWidgets.QPushButton('>', self)
        self.hideBtn.setFixedWidth(20)
        self.hideBtn.setFixedHeight(20)
        self.ctrlSize = 200
        self.sizeApplied = False
        self.hideBtn.clicked.connect(self.hideBtnClicked)
        self.ui.splitter.splitterMoved.connect(self.splitterMoved)
        self.ui.itemList.itemChanged.connect(self.treeItemChanged)
        self.ui.itemList.sigItemMoved.connect(self.treeItemMoved)
        self.ui.itemList.itemSelectionChanged.connect(self.treeItemSelected)
        self.ui.autoRangeBtn.clicked.connect(self.autoRange)
        self.ui.redirectCheck.toggled.connect(self.updateRedirect)
        self.ui.redirectCombo.currentIndexChanged.connect(self.updateRedirect)
        self.multiSelectBox.sigRegionChanged.connect(self.multiSelectBoxChanged)
        self.multiSelectBox.sigRegionChangeFinished.connect(self.multiSelectBoxChangeFinished)
        self.ui.mirrorSelectionBtn.clicked.connect(self.mirrorSelectionClicked)
        self.ui.reflectSelectionBtn.clicked.connect(self.reflectSelectionClicked)
        self.ui.resetTransformsBtn.clicked.connect(self.resetTransformsClicked)
        self.resizeEvent()
        if hideCtrl:
            self.hideBtnClicked()
        if name is not None:
            self.registeredName = CanvasManager.instance().registerCanvas(self, name)
            self.ui.redirectCombo.setHostName(self.registeredName)
        self.menu = QtWidgets.QMenu()
        remAct = QtGui.QAction(translate('Context Menu', 'Remove item'), self.menu)
        remAct.triggered.connect(self.removeClicked)
        self.menu.addAction(remAct)
        self.menu.remAct = remAct
        self.ui.itemList.contextMenuEvent = self.itemListContextMenuEvent

    def splitterMoved(self):
        if False:
            print('Hello World!')
        self.resizeEvent()

    def hideBtnClicked(self):
        if False:
            print('Hello World!')
        ctrlSize = self.ui.splitter.sizes()[1]
        if ctrlSize == 0:
            cs = self.ctrlSize
            w = self.ui.splitter.size().width()
            if cs > w:
                cs = w - 20
            self.ui.splitter.setSizes([w - cs, cs])
            self.hideBtn.setText('>')
        else:
            self.ctrlSize = ctrlSize
            self.ui.splitter.setSizes([100, 0])
            self.hideBtn.setText('<')
        self.resizeEvent()

    def autoRange(self):
        if False:
            i = 10
            return i + 15
        self.view.autoRange()

    def resizeEvent(self, ev=None):
        if False:
            for i in range(10):
                print('nop')
        if ev is not None:
            super().resizeEvent(ev)
        self.hideBtn.move(self.ui.view.size().width() - self.hideBtn.width(), 0)
        if not self.sizeApplied:
            self.sizeApplied = True
            s = int(min(self.width(), max(100, min(200, self.width() // 4))))
            s2 = self.width() - s
            self.ui.splitter.setSizes([s2, s])

    def updateRedirect(self, *args):
        if False:
            print('Hello World!')
        cname = str(self.ui.redirectCombo.currentText())
        man = CanvasManager.instance()
        if self.ui.redirectCheck.isChecked() and cname != '':
            redirect = man.getCanvas(cname)
        else:
            redirect = None
        if self.redirect is redirect:
            return
        self.redirect = redirect
        if redirect is None:
            self.reclaimItems()
        else:
            self.redirectItems(redirect)

    def redirectItems(self, canvas):
        if False:
            while True:
                i = 10
        for i in self.items:
            if i is self.grid:
                continue
            li = i.listItem
            parent = li.parent()
            if parent is None:
                tree = li.treeWidget()
                if tree is None:
                    print('Skipping item', i, i.name)
                    continue
                tree.removeTopLevelItem(li)
            else:
                parent.removeChild(li)
            canvas.addItem(i)

    def reclaimItems(self):
        if False:
            while True:
                i = 10
        items = self.items
        self.items = [self.grid]
        items.remove(self.grid)
        for i in items:
            i.canvas.removeItem(i)
            self.addItem(i)

    def treeItemChanged(self, item, col):
        if False:
            i = 10
            return i + 15
        try:
            citem = item.canvasItem()
        except AttributeError:
            return
        if item.checkState(0) == QtCore.Qt.CheckState.Checked:
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, QtCore.Qt.CheckState.Checked)
            citem.show()
        else:
            for i in range(item.childCount()):
                item.child(i).setCheckState(0, QtCore.Qt.CheckState.Unchecked)
            citem.hide()

    def treeItemSelected(self):
        if False:
            return 10
        sel = self.selectedItems()
        if len(sel) == 0:
            return
        multi = len(sel) > 1
        for i in self.items:
            i.selectionChanged(i in sel, multi)
        if len(sel) == 1:
            self.multiSelectBox.hide()
            self.ui.mirrorSelectionBtn.hide()
            self.ui.reflectSelectionBtn.hide()
            self.ui.resetTransformsBtn.hide()
        elif len(sel) > 1:
            self.showMultiSelectBox()
        self.sigSelectionChanged.emit(self, sel)

    def selectedItems(self):
        if False:
            i = 10
            return i + 15
        '\n        Return list of all selected canvasItems\n        '
        return [item.canvasItem() for item in self.itemList.selectedItems() if item.canvasItem() is not None]

    def selectItem(self, item):
        if False:
            i = 10
            return i + 15
        li = item.listItem
        self.itemList.setCurrentItem(li)

    def showMultiSelectBox(self):
        if False:
            while True:
                i = 10
        items = self.selectedItems()
        rect = self.view.itemBoundingRect(items[0].graphicsItem())
        for i in items:
            if not i.isMovable():
                return
            br = self.view.itemBoundingRect(i.graphicsItem())
            rect = rect | br
        self.multiSelectBox.blockSignals(True)
        self.multiSelectBox.setPos([rect.x(), rect.y()])
        self.multiSelectBox.setSize(rect.size())
        self.multiSelectBox.setAngle(0)
        self.multiSelectBox.blockSignals(False)
        self.multiSelectBox.show()
        self.ui.mirrorSelectionBtn.show()
        self.ui.reflectSelectionBtn.show()
        self.ui.resetTransformsBtn.show()

    def mirrorSelectionClicked(self):
        if False:
            i = 10
            return i + 15
        for ci in self.selectedItems():
            ci.mirrorY()
        self.showMultiSelectBox()

    def reflectSelectionClicked(self):
        if False:
            for i in range(10):
                print('nop')
        for ci in self.selectedItems():
            ci.mirrorXY()
        self.showMultiSelectBox()

    def resetTransformsClicked(self):
        if False:
            while True:
                i = 10
        for i in self.selectedItems():
            i.resetTransformClicked()
        self.showMultiSelectBox()

    def multiSelectBoxChanged(self):
        if False:
            return 10
        self.multiSelectBoxMoved()

    def multiSelectBoxChangeFinished(self):
        if False:
            i = 10
            return i + 15
        for ci in self.selectedItems():
            ci.applyTemporaryTransform()
            ci.sigTransformChangeFinished.emit(ci)

    def multiSelectBoxMoved(self):
        if False:
            for i in range(10):
                print('nop')
        transform = self.multiSelectBox.getGlobalTransform()
        for ci in self.selectedItems():
            ci.setTemporaryTransform(transform)
            ci.sigTransformChanged.emit(ci)

    def addGraphicsItem(self, item, **opts):
        if False:
            while True:
                i = 10
        'Add a new GraphicsItem to the scene at pos.\n        Common options are name, pos, scale, and z\n        '
        citem = CanvasItem(item, **opts)
        item._canvasItem = citem
        self.addItem(citem)
        return citem

    def addGroup(self, name, **kargs):
        if False:
            while True:
                i = 10
        group = GroupCanvasItem(name=name)
        self.addItem(group, **kargs)
        return group

    def addItem(self, citem):
        if False:
            for i in range(10):
                print('nop')
        '\n        Add an item to the canvas. \n        '
        if self.redirect is not None:
            name = self.redirect.addItem(citem)
            self.items.append(citem)
            return name
        if not self.allowTransforms:
            citem.setMovable(False)
        citem.sigTransformChanged.connect(self.itemTransformChanged)
        citem.sigTransformChangeFinished.connect(self.itemTransformChangeFinished)
        citem.sigVisibilityChanged.connect(self.itemVisibilityChanged)
        name = citem.opts['name']
        if name is None:
            name = 'item'
        insertLocation = 0
        parent = citem.parentItem()
        if parent in (None, self.view.childGroup):
            parent = self.itemList.invisibleRootItem()
        else:
            parent = parent.listItem
        siblings = [parent.child(i).canvasItem() for i in range(parent.childCount())]
        z = citem.zValue()
        if z is None:
            zvals = [i.zValue() for i in siblings]
            if parent is self.itemList.invisibleRootItem():
                if len(zvals) == 0:
                    z = 0
                else:
                    z = max(zvals) + 10
            elif len(zvals) == 0:
                z = parent.canvasItem().zValue()
            else:
                z = max(zvals) + 1
            citem.setZValue(z)
        for i in range(parent.childCount()):
            ch = parent.child(i)
            zval = ch.canvasItem().graphicsItem().zValue()
            if zval < z:
                insertLocation = i
                break
            else:
                insertLocation = i + 1
        node = QtWidgets.QTreeWidgetItem([name])
        flags = node.flags() | QtCore.Qt.ItemFlag.ItemIsUserCheckable | QtCore.Qt.ItemFlag.ItemIsDragEnabled
        if not isinstance(citem, GroupCanvasItem):
            flags = flags & ~QtCore.Qt.ItemFlag.ItemIsDropEnabled
        node.setFlags(flags)
        if citem.opts['visible']:
            node.setCheckState(0, QtCore.Qt.CheckState.Checked)
        else:
            node.setCheckState(0, QtCore.Qt.CheckState.Unchecked)
        node.name = name
        parent.insertChild(insertLocation, node)
        citem.name = name
        citem.listItem = node
        node.canvasItem = weakref.ref(citem)
        self.items.append(citem)
        ctrl = citem.ctrlWidget()
        ctrl.hide()
        self.ui.ctrlLayout.addWidget(ctrl)
        citem.setCanvas(self)
        if len(self.items) == 2:
            self.autoRange()
        return citem

    def treeItemMoved(self, item, parent, index):
        if False:
            for i in range(10):
                print('nop')
        if parent is self.itemList.invisibleRootItem():
            item.canvasItem().setParentItem(self.view.childGroup)
        else:
            item.canvasItem().setParentItem(parent.canvasItem())
        siblings = [parent.child(i).canvasItem() for i in range(parent.childCount())]
        zvals = [i.zValue() for i in siblings]
        zvals.sort(reverse=True)
        for i in range(len(siblings)):
            item = siblings[i]
            item.setZValue(zvals[i])

    def itemVisibilityChanged(self, item):
        if False:
            for i in range(10):
                print('nop')
        listItem = item.listItem
        checked = listItem.checkState(0) == QtCore.Qt.CheckState.Checked
        vis = item.isVisible()
        if vis != checked:
            if vis:
                listItem.setCheckState(0, QtCore.Qt.CheckState.Checked)
            else:
                listItem.setCheckState(0, QtCore.Qt.CheckState.Unchecked)

    def removeItem(self, item):
        if False:
            return 10
        if isinstance(item, QtWidgets.QTreeWidgetItem):
            item = item.canvasItem()
        if isinstance(item, CanvasItem):
            item.setCanvas(None)
            listItem = item.listItem
            listItem.canvasItem = None
            item.listItem = None
            self.itemList.removeTopLevelItem(listItem)
            self.items.remove(item)
            ctrl = item.ctrlWidget()
            ctrl.hide()
            self.ui.ctrlLayout.removeWidget(ctrl)
            ctrl.setParent(None)
        elif hasattr(item, '_canvasItem'):
            self.removeItem(item._canvasItem)
        else:
            self.view.removeItem(item)
        gc.collect()

    def clear(self):
        if False:
            while True:
                i = 10
        while len(self.items) > 0:
            self.removeItem(self.items[0])

    def addToScene(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.view.addItem(item)

    def removeFromScene(self, item):
        if False:
            print('Hello World!')
        self.view.removeItem(item)

    def listItems(self):
        if False:
            i = 10
            return i + 15
        'Return a dictionary of name:item pairs'
        return self.items

    def getListItem(self, name):
        if False:
            for i in range(10):
                print('nop')
        return self.items[name]

    def itemTransformChanged(self, item):
        if False:
            return 10
        self.sigItemTransformChanged.emit(self, item)

    def itemTransformChangeFinished(self, item):
        if False:
            for i in range(10):
                print('nop')
        self.sigItemTransformChangeFinished.emit(self, item)

    def itemListContextMenuEvent(self, ev):
        if False:
            print('Hello World!')
        self.menuItem = self.itemList.itemAt(ev.pos())
        self.menu.popup(ev.globalPos())

    def removeClicked(self):
        if False:
            for i in range(10):
                print('nop')
        for item in self.selectedItems():
            self.removeItem(item)
        self.menuItem = None
        import gc
        gc.collect()

class SelectBox(ROI):

    def __init__(self, scalable=False):
        if False:
            return 10
        ROI.__init__(self, [0, 0], [1, 1])
        center = [0.5, 0.5]
        if scalable:
            self.addScaleHandle([1, 1], center, lockAspect=True)
            self.addScaleHandle([0, 0], center, lockAspect=True)
        self.addRotateHandle([0, 1], center)
        self.addRotateHandle([1, 0], center)