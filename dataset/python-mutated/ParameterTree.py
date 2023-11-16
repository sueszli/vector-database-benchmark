from .parameterTypes import GroupParameterItem
from ..Qt import QtCore, QtWidgets, mkQApp
from ..widgets.TreeWidget import TreeWidget
from .ParameterItem import ParameterItem

class ParameterTree(TreeWidget):
    """Widget used to display or control data from a hierarchy of Parameters"""

    def __init__(self, parent=None, showHeader=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        ============== ========================================================\n        **Arguments:**\n        parent         (QWidget) An optional parent widget\n        showHeader     (bool) If True, then the QTreeView header is displayed.\n        ============== ========================================================\n        '
        TreeWidget.__init__(self, parent)
        self.setVerticalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setHorizontalScrollMode(self.ScrollMode.ScrollPerPixel)
        self.setAnimated(False)
        self.setColumnCount(2)
        self.setHeaderLabels(['Parameter', 'Value'])
        self.setAlternatingRowColors(True)
        self.paramSet = None
        self.header().setSectionResizeMode(QtWidgets.QHeaderView.ResizeMode.ResizeToContents)
        self.setHeaderHidden(not showHeader)
        self.itemChanged.connect(self.itemChangedEvent)
        self.itemExpanded.connect(self.itemExpandedEvent)
        self.itemCollapsed.connect(self.itemCollapsedEvent)
        self.lastSel = None
        self.setRootIsDecorated(False)
        app = mkQApp()
        app.paletteChanged.connect(self.updatePalette)

    def setParameters(self, param, showTop=True):
        if False:
            for i in range(10):
                print('nop')
        '\n        Set the top-level :class:`Parameter <pyqtgraph.parametertree.Parameter>`\n        to be displayed in this ParameterTree.\n\n        If *showTop* is False, then the top-level parameter is hidden and only \n        its children will be visible. This is a convenience method equivalent \n        to::\n        \n            tree.clear()\n            tree.addParameters(param, showTop)\n        '
        self.clear()
        self.addParameters(param, showTop=showTop)

    def addParameters(self, param, root=None, depth=0, showTop=True):
        if False:
            print('Hello World!')
        '\n        Adds one top-level :class:`Parameter <pyqtgraph.parametertree.Parameter>`\n        to the view. \n        \n        ============== ==========================================================\n        **Arguments:** \n        param          The :class:`Parameter <pyqtgraph.parametertree.Parameter>` \n                       to add.\n        root           The item within the tree to which *param* should be added.\n                       By default, *param* is added as a top-level item.\n        showTop        If False, then *param* will be hidden, and only its \n                       children will be visible in the tree.\n        ============== ==========================================================\n        '
        item = param.makeTreeItem(depth=depth)
        if root is None:
            root = self.invisibleRootItem()
            if not showTop:
                item.setText(0, '')
                item.setSizeHint(0, QtCore.QSize(1, 1))
                item.setSizeHint(1, QtCore.QSize(1, 1))
                depth -= 1
        root.addChild(item)
        item.treeWidgetChanged()
        for ch in param:
            self.addParameters(ch, root=item, depth=depth + 1)

    def clear(self):
        if False:
            i = 10
            return i + 15
        '\n        Remove all parameters from the tree.        \n        '
        self.invisibleRootItem().takeChildren()

    def focusNext(self, item, forward=True):
        if False:
            i = 10
            return i + 15
        'Give input focus to the next (or previous) item after *item*\n        '
        while True:
            parent = item.parent()
            if parent is None:
                return
            nextItem = self.nextFocusableChild(parent, item, forward=forward)
            if nextItem is not None:
                nextItem.setFocus()
                self.setCurrentItem(nextItem)
                return
            item = parent

    def focusPrevious(self, item):
        if False:
            return 10
        self.focusNext(item, forward=False)

    def nextFocusableChild(self, root, startItem=None, forward=True):
        if False:
            while True:
                i = 10
        if startItem is None:
            if forward:
                index = 0
            else:
                index = root.childCount() - 1
        elif forward:
            index = root.indexOfChild(startItem) + 1
        else:
            index = root.indexOfChild(startItem) - 1
        if forward:
            inds = list(range(index, root.childCount()))
        else:
            inds = list(range(index, -1, -1))
        for i in inds:
            item = root.child(i)
            if hasattr(item, 'isFocusable') and item.isFocusable():
                return item
            else:
                item = self.nextFocusableChild(item, forward=forward)
                if item is not None:
                    return item
        return None

    def contextMenuEvent(self, ev):
        if False:
            return 10
        item = self.currentItem()
        if hasattr(item, 'contextMenuEvent'):
            item.contextMenuEvent(ev)

    def updatePalette(self):
        if False:
            i = 10
            return i + 15
        '\n        called when application palette changes\n        This should ensure that the color theme of the OS is applied to the GroupParameterItems\n        should the theme chang while the application is running.\n        '
        for item in self.listAllItems():
            if isinstance(item, GroupParameterItem):
                item.updateDepth(item.depth)

    def itemChangedEvent(self, item, col):
        if False:
            i = 10
            return i + 15
        if hasattr(item, 'columnChangedEvent'):
            item.columnChangedEvent(col)

    def itemExpandedEvent(self, item):
        if False:
            while True:
                i = 10
        if hasattr(item, 'expandedChangedEvent'):
            item.expandedChangedEvent(True)

    def itemCollapsedEvent(self, item):
        if False:
            return 10
        if hasattr(item, 'expandedChangedEvent'):
            item.expandedChangedEvent(False)

    def selectionChanged(self, *args):
        if False:
            i = 10
            return i + 15
        sel = self.selectedItems()
        if len(sel) != 1:
            sel = None
        if self.lastSel is not None and isinstance(self.lastSel, ParameterItem):
            self.lastSel.selected(False)
        if sel is None:
            self.lastSel = None
            return
        self.lastSel = sel[0]
        if hasattr(sel[0], 'selected'):
            sel[0].selected(True)
        return super().selectionChanged(*args)

    def sizeHint(self):
        if False:
            i = 10
            return i + 15
        (w, h) = (0, 0)
        ind = self.indentation()
        for x in self.listAllItems():
            if x.isHidden():
                continue
            try:
                depth = x.depth
            except AttributeError:
                depth = 0
            s0 = x.sizeHint(0)
            s1 = x.sizeHint(1)
            w = max(w, depth * ind + max(0, s0.width()) + max(0, s1.width()))
            h += max(0, s0.height(), s1.height())
        if not self.header().isHidden():
            h += self.header().height()
        return QtCore.QSize(w, h)