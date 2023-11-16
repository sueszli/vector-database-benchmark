from ..Qt import QtCore, QtGui, QtWidgets
translate = QtCore.QCoreApplication.translate

class ParameterItem(QtWidgets.QTreeWidgetItem):
    """
    Abstract ParameterTree item. 
    Used to represent the state of a Parameter from within a ParameterTree.
    
      - Sets first column of item to name
      - generates context menu if item is renamable or removable
      - handles child added / removed events
      - provides virtual functions for handling changes from parameter
    
    For more ParameterItem types, see ParameterTree.parameterTypes module.
    """

    def __init__(self, param, depth=0):
        if False:
            return 10
        QtWidgets.QTreeWidgetItem.__init__(self, [param.title(), ''])
        self.param = param
        self.param.registerItem(self)
        self.depth = depth
        param.sigValueChanged.connect(self.valueChanged)
        param.sigChildAdded.connect(self.childAdded)
        param.sigChildRemoved.connect(self.childRemoved)
        param.sigNameChanged.connect(self.nameChanged)
        param.sigLimitsChanged.connect(self.limitsChanged)
        param.sigDefaultChanged.connect(self.defaultChanged)
        param.sigOptionsChanged.connect(self.optsChanged)
        param.sigParentChanged.connect(self.parentChanged)
        self.updateFlags()
        self.ignoreNameColumnChange = False

    def updateFlags(self):
        if False:
            print('Hello World!')
        opts = self.param.opts
        flags = QtCore.Qt.ItemFlag.ItemIsSelectable | QtCore.Qt.ItemFlag.ItemIsEnabled
        if opts.get('renamable', False):
            if opts.get('title', None) is not None:
                raise Exception('Cannot make parameter with both title != None and renamable == True.')
            flags |= QtCore.Qt.ItemFlag.ItemIsEditable
        if opts.get('movable', False):
            flags |= QtCore.Qt.ItemFlag.ItemIsDragEnabled
        if opts.get('dropEnabled', False):
            flags |= QtCore.Qt.ItemFlag.ItemIsDropEnabled
        self.setFlags(flags)

    def valueChanged(self, param, val):
        if False:
            for i in range(10):
                print('nop')
        pass

    def isFocusable(self):
        if False:
            print('Hello World!')
        'Return True if this item should be included in the tab-focus order'
        return False

    def setFocus(self):
        if False:
            for i in range(10):
                print('nop')
        'Give input focus to this item.\n        Can be reimplemented to display editor widgets, etc.\n        '
        pass

    def focusNext(self, forward=True):
        if False:
            i = 10
            return i + 15
        'Give focus to the next (or previous) focusable item in the parameter tree'
        self.treeWidget().focusNext(self, forward=forward)

    def treeWidgetChanged(self):
        if False:
            i = 10
            return i + 15
        'Called when this item is added or removed from a tree.\n        Expansion, visibility, and column widgets must all be configured AFTER \n        the item is added to a tree, not during __init__.\n        '
        self.setHidden(not self.param.opts.get('visible', True))
        self.setExpanded(self.param.opts.get('expanded', True))

    def childAdded(self, param, child, pos):
        if False:
            for i in range(10):
                print('nop')
        item = child.makeTreeItem(depth=self.depth + 1)
        self.insertChild(pos, item)
        item.treeWidgetChanged()
        for (i, ch) in enumerate(child):
            item.childAdded(child, ch, i)

    def childRemoved(self, param, child):
        if False:
            for i in range(10):
                print('nop')
        for i in range(self.childCount()):
            item = self.child(i)
            if item.param is child:
                self.takeChild(i)
                break

    def parentChanged(self, param, parent):
        if False:
            while True:
                i = 10
        pass

    def contextMenuEvent(self, ev):
        if False:
            print('Hello World!')
        opts = self.param.opts
        if not opts.get('removable', False) and (not opts.get('renamable', False)) and ('context' not in opts):
            return
        self.contextMenu = QtWidgets.QMenu()
        self.contextMenu.addSeparator()
        if opts.get('renamable', False):
            self.contextMenu.addAction(translate('ParameterItem', 'Rename')).triggered.connect(self.editName)
        if opts.get('removable', False):
            self.contextMenu.addAction(translate('ParameterItem', 'Remove')).triggered.connect(self.requestRemove)
        context = opts.get('context', None)
        if isinstance(context, list):
            for name in context:
                self.contextMenu.addAction(name).triggered.connect(self.contextMenuTriggered(name))
        elif isinstance(context, dict):
            for (name, title) in context.items():
                self.contextMenu.addAction(title).triggered.connect(self.contextMenuTriggered(name))
        self.contextMenu.popup(ev.globalPos())

    def columnChangedEvent(self, col):
        if False:
            while True:
                i = 10
        'Called when the text in a column has been edited (or otherwise changed).\n        By default, we only use changes to column 0 to rename the parameter.\n        '
        if col == 0 and self.param.opts.get('title', None) is None:
            if self.ignoreNameColumnChange:
                return
            try:
                newName = self.param.setName(self.text(col))
            except Exception:
                self.setText(0, self.param.name())
                raise
            try:
                self.ignoreNameColumnChange = True
                self.nameChanged(self, newName)
            finally:
                self.ignoreNameColumnChange = False

    def expandedChangedEvent(self, expanded):
        if False:
            i = 10
            return i + 15
        if self.param.opts['syncExpanded']:
            self.param.setOpts(expanded=expanded)

    def nameChanged(self, param, name):
        if False:
            while True:
                i = 10
        if self.param.opts.get('title', None) is None:
            self.titleChanged()

    def titleChanged(self):
        if False:
            i = 10
            return i + 15
        title = self.param.title()
        if not title or title == 'params':
            return
        self.setText(0, title)
        fm = QtGui.QFontMetrics(self.font(0))
        textFlags = QtCore.Qt.TextFlag.TextSingleLine
        size = fm.size(textFlags, self.text(0))
        size.setHeight(int(size.height() * 1.35))
        size.setWidth(int(size.width() * 1.15))
        self.setSizeHint(0, size)

    def limitsChanged(self, param, limits):
        if False:
            i = 10
            return i + 15
        "Called when the parameter's limits have changed"
        pass

    def defaultChanged(self, param, default):
        if False:
            while True:
                i = 10
        "Called when the parameter's default value has changed"
        pass

    def optsChanged(self, param, opts):
        if False:
            i = 10
            return i + 15
        'Called when any options are changed that are not\n        name, value, default, or limits'
        if 'visible' in opts:
            self.setHidden(not opts['visible'])
        if 'expanded' in opts:
            if self.isExpanded() != opts['expanded']:
                self.setExpanded(opts['expanded'])
        if 'title' in opts:
            self.titleChanged()
        self.updateFlags()

    def contextMenuTriggered(self, name):
        if False:
            i = 10
            return i + 15

        def trigger():
            if False:
                return 10
            self.param.contextMenu(name)
        return trigger

    def editName(self):
        if False:
            print('Hello World!')
        self.treeWidget().editItem(self, 0)

    def selected(self, sel):
        if False:
            for i in range(10):
                print('nop')
        'Called when this item has been selected (sel=True) OR deselected (sel=False)'
        pass

    def requestRemove(self):
        if False:
            return 10
        QtCore.QTimer.singleShot(0, self.param.remove)

    def __hash__(self):
        if False:
            print('Hello World!')
        return id(self)

    def __eq__(self, x):
        if False:
            while True:
                i = 10
        return x is self