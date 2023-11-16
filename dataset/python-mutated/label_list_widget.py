from qtpy import QtCore
from qtpy.QtCore import Qt
from qtpy import QtGui
from qtpy.QtGui import QPalette
from qtpy import QtWidgets
from qtpy.QtWidgets import QStyle

class HTMLDelegate(QtWidgets.QStyledItemDelegate):

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super(HTMLDelegate, self).__init__()
        self.doc = QtGui.QTextDocument(self)

    def paint(self, painter, option, index):
        if False:
            i = 10
            return i + 15
        painter.save()
        options = QtWidgets.QStyleOptionViewItem(option)
        self.initStyleOption(options, index)
        self.doc.setHtml(options.text)
        options.text = ''
        style = QtWidgets.QApplication.style() if options.widget is None else options.widget.style()
        style.drawControl(QStyle.CE_ItemViewItem, options, painter)
        ctx = QtGui.QAbstractTextDocumentLayout.PaintContext()
        if option.state & QStyle.State_Selected:
            ctx.palette.setColor(QPalette.Text, option.palette.color(QPalette.Active, QPalette.HighlightedText))
        else:
            ctx.palette.setColor(QPalette.Text, option.palette.color(QPalette.Active, QPalette.Text))
        textRect = style.subElementRect(QStyle.SE_ItemViewItemText, options)
        if index.column() != 0:
            textRect.adjust(5, 0, 0, 0)
        thefuckyourshitup_constant = 4
        margin = (option.rect.height() - options.fontMetrics.height()) // 2
        margin = margin - thefuckyourshitup_constant
        textRect.setTop(textRect.top() + margin)
        painter.translate(textRect.topLeft())
        painter.setClipRect(textRect.translated(-textRect.topLeft()))
        self.doc.documentLayout().draw(painter, ctx)
        painter.restore()

    def sizeHint(self, option, index):
        if False:
            i = 10
            return i + 15
        thefuckyourshitup_constant = 4
        return QtCore.QSize(int(self.doc.idealWidth()), int(self.doc.size().height() - thefuckyourshitup_constant))

class LabelListWidgetItem(QtGui.QStandardItem):

    def __init__(self, text=None, shape=None):
        if False:
            for i in range(10):
                print('nop')
        super(LabelListWidgetItem, self).__init__()
        self.setText(text or '')
        self.setShape(shape)
        self.setCheckable(True)
        self.setCheckState(Qt.Checked)
        self.setEditable(False)
        self.setTextAlignment(Qt.AlignBottom)

    def clone(self):
        if False:
            i = 10
            return i + 15
        return LabelListWidgetItem(self.text(), self.shape())

    def setShape(self, shape):
        if False:
            return 10
        self.setData(shape, Qt.UserRole)

    def shape(self):
        if False:
            print('Hello World!')
        return self.data(Qt.UserRole)

    def __hash__(self):
        if False:
            print('Hello World!')
        return id(self)

    def __repr__(self):
        if False:
            return 10
        return '{}("{}")'.format(self.__class__.__name__, self.text())

class StandardItemModel(QtGui.QStandardItemModel):
    itemDropped = QtCore.Signal()

    def removeRows(self, *args, **kwargs):
        if False:
            while True:
                i = 10
        ret = super().removeRows(*args, **kwargs)
        self.itemDropped.emit()
        return ret

class LabelListWidget(QtWidgets.QListView):
    itemDoubleClicked = QtCore.Signal(LabelListWidgetItem)
    itemSelectionChanged = QtCore.Signal(list, list)

    def __init__(self):
        if False:
            while True:
                i = 10
        super(LabelListWidget, self).__init__()
        self._selectedItems = []
        self.setWindowFlags(Qt.Window)
        self.setModel(StandardItemModel())
        self.model().setItemPrototype(LabelListWidgetItem())
        self.setItemDelegate(HTMLDelegate())
        self.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.setDragDropMode(QtWidgets.QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)
        self.doubleClicked.connect(self.itemDoubleClickedEvent)
        self.selectionModel().selectionChanged.connect(self.itemSelectionChangedEvent)

    def __len__(self):
        if False:
            return 10
        return self.model().rowCount()

    def __getitem__(self, i):
        if False:
            return 10
        return self.model().item(i)

    def __iter__(self):
        if False:
            while True:
                i = 10
        for i in range(len(self)):
            yield self[i]

    @property
    def itemDropped(self):
        if False:
            while True:
                i = 10
        return self.model().itemDropped

    @property
    def itemChanged(self):
        if False:
            while True:
                i = 10
        return self.model().itemChanged

    def itemSelectionChangedEvent(self, selected, deselected):
        if False:
            print('Hello World!')
        selected = [self.model().itemFromIndex(i) for i in selected.indexes()]
        deselected = [self.model().itemFromIndex(i) for i in deselected.indexes()]
        self.itemSelectionChanged.emit(selected, deselected)

    def itemDoubleClickedEvent(self, index):
        if False:
            i = 10
            return i + 15
        self.itemDoubleClicked.emit(self.model().itemFromIndex(index))

    def selectedItems(self):
        if False:
            print('Hello World!')
        return [self.model().itemFromIndex(i) for i in self.selectedIndexes()]

    def scrollToItem(self, item):
        if False:
            while True:
                i = 10
        self.scrollTo(self.model().indexFromItem(item))

    def addItem(self, item):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(item, LabelListWidgetItem):
            raise TypeError('item must be LabelListWidgetItem')
        self.model().setItem(self.model().rowCount(), 0, item)
        item.setSizeHint(self.itemDelegate().sizeHint(None, None))

    def removeItem(self, item):
        if False:
            while True:
                i = 10
        index = self.model().indexFromItem(item)
        self.model().removeRows(index.row(), 1)

    def selectItem(self, item):
        if False:
            return 10
        index = self.model().indexFromItem(item)
        self.selectionModel().select(index, QtCore.QItemSelectionModel.Select)

    def findItemByShape(self, shape):
        if False:
            i = 10
            return i + 15
        for row in range(self.model().rowCount()):
            item = self.model().item(row, 0)
            if item.shape() == shape:
                return item
        raise ValueError('cannot find shape: {}'.format(shape))

    def clear(self):
        if False:
            while True:
                i = 10
        self.model().clear()