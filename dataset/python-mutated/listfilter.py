from AnyQt.QtWidgets import QListView, QLineEdit, QCompleter, QAbstractItemView
from AnyQt.QtGui import QDrag
from AnyQt.QtCore import Qt, QObject, QEvent, QModelIndex, QAbstractItemModel, QSortFilterProxyModel, QStringListModel
from AnyQt.QtCore import pyqtSignal as Signal
from Orange.widgets.utils.itemmodels import VariableListModel, PyListModel
import Orange

def slices(indices):
    if False:
        i = 10
        return i + 15
    ' Group the given integer indices into slices\n    '
    indices = list(sorted(indices))
    if indices:
        first = last = indices[0]
        for i in indices[1:]:
            if i == last + 1:
                last = i
            else:
                yield (first, last + 1)
                first = last = i
        yield (first, last + 1)

def delslice(model, start, end):
    if False:
        print('Hello World!')
    ' Delete the start, end slice (rows) from the model.\n    '
    if isinstance(model, PyListModel):
        del model[start:end]
    elif isinstance(model, QAbstractItemModel):
        model.removeRows(start, end - start)
    else:
        raise TypeError(type(model))

class VariablesListItemView(QListView):
    """ A Simple QListView subclass initialized for displaying
    variables.
    """
    dragDropActionDidComplete = Signal(Qt.DropAction)

    def __init__(self, parent=None, acceptedType=Orange.data.Variable):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.setSelectionMode(self.ExtendedSelection)
        self.setAcceptDrops(True)
        self.setDragEnabled(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(self.DragDrop)
        self.setDefaultDropAction(Qt.MoveAction)
        self.setDragDropOverwriteMode(False)
        self.setUniformItemSizes(True)
        self.viewport().setAcceptDrops(True)
        self.__acceptedType = acceptedType

    def startDrag(self, supported_actions):
        if False:
            return 10
        indices = self.selectionModel().selectedIndexes()
        indices = [i for i in indices if i.flags() & Qt.ItemIsDragEnabled]
        if indices:
            data = self.model().mimeData(indices)
            if not data:
                return
            drag = QDrag(self)
            drag.setMimeData(data)
            default_action = Qt.IgnoreAction
            if self.defaultDropAction() != Qt.IgnoreAction and supported_actions & self.defaultDropAction():
                default_action = self.defaultDropAction()
            elif supported_actions & Qt.CopyAction and self.dragDropMode() != self.InternalMove:
                default_action = Qt.CopyAction
            res = drag.exec(supported_actions, default_action)
            if res == Qt.MoveAction:
                selected = self.selectionModel().selectedIndexes()
                rows = list(map(QModelIndex.row, selected))
                for (s1, s2) in reversed(list(slices(rows))):
                    delslice(self.model(), s1, s2)
            self.dragDropActionDidComplete.emit(res)

    def dropEvent(self, event):
        if False:
            print('Hello World!')
        QAbstractItemView.dropEvent(self, event)

    def dragEnterEvent(self, event):
        if False:
            print('Hello World!')
        '\n        Reimplemented from QListView.dragEnterEvent\n        '
        if self.acceptsDropEvent(event):
            event.accept()
        else:
            event.ignore()

    def acceptsDropEvent(self, event):
        if False:
            while True:
                i = 10
        '\n        Should the drop event be accepted?\n        '
        if event.source() is not None and event.source().window() is not self.window():
            return False
        mime = event.mimeData()
        vars = mime.property('_items')
        if vars is None:
            return False
        if not all((isinstance(var, self.__acceptedType) for var in vars)):
            return False
        event.accept()
        return True

class VariableFilterProxyModel(QSortFilterProxyModel):
    """ A proxy model for filtering a list of variables based on
    their names and labels.

    """

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._filter_string = ''

    def set_filter_string(self, filter):
        if False:
            while True:
                i = 10
        self._filter_string = str(filter).lower()
        self.invalidateFilter()

    def filter_string(self):
        if False:
            for i in range(10):
                print('nop')
        return self._filter_string

    def filter_accepts_variable(self, var):
        if False:
            for i in range(10):
                print('nop')
        row_str = var.name + ' '.join(('%s=%s' % item for item in var.attributes.items()))
        row_str = row_str.lower()
        filters = self._filter_string.split()
        return all((f in row_str for f in filters))

    def filterAcceptsRow(self, source_row, source_parent):
        if False:
            i = 10
            return i + 15
        model = self.sourceModel()
        if isinstance(model, VariableListModel):
            var = model[source_row]
            return self.filter_accepts_variable(var)
        else:
            return True

class CompleterNavigator(QObject):
    """ An event filter to be installed on a QLineEdit, to enable
    Key up/ down to navigate between posible completions.
    """

    def eventFilter(self, obj, event):
        if False:
            i = 10
            return i + 15
        if event.type() == QEvent.KeyPress and isinstance(obj, QLineEdit):
            if event.key() == Qt.Key_Down:
                diff = 1
            elif event.key() == Qt.Key_Up:
                diff = -1
            else:
                return False
            completer = obj.completer()
            if completer is not None and completer.completionCount() > 0:
                current = completer.currentRow()
                current += diff
                completer.setCurrentRow(current % completer.completionCount())
                completer.complete()
            return True
        else:
            return False

def variables_filter(model, parent=None, accepted_type=Orange.data.Variable, view_type=VariablesListItemView):
    if False:
        while True:
            i = 10
    '\n    GUI components: ListView with a lineedit which works as a filter. One can write\n    a variable name in a edit box and possible matches are then shown in a listview.\n    '

    def update_completer_model():
        if False:
            while True:
                i = 10
        ' This gets called when the model for available attributes changes\n        through either drag/drop or the left/right button actions.\n\n        '
        nonlocal original_completer_items
        items = ['%s=%s' % item for v in model for item in v.attributes.items()]
        new = sorted(set(items))
        if new != original_completer_items:
            original_completer_items = new
            completer_model.setStringList(original_completer_items)

    def update_completer_prefix():
        if False:
            print('Hello World!')
        ' Prefixes all items in the completer model with the current\n        already done completion to enable the completion of multiple keywords.\n        '
        nonlocal original_completer_items
        prefix = str(completer.completionPrefix())
        if not prefix.endswith(' ') and ' ' in prefix:
            (prefix, _) = prefix.rsplit(' ', 1)
            items = [prefix + ' ' + item for item in original_completer_items]
        else:
            items = original_completer_items
        old = list(map(str, completer_model.stringList()))
        if set(old) != set(items):
            completer_model.setStringList(items)
    original_completer_items = []
    filter_edit = QLineEdit()
    filter_edit.setToolTip('Filter the list of available variables.')
    filter_edit.setPlaceholderText('Filter')
    completer_model = QStringListModel()
    completer = QCompleter(completer_model, filter_edit)
    completer.setCompletionMode(QCompleter.InlineCompletion)
    completer.setModelSorting(QCompleter.CaseSensitivelySortedModel)
    filter_edit.setCompleter(completer)
    completer_navigator = CompleterNavigator(parent)
    filter_edit.installEventFilter(completer_navigator)
    proxy = VariableFilterProxyModel()
    proxy.setSourceModel(model)
    view = view_type(acceptedType=accepted_type)
    view.setModel(proxy)
    model.dataChanged.connect(update_completer_model)
    model.rowsInserted.connect(update_completer_model)
    model.rowsRemoved.connect(update_completer_model)
    filter_edit.textChanged.connect(update_completer_prefix)
    filter_edit.textChanged.connect(proxy.set_filter_string)
    return (filter_edit, view)