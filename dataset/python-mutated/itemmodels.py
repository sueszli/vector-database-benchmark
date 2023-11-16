from numbers import Number, Integral
from math import isnan, isinf
import operator
from collections import namedtuple, defaultdict
from collections.abc import Sequence
from contextlib import contextmanager
from functools import reduce, partial, wraps
from itertools import chain
from warnings import warn
from xml.sax.saxutils import escape
from AnyQt.QtCore import Qt, QObject, QAbstractListModel, QModelIndex, QItemSelectionModel, QItemSelection
from AnyQt.QtCore import pyqtSignal as Signal
from AnyQt.QtGui import QColor, QBrush
from AnyQt.QtWidgets import QWidget, QBoxLayout, QToolButton, QAbstractButton, QAction
import numpy
from orangewidget.utils.itemmodels import PyListModel, AbstractSortTableModel as _AbstractSortTableModel, LabelledSeparator, SeparatorItem
from Orange.widgets.utils.colorpalettes import ContinuousPalettes, ContinuousPalette
from Orange.data import Value, Variable, Storage, DiscreteVariable, ContinuousVariable
from Orange.data.domain import filter_visible
from Orange.widgets import gui
from Orange.widgets.utils import datacaching
from Orange.statistics import basic_stats
from Orange.util import deprecated
__all__ = ['PyListModel', 'VariableListModel', 'PyListModelTooltip', 'DomainModel', 'AbstractSortTableModel', 'PyTableModel', 'TableModel', 'ModelActionsWidget', 'ListSingleSelectionModel']

@contextmanager
def signal_blocking(obj):
    if False:
        for i in range(10):
            print('nop')
    blocked = obj.signalsBlocked()
    obj.blockSignals(True)
    try:
        yield
    finally:
        obj.blockSignals(blocked)

def _as_contiguous_range(the_slice, length):
    if False:
        print('Hello World!')
    (start, stop, step) = the_slice.indices(length)
    if step == -1:
        (start, stop, step) = (stop + 1, start + 1, 1)
    elif not (step == 1 or step is None):
        raise IndexError('Non-contiguous range.')
    return (start, stop, step)

class AbstractSortTableModel(_AbstractSortTableModel):

    @deprecated('Orange.widgets.utils.itemmodels.AbstractSortTableModel.mapFromSourceRows')
    def mapFromTableRows(self, rows):
        if False:
            for i in range(10):
                print('nop')
        return self.mapFromSourceRows(rows)

    @deprecated('Orange.widgets.utils.itemmodels.AbstractSortTableModel.mapToSourceRows')
    def mapToTableRows(self, rows):
        if False:
            print('Hello World!')
        return self.mapToSourceRows(rows)

class PyTableModel(AbstractSortTableModel):
    """ A model for displaying python tables (sequences of sequences) in
    QTableView objects.

    Parameters
    ----------
    sequence : list
        The initial list to wrap.
    parent : QObject
        Parent QObject.
    editable: bool or sequence
        If True, all items are flagged editable. If sequence, the True-ish
        fields mark their respective columns editable.

    Notes
    -----
    The model rounds numbers to human readable precision, e.g.:
    1.23e-04, 1.234, 1234.5, 12345, 1.234e06.

    To set additional item roles, use setData().
    """

    @staticmethod
    def _RoleData():
        if False:
            i = 10
            return i + 15
        return defaultdict(lambda : defaultdict(dict))

    def __init__(self, sequence=None, parent=None, editable=False):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._rows = self._cols = 0
        self._headers = {}
        self._editable = editable
        self._table = None
        self._roleData = {}
        if sequence is None:
            sequence = []
        self.wrap(sequence)

    def rowCount(self, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        return 0 if parent.isValid() else self._rows

    def columnCount(self, parent=QModelIndex()):
        if False:
            print('Hello World!')
        return 0 if parent.isValid() else self._cols

    def flags(self, index):
        if False:
            i = 10
            return i + 15
        flags = super().flags(index)
        if not self._editable or not index.isValid():
            return flags
        if isinstance(self._editable, Sequence):
            return flags | Qt.ItemIsEditable if self._editable[index.column()] else flags
        return flags | Qt.ItemIsEditable

    def setData(self, index, value, role):
        if False:
            while True:
                i = 10
        row = self.mapFromSourceRows(index.row())
        if role == Qt.EditRole:
            self[row][index.column()] = value
            self.dataChanged.emit(index, index)
        else:
            self._roleData[row][index.column()][role] = value
        return True

    def data(self, index, role=Qt.DisplayRole):
        if False:
            while True:
                i = 10
        if not index.isValid():
            return
        (row, column) = (self.mapToSourceRows(index.row()), index.column())
        role_value = self._roleData.get(row, {}).get(column, {}).get(role)
        if role_value is not None:
            return role_value
        try:
            value = self[row][column]
        except IndexError:
            return
        if role == Qt.EditRole:
            return value
        if role == Qt.DecorationRole and isinstance(value, Variable):
            return gui.attributeIconDict[value]
        if role == Qt.DisplayRole:
            if isinstance(value, Number) and (not (isnan(value) or isinf(value) or isinstance(value, Integral))):
                absval = abs(value)
                strlen = len(str(int(absval)))
                value = '{:.{}{}}'.format(value, 2 if absval < 0.001 else 3 if strlen < 2 else 1 if strlen < 5 else 0 if strlen < 6 else 3, 'f' if absval == 0 or (absval >= 0.001 and strlen < 6) else 'e')
            return str(value)
        if role == Qt.TextAlignmentRole and isinstance(value, Number):
            return Qt.AlignRight | Qt.AlignVCenter
        if role == Qt.ToolTipRole:
            return str(value)

    def sortColumnData(self, column):
        if False:
            print('Hello World!')
        return [row[column] for row in self._table]

    def setHorizontalHeaderLabels(self, labels):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        labels : list of str or list of Variable\n        '
        self._headers[Qt.Horizontal] = tuple(labels)

    def setVerticalHeaderLabels(self, labels):
        if False:
            i = 10
            return i + 15
        '\n        Parameters\n        ----------\n        labels : list of str or list of Variable\n        '
        self._headers[Qt.Vertical] = tuple(labels)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if False:
            for i in range(10):
                print('nop')
        headers = self._headers.get(orientation)
        if headers and section < len(headers):
            section = self.mapToSourceRows(section) if orientation == Qt.Vertical else section
            value = headers[section]
            if role == Qt.ToolTipRole:
                role = Qt.DisplayRole
            if role == Qt.DisplayRole:
                return value.name if isinstance(value, Variable) else value
            if role == Qt.DecorationRole:
                if isinstance(value, Variable):
                    return gui.attributeIconDict[value]
        return super().headerData(section, orientation, role)

    def removeRows(self, row, count, parent=QModelIndex()):
        if False:
            return 10
        if not parent.isValid():
            del self[row:row + count]
            for rowidx in range(row, row + count):
                self._roleData.pop(rowidx, None)
            self._rows = self._table_dim()[0]
            return True
        return False

    def removeColumns(self, column, count, parent=QModelIndex()):
        if False:
            i = 10
            return i + 15
        self.beginRemoveColumns(parent, column, column + count - 1)
        for row in self._table:
            del row[column:column + count]
        for cols in self._roleData.values():
            for col in range(column, column + count):
                cols.pop(col, None)
        del self._headers.get(Qt.Horizontal, [])[column:column + count]
        self._cols = self._table_dim()[1]
        self.endRemoveColumns()
        return True

    def _table_dim(self):
        if False:
            while True:
                i = 10
        return (len(self._table), max(map(len, self), default=0))

    def insertRows(self, row, count, parent=QModelIndex()):
        if False:
            return 10
        self.beginInsertRows(parent, row, row + count - 1)
        self._table[row:row] = [[''] * self.columnCount() for _ in range(count)]
        self._rows = self._table_dim()[0]
        self.endInsertRows()
        return True

    def insertColumns(self, column, count, parent=QModelIndex()):
        if False:
            print('Hello World!')
        self.beginInsertColumns(parent, column, column + count - 1)
        for row in self._table:
            row[column:column] = [''] * count
        self._cols = self._table_dim()[1]
        self.endInsertColumns()
        return True

    def __len__(self):
        if False:
            i = 10
            return i + 15
        return len(self._table)

    def __bool__(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self) != 0

    def __iter__(self):
        if False:
            print('Hello World!')
        return iter(self._table)

    def __getitem__(self, item):
        if False:
            for i in range(10):
                print('nop')
        return self._table[item]

    def __delitem__(self, i):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(i, slice):
            (start, stop, _) = _as_contiguous_range(i, len(self))
            stop -= 1
        else:
            start = stop = i = i if i >= 0 else len(self) + i
        if stop < start:
            return
        self._check_sort_order()
        self.beginRemoveRows(QModelIndex(), start, stop)
        del self._table[i]
        rows = self._table_dim()[0]
        self._rows = rows
        self.endRemoveRows()
        self._update_column_count()

    def __setitem__(self, i, value):
        if False:
            for i in range(10):
                print('nop')
        self._check_sort_order()
        if isinstance(i, slice):
            (start, stop, _) = _as_contiguous_range(i, len(self))
            self.removeRows(start, stop - start)
            if len(value) == 0:
                return
            self.beginInsertRows(QModelIndex(), start, start + len(value) - 1)
            self._table[start:start] = value
            self._rows = self._table_dim()[0]
            self.endInsertRows()
            self._update_column_count()
        else:
            self._table[i] = value
            self.dataChanged.emit(self.index(i, 0), self.index(i, self.columnCount() - 1))

    def _update_column_count(self):
        if False:
            for i in range(10):
                print('nop')
        cols_before = self._cols
        cols_after = self._table_dim()[1]
        if cols_before < cols_after:
            self.beginInsertColumns(QModelIndex(), cols_before, cols_after - 1)
            self._cols = cols_after
            self.endInsertColumns()
        elif cols_before > cols_after:
            self.beginRemoveColumns(QModelIndex(), cols_after, cols_before - 1)
            self._cols = cols_after
            self.endRemoveColumns()

    def _check_sort_order(self):
        if False:
            while True:
                i = 10
        if self.mapToSourceRows(Ellipsis) is not Ellipsis:
            warn("Can't modify PyTableModel when it's sorted", RuntimeWarning, stacklevel=3)
            raise RuntimeError("Can't modify PyTableModel when it's sorted")

    def wrap(self, table):
        if False:
            i = 10
            return i + 15
        self.beginResetModel()
        self._table = table
        self._roleData = self._RoleData()
        (self._rows, self._cols) = self._table_dim()
        self.resetSorting()
        self.endResetModel()

    def tolist(self):
        if False:
            for i in range(10):
                print('nop')
        return self._table

    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        self.beginResetModel()
        self._table.clear()
        self.resetSorting()
        self._roleData.clear()
        (self._rows, self._cols) = self._table_dim()
        self.endResetModel()

    def append(self, row):
        if False:
            return 10
        self.extend([row])

    def _insertColumns(self, rows):
        if False:
            print('Hello World!')
        n_max = max(map(len, rows))
        if self.columnCount() < n_max:
            self.insertColumns(self.columnCount(), n_max - self.columnCount())

    def extend(self, rows):
        if False:
            for i in range(10):
                print('nop')
        (i, rows) = (len(self), list(rows))
        self.insertRows(i, len(rows))
        self._insertColumns(rows)
        self[i:] = rows

    def insert(self, i, row):
        if False:
            return 10
        self.insertRows(i, 1)
        self._insertColumns((row,))
        self[i] = row

    def remove(self, val):
        if False:
            return 10
        del self[self._table.index(val)]

class PyListModelTooltip(PyListModel):

    def __init__(self, iterable=None, tooltips=(), **kwargs):
        if False:
            while True:
                i = 10
        super().__init__(iterable, **kwargs)
        if not isinstance(tooltips, Sequence):
            tooltips = list(tooltips)
        self.tooltips = tooltips

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10
        if role == Qt.ToolTipRole:
            if index.row() >= len(self.tooltips):
                return None
            return self.tooltips[index.row()]
        else:
            return super().data(index, role)

class VariableListModel(PyListModel):
    MIME_TYPE = 'application/x-Orange-VariableList'

    def __init__(self, *args, placeholder=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(*args, **kwargs)
        self.placeholder = placeholder

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10
        if self._is_index_valid(index):
            var = self[index.row()]
            if var is None and role == Qt.DisplayRole:
                return self.placeholder or 'None'
            if not isinstance(var, Variable):
                return super().data(index, role)
            elif role == Qt.DisplayRole:
                return var.name
            elif role == Qt.DecorationRole:
                return gui.attributeIconDict[var]
            elif role == Qt.ToolTipRole:
                return self.variable_tooltip(var)
            elif role == gui.TableVariable:
                return var
            else:
                return PyListModel.data(self, index, role)

    def variable_tooltip(self, var):
        if False:
            print('Hello World!')
        if var.is_discrete:
            return self.discrete_variable_tooltip(var)
        elif var.is_time:
            return self.time_variable_toltip(var)
        elif var.is_continuous:
            return self.continuous_variable_toltip(var)
        elif var.is_string:
            return self.string_variable_tooltip(var)

    def variable_labels_tooltip(self, var):
        if False:
            print('Hello World!')
        text = ''
        if var.attributes:
            items = [(safe_text(key), safe_text(value)) for (key, value) in var.attributes.items()]
            labels = list(map('%s = %s'.__mod__, items))
            text += '<br/>Variable Labels:<br/>'
            text += '<br/>'.join(labels)
        return text

    def discrete_variable_tooltip(self, var):
        if False:
            return 10
        text = '<b>%s</b><br/>Categorical with %i values: ' % (safe_text(var.name), len(var.values))
        text += ', '.join(('%r' % safe_text(v) for v in var.values))
        text += self.variable_labels_tooltip(var)
        return text

    def time_variable_toltip(self, var):
        if False:
            return 10
        text = '<b>%s</b><br/>Time' % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

    def continuous_variable_toltip(self, var):
        if False:
            i = 10
            return i + 15
        text = '<b>%s</b><br/>Numeric' % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

    def string_variable_tooltip(self, var):
        if False:
            print('Hello World!')
        text = '<b>%s</b><br/>Text' % safe_text(var.name)
        text += self.variable_labels_tooltip(var)
        return text

class DomainModel(VariableListModel):
    (ATTRIBUTES, CLASSES, METAS) = (1, 2, 4)
    MIXED = ATTRIBUTES | CLASSES | METAS
    SEPARATED = (CLASSES, PyListModel.Separator, METAS, PyListModel.Separator, ATTRIBUTES)
    PRIMITIVE = (DiscreteVariable, ContinuousVariable)

    def __init__(self, order=SEPARATED, separators=True, placeholder=None, valid_types=None, alphabetical=False, skip_hidden_vars=True, *, strict_type=False, **kwargs):
        if False:
            while True:
                i = 10
        '\n\n        Parameters\n        ----------\n        order: tuple or int\n            Order of attributes, metas, classes, separators and other options\n        separators: bool\n            If False, remove separators from `order`.\n        placeholder: str\n            The text that is shown when no variable is selected\n        valid_types: tuple\n            (Sub)types of `Variable` that are included in the model\n        alphabetical: bool\n            If True, variables are sorted alphabetically.\n        skip_hidden_vars: bool\n            If True, variables marked as "hidden" are skipped.\n        strict_type: bool\n            If True, variable must be one of specified valid_types and not a\n            derived type (i.e. TimeVariable is not accepted as\n            ContinuousVariable)\n        '
        super().__init__(placeholder=placeholder, **kwargs)
        if isinstance(order, int):
            order = (order,)
        if placeholder is not None and None not in order:
            order = (None,) + (self.Separator,) * (self.Separator in order) + order
        if not separators:
            order = [e for e in order if not isinstance(e, SeparatorItem)]
        self.order = order
        self.valid_types = valid_types
        self.strict_type = strict_type
        self.alphabetical = alphabetical
        self.skip_hidden_vars = skip_hidden_vars
        self._within_set_domain = False
        self.set_domain(None)

    def set_domain(self, domain):
        if False:
            return 10
        self.beginResetModel()
        content = []
        add_separator = None
        for section in self.order:
            if isinstance(section, SeparatorItem):
                add_separator = section
                continue
            if isinstance(section, int):
                if domain is None:
                    continue
                to_add = list(chain(*(vars for (i, vars) in enumerate((domain.attributes, domain.class_vars, domain.metas)) if 1 << i & section)))
                if self.skip_hidden_vars:
                    to_add = list(filter_visible(to_add))
                if self.valid_types is not None:
                    to_add = [var for var in to_add if (type(var) in self.valid_types if self.strict_type else isinstance(var, self.valid_types))]
                if self.alphabetical:
                    to_add = sorted(to_add, key=lambda x: x.name)
            elif isinstance(section, list):
                to_add = section
            else:
                to_add = [section]
            if to_add:
                if add_separator and (content or isinstance(add_separator, LabelledSeparator)):
                    content.append(add_separator)
                    add_separator = None
                content += to_add
        try:
            self._within_set_domain = True
            self[:] = content
        finally:
            self._within_set_domain = False
        self.endResetModel()

    def prevent_modification(method):
        if False:
            print('Hello World!')

        @wraps(method)
        def e(self, *args, **kwargs):
            if False:
                return 10
            if self._within_set_domain:
                method(self, *args, **kwargs)
            else:
                raise TypeError("{} can be modified only by calling 'set_domain'".format(type(self).__name__))
        return e

    @prevent_modification
    def extend(self, iterable):
        if False:
            for i in range(10):
                print('nop')
        return super().extend(iterable)

    @prevent_modification
    def append(self, item):
        if False:
            print('Hello World!')
        return super().append(item)

    @prevent_modification
    def insert(self, i, val):
        if False:
            while True:
                i = 10
        return super().insert(i, val)

    @prevent_modification
    def remove(self, val):
        if False:
            for i in range(10):
                print('nop')
        return super().remove(val)

    @prevent_modification
    def pop(self, i):
        if False:
            i = 10
            return i + 15
        return super().pop(i)

    @prevent_modification
    def clear(self):
        if False:
            for i in range(10):
                print('nop')
        return super().clear()

    @prevent_modification
    def __delitem__(self, s):
        if False:
            print('Hello World!')
        return super().__delitem__(s)

    @prevent_modification
    def __setitem__(self, s, value):
        if False:
            print('Hello World!')
        return super().__setitem__(s, value)

    @prevent_modification
    def reverse(self):
        if False:
            print('Hello World!')
        return super().reverse()

    @prevent_modification
    def sort(self, *args, **kwargs):
        if False:
            i = 10
            return i + 15
        return super().sort(*args, **kwargs)

    def setData(self, index, value, role=Qt.EditRole):
        if False:
            return 10
        if role == Qt.EditRole:
            return False
        else:
            return super().setData(index, value, role)

    def setItemData(self, index, data):
        if False:
            i = 10
            return i + 15
        if Qt.EditRole in data:
            return False
        else:
            return super().setItemData(index, data)

    def insertRows(self, row, count, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        return False

    def removeRows(self, row, count, parent=QModelIndex()):
        if False:
            for i in range(10):
                print('nop')
        return False
_html_replace = [('<', '&lt;'), ('>', '&gt;')]

def safe_text(text):
    if False:
        i = 10
        return i + 15
    for (old, new) in _html_replace:
        text = str(text).replace(old, new)
    return text

class ContinuousPalettesModel(QAbstractListModel):
    """
    Model for combo boxes
    """
    KeyRole = Qt.UserRole + 1

    def __init__(self, parent=None, categories=None, icon_width=64):
        if False:
            return 10
        super().__init__(parent)
        self.icon_width = icon_width
        palettes = list(ContinuousPalettes.values())
        if categories is None:
            categories = dict.fromkeys((palette.category for palette in palettes))
        self.items = []
        for category in categories:
            self.items.append(category)
            self.items += [palette for palette in palettes if palette.category == category]
        if len(categories) == 1:
            del self.items[0]

    def rowCount(self, parent):
        if False:
            for i in range(10):
                print('nop')
        return 0 if parent.isValid() else len(self.items)

    @staticmethod
    def columnCount(parent):
        if False:
            return 10
        return 0 if parent.isValid() else 1

    def data(self, index, role):
        if False:
            while True:
                i = 10
        item = self.items[index.row()]
        if isinstance(item, str):
            if role in [Qt.EditRole, Qt.DisplayRole]:
                return item
        else:
            if role in [Qt.EditRole, Qt.DisplayRole]:
                return item.friendly_name
            if role == Qt.DecorationRole:
                return item.color_strip(self.icon_width, 16)
            if role == Qt.UserRole:
                return item
            if role == self.KeyRole:
                return item.name
        return None

    def flags(self, index):
        if False:
            for i in range(10):
                print('nop')
        item = self.items[index.row()]
        if isinstance(item, ContinuousPalette):
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable
        else:
            return Qt.NoItemFlags

    def indexOf(self, x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, str):
            for (i, item) in enumerate(self.items):
                if not isinstance(item, str) and x in (item.name, item.friendly_name):
                    return i
        elif isinstance(x, ContinuousPalette):
            return self.items.index(x)
        return None

class ListSingleSelectionModel(QItemSelectionModel):
    """ Item selection model for list item models with single selection.

    Defines signal:
        - selectedIndexChanged(QModelIndex)

    """
    selectedIndexChanged = Signal(QModelIndex)

    def __init__(self, model, parent=None):
        if False:
            for i in range(10):
                print('nop')
        QItemSelectionModel.__init__(self, model, parent)
        self.selectionChanged.connect(self.onSelectionChanged)

    def onSelectionChanged(self, new, _):
        if False:
            for i in range(10):
                print('nop')
        index = list(new.indexes())
        if index:
            index = index.pop()
        else:
            index = QModelIndex()
        self.selectedIndexChanged.emit(index)

    def selectedRow(self):
        if False:
            while True:
                i = 10
        ' Return QModelIndex of the selected row or invalid if no selection.\n        '
        rows = self.selectedRows()
        if rows:
            return rows[0]
        else:
            return QModelIndex()

    def select(self, index, flags=QItemSelectionModel.ClearAndSelect):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(index, int):
            index = self.model().index(index)
        return QItemSelectionModel.select(self, index, flags)

def select_row(view, row):
    if False:
        print('Hello World!')
    '\n    Select a `row` in an item view.\n    '
    selmodel = view.selectionModel()
    selmodel.select(view.model().index(row, 0), QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)

def select_rows(view, row_indices, command=QItemSelectionModel.ClearAndSelect):
    if False:
        i = 10
        return i + 15
    '\n    Select several rows in view.\n\n    :param QAbstractItemView view:\n    :param row_indices: Integer indices of rows to select.\n    :param command: QItemSelectionModel.SelectionFlags\n    '
    selmodel = view.selectionModel()
    model = view.model()
    selection = QItemSelection()
    for row in row_indices:
        index = model.index(row, 0)
        selection.select(index, index)
    selmodel.select(selection, command | QItemSelectionModel.Rows)

class ModelActionsWidget(QWidget):

    def __init__(self, actions=None, parent=None, direction=QBoxLayout.LeftToRight):
        if False:
            for i in range(10):
                print('nop')
        QWidget.__init__(self, parent)
        self.actions = []
        self.buttons = []
        layout = QBoxLayout(direction)
        layout.setContentsMargins(0, 0, 0, 0)
        self.setContentsMargins(0, 0, 0, 0)
        self.setLayout(layout)
        if actions is not None:
            for action in actions:
                self.addAction(action)
        self.setLayout(layout)

    def actionButton(self, action):
        if False:
            return 10
        if isinstance(action, QAction):
            button = QToolButton(self)
            button.setDefaultAction(action)
            return button
        elif isinstance(action, QAbstractButton):
            return action

    def insertAction(self, ind, action, *args):
        if False:
            for i in range(10):
                print('nop')
        button = self.actionButton(action)
        self.layout().insertWidget(ind, button, *args)
        self.buttons.insert(ind, button)
        self.actions.insert(ind, action)
        return button

    def addAction(self, action, *args):
        if False:
            return 10
        return self.insertAction(-1, action, *args)

class TableModel(AbstractSortTableModel):
    """
    An adapter for using Orange.data.Table within Qt's Item View Framework.

    :param Orange.data.Table sourcedata: Source data table.
    :param QObject parent:
    """
    ValueRole = gui.TableValueRole
    ClassValueRole = gui.TableClassValueRole
    VariableRole = gui.TableVariable
    VariableStatsRole = next(gui.OrangeUserRole)
    DomainRole = next(gui.OrangeUserRole)
    (ClassVar, Meta, Attribute) = range(3)
    ColorForRole = {ClassVar: QColor(160, 160, 160), Meta: QColor(220, 220, 200), Attribute: None}
    Column = namedtuple('Column', ['var', 'role', 'background', 'format'])
    Basket = namedtuple('Basket', ['vars', 'role', 'background', 'density', 'format'])

    def __init__(self, sourcedata, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.source = sourcedata
        self.domain = domain = sourcedata.domain
        self.X_density = sourcedata.X_density()
        self.Y_density = sourcedata.Y_density()
        self.M_density = sourcedata.metas_density()
        brush_for_role = {role: QBrush(c) if c is not None else None for (role, c) in self.ColorForRole.items()}

        def format_sparse(vars, row):
            if False:
                for i in range(10):
                    print('nop')
            row = row.tocsr()
            return ', '.join(('{}={}'.format(vars[i].name, vars[i].str_val(v)) for (i, v) in zip(row.indices, row.data)))

        def format_sparse_bool(vars, row):
            if False:
                while True:
                    i = 10
            row = row.tocsr()
            return ', '.join((vars[i].name for i in row.indices))

        def format_dense(var, val):
            if False:
                while True:
                    i = 10
            return var.str_val(val)

        def make_basket_formatter(vars, density):
            if False:
                i = 10
                return i + 15
            formatter = format_sparse if density == Storage.SPARSE else format_sparse_bool
            return partial(formatter, vars)

        def make_basket(vars, density, role):
            if False:
                for i in range(10):
                    print('nop')
            return TableModel.Basket(vars, role, brush_for_role[role], density, make_basket_formatter(vars, density))

        def make_column(var, role):
            if False:
                print('Hello World!')
            return TableModel.Column(var, role, brush_for_role[role], partial(format_dense, var))
        columns = []
        if self.Y_density != Storage.DENSE and domain.class_vars:
            coldesc = make_basket(domain.class_vars, self.Y_density, TableModel.ClassVar)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.ClassVar) for var in domain.class_vars]
        if self.M_density != Storage.DENSE and domain.metas:
            coldesc = make_basket(domain.metas, self.M_density, TableModel.Meta)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.Meta) for var in domain.metas]
        if self.X_density != Storage.DENSE and domain.attributes:
            coldesc = make_basket(domain.attributes, self.X_density, TableModel.Attribute)
            columns.append(coldesc)
        else:
            columns += [make_column(var, TableModel.Attribute) for var in domain.attributes]
        self.vars = domain.class_vars + domain.metas + domain.attributes
        self.columns = columns
        self._labels = sorted(reduce(operator.ior, [set(var.attributes) for var in self.vars], set()))
        self.__stats = None
        self.__rowCount = sourcedata.approx_len()
        self.__columnCount = len(self.columns)
        if self.__rowCount > 2 ** 31 - 1:
            raise ValueError('len(sourcedata) > 2 ** 31 - 1')

    def _get_source_item(self, row, coldesc):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(coldesc, self.Basket):
            if coldesc.role is self.Meta:
                return self.source[row:row + 1].metas
            if coldesc.role is self.Attribute:
                return self.source[row:row + 1].X
        return self.source[row, coldesc.var]

    def sortColumnData(self, column):
        if False:
            while True:
                i = 10
        return self._columnSortKeyData(column, TableModel.ValueRole)

    @deprecated('Orange.widgets.utils.itemmodels.TableModel.sortColumnData')
    def columnSortKeyData(self, column, role):
        if False:
            for i in range(10):
                print('nop')
        return self._columnSortKeyData(column, role)

    def _columnSortKeyData(self, column, role):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return a sequence of source table objects which can be used as\n        `keys` for sorting.\n\n        :param int column: Sort column.\n        :param Qt.ItemRole role: Sort item role.\n\n        '
        coldesc = self.columns[column]
        if isinstance(coldesc, TableModel.Column) and role == TableModel.ValueRole:
            return self.source.get_column(coldesc.var)
        else:
            return numpy.asarray([self.index(i, column).data(role) for i in range(self.rowCount())])

    def data(self, index, role, _str=str, _Qt_DisplayRole=Qt.DisplayRole, _Qt_EditRole=Qt.EditRole, _Qt_BackgroundRole=Qt.BackgroundRole, _Qt_ForegroundRole=Qt.ForegroundRole, _ValueRole=ValueRole, _ClassValueRole=ClassValueRole, _VariableRole=VariableRole, _DomainRole=DomainRole, _VariableStatsRole=VariableStatsRole, _recognizedRoles=frozenset([Qt.DisplayRole, Qt.EditRole, Qt.BackgroundRole, Qt.ForegroundRole, ValueRole, ClassValueRole, VariableRole, DomainRole, VariableStatsRole])):
        if False:
            while True:
                i = 10
        '\n        Reimplemented from `QAbstractItemModel.data`\n        '
        if role not in _recognizedRoles:
            return None
        row = index.row()
        if not 0 <= row <= self.__rowCount:
            return None
        row = self.mapToSourceRows(row)
        col = 0 if role is _ClassValueRole else index.column()
        try:
            coldesc = self.columns[col]
            instance = self._get_source_item(row, coldesc)
        except IndexError:
            self.layoutAboutToBeChanged.emit()
            self.beginRemoveRows(self.parent(), row, max(self.rowCount(), row))
            self.__rowCount = min(row, self.__rowCount)
            self.endRemoveRows()
            self.layoutChanged.emit()
            return None
        if role == _Qt_DisplayRole:
            return coldesc.format(instance)
        elif role in (_Qt_EditRole, _ValueRole) and isinstance(coldesc, TableModel.Column):
            return Value(coldesc.var, instance)
        elif role == _Qt_BackgroundRole:
            return coldesc.background
        elif role == _Qt_ForegroundRole:
            return coldesc.background and QColor(0, 0, 0, 200)
        elif role == _ClassValueRole and isinstance(coldesc, TableModel.Column) and (len(self.domain.class_vars) == 1):
            return Value(coldesc.var, instance)
        elif role == _VariableRole and isinstance(coldesc, TableModel.Column):
            return coldesc.var
        elif role == _DomainRole:
            return coldesc.role
        elif role == _VariableStatsRole:
            return self._stats_for_column(col)
        else:
            return None

    def setData(self, index, value, role):
        if False:
            print('Hello World!')
        row = self.mapFromSourceRows(index.row())
        if role == Qt.EditRole:
            try:
                self.source[row, index.column()] = value
            except (TypeError, IndexError):
                return False
            else:
                self.dataChanged.emit(index, index)
                return True
        else:
            return False

    def parent(self, index=QModelIndex()):
        if False:
            i = 10
            return i + 15
        'Reimplemented from `QAbstractTableModel.parent`.'
        return QModelIndex()

    def rowCount(self, parent=QModelIndex()):
        if False:
            return 10
        'Reimplemented from `QAbstractTableModel.rowCount`.'
        return 0 if parent.isValid() else self.__rowCount

    def columnCount(self, parent=QModelIndex()):
        if False:
            return 10
        'Reimplemented from `QAbstractTableModel.columnCount`.'
        return 0 if parent.isValid() else self.__columnCount

    def headerData(self, section, orientation, role):
        if False:
            return 10
        'Reimplemented from `QAbstractTableModel.headerData`.'
        if orientation == Qt.Vertical:
            if role == Qt.DisplayRole:
                return int(self.mapToSourceRows(section) + 1)
            return None
        coldesc = self.columns[section]
        if role == Qt.DisplayRole:
            if isinstance(coldesc, TableModel.Basket):
                return '{...}'
            else:
                return coldesc.var.name
        elif role == Qt.ToolTipRole:
            return self._tooltip(coldesc)
        elif role == TableModel.VariableRole and isinstance(coldesc, TableModel.Column):
            return coldesc.var
        elif role == TableModel.VariableStatsRole:
            return self._stats_for_column(section)
        elif role == TableModel.DomainRole:
            return coldesc.role
        else:
            return None

    def _tooltip(self, coldesc):
        if False:
            print('Hello World!')
        '\n        Return an header tool tip text for an `column` descriptor.\n        '
        if isinstance(coldesc, TableModel.Basket):
            return None
        labels = self._labels
        variable = coldesc.var
        pairs = [(escape(key), escape(str(variable.attributes[key]))) for key in labels if key in variable.attributes]
        tip = '<b>%s</b>' % escape(variable.name)
        tip = '<br/>'.join([tip] + ['%s = %s' % pair for pair in pairs])
        return tip

    def _stats_for_column(self, column):
        if False:
            i = 10
            return i + 15
        '\n        Return BasicStats for `column` index.\n        '
        coldesc = self.columns[column]
        if isinstance(coldesc, TableModel.Basket):
            return None
        if self.__stats is None:
            self.__stats = datacaching.getCached(self.source, basic_stats.DomainBasicStats, (self.source, True))
        return self.__stats[coldesc.var]