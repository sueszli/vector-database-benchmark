"""
NumPy Array Editor Dialog based on Qt
"""
import io
from qtpy.compat import from_qvariant, to_qvariant
from qtpy.QtCore import QAbstractTableModel, QItemSelection, QLocale, QItemSelectionRange, QModelIndex, Qt, Slot
from qtpy.QtGui import QColor, QCursor, QDoubleValidator, QKeySequence
from qtpy.QtWidgets import QAbstractItemDelegate, QApplication, QComboBox, QDialog, QGridLayout, QHBoxLayout, QInputDialog, QItemDelegate, QLabel, QLineEdit, QMenu, QMessageBox, QPushButton, QSpinBox, QStackedWidget, QTableView, QVBoxLayout, QWidget
from spyder_kernels.utils.nsview import value_to_display
from spyder_kernels.utils.lazymodules import numpy as np
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.api.widgets.mixins import SpyderWidgetMixin
from spyder.api.widgets.toolbars import SpyderToolbar
from spyder.config.base import _
from spyder.config.manager import CONF
from spyder.plugins.variableexplorer.widgets.basedialog import BaseDialog
from spyder.py3compat import is_binary_string, is_string, is_text_string, to_binary_string, to_text_string
from spyder.utils.icon_manager import ima
from spyder.utils.qthelpers import add_actions, create_action, keybinding
from spyder.utils.stylesheet import PANES_TOOLBAR_STYLESHEET

class ArrayEditorActions:
    Copy = 'copy_action'
    Edit = 'edit_action'
    Format = 'format_action'
    Resize = 'resize_action'
    ToggleBackgroundColor = 'toggle_background_color_action'
SUPPORTED_FORMATS = {'single': '.6g', 'double': '.6g', 'float_': '.6g', 'longfloat': '.6g', 'float16': '.6g', 'float32': '.6g', 'float64': '.6g', 'float96': '.6g', 'float128': '.6g', 'csingle': '.6g', 'complex_': '.6g', 'clongfloat': '.6g', 'complex64': '.6g', 'complex128': '.6g', 'complex192': '.6g', 'complex256': '.6g', 'byte': 'd', 'bytes8': 's', 'short': 'd', 'intc': 'd', 'int_': 'd', 'longlong': 'd', 'intp': 'd', 'int8': 'd', 'int16': 'd', 'int32': 'd', 'int64': 'd', 'ubyte': 'd', 'ushort': 'd', 'uintc': 'd', 'uint': 'd', 'ulonglong': 'd', 'uintp': 'd', 'uint8': 'd', 'uint16': 'd', 'uint32': 'd', 'uint64': 'd', 'bool_': '', 'bool8': '', 'bool': ''}
LARGE_SIZE = 500000.0
LARGE_NROWS = 100000.0
LARGE_COLS = 60

def is_float(dtype):
    if False:
        while True:
            i = 10
    'Return True if datatype dtype is a float kind'
    return 'float' in dtype.name or dtype.name in ['single', 'double']

def is_number(dtype):
    if False:
        print('Hello World!')
    'Return True is datatype dtype is a number kind'
    return is_float(dtype) or 'int' in dtype.name or 'long' in dtype.name or ('short' in dtype.name)

def get_idx_rect(index_list):
    if False:
        i = 10
        return i + 15
    'Extract the boundaries from a list of indexes'
    (rows, cols) = list(zip(*[(i.row(), i.column()) for i in index_list]))
    return (min(rows), max(rows), min(cols), max(cols))

class ArrayModel(QAbstractTableModel, SpyderFontsMixin):
    """Array Editor Table Model"""
    ROWS_TO_LOAD = 500
    COLS_TO_LOAD = 40

    def __init__(self, data, format_spec='.6g', xlabels=None, ylabels=None, readonly=False, parent=None):
        if False:
            return 10
        QAbstractTableModel.__init__(self)
        self.dialog = parent
        self.changes = {}
        self.xlabels = xlabels
        self.ylabels = ylabels
        self.readonly = readonly
        self.test_array = np.array([0], dtype=data.dtype)
        if data.dtype in (np.complex64, np.complex128):
            self.color_func = np.abs
        else:
            self.color_func = np.real
        huerange = [0.66, 0.99]
        self.sat = 0.7
        self.val = 1.0
        self.alp = 0.6
        self._data = data
        self._format_spec = format_spec
        self.total_rows = self._data.shape[0]
        self.total_cols = self._data.shape[1]
        size = self.total_rows * self.total_cols
        if not self._data.dtype.name == 'object':
            try:
                self.vmin = np.nanmin(self.color_func(data))
                self.vmax = np.nanmax(self.color_func(data))
                if self.vmax == self.vmin:
                    self.vmin -= 1
                self.hue0 = huerange[0]
                self.dhue = huerange[1] - huerange[0]
                self.bgcolor_enabled = True
            except (AttributeError, TypeError, ValueError):
                self.vmin = None
                self.vmax = None
                self.hue0 = None
                self.dhue = None
                self.bgcolor_enabled = False
        self.has_inf = False
        if data.dtype.kind in ['f', 'c']:
            self.has_inf = np.any(np.isinf(data))
        if self._data.dtype.name == 'object' or self.has_inf:
            self.bgcolor_enabled = False
        if size > LARGE_SIZE:
            self.rows_loaded = self.ROWS_TO_LOAD
            self.cols_loaded = self.COLS_TO_LOAD
        else:
            if self.total_rows > LARGE_NROWS:
                self.rows_loaded = self.ROWS_TO_LOAD
            else:
                self.rows_loaded = self.total_rows
            if self.total_cols > LARGE_COLS:
                self.cols_loaded = self.COLS_TO_LOAD
            else:
                self.cols_loaded = self.total_cols

    def get_format_spec(self):
        if False:
            while True:
                i = 10
        'Return current format'
        return self._format_spec

    def get_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Return data'
        return self._data

    def set_format_spec(self, format_spec):
        if False:
            print('Hello World!')
        'Change display format'
        self._format_spec = format_spec
        self.reset()

    def columnCount(self, qindex=QModelIndex()):
        if False:
            for i in range(10):
                print('nop')
        'Array column number'
        if self.total_cols <= self.cols_loaded:
            return self.total_cols
        else:
            return self.cols_loaded

    def rowCount(self, qindex=QModelIndex()):
        if False:
            for i in range(10):
                print('nop')
        'Array row number'
        if self.total_rows <= self.rows_loaded:
            return self.total_rows
        else:
            return self.rows_loaded

    def can_fetch_more(self, rows=False, columns=False):
        if False:
            print('Hello World!')
        if rows:
            if self.total_rows > self.rows_loaded:
                return True
            else:
                return False
        if columns:
            if self.total_cols > self.cols_loaded:
                return True
            else:
                return False

    def fetch_more(self, rows=False, columns=False):
        if False:
            while True:
                i = 10
        if self.can_fetch_more(rows=rows):
            reminder = self.total_rows - self.rows_loaded
            items_to_fetch = min(reminder, self.ROWS_TO_LOAD)
            self.beginInsertRows(QModelIndex(), self.rows_loaded, self.rows_loaded + items_to_fetch - 1)
            self.rows_loaded += items_to_fetch
            self.endInsertRows()
        if self.can_fetch_more(columns=columns):
            reminder = self.total_cols - self.cols_loaded
            items_to_fetch = min(reminder, self.COLS_TO_LOAD)
            self.beginInsertColumns(QModelIndex(), self.cols_loaded, self.cols_loaded + items_to_fetch - 1)
            self.cols_loaded += items_to_fetch
            self.endInsertColumns()

    def bgcolor(self, state):
        if False:
            while True:
                i = 10
        'Toggle backgroundcolor'
        self.bgcolor_enabled = state > 0
        self.reset()

    def get_value(self, index):
        if False:
            while True:
                i = 10
        i = index.row()
        j = index.column()
        if len(self._data.shape) == 1:
            value = self._data[j]
        else:
            value = self._data[i, j]
        return self.changes.get((i, j), value)

    def data(self, index, role=Qt.DisplayRole):
        if False:
            print('Hello World!')
        'Cell content.'
        if not index.isValid():
            return to_qvariant()
        value = self.get_value(index)
        dtn = self._data.dtype.name
        if is_binary_string(value):
            try:
                value = to_text_string(value, 'utf8')
            except Exception:
                pass
        if role == Qt.DisplayRole:
            if value is np.ma.masked:
                return ''
            elif dtn == 'object':
                return value_to_display(value)
            else:
                try:
                    format_spec = self._format_spec
                    return to_qvariant(format(value, format_spec))
                except TypeError:
                    self.readonly = True
                    return repr(value)
        elif role == Qt.TextAlignmentRole:
            return to_qvariant(int(Qt.AlignCenter | Qt.AlignVCenter))
        elif role == Qt.BackgroundColorRole and self.bgcolor_enabled and (value is not np.ma.masked) and (not self.has_inf):
            try:
                hue = self.hue0 + self.dhue * (float(self.vmax) - self.color_func(value)) / (float(self.vmax) - self.vmin)
                hue = float(np.abs(hue))
                color = QColor.fromHsvF(hue, self.sat, self.val, self.alp)
                return to_qvariant(color)
            except (TypeError, ValueError):
                return to_qvariant()
        elif role == Qt.FontRole:
            return self.get_font(SpyderFontType.MonospaceInterface)
        return to_qvariant()

    def setData(self, index, value, role=Qt.EditRole):
        if False:
            for i in range(10):
                print('nop')
        'Cell content change'
        if not index.isValid() or self.readonly:
            return False
        i = index.row()
        j = index.column()
        value = from_qvariant(value, str)
        dtype = self._data.dtype.name
        if dtype == 'bool':
            try:
                val = bool(float(value))
            except ValueError:
                val = value.lower() == 'true'
        elif dtype.startswith('string') or dtype.startswith('bytes'):
            val = to_binary_string(value, 'utf8')
        elif dtype.startswith('unicode') or dtype.startswith('str'):
            val = to_text_string(value)
        else:
            if value.lower().startswith('e') or value.lower().endswith('e'):
                return False
            try:
                val = complex(value)
                if not val.imag:
                    val = val.real
            except ValueError as e:
                QMessageBox.critical(self.dialog, 'Error', 'Value error: %s' % str(e))
                return False
        try:
            self.test_array[0] = val
        except OverflowError as e:
            print('OverflowError: ' + str(e))
            QMessageBox.critical(self.dialog, 'Error', 'Overflow error: %s' % str(e))
            return False
        self.changes[i, j] = self.test_array[0]
        self.dataChanged.emit(index, index)
        if not is_string(val):
            val = self.color_func(val)
            if val > self.vmax:
                self.vmax = val
            if val < self.vmin:
                self.vmin = val
        return True

    def flags(self, index):
        if False:
            print('Hello World!')
        'Set editable flag'
        if not index.isValid():
            return Qt.ItemIsEnabled
        return Qt.ItemFlags(int(QAbstractTableModel.flags(self, index) | Qt.ItemIsEditable))

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if False:
            return 10
        'Set header data'
        if role != Qt.DisplayRole:
            return to_qvariant()
        labels = self.xlabels if orientation == Qt.Horizontal else self.ylabels
        if labels is None:
            return to_qvariant(int(section))
        else:
            return to_qvariant(labels[section])

    def reset(self):
        if False:
            i = 10
            return i + 15
        self.beginResetModel()
        self.endResetModel()

class ArrayDelegate(QItemDelegate, SpyderFontsMixin):
    """Array Editor Item Delegate"""

    def __init__(self, dtype, parent=None):
        if False:
            while True:
                i = 10
        QItemDelegate.__init__(self, parent)
        self.dtype = dtype

    def createEditor(self, parent, option, index):
        if False:
            for i in range(10):
                print('nop')
        'Create editor widget'
        model = index.model()
        value = model.get_value(index)
        if type(value) == np.ndarray or model.readonly:
            return
        elif model._data.dtype.name == 'bool':
            value = not value
            model.setData(index, to_qvariant(value))
            return
        elif value is not np.ma.masked:
            editor = QLineEdit(parent)
            editor.setFont(self.get_font(SpyderFontType.MonospaceInterface))
            editor.setAlignment(Qt.AlignCenter)
            if is_number(self.dtype):
                validator = QDoubleValidator(editor)
                validator.setLocale(QLocale('C'))
                editor.setValidator(validator)
            editor.returnPressed.connect(self.commitAndCloseEditor)
            return editor

    def commitAndCloseEditor(self):
        if False:
            print('Hello World!')
        'Commit and close editor'
        editor = self.sender()
        try:
            self.commitData.emit(editor)
        except AttributeError:
            pass
        self.closeEditor.emit(editor, QAbstractItemDelegate.NoHint)

    def setEditorData(self, editor, index):
        if False:
            for i in range(10):
                print('nop')
        "Set editor widget's data"
        text = from_qvariant(index.model().data(index, Qt.DisplayRole), str)
        editor.setText(text)

class ArrayView(QTableView):
    """Array view class"""

    def __init__(self, parent, model, dtype, shape):
        if False:
            while True:
                i = 10
        QTableView.__init__(self, parent)
        self.setModel(model)
        self.setItemDelegate(ArrayDelegate(dtype, self))
        total_width = 0
        for k in range(shape[1]):
            total_width += self.columnWidth(k)
        self.viewport().resize(min(total_width, 1024), self.height())
        self.shape = shape
        self.menu = self.setup_menu()
        CONF.config_shortcut(self.copy, context='variable_explorer', name='copy', parent=self)
        self.horizontalScrollBar().valueChanged.connect(self._load_more_columns)
        self.verticalScrollBar().valueChanged.connect(self._load_more_rows)

    def _load_more_columns(self, value):
        if False:
            print('Hello World!')
        'Load more columns to display.'
        try:
            self.load_more_data(value, columns=True)
        except NameError:
            pass

    def _load_more_rows(self, value):
        if False:
            print('Hello World!')
        'Load more rows to display.'
        try:
            self.load_more_data(value, rows=True)
        except NameError:
            pass

    def load_more_data(self, value, rows=False, columns=False):
        if False:
            while True:
                i = 10
        try:
            old_selection = self.selectionModel().selection()
            old_rows_loaded = old_cols_loaded = None
            if rows and value == self.verticalScrollBar().maximum():
                old_rows_loaded = self.model().rows_loaded
                self.model().fetch_more(rows=rows)
            if columns and value == self.horizontalScrollBar().maximum():
                old_cols_loaded = self.model().cols_loaded
                self.model().fetch_more(columns=columns)
            if old_rows_loaded is not None or old_cols_loaded is not None:
                new_selection = QItemSelection()
                for part in old_selection:
                    top = part.top()
                    bottom = part.bottom()
                    if old_rows_loaded is not None and top == 0 and (bottom == old_rows_loaded - 1):
                        bottom = self.model().rows_loaded - 1
                    left = part.left()
                    right = part.right()
                    if old_cols_loaded is not None and left == 0 and (right == old_cols_loaded - 1):
                        right = self.model().cols_loaded - 1
                    top_left = self.model().index(top, left)
                    bottom_right = self.model().index(bottom, right)
                    part = QItemSelectionRange(top_left, bottom_right)
                    new_selection.append(part)
                self.selectionModel().select(new_selection, self.selectionModel().ClearAndSelect)
        except NameError:
            pass

    @Slot()
    def resize_to_contents(self):
        if False:
            return 10
        'Resize cells to contents'
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.resizeColumnsToContents()
        self.model().fetch_more(columns=True)
        self.resizeColumnsToContents()
        QApplication.restoreOverrideCursor()

    def setup_menu(self):
        if False:
            while True:
                i = 10
        'Setup context menu'
        self.copy_action = create_action(self, _('Copy'), shortcut=keybinding('Copy'), icon=ima.icon('editcopy'), triggered=self.copy, context=Qt.WidgetShortcut)
        menu = QMenu(self)
        add_actions(menu, [self.copy_action])
        return menu

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Reimplement Qt method'
        self.menu.popup(event.globalPos())
        event.accept()

    def keyPressEvent(self, event):
        if False:
            return 10
        'Reimplement Qt method'
        if event == QKeySequence.Copy:
            self.copy()
        else:
            QTableView.keyPressEvent(self, event)

    def _sel_to_text(self, cell_range):
        if False:
            while True:
                i = 10
        'Copy an array portion to a unicode string'
        if not cell_range:
            return
        (row_min, row_max, col_min, col_max) = get_idx_rect(cell_range)
        if col_min == 0 and col_max == self.model().cols_loaded - 1:
            col_max = self.model().total_cols - 1
        if row_min == 0 and row_max == self.model().rows_loaded - 1:
            row_max = self.model().total_rows - 1
        _data = self.model().get_data()
        output = io.BytesIO()
        try:
            fmt = '%' + self.model().get_format_spec()
            np.savetxt(output, _data[row_min:row_max + 1, col_min:col_max + 1], delimiter='\t', fmt=fmt)
        except:
            QMessageBox.warning(self, _('Warning'), _('It was not possible to copy values for this array'))
            return
        contents = output.getvalue().decode('utf-8')
        output.close()
        return contents

    @Slot()
    def copy(self):
        if False:
            for i in range(10):
                print('nop')
        'Copy text to clipboard'
        cliptxt = self._sel_to_text(self.selectedIndexes())
        clipboard = QApplication.clipboard()
        clipboard.setText(cliptxt)

    def edit_item(self):
        if False:
            return 10
        'Edit item'
        index = self.currentIndex()
        if index.isValid():
            self.edit(index)

class ArrayEditorWidget(QWidget):

    def __init__(self, parent, data, readonly=False, xlabels=None, ylabels=None):
        if False:
            while True:
                i = 10
        QWidget.__init__(self, parent)
        self.data = data
        self.old_data_shape = None
        if len(self.data.shape) == 1:
            self.old_data_shape = self.data.shape
            self.data.shape = (self.data.shape[0], 1)
        elif len(self.data.shape) == 0:
            self.old_data_shape = self.data.shape
            self.data.shape = (1, 1)
        format_spec = SUPPORTED_FORMATS.get(data.dtype.name, 's')
        self.model = ArrayModel(self.data, format_spec=format_spec, xlabels=xlabels, ylabels=ylabels, readonly=readonly, parent=self)
        self.view = ArrayView(self, self.model, data.dtype, data.shape)
        layout = QVBoxLayout()
        layout.addWidget(self.view)
        self.setLayout(layout)

    def accept_changes(self):
        if False:
            print('Hello World!')
        'Accept changes'
        for ((i, j), value) in list(self.model.changes.items()):
            self.data[i, j] = value
        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    def reject_changes(self):
        if False:
            for i in range(10):
                print('nop')
        'Reject changes'
        if self.old_data_shape is not None:
            self.data.shape = self.old_data_shape

    @Slot()
    def change_format(self):
        if False:
            return 10
        'Change display format'
        (format_spec, valid) = QInputDialog.getText(self, _('Format'), _('Float formatting'), QLineEdit.Normal, self.model.get_format_spec())
        if valid:
            format_spec = str(format_spec)
            try:
                format(1.1, format_spec)
            except:
                QMessageBox.critical(self, _('Error'), _('Format (%s) is incorrect') % format_spec)
                return
            self.model.set_format_spec(format_spec)

class ArrayEditor(BaseDialog, SpyderWidgetMixin):
    """Array Editor Dialog"""
    CONF_SECTION = 'variable_explorer'

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.data = None
        self.arraywidget = None
        self.stack = None
        self.layout = None
        self.btn_save_and_close = None
        self.btn_close = None
        self.dim_indexes = [{}, {}, {}]
        self.last_dim = 0

    def setup_and_check(self, data, title='', readonly=False, xlabels=None, ylabels=None):
        if False:
            print('Hello World!')
        '\n        Setup ArrayEditor:\n        return False if data is not supported, True otherwise\n        '
        self.data = data
        readonly = readonly or not self.data.flags.writeable
        is_masked_array = isinstance(data, np.ma.MaskedArray)
        if hasattr(data.dtype, 'names'):
            is_record_array = data.dtype.names is not None
        else:
            is_record_array = False
        if data.ndim > 3:
            self.error(_('Arrays with more than 3 dimensions are not supported'))
            return False
        if xlabels is not None and len(xlabels) != self.data.shape[1]:
            self.error(_("The 'xlabels' argument length do no match array column number"))
            return False
        if ylabels is not None and len(ylabels) != self.data.shape[0]:
            self.error(_("The 'ylabels' argument length do no match array row number"))
            return False
        if not is_record_array:
            if hasattr(data.dtype, 'name'):
                dtn = data.dtype.name
            else:
                dtn = 'Unknown'
            if dtn == 'object':
                if data.shape == ():
                    self.error(_('Object arrays without shape are not supported'))
                    return False
                self.readonly = readonly = True
            elif dtn not in SUPPORTED_FORMATS and (not dtn.startswith('str')) and (not dtn.startswith('unicode')):
                arr = _('%s arrays') % dtn
                self.error(_('%s are currently not supported') % arr)
                return False
        self.layout = QGridLayout()
        self.setLayout(self.layout)
        if title:
            title = to_text_string(title) + ' - ' + _('NumPy object array')
        else:
            title = _('Array editor')
        if readonly:
            title += ' (' + _('read only') + ')'
        self.setWindowTitle(title)
        self.stack = QStackedWidget(self)
        if is_record_array:
            for name in data.dtype.names:
                self.stack.addWidget(ArrayEditorWidget(self, data[name], readonly, xlabels, ylabels))
        elif is_masked_array:
            self.stack.addWidget(ArrayEditorWidget(self, data, readonly, xlabels, ylabels))
            self.stack.addWidget(ArrayEditorWidget(self, data.data, readonly, xlabels, ylabels))
            self.stack.addWidget(ArrayEditorWidget(self, data.mask, readonly, xlabels, ylabels))
        elif data.ndim == 3:
            self.index_spin = QSpinBox(self, keyboardTracking=False)
            self.index_spin.valueChanged.connect(self.change_active_widget)
            self.shape_label = QLabel()
            self.slicing_label = QLabel()
            self.current_dim_changed(self.last_dim)
        else:
            self.stack.addWidget(ArrayEditorWidget(self, data, readonly, xlabels, ylabels))
        self.arraywidget = self.stack.currentWidget()
        self.arraywidget.model.dataChanged.connect(self.save_and_close_enable)
        self.stack.currentChanged.connect(self.current_widget_changed)
        self.layout.addWidget(self.stack, 1, 0)
        toolbar = SpyderToolbar(parent=self, title='Editor toolbar')
        toolbar.setStyleSheet(str(PANES_TOOLBAR_STYLESHEET))
        self.copy_action = self.create_action(ArrayEditorActions.Copy, text=_('Copy'), icon=self.create_icon('editcopy'), triggered=self.arraywidget.view.copy)
        toolbar.add_item(self.copy_action)
        self.edit_action = self.create_action(ArrayEditorActions.Edit, text=_('Edit'), icon=self.create_icon('edit'), triggered=self.arraywidget.view.edit_item)
        toolbar.add_item(self.edit_action)
        self.format_action = self.create_action(ArrayEditorActions.Format, text=_('Format'), icon=self.create_icon('format_float'), tip=_('Set format of floating-point numbers'), triggered=self.arraywidget.change_format)
        self.format_action.setEnabled(is_float(self.arraywidget.data.dtype))
        toolbar.add_item(self.format_action)
        self.resize_action = self.create_action(ArrayEditorActions.Resize, text=_('Resize'), icon=self.create_icon('collapse_column'), tip=_('Resize columns to contents'), triggered=self.arraywidget.view.resize_to_contents)
        toolbar.add_item(self.resize_action)
        self.toggle_bgcolor_action = self.create_action(ArrayEditorActions.ToggleBackgroundColor, text=_('Background color'), icon=self.create_icon('background_color'), toggled=lambda state: self.arraywidget.model.bgcolor(state), initial=self.arraywidget.model.bgcolor_enabled)
        self.toggle_bgcolor_action.setEnabled(self.arraywidget.model.bgcolor_enabled)
        toolbar.add_item(self.toggle_bgcolor_action)
        toolbar._render()
        self.layout.addWidget(toolbar, 0, 0)
        btn_layout = QHBoxLayout()
        if is_record_array or is_masked_array or data.ndim == 3:
            if is_record_array:
                btn_layout.addWidget(QLabel(_('Record array fields:')))
                names = []
                for name in data.dtype.names:
                    field = data.dtype.fields[name]
                    text = name
                    if len(field) >= 3:
                        title = field[2]
                        if not is_text_string(title):
                            title = repr(title)
                        text += ' - ' + title
                    names.append(text)
            else:
                names = [_('Masked data'), _('Data'), _('Mask')]
            if data.ndim == 3:
                names = [str(i) for i in range(3)]
                ra_combo = QComboBox(self)
                ra_combo.addItems(names)
                ra_combo.currentIndexChanged.connect(self.current_dim_changed)
                label = QLabel(_('Axis:'))
                btn_layout.addWidget(label)
                btn_layout.addWidget(ra_combo)
                btn_layout.addWidget(self.shape_label)
                label = QLabel(_('Index:'))
                btn_layout.addWidget(label)
                btn_layout.addWidget(self.index_spin)
                btn_layout.addWidget(self.slicing_label)
            else:
                ra_combo = QComboBox(self)
                ra_combo.currentIndexChanged.connect(self.stack.setCurrentIndex)
                ra_combo.addItems(names)
                btn_layout.addWidget(ra_combo)
            if is_masked_array:
                label = QLabel(_('<u>Warning</u>: Changes are applied separately'))
                label.setToolTip(_("For performance reasons, changes applied to masked arrays won't be reflected in array's data (and vice-versa)."))
                btn_layout.addWidget(label)
        btn_layout.addStretch()
        if not readonly:
            self.btn_save_and_close = QPushButton(_('Save and Close'))
            self.btn_save_and_close.setDisabled(True)
            self.btn_save_and_close.clicked.connect(self.accept)
            btn_layout.addWidget(self.btn_save_and_close)
        self.btn_close = QPushButton(_('Close'))
        self.btn_close.setAutoDefault(True)
        self.btn_close.setDefault(True)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)
        btn_layout.setContentsMargins(4, 4, 4, 4)
        self.layout.addLayout(btn_layout, 2, 0)
        self.setMinimumSize(500, 300)
        self.setWindowFlags(Qt.Window)
        return True

    @Slot(QModelIndex, QModelIndex)
    def save_and_close_enable(self, left_top, bottom_right):
        if False:
            print('Hello World!')
        'Handle the data change event to enable the save and close button.'
        if self.btn_save_and_close:
            self.btn_save_and_close.setEnabled(True)
            self.btn_save_and_close.setAutoDefault(True)
            self.btn_save_and_close.setDefault(True)

    def current_widget_changed(self, index):
        if False:
            for i in range(10):
                print('nop')
        self.arraywidget = self.stack.widget(index)
        self.arraywidget.model.dataChanged.connect(self.save_and_close_enable)
        self.toggle_bgcolor_action.setChecked(self.arraywidget.model.bgcolor_enabled)

    def change_active_widget(self, index):
        if False:
            print('Hello World!')
        '\n        This is implemented for handling negative values in index for\n        3d arrays, to give the same behavior as slicing\n        '
        string_index = [':'] * 3
        string_index[self.last_dim] = '<font color=red>%i</font>'
        self.slicing_label.setText(('Slicing: [' + ', '.join(string_index) + ']') % index)
        if index < 0:
            data_index = self.data.shape[self.last_dim] + index
        else:
            data_index = index
        slice_index = [slice(None)] * 3
        slice_index[self.last_dim] = data_index
        stack_index = self.dim_indexes[self.last_dim].get(data_index)
        if stack_index is None:
            stack_index = self.stack.count()
            try:
                self.stack.addWidget(ArrayEditorWidget(self, self.data[tuple(slice_index)]))
            except IndexError:
                self.stack.addWidget(ArrayEditorWidget(self, self.data))
            self.dim_indexes[self.last_dim][data_index] = stack_index
            self.stack.update()
        self.stack.setCurrentIndex(stack_index)

    def current_dim_changed(self, index):
        if False:
            print('Hello World!')
        '\n        This change the active axis the array editor is plotting over\n        in 3D\n        '
        self.last_dim = index
        string_size = ['%i'] * 3
        string_size[index] = '<font color=red>%i</font>'
        self.shape_label.setText(('Shape: (' + ', '.join(string_size) + ')    ') % self.data.shape)
        if self.index_spin.value() != 0:
            self.index_spin.setValue(0)
        else:
            self.change_active_widget(0)
        self.index_spin.setRange(-self.data.shape[index], self.data.shape[index] - 1)

    @Slot()
    def accept(self):
        if False:
            print('Hello World!')
        'Reimplement Qt method.'
        try:
            for index in range(self.stack.count()):
                self.stack.widget(index).accept_changes()
            QDialog.accept(self)
        except RuntimeError:
            pass

    def get_value(self):
        if False:
            for i in range(10):
                print('nop')
        'Return modified array -- this is *not* a copy'
        return self.data

    def error(self, message):
        if False:
            for i in range(10):
                print('nop')
        'An error occurred, closing the dialog box'
        QMessageBox.critical(self, _('Array editor'), message)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.reject()

    @Slot()
    def reject(self):
        if False:
            return 10
        'Reimplement Qt method'
        if self.arraywidget is not None:
            for index in range(self.stack.count()):
                self.stack.widget(index).reject_changes()
        QDialog.reject(self)