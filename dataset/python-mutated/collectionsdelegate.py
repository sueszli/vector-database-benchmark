import copy
import datetime
import functools
import operator
from qtpy.compat import to_qvariant
from qtpy.QtCore import QDateTime, Qt, Signal
from qtpy.QtWidgets import QAbstractItemDelegate, QDateEdit, QDateTimeEdit, QItemDelegate, QLineEdit, QMessageBox, QTableView
from spyder_kernels.utils.lazymodules import FakeObject, numpy as np, pandas as pd, PIL
from spyder_kernels.utils.nsview import display_to_value, is_editable_type, is_known_type
from spyder.api.config.fonts import SpyderFontsMixin, SpyderFontType
from spyder.config.base import _, is_conda_based_app
from spyder.py3compat import is_binary_string, is_text_string, to_text_string
from spyder.plugins.variableexplorer.widgets.arrayeditor import ArrayEditor
from spyder.plugins.variableexplorer.widgets.dataframeeditor import DataFrameEditor
from spyder.plugins.variableexplorer.widgets.texteditor import TextEditor
LARGE_COLLECTION = 100000.0
LARGE_ARRAY = 5000000.0

class CollectionsDelegate(QItemDelegate, SpyderFontsMixin):
    """CollectionsEditor Item Delegate"""
    sig_free_memory_requested = Signal()
    sig_editor_creation_started = Signal()
    sig_editor_shown = Signal()

    def __init__(self, parent=None, namespacebrowser=None):
        if False:
            return 10
        QItemDelegate.__init__(self, parent)
        self.namespacebrowser = namespacebrowser
        self._editors = {}

    def get_value(self, index):
        if False:
            i = 10
            return i + 15
        if index.isValid():
            return index.model().get_value(index)

    def set_value(self, index, value):
        if False:
            return 10
        if index.isValid():
            index.model().set_value(index, value)

    def show_warning(self, index):
        if False:
            return 10
        "\n        Decide if showing a warning when the user is trying to view\n        a big variable associated to a Tablemodel index.\n\n        This avoids getting the variables' value to know its\n        size and type, using instead those already computed by\n        the TableModel.\n\n        The problem is when a variable is too big, it can take a\n        lot of time just to get its value.\n        "
        val_type = index.sibling(index.row(), 1).data()
        val_size = index.sibling(index.row(), 2).data()
        if val_type in ['list', 'set', 'tuple', 'dict']:
            if int(val_size) > LARGE_COLLECTION:
                return True
        elif val_type in ['DataFrame', 'Series'] or 'Array' in val_type or 'Index' in val_type:
            try:
                shape = [int(s) for s in val_size.strip('()').split(',') if s]
                size = functools.reduce(operator.mul, shape)
                if size > LARGE_ARRAY:
                    return True
            except Exception:
                pass
        return False

    def createEditor(self, parent, option, index, object_explorer=False):
        if False:
            for i in range(10):
                print('nop')
        'Overriding method createEditor'
        val_type = index.sibling(index.row(), 1).data()
        self.sig_editor_creation_started.emit()
        if index.column() < 3:
            return None
        if self.show_warning(index):
            answer = QMessageBox.warning(self.parent(), _('Warning'), _('Opening this variable can be slow\n\nDo you want to continue anyway?'), QMessageBox.Yes | QMessageBox.No)
            if answer == QMessageBox.No:
                self.sig_editor_shown.emit()
                return None
        try:
            value = self.get_value(index)
            if value is None:
                return None
        except ImportError as msg:
            self.sig_editor_shown.emit()
            module = str(msg).split("'")[1]
            if module in ['pandas', 'numpy']:
                if module == 'numpy':
                    val_type = 'array'
                else:
                    val_type = 'dataframe or series'
                message = _("Spyder is unable to show the {val_type} object you're trying to view because <tt>{module}</tt> is missing. Please install that package in your Spyder environment to fix this problem.")
                QMessageBox.critical(self.parent(), _('Error'), message.format(val_type=val_type, module=module))
                return
            else:
                if is_conda_based_app():
                    message = _("Spyder is unable to show the variable you're trying to view because the module <tt>{module}</tt> is not supported by Spyder's standalone application.<br>")
                else:
                    message = _("Spyder is unable to show the variable you're trying to view because the module <tt>{module}</tt> is not found in your Spyder environment. Please install this package in this environment.<br>")
                QMessageBox.critical(self.parent(), _('Error'), message.format(module=module))
                return
        except Exception as msg:
            self.sig_editor_shown.emit()
            QMessageBox.critical(self.parent(), _('Error'), _('Spyder was unable to retrieve the value of this variable from the console.<br><br>The error message was:<br>%s') % to_text_string(msg))
            return
        key = index.model().get_key(index)
        readonly = isinstance(value, (tuple, set)) or self.parent().readonly or (not is_known_type(value))
        if isinstance(value, np.void):
            self.sig_editor_shown.emit()
            return None
        elif isinstance(value, (list, set, tuple, dict)) and (not object_explorer):
            from spyder.widgets.collectionseditor import CollectionsEditor
            editor = CollectionsEditor(parent=parent, namespacebrowser=self.namespacebrowser)
            editor.setup(value, key, icon=self.parent().windowIcon(), readonly=readonly)
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif isinstance(value, (np.ndarray, np.ma.MaskedArray)) and np.ndarray is not FakeObject and (not object_explorer):
            from .arrayeditor import ArrayEditor
            editor = ArrayEditor(parent=parent)
            if not editor.setup_and_check(value, title=key, readonly=readonly):
                self.sig_editor_shown.emit()
                return
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif isinstance(value, PIL.Image.Image) and np.ndarray is not FakeObject and (PIL.Image is not FakeObject) and (not object_explorer):
            from .arrayeditor import ArrayEditor
            arr = np.array(value)
            editor = ArrayEditor(parent=parent)
            if not editor.setup_and_check(arr, title=key, readonly=readonly):
                self.sig_editor_shown.emit()
                return
            conv_func = lambda arr: PIL.Image.fromarray(arr, mode=value.mode)
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly, conv=conv_func))
            return None
        elif isinstance(value, (pd.DataFrame, pd.Index, pd.Series)) and pd.DataFrame is not FakeObject and (not object_explorer):
            from .dataframeeditor import DataFrameEditor
            editor = DataFrameEditor(parent=parent)
            if not editor.setup_and_check(value, title=key):
                self.sig_editor_shown.emit()
                return
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif isinstance(value, datetime.date) and (not object_explorer):
            if readonly:
                self.sig_editor_shown.emit()
                return None
            else:
                if isinstance(value, datetime.datetime):
                    editor = QDateTimeEdit(value, parent=parent)
                    try:
                        value.time()
                    except ValueError:
                        self.sig_editor_shown.emit()
                        return None
                else:
                    editor = QDateEdit(value, parent=parent)
                editor.setCalendarPopup(True)
                editor.setFont(self.get_font(SpyderFontType.MonospaceInterface))
                self.sig_editor_shown.emit()
                return editor
        elif is_text_string(value) and len(value) > 40 and (not object_explorer):
            te = TextEditor(None, parent=parent)
            if te.setup_and_check(value):
                editor = TextEditor(value, key, readonly=readonly, parent=parent)
                self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif is_editable_type(value) and (not object_explorer):
            if readonly:
                self.sig_editor_shown.emit()
                return None
            else:
                editor = QLineEdit(parent=parent)
                editor.setFont(self.get_font(SpyderFontType.MonospaceInterface))
                editor.setAlignment(Qt.AlignLeft)
                self.sig_editor_shown.emit()
                return editor
        else:
            from spyder.plugins.variableexplorer.widgets.objectexplorer import ObjectExplorer
            editor = ObjectExplorer(value, name=key, parent=parent, namespacebrowser=self.namespacebrowser, readonly=readonly)
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None

    def create_dialog(self, editor, data):
        if False:
            print('Hello World!')
        self._editors[id(editor)] = data
        editor.accepted.connect(lambda eid=id(editor): self.editor_accepted(eid))
        editor.rejected.connect(lambda eid=id(editor): self.editor_rejected(eid))
        self.sig_editor_shown.emit()
        editor.show()

    def editor_accepted(self, editor_id):
        if False:
            while True:
                i = 10
        data = self._editors[editor_id]
        if not data['readonly']:
            index = data['model'].get_index_from_key(data['key'])
            value = data['editor'].get_value()
            conv_func = data.get('conv', lambda v: v)
            self.set_value(index, conv_func(value))
        try:
            self._editors.pop(editor_id)
        except KeyError:
            pass
        self.free_memory()

    def editor_rejected(self, editor_id):
        if False:
            while True:
                i = 10
        try:
            self._editors.pop(editor_id)
        except KeyError:
            pass
        self.free_memory()

    def free_memory(self):
        if False:
            return 10
        'Free memory after closing an editor.'
        try:
            self.sig_free_memory_requested.emit()
        except RuntimeError:
            pass

    def commitAndCloseEditor(self):
        if False:
            for i in range(10):
                print('nop')
        'Overriding method commitAndCloseEditor'
        editor = self.sender()
        try:
            self.commitData.emit(editor)
        except AttributeError:
            pass
        self.closeEditor.emit(editor, QAbstractItemDelegate.NoHint)

    def setEditorData(self, editor, index):
        if False:
            return 10
        '\n        Overriding method setEditorData\n        Model --> Editor\n        '
        value = self.get_value(index)
        if isinstance(editor, QLineEdit):
            if is_binary_string(value):
                try:
                    value = to_text_string(value, 'utf8')
                except Exception:
                    pass
            if not is_text_string(value):
                value = repr(value)
            editor.setText(value)
        elif isinstance(editor, QDateEdit):
            editor.setDate(value)
        elif isinstance(editor, QDateTimeEdit):
            editor.setDateTime(QDateTime(value.date(), value.time()))

    def setModelData(self, editor, model, index):
        if False:
            for i in range(10):
                print('nop')
        '\n        Overriding method setModelData\n        Editor --> Model\n        '
        if hasattr(model, 'sourceModel') and (not hasattr(model.sourceModel(), 'set_value')) or not hasattr(model, 'set_value'):
            return
        if isinstance(editor, QLineEdit):
            value = editor.text()
            try:
                value = display_to_value(to_qvariant(value), self.get_value(index), ignore_errors=False)
            except Exception as msg:
                QMessageBox.critical(editor, _('Edit item'), _('<b>Unable to assign data to item.</b><br><br>Error message:<br>%s') % str(msg))
                return
        elif isinstance(editor, QDateEdit):
            qdate = editor.date()
            value = datetime.date(qdate.year(), qdate.month(), qdate.day())
        elif isinstance(editor, QDateTimeEdit):
            qdatetime = editor.dateTime()
            qdate = qdatetime.date()
            qtime = qdatetime.time()
            value = datetime.datetime(qdate.year(), qdate.month(), qdate.day(), qtime.hour(), qtime.minute(), qtime.second(), qtime.msec() * 1000)
        else:
            raise RuntimeError('Unsupported editor widget')
        self.set_value(index, value)

    def updateEditorGeometry(self, editor, option, index):
        if False:
            print('Hello World!')
        "\n        Overriding method updateEditorGeometry.\n\n        This is necessary to set the correct position of the QLineEdit\n        editor since option.rect doesn't have values -> QRect() and\n        makes the editor to be invisible (i.e. it has 0 as x, y, width\n        and height) when doing double click over a cell.\n        See spyder-ide/spyder#9945\n        "
        table_view = editor.parent().parent()
        if isinstance(table_view, QTableView):
            row = index.row()
            column = index.column()
            y0 = table_view.rowViewportPosition(row)
            x0 = table_view.columnViewportPosition(column)
            width = table_view.columnWidth(column)
            height = table_view.rowHeight(row)
            editor.setGeometry(x0, y0, width, height)
        else:
            super(CollectionsDelegate, self).updateEditorGeometry(editor, option, index)

class ToggleColumnDelegate(CollectionsDelegate):
    """ToggleColumn Item Delegate"""

    def __init__(self, parent=None, namespacebrowser=None):
        if False:
            i = 10
            return i + 15
        CollectionsDelegate.__init__(self, parent, namespacebrowser)
        self.current_index = None
        self.old_obj = None

    def restore_object(self):
        if False:
            for i in range(10):
                print('nop')
        'Discart changes made to the current object in edition.'
        if self.current_index and self.old_obj is not None:
            index = self.current_index
            index.model().treeItem(index).obj = self.old_obj

    def get_value(self, index):
        if False:
            i = 10
            return i + 15
        'Get object value in index.'
        if index.isValid():
            value = index.model().treeItem(index).obj
            return value

    def set_value(self, index, value):
        if False:
            for i in range(10):
                print('nop')
        if index.isValid():
            index.model().set_value(index, value)

    def createEditor(self, parent, option, index):
        if False:
            while True:
                i = 10
        'Overriding method createEditor'
        if self.show_warning(index):
            answer = QMessageBox.warning(self.parent(), _('Warning'), _('Opening this variable can be slow\n\nDo you want to continue anyway?'), QMessageBox.Yes | QMessageBox.No)
            if answer == QMessageBox.No:
                return None
        try:
            value = self.get_value(index)
            try:
                self.old_obj = value.copy()
            except AttributeError:
                self.old_obj = copy.deepcopy(value)
            if value is None:
                return None
        except Exception as msg:
            QMessageBox.critical(self.parent(), _('Error'), _('Spyder was unable to retrieve the value of this variable from the console.<br><br>The error message was:<br><i>%s</i>') % to_text_string(msg))
            return
        self.current_index = index
        key = index.model().get_key(index).obj_name
        readonly = isinstance(value, (tuple, set)) or self.parent().readonly or (not is_known_type(value))
        if isinstance(value, (list, set, tuple, dict)):
            from spyder.widgets.collectionseditor import CollectionsEditor
            editor = CollectionsEditor(parent=parent, namespacebrowser=self.namespacebrowser)
            editor.setup(value, key, icon=self.parent().windowIcon(), readonly=readonly)
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif isinstance(value, (np.ndarray, np.ma.MaskedArray)) and np.ndarray is not FakeObject:
            editor = ArrayEditor(parent=parent)
            if not editor.setup_and_check(value, title=key, readonly=readonly):
                return
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif isinstance(value, PIL.Image.Image) and np.ndarray is not FakeObject and (PIL.Image is not FakeObject):
            arr = np.array(value)
            editor = ArrayEditor(parent=parent)
            if not editor.setup_and_check(arr, title=key, readonly=readonly):
                return
            conv_func = lambda arr: PIL.Image.fromarray(arr, mode=value.mode)
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly, conv=conv_func))
            return None
        elif isinstance(value, (pd.DataFrame, pd.Index, pd.Series)) and pd.DataFrame is not FakeObject:
            editor = DataFrameEditor(parent=parent)
            if not editor.setup_and_check(value, title=key):
                return
            self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif isinstance(value, datetime.date):
            if readonly:
                return None
            else:
                if isinstance(value, datetime.datetime):
                    editor = QDateTimeEdit(value, parent=parent)
                else:
                    editor = QDateEdit(value, parent=parent)
                editor.setCalendarPopup(True)
                editor.setFont(self.get_font(SpyderFontType.MonospaceInterface))
                return editor
        elif is_text_string(value) and len(value) > 40:
            te = TextEditor(None, parent=parent)
            if te.setup_and_check(value):
                editor = TextEditor(value, key, readonly=readonly, parent=parent)
                self.create_dialog(editor, dict(model=index.model(), editor=editor, key=key, readonly=readonly))
            return None
        elif is_editable_type(value):
            if readonly:
                return None
            else:
                editor = QLineEdit(parent=parent)
                editor.setFont(self.get_font(SpyderFontType.MonospaceInterface))
                editor.setAlignment(Qt.AlignLeft)
                return editor
        else:
            return None

    def editor_accepted(self, editor_id):
        if False:
            return 10
        'Actions to execute when the editor has been closed.'
        data = self._editors[editor_id]
        if not data['readonly'] and self.current_index:
            index = self.current_index
            value = data['editor'].get_value()
            conv_func = data.get('conv', lambda v: v)
            self.set_value(index, conv_func(value))
        try:
            self._editors.pop(editor_id)
        except KeyError:
            pass
        self.free_memory()

    def editor_rejected(self, editor_id):
        if False:
            i = 10
            return i + 15
        'Actions to do when the editor was rejected.'
        self.restore_object()
        super(ToggleColumnDelegate, self).editor_rejected(editor_id)