"""
Text data Importing Wizard based on Qt
"""
import datetime
from functools import partial as ft_partial
import io
from itertools import zip_longest
from qtpy.compat import to_qvariant
from qtpy.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal, Slot
from qtpy.QtGui import QColor, QIntValidator
from qtpy.QtWidgets import QCheckBox, QDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QPushButton, QMenu, QMessageBox, QRadioButton, QSizePolicy, QSpacerItem, QTableView, QTabWidget, QTextEdit, QVBoxLayout, QWidget
from spyder_kernels.utils.lazymodules import FakeObject, numpy as np, pandas as pd
from spyder.config.base import _
from spyder.py3compat import INT_TYPES, TEXT_TYPES, to_text_string
from spyder.utils import programs
from spyder.utils.icon_manager import ima
from spyder.utils.qthelpers import add_actions, create_action
from spyder.plugins.variableexplorer.widgets.basedialog import BaseDialog
from spyder.utils.palette import SpyderPalette

def try_to_parse(value):
    if False:
        print('Hello World!')
    _types = ('int', 'float')
    for _t in _types:
        try:
            _val = eval("%s('%s')" % (_t, value))
            return _val
        except (ValueError, SyntaxError):
            pass
    return value

def try_to_eval(value):
    if False:
        i = 10
        return i + 15
    try:
        return eval(value)
    except (NameError, SyntaxError, ImportError):
        return value
try:
    from dateutil.parser import parse as dateparse
except:

    def dateparse(datestr, dayfirst=True):
        if False:
            for i in range(10):
                print('nop')
        "Just for 'day/month/year' strings"
        (_a, _b, _c) = list(map(int, datestr.split('/')))
        if dayfirst:
            return datetime.datetime(_c, _b, _a)
        return datetime.datetime(_c, _a, _b)

def datestr_to_datetime(value, dayfirst=True):
    if False:
        i = 10
        return i + 15
    return dateparse(value, dayfirst=dayfirst)

def get_color(value, alpha):
    if False:
        while True:
            i = 10
    'Return color depending on value type'
    colors = {bool: SpyderPalette.GROUP_1, tuple([float] + list(INT_TYPES)): SpyderPalette.GROUP_2, TEXT_TYPES: SpyderPalette.GROUP_3, datetime.date: SpyderPalette.GROUP_4, list: SpyderPalette.GROUP_5, set: SpyderPalette.GROUP_6, tuple: SpyderPalette.GROUP_7, dict: SpyderPalette.GROUP_8, np.ndarray: SpyderPalette.GROUP_9}
    color = QColor()
    for typ in colors:
        if isinstance(value, typ):
            color = QColor(colors[typ])
    color.setAlphaF(alpha)
    return color

class ContentsWidget(QWidget):
    """Import wizard contents widget"""
    asDataChanged = Signal(bool)

    def __init__(self, parent, text):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, parent)
        self.text_editor = QTextEdit(self)
        self.text_editor.setText(text)
        self.text_editor.setReadOnly(True)
        type_layout = QHBoxLayout()
        type_label = QLabel(_('Import as'))
        type_layout.addWidget(type_label)
        data_btn = QRadioButton(_('data'))
        data_btn.setChecked(True)
        self._as_data = True
        type_layout.addWidget(data_btn)
        code_btn = QRadioButton(_('code'))
        self._as_code = False
        type_layout.addWidget(code_btn)
        txt_btn = QRadioButton(_('text'))
        type_layout.addWidget(txt_btn)
        h_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        type_layout.addItem(h_spacer)
        type_frame = QFrame()
        type_frame.setLayout(type_layout)
        grid_layout = QGridLayout()
        grid_layout.setSpacing(0)
        col_label = QLabel(_('Column separator:'))
        grid_layout.addWidget(col_label, 0, 0)
        col_w = QWidget()
        col_btn_layout = QHBoxLayout()
        self.tab_btn = QRadioButton(_('Tab'))
        self.tab_btn.setChecked(False)
        col_btn_layout.addWidget(self.tab_btn)
        self.ws_btn = QRadioButton(_('Whitespace'))
        self.ws_btn.setChecked(False)
        col_btn_layout.addWidget(self.ws_btn)
        other_btn_col = QRadioButton(_('other'))
        other_btn_col.setChecked(True)
        col_btn_layout.addWidget(other_btn_col)
        col_w.setLayout(col_btn_layout)
        grid_layout.addWidget(col_w, 0, 1)
        self.line_edt = QLineEdit(',')
        self.line_edt.setMaximumWidth(30)
        self.line_edt.setEnabled(True)
        other_btn_col.toggled.connect(self.line_edt.setEnabled)
        grid_layout.addWidget(self.line_edt, 0, 2)
        row_label = QLabel(_('Row separator:'))
        grid_layout.addWidget(row_label, 1, 0)
        row_w = QWidget()
        row_btn_layout = QHBoxLayout()
        self.eol_btn = QRadioButton(_('EOL'))
        self.eol_btn.setChecked(True)
        row_btn_layout.addWidget(self.eol_btn)
        other_btn_row = QRadioButton(_('other'))
        row_btn_layout.addWidget(other_btn_row)
        row_w.setLayout(row_btn_layout)
        grid_layout.addWidget(row_w, 1, 1)
        self.line_edt_row = QLineEdit(';')
        self.line_edt_row.setMaximumWidth(30)
        self.line_edt_row.setEnabled(False)
        other_btn_row.toggled.connect(self.line_edt_row.setEnabled)
        grid_layout.addWidget(self.line_edt_row, 1, 2)
        grid_layout.setRowMinimumHeight(2, 15)
        other_group = QGroupBox(_('Additional options'))
        other_layout = QGridLayout()
        other_group.setLayout(other_layout)
        skiprows_label = QLabel(_('Skip rows:'))
        other_layout.addWidget(skiprows_label, 0, 0)
        self.skiprows_edt = QLineEdit('0')
        self.skiprows_edt.setMaximumWidth(30)
        intvalid = QIntValidator(0, len(to_text_string(text).splitlines()), self.skiprows_edt)
        self.skiprows_edt.setValidator(intvalid)
        self.skiprows_edt.textChanged.connect(lambda text: self.get_skiprows())
        other_layout.addWidget(self.skiprows_edt, 0, 1)
        other_layout.setColumnMinimumWidth(2, 5)
        comments_label = QLabel(_('Comments:'))
        other_layout.addWidget(comments_label, 0, 3)
        self.comments_edt = QLineEdit('#')
        self.comments_edt.setMaximumWidth(30)
        other_layout.addWidget(self.comments_edt, 0, 4)
        self.trnsp_box = QCheckBox(_('Transpose'))
        other_layout.addWidget(self.trnsp_box, 1, 0, 2, 0)
        grid_layout.addWidget(other_group, 3, 0, 2, 0)
        opts_frame = QFrame()
        opts_frame.setLayout(grid_layout)
        data_btn.toggled.connect(opts_frame.setEnabled)
        data_btn.toggled.connect(self.set_as_data)
        code_btn.toggled.connect(self.set_as_code)
        layout = QVBoxLayout()
        layout.addWidget(type_frame)
        layout.addWidget(self.text_editor)
        layout.addWidget(opts_frame)
        self.setLayout(layout)

    def get_as_data(self):
        if False:
            i = 10
            return i + 15
        'Return if data type conversion'
        return self._as_data

    def get_as_code(self):
        if False:
            for i in range(10):
                print('nop')
        'Return if code type conversion'
        return self._as_code

    def get_as_num(self):
        if False:
            while True:
                i = 10
        'Return if numeric type conversion'
        return self._as_num

    def get_col_sep(self):
        if False:
            return 10
        'Return the column separator'
        if self.tab_btn.isChecked():
            return u'\t'
        elif self.ws_btn.isChecked():
            return None
        return to_text_string(self.line_edt.text())

    def get_row_sep(self):
        if False:
            i = 10
            return i + 15
        'Return the row separator'
        if self.eol_btn.isChecked():
            return u'\n'
        return to_text_string(self.line_edt_row.text())

    def get_skiprows(self):
        if False:
            while True:
                i = 10
        'Return number of lines to be skipped'
        skip_rows = to_text_string(self.skiprows_edt.text())
        if skip_rows and skip_rows != '+':
            return int(skip_rows)
        else:
            self.skiprows_edt.clear()
            return 0

    def get_comments(self):
        if False:
            return 10
        'Return comment string'
        return to_text_string(self.comments_edt.text())

    @Slot(bool)
    def set_as_data(self, as_data):
        if False:
            while True:
                i = 10
        'Set if data type conversion'
        self._as_data = as_data
        self.asDataChanged.emit(as_data)

    @Slot(bool)
    def set_as_code(self, as_code):
        if False:
            while True:
                i = 10
        'Set if code type conversion'
        self._as_code = as_code

class PreviewTableModel(QAbstractTableModel):
    """Import wizard preview table model"""

    def __init__(self, data=[], parent=None):
        if False:
            for i in range(10):
                print('nop')
        QAbstractTableModel.__init__(self, parent)
        self._data = data

    def rowCount(self, parent=QModelIndex()):
        if False:
            while True:
                i = 10
        'Return row count'
        return len(self._data)

    def columnCount(self, parent=QModelIndex()):
        if False:
            print('Hello World!')
        'Return column count'
        return len(self._data[0])

    def _display_data(self, index):
        if False:
            return 10
        'Return a data element'
        return to_qvariant(self._data[index.row()][index.column()])

    def data(self, index, role=Qt.DisplayRole):
        if False:
            return 10
        'Return a model data element'
        if not index.isValid():
            return to_qvariant()
        if role == Qt.DisplayRole:
            return self._display_data(index)
        elif role == Qt.BackgroundColorRole:
            return to_qvariant(get_color(self._data[index.row()][index.column()], 0.5))
        elif role == Qt.TextAlignmentRole:
            return to_qvariant(int(Qt.AlignRight | Qt.AlignVCenter))
        return to_qvariant()

    def setData(self, index, value, role=Qt.EditRole):
        if False:
            for i in range(10):
                print('nop')
        'Set model data'
        return False

    def get_data(self):
        if False:
            print('Hello World!')
        'Return a copy of model data'
        return self._data[:][:]

    def parse_data_type(self, index, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Parse a type to an other type'
        if not index.isValid():
            return False
        try:
            if kwargs['atype'] == 'date':
                self._data[index.row()][index.column()] = datestr_to_datetime(self._data[index.row()][index.column()], kwargs['dayfirst']).date()
            elif kwargs['atype'] == 'perc':
                _tmp = self._data[index.row()][index.column()].replace('%', '')
                self._data[index.row()][index.column()] = eval(_tmp) / 100.0
            elif kwargs['atype'] == 'account':
                _tmp = self._data[index.row()][index.column()].replace(',', '')
                self._data[index.row()][index.column()] = eval(_tmp)
            elif kwargs['atype'] == 'unicode':
                self._data[index.row()][index.column()] = to_text_string(self._data[index.row()][index.column()])
            elif kwargs['atype'] == 'int':
                self._data[index.row()][index.column()] = int(self._data[index.row()][index.column()])
            elif kwargs['atype'] == 'float':
                self._data[index.row()][index.column()] = float(self._data[index.row()][index.column()])
            self.dataChanged.emit(index, index)
        except Exception as instance:
            print(instance)

    def reset(self):
        if False:
            return 10
        self.beginResetModel()
        self.endResetModel()

class PreviewTable(QTableView):
    """Import wizard preview widget"""

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        QTableView.__init__(self, parent)
        self._model = None
        self.date_dayfirst_action = create_action(self, 'dayfirst', triggered=ft_partial(self.parse_to_type, atype='date', dayfirst=True))
        self.date_monthfirst_action = create_action(self, 'monthfirst', triggered=ft_partial(self.parse_to_type, atype='date', dayfirst=False))
        self.perc_action = create_action(self, 'perc', triggered=ft_partial(self.parse_to_type, atype='perc'))
        self.acc_action = create_action(self, 'account', triggered=ft_partial(self.parse_to_type, atype='account'))
        self.str_action = create_action(self, 'unicode', triggered=ft_partial(self.parse_to_type, atype='unicode'))
        self.int_action = create_action(self, 'int', triggered=ft_partial(self.parse_to_type, atype='int'))
        self.float_action = create_action(self, 'float', triggered=ft_partial(self.parse_to_type, atype='float'))
        self.date_menu = QMenu()
        self.date_menu.setTitle('Date')
        add_actions(self.date_menu, (self.date_dayfirst_action, self.date_monthfirst_action))
        self.parse_menu = QMenu(self)
        self.parse_menu.addMenu(self.date_menu)
        add_actions(self.parse_menu, (self.perc_action, self.acc_action))
        self.parse_menu.setTitle('String to')
        self.opt_menu = QMenu(self)
        self.opt_menu.addMenu(self.parse_menu)
        add_actions(self.opt_menu, (self.str_action, self.int_action, self.float_action))

    def _shape_text(self, text, colsep=u'\t', rowsep=u'\n', transpose=False, skiprows=0, comments='#'):
        if False:
            while True:
                i = 10
        'Decode the shape of the given text'
        assert colsep != rowsep, 'Column sep should not equal Row sep'
        out = []
        text_rows = text.split(rowsep)
        assert skiprows < len(text_rows), 'Skip Rows > Line Count'
        text_rows = text_rows[skiprows:]
        for row in text_rows:
            stripped = to_text_string(row).strip()
            if len(stripped) == 0 or (comments and stripped.startswith(comments)):
                continue
            line = to_text_string(row).split(colsep)
            line = [try_to_parse(to_text_string(x)) for x in line]
            out.append(line)
        if programs.is_module_installed('numpy'):
            from numpy import nan
            out = list(zip_longest(*out, fillvalue=nan))
        else:
            out = list(zip_longest(*out, fillvalue=None))
        out = [[r[col] for r in out] for col in range(len(out[0]))]
        if transpose:
            return [[r[col] for r in out] for col in range(len(out[0]))]
        return out

    def get_data(self):
        if False:
            for i in range(10):
                print('nop')
        'Return model data'
        if self._model is None:
            return None
        return self._model.get_data()

    def process_data(self, text, colsep=u'\t', rowsep=u'\n', transpose=False, skiprows=0, comments='#'):
        if False:
            print('Hello World!')
        'Put data into table model'
        data = self._shape_text(text, colsep, rowsep, transpose, skiprows, comments)
        self._model = PreviewTableModel(data)
        self.setModel(self._model)

    @Slot()
    def parse_to_type(self, **kwargs):
        if False:
            i = 10
            return i + 15
        'Parse to a given type'
        indexes = self.selectedIndexes()
        if not indexes:
            return
        for index in indexes:
            self.model().parse_data_type(index, **kwargs)

    def contextMenuEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Reimplement Qt method'
        self.opt_menu.popup(event.globalPos())
        event.accept()

class PreviewWidget(QWidget):
    """Import wizard preview widget"""

    def __init__(self, parent):
        if False:
            return 10
        QWidget.__init__(self, parent)
        vert_layout = QVBoxLayout()
        type_layout = QHBoxLayout()
        type_label = QLabel(_('Import as'))
        type_layout.addWidget(type_label)
        self.array_btn = array_btn = QRadioButton(_('array'))
        available_array = np.ndarray is not FakeObject
        array_btn.setEnabled(available_array)
        array_btn.setChecked(available_array)
        type_layout.addWidget(array_btn)
        list_btn = QRadioButton(_('list'))
        list_btn.setChecked(not array_btn.isChecked())
        type_layout.addWidget(list_btn)
        if pd:
            self.df_btn = df_btn = QRadioButton(_('DataFrame'))
            df_btn.setChecked(False)
            type_layout.addWidget(df_btn)
        h_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        type_layout.addItem(h_spacer)
        type_frame = QFrame()
        type_frame.setLayout(type_layout)
        self._table_view = PreviewTable(self)
        vert_layout.addWidget(type_frame)
        vert_layout.addWidget(self._table_view)
        self.setLayout(vert_layout)

    def open_data(self, text, colsep=u'\t', rowsep=u'\n', transpose=False, skiprows=0, comments='#'):
        if False:
            while True:
                i = 10
        'Open clipboard text as table'
        if pd:
            self.pd_text = text
            self.pd_info = dict(sep=colsep, lineterminator=rowsep, skiprows=skiprows, comment=comments)
            if colsep is None:
                self.pd_info = dict(lineterminator=rowsep, skiprows=skiprows, comment=comments, delim_whitespace=True)
        self._table_view.process_data(text, colsep, rowsep, transpose, skiprows, comments)

    def get_data(self):
        if False:
            return 10
        'Return table data'
        return self._table_view.get_data()

class ImportWizard(BaseDialog):
    """Text data import wizard"""

    def __init__(self, parent, text, title=None, icon=None, contents_title=None, varname=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        if title is None:
            title = _('Import wizard')
        self.setWindowTitle(title)
        if icon is None:
            self.setWindowIcon(ima.icon('fileimport'))
        if contents_title is None:
            contents_title = _('Raw text')
        if varname is None:
            varname = _('variable_name')
        (self.var_name, self.clip_data) = (None, None)
        self.tab_widget = QTabWidget(self)
        self.text_widget = ContentsWidget(self, text)
        self.table_widget = PreviewWidget(self)
        self.tab_widget.addTab(self.text_widget, _('text'))
        self.tab_widget.setTabText(0, contents_title)
        self.tab_widget.addTab(self.table_widget, _('table'))
        self.tab_widget.setTabText(1, _('Preview'))
        self.tab_widget.setTabEnabled(1, False)
        name_layout = QHBoxLayout()
        name_label = QLabel(_('Variable Name'))
        name_layout.addWidget(name_label)
        self.name_edt = QLineEdit()
        self.name_edt.setText(varname)
        name_layout.addWidget(self.name_edt)
        btns_layout = QHBoxLayout()
        cancel_btn = QPushButton(_('Cancel'))
        btns_layout.addWidget(cancel_btn)
        cancel_btn.clicked.connect(self.reject)
        h_spacer = QSpacerItem(40, 20, QSizePolicy.Expanding, QSizePolicy.Minimum)
        btns_layout.addItem(h_spacer)
        self.back_btn = QPushButton(_('Previous'))
        self.back_btn.setEnabled(False)
        btns_layout.addWidget(self.back_btn)
        self.back_btn.clicked.connect(ft_partial(self._set_step, step=-1))
        self.fwd_btn = QPushButton(_('Next'))
        if not text:
            self.fwd_btn.setEnabled(False)
        btns_layout.addWidget(self.fwd_btn)
        self.fwd_btn.clicked.connect(ft_partial(self._set_step, step=1))
        self.done_btn = QPushButton(_('Done'))
        self.done_btn.setEnabled(False)
        btns_layout.addWidget(self.done_btn)
        self.done_btn.clicked.connect(self.process)
        self.text_widget.asDataChanged.connect(self.fwd_btn.setEnabled)
        self.text_widget.asDataChanged.connect(self.done_btn.setDisabled)
        layout = QVBoxLayout()
        layout.addLayout(name_layout)
        layout.addWidget(self.tab_widget)
        layout.addLayout(btns_layout)
        self.setLayout(layout)

    def _focus_tab(self, tab_idx):
        if False:
            i = 10
            return i + 15
        'Change tab focus'
        for i in range(self.tab_widget.count()):
            self.tab_widget.setTabEnabled(i, False)
        self.tab_widget.setTabEnabled(tab_idx, True)
        self.tab_widget.setCurrentIndex(tab_idx)

    def _set_step(self, step):
        if False:
            i = 10
            return i + 15
        'Proceed to a given step'
        new_tab = self.tab_widget.currentIndex() + step
        assert new_tab < self.tab_widget.count() and new_tab >= 0
        if new_tab == self.tab_widget.count() - 1:
            try:
                self.table_widget.open_data(self._get_plain_text(), self.text_widget.get_col_sep(), self.text_widget.get_row_sep(), self.text_widget.trnsp_box.isChecked(), self.text_widget.get_skiprows(), self.text_widget.get_comments())
                self.done_btn.setEnabled(True)
                self.done_btn.setDefault(True)
                self.fwd_btn.setEnabled(False)
                self.back_btn.setEnabled(True)
            except (SyntaxError, AssertionError) as error:
                QMessageBox.critical(self, _('Import wizard'), _('<b>Unable to proceed to next step</b><br><br>Please check your entries.<br><br>Error message:<br>%s') % str(error))
                return
        elif new_tab == 0:
            self.done_btn.setEnabled(False)
            self.fwd_btn.setEnabled(True)
            self.back_btn.setEnabled(False)
        self._focus_tab(new_tab)

    def get_data(self):
        if False:
            i = 10
            return i + 15
        'Return processed data'
        return (self.var_name, self.clip_data)

    def _simplify_shape(self, alist, rec=0):
        if False:
            i = 10
            return i + 15
        'Reduce the alist dimension if needed'
        if rec != 0:
            if len(alist) == 1:
                return alist[-1]
            return alist
        if len(alist) == 1:
            return self._simplify_shape(alist[-1], 1)
        return [self._simplify_shape(al, 1) for al in alist]

    def _get_table_data(self):
        if False:
            return 10
        'Return clipboard processed as data'
        data = self._simplify_shape(self.table_widget.get_data())
        if self.table_widget.array_btn.isChecked():
            return np.array(data)
        elif pd.read_csv is not FakeObject and self.table_widget.df_btn.isChecked():
            info = self.table_widget.pd_info
            buf = io.StringIO(self.table_widget.pd_text)
            return pd.read_csv(buf, **info)
        return data

    def _get_plain_text(self):
        if False:
            print('Hello World!')
        'Return clipboard as text'
        return self.text_widget.text_editor.toPlainText()

    @Slot()
    def process(self):
        if False:
            i = 10
            return i + 15
        'Process the data from clipboard'
        var_name = self.name_edt.text()
        try:
            self.var_name = str(var_name)
        except UnicodeEncodeError:
            self.var_name = to_text_string(var_name)
        if self.text_widget.get_as_data():
            self.clip_data = self._get_table_data()
        elif self.text_widget.get_as_code():
            self.clip_data = try_to_eval(to_text_string(self._get_plain_text()))
        else:
            self.clip_data = to_text_string(self._get_plain_text())
        self.accept()

def test(text):
    if False:
        while True:
            i = 10
    'Test'
    from spyder.utils.qthelpers import qapplication
    _app = qapplication()
    dialog = ImportWizard(None, text)
    if dialog.exec_():
        print(dialog.get_data())
if __name__ == '__main__':
    test(u'17/11/1976\t1.34\n14/05/09\t3.14')