"""Customized combobox widgets."""
import glob
import os
import os.path as osp
from qtpy.QtCore import QEvent, Qt, QTimer, QUrl, Signal, QSize
from qtpy.QtGui import QFont
from qtpy.QtWidgets import QComboBox, QCompleter, QLineEdit, QSizePolicy, QToolTip
from spyder.config.base import _
from spyder.py3compat import to_text_string
from spyder.utils.stylesheet import APP_STYLESHEET
from spyder.widgets.helperwidgets import ClearLineEdit, IconLineEdit

class BaseComboBox(QComboBox):
    """Editable combo box base class"""
    valid = Signal(bool, bool)
    sig_tab_pressed = Signal(bool)
    sig_resized = Signal(QSize, QSize)
    '\n    This signal is emitted to inform the widget has been resized.\n\n    Parameters\n    ----------\n    size: QSize\n        The new size of the widget.\n    old_size: QSize\n        The previous size of the widget.\n    '

    def __init__(self, parent):
        if False:
            i = 10
            return i + 15
        QComboBox.__init__(self, parent)
        self.setEditable(True)
        self.setCompleter(QCompleter(self))
        self.selected_text = self.currentText()

    def event(self, event):
        if False:
            return 10
        'Qt Override.\n\n        Filter tab keys and process double tab keys.\n        '
        if not isinstance(event, QEvent):
            return True
        if event.type() == QEvent.KeyPress and event.key() == Qt.Key_Tab:
            self.sig_tab_pressed.emit(True)
            return True
        return QComboBox.event(self, event)

    def keyPressEvent(self, event):
        if False:
            return 10
        'Qt Override.\n\n        Handle key press events.\n        '
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.add_current_text_if_valid():
                self.selected()
                self.hide_completer()
        elif event.key() == Qt.Key_Escape:
            self.set_current_text(self.selected_text)
            self.hide_completer()
        else:
            QComboBox.keyPressEvent(self, event)

    def resizeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        '\n        Emit a resize signal for widgets that need to adapt its size.\n        '
        super().resizeEvent(event)
        self.sig_resized.emit(event.size(), event.oldSize())

    def is_valid(self, qstr):
        if False:
            for i in range(10):
                print('nop')
        "\n        Return True if string is valid\n        Return None if validation can't be done\n        "
        pass

    def selected(self):
        if False:
            print('Hello World!')
        'Action to be executed when a valid item has been selected'
        self.valid.emit(True, True)

    def add_text(self, text):
        if False:
            return 10
        'Add text to combo box: add a new item if text is not found in\n        combo box items.'
        index = self.findText(text)
        while index != -1:
            self.removeItem(index)
            index = self.findText(text)
        self.insertItem(0, text)
        index = self.findText('')
        if index != -1:
            self.removeItem(index)
            self.insertItem(0, '')
            if text != '':
                self.setCurrentIndex(1)
            else:
                self.setCurrentIndex(0)
        else:
            self.setCurrentIndex(0)

    def set_current_text(self, text):
        if False:
            while True:
                i = 10
        'Sets the text of the QLineEdit of the QComboBox.'
        self.lineEdit().setText(to_text_string(text))

    def add_current_text(self):
        if False:
            while True:
                i = 10
        'Add current text to combo box history (convenient method)'
        text = self.currentText()
        self.add_text(text)

    def add_current_text_if_valid(self):
        if False:
            return 10
        'Add current text to combo box history if valid'
        valid = self.is_valid(self.currentText())
        if valid or valid is None:
            self.add_current_text()
            return True
        else:
            self.set_current_text(self.selected_text)

    def hide_completer(self):
        if False:
            print('Hello World!')
        'Hides the completion widget.'
        self.setCompleter(QCompleter([], self))

class PatternComboBox(BaseComboBox):
    """Search pattern combo box"""

    def __init__(self, parent, items=None, tip=None, adjust_to_minimum=True, id_=None):
        if False:
            for i in range(10):
                print('nop')
        BaseComboBox.__init__(self, parent)
        if adjust_to_minimum:
            self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        if items is not None:
            self.addItems(items)
        if tip is not None:
            self.setToolTip(tip)
        if id_ is not None:
            self.ID = id_
        self.setLineEdit(ClearLineEdit(self, reposition_button=True))

class EditableComboBox(BaseComboBox):
    """
    Editable combo box + Validate
    """

    def __init__(self, parent):
        if False:
            for i in range(10):
                print('nop')
        BaseComboBox.__init__(self, parent)
        self.font = QFont()
        self.selected_text = self.currentText()
        self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
        self.editTextChanged.connect(self.validate)
        self.tips = {True: _('Press enter to validate this entry'), False: _('This entry is incorrect')}

    def show_tip(self, tip=''):
        if False:
            print('Hello World!')
        'Show tip'
        QToolTip.showText(self.mapToGlobal(self.pos()), tip, self)

    def selected(self):
        if False:
            print('Hello World!')
        'Action to be executed when a valid item has been selected'
        BaseComboBox.selected(self)
        self.selected_text = self.currentText()

    def validate(self, qstr, editing=True):
        if False:
            i = 10
            return i + 15
        'Validate entered path'
        if self.selected_text == qstr and qstr != '':
            self.valid.emit(True, True)
            return
        valid = self.is_valid(qstr)
        if editing:
            if valid:
                self.valid.emit(True, False)
            else:
                self.valid.emit(False, False)

class PathComboBox(EditableComboBox):
    """
    QComboBox handling path locations
    """
    open_dir = Signal(str)

    def __init__(self, parent, adjust_to_contents=False, id_=None, elide_text=False, ellipsis_place=Qt.ElideLeft):
        if False:
            return 10
        EditableComboBox.__init__(self, parent)
        lineedit = IconLineEdit(self, elide_text=elide_text, ellipsis_place=ellipsis_place)
        if adjust_to_contents:
            self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        else:
            self.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLength)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.tips = {True: _('Press enter to validate this path'), False: ''}
        self.setLineEdit(lineedit)
        self.highlighted.connect(self.add_tooltip_to_highlighted_item)
        self.sig_tab_pressed.connect(self.tab_complete)
        self.valid.connect(lineedit.update_status)
        if id_ is not None:
            self.ID = id_

    def focusInEvent(self, event):
        if False:
            while True:
                i = 10
        'Handle focus in event restoring to display the status icon.'
        show_status = getattr(self.lineEdit(), 'show_status_icon', None)
        if show_status:
            show_status()
        QComboBox.focusInEvent(self, event)

    def focusOutEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        'Handle focus out event restoring the last valid selected path.'
        if not self.is_valid():
            lineedit = self.lineEdit()
            QTimer.singleShot(50, lambda : lineedit.setText(self.selected_text))
        hide_status = getattr(self.lineEdit(), 'hide_status_icon', None)
        if hide_status:
            hide_status()
        QComboBox.focusOutEvent(self, event)

    def _complete_options(self):
        if False:
            return 10
        'Find available completion options.'
        text = to_text_string(self.currentText())
        opts = glob.glob(text + '*')
        opts = sorted([opt for opt in opts if osp.isdir(opt)])
        completer = QCompleter(opts, self)
        qss = str(APP_STYLESHEET)
        completer.popup().setStyleSheet(qss)
        self.setCompleter(completer)
        return opts

    def tab_complete(self):
        if False:
            return 10
        '\n        If there is a single option available one tab completes the option.\n        '
        opts = self._complete_options()
        if len(opts) == 0:
            return
        if len(opts) == 1:
            self.set_current_text(opts[0] + os.sep)
            self.hide_completer()
        else:
            self.set_current_text(osp.commonprefix(opts))
            self.completer().complete()

    def is_valid(self, qstr=None):
        if False:
            print('Hello World!')
        'Return True if string is valid'
        if qstr is None:
            qstr = self.currentText()
        return osp.isdir(to_text_string(qstr))

    def selected(self):
        if False:
            print('Hello World!')
        'Action to be executed when a valid item has been selected'
        self.selected_text = self.currentText()
        self.valid.emit(True, True)
        self.open_dir.emit(self.selected_text)

    def add_current_text(self):
        if False:
            while True:
                i = 10
        '\n        Add current text to combo box history (convenient method).\n        If path ends in os separator ("" windows, "/" unix) remove it.\n        '
        text = self.currentText()
        if osp.isdir(text) and text:
            if text[-1] == os.sep:
                text = text[:-1]
        self.add_text(text)

    def add_tooltip_to_highlighted_item(self, index):
        if False:
            while True:
                i = 10
        '\n        Add a tooltip showing the full path of the currently highlighted item\n        of the PathComboBox.\n        '
        self.setItemData(index, self.itemText(index), Qt.ToolTipRole)

class UrlComboBox(PathComboBox):
    """
    QComboBox handling urls
    """

    def __init__(self, parent, adjust_to_contents=False, id_=None):
        if False:
            while True:
                i = 10
        PathComboBox.__init__(self, parent, adjust_to_contents)
        line_edit = QLineEdit(self)
        self.setLineEdit(line_edit)
        self.editTextChanged.disconnect(self.validate)
        if id_ is not None:
            self.ID = id_

    def is_valid(self, qstr=None):
        if False:
            for i in range(10):
                print('nop')
        'Return True if string is valid'
        if qstr is None:
            qstr = self.currentText()
        return QUrl(qstr).isValid()

class FileComboBox(PathComboBox):
    """
    QComboBox handling File paths
    """

    def __init__(self, parent=None, adjust_to_contents=False, default_line_edit=False):
        if False:
            return 10
        PathComboBox.__init__(self, parent, adjust_to_contents)
        if default_line_edit:
            line_edit = QLineEdit(self)
            self.setLineEdit(line_edit)
        if adjust_to_contents:
            self.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        else:
            self.setSizeAdjustPolicy(QComboBox.AdjustToContentsOnFirstShow)
            self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def is_valid(self, qstr=None):
        if False:
            for i in range(10):
                print('nop')
        'Return True if string is valid.'
        if qstr is None:
            qstr = self.currentText()
        valid = osp.isfile(to_text_string(qstr)) or osp.isdir(to_text_string(qstr))
        return valid

    def tab_complete(self):
        if False:
            while True:
                i = 10
        '\n        If there is a single option available one tab completes the option.\n        '
        opts = self._complete_options()
        if len(opts) == 1:
            text = opts[0]
            if osp.isdir(text):
                text = text + os.sep
            self.set_current_text(text)
            self.hide_completer()
        else:
            self.completer().complete()

    def _complete_options(self):
        if False:
            for i in range(10):
                print('nop')
        'Find available completion options.'
        text = to_text_string(self.currentText())
        opts = glob.glob(text + '*')
        opts = sorted([opt for opt in opts if osp.isdir(opt) or osp.isfile(opt)])
        completer = QCompleter(opts, self)
        qss = str(APP_STYLESHEET)
        completer.popup().setStyleSheet(qss)
        self.setCompleter(completer)
        return opts

def is_module_or_package(path):
    if False:
        while True:
            i = 10
    'Return True if path is a Python module/package'
    is_module = osp.isfile(path) and osp.splitext(path)[1] in ('.py', '.pyw')
    is_package = osp.isdir(path) and osp.isfile(osp.join(path, '__init__.py'))
    return is_module or is_package

class PythonModulesComboBox(PathComboBox):
    """
    QComboBox handling Python modules or packages path
    (i.e. .py, .pyw files *and* directories containing __init__.py)
    """

    def __init__(self, parent, adjust_to_contents=False, id_=None):
        if False:
            while True:
                i = 10
        PathComboBox.__init__(self, parent, adjust_to_contents)
        if id_ is not None:
            self.ID = id_

    def is_valid(self, qstr=None):
        if False:
            while True:
                i = 10
        'Return True if string is valid'
        if qstr is None:
            qstr = self.currentText()
        return is_module_or_package(to_text_string(qstr))

    def selected(self):
        if False:
            while True:
                i = 10
        'Action to be executed when a valid item has been selected'
        EditableComboBox.selected(self)
        self.open_dir.emit(self.currentText())