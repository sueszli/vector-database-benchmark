from __future__ import print_function
import math
import collections
import time
import sys
import logging
import psutil
import traceback as tb
import vaex
import smtplib
import platform
import getpass
import os
import vaex.utils
try:
    from urllib.request import urlopen
    from urllib.parse import urlparse, urlencode
except ImportError:
    from urlparse import urlparse
    from urllib import urlopen, urlencode, quote as urlquote
logger = logging.getLogger('vaex.ui.qt')
try:
    import sip
    sip.setapi('QVariant', 2)
    sip.setapi('QString', 2)
    from PyQt4 import QtGui, QtCore, QtTest
    sip.setapi('QVariant', 2)
    qt_version = QtCore.PYQT_VERSION_STR
    qt_mayor = int(qt_version[0])
except ImportError as e1:
    try:
        from PyQt5 import QtGui, QtCore, QtTest, QtWidgets
        for name in dir(QtWidgets):
            if name[0].lower() == 'q':
                setattr(QtGui, name, getattr(QtWidgets, name))
        qt_version = QtCore.PYQT_VERSION_STR
        qt_mayor = int(qt_version[0])
        QtGui.QStringListModel = QtCore.QStringListModel
    except ImportError as e1b:
        try:
            from PySide import QtGui, QtCore, QtTest
            QtCore.pyqtSignal = QtCore.Signal
            qt_version = QtCore.__version__
            qt_mayor = 4
        except ImportError as e2:
            print('could not import PyQt4, PyQt5, or PySide, please install', file=sys.stderr)
            print('errors: ', repr(e1), repr(e2), file=sys.stderr)
            sys.exit(1)

def attrsetter(object, attr_name):
    if False:
        while True:
            i = 10

    def setter(value):
        if False:
            for i in range(10):
                print('nop')
        setattr(object, attr_name, value)
    return setter

def attrgetter(object, attr_name):
    if False:
        print('Hello World!')

    def getter():
        if False:
            for i in range(10):
                print('nop')
        return getattr(object, attr_name)
    return getter

class ProgressExecution(object):

    def __init__(self, parent, title, cancel_text='Cancel', executor=None):
        if False:
            for i in range(10):
                print('nop')
        self.parent = parent
        self.title = title
        self.cancel_text = cancel_text
        self.executor = executor
        if self.executor:

            def begin():
                if False:
                    return 10
                self.time_begin = time.time()
                self.cancelled = False

            def end():
                if False:
                    for i in range(10):
                        print('nop')
                self.cancelled = False
                time_total = time.time() - self.time_begin

            def progress(fraction):
                if False:
                    i = 10
                    return i + 15
                return self.progress(fraction * 100)

            def cancel():
                if False:
                    while True:
                        i = 10
                pass
        self.tasks = []
        self._task_signals = []

    def execute(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('show dialog')
        if isinstance(self.executor, vaex.remote.ServerExecutor):
            self.dialog.exec_()
            logger.debug('dialog stopped')
            while any((task.isPending for task in self.tasks)):
                QtCore.QCoreApplication.instance().processEvents()
                time.sleep(0.01)
            self.finished_tasks()
        else:
            self.dialog.show()
            self.executor.execute()
            self.dialog.hide()
        logger.debug('self.dialog.wasCanceled() = %r', self.dialog.wasCanceled())
        return not self.dialog.wasCanceled()

    def add_task(self, task):
        if False:
            i = 10
            return i + 15
        self._task_signals.append(task.signal_progress.connect(self._on_progress))
        self.tasks.append(task)
        return task

    def _on_progress(self, fraction):
        if False:
            i = 10
            return i + 15
        total = self.get_progress_fraction()
        self.progress(total * 100)
        QtCore.QCoreApplication.instance().processEvents()
        ok = not self.dialog.wasCanceled()
        if total == 1:
            self.dialog.hide()
        return ok

    def get_progress_fraction(self):
        if False:
            for i in range(10):
                print('nop')
        total_fraction = 0
        for task in self.tasks:
            total_fraction += task.progress_fraction
        fraction = total_fraction / len(self.tasks)
        return fraction

    def finished_tasks(self):
        if False:
            print('Hello World!')
        for (task, signal) in zip(self.tasks, self._task_signals):
            task.signal_progress.disconnect(signal)
        self.tasks = []
        self._task_signals = []

    def __enter__(self):
        if False:
            return 10
        self.dialog = QtGui.QProgressDialog(self.title, self.cancel_text, 0, 1000, self.parent)
        self.dialog.setWindowModality(QtCore.Qt.WindowModal)
        self.dialog.setMinimumDuration(0)
        self.dialog.setAutoClose(True)
        self.dialog.setAutoReset(True)
        return self

    def progress(self, percentage):
        if False:
            print('Hello World!')
        self.dialog.setValue(int(percentage * 10))
        QtCore.QCoreApplication.instance().processEvents()
        return not self.dialog.wasCanceled()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        self.dialog.hide()
        if 0:
            self.executor.signal_begin.disconnect(self._begin_signal)
            self.executor.signal_progress.disconnect(self._progress_signal)
            self.executor.signal_end.disconnect(self._end_signal)
            self.executor.signal_cancel.disconnect(self._cancel_signal)

class assertError(object):

    def __init__(self, calls_expected=1):
        if False:
            i = 10
            return i + 15
        self.calls_expected = calls_expected

    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.called += 1

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        global dialog_error
        self.remember = dialog_error
        self.called = 0
        dialog_error = self.wrapper
        logger.debug('wrapped dialog_error')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            return 10
        global dialog_error
        assert self.called == self.calls_expected, 'expected the error dialog to be invoked %i time(s), was called %i times(s)' % (self.calls_expected, self.called)
        dialog_error = self.remember
        logger.debug('unwrapped dialog_error')

class settext(object):

    def __init__(self, value, calls_expected=1):
        if False:
            return 10
        self.value = value
        self.calls_expected = calls_expected

    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.called += 1
        return self.value

    def __enter__(self):
        if False:
            i = 10
            return i + 15
        global gettext
        self.remember = gettext
        self.called = 0
        gettext = self.wrapper
        logger.debug('wrapped gettext')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        global gettext
        assert self.called == self.calls_expected, 'expected the error dialog to be invoked %i time(s), was called %i times(s)' % (self.calls_expected, self.called)
        gettext = self.remember
        logger.debug('unwrapped gettext')

class setchoose(object):

    def __init__(self, value, calls_expected=1):
        if False:
            while True:
                i = 10
        self.value = value
        self.calls_expected = calls_expected

    def wrapper(self, *args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        self.called += 1
        return self.value

    def __enter__(self):
        if False:
            print('Hello World!')
        global choose
        self.remember = choose
        self.called = 0
        choose = self.wrapper
        logger.debug('wrapped choose')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            while True:
                i = 10
        global choose
        assert self.called == self.calls_expected, 'expected the error dialog to be invoked %i time(s), was called %i times(s)' % (self.calls_expected, self.called)
        choose = self.remember
        logger.debug('unwrapped choose')

class FakeProgressExecution(object):

    def __init__(self, *args):
        if False:
            i = 10
            return i + 15
        pass

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if False:
            i = 10
            return i + 15
        pass

    def progress(self, percentage):
        if False:
            print('Hello World!')
        return True

class OptionBase(object):

    def setVisible(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value:
            self.show()
        else:
            self.hide()

    def show(self):
        if False:
            for i in range(10):
                print('nop')
        self.combobox.show()
        self.label.show()

    def hide(self):
        if False:
            return 10
        self.combobox.hide()
        self.label.hide()

class Codeline(OptionBase):

    def __init__(self, parent, label, options, getter, setter, update=lambda : None):
        if False:
            while True:
                i = 10
        self.update = update
        self.options = options
        self.label = QtGui.QLabel(label, parent)
        self.combobox = QtGui.QComboBox(parent)
        self.combobox.addItems(options)
        self.combobox.setEditable(True)

        def wrap_setter(value, update=True):
            if False:
                for i in range(10):
                    print('nop')
            self.combobox.lineEdit().setText(value)
            setter(value)
            if update:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def on_change(index):
            if False:
                i = 10
                return i + 15
            on_edit_finished()

        def on_edit_finished():
            if False:
                i = 10
                return i + 15
            new_value = text = str(self.combobox.lineEdit().text())
            if new_value != self.current_value:
                self.current_value = new_value
                setter(self.current_value)
                update()
        self.combobox.currentIndexChanged.connect(on_change)
        self.combobox.lineEdit().editingFinished.connect(on_edit_finished)
        self.current_value = getter()
        self.combobox.lineEdit().setText(self.current_value)

    def add_to_grid_layout(self, row, grid_layout):
        if False:
            print('Hello World!')
        grid_layout.addWidget(self.label, row, 0)
        grid_layout.addWidget(self.combobox, row, 1)
        return row + 1

class Option(OptionBase):

    def __init__(self, parent, label, options, getter, setter, update=lambda : None):
        if False:
            while True:
                i = 10
        self.update = update
        self.options = options
        self.label = QtGui.QLabel(label, parent)
        self.combobox = QtGui.QComboBox(parent)
        self.combobox.addItems(options)

        def wrap_setter(value, update=True):
            if False:
                print('Hello World!')
            self.combobox.setCurrentIndex(options.index(getter()))
            setter(value)
            if update:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def on_change(index):
            if False:
                i = 10
                return i + 15
            setter(self.options[index])
            update()
        self.combobox.setCurrentIndex(options.index(getter()))
        self.combobox.currentIndexChanged.connect(on_change)

    def add_to_grid_layout(self, row, grid_layout):
        if False:
            for i in range(10):
                print('nop')
        grid_layout.addWidget(self.label, row, 0)
        grid_layout.addWidget(self.combobox, row, 1)
        return row + 1

class TextOption(OptionBase):

    def __init__(self, parent, label, value, placeholder, getter, setter, update=lambda : None):
        if False:
            print('Hello World!')
        self.update = update
        self.value = value
        self.placeholder = placeholder
        self.label = QtGui.QLabel(label, parent)
        self.textfield = QtGui.QLineEdit(parent)
        self.textfield.setPlaceholderText(self.get_placeholder())

        def wrap_setter(value, update=True):
            if False:
                i = 10
                return i + 15
            self.textfield.setText(value)
            setter(value)
            if update:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def on_change(*ignore):
            if False:
                i = 10
                return i + 15
            setter(self.textfield.text())
            update()
        self.textfield.returnPressed.connect(on_change)
        if 1:
            from vaex.ui.icons import iconfile
            self.tool_button = QtGui.QToolButton(parent)
            self.tool_button.setIcon(QtGui.QIcon(iconfile('gear')))
            self.tool_menu = QtGui.QMenu()
            self.tool_button.setMenu(self.tool_menu)
            self.tool_button.setPopupMode(QtGui.QToolButton.InstantPopup)
            if self.placeholder:

                def fill(_=None):
                    if False:
                        return 10
                    value = self.get_placeholder()
                    if value:
                        setter(value)
                        self.set_value(value)
                self.action_fill = QtGui.QAction('Fill in default value', parent)
                self.action_fill.triggered.connect(fill)
                self.tool_menu.addAction(self.action_fill)

            def copy(_=None):
                if False:
                    i = 10
                    return i + 15
                value = self.textfield.text()
                if value:
                    clipboard = QtGui.QApplication.clipboard()
                    text = str(value)
                    clipboard.setText(text)
            self.action_copy = QtGui.QAction('Copy', parent)
            self.action_copy.triggered.connect(copy)
            self.tool_menu.addAction(self.action_copy)

            def paste(_=None):
                if False:
                    return 10
                clipboard = QtGui.QApplication.clipboard()
                text = clipboard.text()
                setter(text)
                self.set_value(text)
            self.action_paste = QtGui.QAction('Paste', parent)
            self.action_paste.triggered.connect(paste)
            self.tool_menu.addAction(self.action_paste)

    def set_unit_completer(self):
        if False:
            while True:
                i = 10
        self.completer = vaex.ui.completer.UnitCompleter(self.textfield)
        self.textfield.setCompleter(self.completer)

    def get_placeholder(self):
        if False:
            return 10
        if callable(self.placeholder):
            return self.placeholder()
        else:
            return self.placeholder

    def add_to_grid_layout(self, row, grid_layout):
        if False:
            print('Hello World!')
        grid_layout.addWidget(self.label, row, 0)
        grid_layout.addWidget(self.textfield, row, 1)
        grid_layout.addWidget(self.tool_button, row, 2)
        return row + 1

class RangeOption(object):

    def __init__(self, parent, label, values, getter, setter, update=lambda : None):
        if False:
            while True:
                i = 10
        self.update = update
        self.values = [str(k) for k in values]
        self.label = QtGui.QLabel(label, parent)
        self.combobox_min = QtGui.QComboBox(parent)
        self.combobox_max = QtGui.QComboBox(parent)
        self.combobox_min.setEditable(True)
        self.combobox_max.setEditable(True)
        self.vmin = None
        self.vmax = None

        def wrap_setter(value, update=True):
            if False:
                i = 10
                return i + 15
            if value is None:
                (vmin, vmax) = (None, None)
            else:
                (vmin, vmax) = value
            self.combobox_min.blockSignals(True)
            self.combobox_max.blockSignals(True)
            changed = False
            if vmin != self.vmin:
                self.vmin = vmin
                self.combobox_min.lineEdit().setText(str(self.vmin) if self.vmin is not None else '')
                changed = True
            if vmax != self.vmax:
                self.vmax = vmax
                self.combobox_max.lineEdit().setText(str(self.vmax) if self.vmax is not None else '')
                changed = True
            self.combobox_min.blockSignals(False)
            self.combobox_max.blockSignals(False)
            if update and changed:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def get():
            if False:
                return 10
            if self.combobox_min.lineEdit().text().strip():
                try:
                    self.vmin = float(self.combobox_min.lineEdit().text())
                except:
                    logger.exception('parsing vmin')
                    dialog_error(self.combobox_min, 'Error parsing number', 'Cannot parse number: %s' % self.combobox_min.lineEdit().text())
            if self.combobox_max.lineEdit().text().strip():
                try:
                    self.vmax = float(self.combobox_max.lineEdit().text())
                except:
                    logger.exception('parsing vmax')
                    dialog_error(self.combobox_max, 'Error parsing number', 'Cannot parse number: %s' % self.combobox_max.lineEdit().text())
            return (self.vmin, self.vmax) if self.vmin is not None and self.vmax is not None else None

        def on_change(_ignore=None):
            if False:
                i = 10
                return i + 15
            value = get()
            if value:
                (vmin, vmax) = value
                if setter((vmin, vmax)):
                    update()
        self.combobox_min.lineEdit().returnPressed.connect(on_change)
        self.combobox_max.lineEdit().returnPressed.connect(on_change)
        self.combobox_layout = QtGui.QHBoxLayout(parent)
        self.combobox_layout.addWidget(self.combobox_min)
        self.combobox_layout.addWidget(self.combobox_max)
        if 1:
            from vaex.ui.icons import iconfile
            self.tool_button = QtGui.QToolButton(parent)
            self.tool_button.setIcon(QtGui.QIcon(iconfile('gear')))
            self.tool_menu = QtGui.QMenu()
            self.tool_button.setMenu(self.tool_menu)
            self.tool_button.setPopupMode(QtGui.QToolButton.InstantPopup)

            def flip(_=None):
                if False:
                    i = 10
                    return i + 15
                value = get()
                if value:
                    (vmin, vmax) = value
                    setter((vmax, vmin))
                    self.set_value((vmax, vmin))
            self.action_flip = QtGui.QAction('Flip axis', parent)
            self.action_flip.triggered.connect(flip)
            self.tool_menu.addAction(self.action_flip)

            def copy(_=None):
                if False:
                    i = 10
                    return i + 15
                value = get()
                if value:
                    clipboard = QtGui.QApplication.clipboard()
                    text = str(value)
                    clipboard.setText(text)
            self.action_copy = QtGui.QAction('Copy', parent)
            self.action_copy.triggered.connect(copy)
            self.tool_menu.addAction(self.action_copy)

            def paste(_=None):
                if False:
                    i = 10
                    return i + 15
                clipboard = QtGui.QApplication.clipboard()
                text = clipboard.text()
                try:
                    (vmin, vmax) = eval(text)
                    setter((vmin, vmax))
                    self.set_value((vmin, vmax))
                except Exception as e:
                    dialog_error(parent, 'Could not parse min/max values', 'Could not parse min/max values: %r' % e)
            self.action_paste = QtGui.QAction('Paste', parent)
            self.action_paste.triggered.connect(paste)
            self.tool_menu.addAction(self.action_paste)

    def add_to_grid_layout(self, row, grid_layout):
        if False:
            print('Hello World!')
        grid_layout.addWidget(self.label, row, 0)
        grid_layout.addWidget(self.combobox_min, row, 1)
        grid_layout.addWidget(self.tool_button, row, 2)
        row += 1
        grid_layout.addWidget(self.combobox_max, row, 1)
        return row + 1
color_list = [(255, 187, 120), (255, 127, 14), (174, 199, 232), (44, 160, 44), (31, 119, 180), (255, 152, 150), (214, 39, 40), (197, 176, 213), (152, 223, 138), (148, 103, 189), (247, 182, 210), (227, 119, 194), (196, 156, 148), (140, 86, 75), (127, 127, 127), (219, 219, 141), (199, 199, 199), (188, 189, 34), (158, 218, 229), (23, 190, 207)]

class ColorOption(object):

    def __init__(self, parent, label, getter, setter, update=lambda : None):
        if False:
            return 10
        self.update = update
        self.label = QtGui.QLabel(label, parent)
        self.combobox = QtGui.QComboBox(parent)
        index = 0
        self.qt_colors = []
        for color_tuple in color_list:
            self.combobox.addItem(','.join(map(str, color_tuple)))
            model = self.combobox.model().index(index, 0)
            color = QtGui.QColor(*color_tuple)
            self.combobox.model().setData(model, color, QtCore.Qt.BackgroundColorRole)
            index += 1
            self.qt_colors.append(color)

        def wrap_setter(value, update=True):
            if False:
                while True:
                    i = 10
            index = color_list.index(getter())
            self.combobox.setCurrentIndex(index)
            self.combobox.palette().setColor(QtGui.QPalette.Background, self.qt_colors[index])
            self.combobox.palette().setColor(QtGui.QPalette.Highlight, self.qt_colors[index])
            setter([c / 255.0 for c in value])
            if update:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def on_change(index):
            if False:
                return 10
            self.set_value(color_list[index])
            update()
        self.combobox.currentIndexChanged.connect(on_change)
        self.combobox.setCurrentIndex(color_list.index(getter()))

    def add_to_grid_layout(self, row, grid_layout):
        if False:
            for i in range(10):
                print('nop')
        grid_layout.addWidget(self.label, row, 0)
        grid_layout.addWidget(self.combobox, row, 1)
        return row + 1

class Checkbox(object):

    def __init__(self, parent, label_text, getter, setter, update=lambda : None):
        if False:
            i = 10
            return i + 15
        self.update = update
        self.label = QtGui.QLabel(label_text, parent)
        self.checkbox = QtGui.QCheckBox(parent)

        def wrap_setter(value, update=True):
            if False:
                for i in range(10):
                    print('nop')
            self.checkbox.setChecked(value)
            setter(value)
            if update:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def on_change(state):
            if False:
                while True:
                    i = 10
            value = state == QtCore.Qt.Checked
            setter(value)
            self.update()
        self.checkbox.setChecked(getter())
        self.checkbox.stateChanged.connect(on_change)

    def add_to_grid_layout(self, row, grid_layout, column_start=0):
        if False:
            i = 10
            return i + 15
        grid_layout.addWidget(self.label, row, column_start + 0)
        grid_layout.addWidget(self.checkbox, row, column_start + 1)
        return row + 1

class Slider(object):

    def __init__(self, parent, label_text, value_min, value_max, value_steps, getter, setter, name=None, format='{0:<0.3f}', transform=lambda x: x, inverse=lambda x: x, update=lambda : None, uselog=False, numeric_type=float):
        if False:
            while True:
                i = 10
        if name is None:
            name = label_text
        if uselog:

            def transform(x):
                if False:
                    while True:
                        i = 10
                return 10 ** x

            def inverse(x):
                if False:
                    return 10
                return math.log10(x)
        self.update = update
        self.label = QtGui.QLabel(label_text, parent)
        self.label_value = QtGui.QLabel(label_text, parent)
        self.slider = QtGui.QSlider(parent)
        self.slider.setOrientation(QtCore.Qt.Horizontal)
        self.slider.setRange(0, value_steps)

        def wrap_setter(value, update=True):
            if False:
                print('Hello World!')
            self.slider.setValue((inverse(value) - inverse(value_min)) / (inverse(value_max) - inverse(value_min)) * value_steps)
            setter(value)
            if update:
                self.update()
        setattr(self, 'get_value', getter)
        setattr(self, 'set_value', wrap_setter)

        def update_text():
            if False:
                for i in range(10):
                    print('nop')
            self.label_value.setText(format.format(getter()))

        def on_change(index, slider=self.slider):
            if False:
                return 10
            value = numeric_type(index / float(value_steps) * (inverse(value_max) - inverse(value_min)) + inverse(value_min))
            setter(transform(value))
            self.update()
            update_text()
        self.slider.setValue((inverse(getter()) - inverse(value_min)) * value_steps / (inverse(value_max) - inverse(value_min)))
        update_text()
        self.slider.valueChanged.connect(on_change)

    def add_to_grid_layout(self, row, grid_layout):
        if False:
            for i in range(10):
                print('nop')
        grid_layout.addWidget(self.label, row, 0)
        grid_layout.addWidget(self.slider, row, 1)
        grid_layout.addWidget(self.label_value, row, 2)
        return row + 1

class QuickDialog(QtGui.QDialog):

    def __init__(self, parent, title, validate=None):
        if False:
            while True:
                i = 10
        QtGui.QDialog.__init__(self, parent)
        self.setWindowTitle(title)
        self.layout = QtGui.QFormLayout(self)
        self.layout.setFieldGrowthPolicy(QtGui.QFormLayout.AllNonFixedFieldsGrow)
        self.values = {}
        self.widgets = {}
        self.validate = validate
        self.button_box = QtGui.QDialogButtonBox(QtGui.QDialogButtonBox.Ok | QtGui.QDialogButtonBox.Cancel, QtCore.Qt.Horizontal, self)
        self.button_box.accepted.connect(self.check_accept)
        self.button_box.rejected.connect(self.reject)
        self.setLayout(self.layout)

    def get(self):
        if False:
            i = 10
            return i + 15
        self.layout.addWidget(self.button_box)
        if self.exec_() == QtGui.QDialog.Accepted:
            return self.values
        else:
            return None

    def check_accept(self):
        if False:
            i = 10
            return i + 15
        logger.debug('on accepted')
        for (name, widget) in self.widgets.items():
            if isinstance(widget, QtGui.QLabel):
                pass
            elif isinstance(widget, QtGui.QLineEdit):
                self.values[name] = widget.text()
            elif isinstance(widget, QtGui.QComboBox):
                self.values[name] = widget.currentText()
            else:
                raise NotImplementedError
        if self.validate is None or self.validate(self, self.values):
            self.accept()

    def accept(self):
        if False:
            for i in range(10):
                print('nop')
        return QtGui.QDialog.accept(self)

    def add_label(self, name, label=''):
        if False:
            while True:
                i = 10
        self.widgets[name] = widget = QtGui.QLabel(label)
        self.layout.addRow('', widget)

    def add_text(self, name, label='', value='', placeholder=None):
        if False:
            return 10
        self.widgets[name] = widget = QtGui.QLineEdit(value, self)
        if placeholder:
            widget.setPlaceholderText(placeholder)
        self.layout.addRow(label, widget)

    def add_password(self, name, label='', value='', placeholder=None):
        if False:
            i = 10
            return i + 15
        self.widgets[name] = widget = QtGui.QLineEdit(value, self)
        widget.setEchoMode(QtGui.QLineEdit.Password)
        if placeholder:
            widget.setPlaceholderText(placeholder)
        self.layout.addRow(label, widget)

    def add_ucd(self, name, label='', value=''):
        if False:
            return 10
        self.widgets[name] = widget = QtGui.QLineEdit(value, self)
        widget.setCompleter(vaex.ui.completer.UCDCompleter(widget))
        self.layout.addRow(label, widget)

    def add_combo_edit(self, name, label='', value='', values=[]):
        if False:
            print('Hello World!')
        self.widgets[name] = widget = QtGui.QComboBox(self)
        widget.addItems([value] + values)
        widget.setEditable(True)
        self.layout.addRow(label, widget)

    def add_expression(self, name, label, value, dataset):
        if False:
            i = 10
            return i + 15
        import vaex.ui.completer
        self.widgets[name] = widget = vaex.ui.completer.ExpressionCombobox(self, dataset)
        if value is not None:
            widget.lineEdit().setText(value)
        self.layout.addRow(label, widget)

    def add_variable_expression(self, name, label, value, dataset):
        if False:
            return 10
        import vaex.ui.completer
        self.widgets[name] = widget = vaex.ui.completer.ExpressionCombobox(self, dataset, variables=True)
        if value is not None:
            widget.lineEdit().setText(value)
        self.layout.addRow(label, widget)

    def add_combo(self, name, label='', values=[]):
        if False:
            for i in range(10):
                print('nop')
        self.widgets[name] = widget = QtGui.QComboBox(self)
        widget.addItems(values)
        self.layout.addRow(label, widget)

def get_path_save(parent, title='Save file', path='', file_mask='HDF5 *.hdf5'):
    if False:
        for i in range(10):
            print('nop')
    path = QtGui.QFileDialog.getSaveFileName(parent, title, path, file_mask)
    if isinstance(path, tuple):
        path = str(path[0])
    return str(path)

def get_path_open(parent, title='Select file', path='', file_mask='HDF5 *.hdf5'):
    if False:
        return 10
    path = QtGui.QFileDialog.getOpenFileName(parent, title, path, file_mask)
    if isinstance(path, tuple):
        path = str(path[0])
    return str(path)

def getdir(parent, title, start_directory=''):
    if False:
        while True:
            i = 10
    result = QtGui.QFileDialog.getExistingDirectory(parent, title, '', QtGui.QFileDialog.ShowDirsOnly | QtGui.QFileDialog.DontResolveSymlinks)
    return None if result is None else str(result)

def gettext(parent, title, label, default=''):
    if False:
        for i in range(10):
            print('nop')
    (text, ok) = QtGui.QInputDialog.getText(parent, title, label, QtGui.QLineEdit.Normal, default)
    return str(text) if ok else None

class Thenner(object):

    def __init__(self, callback):
        if False:
            for i in range(10):
                print('nop')
        self.callback = callback
        self.thennable = False

    def then(self, *args, **kwargs):
        if False:
            print('Hello World!')
        self.thennable = True
        self.args = args
        self.kwargs = kwargs

    def do(self):
        if False:
            print('Hello World!')
        if self.thennable:
            self.callback(*self.args, **self.kwargs)

def set_choose(value, ok=None):
    if False:
        while True:
            i = 10
    global choose
    thenner = Thenner(set_choose)

    def wrapper(*args):
        if False:
            i = 10
            return i + 15
        thenner.do()
        return value
    choose = wrapper
    return thenner
QtGui.QInputDialog_getItem = QtGui.QInputDialog.getItem

def choose(parent, title, label, options, index=0, editable=False):
    if False:
        while True:
            i = 10
    (text, ok) = QtGui.QInputDialog_getItem(parent, title, label, options, index, editable)
    if editable:
        return text if ok else None
    else:
        return options.index(text) if ok else None

def set_select_many(ok, mask):
    if False:
        print('Hello World!')
    global select_many

    def wrapper(*args):
        if False:
            i = 10
            return i + 15
        return (ok, mask)
    select_many = wrapper

def select_many(parent, title, options):
    if False:
        while True:
            i = 10
    dialog = QtGui.QDialog(parent)
    dialog.setWindowTitle(title)
    dialog.setModal(True)
    layout = QtGui.QGridLayout(dialog)
    dialog.setLayout(layout)
    scroll_area = QtGui.QScrollArea(dialog)
    scroll_area.setWidgetResizable(True)
    frame = QtGui.QWidget(scroll_area)
    layout_frame = QtGui.QVBoxLayout(frame)
    frame.setLayout(layout_frame)
    checkboxes = [QtGui.QCheckBox(option, frame) for option in options]
    scroll_area.setWidget(frame)
    row = 0
    for checkbox in checkboxes:
        checkbox.setCheckState(QtCore.Qt.Checked)
        layout_frame.addWidget(checkbox)
        row += 1
    buttonLayout = QtGui.QHBoxLayout()
    button_ok = QtGui.QPushButton('Ok', dialog)
    button_cancel = QtGui.QPushButton('Cancel', dialog)
    button_cancel.clicked.connect(dialog.reject)
    button_ok.clicked.connect(dialog.accept)
    buttonLayout.addWidget(button_cancel)
    buttonLayout.addWidget(button_ok)
    layout.addWidget(scroll_area)
    layout.addLayout(buttonLayout, row, 0)
    value = dialog.exec_()
    mask = [checkbox.checkState() == QtCore.Qt.Checked for checkbox in checkboxes]
    return (value == QtGui.QDialog.Accepted, mask)

def memory_check_ok(parent, bytes_needed):
    if False:
        print('Hello World!')
    bytes_available = psutil.virtual_memory().available
    alot = bytes_needed / bytes_available > 0.5
    required = vaex.utils.filesize_format(bytes_needed)
    available = vaex.utils.filesize_format(bytes_available)
    msg = 'This action required {required} of memory, while you have {available}. Are you sure you want to continue?'.format(**locals())
    return not alot or dialog_confirm(parent, 'A lot of memory requested', msg)

def dialog_error(parent, title, msg):
    if False:
        return 10
    QtGui.QMessageBox.warning(parent, title, msg)

def dialog_info(parent, title, msg):
    if False:
        for i in range(10):
            print('nop')
    QtGui.QMessageBox.information(parent, title, msg)

def dialog_confirm(parent, title, msg, to_all=False):
    if False:
        print('Hello World!')
    msgbox = QtGui.QMessageBox(parent)
    msgbox.setText(msg)
    msgbox.setWindowTitle(title)
    msgbox.addButton(QtGui.QMessageBox.Yes)
    if to_all:
        msgbox.addButton(QtGui.QMessageBox.YesToAll)
        msgbox.setDefaultButton(QtGui.QMessageBox.YesToAll)
    else:
        msgbox.setDefaultButton(QtGui.QMessageBox.Yes)
    msgbox.addButton(QtGui.QMessageBox.No)
    result = msgbox.exec_()
    if to_all:
        return (result in [QtGui.QMessageBox.Yes, QtGui.QMessageBox.YesToAll], result == QtGui.QMessageBox.YesToAll)
    else:
        return result in [QtGui.QMessageBox.Yes]
confirm = dialog_confirm

def email(text):
    if False:
        i = 10
        return i + 15
    osname = platform.system().lower()
    if osname == 'linux':
        text = text.replace('#', '%23')
    body = urlquote(text)
    subject = urlquote('Error report for: ' + vaex.__full_name__)
    mailto = 'mailto:maartenbreddels@gmail.com?subject={subject}&body={body}'.format(**locals())
    print('open:', mailto)
    vaex.utils.os_open(mailto)

def old_email(text):
    if False:
        for i in range(10):
            print('nop')
    msg = MIMEText(text)
    msg['Subject'] = 'Error report for: ' + vaex.__full_name__
    email_from = 'vaex@astro.rug.nl'
    email_to = 'maartenbreddels@gmail.com'
    msg['From'] = email_to
    msg['To'] = email_to
    s = smtplib.SMTP('smtp.gmail.com')
    s.helo('fw1.astro.rug.nl')
    s.sendmail(email_to, [email_to], msg.as_string())
    s.quit()

def qt_exception(parent, exctype, value, traceback):
    if False:
        return 10
    trace_lines = tb.format_exception(exctype, value, traceback)
    trace = ''.join(trace_lines)
    print(trace)
    info = 'username: %r\n' % (getpass.getuser(),)
    info += 'program: %r\n' % vaex.__program_name__
    info += 'version: %r\n' % vaex.__version__
    info += 'full name: %r\n' % vaex.__full_name__
    info += 'arguments: %r\n' % sys.argv
    info += 'Qt version: %r\n' % qt_version
    attrs = sorted(dir(platform))
    for attr in attrs:
        if not attr.startswith('_') and attr not in ['popen', 'system_alias']:
            f = getattr(platform, attr)
            if isinstance(f, collections.Callable):
                try:
                    info += '%s: %r\n' % (attr, f())
                except:
                    pass
    report = info + '\n' + trace
    text = 'An unexpected error occured, you may press ok and continue, but the program might be unstable.\n\n' + report
    dialog = QtGui.QMessageBox(parent)
    dialog.setText('Unexpected error: %s\nDo you want to continue' % (exctype,))
    dialog.setDetailedText(text)
    buttonSend = QtGui.QPushButton('Email report', dialog)
    buttonQuit = QtGui.QPushButton('Quit program', dialog)
    buttonContinue = QtGui.QPushButton('Continue', dialog)

    def exit(ignore=None):
        if False:
            for i in range(10):
                print('nop')
        print('exit')
        sys.exit(1)

    def _email(ignore=None):
        if False:
            i = 10
            return i + 15
        if QtGui.QMessageBox.information(dialog, 'Send report', 'Confirm that you want to send a report', QtGui.QMessageBox.Abort | QtGui.QMessageBox.Yes) == QtGui.QMessageBox.Yes:
            email(report)
    buttonQuit.clicked.connect(exit)
    buttonSend.clicked.connect(_email)
    dialog.addButton(buttonSend, QtGui.QMessageBox.YesRole)
    dialog.addButton(buttonQuit, QtGui.QMessageBox.NoRole)
    dialog.addButton(buttonContinue, QtGui.QMessageBox.YesRole)
    dialog.setDefaultButton(buttonSend)
    dialog.setEscapeButton(buttonContinue)
    dialog.raise_()
    dialog.exec_()