__license__ = 'GPL v3'
__copyright__ = '2010, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import os
from collections import OrderedDict
from functools import partial
from qt.core import QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QGridLayout, QGroupBox, QHBoxLayout, QIcon, QLabel, QLineEdit, QMessageBox, QPlainTextEdit, QSizePolicy, QSpacerItem, QSpinBox, QStyle, Qt, QToolButton, QUrl, QVBoxLayout, QWidget
from calibre.ebooks.metadata import title_sort
from calibre.gui2 import UNDEFINED_QDATETIME, elided_text, error_dialog, gprefs
from calibre.gui2.comments_editor import Editor as CommentsEditor
from calibre.gui2.complete2 import EditWithComplete as EWC
from calibre.gui2.dialogs.tag_editor import TagEditor
from calibre.gui2.library.delegates import ClearingDoubleSpinBox, ClearingSpinBox
from calibre.gui2.markdown_editor import Editor as MarkdownEditor
from calibre.gui2.widgets2 import DateTimeEdit as DateTimeEditBase, RatingEditor
from calibre.library.comments import comments_to_html
from calibre.utils.config import tweaks
from calibre.utils.date import as_local_time, as_utc, internal_iso_format_string, is_date_undefined, now, qt_from_dt, qt_to_dt
from calibre.utils.icu import lower as icu_lower, sort_key

class EditWithComplete(EWC):

    def __init__(self, *a, **kw):
        if False:
            i = 10
            return i + 15
        super().__init__(*a, **kw)
        self.set_clear_button_enabled(False)

def safe_disconnect(signal):
    if False:
        for i in range(10):
            print('nop')
    try:
        signal.disconnect()
    except Exception:
        pass

def label_string(txt):
    if False:
        i = 10
        return i + 15
    if txt:
        try:
            if txt[0].isalnum():
                return '&' + txt
        except:
            pass
    return txt

def get_tooltip(col_metadata, add_index=False):
    if False:
        print('Hello World!')
    key = col_metadata['label'] + ('_index' if add_index else '')
    label = col_metadata['name'] + (_(' index') if add_index else '')
    description = col_metadata.get('display', {}).get('description', '')
    return '{} (#{}){} {}'.format(label, key, ':' if description else '', description).strip()

class Base:

    def __init__(self, db, col_id, parent=None):
        if False:
            for i in range(10):
                print('nop')
        (self.db, self.col_id) = (db, col_id)
        self.book_id = None
        self.col_metadata = db.custom_column_num_map[col_id]
        self.initial_val = self.widgets = None
        self.signals_to_disconnect = []
        self.setup_ui(parent)
        description = get_tooltip(self.col_metadata)
        try:
            self.widgets[0].setToolTip(description)
            self.widgets[1].setToolTip(description)
        except:
            try:
                self.widgets[1].setToolTip(description)
            except:
                pass

    def finish_ui_setup(self, parent, edit_widget):
        if False:
            while True:
                i = 10
        self.was_none = False
        w = QWidget(parent)
        self.widgets.append(w)
        l = QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        w.setLayout(l)
        self.editor = editor = edit_widget(parent)
        l.addWidget(editor)
        self.clear_button = QToolButton(parent)
        self.clear_button.setIcon(QIcon.ic('trash.png'))
        self.clear_button.clicked.connect(self.set_to_undefined)
        self.clear_button.setToolTip(_('Clear {0}').format(self.col_metadata['name']))
        l.addWidget(self.clear_button)

    def initialize(self, book_id):
        if False:
            print('Hello World!')
        self.book_id = book_id
        val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
        val = self.normalize_db_val(val)
        self.setter(val)
        self.initial_val = self.current_val

    @property
    def current_val(self):
        if False:
            i = 10
            return i + 15
        return self.normalize_ui_val(self.gui_val)

    @property
    def gui_val(self):
        if False:
            while True:
                i = 10
        return self.getter()

    def commit(self, book_id, notify=False):
        if False:
            return 10
        val = self.current_val
        if val != self.initial_val:
            return self.db.set_custom(book_id, val, num=self.col_id, notify=notify, commit=False, allow_case_change=True)
        else:
            return set()

    def apply_to_metadata(self, mi):
        if False:
            i = 10
            return i + 15
        mi.set('#' + self.col_metadata['label'], self.current_val)

    def normalize_db_val(self, val):
        if False:
            i = 10
            return i + 15
        return val

    def normalize_ui_val(self, val):
        if False:
            print('Hello World!')
        return val

    def break_cycles(self):
        if False:
            return 10
        self.db = self.widgets = self.initial_val = None
        for signal in self.signals_to_disconnect:
            safe_disconnect(signal)
        self.signals_to_disconnect = []

    def connect_data_changed(self, slot):
        if False:
            for i in range(10):
                print('nop')
        pass

    def values_changed(self):
        if False:
            i = 10
            return i + 15
        return self.getter() != self.initial_val and (self.getter() or self.initial_val)

    def edit(self):
        if False:
            for i in range(10):
                print('nop')
        if self.values_changed():
            d = _save_dialog(self.parent, _('Values changed'), _('You have changed the values. In order to use this editor, you must either discard or apply these changes. Apply changes?'))
            if d == QMessageBox.StandardButton.Cancel:
                return
            if d == QMessageBox.StandardButton.Yes:
                self.commit(self.book_id)
                self.db.commit()
                self.initial_val = self.current_val
            else:
                self.setter(self.initial_val)
        from calibre.gui2.ui import get_gui
        get_gui().do_tags_list_edit(None, self.key)
        self.initialize(self.book_id)

class SimpleText(Base):

    def setup_ui(self, parent):
        if False:
            for i in range(10):
                print('nop')
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, QLineEdit)

    def set_to_undefined(self):
        if False:
            while True:
                i = 10
        self.editor.setText('')

    def setter(self, val):
        if False:
            while True:
                i = 10
        self.editor.setText(str(val or ''))

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        return self.editor.text().strip()

    def connect_data_changed(self, slot):
        if False:
            for i in range(10):
                print('nop')
        self.editor.textChanged.connect(slot)
        self.signals_to_disconnect.append(self.editor.textChanged)

class LongText(Base):

    def setup_ui(self, parent):
        if False:
            while True:
                i = 10
        self._box = QGroupBox(parent)
        self._box.setTitle(label_string(self.col_metadata['name']))
        self._layout = QVBoxLayout()
        self._tb = QPlainTextEdit(self._box)
        self._tb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout.addWidget(self._tb)
        self._box.setLayout(self._layout)
        self.widgets = [self._box]

    def setter(self, val):
        if False:
            while True:
                i = 10
        self._tb.setPlainText(str(val or ''))

    def getter(self):
        if False:
            i = 10
            return i + 15
        return self._tb.toPlainText()

    def connect_data_changed(self, slot):
        if False:
            for i in range(10):
                print('nop')
        self._tb.textChanged.connect(slot)
        self.signals_to_disconnect.append(self._tb.textChanged)

class Bool(Base):

    def setup_ui(self, parent):
        if False:
            i = 10
            return i + 15
        name = self.col_metadata['name']
        self.widgets = [QLabel(label_string(name), parent)]
        w = QWidget(parent)
        self.widgets.append(w)
        l = QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        w.setLayout(l)
        self.combobox = QComboBox(parent)
        l.addWidget(self.combobox)
        c = QToolButton(parent)
        c.setIcon(QIcon.ic('ok.png'))
        c.setToolTip(_('Set {} to yes').format(name))
        l.addWidget(c)
        c.clicked.connect(self.set_to_yes)
        c = QToolButton(parent)
        c.setIcon(QIcon.ic('list_remove.png'))
        c.setToolTip(_('Set {} to no').format(name))
        l.addWidget(c)
        c.clicked.connect(self.set_to_no)
        if self.db.new_api.pref('bools_are_tristate'):
            c = QToolButton(parent)
            c.setIcon(QIcon.ic('trash.png'))
            c.setToolTip(_('Clear {}').format(name))
            l.addWidget(c)
            c.clicked.connect(self.set_to_cleared)
        w = self.combobox
        items = [_('Yes'), _('No'), _('Undefined')]
        icons = ['ok.png', 'list_remove.png', 'blank.png']
        if not self.db.new_api.pref('bools_are_tristate'):
            items = items[:-1]
            icons = icons[:-1]
        for (icon, text) in zip(icons, items):
            w.addItem(QIcon.ic(icon), text)

    def setter(self, val):
        if False:
            i = 10
            return i + 15
        val = {None: 2, False: 1, True: 0}[val]
        if not self.db.new_api.pref('bools_are_tristate') and val == 2:
            val = 1
        self.combobox.setCurrentIndex(val)

    def getter(self):
        if False:
            while True:
                i = 10
        val = self.combobox.currentIndex()
        return {2: None, 1: False, 0: True}[val]

    def set_to_yes(self):
        if False:
            i = 10
            return i + 15
        self.combobox.setCurrentIndex(0)

    def set_to_no(self):
        if False:
            for i in range(10):
                print('nop')
        self.combobox.setCurrentIndex(1)

    def set_to_cleared(self):
        if False:
            while True:
                i = 10
        self.combobox.setCurrentIndex(2)

    def connect_data_changed(self, slot):
        if False:
            return 10
        self.combobox.currentTextChanged.connect(slot)
        self.signals_to_disconnect.append(self.combobox.currentTextChanged)

class Int(Base):

    def setup_ui(self, parent):
        if False:
            return 10
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, ClearingSpinBox)
        self.editor.setRange(-1000000, 100000000)

    def finish_ui_setup(self, parent, edit_widget):
        if False:
            i = 10
            return i + 15
        Base.finish_ui_setup(self, parent, edit_widget)
        self.editor.setSpecialValueText(_('Undefined'))
        self.editor.setSingleStep(1)
        self.editor.valueChanged.connect(self.valueChanged)

    def setter(self, val):
        if False:
            return 10
        if val is None:
            val = self.editor.minimum()
        self.editor.setValue(val)
        self.was_none = val == self.editor.minimum()

    def getter(self):
        if False:
            print('Hello World!')
        val = self.editor.value()
        if val == self.editor.minimum():
            val = None
        return val

    def valueChanged(self, to_what):
        if False:
            for i in range(10):
                print('nop')
        if self.was_none and to_what == -999999:
            self.setter(0)
        self.was_none = to_what == self.editor.minimum()

    def connect_data_changed(self, slot):
        if False:
            while True:
                i = 10
        self.editor.valueChanged.connect(slot)
        self.signals_to_disconnect.append(self.editor.valueChanged)

    def set_to_undefined(self):
        if False:
            i = 10
            return i + 15
        self.editor.setValue(-1000000)

class Float(Int):

    def setup_ui(self, parent):
        if False:
            return 10
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, ClearingDoubleSpinBox)
        self.editor.setRange(-1000000.0, float(100000000))
        self.editor.setDecimals(int(self.col_metadata['display'].get('decimals', 2)))

class Rating(Base):

    def setup_ui(self, parent):
        if False:
            i = 10
            return i + 15
        allow_half_stars = self.col_metadata['display'].get('allow_half_stars', False)
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, partial(RatingEditor, is_half_star=allow_half_stars))

    def set_to_undefined(self):
        if False:
            while True:
                i = 10
        self.editor.setCurrentIndex(0)

    def setter(self, val):
        if False:
            for i in range(10):
                print('nop')
        val = max(0, min(int(val or 0), 10))
        self.editor.rating_value = val

    def getter(self):
        if False:
            while True:
                i = 10
        return self.editor.rating_value or None

    def connect_data_changed(self, slot):
        if False:
            return 10
        self.editor.currentTextChanged.connect(slot)
        self.signals_to_disconnect.append(self.editor.currentTextChanged)

class DateTimeEdit(DateTimeEditBase):

    def focusInEvent(self, x):
        if False:
            i = 10
            return i + 15
        self.setSpecialValueText('')
        DateTimeEditBase.focusInEvent(self, x)

    def focusOutEvent(self, x):
        if False:
            print('Hello World!')
        self.setSpecialValueText(_('Undefined'))
        DateTimeEditBase.focusOutEvent(self, x)

    def set_to_today(self):
        if False:
            while True:
                i = 10
        self.setDateTime(qt_from_dt(now()))

    def set_to_clear(self):
        if False:
            print('Hello World!')
        self.setDateTime(UNDEFINED_QDATETIME)

class DateTime(Base):

    def setup_ui(self, parent):
        if False:
            while True:
                i = 10
        cm = self.col_metadata
        self.widgets = [QLabel(label_string(cm['name']), parent)]
        w = QWidget(parent)
        self.widgets.append(w)
        l = QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        w.setLayout(l)
        self.dte = dte = DateTimeEdit(parent)
        format_ = cm['display'].get('date_format', '')
        if not format_:
            format_ = 'dd MMM yyyy hh:mm'
        elif format_ == 'iso':
            format_ = internal_iso_format_string()
        dte.setDisplayFormat(format_)
        dte.setCalendarPopup(True)
        dte.setMinimumDateTime(UNDEFINED_QDATETIME)
        dte.setSpecialValueText(_('Undefined'))
        l.addWidget(dte)
        self.today_button = QToolButton(parent)
        self.today_button.setText(_('Today'))
        self.today_button.clicked.connect(dte.set_to_today)
        l.addWidget(self.today_button)
        self.clear_button = QToolButton(parent)
        self.clear_button.setIcon(QIcon.ic('trash.png'))
        self.clear_button.clicked.connect(dte.set_to_clear)
        self.clear_button.setToolTip(_('Clear {0}').format(self.col_metadata['name']))
        l.addWidget(self.clear_button)
        self.connect_data_changed(self.set_tooltip)

    def set_tooltip(self, val):
        if False:
            for i in range(10):
                print('nop')
        if is_date_undefined(val):
            self.dte.setToolTip(get_tooltip(self.col_metadata, False))
        else:
            self.dte.setToolTip(get_tooltip(self.col_metadata, False) + '\n' + _('Exact time: {}').format(as_local_time(qt_to_dt(val))))

    def setter(self, val):
        if False:
            for i in range(10):
                print('nop')
        if val is None:
            val = self.dte.minimumDateTime()
        else:
            val = qt_from_dt(val)
        self.dte.setDateTime(val)

    def getter(self):
        if False:
            print('Hello World!')
        val = self.dte.dateTime()
        if is_date_undefined(val):
            val = None
        else:
            val = qt_to_dt(val)
        return val

    def normalize_db_val(self, val):
        if False:
            while True:
                i = 10
        return as_local_time(val) if val is not None else None

    def normalize_ui_val(self, val):
        if False:
            for i in range(10):
                print('nop')
        return as_utc(val) if val is not None else None

    def connect_data_changed(self, slot):
        if False:
            print('Hello World!')
        self.dte.dateTimeChanged.connect(slot)
        self.signals_to_disconnect.append(self.dte.dateTimeChanged)

class Comments(Base):

    def setup_ui(self, parent):
        if False:
            for i in range(10):
                print('nop')
        self._box = QGroupBox(parent)
        self._box.setTitle(label_string(self.col_metadata['name']))
        self._layout = QVBoxLayout()
        self._tb = CommentsEditor(self._box, toolbar_prefs_name='metadata-comments-editor-widget-hidden-toolbars')
        self._tb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout.addWidget(self._tb)
        self._box.setLayout(self._layout)
        self.widgets = [self._box]

    def initialize(self, book_id):
        if False:
            while True:
                i = 10
        path = self.db.abspath(book_id, index_is_id=True)
        if path:
            self._tb.set_base_url(QUrl.fromLocalFile(os.path.join(path, 'metadata.html')))
        return Base.initialize(self, book_id)

    def setter(self, val):
        if False:
            i = 10
            return i + 15
        if not val or not val.strip():
            val = ''
        else:
            val = comments_to_html(val)
        self._tb.html = val
        self._tb.wyswyg_dirtied()

    def getter(self):
        if False:
            i = 10
            return i + 15
        val = str(self._tb.html).strip()
        if not val:
            val = None
        return val

    @property
    def tab(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tb.tab

    @tab.setter
    def tab(self, val):
        if False:
            for i in range(10):
                print('nop')
        self._tb.tab = val

    def connect_data_changed(self, slot):
        if False:
            while True:
                i = 10
        self._tb.data_changed.connect(slot)
        self.signals_to_disconnect.append(self._tb.data_changed)

class Markdown(Base):

    def setup_ui(self, parent):
        if False:
            for i in range(10):
                print('nop')
        self._box = QGroupBox(parent)
        self._box.setTitle(label_string(self.col_metadata['name']))
        self._layout = QVBoxLayout()
        self._tb = MarkdownEditor(self._box)
        self._tb.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self._layout.addWidget(self._tb)
        self._box.setLayout(self._layout)
        self.widgets = [self._box]

    def initialize(self, book_id):
        if False:
            i = 10
            return i + 15
        path = self.db.abspath(book_id, index_is_id=True)
        if path:
            self._tb.set_base_url(QUrl.fromLocalFile(os.path.join(path, 'metadata.html')))
        return super().initialize(book_id)

    def setter(self, val):
        if False:
            while True:
                i = 10
        self._tb.markdown = str(val or '').strip()

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        val = self._tb.markdown.strip()
        if not val:
            val = None
        return val

    @property
    def tab(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tb.tab

    @tab.setter
    def tab(self, val):
        if False:
            return 10
        self._tb.tab = val

    def connect_data_changed(self, slot):
        if False:
            print('Hello World!')
        self._tb.editor.textChanged.connect(slot)
        self.signals_to_disconnect.append(self._tb.editor.textChanged)

class MultipleWidget(QWidget):

    def __init__(self, parent, only_manage_items=False, widget=EditWithComplete, name=None):
        if False:
            return 10
        QWidget.__init__(self, parent)
        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        self.edit_widget = widget(parent)
        layout.addWidget(self.edit_widget, stretch=1000)
        self.editor_button = QToolButton(self)
        if name is None:
            name = _('items')
        if only_manage_items:
            self.editor_button.setToolTip(_('Open the Manage {} window').format(name))
        else:
            self.editor_button.setToolTip(_('Open the {0} editor. If Ctrl or Shift is pressed, open the Manage {0} window').format(name))
        self.editor_button.setIcon(QIcon.ic('chapters.png'))
        layout.addWidget(self.editor_button)
        self.setLayout(layout)

    def get_editor_button(self):
        if False:
            print('Hello World!')
        return self.editor_button

    def update_items_cache(self, values):
        if False:
            while True:
                i = 10
        self.edit_widget.update_items_cache(values)

    def clear(self):
        if False:
            print('Hello World!')
        self.edit_widget.clear()

    def setEditText(self):
        if False:
            return 10
        self.edit_widget.setEditText()

    def addItem(self, itm):
        if False:
            return 10
        self.edit_widget.addItem(itm)

    def set_separator(self, sep):
        if False:
            for i in range(10):
                print('nop')
        self.edit_widget.set_separator(sep)

    def set_add_separator(self, sep):
        if False:
            for i in range(10):
                print('nop')
        self.edit_widget.set_add_separator(sep)

    def set_space_before_sep(self, v):
        if False:
            print('Hello World!')
        self.edit_widget.set_space_before_sep(v)

    def setSizePolicy(self, v1, v2):
        if False:
            for i in range(10):
                print('nop')
        self.edit_widget.setSizePolicy(v1, v2)

    def setText(self, v):
        if False:
            return 10
        self.edit_widget.setText(v)

    def text(self):
        if False:
            while True:
                i = 10
        return self.edit_widget.text()

def _save_dialog(parent, title, msg, det_msg=''):
    if False:
        print('Hello World!')
    d = QMessageBox(parent)
    d.setWindowTitle(title)
    d.setText(msg)
    d.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No | QMessageBox.StandardButton.Cancel)
    return d.exec()

class Text(Base):

    def setup_ui(self, parent):
        if False:
            while True:
                i = 10
        self.sep = self.col_metadata['multiple_seps']
        self.key = self.db.field_metadata.label_to_key(self.col_metadata['label'], prefer_custom=True)
        self.parent = parent
        if self.col_metadata['is_multiple']:
            w = MultipleWidget(parent, name=self.col_metadata['name'])
            w.set_separator(self.sep['ui_to_list'])
            if self.sep['ui_to_list'] == '&':
                w.set_space_before_sep(True)
                w.set_add_separator(tweaks['authors_completer_append_separator'])
            w.get_editor_button().clicked.connect(self.edit)
        else:
            w = MultipleWidget(parent, only_manage_items=True, name=self.col_metadata['name'])
            w.set_separator(None)
            w.get_editor_button().clicked.connect(super().edit)
        w.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.set_to_undefined = w.clear
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, lambda parent: w)

    def initialize(self, book_id):
        if False:
            print('Hello World!')
        values = list(self.db.all_custom(num=self.col_id))
        values.sort(key=sort_key)
        self.book_id = book_id
        self.editor.clear()
        self.editor.update_items_cache(values)
        val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
        if isinstance(val, list):
            if not self.col_metadata.get('display', {}).get('is_names', False):
                val.sort(key=sort_key)
        val = self.normalize_db_val(val)
        if self.col_metadata['is_multiple']:
            self.setter(val)
        else:
            self.editor.setText(val)
        self.initial_val = self.current_val

    def setter(self, val):
        if False:
            return 10
        if self.col_metadata['is_multiple']:
            if not val:
                val = []
            self.editor.setText(self.sep['list_to_ui'].join(val))

    def getter(self):
        if False:
            i = 10
            return i + 15
        val = str(self.editor.text()).strip()
        if self.col_metadata['is_multiple']:
            ans = [x.strip() for x in val.split(self.sep['ui_to_list']) if x.strip()]
            if not ans:
                ans = None
            return ans
        if not val:
            val = None
        return val

    def edit(self):
        if False:
            i = 10
            return i + 15
        ctrl_or_shift_pressed = QApplication.keyboardModifiers() & (Qt.KeyboardModifier.ControlModifier | Qt.KeyboardModifier.ShiftModifier)
        if self.getter() != self.initial_val and (self.getter() or self.initial_val):
            d = _save_dialog(self.parent, _('Values changed'), _('You have changed the values. In order to use this editor, you must either discard or apply these changes. Apply changes?'))
            if d == QMessageBox.StandardButton.Cancel:
                return
            if d == QMessageBox.StandardButton.Yes:
                self.commit(self.book_id)
                self.db.commit()
                self.initial_val = self.current_val
            else:
                self.setter(self.initial_val)
        if ctrl_or_shift_pressed:
            from calibre.gui2.ui import get_gui
            get_gui().do_tags_list_edit(None, self.key)
            self.initialize(self.book_id)
        else:
            d = TagEditor(self.parent, self.db, self.book_id, self.key)
            if d.exec() == QDialog.DialogCode.Accepted:
                self.setter(d.tags)

    def connect_data_changed(self, slot):
        if False:
            while True:
                i = 10
        s = self.editor.edit_widget.currentTextChanged
        s.connect(slot)
        self.signals_to_disconnect.append(s)

class Series(Base):

    def setup_ui(self, parent):
        if False:
            return 10
        self.parent = parent
        self.key = self.db.field_metadata.label_to_key(self.col_metadata['label'], prefer_custom=True)
        w = MultipleWidget(parent, only_manage_items=True, name=self.col_metadata['name'])
        w.get_editor_button().clicked.connect(self.edit)
        w.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.set_to_undefined = w.clear
        w.set_separator(None)
        self.name_widget = w.edit_widget
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, lambda parent: w)
        self.name_widget.editTextChanged.connect(self.series_changed)
        w = QLabel(label_string(self.col_metadata['name']) + _(' index'), parent)
        w.setToolTip(get_tooltip(self.col_metadata, add_index=True))
        self.widgets.append(w)
        w = QDoubleSpinBox(parent)
        w.setRange(-10000.0, float(100000000))
        w.setDecimals(2)
        w.setSingleStep(1)
        self.idx_widget = w
        w.setToolTip(get_tooltip(self.col_metadata, add_index=True))
        self.widgets.append(w)

    def set_to_undefined(self):
        if False:
            return 10
        self.name_widget.clearEditText()
        self.idx_widget.setValue(1.0)

    def initialize(self, book_id):
        if False:
            print('Hello World!')
        self.book_id = book_id
        values = list(self.db.all_custom(num=self.col_id))
        values.sort(key=sort_key)
        val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
        s_index = self.db.get_custom_extra(book_id, num=self.col_id, index_is_id=True)
        try:
            s_index = float(s_index)
        except (ValueError, TypeError):
            s_index = 1.0
        self.idx_widget.setValue(s_index)
        val = self.normalize_db_val(val)
        self.name_widget.blockSignals(True)
        self.name_widget.update_items_cache(values)
        self.name_widget.setText(val)
        self.name_widget.blockSignals(False)
        (self.initial_val, self.initial_index) = self.current_val

    def getter(self):
        if False:
            print('Hello World!')
        n = str(self.name_widget.currentText()).strip()
        i = self.idx_widget.value()
        return (n, i)

    def series_changed(self, val):
        if False:
            return 10
        (val, s_index) = self.gui_val
        if tweaks['series_index_auto_increment'] == 'no_change':
            pass
        elif tweaks['series_index_auto_increment'] == 'const':
            s_index = 1.0
        else:
            s_index = self.db.get_next_cc_series_num_for(val, num=self.col_id)
        self.idx_widget.setValue(s_index)

    def values_changed(self):
        if False:
            return 10
        (val, s_index) = self.current_val
        return val != self.initial_val or s_index != self.initial_index

    @property
    def current_val(self):
        if False:
            print('Hello World!')
        (val, s_index) = self.gui_val
        val = self.normalize_ui_val(val)
        return (val, s_index)

    def commit(self, book_id, notify=False):
        if False:
            return 10
        (val, s_index) = self.current_val
        if val != self.initial_val or s_index != self.initial_index:
            if not val:
                val = s_index = None
            return self.db.set_custom(book_id, val, extra=s_index, num=self.col_id, notify=notify, commit=False, allow_case_change=True)
        else:
            return set()

    def apply_to_metadata(self, mi):
        if False:
            while True:
                i = 10
        (val, s_index) = self.current_val
        mi.set('#' + self.col_metadata['label'], val, extra=s_index)

    def connect_data_changed(self, slot):
        if False:
            for i in range(10):
                print('nop')
        for s in (self.name_widget.editTextChanged, self.idx_widget.valueChanged):
            s.connect(slot)
            self.signals_to_disconnect.append(s)

class Enumeration(Base):

    def setup_ui(self, parent):
        if False:
            print('Hello World!')
        self.parent = parent
        self.key = self.db.field_metadata.label_to_key(self.col_metadata['label'], prefer_custom=True)
        w = MultipleWidget(parent, only_manage_items=True, widget=QComboBox, name=self.col_metadata['name'])
        w.get_editor_button().clicked.connect(self.edit)
        w.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Fixed)
        self.set_to_undefined = w.clear
        self.name_widget = w.edit_widget
        self.widgets = [QLabel(label_string(self.col_metadata['name']), parent)]
        self.finish_ui_setup(parent, lambda parent: w)
        self.editor = self.name_widget
        vals = self.col_metadata['display']['enum_values']
        self.editor.addItem('')
        for v in vals:
            self.editor.addItem(v)

    def initialize(self, book_id):
        if False:
            while True:
                i = 10
        self.book_id = book_id
        val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
        val = self.normalize_db_val(val)
        idx = self.editor.findText(val)
        if idx < 0:
            error_dialog(self.parent, '', _('The enumeration "{0}" contains an invalid value that will be set to the default').format(self.col_metadata['name']), show=True, show_copy_button=False)
            idx = 0
        self.editor.setCurrentIndex(idx)
        self.initial_val = val

    def setter(self, val):
        if False:
            return 10
        self.editor.setCurrentIndex(self.editor.findText(val))

    def getter(self):
        if False:
            return 10
        return str(self.editor.currentText())

    def normalize_db_val(self, val):
        if False:
            i = 10
            return i + 15
        if val is None:
            val = ''
        return val

    def normalize_ui_val(self, val):
        if False:
            i = 10
            return i + 15
        if not val:
            val = None
        return val

    def set_to_undefined(self):
        if False:
            return 10
        self.editor.setCurrentIndex(0)

    def connect_data_changed(self, slot):
        if False:
            i = 10
            return i + 15
        self.editor.currentIndexChanged.connect(slot)
        self.signals_to_disconnect.append(self.editor.currentIndexChanged)

def comments_factory(db, key, parent):
    if False:
        for i in range(10):
            print('nop')
    fm = db.custom_column_num_map[key]
    ctype = fm.get('display', {}).get('interpret_as', 'html')
    if ctype == 'short-text':
        return SimpleText(db, key, parent)
    if ctype == 'long-text':
        return LongText(db, key, parent)
    if ctype == 'markdown':
        return Markdown(db, key, parent)
    return Comments(db, key, parent)
widgets = {'bool': Bool, 'rating': Rating, 'int': Int, 'float': Float, 'datetime': DateTime, 'text': Text, 'comments': comments_factory, 'series': Series, 'enumeration': Enumeration}

def field_sort_key(y, fm=None):
    if False:
        i = 10
        return i + 15
    m1 = fm[y]
    name = icu_lower(m1['name'])
    n1 = 'zzzzz' + name if column_is_comments(y, fm) else name
    return sort_key(n1)

def column_is_comments(key, fm):
    if False:
        print('Hello World!')
    return fm[key]['datatype'] == 'comments' and fm[key].get('display', {}).get('interpret_as') != 'short-text'

def get_field_list(db, use_defaults=False, pref_data_override=None):
    if False:
        i = 10
        return i + 15
    fm = db.field_metadata
    fields = fm.custom_field_keys(include_composites=False)
    if pref_data_override is not None:
        displayable = pref_data_override
    else:
        displayable = db.prefs.get('edit_metadata_custom_columns_to_display', None)
    if use_defaults or displayable is None:
        fields.sort(key=partial(field_sort_key, fm=fm))
        return [(k, True) for k in fields]
    else:
        field_set = set(fields)
        result = OrderedDict({k: v for (k, v) in displayable if k in field_set})
        for k in fields:
            if k not in result:
                result[k] = True
        return [(k, v) for (k, v) in result.items()]

def populate_metadata_page(layout, db, book_id, bulk=False, two_column=False, parent=None):
    if False:
        return 10

    def widget_factory(typ, key):
        if False:
            i = 10
            return i + 15
        if bulk:
            w = bulk_widgets[typ](db, key, parent)
        else:
            w = widgets[typ](db, key, parent)
        if book_id is not None:
            w.initialize(book_id)
        return w
    fm = db.field_metadata
    cols = [k[0] for k in get_field_list(db, use_defaults=db.prefs['edit_metadata_ignore_display_order']) if k[1]]
    comments_at_end = 0
    for k in cols[::-1]:
        if not column_is_comments(k, fm):
            break
        comments_at_end += 1
    comments_not_at_end = len([k for k in cols if column_is_comments(k, fm)]) - comments_at_end
    count = len(cols)
    layout_rows_for_comments = 9
    if two_column:
        turnover_point = int((count - comments_at_end + 1 + int(comments_not_at_end * (layout_rows_for_comments - 1))) / 2)
    else:
        turnover_point = count + 1000
    ans = []
    column = row = base_row = max_row = 0
    label_width = 0
    do_elision = gprefs['edit_metadata_elide_labels']
    elide_pos = gprefs['edit_metadata_elision_point']
    elide_pos = elide_pos if elide_pos in {'left', 'middle', 'right'} else 'right'
    sb_width = QApplication.instance().style().pixelMetric(QStyle.PixelMetric.PM_ScrollBarExtent)
    layout.setContentsMargins(0, 0, sb_width, 0)
    for key in cols:
        if not fm[key]['is_editable']:
            continue
        dt = fm[key]['datatype']
        if dt == 'composite' or (bulk and dt == 'comments'):
            continue
        is_comments = column_is_comments(key, fm)
        w = widget_factory(dt, fm[key]['colnum'])
        ans.append(w)
        if two_column and is_comments:
            comments_not_at_end -= 1
            if comments_not_at_end < 0 and comments_at_end > 0:
                column = 0
                row = max_row
                base_row = row
                turnover_point = row + int(comments_at_end * layout_rows_for_comments / 2)
                comments_at_end = 0
        l = QGridLayout()
        if is_comments:
            layout.addLayout(l, row, column, layout_rows_for_comments, 1)
            layout.setColumnStretch(column, 100)
            row += layout_rows_for_comments
        else:
            layout.addLayout(l, row, column, 1, 1)
            layout.setColumnStretch(column, 100)
            row += 1
        for c in range(0, len(w.widgets), 2):
            if not is_comments:
                wij = w.widgets[c]
                if label_width == 0:
                    font_metrics = wij.fontMetrics()
                    colon_width = font_metrics.horizontalAdvance(':')
                    if bulk:
                        label_width = font_metrics.averageCharWidth() * gprefs['edit_metadata_bulk_cc_label_length'] - colon_width
                    else:
                        label_width = font_metrics.averageCharWidth() * gprefs['edit_metadata_single_cc_label_length'] - colon_width
                wij.setMaximumWidth(label_width)
                if c == 0:
                    wij.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
                    l.setColumnMinimumWidth(0, label_width)
                wij.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                t = str(wij.text())
                if t:
                    if do_elision:
                        wij.setText(elided_text(t, font=font_metrics, width=label_width, pos=elide_pos) + ':')
                    else:
                        wij.setText(t + ':')
                        wij.setWordWrap(True)
                wij.setBuddy(w.widgets[c + 1])
                l.addWidget(wij, c, 0)
                l.addWidget(w.widgets[c + 1], c, 1)
            else:
                l.addWidget(w.widgets[0], 0, 0, 1, 2)
        max_row = max(max_row, row)
        if row >= turnover_point:
            column = 1
            turnover_point = count + 1000
            row = base_row
    items = []
    if len(ans) > 0:
        items.append(QSpacerItem(10, 10, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        layout.addItem(items[-1], layout.rowCount(), 0, 1, 1)
        layout.setRowStretch(layout.rowCount() - 1, 100)
    return (ans, items)

class BulkBase(Base):

    @property
    def gui_val(self):
        if False:
            i = 10
            return i + 15
        if not hasattr(self, '_cached_gui_val_'):
            self._cached_gui_val_ = self.getter()
        return self._cached_gui_val_

    def get_initial_value(self, book_ids):
        if False:
            i = 10
            return i + 15
        values = set()
        for book_id in book_ids:
            val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
            if isinstance(val, list):
                val = frozenset(val)
            values.add(val)
            if len(values) > 1:
                break
        ans = None
        if len(values) == 1:
            ans = next(iter(values))
        if isinstance(ans, frozenset):
            ans = list(ans)
        return ans

    def finish_ui_setup(self, parent, is_bool=False, add_edit_tags_button=(False,)):
        if False:
            while True:
                i = 10
        self.was_none = False
        l = self.widgets[1].layout()
        if not is_bool or self.bools_are_tristate:
            self.clear_button = QToolButton(parent)
            self.clear_button.setIcon(QIcon.ic('trash.png'))
            self.clear_button.setToolTip(_('Clear {0}').format(self.col_metadata['name']))
            self.clear_button.clicked.connect(self.set_to_undefined)
            l.insertWidget(1, self.clear_button)
        if is_bool:
            self.set_no_button = QToolButton(parent)
            self.set_no_button.setIcon(QIcon.ic('list_remove.png'))
            self.set_no_button.clicked.connect(lambda : self.main_widget.setCurrentIndex(1))
            self.set_no_button.setToolTip(_('Set {0} to No').format(self.col_metadata['name']))
            l.insertWidget(1, self.set_no_button)
            self.set_yes_button = QToolButton(parent)
            self.set_yes_button.setIcon(QIcon.ic('ok.png'))
            self.set_yes_button.clicked.connect(lambda : self.main_widget.setCurrentIndex(0))
            self.set_yes_button.setToolTip(_('Set {0} to Yes').format(self.col_metadata['name']))
            l.insertWidget(1, self.set_yes_button)
        if add_edit_tags_button[0]:
            self.edit_tags_button = QToolButton(parent)
            self.edit_tags_button.setToolTip(_('Open Item editor'))
            self.edit_tags_button.setIcon(QIcon.ic('chapters.png'))
            self.edit_tags_button.clicked.connect(add_edit_tags_button[1])
            l.insertWidget(1, self.edit_tags_button)
        l.insertStretch(2)

    def initialize(self, book_ids):
        if False:
            i = 10
            return i + 15
        self.initial_val = val = self.get_initial_value(book_ids)
        val = self.normalize_db_val(val)
        self.setter(val)

    def commit(self, book_ids, notify=False):
        if False:
            i = 10
            return i + 15
        if not self.a_c_checkbox.isChecked():
            return
        val = self.gui_val
        val = self.normalize_ui_val(val)
        self.db.set_custom_bulk(book_ids, val, num=self.col_id, notify=notify)

    def make_widgets(self, parent, main_widget_class):
        if False:
            return 10
        w = QWidget(parent)
        self.widgets = [QLabel(label_string(self.col_metadata['name']), w), w]
        l = QHBoxLayout()
        l.setContentsMargins(0, 0, 0, 0)
        w.setLayout(l)
        self.main_widget = main_widget_class(w)
        l.addWidget(self.main_widget)
        l.setStretchFactor(self.main_widget, 10)
        self.a_c_checkbox = QCheckBox(_('Apply changes'), w)
        l.addWidget(self.a_c_checkbox)
        self.ignore_change_signals = True
        if hasattr(self.main_widget, 'editTextChanged'):
            self.main_widget.editTextChanged.connect(self.a_c_checkbox_changed)
        if hasattr(self.main_widget, 'textChanged'):
            self.main_widget.textChanged.connect(self.a_c_checkbox_changed)
        if hasattr(self.main_widget, 'currentIndexChanged'):
            self.main_widget.currentIndexChanged.connect(self.a_c_checkbox_changed)
        if hasattr(self.main_widget, 'valueChanged'):
            self.main_widget.valueChanged.connect(self.a_c_checkbox_changed)
        if hasattr(self.main_widget, 'dateTimeChanged'):
            self.main_widget.dateTimeChanged.connect(self.a_c_checkbox_changed)

    def a_c_checkbox_changed(self):
        if False:
            i = 10
            return i + 15
        if not self.ignore_change_signals:
            self.a_c_checkbox.setChecked(True)

class BulkBool(BulkBase, Bool):

    def get_initial_value(self, book_ids):
        if False:
            for i in range(10):
                print('nop')
        value = None
        for book_id in book_ids:
            val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
            if not self.db.new_api.pref('bools_are_tristate') and val is None:
                val = False
            if value is not None and value != val:
                return None
            value = val
        return value

    def setup_ui(self, parent):
        if False:
            print('Hello World!')
        self.make_widgets(parent, QComboBox)
        items = [_('Yes'), _('No')]
        self.bools_are_tristate = self.db.new_api.pref('bools_are_tristate')
        if not self.bools_are_tristate:
            items.append('')
        else:
            items.append(_('Undefined'))
        icons = ['ok.png', 'list_remove.png', 'blank.png']
        self.main_widget.blockSignals(True)
        for (icon, text) in zip(icons, items):
            self.main_widget.addItem(QIcon.ic(icon), text)
        self.main_widget.blockSignals(False)
        self.finish_ui_setup(parent, is_bool=True)

    def set_to_undefined(self):
        if False:
            for i in range(10):
                print('nop')
        self.main_widget.setCurrentIndex(2)

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        val = self.main_widget.currentIndex()
        if not self.bools_are_tristate:
            return {2: False, 1: False, 0: True}[val]
        else:
            return {2: None, 1: False, 0: True}[val]

    def setter(self, val):
        if False:
            while True:
                i = 10
        val = {None: 2, False: 1, True: 0}[val]
        self.main_widget.setCurrentIndex(val)
        self.ignore_change_signals = False

    def commit(self, book_ids, notify=False):
        if False:
            return 10
        if not self.a_c_checkbox.isChecked():
            return
        val = self.gui_val
        val = self.normalize_ui_val(val)
        if not self.bools_are_tristate and val is None:
            val = False
        self.db.set_custom_bulk(book_ids, val, num=self.col_id, notify=notify)

    def a_c_checkbox_changed(self):
        if False:
            return 10
        if not self.ignore_change_signals:
            if not self.bools_are_tristate and self.main_widget.currentIndex() == 2:
                self.a_c_checkbox.setChecked(False)
            else:
                self.a_c_checkbox.setChecked(True)

class BulkInt(BulkBase):

    def setup_ui(self, parent):
        if False:
            i = 10
            return i + 15
        self.make_widgets(parent, QSpinBox)
        self.main_widget.setRange(-1000000, 100000000)
        self.finish_ui_setup(parent)

    def finish_ui_setup(self, parent):
        if False:
            return 10
        BulkBase.finish_ui_setup(self, parent)
        self.main_widget.setSpecialValueText(_('Undefined'))
        self.main_widget.setSingleStep(1)
        self.main_widget.valueChanged.connect(self.valueChanged)

    def setter(self, val):
        if False:
            print('Hello World!')
        if val is None:
            val = self.main_widget.minimum()
        self.main_widget.setValue(val)
        self.ignore_change_signals = False
        self.was_none = val == self.main_widget.minimum()

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        val = self.main_widget.value()
        if val == self.main_widget.minimum():
            val = None
        return val

    def valueChanged(self, to_what):
        if False:
            i = 10
            return i + 15
        if self.was_none and to_what == -999999:
            self.setter(0)
        self.was_none = to_what == self.main_widget.minimum()

    def set_to_undefined(self):
        if False:
            print('Hello World!')
        self.main_widget.setValue(-1000000)

class BulkFloat(BulkInt):

    def setup_ui(self, parent):
        if False:
            for i in range(10):
                print('nop')
        self.make_widgets(parent, QDoubleSpinBox)
        self.main_widget.setRange(-1000000.0, float(100000000))
        self.main_widget.setDecimals(int(self.col_metadata['display'].get('decimals', 2)))
        self.finish_ui_setup(parent)

    def set_to_undefined(self):
        if False:
            while True:
                i = 10
        self.main_widget.setValue(-1000000.0)

class BulkRating(BulkBase):

    def setup_ui(self, parent):
        if False:
            print('Hello World!')
        allow_half_stars = self.col_metadata['display'].get('allow_half_stars', False)
        self.make_widgets(parent, partial(RatingEditor, is_half_star=allow_half_stars))
        self.finish_ui_setup(parent)

    def set_to_undefined(self):
        if False:
            print('Hello World!')
        self.main_widget.setCurrentIndex(0)

    def setter(self, val):
        if False:
            print('Hello World!')
        val = max(0, min(int(val or 0), 10))
        self.main_widget.rating_value = val
        self.ignore_change_signals = False

    def getter(self):
        if False:
            return 10
        return self.main_widget.rating_value or None

class BulkDateTime(BulkBase):

    def setup_ui(self, parent):
        if False:
            return 10
        cm = self.col_metadata
        self.make_widgets(parent, DateTimeEdit)
        l = self.widgets[1].layout()
        self.today_button = QToolButton(parent)
        self.today_button.setText(_('Today'))
        l.insertWidget(1, self.today_button)
        self.clear_button = QToolButton(parent)
        self.clear_button.setIcon(QIcon.ic('trash.png'))
        self.clear_button.setToolTip(_('Clear {0}').format(self.col_metadata['name']))
        l.insertWidget(2, self.clear_button)
        l.insertStretch(3)
        w = self.main_widget
        format_ = cm['display'].get('date_format', '')
        if not format_:
            format_ = 'dd MMM yyyy'
        elif format_ == 'iso':
            format_ = internal_iso_format_string()
        w.setDisplayFormat(format_)
        w.setCalendarPopup(True)
        w.setMinimumDateTime(UNDEFINED_QDATETIME)
        w.setSpecialValueText(_('Undefined'))
        self.today_button.clicked.connect(w.set_to_today)
        self.clear_button.clicked.connect(w.set_to_clear)

    def setter(self, val):
        if False:
            i = 10
            return i + 15
        if val is None:
            val = self.main_widget.minimumDateTime()
        else:
            val = qt_from_dt(val)
        self.main_widget.setDateTime(val)
        self.ignore_change_signals = False

    def getter(self):
        if False:
            print('Hello World!')
        val = self.main_widget.dateTime()
        if is_date_undefined(val):
            val = None
        else:
            val = qt_to_dt(val)
        return val

    def normalize_db_val(self, val):
        if False:
            return 10
        return as_local_time(val) if val is not None else None

    def normalize_ui_val(self, val):
        if False:
            return 10
        return as_utc(val) if val is not None else None

class BulkSeries(BulkBase):

    def setup_ui(self, parent):
        if False:
            i = 10
            return i + 15
        self.make_widgets(parent, partial(EditWithComplete, sort_func=title_sort))
        values = self.all_values = list(self.db.all_custom(num=self.col_id))
        values.sort(key=sort_key)
        self.main_widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.main_widget.setMinimumContentsLength(25)
        self.widgets.append(QLabel('', parent))
        w = QWidget(parent)
        layout = QHBoxLayout(w)
        layout.setContentsMargins(0, 0, 0, 0)
        self.remove_series = QCheckBox(parent)
        self.remove_series.setText(_('Clear series'))
        layout.addWidget(self.remove_series)
        self.idx_widget = QCheckBox(parent)
        self.idx_widget.setText(_('Automatically number books'))
        self.idx_widget.setToolTip('<p>' + _('If not checked, the series number for the books will be set to 1. If checked, selected books will be automatically numbered, in the order you selected them. So if you selected Book A and then Book B, Book A will have series number 1 and Book B series number 2.') + '</p>')
        layout.addWidget(self.idx_widget)
        self.force_number = QCheckBox(parent)
        self.force_number.setText(_('Force numbers to start with '))
        self.force_number.setToolTip('<p>' + _('Series will normally be renumbered from the highest number in the database for that series. Checking this box will tell calibre to start numbering from the value in the box') + '</p>')
        layout.addWidget(self.force_number)
        self.series_start_number = QDoubleSpinBox(parent)
        self.series_start_number.setMinimum(0.0)
        self.series_start_number.setMaximum(9999999.0)
        self.series_start_number.setProperty('value', 1.0)
        layout.addWidget(self.series_start_number)
        self.series_increment = QDoubleSpinBox(parent)
        self.series_increment.setMinimum(0.0)
        self.series_increment.setMaximum(99999.0)
        self.series_increment.setProperty('value', 1.0)
        self.series_increment.setToolTip('<p>' + _('The amount by which to increment the series number for successive books. Only applicable when using force series numbers.') + '</p>')
        self.series_increment.setPrefix('+')
        layout.addWidget(self.series_increment)
        layout.addItem(QSpacerItem(20, 10, QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum))
        self.widgets.append(w)
        self.idx_widget.stateChanged.connect(self.a_c_checkbox_changed)
        self.force_number.stateChanged.connect(self.a_c_checkbox_changed)
        self.series_start_number.valueChanged.connect(self.a_c_checkbox_changed)
        self.series_increment.valueChanged.connect(self.a_c_checkbox_changed)
        self.remove_series.stateChanged.connect(self.a_c_checkbox_changed)
        self.main_widget
        self.ignore_change_signals = False

    def a_c_checkbox_changed(self):
        if False:
            for i in range(10):
                print('nop')

        def disable_numbering_checkboxes(idx_widget_enable):
            if False:
                while True:
                    i = 10
            if idx_widget_enable:
                self.idx_widget.setEnabled(True)
            else:
                self.idx_widget.setChecked(False)
                self.idx_widget.setEnabled(False)
            self.force_number.setChecked(False)
            self.force_number.setEnabled(False)
            self.series_start_number.setEnabled(False)
            self.series_increment.setEnabled(False)
        if self.ignore_change_signals:
            return
        self.ignore_change_signals = True
        apply_changes = False
        if self.remove_series.isChecked():
            self.main_widget.setText('')
            self.main_widget.setEnabled(False)
            disable_numbering_checkboxes(idx_widget_enable=False)
            apply_changes = True
        elif self.main_widget.text():
            self.remove_series.setEnabled(False)
            self.idx_widget.setEnabled(True)
            apply_changes = True
        else:
            self.main_widget.setEnabled(True)
            self.remove_series.setEnabled(True)
            disable_numbering_checkboxes(idx_widget_enable=False)
            apply_changes = False
        self.force_number.setEnabled(self.idx_widget.isChecked())
        self.series_start_number.setEnabled(self.force_number.isChecked())
        self.series_increment.setEnabled(self.force_number.isChecked())
        self.ignore_change_signals = False
        self.a_c_checkbox.setChecked(apply_changes)

    def initialize(self, book_id):
        if False:
            return 10
        self.idx_widget.setChecked(False)
        self.main_widget.set_separator(None)
        self.main_widget.update_items_cache(self.all_values)
        self.main_widget.setEditText('')
        self.a_c_checkbox.setChecked(False)

    def getter(self):
        if False:
            i = 10
            return i + 15
        n = str(self.main_widget.currentText()).strip()
        autonumber = self.idx_widget.isChecked()
        force = self.force_number.isChecked()
        start = self.series_start_number.value()
        remove = self.remove_series.isChecked()
        increment = self.series_increment.value()
        return (n, autonumber, force, start, remove, increment)

    def commit(self, book_ids, notify=False):
        if False:
            print('Hello World!')
        if not self.a_c_checkbox.isChecked():
            return
        (val, update_indices, force_start, at_value, clear, increment) = self.gui_val
        val = None if clear else self.normalize_ui_val(val)
        if clear or val != '':
            extras = []
            for book_id in book_ids:
                if clear:
                    extras.append(None)
                    continue
                if update_indices:
                    if force_start:
                        s_index = at_value
                        at_value += increment
                    elif tweaks['series_index_auto_increment'] != 'const':
                        s_index = self.db.get_next_cc_series_num_for(val, num=self.col_id)
                    else:
                        s_index = 1.0
                else:
                    s_index = self.db.get_custom_extra(book_id, num=self.col_id, index_is_id=True)
                extras.append(s_index)
            self.db.set_custom_bulk(book_ids, val, extras=extras, num=self.col_id, notify=notify)

class BulkEnumeration(BulkBase, Enumeration):

    def get_initial_value(self, book_ids):
        if False:
            print('Hello World!')
        value = None
        first = True
        dialog_shown = False
        for book_id in book_ids:
            val = self.db.get_custom(book_id, num=self.col_id, index_is_id=True)
            if val and val not in self.col_metadata['display']['enum_values']:
                if not dialog_shown:
                    error_dialog(self.parent, '', _('The enumeration "{0}" contains invalid values that will not appear in the list').format(self.col_metadata['name']), show=True, show_copy_button=False)
                    dialog_shown = True
            if first:
                value = val
                first = False
            elif value != val:
                value = None
        if not value:
            self.ignore_change_signals = False
        return value

    def setup_ui(self, parent):
        if False:
            i = 10
            return i + 15
        self.parent = parent
        self.make_widgets(parent, QComboBox)
        self.finish_ui_setup(parent)
        vals = self.col_metadata['display']['enum_values']
        self.main_widget.blockSignals(True)
        self.main_widget.addItem('')
        self.main_widget.addItems(vals)
        self.main_widget.blockSignals(False)

    def set_to_undefined(self):
        if False:
            while True:
                i = 10
        self.main_widget.setCurrentIndex(0)

    def getter(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self.main_widget.currentText())

    def setter(self, val):
        if False:
            while True:
                i = 10
        if val is None:
            self.main_widget.setCurrentIndex(0)
        else:
            self.main_widget.setCurrentIndex(self.main_widget.findText(val))
        self.ignore_change_signals = False

class RemoveTags(QWidget):

    def __init__(self, parent, values):
        if False:
            while True:
                i = 10
        QWidget.__init__(self, parent)
        layout = QHBoxLayout()
        layout.setSpacing(5)
        layout.setContentsMargins(0, 0, 0, 0)
        self.tags_box = EditWithComplete(parent)
        self.tags_box.update_items_cache(values)
        layout.addWidget(self.tags_box, stretch=3)
        self.remove_tags_button = QToolButton(parent)
        self.remove_tags_button.setToolTip(_('Open Item editor'))
        self.remove_tags_button.setIcon(QIcon.ic('chapters.png'))
        layout.addWidget(self.remove_tags_button)
        self.checkbox = QCheckBox(_('Remove all tags'), parent)
        layout.addWidget(self.checkbox)
        layout.addStretch(1)
        self.setLayout(layout)
        self.checkbox.stateChanged[int].connect(self.box_touched)

    def box_touched(self, state):
        if False:
            for i in range(10):
                print('nop')
        if state:
            self.tags_box.setText('')
            self.tags_box.setEnabled(False)
        else:
            self.tags_box.setEnabled(True)

class BulkText(BulkBase):

    def setup_ui(self, parent):
        if False:
            while True:
                i = 10
        values = self.all_values = list(self.db.all_custom(num=self.col_id))
        values.sort(key=sort_key)
        is_tags = False
        if self.col_metadata['is_multiple']:
            is_tags = not self.col_metadata['display'].get('is_names', False)
            self.make_widgets(parent, EditWithComplete)
            self.main_widget.setSizePolicy(QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Preferred)
            self.adding_widget = self.main_widget
            if is_tags:
                w = RemoveTags(parent, values)
                w.remove_tags_button.clicked.connect(self.edit_remove)
                l = QLabel(label_string(self.col_metadata['name']) + ': ' + _('tags to remove'), parent)
                tt = get_tooltip(self.col_metadata) + ': ' + _('tags to remove')
                l.setToolTip(tt)
                self.widgets.append(l)
                w.setToolTip(tt)
                self.widgets.append(w)
                self.removing_widget = w
                self.main_widget.set_separator(',')
                w.tags_box.textChanged.connect(self.a_c_checkbox_changed)
                w.checkbox.stateChanged.connect(self.a_c_checkbox_changed)
            else:
                self.main_widget.set_separator('&')
                self.main_widget.set_space_before_sep(True)
                self.main_widget.set_add_separator(tweaks['authors_completer_append_separator'])
        else:
            self.make_widgets(parent, EditWithComplete)
            self.main_widget.set_separator(None)
            self.main_widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            self.main_widget.setMinimumContentsLength(25)
        self.ignore_change_signals = False
        self.parent = parent
        self.finish_ui_setup(parent, add_edit_tags_button=(is_tags, self.edit_add))

    def set_to_undefined(self):
        if False:
            print('Hello World!')
        self.main_widget.clearEditText()

    def initialize(self, book_ids):
        if False:
            return 10
        self.main_widget.update_items_cache(self.all_values)
        if not self.col_metadata['is_multiple']:
            val = self.get_initial_value(book_ids)
            self.initial_val = val = self.normalize_db_val(val)
            self.ignore_change_signals = True
            self.main_widget.blockSignals(True)
            self.main_widget.setText(val)
            self.main_widget.blockSignals(False)
            self.ignore_change_signals = False

    def commit(self, book_ids, notify=False):
        if False:
            while True:
                i = 10
        if not self.a_c_checkbox.isChecked():
            return
        if self.col_metadata['is_multiple']:
            ism = self.col_metadata['multiple_seps']
            if self.col_metadata['display'].get('is_names', False):
                val = self.gui_val
                add = [v.strip() for v in val.split(ism['ui_to_list']) if v.strip()]
                self.db.set_custom_bulk(book_ids, add, num=self.col_id)
            else:
                (remove_all, adding, rtext) = self.gui_val
                remove = set()
                if remove_all:
                    remove = set(self.db.all_custom(num=self.col_id))
                else:
                    txt = rtext
                    if txt:
                        remove = {v.strip() for v in txt.split(ism['ui_to_list'])}
                txt = adding
                if txt:
                    add = {v.strip() for v in txt.split(ism['ui_to_list'])}
                else:
                    add = set()
                self.db.set_custom_bulk_multiple(book_ids, add=add, remove=remove, num=self.col_id)
        else:
            val = self.gui_val
            val = self.normalize_ui_val(val)
            self.db.set_custom_bulk(book_ids, val, num=self.col_id, notify=notify)

    def getter(self):
        if False:
            return 10
        if self.col_metadata['is_multiple']:
            if not self.col_metadata['display'].get('is_names', False):
                return (self.removing_widget.checkbox.isChecked(), str(self.adding_widget.text()), str(self.removing_widget.tags_box.text()))
            return str(self.adding_widget.text())
        val = str(self.main_widget.currentText()).strip()
        if not val:
            val = None
        return val

    def edit_remove(self):
        if False:
            print('Hello World!')
        self.edit(widget=self.removing_widget.tags_box)

    def edit_add(self):
        if False:
            i = 10
            return i + 15
        self.edit(widget=self.main_widget)

    def edit(self, widget):
        if False:
            for i in range(10):
                print('nop')
        if widget.text():
            d = _save_dialog(self.parent, _('Values changed'), _('You have entered values. In order to use this editor you must first discard them. Discard the values?'))
            if d == QMessageBox.StandardButton.Cancel or d == QMessageBox.StandardButton.No:
                return
            widget.setText('')
        d = TagEditor(self.parent, self.db, key='#' + self.col_metadata['label'])
        if d.exec() == QDialog.DialogCode.Accepted:
            val = d.tags
            if not val:
                val = []
            widget.setText(self.col_metadata['multiple_seps']['list_to_ui'].join(val))
bulk_widgets = {'bool': BulkBool, 'rating': BulkRating, 'int': BulkInt, 'float': BulkFloat, 'datetime': BulkDateTime, 'text': BulkText, 'series': BulkSeries, 'enumeration': BulkEnumeration}