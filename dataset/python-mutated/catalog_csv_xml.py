__license__ = 'GPL v3'
__copyright__ = '2009, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from qt.core import QAbstractItemView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QPushButton, Qt, QVBoxLayout, QWidget
from calibre.constants import ismacos
from calibre.gui2 import gprefs
from calibre.gui2.ui import get_gui

def get_saved_field_data(name, all_fields):
    if False:
        while True:
            i = 10
    db = get_gui().current_db
    val = db.new_api.pref('catalog-field-data-for-' + name)
    if val is None:
        sort_order = gprefs.get(name + '_db_fields_sort_order', {})
        fields = frozenset(gprefs.get(name + '_db_fields', all_fields))
    else:
        sort_order = val['sort_order']
        fields = frozenset(val['fields'])
    return (sort_order, fields)

def set_saved_field_data(name, fields, sort_order):
    if False:
        while True:
            i = 10
    db = get_gui().current_db
    db.new_api.set_pref('catalog-field-data-for-' + name, {'fields': fields, 'sort_order': sort_order})
    gprefs.set(name + '_db_fields', fields)
    gprefs.set(name + '_db_fields_sort_order', sort_order)

class PluginWidget(QWidget):
    TITLE = _('CSV/XML options')
    HELP = _('Options specific to') + ' CSV/XML ' + _('output')
    sync_enabled = False
    formats = {'csv', 'xml'}
    handles_scrolling = True

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, parent)
        self.l = l = QVBoxLayout(self)
        self.la = la = QLabel(_('Fields to include in output:'))
        la.setWordWrap(True)
        l.addWidget(la)
        self.db_fields = QListWidget(self)
        l.addWidget(self.db_fields)
        self.la2 = la = QLabel(_('Drag and drop to re-arrange fields'))
        self.db_fields.setDragEnabled(True)
        self.db_fields.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.db_fields.setDefaultDropAction(Qt.DropAction.CopyAction if ismacos else Qt.DropAction.MoveAction)
        self.db_fields.setAlternatingRowColors(True)
        self.db_fields.setObjectName('db_fields')
        h = QHBoxLayout()
        l.addLayout(h)
        (h.addWidget(la), h.addStretch(10))
        self.select_all_button = b = QPushButton(_('Select &all'))
        b.clicked.connect(self.select_all)
        h.addWidget(b)
        self.select_all_button = b = QPushButton(_('Select &none'))
        b.clicked.connect(self.select_none)
        h.addWidget(b)
        self.select_visible_button = b = QPushButton(_('Select &visible'))
        b.clicked.connect(self.select_visible)
        b.setToolTip(_('Select the fields currently shown in the book list'))
        h.addWidget(b)

    def select_all(self):
        if False:
            return 10
        for row in range(self.db_fields.count()):
            item = self.db_fields.item(row)
            item.setCheckState(Qt.CheckState.Checked)

    def select_none(self):
        if False:
            for i in range(10):
                print('nop')
        for row in range(self.db_fields.count()):
            item = self.db_fields.item(row)
            item.setCheckState(Qt.CheckState.Unchecked)

    def select_visible(self):
        if False:
            return 10
        state = get_gui().library_view.get_state()
        hidden = frozenset(state['hidden_columns'])
        for row in range(self.db_fields.count()):
            item = self.db_fields.item(row)
            field = item.data(Qt.ItemDataRole.UserRole)
            item.setCheckState(Qt.CheckState.Unchecked if field in hidden else Qt.CheckState.Checked)

    def initialize(self, catalog_name, db):
        if False:
            for i in range(10):
                print('nop')
        self.name = catalog_name
        from calibre.library.catalogs import FIELDS
        db = get_gui().current_db
        self.all_fields = {x for x in FIELDS if x != 'all'} | set(db.custom_field_keys())
        (sort_order, fields) = get_saved_field_data(self.name, self.all_fields)
        fm = db.field_metadata

        def name(x):
            if False:
                while True:
                    i = 10
            if x == 'isbn':
                return 'ISBN'
            if x == 'library_name':
                return _('Library name')
            if x.endswith('_index'):
                return name(x[:-len('_index')]) + ' ' + _('Number')
            return fm[x].get('name') or x

        def key(x):
            if False:
                while True:
                    i = 10
            return (sort_order.get(x, 10000), name(x))
        self.db_fields.clear()
        for x in sorted(self.all_fields, key=key):
            QListWidgetItem(name(x) + ' (%s)' % x, self.db_fields).setData(Qt.ItemDataRole.UserRole, x)
            if x.startswith('#') and fm[x]['datatype'] == 'series':
                x += '_index'
                QListWidgetItem(name(x) + ' (%s)' % x, self.db_fields).setData(Qt.ItemDataRole.UserRole, x)
        for x in range(self.db_fields.count()):
            item = self.db_fields.item(x)
            item.setCheckState(Qt.CheckState.Checked if str(item.data(Qt.ItemDataRole.UserRole)) in fields else Qt.CheckState.Unchecked)

    def options(self):
        if False:
            print('Hello World!')
        (fields, all_fields) = ([], [])
        for x in range(self.db_fields.count()):
            item = self.db_fields.item(x)
            all_fields.append(str(item.data(Qt.ItemDataRole.UserRole)))
            if item.checkState() == Qt.CheckState.Checked:
                fields.append(str(item.data(Qt.ItemDataRole.UserRole)))
        set_saved_field_data(self.name, fields, {x: i for (i, x) in enumerate(all_fields)})
        if len(fields):
            return {'fields': fields}
        else:
            return {'fields': ['all']}