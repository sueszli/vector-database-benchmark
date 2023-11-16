__license__ = 'GPL v3'
__copyright__ = '2009, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
from qt.core import QWidget, QListWidgetItem
from calibre.gui2 import gprefs
from calibre.gui2.catalog.catalog_bibtex_ui import Ui_Form

class PluginWidget(QWidget, Ui_Form):
    TITLE = _('BibTeX options')
    HELP = _('Options specific to') + ' BibTeX ' + _('output')
    OPTION_FIELDS = [('bib_cit', '{authors}{id}'), ('bib_entry', 0), ('bibfile_enc', 0), ('bibfile_enctag', 0), ('impcit', True), ('addfiles', False)]
    sync_enabled = False
    formats = {'bib'}

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        QWidget.__init__(self, parent)
        self.setupUi(self)

    def initialize(self, name, db):
        if False:
            i = 10
            return i + 15
        from calibre.library.catalogs import FIELDS
        self.all_fields = [x for x in FIELDS if x != 'all']
        for x in sorted(db.custom_field_keys()):
            self.all_fields.append(x)
            if db.field_metadata[x]['datatype'] == 'series':
                self.all_fields.append(x + '_index')
        for x in self.all_fields:
            QListWidgetItem(x, self.db_fields)
        self.name = name
        fields = gprefs.get(name + '_db_fields', self.all_fields)
        for x in range(self.db_fields.count()):
            item = self.db_fields.item(x)
            item.setSelected(str(item.text()) in fields)
        self.bibfile_enc.clear()
        self.bibfile_enc.addItems(['utf-8', 'cp1252', 'ascii/LaTeX'])
        self.bibfile_enctag.clear()
        self.bibfile_enctag.addItems(['strict', 'replace', 'ignore', 'backslashreplace'])
        self.bib_entry.clear()
        self.bib_entry.addItems(['mixed', 'misc', 'book'])
        for opt in self.OPTION_FIELDS:
            opt_value = gprefs.get(self.name + '_' + opt[0], opt[1])
            if opt[0] in ['bibfile_enc', 'bibfile_enctag', 'bib_entry']:
                getattr(self, opt[0]).setCurrentIndex(opt_value)
            elif opt[0] in ['impcit', 'addfiles']:
                getattr(self, opt[0]).setChecked(opt_value)
            else:
                getattr(self, opt[0]).setText(opt_value)

    def options(self):
        if False:
            print('Hello World!')
        fields = []
        for x in range(self.db_fields.count()):
            item = self.db_fields.item(x)
            if item.isSelected():
                fields.append(str(item.text()))
        gprefs.set(self.name + '_db_fields', fields)
        if len(self.db_fields.selectedItems()):
            opts_dict = {'fields': [str(i.text()) for i in self.db_fields.selectedItems()]}
        else:
            opts_dict = {'fields': ['all']}
        for opt in self.OPTION_FIELDS:
            if opt[0] in ['bibfile_enc', 'bibfile_enctag', 'bib_entry']:
                opt_value = getattr(self, opt[0]).currentIndex()
            elif opt[0] in ['impcit', 'addfiles']:
                opt_value = getattr(self, opt[0]).isChecked()
            else:
                opt_value = str(getattr(self, opt[0]).text())
            gprefs.set(self.name + '_' + opt[0], opt_value)
            opts_dict[opt[0]] = opt_value
        return opts_dict