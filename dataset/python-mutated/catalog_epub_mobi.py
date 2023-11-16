__license__ = 'GPL v3'
__copyright__ = '2009, Kovid Goyal <kovid@kovidgoyal.net>'
__docformat__ = 'restructuredtext en'
import re, sys
from functools import partial
from calibre.ebooks.conversion.config import load_defaults
from calibre.gui2 import gprefs, open_url, question_dialog, error_dialog
from calibre.utils.config import JSONConfig
from calibre.utils.icu import sort_key
from calibre.utils.localization import localize_user_manual_link
from .catalog_epub_mobi_ui import Ui_Form
from qt.core import Qt, QAbstractItemView, QCheckBox, QComboBox, QDoubleSpinBox, QIcon, QInputDialog, QLineEdit, QRadioButton, QSize, QSizePolicy, QTableWidget, QTableWidgetItem, QTextEdit, QToolButton, QUrl, QVBoxLayout, QWidget, sip

class PluginWidget(QWidget, Ui_Form):
    TITLE = _('E-book options')
    HELP = _('Options specific to') + ' AZW3/EPUB/MOBI ' + _('output')
    DEBUG = False
    handles_scrolling = True
    sync_enabled = True
    formats = {'azw3', 'epub', 'mobi'}

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self, parent)
        self.setupUi(self)
        self._initControlArrays()
        self.blocking_all_signals = None
        self.parent_ref = lambda : None

    def _initControlArrays(self):
        if False:
            return 10
        CheckBoxControls = []
        ComboBoxControls = []
        DoubleSpinBoxControls = []
        LineEditControls = []
        RadioButtonControls = []
        TableWidgetControls = []
        TextEditControls = []
        for item in self.__dict__:
            if type(self.__dict__[item]) is QCheckBox:
                CheckBoxControls.append(self.__dict__[item].objectName())
            elif type(self.__dict__[item]) is QComboBox:
                ComboBoxControls.append(self.__dict__[item].objectName())
            elif type(self.__dict__[item]) is QDoubleSpinBox:
                DoubleSpinBoxControls.append(self.__dict__[item].objectName())
            elif type(self.__dict__[item]) is QLineEdit:
                LineEditControls.append(self.__dict__[item].objectName())
            elif type(self.__dict__[item]) is QRadioButton:
                RadioButtonControls.append(self.__dict__[item].objectName())
            elif type(self.__dict__[item]) is QTableWidget:
                TableWidgetControls.append(self.__dict__[item].objectName())
            elif type(self.__dict__[item]) is QTextEdit:
                TextEditControls.append(self.__dict__[item].objectName())
        option_fields = list(zip(CheckBoxControls, [True for i in CheckBoxControls], ['check_box' for i in CheckBoxControls]))
        option_fields += list(zip(ComboBoxControls, [None for i in ComboBoxControls], ['combo_box' for i in ComboBoxControls]))
        option_fields += list(zip(RadioButtonControls, [None for i in RadioButtonControls], ['radio_button' for i in RadioButtonControls]))
        option_fields += list(zip(['exclude_genre'], ['\\[.+\\]|^\\+$'], ['line_edit']))
        option_fields += list(zip(['thumb_width'], [1.0], ['spin_box']))
        option_fields += list(zip(['exclusion_rules_tw'], [{'ordinal': 0, 'enabled': True, 'name': _('Catalogs'), 'field': _('Tags'), 'pattern': 'Catalog'}], ['table_widget']))
        option_fields += list(zip(['prefix_rules_tw', 'prefix_rules_tw'], [{'ordinal': 0, 'enabled': True, 'name': _('Read book'), 'field': _('Tags'), 'pattern': '+', 'prefix': '✓'}, {'ordinal': 1, 'enabled': True, 'name': _('Wishlist item'), 'field': _('Tags'), 'pattern': 'Wishlist', 'prefix': '×'}], ['table_widget', 'table_widget']))
        self.OPTION_FIELDS = option_fields

    def block_all_signals(self, bool):
        if False:
            i = 10
            return i + 15
        if self.DEBUG:
            print('block_all_signals: %s' % bool)
        self.blocking_all_signals = bool
        for opt in self.OPTION_FIELDS:
            (c_name, c_def, c_type) = opt
            if c_name in ['exclusion_rules_tw', 'prefix_rules_tw']:
                continue
            getattr(self, c_name).blockSignals(bool)

    def construct_tw_opts_object(self, c_name, opt_value, opts_dict):
        if False:
            return 10
        "\n        Build an opts object from the UI settings to pass to the catalog builder\n        Handles two types of rules sets, with and without ['prefix'] field\n        Store processed opts object to opt_dict\n        "
        rule_set = []
        for stored_rule in opt_value:
            rule = stored_rule.copy()
            if not rule['enabled']:
                continue
            if not rule['field'] or not rule['pattern']:
                continue
            if 'prefix' in rule and rule['prefix'] is None:
                continue
            if rule['field'] != _('Tags'):
                rule['field'] = self.eligible_custom_fields[rule['field']]['field']
                if rule['pattern'] in [_('any value'), _('any date')]:
                    rule['pattern'] = '.*'
                elif rule['pattern'] == _('unspecified'):
                    rule['pattern'] = 'None'
            if 'prefix' in rule:
                pr = (rule['name'], rule['field'], rule['pattern'], rule['prefix'])
            else:
                pr = (rule['name'], rule['field'], rule['pattern'])
            rule_set.append(pr)
        opt_value = tuple(rule_set)
        opts_dict[c_name[:-3]] = opt_value

    def exclude_genre_changed(self):
        if False:
            print('Hello World!')
        ' Dynamically compute excluded genres.\n\n        Run exclude_genre regex against selected genre_source_field to show excluded tags.\n\n        Inputs:\n            current regex\n            genre_source_field\n\n        Output:\n         self.exclude_genre_results (QLabel): updated to show tags to be excluded as genres\n        '

        def _truncated_results(excluded_tags, limit=180):
            if False:
                print('Hello World!')
            '\n            Limit number of genres displayed to avoid dialog explosion\n            '
            start = []
            end = []
            lower = 0
            upper = len(excluded_tags) - 1
            excluded_tags.sort()
            while True:
                if lower > upper:
                    break
                elif lower == upper:
                    start.append(excluded_tags[lower])
                    break
                start.append(excluded_tags[lower])
                end.insert(0, excluded_tags[upper])
                if len(', '.join(start)) + len(', '.join(end)) > limit:
                    break
                lower += 1
                upper -= 1
            if excluded_tags == start + end:
                return ', '.join(excluded_tags)
            else:
                return '{}  ...  {}'.format(', '.join(start), ', '.join(end))
        results = _('No genres will be excluded')
        regex = str(getattr(self, 'exclude_genre').text()).strip()
        if not regex:
            self.exclude_genre_results.clear()
            self.exclude_genre_results.setText(results)
            return
        if self.genre_source_field_name == _('Tags'):
            all_genre_tags = self.db.all_tags()
        else:
            all_genre_tags = list(self.db.all_custom(self.db.field_metadata.key_to_label(self.genre_source_field_name)))
        try:
            pattern = re.compile(regex)
        except:
            results = _('regex error: %s') % sys.exc_info()[1]
        else:
            excluded_tags = []
            for tag in all_genre_tags:
                hit = pattern.search(tag)
                if hit:
                    excluded_tags.append(hit.string)
            if excluded_tags:
                if set(excluded_tags) == set(all_genre_tags):
                    results = _('All genres will be excluded')
                else:
                    results = _truncated_results(excluded_tags)
        finally:
            if False and self.DEBUG:
                print('exclude_genre_changed(): %s' % results)
            self.exclude_genre_results.clear()
            self.exclude_genre_results.setText(results)

    def exclude_genre_reset(self):
        if False:
            while True:
                i = 10
        for default in self.OPTION_FIELDS:
            if default[0] == 'exclude_genre':
                self.exclude_genre.setText(default[1])
                break

    def fetch_eligible_custom_fields(self):
        if False:
            return 10
        self.all_custom_fields = self.db.custom_field_keys()
        custom_fields = {}
        custom_fields[_('Tags')] = {'field': 'tag', 'datatype': 'text'}
        for custom_field in self.all_custom_fields:
            field_md = self.db.metadata_for_field(custom_field)
            if field_md['datatype'] in ['bool', 'composite', 'datetime', 'enumeration', 'text']:
                custom_fields[field_md['name']] = {'field': custom_field, 'datatype': field_md['datatype']}
        self.eligible_custom_fields = custom_fields

    def generate_descriptions_changed(self, enabled):
        if False:
            while True:
                i = 10
        '\n        Toggle Description-related controls\n        '
        self.header_note_source_field.setEnabled(enabled)
        self.include_hr.setEnabled(enabled)
        self.merge_after.setEnabled(enabled)
        self.merge_before.setEnabled(enabled)
        self.merge_source_field.setEnabled(enabled)
        self.thumb_width.setEnabled(enabled)

    def generate_genres_changed(self, enabled):
        if False:
            print('Hello World!')
        '\n        Toggle Genres-related controls\n        '
        self.genre_source_field.setEnabled(enabled)

    def genre_source_field_changed(self, new_index):
        if False:
            while True:
                i = 10
        '\n        Process changes in the genre_source_field combo box\n        Update Excluded genres preview\n        '
        new_source = self.genre_source_field.currentText()
        self.genre_source_field_name = new_source
        if new_source != _('Tags'):
            genre_source_spec = self.genre_source_fields[str(new_source)]
            self.genre_source_field_name = genre_source_spec['field']
        self.exclude_genre_changed()

    def get_format_and_title(self):
        if False:
            return 10
        current_format = None
        current_title = None
        parent = self.parent_ref()
        if parent is not None:
            current_title = parent.title.text().strip()
            current_format = parent.format.currentText().strip()
        return (current_format, current_title)

    def header_note_source_field_changed(self, new_index):
        if False:
            i = 10
            return i + 15
        '\n        Process changes in the header_note_source_field combo box\n        '
        new_source = self.header_note_source_field.currentText()
        self.header_note_source_field_name = new_source
        if new_source:
            header_note_source_spec = self.header_note_source_fields[str(new_source)]
            self.header_note_source_field_name = header_note_source_spec['field']

    def initialize(self, name, db):
        if False:
            return 10
        "\n        CheckBoxControls (c_type: check_box):\n            ['cross_reference_authors',\n             'generate_titles','generate_series','generate_genres',\n             'generate_recently_added','generate_descriptions',\n             'include_hr']\n        ComboBoxControls (c_type: combo_box):\n            ['exclude_source_field','genre_source_field',\n             'header_note_source_field','merge_source_field']\n        LineEditControls (c_type: line_edit):\n            ['exclude_genre']\n        RadioButtonControls (c_type: radio_button):\n            ['merge_before','merge_after','generate_new_cover', 'use_existing_cover']\n        SpinBoxControls (c_type: spin_box):\n            ['thumb_width']\n        TableWidgetControls (c_type: table_widget):\n            ['exclusion_rules_tw','prefix_rules_tw']\n        TextEditControls (c_type: text_edit):\n            ['exclude_genre_results']\n\n        "
        self.name = name
        self.db = db
        self.all_genre_tags = []
        self.fetch_eligible_custom_fields()
        self.populate_combo_boxes()
        self.blocking_all_signals = True
        exclusion_rules = []
        prefix_rules = []
        for opt in self.OPTION_FIELDS:
            (c_name, c_def, c_type) = opt
            opt_value = gprefs.get(self.name + '_' + c_name, c_def)
            if c_type in ['check_box']:
                getattr(self, c_name).setChecked(eval(str(opt_value)))
                getattr(self, c_name).clicked.connect(partial(self.settings_changed, c_name))
            elif c_type in ['combo_box']:
                if opt_value is None:
                    index = 0
                    if c_name == 'genre_source_field':
                        index = self.genre_source_field.findText(_('Tags'))
                else:
                    index = getattr(self, c_name).findText(opt_value)
                    if index == -1:
                        if c_name == 'read_source_field':
                            index = self.read_source_field.findText(_('Tags'))
                        elif c_name == 'genre_source_field':
                            index = self.genre_source_field.findText(_('Tags'))
                getattr(self, c_name).setCurrentIndex(index)
                if c_name != 'preset_field':
                    getattr(self, c_name).currentIndexChanged.connect(partial(self.settings_changed, c_name))
            elif c_type in ['line_edit']:
                getattr(self, c_name).setText(opt_value if opt_value else '')
                getattr(self, c_name).editingFinished.connect(partial(self.settings_changed, c_name))
            elif c_type in ['radio_button'] and opt_value is not None:
                getattr(self, c_name).setChecked(opt_value)
                getattr(self, c_name).clicked.connect(partial(self.settings_changed, c_name))
            elif c_type in ['spin_box']:
                getattr(self, c_name).setValue(float(opt_value))
                getattr(self, c_name).valueChanged.connect(partial(self.settings_changed, c_name))
            if c_type == 'table_widget':
                if c_name == 'exclusion_rules_tw':
                    if opt_value not in exclusion_rules:
                        exclusion_rules.append(opt_value)
                if c_name == 'prefix_rules_tw':
                    if opt_value not in prefix_rules:
                        prefix_rules.append(opt_value)
        self.reset_exclude_genres_tb.setIcon(QIcon.ic('trash.png'))
        self.reset_exclude_genres_tb.clicked.connect(self.exclude_genre_reset)
        self.exclude_genre.textChanged.connect(self.exclude_genre_changed)
        self.generate_descriptions.clicked.connect(self.generate_descriptions_changed)
        self.generate_descriptions_changed(self.generate_descriptions.isChecked())
        self.merge_source_field_name = ''
        cs = str(self.merge_source_field.currentText())
        if cs:
            merge_source_spec = self.merge_source_fields[cs]
            self.merge_source_field_name = merge_source_spec['field']
        self.header_note_source_field_name = ''
        cs = str(self.header_note_source_field.currentText())
        if cs:
            header_note_source_spec = self.header_note_source_fields[cs]
            self.header_note_source_field_name = header_note_source_spec['field']
        self.genre_source_field_name = _('Tags')
        cs = str(self.genre_source_field.currentText())
        if cs != _('Tags'):
            genre_source_spec = self.genre_source_fields[cs]
            self.genre_source_field_name = genre_source_spec['field']
        self.generate_genres.clicked.connect(self.generate_genres_changed)
        self.generate_genres_changed(self.generate_genres.isChecked())
        self.exclusion_rules_table = ExclusionRules(self, self.exclusion_rules_gb, 'exclusion_rules_tw', exclusion_rules)
        self.prefix_rules_table = PrefixRules(self, self.prefix_rules_gb, 'prefix_rules_tw', prefix_rules)
        self.exclude_genre_changed()
        self.preset_delete_pb.clicked.connect(self.preset_remove)
        self.preset_save_pb.clicked.connect(self.preset_save)
        self.preset_field.currentIndexChanged.connect(self.preset_change)
        self.blocking_all_signals = False

    def merge_source_field_changed(self, new_index):
        if False:
            return 10
        '\n        Process changes in the merge_source_field combo box\n        '
        new_source = self.merge_source_field.currentText()
        self.merge_source_field_name = new_source
        if new_source:
            merge_source_spec = self.merge_source_fields[str(new_source)]
            self.merge_source_field_name = merge_source_spec['field']
            if not self.merge_before.isChecked() and (not self.merge_after.isChecked()):
                self.merge_after.setChecked(True)
            self.merge_before.setEnabled(True)
            self.merge_after.setEnabled(True)
            self.include_hr.setEnabled(True)
        else:
            self.merge_before.setEnabled(False)
            self.merge_after.setEnabled(False)
            self.include_hr.setEnabled(False)

    def options(self):
        if False:
            print('Hello World!')
        '\n        Return, optionally save current options\n        exclude_genre stores literally\n        Section switches store as True/False\n        others store as lists\n        '
        opts_dict = {}
        prefix_rules_processed = False
        exclusion_rules_processed = False
        for opt in self.OPTION_FIELDS:
            (c_name, c_def, c_type) = opt
            if c_name == 'exclusion_rules_tw' and exclusion_rules_processed:
                continue
            if c_name == 'prefix_rules_tw' and prefix_rules_processed:
                continue
            if c_type in ['check_box', 'radio_button']:
                opt_value = getattr(self, c_name).isChecked()
            elif c_type in ['combo_box']:
                opt_value = str(getattr(self, c_name).currentText()).strip()
            elif c_type in ['line_edit']:
                opt_value = str(getattr(self, c_name).text()).strip()
            elif c_type in ['spin_box']:
                opt_value = str(getattr(self, c_name).value())
            elif c_type in ['table_widget']:
                if c_name == 'prefix_rules_tw':
                    opt_value = self.prefix_rules_table.get_data()
                    prefix_rules_processed = True
                if c_name == 'exclusion_rules_tw':
                    opt_value = self.exclusion_rules_table.get_data()
                    exclusion_rules_processed = True
            gprefs.set(self.name + '_' + c_name, opt_value)
            if c_name in ['exclusion_rules_tw', 'prefix_rules_tw']:
                self.construct_tw_opts_object(c_name, opt_value, opts_dict)
            else:
                opts_dict[c_name] = opt_value
        checked = ''
        if self.merge_before.isChecked():
            checked = 'before'
        elif self.merge_after.isChecked():
            checked = 'after'
        include_hr = self.include_hr.isChecked()
        self.merge_source_field_name = ''
        cs = str(self.merge_source_field.currentText())
        if cs and cs in self.merge_source_fields:
            merge_source_spec = self.merge_source_fields[cs]
            self.merge_source_field_name = merge_source_spec['field']
        self.header_note_source_field_name = ''
        cs = str(self.header_note_source_field.currentText())
        if cs and cs in self.header_note_source_fields:
            header_note_source_spec = self.header_note_source_fields[cs]
            self.header_note_source_field_name = header_note_source_spec['field']
        self.genre_source_field_name = _('Tags')
        cs = str(self.genre_source_field.currentText())
        if cs != _('Tags') and cs and (cs in self.genre_source_fields):
            genre_source_spec = self.genre_source_fields[cs]
            self.genre_source_field_name = genre_source_spec['field']
        opts_dict['merge_comments_rule'] = '%s:%s:%s' % (self.merge_source_field_name, checked, include_hr)
        opts_dict['header_note_source_field'] = self.header_note_source_field_name
        opts_dict['genre_source_field'] = self.genre_source_field_name
        if opts_dict['exclude_genre'] == '':
            opts_dict['exclude_genre'] = 'a^'
        try:
            opts_dict['output_profile'] = [load_defaults('page_setup')['output_profile']]
        except:
            opts_dict['output_profile'] = ['default']
        if False and self.DEBUG:
            print('opts_dict')
            for opt in sorted(opts_dict.keys(), key=sort_key):
                print(f' {opt}: {repr(opts_dict[opt])}')
        return opts_dict

    def populate_combo_boxes(self):
        if False:
            for i in range(10):
                print('nop')
        custom_fields = {}
        for custom_field in self.all_custom_fields:
            field_md = self.db.metadata_for_field(custom_field)
            if field_md['datatype'] in ['bool', 'composite', 'datetime', 'enumeration', 'text']:
                custom_fields[field_md['name']] = {'field': custom_field, 'datatype': field_md['datatype']}
        custom_fields = {}
        for custom_field in self.all_custom_fields:
            field_md = self.db.metadata_for_field(custom_field)
            if field_md['datatype'] in ['bool', 'composite', 'datetime', 'enumeration', 'text']:
                custom_fields[field_md['name']] = {'field': custom_field, 'datatype': field_md['datatype']}
        self.header_note_source_field.addItem('')
        for cf in sorted(custom_fields, key=sort_key):
            self.header_note_source_field.addItem(cf)
        self.header_note_source_fields = custom_fields
        self.header_note_source_field.currentIndexChanged.connect(self.header_note_source_field_changed)
        custom_fields = {}
        for custom_field in self.all_custom_fields:
            field_md = self.db.metadata_for_field(custom_field)
            if field_md['datatype'] in ['text', 'comments', 'composite']:
                custom_fields[field_md['name']] = {'field': custom_field, 'datatype': field_md['datatype']}
        self.merge_source_field.addItem('')
        for cf in sorted(custom_fields, key=sort_key):
            self.merge_source_field.addItem(cf)
        self.merge_source_fields = custom_fields
        self.merge_source_field.currentIndexChanged.connect(self.merge_source_field_changed)
        self.merge_before.setEnabled(False)
        self.merge_after.setEnabled(False)
        self.include_hr.setEnabled(False)
        custom_fields = {_('Tags'): {'field': None, 'datatype': None}}
        for custom_field in self.all_custom_fields:
            field_md = self.db.metadata_for_field(custom_field)
            if field_md['datatype'] in ['text', 'enumeration']:
                custom_fields[field_md['name']] = {'field': custom_field, 'datatype': field_md['datatype']}
        for cf in sorted(custom_fields, key=sort_key):
            self.genre_source_field.addItem(cf)
        self.genre_source_fields = custom_fields
        self.genre_source_field.currentIndexChanged.connect(self.genre_source_field_changed)
        self.presets = JSONConfig('catalog_presets')
        self.preset_field.addItem('')
        self.preset_field_values = sorted(self.presets, key=sort_key)
        self.preset_field.addItems(self.preset_field_values)

    def preset_change(self, idx):
        if False:
            print('Hello World!')
        '\n        Update catalog options from current preset\n        '
        if idx <= 0:
            return
        current_preset = self.preset_field.currentText()
        options = self.presets[current_preset]
        exclusion_rules = []
        prefix_rules = []
        self.block_all_signals(True)
        for opt in self.OPTION_FIELDS:
            (c_name, c_def, c_type) = opt
            if c_name == 'preset_field':
                continue
            if c_name in options:
                opt_value = options[c_name]
            else:
                continue
            if c_type in ['check_box']:
                getattr(self, c_name).setChecked(eval(str(opt_value)))
                if c_name == 'generate_genres':
                    self.genre_source_field.setEnabled(eval(str(opt_value)))
            elif c_type in ['combo_box']:
                if opt_value is None:
                    index = 0
                    if c_name == 'genre_source_field':
                        index = self.genre_source_field.findText(_('Tags'))
                else:
                    index = getattr(self, c_name).findText(opt_value)
                    if index == -1:
                        if c_name == 'read_source_field':
                            index = self.read_source_field.findText(_('Tags'))
                        elif c_name == 'genre_source_field':
                            index = self.genre_source_field.findText(_('Tags'))
                getattr(self, c_name).setCurrentIndex(index)
            elif c_type in ['line_edit']:
                getattr(self, c_name).setText(opt_value if opt_value else '')
            elif c_type in ['radio_button'] and opt_value is not None:
                getattr(self, c_name).setChecked(opt_value)
            elif c_type in ['spin_box']:
                getattr(self, c_name).setValue(float(opt_value))
            if c_type == 'table_widget':
                if c_name == 'exclusion_rules_tw':
                    if opt_value not in exclusion_rules:
                        exclusion_rules.append(opt_value)
                if c_name == 'prefix_rules_tw':
                    if opt_value not in prefix_rules:
                        prefix_rules.append(opt_value)
        self.exclusion_rules_table.clearLayout()
        self.exclusion_rules_table = ExclusionRules(self, self.exclusion_rules_gb, 'exclusion_rules_tw', exclusion_rules)
        self.prefix_rules_table.clearLayout()
        self.prefix_rules_table = PrefixRules(self, self.prefix_rules_gb, 'prefix_rules_tw', prefix_rules)
        self.exclude_genre_changed()
        format = options['format']
        title = options['catalog_title']
        self.set_format_and_title(format, title)
        self.generate_descriptions_changed(self.generate_descriptions.isChecked())
        self.block_all_signals(False)

    def preset_remove(self):
        if False:
            while True:
                i = 10
        if self.preset_field.currentIndex() == 0:
            return
        if not question_dialog(self, _('Delete saved catalog preset'), _('The selected saved catalog preset will be deleted. Are you sure?')):
            return
        item_id = self.preset_field.currentIndex()
        item_name = str(self.preset_field.currentText())
        self.preset_field.blockSignals(True)
        self.preset_field.removeItem(item_id)
        self.preset_field.blockSignals(False)
        self.preset_field.setCurrentIndex(0)
        if item_name in self.presets.keys():
            del self.presets[item_name]
            self.presets.commit()

    def preset_save(self):
        if False:
            print('Hello World!')
        names = ['']
        names.extend(self.preset_field_values)
        try:
            dex = names.index(self.preset_search_name)
        except:
            dex = 0
        name = ''
        while not name:
            (name, ok) = QInputDialog.getItem(self, _('Save catalog preset'), _('Preset name:'), names, dex, True)
            if not ok:
                return
            if not name:
                error_dialog(self, _('Save catalog preset'), _('You must provide a name.'), show=True)
        new = True
        name = str(name)
        if name in self.presets.keys():
            if not question_dialog(self, _('Save catalog preset'), _('That saved preset already exists and will be overwritten. Are you sure?')):
                return
            new = False
        preset = {}
        prefix_rules_processed = False
        exclusion_rules_processed = False
        for opt in self.OPTION_FIELDS:
            (c_name, c_def, c_type) = opt
            if c_name == 'exclusion_rules_tw' and exclusion_rules_processed:
                continue
            if c_name == 'prefix_rules_tw' and prefix_rules_processed:
                continue
            if c_type in ['check_box', 'radio_button']:
                opt_value = getattr(self, c_name).isChecked()
            elif c_type in ['combo_box']:
                if c_name == 'preset_field':
                    continue
                opt_value = str(getattr(self, c_name).currentText()).strip()
            elif c_type in ['line_edit']:
                opt_value = str(getattr(self, c_name).text()).strip()
            elif c_type in ['spin_box']:
                opt_value = str(getattr(self, c_name).value())
            elif c_type in ['table_widget']:
                if c_name == 'prefix_rules_tw':
                    opt_value = self.prefix_rules_table.get_data()
                    prefix_rules_processed = True
                if c_name == 'exclusion_rules_tw':
                    opt_value = self.exclusion_rules_table.get_data()
                    exclusion_rules_processed = True
            preset[c_name] = opt_value
            if c_name in ['exclusion_rules_tw', 'prefix_rules_tw']:
                self.construct_tw_opts_object(c_name, opt_value, preset)
        (format, title) = self.get_format_and_title()
        preset['format'] = format
        preset['catalog_title'] = title
        checked = ''
        if self.merge_before.isChecked():
            checked = 'before'
        elif self.merge_after.isChecked():
            checked = 'after'
        include_hr = self.include_hr.isChecked()
        preset['merge_comments_rule'] = '%s:%s:%s' % (self.merge_source_field_name, checked, include_hr)
        preset['header_note_source_field'] = str(self.header_note_source_field.currentText())
        preset['genre_source_field'] = str(self.genre_source_field.currentText())
        try:
            preset['output_profile'] = load_defaults('page_setup')['output_profile']
        except:
            preset['output_profile'] = 'default'
        self.presets[name] = preset
        self.presets.commit()
        if new:
            self.preset_field.blockSignals(True)
            self.preset_field.clear()
            self.preset_field.addItem('')
            self.preset_field_values = sorted(self.presets, key=sort_key)
            self.preset_field.addItems(self.preset_field_values)
            self.preset_field.blockSignals(False)
        self.preset_field.setCurrentIndex(self.preset_field.findText(name))

    def set_format_and_title(self, format, title):
        if False:
            i = 10
            return i + 15
        parent = self.parent_ref()
        if parent is not None:
            if format:
                index = parent.format.findText(format)
                parent.format.blockSignals(True)
                parent.format.setCurrentIndex(index)
                parent.format.blockSignals(False)
            if title:
                parent.title.setText(title)

    def settings_changed(self, source):
        if False:
            print('Hello World!')
        '\n        When anything changes, clear Preset combobox\n        '
        if self.DEBUG:
            print('settings_changed: %s' % source)
        self.preset_field.setCurrentIndex(0)

    def show_help(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Display help file\n        '
        open_url(QUrl(localize_user_manual_link('https://manual.calibre-ebook.com/catalogs.html')))

class CheckableTableWidgetItem(QTableWidgetItem):
    """
    Borrowed from kiwidude
    """

    def __init__(self, checked=False, is_tristate=False):
        if False:
            return 10
        QTableWidgetItem.__init__(self, '')
        self.setFlags(Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
        if is_tristate:
            self.setFlags(self.flags() | Qt.ItemFlag.ItemIsTristate)
        if checked:
            self.setCheckState(Qt.CheckState.Checked)
        elif is_tristate and checked is None:
            self.setCheckState(Qt.CheckState.PartiallyChecked)
        else:
            self.setCheckState(Qt.CheckState.Unchecked)

    def get_boolean_value(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a boolean value indicating whether checkbox is checked\n        If this is a tristate checkbox, a partially checked value is returned as None\n        '
        if self.checkState() == Qt.CheckState.PartiallyChecked:
            return None
        else:
            return self.checkState() == Qt.CheckState.Checked

class NoWheelComboBox(QComboBox):

    def wheelEvent(self, event):
        if False:
            return 10
        event.ignore()

class ComboBox(NoWheelComboBox):

    def __init__(self, parent, items, selected_text, insert_blank=True):
        if False:
            for i in range(10):
                print('nop')
        NoWheelComboBox.__init__(self, parent)
        self.populate_combo(items, selected_text, insert_blank)

    def populate_combo(self, items, selected_text, insert_blank):
        if False:
            while True:
                i = 10
        if insert_blank:
            self.addItems([''])
        self.addItems(items)
        if selected_text:
            idx = self.findText(selected_text)
            self.setCurrentIndex(idx)
        else:
            self.setCurrentIndex(0)

class GenericRulesTable(QTableWidget):
    """
    Generic methods for managing rows in a QTableWidget
    """
    DEBUG = False
    MAXIMUM_TABLE_HEIGHT = 113
    NAME_FIELD_WIDTH = 225

    def __init__(self, parent, parent_gb, object_name, rules):
        if False:
            while True:
                i = 10
        self.parent = parent
        self.rules = rules
        self.eligible_custom_fields = parent.eligible_custom_fields
        self.db = parent.db
        QTableWidget.__init__(self)
        self.setObjectName(object_name)
        self.layout = parent_gb.layout()
        sizePolicy = QSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        self.setSizePolicy(sizePolicy)
        self.setMaximumSize(QSize(16777215, self.MAXIMUM_TABLE_HEIGHT))
        self.setColumnCount(0)
        self.setRowCount(0)
        self.layout.addWidget(self)
        self.last_row_selected = self.currentRow()
        self.last_rows_selected = self.selectionModel().selectedRows()
        self._init_controls()
        self.cellChanged.connect(self.enabled_state_changed)

    def _init_controls(self):
        if False:
            while True:
                i = 10
        vbl = QVBoxLayout()
        self.move_rule_up_tb = QToolButton()
        self.move_rule_up_tb.setObjectName('move_rule_up_tb')
        self.move_rule_up_tb.setToolTip('Move rule up')
        self.move_rule_up_tb.setIcon(QIcon.ic('arrow-up.png'))
        self.move_rule_up_tb.clicked.connect(self.move_row_up)
        vbl.addWidget(self.move_rule_up_tb)
        self.add_rule_tb = QToolButton()
        self.add_rule_tb.setObjectName('add_rule_tb')
        self.add_rule_tb.setToolTip('Add a new rule')
        self.add_rule_tb.setIcon(QIcon.ic('plus.png'))
        self.add_rule_tb.clicked.connect(self.add_row)
        vbl.addWidget(self.add_rule_tb)
        self.delete_rule_tb = QToolButton()
        self.delete_rule_tb.setObjectName('delete_rule_tb')
        self.delete_rule_tb.setToolTip('Delete selected rule')
        self.delete_rule_tb.setIcon(QIcon.ic('list_remove.png'))
        self.delete_rule_tb.clicked.connect(self.delete_row)
        vbl.addWidget(self.delete_rule_tb)
        self.move_rule_down_tb = QToolButton()
        self.move_rule_down_tb.setObjectName('move_rule_down_tb')
        self.move_rule_down_tb.setToolTip('Move rule down')
        self.move_rule_down_tb.setIcon(QIcon.ic('arrow-down.png'))
        self.move_rule_down_tb.clicked.connect(self.move_row_down)
        vbl.addWidget(self.move_rule_down_tb)
        self.layout.addLayout(vbl)

    def add_row(self):
        if False:
            while True:
                i = 10
        self.setFocus()
        row = self.last_row_selected + 1
        if self.DEBUG:
            print('%s:add_row(): at row: %d' % (self.objectName(), row))
        self.insertRow(row)
        self.populate_table_row(row, self.create_blank_row_data())
        self.select_and_scroll_to_row(row)
        self.resizeColumnsToContents()
        self.horizontalHeader().setStretchLastSection(True)

    def clearLayout(self):
        if False:
            i = 10
            return i + 15
        if self.layout is not None:
            old_layout = self.layout
            for child in old_layout.children():
                for i in reversed(range(child.count())):
                    if child.itemAt(i).widget() is not None:
                        child.itemAt(i).widget().setParent(None)
                sip.delete(child)
            for i in reversed(range(old_layout.count())):
                if old_layout.itemAt(i).widget() is not None:
                    old_layout.itemAt(i).widget().setParent(None)

    def delete_row(self):
        if False:
            print('Hello World!')
        if self.DEBUG:
            print('%s:delete_row()' % self.objectName())
        self.setFocus()
        rows = self.last_rows_selected
        if len(rows) == 0:
            return
        first = rows[0].row() + 1
        last = rows[-1].row() + 1
        first_rule_name = str(self.cellWidget(first - 1, self.COLUMNS['NAME']['ordinal']).text()).strip()
        message = _("Are you sure you want to delete '%s'?") % first_rule_name
        if len(rows) > 1:
            message = _('Are you sure you want to delete rules #%(first)d-%(last)d?') % dict(first=first, last=last)
        if not question_dialog(self, _('Delete Rule'), message, show_copy_button=False):
            return
        first_sel_row = self.currentRow()
        for selrow in reversed(rows):
            self.removeRow(selrow.row())
        if first_sel_row < self.rowCount():
            self.select_and_scroll_to_row(first_sel_row)
        elif self.rowCount() > 0:
            self.select_and_scroll_to_row(first_sel_row - 1)

    def enabled_state_changed(self, row, col):
        if False:
            while True:
                i = 10
        if col in [self.COLUMNS['ENABLED']['ordinal']]:
            self.select_and_scroll_to_row(row)
            self.settings_changed('enabled_state_changed')
            if self.DEBUG:
                print('%s:enabled_state_changed(): row %d col %d' % (self.objectName(), row, col))

    def focusInEvent(self, e):
        if False:
            while True:
                i = 10
        if self.DEBUG:
            print('%s:focusInEvent()' % self.objectName())

    def focusOutEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.last_row_selected = self.currentRow()
        self.last_rows_selected = self.selectionModel().selectedRows()
        self.clearSelection()
        if self.DEBUG:
            print('%s:focusOutEvent(): self.last_row_selected: %d' % (self.objectName(), self.last_row_selected))

    def move_row_down(self):
        if False:
            print('Hello World!')
        self.setFocus()
        rows = self.last_rows_selected
        if len(rows) == 0:
            return
        last_sel_row = rows[-1].row()
        if last_sel_row == self.rowCount() - 1:
            return
        self.blockSignals(True)
        for selrow in reversed(rows):
            dest_row = selrow.row() + 1
            src_row = selrow.row()
            if self.DEBUG:
                print('%s:move_row_down() %d -> %d' % (self.objectName(), src_row, dest_row))
            saved_data = self.convert_row_to_data(dest_row)
            self.removeRow(dest_row)
            self.insertRow(src_row)
            self.populate_table_row(src_row, saved_data)
        scroll_to_row = last_sel_row + 1
        self.select_and_scroll_to_row(scroll_to_row)
        self.blockSignals(False)

    def move_row_up(self):
        if False:
            for i in range(10):
                print('nop')
        self.setFocus()
        rows = self.last_rows_selected
        if len(rows) == 0:
            return
        first_sel_row = rows[0].row()
        if first_sel_row <= 0:
            return
        self.blockSignals(True)
        for selrow in rows:
            if self.DEBUG:
                print('%s:move_row_up() %d -> %d' % (self.objectName(), selrow.row(), selrow.row() - 1))
            saved_data = self.convert_row_to_data(selrow.row() - 1)
            self.insertRow(selrow.row() + 1)
            self.populate_table_row(selrow.row() + 1, saved_data)
            self.removeRow(selrow.row() - 1)
        scroll_to_row = first_sel_row
        if scroll_to_row > 0:
            scroll_to_row = scroll_to_row - 1
        self.select_and_scroll_to_row(scroll_to_row)
        self.blockSignals(False)

    def populate_table(self):
        if False:
            for i in range(10):
                print('nop')
        rules = self.rules
        if rules and type(rules[0]) is list:
            rules = rules[0]
        self.setFocus()
        rules = sorted(rules, key=lambda k: k['ordinal'])
        for (row, rule) in enumerate(rules):
            self.insertRow(row)
            self.select_and_scroll_to_row(row)
            self.populate_table_row(row, rule)
        self.selectRow(0)

    def resize_name(self):
        if False:
            for i in range(10):
                print('nop')
        self.setColumnWidth(1, self.NAME_FIELD_WIDTH)

    def rule_name_edited(self):
        if False:
            while True:
                i = 10
        if self.DEBUG:
            print('%s:rule_name_edited()' % self.objectName())
        current_row = self.currentRow()
        self.cellWidget(current_row, 1).home(False)
        self.select_and_scroll_to_row(current_row)
        self.settings_changed('rule_name_edited')

    def select_and_scroll_to_row(self, row):
        if False:
            while True:
                i = 10
        self.setFocus()
        self.selectRow(row)
        self.scrollToItem(self.currentItem())
        self.last_row_selected = self.currentRow()
        self.last_rows_selected = self.selectionModel().selectedRows()

    def settings_changed(self, source):
        if False:
            print('Hello World!')
        if not self.parent.blocking_all_signals:
            self.parent.settings_changed(source)

    def _source_index_changed(self, combo):
        if False:
            return 10
        for row in range(self.rowCount()):
            if self.cellWidget(row, self.COLUMNS['FIELD']['ordinal']) is combo:
                break
        if self.DEBUG:
            print('%s:_source_index_changed(): calling source_index_changed with row: %d ' % (self.objectName(), row))
        self.source_index_changed(combo, row)

    def source_index_changed(self, combo, row, pattern=''):
        if False:
            while True:
                i = 10
        source_field = combo.currentText()
        if source_field == '':
            values = []
        elif source_field == _('Tags'):
            values = sorted(self.db.all_tags(), key=sort_key)
        elif self.eligible_custom_fields[str(source_field)]['datatype'] in ['enumeration', 'text']:
            values = self.db.all_custom(self.db.field_metadata.key_to_label(self.eligible_custom_fields[str(source_field)]['field']))
            values = sorted(values, key=sort_key)
        elif self.eligible_custom_fields[str(source_field)]['datatype'] in ['bool']:
            values = [_('True'), _('False'), _('unspecified')]
        elif self.eligible_custom_fields[str(source_field)]['datatype'] in ['composite']:
            values = [_('any value'), _('unspecified')]
        elif self.eligible_custom_fields[str(source_field)]['datatype'] in ['datetime']:
            values = [_('any date'), _('unspecified')]
        values_combo = ComboBox(self, values, pattern)
        values_combo.currentIndexChanged.connect(partial(self.values_index_changed, values_combo))
        self.setCellWidget(row, self.COLUMNS['PATTERN']['ordinal'], values_combo)
        self.select_and_scroll_to_row(row)
        self.settings_changed('source_index_changed')

    def values_index_changed(self, combo):
        if False:
            for i in range(10):
                print('nop')
        for row in range(self.rowCount()):
            if self.cellWidget(row, self.COLUMNS['PATTERN']['ordinal']) is combo:
                self.select_and_scroll_to_row(row)
                self.settings_changed('values_index_changed')
                break
        if self.DEBUG:
            print('%s:values_index_changed(): row %d ' % (self.objectName(), row))

class ExclusionRules(GenericRulesTable):
    COLUMNS = {'ENABLED': {'ordinal': 0, 'name': ''}, 'NAME': {'ordinal': 1, 'name': _('Name')}, 'FIELD': {'ordinal': 2, 'name': _('Field')}, 'PATTERN': {'ordinal': 3, 'name': _('Value')}}

    def __init__(self, parent, parent_gb_hl, object_name, rules):
        if False:
            i = 10
            return i + 15
        super().__init__(parent, parent_gb_hl, object_name, rules)
        self.setObjectName('exclusion_rules_table')
        self._init_table_widget()
        self._initialize()

    def _init_table_widget(self):
        if False:
            for i in range(10):
                print('nop')
        header_labels = [self.COLUMNS[index]['name'] for index in sorted(self.COLUMNS.keys(), key=lambda c: self.COLUMNS[c]['ordinal'])]
        self.setColumnCount(len(header_labels))
        self.setHorizontalHeaderLabels(header_labels)
        self.setSortingEnabled(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

    def _initialize(self):
        if False:
            return 10
        self.populate_table()
        self.resizeColumnsToContents()
        self.resize_name()
        self.horizontalHeader().setStretchLastSection(True)
        self.clearSelection()

    def convert_row_to_data(self, row):
        if False:
            for i in range(10):
                print('nop')
        data = self.create_blank_row_data()
        data['ordinal'] = row
        data['enabled'] = self.item(row, self.COLUMNS['ENABLED']['ordinal']).checkState() == Qt.CheckState.Checked
        data['name'] = str(self.cellWidget(row, self.COLUMNS['NAME']['ordinal']).text()).strip()
        data['field'] = str(self.cellWidget(row, self.COLUMNS['FIELD']['ordinal']).currentText()).strip()
        data['pattern'] = str(self.cellWidget(row, self.COLUMNS['PATTERN']['ordinal']).currentText()).strip()
        return data

    def create_blank_row_data(self):
        if False:
            while True:
                i = 10
        data = {}
        data['ordinal'] = -1
        data['enabled'] = True
        data['name'] = 'New rule'
        data['field'] = ''
        data['pattern'] = ''
        return data

    def get_data(self):
        if False:
            for i in range(10):
                print('nop')
        data_items = []
        for row in range(self.rowCount()):
            data = self.convert_row_to_data(row)
            data_items.append({'ordinal': data['ordinal'], 'enabled': data['enabled'], 'name': data['name'], 'field': data['field'], 'pattern': data['pattern']})
        return data_items

    def populate_table_row(self, row, data):
        if False:
            print('Hello World!')

        def set_rule_name_in_row(row, col, name=''):
            if False:
                i = 10
                return i + 15
            rule_name = QLineEdit(name)
            rule_name.home(False)
            rule_name.editingFinished.connect(self.rule_name_edited)
            self.setCellWidget(row, col, rule_name)

        def set_source_field_in_row(row, col, field=''):
            if False:
                print('Hello World!')
            source_combo = ComboBox(self, sorted(self.eligible_custom_fields.keys(), key=sort_key), field)
            source_combo.currentIndexChanged.connect(partial(self._source_index_changed, source_combo))
            self.setCellWidget(row, col, source_combo)
            return source_combo
        self.blockSignals(True)
        check_box = CheckableTableWidgetItem(data['enabled'])
        self.setItem(row, self.COLUMNS['ENABLED']['ordinal'], check_box)
        set_rule_name_in_row(row, self.COLUMNS['NAME']['ordinal'], name=data['name'])
        source_combo = set_source_field_in_row(row, self.COLUMNS['FIELD']['ordinal'], field=data['field'])
        self.source_index_changed(source_combo, row, pattern=data['pattern'])
        self.blockSignals(False)

class PrefixRules(GenericRulesTable):
    COLUMNS = {'ENABLED': {'ordinal': 0, 'name': ''}, 'NAME': {'ordinal': 1, 'name': _('Name')}, 'PREFIX': {'ordinal': 2, 'name': _('Prefix')}, 'FIELD': {'ordinal': 3, 'name': _('Field')}, 'PATTERN': {'ordinal': 4, 'name': _('Value')}}

    def __init__(self, parent, parent_gb_hl, object_name, rules):
        if False:
            while True:
                i = 10
        super().__init__(parent, parent_gb_hl, object_name, rules)
        self.setObjectName('prefix_rules_table')
        self._init_table_widget()
        self._initialize()

    def _init_table_widget(self):
        if False:
            for i in range(10):
                print('nop')
        header_labels = [self.COLUMNS[index]['name'] for index in sorted(self.COLUMNS.keys(), key=lambda c: self.COLUMNS[c]['ordinal'])]
        self.setColumnCount(len(header_labels))
        self.setHorizontalHeaderLabels(header_labels)
        self.setSortingEnabled(False)
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)

    def _initialize(self):
        if False:
            i = 10
            return i + 15
        self.generate_prefix_list()
        self.populate_table()
        self.resizeColumnsToContents()
        self.resize_name()
        self.horizontalHeader().setStretchLastSection(True)
        self.clearSelection()

    def convert_row_to_data(self, row):
        if False:
            print('Hello World!')
        data = self.create_blank_row_data()
        data['ordinal'] = row
        data['enabled'] = self.item(row, self.COLUMNS['ENABLED']['ordinal']).checkState() == Qt.CheckState.Checked
        data['name'] = str(self.cellWidget(row, self.COLUMNS['NAME']['ordinal']).text()).strip()
        data['prefix'] = str(self.cellWidget(row, self.COLUMNS['PREFIX']['ordinal']).currentText()).strip()
        data['field'] = str(self.cellWidget(row, self.COLUMNS['FIELD']['ordinal']).currentText()).strip()
        data['pattern'] = str(self.cellWidget(row, self.COLUMNS['PATTERN']['ordinal']).currentText()).strip()
        return data

    def create_blank_row_data(self):
        if False:
            while True:
                i = 10
        data = {}
        data['ordinal'] = -1
        data['enabled'] = True
        data['name'] = 'New rule'
        data['field'] = ''
        data['pattern'] = ''
        data['prefix'] = ''
        return data

    def generate_prefix_list(self):
        if False:
            return 10

        def prefix_sorter(item):
            if False:
                for i in range(10):
                    print('nop')
            key = item
            if item[0] == '_':
                key = 'zzz' + item
            return key
        raw_prefix_list = [('Ampersand', '&'), ('Angle left double', '«'), ('Angle left', '‹'), ('Angle right double', '»'), ('Angle right', '›'), ('Arrow carriage return', '↵'), ('Arrow double', '↔'), ('Arrow down', '↓'), ('Arrow left', '←'), ('Arrow right', '→'), ('Arrow up', '↑'), ('Asterisk', '*'), ('At sign', '@'), ('Bullet smallest', '⋅'), ('Bullet small', '·'), ('Bullet', '•'), ('Cards clubs', '♣'), ('Cards diamonds', '♦'), ('Cards hearts', '♥'), ('Cards spades', '♠'), ('Caret', '^'), ('Checkmark', '✓'), ('Copyright circle c', '©'), ('Copyright circle r', '®'), ('Copyright trademark', '™'), ('Currency cent', '¢'), ('Currency dollar', '$'), ('Currency euro', '€'), ('Currency pound', '£'), ('Currency yen', '¥'), ('Dagger double', '‡'), ('Dagger', '†'), ('Degree', '°'), ('Dots3', '∴'), ('Hash', '#'), ('Infinity', '∞'), ('Lozenge', '◊'), ('Math divide', '÷'), ('Math empty', '∅'), ('Math equals', '='), ('Math minus', '−'), ('Math plus circled', '⊕'), ('Math times circled', '⊗'), ('Math times', '×'), ('Paragraph', '¶'), ('Percent', '%'), ('Plus-or-minus', '±'), ('Plus', '+'), ('Punctuation colon', ':'), ('Punctuation colon-semi', ';'), ('Punctuation exclamation', '!'), ('Punctuation question', '?'), ('Punctuation period', '.'), ('Punctuation slash back', '\\'), ('Punctuation slash forward', '/'), ('Section', '§'), ('Tilde', '~'), ('Vertical bar', '|'), ('Vertical bar broken', '¦'), ('_0', '0'), ('_1', '1'), ('_2', '2'), ('_3', '3'), ('_4', '4'), ('_5', '5'), ('_6', '6'), ('_7', '7'), ('_8', '8'), ('_9', '9'), ('_A', 'A'), ('_B', 'B'), ('_C', 'C'), ('_D', 'D'), ('_E', 'E'), ('_F', 'F'), ('_G', 'G'), ('_H', 'H'), ('_I', 'I'), ('_J', 'J'), ('_K', 'K'), ('_L', 'L'), ('_M', 'M'), ('_N', 'N'), ('_O', 'O'), ('_P', 'P'), ('_Q', 'Q'), ('_R', 'R'), ('_S', 'S'), ('_T', 'T'), ('_U', 'U'), ('_V', 'V'), ('_W', 'W'), ('_X', 'X'), ('_Y', 'Y'), ('_Z', 'Z'), ('_a', 'a'), ('_b', 'b'), ('_c', 'c'), ('_d', 'd'), ('_e', 'e'), ('_f', 'f'), ('_g', 'g'), ('_h', 'h'), ('_i', 'i'), ('_j', 'j'), ('_k', 'k'), ('_l', 'l'), ('_m', 'm'), ('_n', 'n'), ('_o', 'o'), ('_p', 'p'), ('_q', 'q'), ('_r', 'r'), ('_s', 's'), ('_t', 't'), ('_u', 'u'), ('_v', 'v'), ('_w', 'w'), ('_x', 'x'), ('_y', 'y'), ('_z', 'z')]
        raw_prefix_list = sorted(raw_prefix_list, key=prefix_sorter)
        self.prefix_list = [x[1] for x in raw_prefix_list]

    def get_data(self):
        if False:
            for i in range(10):
                print('nop')
        data_items = []
        for row in range(self.rowCount()):
            data = self.convert_row_to_data(row)
            data_items.append({'ordinal': data['ordinal'], 'enabled': data['enabled'], 'name': data['name'], 'field': data['field'], 'pattern': data['pattern'], 'prefix': data['prefix']})
        return data_items

    def populate_table_row(self, row, data):
        if False:
            i = 10
            return i + 15

        def set_prefix_field_in_row(row, col, field=''):
            if False:
                while True:
                    i = 10
            prefix_combo = ComboBox(self, self.prefix_list, field)
            prefix_combo.currentIndexChanged.connect(partial(self.settings_changed, 'set_prefix_field_in_row'))
            self.setCellWidget(row, col, prefix_combo)

        def set_rule_name_in_row(row, col, name=''):
            if False:
                print('Hello World!')
            rule_name = QLineEdit(name)
            rule_name.home(False)
            rule_name.editingFinished.connect(self.rule_name_edited)
            self.setCellWidget(row, col, rule_name)

        def set_source_field_in_row(row, col, field=''):
            if False:
                print('Hello World!')
            source_combo = ComboBox(self, sorted(self.eligible_custom_fields.keys(), key=sort_key), field)
            source_combo.currentIndexChanged.connect(partial(self._source_index_changed, source_combo))
            self.setCellWidget(row, col, source_combo)
            return source_combo
        self.blockSignals(True)
        self.setItem(row, self.COLUMNS['ENABLED']['ordinal'], CheckableTableWidgetItem(data['enabled']))
        set_rule_name_in_row(row, self.COLUMNS['NAME']['ordinal'], name=data['name'])
        set_prefix_field_in_row(row, self.COLUMNS['PREFIX']['ordinal'], field=data['prefix'])
        source_combo = set_source_field_in_row(row, self.COLUMNS['FIELD']['ordinal'], field=data['field'])
        self.source_index_changed(source_combo, row, pattern=data['pattern'])
        self.blockSignals(False)