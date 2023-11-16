import os.path
from PyQt6 import QtWidgets
from PyQt6.QtCore import QStandardPaths
from PyQt6.QtGui import QPalette
from picard.config import BoolOption, TextOption, get_config
from picard.script import ScriptParser
from picard.ui.options import OptionsCheckError, OptionsPage, register_options_page
from picard.ui.options.scripting import ScriptCheckError, ScriptingDocumentationDialog
from picard.ui.scripteditor import ScriptEditorDialog, ScriptEditorExamples, populate_script_selection_combo_box, synchronize_vertical_scrollbars
from picard.ui.ui_options_renaming import Ui_RenamingOptionsPage
_default_music_dir = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.MusicLocation)

class RenamingOptionsPage(OptionsPage):
    NAME = 'filerenaming'
    TITLE = N_('File Naming')
    PARENT = None
    SORT_ORDER = 40
    ACTIVE = True
    HELP_URL = '/config/options_filerenaming.html'
    options = [BoolOption('setting', 'rename_files', False), BoolOption('setting', 'move_files', False), TextOption('setting', 'move_files_to', _default_music_dir), BoolOption('setting', 'move_additional_files', False), TextOption('setting', 'move_additional_files_pattern', '*.jpg *.png'), BoolOption('setting', 'delete_empty_dirs', True)]

    def __init__(self, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self.script_text = ''
        self.compat_options = {}
        self.ui = Ui_RenamingOptionsPage()
        self.ui.setupUi(self)
        self.ui.rename_files.clicked.connect(self.update_examples_from_local)
        self.ui.move_files.clicked.connect(self.update_examples_from_local)
        self.ui.move_files_to.editingFinished.connect(self.update_examples_from_local)
        self.ui.move_files.toggled.connect(self.toggle_file_naming_format)
        self.ui.rename_files.toggled.connect(self.toggle_file_naming_format)
        self.toggle_file_naming_format(None)
        self.ui.open_script_editor.clicked.connect(self.show_script_editing_page)
        self.ui.move_files_to_browse.clicked.connect(self.move_files_to_browse)
        self.ui.naming_script_selector.currentIndexChanged.connect(self.update_selector_in_editor)
        self.ui.example_filename_after.itemSelectionChanged.connect(self.match_before_to_after)
        self.ui.example_filename_before.itemSelectionChanged.connect(self.match_after_to_before)
        script_edit = self.ui.move_additional_files_pattern
        self.script_palette_normal = script_edit.palette()
        self.script_palette_readonly = QPalette(self.script_palette_normal)
        disabled_color = self.script_palette_normal.color(QPalette.ColorGroup.Inactive, QPalette.ColorRole.Window)
        self.script_palette_readonly.setColor(QPalette.ColorGroup.Disabled, QPalette.ColorRole.Base, disabled_color)
        self.ui.example_filename_sample_files_button.clicked.connect(self.update_example_files)
        self.examples = ScriptEditorExamples(tagger=self.tagger)
        self.script_editor_dialog = None
        self.ui.example_selection_note.setText(self.examples.get_notes_text())
        self.ui.example_filename_sample_files_button.setToolTip(self.examples.get_tooltip_text())
        synchronize_vertical_scrollbars((self.ui.example_filename_before, self.ui.example_filename_after))
        self.current_row = -1

    def update_selector_from_editor(self):
        if False:
            print('Hello World!')
        'Update the script selector combo box from the script editor page.\n        '
        self.naming_scripts = self.script_editor_dialog.naming_scripts
        self.selected_naming_script_id = self.script_editor_dialog.selected_script_id
        populate_script_selection_combo_box(self.naming_scripts, self.selected_naming_script_id, self.ui.naming_script_selector)
        self.display_examples()

    def update_selector_from_settings(self):
        if False:
            for i in range(10):
                print('nop')
        'Update the script selector combo box from the settings.\n        '
        populate_script_selection_combo_box(self.naming_scripts, self.selected_naming_script_id, self.ui.naming_script_selector)
        self.update_selector_in_editor()

    def update_selector_in_editor(self):
        if False:
            for i in range(10):
                print('nop')
        'Update the selection in the script editor page to match local selection.\n        '
        idx = self.ui.naming_script_selector.currentIndex()
        if self.script_editor_dialog:
            self.script_editor_dialog.set_selected_script_index(idx)
        else:
            script_item = self.ui.naming_script_selector.itemData(idx)
            self.script_text = script_item['script']
            self.selected_naming_script_id = script_item['id']
            self.examples.update_examples(script_text=self.script_text)
            self.update_examples_from_local()

    def match_after_to_before(self):
        if False:
            i = 10
            return i + 15
        "Sets the selected item in the 'after' list to the corresponding item in the 'before' list.\n        "
        self.examples.synchronize_selected_example_lines(self.current_row, self.ui.example_filename_before, self.ui.example_filename_after)

    def match_before_to_after(self):
        if False:
            while True:
                i = 10
        "Sets the selected item in the 'before' list to the corresponding item in the 'after' list.\n        "
        self.examples.synchronize_selected_example_lines(self.current_row, self.ui.example_filename_after, self.ui.example_filename_before)

    def show_script_editing_page(self):
        if False:
            print('Hello World!')
        self.script_editor_dialog = ScriptEditorDialog.show_instance(parent=self, examples=self.examples)
        self.script_editor_dialog.signal_save.connect(self.save_from_editor)
        self.script_editor_dialog.signal_update.connect(self.display_examples)
        self.script_editor_dialog.signal_selection_changed.connect(self.update_selector_from_editor)
        self.script_editor_dialog.finished.connect(self.script_editor_dialog_close)
        if self.tagger.window.script_editor_dialog is not None:
            self.update_selector_from_editor()
        else:
            self.script_editor_dialog.loading = True
            self.script_editor_dialog.naming_scripts = self.naming_scripts
            self.script_editor_dialog.populate_script_selector()
            self.update_selector_in_editor()
            self.script_editor_dialog.loading = False
            self.update_examples_from_local()
            self.tagger.window.script_editor_dialog = True

    def script_editor_dialog_close(self):
        if False:
            for i in range(10):
                print('nop')
        self.tagger.window.script_editor_dialog = None

    def show_scripting_documentation(self):
        if False:
            print('Hello World!')
        ScriptingDocumentationDialog.show_instance(parent=self)

    def toggle_file_naming_format(self, state):
        if False:
            print('Hello World!')
        active = self.ui.move_files.isChecked() or self.ui.rename_files.isChecked()
        self.ui.open_script_editor.setEnabled(active)

    def save_from_editor(self):
        if False:
            return 10
        self.script_text = self.script_editor_dialog.get_script()
        self.update_selector_from_editor()

    def check_formats(self):
        if False:
            print('Hello World!')
        self.test()
        self.update_examples_from_local()

    def update_example_files(self):
        if False:
            i = 10
            return i + 15
        self.examples.update_sample_example_files()
        self.update_displayed_examples()

    def update_examples_from_local(self):
        if False:
            for i in range(10):
                print('nop')
        override = dict(self.compat_options)
        override['move_files'] = self.ui.move_files.isChecked()
        override['move_files_to'] = os.path.normpath(self.ui.move_files_to.text())
        override['rename_files'] = self.ui.rename_files.isChecked()
        self.examples.update_examples(override=override)
        self.update_displayed_examples()

    def update_displayed_examples(self):
        if False:
            print('Hello World!')
        if self.script_editor_dialog is not None:
            self.script_editor_dialog.display_examples()
        else:
            self.display_examples()

    def display_examples(self):
        if False:
            return 10
        self.current_row = -1
        self.examples.update_example_listboxes(self.ui.example_filename_before, self.ui.example_filename_after)

    def load(self):
        if False:
            i = 10
            return i + 15
        compat_page = self.dialog.get_page('filerenaming_compat')
        self.compat_options = compat_page.get_options()
        compat_page.options_changed.connect(self.on_compat_options_changed)
        config = get_config()
        self.ui.rename_files.setChecked(config.setting['rename_files'])
        self.ui.move_files.setChecked(config.setting['move_files'])
        self.ui.move_files_to.setText(config.setting['move_files_to'])
        self.ui.move_files_to.setCursorPosition(0)
        self.ui.move_additional_files.setChecked(config.setting['move_additional_files'])
        self.ui.move_additional_files_pattern.setText(config.setting['move_additional_files_pattern'])
        self.ui.delete_empty_dirs.setChecked(config.setting['delete_empty_dirs'])
        self.naming_scripts = config.setting['file_renaming_scripts']
        self.selected_naming_script_id = config.setting['selected_file_naming_script_id']
        if self.script_editor_dialog:
            self.script_editor_dialog.load()
        else:
            self.update_selector_from_settings()
        self.update_examples_from_local()

    def on_compat_options_changed(self, options):
        if False:
            while True:
                i = 10
        self.compat_options = options
        self.update_examples_from_local()

    def check(self):
        if False:
            while True:
                i = 10
        self.check_format()
        if self.ui.move_files.isChecked() and (not self.ui.move_files_to.text().strip()):
            raise OptionsCheckError(_('Error'), _('The location to move files to must not be empty.'))

    def check_format(self):
        if False:
            while True:
                i = 10
        parser = ScriptParser()
        try:
            parser.eval(self.script_text)
        except Exception as e:
            raise ScriptCheckError('', str(e))
        if self.ui.rename_files.isChecked():
            if not self.script_text.strip():
                raise ScriptCheckError('', _('The file naming format must not be empty.'))

    def save(self):
        if False:
            return 10
        config = get_config()
        config.setting['rename_files'] = self.ui.rename_files.isChecked()
        config.setting['move_files'] = self.ui.move_files.isChecked()
        config.setting['move_files_to'] = os.path.normpath(self.ui.move_files_to.text())
        config.setting['move_additional_files'] = self.ui.move_additional_files.isChecked()
        config.setting['move_additional_files_pattern'] = self.ui.move_additional_files_pattern.text()
        config.setting['delete_empty_dirs'] = self.ui.delete_empty_dirs.isChecked()
        config.setting['selected_file_naming_script_id'] = self.selected_naming_script_id
        self.tagger.window.enable_renaming_action.setChecked(config.setting['rename_files'])
        self.tagger.window.enable_moving_action.setChecked(config.setting['move_files'])
        self.tagger.window.make_script_selector_menu()

    def display_error(self, error):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(error, ScriptCheckError):
            super().display_error(error)

    def move_files_to_browse(self):
        if False:
            while True:
                i = 10
        path = QtWidgets.QFileDialog.getExistingDirectory(self, '', self.ui.move_files_to.text())
        if path:
            path = os.path.normpath(path)
            self.ui.move_files_to.setText(path)

    def test(self):
        if False:
            return 10
        self.ui.renaming_error.setStyleSheet('')
        self.ui.renaming_error.setText('')
        try:
            self.check_format()
        except ScriptCheckError as e:
            self.ui.renaming_error.setStyleSheet(self.STYLESHEET_ERROR)
            self.ui.renaming_error.setText(e.info)
            return
register_options_page(RenamingOptionsPage)