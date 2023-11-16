"""File explorer configuration page."""
from qtpy.QtWidgets import QGroupBox, QLabel, QPushButton, QVBoxLayout, QWidget
from spyder.api.preferences import PluginConfigPage
from spyder.api.translations import _
from spyder.config.main import NAME_FILTERS
from spyder.plugins.explorer.widgets.fileassociations import FileAssociationsWidget

class ExplorerConfigPage(PluginConfigPage):

    def setup_page(self):
        if False:
            for i in range(10):
                print('nop')
        newcb = self.create_checkbox
        general_widget = QWidget()
        basic_group = QGroupBox(_('General options'))
        check_show_hidden_files = newcb(_('Show hidden files'), 'show_hidden')
        check_single_click = newcb(_('Single click to open files'), 'single_click_to_open')
        basic_layout = QVBoxLayout()
        basic_layout.addWidget(check_show_hidden_files)
        basic_layout.addWidget(check_single_click)
        basic_group.setLayout(basic_layout)
        filter_group = QGroupBox(_('Filter settings'))
        description_label = QLabel(_('Filter files by name, extension, or more using <a href="https://en.wikipedia.org/wiki/Glob_(programming)">glob patterns.</a> Please enter the glob patterns of the files you want to show, separated by commas.'))
        description_label.setOpenExternalLinks(True)
        description_label.setWordWrap(True)
        self.edit_filename_filters = self.create_textedit('', 'name_filters', tip='Enter values separated by commas', content_type=list)
        self.edit_filename_filters.setEnabled(True)
        reset_btn = QPushButton(_('Reset to default values'))
        reset_btn.clicked.connect(self.reset_to_default)
        filter_layout = QVBoxLayout()
        filter_layout.addWidget(description_label)
        filter_layout.addWidget(self.edit_filename_filters)
        filter_layout.addWidget(reset_btn)
        filter_group.setLayout(filter_layout)
        associations_widget = QWidget()
        self.edit_file_associations = self.create_textedit('', 'file_associations', content_type=dict)
        file_associations = FileAssociationsWidget()
        file_associations.load_values(self.get_option('file_associations', {}))
        self.edit_file_associations.setVisible(False)
        layout = QVBoxLayout()
        layout.addWidget(basic_group)
        layout.addWidget(filter_group)
        general_widget.setLayout(layout)
        layout_file = QVBoxLayout()
        layout_file.addWidget(file_associations)
        layout_file.addWidget(self.edit_file_associations)
        associations_widget.setLayout(layout_file)
        self.create_tab(_('General'), general_widget)
        self.create_tab(_('File associations'), associations_widget)
        file_associations.sig_data_changed.connect(self.update_associations)

    def update_associations(self, data):
        if False:
            return 10
        '\n        Update the content of the text edit used to store the config data.\n        '
        textedit = self.edit_file_associations.textbox
        textedit.setPlainText(str(data))

    def reset_to_default(self):
        if False:
            return 10
        'Reset the filter settings to default.'
        self.set_option('name_filters', NAME_FILTERS)
        textedit = self.edit_filename_filters.textbox
        textedit.setPlainText(', '.join(NAME_FILTERS))