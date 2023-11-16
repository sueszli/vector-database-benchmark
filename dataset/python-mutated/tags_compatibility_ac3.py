from picard.config import BoolOption, get_config
from picard.ui.options import OptionsPage, register_options_page
from picard.ui.ui_options_tags_compatibility_ac3 import Ui_TagsCompatibilityOptionsPage

class TagsCompatibilityAC3OptionsPage(OptionsPage):
    NAME = 'tags_compatibility_ac3'
    TITLE = N_('AC3')
    PARENT = 'tags'
    SORT_ORDER = 50
    ACTIVE = True
    HELP_URL = '/config/options_tags_compatibility_ac3.html'
    options = [BoolOption('setting', 'ac3_save_ape', True), BoolOption('setting', 'remove_ape_from_ac3', False)]

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.ui = Ui_TagsCompatibilityOptionsPage()
        self.ui.setupUi(self)
        self.ui.ac3_no_tags.toggled.connect(self.ui.remove_ape_from_ac3.setEnabled)

    def load(self):
        if False:
            while True:
                i = 10
        config = get_config()
        if config.setting['ac3_save_ape']:
            self.ui.ac3_save_ape.setChecked(True)
        else:
            self.ui.ac3_no_tags.setChecked(True)
        self.ui.remove_ape_from_ac3.setChecked(config.setting['remove_ape_from_ac3'])
        self.ui.remove_ape_from_ac3.setEnabled(not config.setting['ac3_save_ape'])

    def save(self):
        if False:
            return 10
        config = get_config()
        config.setting['ac3_save_ape'] = self.ui.ac3_save_ape.isChecked()
        config.setting['remove_ape_from_ac3'] = self.ui.remove_ape_from_ac3.isChecked()
register_options_page(TagsCompatibilityAC3OptionsPage)