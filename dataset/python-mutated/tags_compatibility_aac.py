from picard.config import BoolOption, get_config
from picard.ui.options import OptionsPage, register_options_page
from picard.ui.ui_options_tags_compatibility_aac import Ui_TagsCompatibilityOptionsPage

class TagsCompatibilityAACOptionsPage(OptionsPage):
    NAME = 'tags_compatibility_aac'
    TITLE = N_('AAC')
    PARENT = 'tags'
    SORT_ORDER = 40
    ACTIVE = True
    HELP_URL = '/config/options_tags_compatibility_aac.html'
    options = [BoolOption('setting', 'aac_save_ape', True), BoolOption('setting', 'remove_ape_from_aac', False)]

    def __init__(self, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.ui = Ui_TagsCompatibilityOptionsPage()
        self.ui.setupUi(self)
        self.ui.aac_no_tags.toggled.connect(self.ui.remove_ape_from_aac.setEnabled)

    def load(self):
        if False:
            return 10
        config = get_config()
        if config.setting['aac_save_ape']:
            self.ui.aac_save_ape.setChecked(True)
        else:
            self.ui.aac_no_tags.setChecked(True)
        self.ui.remove_ape_from_aac.setChecked(config.setting['remove_ape_from_aac'])
        self.ui.remove_ape_from_aac.setEnabled(not config.setting['aac_save_ape'])

    def save(self):
        if False:
            print('Hello World!')
        config = get_config()
        config.setting['aac_save_ape'] = self.ui.aac_save_ape.isChecked()
        config.setting['remove_ape_from_aac'] = self.ui.remove_ape_from_aac.isChecked()
register_options_page(TagsCompatibilityAACOptionsPage)