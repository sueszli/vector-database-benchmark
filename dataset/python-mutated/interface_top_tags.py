from picard.config import ListOption, get_config
from picard.ui.options import OptionsPage, register_options_page
from picard.ui.ui_options_interface_top_tags import Ui_InterfaceTopTagsOptionsPage

class InterfaceTopTagsOptionsPage(OptionsPage):
    NAME = 'interface_top_tags'
    TITLE = N_('Top Tags')
    PARENT = 'interface'
    SORT_ORDER = 30
    ACTIVE = True
    HELP_URL = '/config/options_interface_top_tags.html'
    options = [ListOption('setting', 'metadatabox_top_tags', ['title', 'artist', 'album', 'tracknumber', '~length', 'date'])]

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.ui = Ui_InterfaceTopTagsOptionsPage()
        self.ui.setupUi(self)

    def load(self):
        if False:
            return 10
        config = get_config()
        tags = config.setting['metadatabox_top_tags']
        self.ui.top_tags_list.update(tags)

    def save(self):
        if False:
            while True:
                i = 10
        config = get_config()
        tags = list(self.ui.top_tags_list.tags)
        if tags != config.setting['metadatabox_top_tags']:
            config.setting['metadatabox_top_tags'] = tags
            self.tagger.window.metadata_box.update()

    def restore_defaults(self):
        if False:
            print('Hello World!')
        self.ui.top_tags_list.clear()
        super().restore_defaults()
register_options_page(InterfaceTopTagsOptionsPage)