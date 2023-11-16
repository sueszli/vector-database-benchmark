from picard.config import BoolOption, IntOption, TextOption, get_config
from picard.ui.options import OptionsPage, register_options_page
from picard.ui.ui_options_ratings import Ui_RatingsOptionsPage

class RatingsOptionsPage(OptionsPage):
    NAME = 'ratings'
    TITLE = N_('Ratings')
    PARENT = 'metadata'
    SORT_ORDER = 20
    ACTIVE = True
    HELP_URL = '/config/options_ratings.html'
    options = [BoolOption('setting', 'enable_ratings', False), TextOption('setting', 'rating_user_email', 'users@musicbrainz.org'), BoolOption('setting', 'submit_ratings', True), IntOption('setting', 'rating_steps', 6)]

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self.ui = Ui_RatingsOptionsPage()
        self.ui.setupUi(self)

    def load(self):
        if False:
            print('Hello World!')
        config = get_config()
        self.ui.enable_ratings.setChecked(config.setting['enable_ratings'])
        self.ui.rating_user_email.setText(config.setting['rating_user_email'])
        self.ui.submit_ratings.setChecked(config.setting['submit_ratings'])

    def save(self):
        if False:
            while True:
                i = 10
        config = get_config()
        config.setting['enable_ratings'] = self.ui.enable_ratings.isChecked()
        config.setting['rating_user_email'] = self.ui.rating_user_email.text()
        config.setting['submit_ratings'] = self.ui.submit_ratings.isChecked()
register_options_page(RatingsOptionsPage)