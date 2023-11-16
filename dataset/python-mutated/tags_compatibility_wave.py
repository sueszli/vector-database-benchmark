from picard.config import BoolOption, TextOption, get_config
from picard.formats.wav import WAVFile
from picard.ui.options import OptionsPage, register_options_page
from picard.ui.ui_options_tags_compatibility_wave import Ui_TagsCompatibilityOptionsPage

class TagsCompatibilityWaveOptionsPage(OptionsPage):
    NAME = 'tags_compatibility_wave'
    TITLE = N_('WAVE')
    PARENT = 'tags'
    SORT_ORDER = 60
    ACTIVE = True
    HELP_URL = '/config/options_tags_compatibility_wave.html'
    options = [BoolOption('setting', 'write_wave_riff_info', True), BoolOption('setting', 'remove_wave_riff_info', False), TextOption('setting', 'wave_riff_info_encoding', 'windows-1252')]

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.ui = Ui_TagsCompatibilityOptionsPage()
        self.ui.setupUi(self)

    def load(self):
        if False:
            print('Hello World!')
        config = get_config()
        self.ui.write_wave_riff_info.setChecked(config.setting['write_wave_riff_info'])
        self.ui.remove_wave_riff_info.setChecked(config.setting['remove_wave_riff_info'])
        if config.setting['wave_riff_info_encoding'] == 'utf-8':
            self.ui.wave_riff_info_enc_utf8.setChecked(True)
        else:
            self.ui.wave_riff_info_enc_cp1252.setChecked(True)

    def save(self):
        if False:
            i = 10
            return i + 15
        config = get_config()
        config.setting['write_wave_riff_info'] = self.ui.write_wave_riff_info.isChecked()
        config.setting['remove_wave_riff_info'] = self.ui.remove_wave_riff_info.isChecked()
        if self.ui.wave_riff_info_enc_utf8.isChecked():
            config.setting['wave_riff_info_encoding'] = 'utf-8'
        else:
            config.setting['wave_riff_info_encoding'] = 'windows-1252'
if WAVFile.supports_tag('artist'):
    register_options_page(TagsCompatibilityWaveOptionsPage)