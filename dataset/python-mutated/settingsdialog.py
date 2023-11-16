from typing import TYPE_CHECKING
from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QApplication, QWidget
from autokey.qtui import common
if TYPE_CHECKING:
    from autokey.qtapp import Application
logger = __import__('autokey.logger').logger.get_logger(__name__)

class SettingsDialog(*common.inherits_from_ui_file_with_name('settingsdialog')):

    def __init__(self, parent: QWidget=None):
        if False:
            while True:
                i = 10
        super(SettingsDialog, self).__init__(parent)
        self.setupUi(self)
        logger.info('Settings dialog window created.')

    @pyqtSlot()
    def on_show_general_settings_button_clicked(self):
        if False:
            return 10
        logger.debug('User views general settings')
        self.settings_pages.setCurrentWidget(self.general_settings_page)

    @pyqtSlot()
    def on_show_special_hotkeys_button_clicked(self):
        if False:
            return 10
        logger.debug('User views special hotkeys settings')
        self.settings_pages.setCurrentWidget(self.special_hotkeys_page)

    @pyqtSlot()
    def on_show_script_engine_button_clicked(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('User views script engine settings')
        self.settings_pages.setCurrentWidget(self.script_engine_page)

    def accept(self):
        if False:
            for i in range(10):
                print('nop')
        logger.info('User requested to save the settings.')
        app = QApplication.instance()
        self.general_settings_page.save()
        self.special_hotkeys_page.save()
        self.script_engine_page.save()
        app.configManager.config_altered(True)
        app.update_notifier_visibility()
        app.notifier.reset_tray_icon()
        super(SettingsDialog, self).accept()
        logger.debug('Save completed, dialog window hidden.')