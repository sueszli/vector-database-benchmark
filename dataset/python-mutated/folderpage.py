import subprocess
from PyQt5.QtWidgets import QMessageBox
import autokey.configmanager.configmanager_constants as cm_constants
import autokey.qtui.common as ui_common
from autokey.model.folder import Folder
logger = __import__('autokey.logger').logger.get_logger(__name__)
PROBLEM_MSG_PRIMARY = 'Some problems were found'
PROBLEM_MSG_SECONDARY = '{}\n\nYour changes have not been saved.'

class FolderPage(*ui_common.inherits_from_ui_file_with_name('folderpage')):

    def __init__(self):
        if False:
            print('Hello World!')
        super(FolderPage, self).__init__()
        self.setupUi(self)
        self.current_folder = None

    def load(self, folder: Folder):
        if False:
            i = 10
            return i + 15
        self.current_folder = folder
        self.showInTrayCheckbox.setChecked(folder.show_in_tray_menu)
        self.settingsWidget.load(folder)
        if self.is_new_item():
            self.urlLabel.setEnabled(False)
            self.urlLabel.setText('(Unsaved)')
        else:
            ui_common.set_url_label(self.urlLabel, self.current_folder.path)

    def save(self):
        if False:
            print('Hello World!')
        self.current_folder.show_in_tray_menu = self.showInTrayCheckbox.isChecked()
        self.settingsWidget.save()
        self.current_folder.persist()
        ui_common.set_url_label(self.urlLabel, self.current_folder.path)
        return not self.current_folder.path.startswith(cm_constants.CONFIG_DEFAULT_FOLDER)

    def get_current_item(self):
        if False:
            return 10
        'Returns the currently held item.'
        return self.current_folder

    def set_item_title(self, title: str):
        if False:
            print('Hello World!')
        self.current_folder.title = title

    def rebuild_item_path(self):
        if False:
            return 10
        self.current_folder.rebuild_path()

    def is_new_item(self):
        if False:
            for i in range(10):
                print('nop')
        return self.current_folder.path is None

    def reset(self):
        if False:
            while True:
                i = 10
        self.load(self.current_folder)

    def validate(self):
        if False:
            print('Hello World!')
        errors = self.settingsWidget.validate()
        if errors:
            msg = PROBLEM_MSG_SECONDARY.format('\n'.join([str(e) for e in errors]))
            QMessageBox.critical(self.window(), PROBLEM_MSG_PRIMARY, msg)
        return not bool(errors)

    def set_dirty(self):
        if False:
            print('Hello World!')
        self.window().set_dirty()

    def on_showInTrayCheckbox_stateChanged(self, state: bool):
        if False:
            i = 10
            return i + 15
        self.set_dirty()

    @staticmethod
    def on_urlLabel_leftClickedUrl(url: str=None):
        if False:
            return 10
        if url:
            subprocess.Popen(['/usr/bin/xdg-open', url])