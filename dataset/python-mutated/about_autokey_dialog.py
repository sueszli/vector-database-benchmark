import sys
from PyQt5.QtWidgets import QWidget
from PyQt5.QtCore import QSize
import autokey.common
from autokey.qtui import common as ui_common

class AboutAutokeyDialog(*ui_common.inherits_from_ui_file_with_name('about_autokey_dialog')):

    def __init__(self, parent: QWidget=None):
        if False:
            print('Hello World!')
        super(AboutAutokeyDialog, self).__init__(parent)
        self.setupUi(self)
        icon = ui_common.load_icon(ui_common.AutoKeyIcon.AUTOKEY)
        pixmap = icon.pixmap(icon.actualSize(QSize(1024, 1024)))
        self.autokey_icon.setPixmap(pixmap)
        self.autokey_version_label.setText(autokey.common.VERSION)
        self.python_version_label.setText(sys.version.replace('\n', ' '))