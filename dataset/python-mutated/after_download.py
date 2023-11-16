try:
    from PySide6.QtCore import Qt, QSize, QPoint, QTranslator, QCoreApplication, QLocale
    from PySide6.QtGui import QIcon
except:
    from PyQt5.QtCore import Qt, QSize, QPoint, QTranslator, QCoreApplication, QLocale
    from PyQt5.QtGui import QIcon
from persepolis.gui.after_download_ui import AfterDownloadWindow_Ui
from persepolis.scripts.play import playNotification
from persepolis.scripts import osCommands
import os

class AfterDownloadWindow(AfterDownloadWindow_Ui):

    def __init__(self, parent, dict, persepolis_setting):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(persepolis_setting)
        self.persepolis_setting = persepolis_setting
        self.dict = dict
        self.parent = parent
        locale = str(self.persepolis_setting.value('settings/locale'))
        QLocale.setDefault(QLocale(locale))
        self.translator = QTranslator()
        if self.translator.load(':/translations/locales/ui_' + locale, 'ts'):
            QCoreApplication.installTranslator(self.translator)
        self.open_pushButtun.clicked.connect(self.openFile)
        self.open_folder_pushButtun.clicked.connect(self.openFolder)
        self.ok_pushButton.clicked.connect(self.okButtonPressed)
        gid = self.dict['gid']
        self.add_link_dict = self.parent.persepolis_db.searchGidInAddLinkTable(gid)
        file_path = self.add_link_dict['download_path']
        self.save_as_lineEdit.setText(file_path)
        self.save_as_lineEdit.setToolTip(file_path)
        link = str(self.dict['link'])
        self.link_lineEdit.setText(link)
        self.link_lineEdit.setToolTip(link)
        window_title = str(self.dict['file_name'])
        file_name = QCoreApplication.translate('after_download_src_ui_tr', '<b>File name</b>: ') + window_title
        self.setWindowTitle(window_title)
        self.file_name_label.setText(file_name)
        size = QCoreApplication.translate('after_download_src_ui_tr', '<b>Size</b>: ') + str(self.dict['size'])
        self.size_label.setText(size)
        self.link_lineEdit.setEnabled(False)
        self.save_as_lineEdit.setEnabled(False)
        size = self.persepolis_setting.value('AfterDownloadWindow/size', QSize(570, 290))
        position = self.persepolis_setting.value('AfterDownloadWindow/position', QPoint(300, 300))
        self.resize(size)
        self.move(position)

    def openFile(self):
        if False:
            i = 10
            return i + 15
        file_path = self.add_link_dict['download_path']
        if os.path.isfile(file_path):
            osCommands.xdgOpen(file_path)
        self.close()

    def openFolder(self):
        if False:
            while True:
                i = 10
        download_path = self.add_link_dict['download_path']
        if os.path.isfile(download_path):
            osCommands.xdgOpen(download_path, 'folder', 'file')
        self.close()

    def okButtonPressed(self):
        if False:
            print('Hello World!')
        if self.dont_show_checkBox.isChecked():
            self.persepolis_setting.setValue('settings/after-dialog', 'no')
            self.persepolis_setting.sync()
        self.close()

    def keyPressEvent(self, event):
        if False:
            return 10
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.persepolis_setting.setValue('AfterDownloadWindow/size', self.size())
        self.persepolis_setting.setValue('AfterDownloadWindow/position', self.pos())
        self.persepolis_setting.sync()
        event.accept()

    def changeIcon(self, icons):
        if False:
            i = 10
            return i + 15
        icons = ':/' + str(icons) + '/'
        self.ok_pushButton.setIcon(QIcon(icons + 'ok'))
        self.open_folder_pushButtun.setIcon(QIcon(icons + 'folder'))
        self.open_pushButtun.setIcon(QIcon(icons + 'file'))