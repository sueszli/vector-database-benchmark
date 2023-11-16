try:
    from PySide6.QtCore import Qt, QSize, QPoint, QThread, QTranslator, QCoreApplication, QLocale
    from PySide6.QtWidgets import QLineEdit, QWidget, QSizePolicy, QInputDialog
    from PySide6.QtGui import QIcon
except:
    from PyQt5.QtCore import Qt, QSize, QPoint, QThread, QTranslator, QCoreApplication, QLocale
    from PyQt5.QtWidgets import QLineEdit, QWidget, QSizePolicy, QInputDialog
    from PyQt5.QtGui import QIcon
from persepolis.constants import OS
from persepolis.gui.video_finder_progress_ui import VideoFinderProgressWindow_Ui
from persepolis.scripts.shutdown import shutDown
from persepolis.scripts.bubble import notifySend
from persepolis.scripts import download
import subprocess
import platform
import time
os_type = platform.system()

class ShutDownThread(QThread):

    def __init__(self, parent, category, password=None):
        if False:
            print('Hello World!')
        QThread.__init__(self)
        self.category = category
        self.password = password
        self.parent = parent

    def run(self):
        if False:
            print('Hello World!')
        shutDown(self.parent, category=self.category, password=self.password)

class VideoFinderProgressWindow(VideoFinderProgressWindow_Ui):

    def __init__(self, parent, gid_list, persepolis_setting):
        if False:
            return 10
        super().__init__(persepolis_setting)
        self.persepolis_setting = persepolis_setting
        self.parent = parent
        self.gid_list = gid_list
        self.gid = gid_list[0]
        self.video_finder_plus_gid = 'video_finder_' + str(gid_list[0])
        self.resume_pushButton.clicked.connect(self.resumePushButtonPressed)
        self.stop_pushButton.clicked.connect(self.stopPushButtonPressed)
        self.pause_pushButton.clicked.connect(self.pausePushButtonPressed)
        self.download_progressBar.setValue(0)
        self.limit_pushButton.clicked.connect(self.limitPushButtonPressed)
        self.limit_frame.setEnabled(False)
        self.limit_checkBox.toggled.connect(self.limitCheckBoxToggled)
        self.after_frame.setEnabled(False)
        self.after_checkBox.toggled.connect(self.afterCheckBoxToggled)
        self.after_pushButton.clicked.connect(self.afterPushButtonPressed)
        locale = str(self.persepolis_setting.value('settings/locale'))
        QLocale.setDefault(QLocale(locale))
        self.translator = QTranslator()
        if self.translator.load(':/translations/locales/ui_' + locale, 'ts'):
            QCoreApplication.installTranslator(self.translator)
        add_link_dictionary = self.parent.persepolis_db.searchGidInAddLinkTable(gid_list[0])
        limit = str(add_link_dictionary['limit_value'])
        if limit != '0':
            limit_number = limit[:-1]
            limit_unit = limit[-1]
            self.limit_spinBox.setValue(float(limit_number))
            if limit_unit == 'K':
                self.after_comboBox.setCurrentIndex(0)
            else:
                self.after_comboBox.setCurrentIndex(1)
            self.limit_checkBox.setChecked(True)
        self.after_comboBox.currentIndexChanged.connect(self.afterComboBoxChanged)
        self.limit_comboBox.currentIndexChanged.connect(self.limitComboBoxChanged)
        self.limit_spinBox.valueChanged.connect(self.limitComboBoxChanged)
        size = self.persepolis_setting.value('ProgressWindow/size', QSize(595, 274))
        position = self.persepolis_setting.value('ProgressWindow/position', QPoint(300, 300))
        self.resize(size)
        self.move(position)

    def keyPressEvent(self, event):
        if False:
            return 10
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if False:
            return 10
        self.persepolis_setting.setValue('ProgressWindow/size', self.size())
        self.persepolis_setting.setValue('ProgressWindow/position', self.pos())
        self.persepolis_setting.sync()
        self.hide()

    def resumePushButtonPressed(self, button):
        if False:
            print('Hello World!')
        if self.status == 'paused':
            answer = download.downloadUnpause(self.gid)
            if not answer:
                version_answer = download.aria2Version()
                if version_answer == 'did not respond':
                    self.parent.aria2Disconnected()
                    notifySend(QCoreApplication.translate('progress_src_ui_tr', 'Aria2 disconnected!'), QCoreApplication.translate('progress_src_ui_tr', 'Persepolis is trying to connect! be patient!'), 10000, 'warning', parent=self.parent)
                else:
                    notifySend(QCoreApplication.translate('progress_src_ui_tr', 'Aria2 did not respond!'), QCoreApplication.translate('progress_src_ui_tr', 'Please try again.'), 10000, 'warning', parent=self.parent)

    def pausePushButtonPressed(self, button):
        if False:
            print('Hello World!')
        if self.status == 'downloading':
            answer = download.downloadPause(self.gid)
            if not answer:
                version_answer = download.aria2Version()
                if version_answer == 'did not respond':
                    self.parent.aria2Disconnected()
                    download.downloadStop(self.gid, self.parent)
                    notifySend('Aria2 disconnected!', 'Persepolis is trying to connect! be patient!', 10000, 'warning', parent=self.parent)
                else:
                    notifySend(QCoreApplication.translate('progress_src_ui_tr', 'Aria2 did not respond!'), QCoreApplication.translate('progress_src_ui_tr', 'Try again!'), 10000, 'critical', parent=self.parent)

    def stopPushButtonPressed(self, button):
        if False:
            while True:
                i = 10
        dictionary = {'category': self.video_finder_plus_gid, 'shutdown': 'canceled'}
        self.parent.temp_db.updateQueueTable(dictionary)
        answer = download.downloadStop(self.gid, self.parent)
        if answer == 'None':
            version_answer = download.aria2Version()
            if version_answer == 'did not respond':
                self.parent.aria2Disconnected()
                notifySend(QCoreApplication.translate('progress_src_ui_tr', 'Aria2 disconnected!'), QCoreApplication.translate('progress_src_ui_tr', 'Persepolis is trying to connect! be patient!'), 10000, 'warning', parent=self.parent)

    def limitCheckBoxToggled(self, checkBoxes):
        if False:
            print('Hello World!')
        if self.limit_checkBox.isChecked() == True:
            self.limit_frame.setEnabled(True)
            self.limit_pushButton.setEnabled(True)
        else:
            self.limit_frame.setEnabled(False)
            for i in [0, 1]:
                gid = self.gid_list[i]
                dictionary = self.parent.persepolis_db.searchGidInDownloadTable(gid)
                status = dictionary['status']
                if status != 'scheduled':
                    download.limitSpeed(gid, '0')
                else:
                    add_link_dictionary = {'gid': gid, 'limit_value': '0'}
                    self.parent.persepolis_db.updateAddLinkTable([add_link_dictionary])

    def limitComboBoxChanged(self, connect):
        if False:
            return 10
        self.limit_pushButton.setEnabled(True)

    def afterComboBoxChanged(self, connect):
        if False:
            print('Hello World!')
        self.after_pushButton.setEnabled(True)

    def afterCheckBoxToggled(self, checkBoxes):
        if False:
            i = 10
            return i + 15
        if self.after_checkBox.isChecked():
            self.after_frame.setEnabled(True)
        else:
            dictionary = {'category': self.video_finder_plus_gid, 'shutdown': 'canceled'}
            self.parent.temp_db.updateQueueTable(dictionary)

    def afterPushButtonPressed(self, button):
        if False:
            while True:
                i = 10
        self.after_pushButton.setEnabled(False)
        if os_type != OS.WINDOWS:
            (passwd, ok) = QInputDialog.getText(self, 'PassWord', 'Please enter root password:', QLineEdit.Password)
            if ok:
                pipe = subprocess.Popen(['sudo', '-S', 'echo', 'hello'], stdout=subprocess.DEVNULL, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False)
                pipe.communicate(passwd.encode())
                answer = pipe.wait()
                while answer != 0:
                    (passwd, ok) = QInputDialog.getText(self, 'PassWord', 'Wrong Password!\nPlease try again.', QLineEdit.Password)
                    if ok:
                        pipe = subprocess.Popen(['sudo', '-S', 'echo', 'hello'], stdout=subprocess.DEVNULL, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL, shell=False)
                        pipe.communicate(passwd.encode())
                        answer = pipe.wait()
                    else:
                        ok = False
                        break
                if ok != False:
                    shutdown_enable = ShutDownThread(self.parent, self.video_finder_plus_gid, passwd)
                    self.parent.threadPool.append(shutdown_enable)
                    self.parent.threadPool[len(self.parent.threadPool) - 1].start()
                else:
                    self.after_checkBox.setChecked(False)
            else:
                self.after_checkBox.setChecked(False)
        else:
            for gid in self.gid_list:
                shutdown_enable = ShutDownThread(self.parent, self.video_finder_plus_gid)
                self.parent.threadPool.append(shutdown_enable)
                self.parent.threadPool[len(self.parent.threadPool) - 1].start()

    def limitPushButtonPressed(self, button):
        if False:
            i = 10
            return i + 15
        self.limit_pushButton.setEnabled(False)
        if self.limit_comboBox.currentText() == 'KiB/s':
            limit_value = str(self.limit_spinBox.value()) + str('K')
        else:
            limit_value = str(self.limit_spinBox.value()) + str('M')
        for i in [0, 1]:
            gid = self.gid_list[i]
            dictionary = self.parent.persepolis_db.searchGidInDownloadTable(gid)
            status = dictionary['status']
            if status != 'scheduled':
                download.limitSpeed(self.gid, limit_value)
            else:
                add_link_dictionary = {'gid': gid, 'limit_value': limit_value}
                self.parent.persepolis_db.updateAddLinkTable([add_link_dictionary])

    def changeIcon(self, icons):
        if False:
            print('Hello World!')
        icons = ':/' + str(icons) + '/'
        self.resume_pushButton.setIcon(QIcon(icons + 'play'))
        self.pause_pushButton.setIcon(QIcon(icons + 'pause'))
        self.stop_pushButton.setIcon(QIcon(icons + 'stop'))