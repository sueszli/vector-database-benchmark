from persepolis.scripts.useful_tools import determineConfigFolder
from persepolis.gui.log_window_ui import LogWindow_Ui
from persepolis.scripts import osCommands
import os
try:
    from PySide6.QtCore import Qt, QPoint, QSize
    from PySide6.QtGui import QIcon
    from PySide6 import QtWidgets
except:
    from PyQt5.QtCore import Qt, QPoint, QSize
    from PyQt5.QtGui import QIcon
    from PyQt5 import QtWidgets
config_folder = determineConfigFolder()

class LogWindow(LogWindow_Ui):

    def __init__(self, persepolis_setting):
        if False:
            i = 10
            return i + 15
        super().__init__(persepolis_setting)
        self.persepolis_setting = persepolis_setting
        self.copy_log_pushButton.setEnabled(False)
        self.log_file = os.path.join(str(config_folder), 'persepolisdm.log')
        f = open(self.log_file, 'r')
        f_lines = f.readlines()
        f.close()
        self.text = 'Log File:\n'
        for line in f_lines:
            self.text = self.text + str(line) + '\n'
        self.text_edit.insertPlainText(self.text)
        self.text_edit.copyAvailable.connect(self.copyAvailableSignalHandler)
        self.copy_log_pushButton.clicked.connect(self.copyPushButtonPressed)
        self.report_pushButton.clicked.connect(self.reportPushButtonPressed)
        self.close_pushButton.clicked.connect(self.closePushButtonPressed)
        self.refresh_log_pushButton.clicked.connect(self.refreshLogPushButtonPressed)
        self.clear_log_pushButton.clicked.connect(self.clearLogPushButtonPressed)
        size = self.persepolis_setting.value('LogWindow/size', QSize(720, 300))
        position = self.persepolis_setting.value('LogWindow/position', QPoint(300, 300))
        self.resize(size)
        self.move(position)
        self.minimum_height = self.height()

    def clearLogPushButtonPressed(self, button):
        if False:
            for i in range(10):
                print('nop')
        f = open(self.log_file, 'w')
        f.close()
        self.text = 'Log File:\n'
        self.text_edit.clear()
        self.text_edit.insertPlainText(self.text)

    def reportPushButtonPressed(self, button):
        if False:
            while True:
                i = 10
        osCommands.xdgOpen('https://github.com/persepolisdm/persepolis/issues')

    def closePushButtonPressed(self, button):
        if False:
            i = 10
            return i + 15
        self.close()

    def copyAvailableSignalHandler(self, signal):
        if False:
            while True:
                i = 10
        if signal:
            self.copy_log_pushButton.setEnabled(True)
        else:
            self.copy_log_pushButton.setEnabled(False)

    def copyPushButtonPressed(self, button):
        if False:
            print('Hello World!')
        self.text_edit.copy()

    def refreshLogPushButtonPressed(self, button):
        if False:
            for i in range(10):
                print('nop')
        f = open(self.log_file, 'r')
        f_lines = f.readlines()
        f.close()
        self.text = 'Log File:\n'
        for line in f_lines:
            self.text = self.text + str(line) + '\n'
        self.text_edit.clear()
        self.text_edit.insertPlainText(self.text)

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if False:
            i = 10
            return i + 15
        self.layout().setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.setMinimumSize(QSize(self.width(), self.minimum_height))
        self.resize(QSize(self.width(), self.minimum_height))
        self.persepolis_setting.setValue('LogWindow/size', self.size())
        self.persepolis_setting.setValue('LogWindow/position', self.pos())
        self.persepolis_setting.sync()
        event.accept()

    def changeIcon(self, icons):
        if False:
            i = 10
            return i + 15
        icons = ':/' + str(icons) + '/'
        self.close_pushButton.setIcon(QIcon(icons + 'remove'))
        self.copy_log_pushButton.setIcon(QIcon(icons + 'clipboard'))
        self.report_pushButton.setIcon(QIcon(icons + 'about'))
        self.refresh_log_pushButton.setIcon(QIcon(icons + 'refresh'))