"""
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
try:
    from PySide6.QtWidgets import QHBoxLayout, QApplication, QFileDialog, QCheckBox, QLineEdit, QPushButton
    from PySide6.QtCore import Qt, QPoint, QSize, QDir, QThread, Signal
    from PySide6.QtGui import QIcon
except:
    from PyQt5.QtWidgets import QHBoxLayout, QApplication, QFileDialog, QCheckBox, QLineEdit, QPushButton
    from PyQt5.QtCore import Qt, QPoint, QSize, QDir, QThread
    from PyQt5.QtGui import QIcon
    from PyQt5.QtCore import pyqtSignal as Signal
from persepolis.gui.addlink_ui import AddLinkWindow_Ui
from persepolis.scripts.check_proxy import getProxy
from persepolis.scripts import spider
from persepolis.scripts import logger
from functools import partial
import os

class AddLinkSpiderThread(QThread):
    ADDLINKSPIDERSIGNAL = Signal(dict)

    def __init__(self, add_link_dictionary):
        if False:
            while True:
                i = 10
        QThread.__init__(self)
        self.add_link_dictionary = add_link_dictionary

    def run(self):
        if False:
            for i in range(10):
                print('nop')
        try:
            (file_name, file_size) = spider.addLinkSpider(self.add_link_dictionary)
            spider_dict = {'file_size': file_size, 'file_name': file_name}
            self.ADDLINKSPIDERSIGNAL.emit(spider_dict)
            if not file_name:
                logger.sendToLog("Spider couldn't find file name", 'ERROR')
            if not file_size:
                logger.sendToLog("Spider couldn't find file size", 'ERROR')
        except Exception as e:
            logger.sendToLog("Spider couldn't find download information", 'ERROR')
            logger.sendToLog(str(e), 'ERROR')

class AddLinkWindow(AddLinkWindow_Ui):

    def __init__(self, parent, callback, persepolis_setting, plugin_add_link_dictionary={}):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(persepolis_setting)
        self.callback = callback
        self.plugin_add_link_dictionary = plugin_add_link_dictionary
        self.persepolis_setting = persepolis_setting
        self.parent = parent
        connections = int(self.persepolis_setting.value('settings/connections'))
        self.connections_spinBox.setValue(connections)
        download_path = str(self.persepolis_setting.value('settings/download_path'))
        self.download_folder_lineEdit.setText(download_path)
        self.download_folder_lineEdit.setEnabled(False)
        self.ok_pushButton.setEnabled(False)
        self.download_later_pushButton.setEnabled(False)
        self.link_lineEdit.textChanged.connect(self.linkLineChanged)
        if 'link' in self.plugin_add_link_dictionary.keys():
            self.link_lineEdit.setText(str(self.plugin_add_link_dictionary['link']))
        else:
            clipboard = QApplication.clipboard()
            text = clipboard.text()
            if 'tp:/' in text[2:6] or 'tps:/' in text[2:7]:
                self.link_lineEdit.setText(str(text))
        self.detect_proxy_pushButton.clicked.connect(self.detectProxy)
        settings_ip = self.persepolis_setting.value('add_link_initialization/ip', None)
        if settings_ip:
            self.ip_lineEdit.setText(str(settings_ip))
        settings_proxy_user = self.persepolis_setting.value('add_link_initialization/proxy_user', None)
        if settings_proxy_user:
            self.proxy_user_lineEdit.setText(str(settings_proxy_user))
        settings_port = self.persepolis_setting.value('add_link_initialization/port', 0)
        self.port_spinBox.setValue(int(int(settings_port)))
        settings_download_user = self.persepolis_setting.value('add_link_initialization/download_user', None)
        if settings_download_user:
            self.download_user_lineEdit.setText(str(settings_download_user))
        categories_list = self.parent.persepolis_db.categoriesList()
        for queue in categories_list:
            if queue != 'All Downloads':
                self.add_queue_comboBox.addItem(queue)
        self.add_queue_comboBox.setCurrentIndex(0)
        self.add_queue_comboBox.currentIndexChanged.connect(self.queueChanged)
        self.folder_pushButton.clicked.connect(self.changeFolder)
        self.cancel_pushButton.clicked.connect(self.close)
        self.ok_pushButton.clicked.connect(partial(self.okButtonPressed, download_later=False))
        self.download_later_pushButton.clicked.connect(partial(self.okButtonPressed, download_later=True))
        self.proxy_frame.setEnabled(False)
        self.proxy_checkBox.toggled.connect(self.proxyFrame)
        self.download_frame.setEnabled(False)
        self.download_checkBox.toggled.connect(self.downloadFrame)
        self.limit_frame.setEnabled(False)
        self.limit_checkBox.toggled.connect(self.limitFrame)
        self.start_frame.setEnabled(False)
        self.start_checkBox.toggled.connect(self.startFrame)
        self.end_frame.setEnabled(False)
        self.end_checkBox.toggled.connect(self.endFrame)
        self.change_name_lineEdit.setEnabled(False)
        self.change_name_checkBox.toggled.connect(self.changeName)
        self.ok_pushButton.setFocus()
        if 'out' in self.plugin_add_link_dictionary.keys():
            if self.plugin_add_link_dictionary['out']:
                self.change_name_lineEdit.setText(str(self.plugin_add_link_dictionary['out']))
                self.change_name_checkBox.setChecked(True)
        if 'referer' in self.plugin_add_link_dictionary:
            self.referer_lineEdit.setText(str(self.plugin_add_link_dictionary['referer']))
        if 'header' in self.plugin_add_link_dictionary:
            if str(self.plugin_add_link_dictionary['header']) != 'None':
                self.header_lineEdit.setText(str(self.plugin_add_link_dictionary['header']))
        if 'user_agent' in self.plugin_add_link_dictionary:
            self.user_agent_lineEdit.setText(str(self.plugin_add_link_dictionary['user_agent']))
        if 'load_cookies' in self.plugin_add_link_dictionary:
            self.load_cookies_lineEdit.setText(self.plugin_add_link_dictionary['load_cookies'])
        size = self.persepolis_setting.value('AddLinkWindow/size', QSize(520, 425))
        position = self.persepolis_setting.value('AddLinkWindow/position', QPoint(300, 300))
        self.resize(size)
        self.move(position)

    def detectProxy(self, button):
        if False:
            i = 10
            return i + 15
        system_proxy_dict = getProxy()
        enable_proxy_frame = False
        if 'http_proxy_ip' in system_proxy_dict.keys():
            self.ip_lineEdit.setText(str(system_proxy_dict['http_proxy_ip']))
            enable_proxy_frame = True
        if 'http_proxy_port' in system_proxy_dict.keys():
            self.port_spinBox.setValue(int(system_proxy_dict['http_proxy_port']))
            enable_proxy_frame = True
        if enable_proxy_frame:
            self.proxy_checkBox.setChecked(True)
            self.detect_proxy_label.setText('')
        else:
            self.proxy_checkBox.setChecked(False)
            self.detect_proxy_label.setText('No proxy detected!')

    def proxyFrame(self, checkBox):
        if False:
            i = 10
            return i + 15
        if self.proxy_checkBox.isChecked() == True:
            self.proxy_frame.setEnabled(True)
        else:
            self.proxy_frame.setEnabled(False)

    def downloadFrame(self, checkBox):
        if False:
            i = 10
            return i + 15
        if self.download_checkBox.isChecked() == True:
            self.download_frame.setEnabled(True)
        else:
            self.download_frame.setEnabled(False)

    def limitFrame(self, checkBox):
        if False:
            print('Hello World!')
        if self.limit_checkBox.isChecked() == True:
            self.limit_frame.setEnabled(True)
        else:
            self.limit_frame.setEnabled(False)

    def startFrame(self, checkBox):
        if False:
            while True:
                i = 10
        if self.start_checkBox.isChecked() == True:
            self.start_frame.setEnabled(True)
        else:
            self.start_frame.setEnabled(False)

    def endFrame(self, checkBox):
        if False:
            return 10
        if self.end_checkBox.isChecked() == True:
            self.end_frame.setEnabled(True)
        else:
            self.end_frame.setEnabled(False)

    def changeFolder(self, button):
        if False:
            return 10
        download_path = self.download_folder_lineEdit.text()
        fname = QFileDialog.getExistingDirectory(self, 'Select a directory', download_path)
        if fname:
            fname = QDir.toNativeSeparators(fname)
        if os.path.isdir(fname):
            self.download_folder_lineEdit.setText(fname)

    def linkLineChanged(self, lineEdit):
        if False:
            i = 10
            return i + 15
        if str(self.link_lineEdit.text()) == '':
            self.ok_pushButton.setEnabled(False)
            self.download_later_pushButton.setEnabled(False)
        else:
            dict = {'link': str(self.link_lineEdit.text())}
            new_spider = AddLinkSpiderThread(dict)
            self.parent.threadPool.append(new_spider)
            self.parent.threadPool[len(self.parent.threadPool) - 1].start()
            self.parent.threadPool[len(self.parent.threadPool) - 1].ADDLINKSPIDERSIGNAL.connect(partial(self.parent.addLinkSpiderCallBack, child=self))
            self.ok_pushButton.setEnabled(True)
            self.download_later_pushButton.setEnabled(True)

    def changeName(self, checkBoxes):
        if False:
            print('Hello World!')
        if self.change_name_checkBox.isChecked() == True:
            self.change_name_lineEdit.setEnabled(True)
        else:
            self.change_name_lineEdit.setEnabled(False)

    def queueChanged(self, combo):
        if False:
            for i in range(10):
                print('nop')
        if self.add_queue_comboBox.currentIndex() != 0:
            self.start_checkBox.setCheckState(Qt.Unchecked)
            self.start_checkBox.setEnabled(False)
            self.end_checkBox.setCheckState(Qt.Unchecked)
            self.end_checkBox.setEnabled(False)
        else:
            self.start_checkBox.setEnabled(True)
            self.end_checkBox.setEnabled(True)

    def okButtonPressed(self, download_later, button=None):
        if False:
            while True:
                i = 10
        self.persepolis_setting.setValue('add_link_initialization/ip', self.ip_lineEdit.text())
        self.persepolis_setting.setValue('add_link_initialization/port', self.port_spinBox.value())
        self.persepolis_setting.setValue('add_link_initialization/proxy_user', self.proxy_user_lineEdit.text())
        self.persepolis_setting.setValue('add_link_initialization/download_user', self.download_user_lineEdit.text())
        if self.folder_checkBox.isChecked() == True:
            self.persepolis_setting.setValue('settings/download_path', self.download_folder_lineEdit.text())
        if not self.proxy_checkBox.isChecked():
            ip = None
            port = None
            proxy_user = None
            proxy_passwd = None
        else:
            ip = self.ip_lineEdit.text()
            if not ip:
                ip = None
            port = self.port_spinBox.value()
            if not port:
                port = None
            proxy_user = self.proxy_user_lineEdit.text()
            if not proxy_user:
                proxy_user = None
            proxy_passwd = self.proxy_pass_lineEdit.text()
            if not proxy_passwd:
                proxy_passwd = None
        if not self.download_checkBox.isChecked():
            download_user = None
            download_passwd = None
        else:
            download_user = self.download_user_lineEdit.text()
            if not download_user:
                download_user = None
            download_passwd = self.download_pass_lineEdit.text()
            if not download_passwd:
                download_passwd = None
        if not self.limit_checkBox.isChecked():
            limit = 0
        elif self.limit_comboBox.currentText() == 'KiB/s':
            limit = str(self.limit_spinBox.value()) + str('K')
        else:
            limit = str(self.limit_spinBox.value()) + str('M')
        if not self.start_checkBox.isChecked():
            start_time = None
        else:
            start_time = self.start_time_qDataTimeEdit.text()
        if not self.end_checkBox.isChecked():
            end_time = None
        else:
            end_time = self.end_time_qDateTimeEdit.text()
        if self.change_name_checkBox.isChecked():
            out = str(self.change_name_lineEdit.text())
            self.plugin_add_link_dictionary['out'] = out
        else:
            out = None
        link = self.link_lineEdit.text()
        connections = self.connections_spinBox.value()
        download_path = self.download_folder_lineEdit.text()
        if self.referer_lineEdit.text() != '':
            referer = self.referer_lineEdit.text()
        else:
            referer = None
        if self.header_lineEdit.text() != '':
            header = self.header_lineEdit.text()
        else:
            header = None
        if self.user_agent_lineEdit.text() != '':
            user_agent = self.user_agent_lineEdit.text()
        else:
            user_agent = None
        if self.load_cookies_lineEdit.text() != '':
            load_cookies = self.load_cookies_lineEdit.text()
        else:
            load_cookies = None
        self.add_link_dictionary = {'referer': referer, 'header': header, 'user_agent': user_agent, 'load_cookies': load_cookies, 'out': out, 'start_time': start_time, 'end_time': end_time, 'link': link, 'ip': ip, 'port': port, 'proxy_user': proxy_user, 'proxy_passwd': proxy_passwd, 'download_user': download_user, 'download_passwd': download_passwd, 'connections': connections, 'limit_value': limit, 'download_path': download_path}
        category = str(self.add_queue_comboBox.currentText())
        del self.plugin_add_link_dictionary
        self.callback(self.add_link_dictionary, download_later, category)
        self.close()

    def keyPressEvent(self, event):
        if False:
            i = 10
            return i + 15
        if event.key() == Qt.Key_Escape:
            self.close()

    def closeEvent(self, event):
        if False:
            print('Hello World!')
        self.persepolis_setting.setValue('AddLinkWindow/size', self.size())
        self.persepolis_setting.setValue('AddLinkWindow/position', self.pos())
        self.persepolis_setting.sync()
        event.accept()

    def changeIcon(self, icons):
        if False:
            for i in range(10):
                print('nop')
        icons = ':/' + str(icons) + '/'
        self.folder_pushButton.setIcon(QIcon(icons + 'folder'))
        self.download_later_pushButton.setIcon(QIcon(icons + 'stop'))
        self.cancel_pushButton.setIcon(QIcon(icons + 'remove'))
        self.ok_pushButton.setIcon(QIcon(icons + 'ok'))