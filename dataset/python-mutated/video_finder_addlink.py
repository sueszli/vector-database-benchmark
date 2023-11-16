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
    from PySide6.QtWidgets import QCheckBox, QPushButton, QTextEdit, QFrame, QLabel, QComboBox, QHBoxLayout, QApplication
    from PySide6.QtCore import Qt, QThread, Signal, QCoreApplication, QTranslator, QLocale
except:
    from PyQt5.QtWidgets import QCheckBox, QPushButton, QTextEdit, QFrame, QLabel, QComboBox, QHBoxLayout, QApplication
    from PyQt5.QtCore import Qt, QThread, QCoreApplication, QTranslator, QLocale
    from PyQt5.QtCore import pyqtSignal as Signal
from persepolis.scripts.useful_tools import determineConfigFolder
from persepolis.scripts.addlink import AddLinkWindow
from persepolis.scripts import logger, osCommands
from persepolis.scripts.spider import spider
from time import time, sleep
from functools import partial
from random import random
from copy import deepcopy
import youtube_dl
import re
import os
logger.sendToLog('youtube_dl version: ' + str(youtube_dl.version.__version__), 'INFO')
config_folder = determineConfigFolder()
persepolis_tmp = os.path.join(config_folder, 'persepolis_tmp')

class MediaListFetcherThread(QThread):
    RESULT = Signal(dict)
    cookies = '# HTTP cookie file.\n'

    def __init__(self, receiver_slot, video_dict, parent):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.RESULT.connect(receiver_slot)
        self.video_dict = video_dict
        self.cookie_path = os.path.join(persepolis_tmp, '.{}{}'.format(time(), random()))
        self.youtube_dl_options_dict = {'dump_single_json': True, 'quiet': True, 'noplaylist': True, 'no_warnings': True}
        self.youtube_dl_options_dict['cookies'] = str(self.cookie_path)
        if 'referer' in video_dict.keys() and video_dict['referer']:
            self.youtube_dl_options_dict['referer'] = str(video_dict['referer'])
        if 'user_agent' in video_dict.keys() and video_dict['user_agent']:
            self.youtube_dl_options_dict['user-agent'] = str(video_dict['user_agent'])
        if 'load_cookies' in video_dict.keys() and video_dict['load_cookies']:
            self.cookies = self.makeHttpCookie(video_dict['load_cookies'])
        if 'ip' in video_dict.keys() and video_dict['ip']:
            try:
                ip_port = 'http://{}:{}'.format(video_dict['ip'], video_dict['port'])
                if 'referer' in video_dict.keys() and video_dict['proxy_user']:
                    ip_port = 'http://{}:{}@{}'.format(video_dict['proxy_user'], video_dict['proxy_passwd'], ip_port)
                self.youtube_dl_options_dict['proxy'] = str(ip_port)
            except:
                pass
        if 'download_user' in video_dict.keys() and video_dict['download_user']:
            try:
                self.youtube_dl_options_dict['username'] = str(video_dict['download_user'])
                self.youtube_dl_options_dict['password'] = str(video_dict['download_passwd'])
            except:
                pass
        if 'link' in video_dict.keys() and video_dict['link']:
            self.youtube_link = str(video_dict['link'])

    def run(self):
        if False:
            while True:
                i = 10
        ret_val = {}
        try:
            cookie_file = open(self.cookie_path, 'w')
            cookie_file.write(self.cookies)
            cookie_file.close()
            ydl = youtube_dl.YoutubeDL(self.youtube_dl_options_dict)
            with ydl:
                result = ydl.extract_info(self.youtube_link, download=False)
            error = 'error'
            if result:
                ret_val = result
            else:
                ret_val = {'error': str(error)}
        except Exception as ex:
            ret_val = {'error': str(ex)}
        finally:
            try:
                osCommands.remove(self.cookie_path)
            except Exception as ex:
                logger.sendToLog(ex, 'ERROR')
        self.RESULT.emit(ret_val)

    def makeHttpCookie(self, raw_cookie, host_name='.youtube.com'):
        if False:
            return 10
        cookies = '# HTTP cookie file.\n'
        if raw_cookie:
            try:
                raw_cookies = re.split(';\\s*', str(raw_cookie))
                for c in raw_cookies:
                    (key, val) = c.split('=', 1)
                    cookies = cookies + '{}\tTRUE\t/\tFALSE\t{}\t{}\t{}\n'.format(host_name, int(time()) + 259200, key, val)
            except:
                pass
        return cookies

class FileSizeFetcherThread(QThread):
    FOUND = Signal(dict)

    def __init__(self, dictionary, thread_key):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.dictionary = dictionary
        self.key = thread_key

    def run(self):
        if False:
            while True:
                i = 10
        spider_file_size = spider(self.dictionary)[1]
        self.FOUND.emit({'thread_key': self.key, 'file_size': spider_file_size})

class VideoFinderAddLink(AddLinkWindow):
    running_thread = None
    threadPool = {}

    def __init__(self, parent, receiver_slot, settings, video_dict={}):
        if False:
            print('Hello World!')
        super().__init__(parent, receiver_slot, settings, video_dict)
        self.setWindowTitle(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Video Finder'))
        self.size_label.hide()
        self.no_audio_list = []
        self.no_video_list = []
        self.video_audio_list = []
        self.media_title = ''
        locale = str(self.persepolis_setting.value('settings/locale'))
        QLocale.setDefault(QLocale(locale))
        self.translator = QTranslator()
        if self.translator.load(':/translations/locales/ui_' + locale, 'ts'):
            QCoreApplication.installTranslator(self.translator)
        self.extension_label = QLabel(self.link_frame)
        self.change_name_horizontalLayout.addWidget(self.extension_label)
        self.url_submit_pushButtontton = QPushButton(self.link_frame)
        self.link_horizontalLayout.addWidget(self.url_submit_pushButtontton)
        self.status_box_textEdit = QTextEdit(self.link_frame)
        self.status_box_textEdit.setMaximumHeight(150)
        self.link_verticalLayout.addWidget(self.status_box_textEdit)
        select_format_horizontalLayout = QHBoxLayout()
        self.select_format_label = QLabel(self.link_frame)
        select_format_horizontalLayout.addWidget(self.select_format_label)
        self.media_comboBox = QComboBox(self.link_frame)
        self.media_comboBox.setMinimumWidth(200)
        select_format_horizontalLayout.addWidget(self.media_comboBox)
        self.duration_label = QLabel(self.link_frame)
        select_format_horizontalLayout.addWidget(self.duration_label)
        self.format_selection_frame = QFrame(self)
        self.format_selection_frame.setLayout(select_format_horizontalLayout)
        self.link_verticalLayout.addWidget(self.format_selection_frame)
        self.advanced_format_selection_checkBox = QCheckBox(self)
        self.link_verticalLayout.addWidget(self.advanced_format_selection_checkBox)
        self.advanced_format_selection_frame = QFrame(self)
        self.link_verticalLayout.addWidget(self.advanced_format_selection_frame)
        advanced_format_selection_horizontalLayout = QHBoxLayout(self.advanced_format_selection_frame)
        self.video_format_selection_label = QLabel(self.advanced_format_selection_frame)
        self.video_format_selection_comboBox = QComboBox(self.advanced_format_selection_frame)
        self.audio_format_selection_label = QLabel(self.advanced_format_selection_frame)
        self.audio_format_selection_comboBox = QComboBox(self.advanced_format_selection_frame)
        for widget in [self.video_format_selection_label, self.video_format_selection_comboBox, self.audio_format_selection_label, self.audio_format_selection_comboBox]:
            advanced_format_selection_horizontalLayout.addWidget(widget)
        self.url_submit_pushButtontton.setText(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Fetch Media List'))
        self.select_format_label.setText(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Select a format'))
        self.video_format_selection_label.setText(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Video format:'))
        self.audio_format_selection_label.setText(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Audio format:'))
        self.advanced_format_selection_checkBox.setText(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Advanced options'))
        self.url_submit_pushButtontton.setEnabled(False)
        self.change_name_lineEdit.setEnabled(False)
        self.ok_pushButton.setEnabled(False)
        self.download_later_pushButton.setEnabled(False)
        self.format_selection_frame.setEnabled(True)
        self.advanced_format_selection_frame.setEnabled(False)
        self.advanced_format_selection_checkBox.toggled.connect(self.advancedFormatFrame)
        self.url_submit_pushButtontton.clicked.connect(self.submitClicked)
        self.media_comboBox.activated.connect(partial(self.mediaSelectionChanged, 'video_audio'))
        self.video_format_selection_comboBox.activated.connect(partial(self.mediaSelectionChanged, 'video'))
        self.audio_format_selection_comboBox.activated.connect(partial(self.mediaSelectionChanged, 'audio'))
        self.link_lineEdit.textChanged.disconnect(super().linkLineChanged)
        self.link_lineEdit.textChanged.connect(self.linkLineChangedHere)
        self.setMinimumSize(650, 480)
        self.status_box_textEdit.hide()
        self.format_selection_frame.hide()
        self.advanced_format_selection_frame.hide()
        self.advanced_format_selection_checkBox.hide()
        if 'link' in video_dict.keys() and video_dict['link']:
            self.link_lineEdit.setText(video_dict['link'])
            self.url_submit_pushButtontton.setEnabled(True)
        else:
            clipboard = QApplication.clipboard()
            text = clipboard.text()
            if 'tp:/' in text[2:6] or 'tps:/' in text[2:7]:
                self.link_lineEdit.setText(str(text))
            self.url_submit_pushButtontton.setEnabled(True)

    def advancedFormatFrame(self, button):
        if False:
            print('Hello World!')
        if self.advanced_format_selection_checkBox.isChecked():
            self.advanced_format_selection_frame.setEnabled(True)
            self.format_selection_frame.setEnabled(False)
            self.mediaSelectionChanged('video', int(self.video_format_selection_comboBox.currentIndex()))
        else:
            self.advanced_format_selection_frame.setEnabled(False)
            self.format_selection_frame.setEnabled(True)
            self.mediaSelectionChanged('video_audio', int(self.media_comboBox.currentIndex()))

    def getReadableSize(self, size):
        if False:
            for i in range(10):
                print('nop')
        try:
            return '{:1.2f} MB'.format(int(size) / 1048576)
        except:
            return str(size)

    def getReadableDuration(self, seconds):
        if False:
            while True:
                i = 10
        try:
            seconds = int(seconds)
            hours = seconds // 3600
            seconds = seconds % 3600
            minutes = seconds // 60
            seconds = seconds % 60
            return '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
        except:
            return str(seconds)

    def urlChanged(self, value):
        if False:
            print('Hello World!')
        if ' ' in value or value == '':
            self.url_submit_pushButtontton.setEnabled(False)
            self.url_submit_pushButtontton.setToolTip(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Please enter a valid video link'))
        else:
            self.url_submit_pushButtontton.setEnabled(True)
            self.url_submit_pushButtontton.setToolTip('')

    def submitClicked(self, button=None):
        if False:
            i = 10
            return i + 15
        self.media_comboBox.clear()
        self.format_selection_frame.hide()
        self.advanced_format_selection_checkBox.hide()
        self.advanced_format_selection_frame.hide()
        self.video_format_selection_comboBox.clear()
        self.audio_format_selection_comboBox.clear()
        self.change_name_lineEdit.clear()
        self.threadPool.clear()
        self.change_name_checkBox.setChecked(False)
        self.video_audio_list.clear()
        self.no_video_list.clear()
        self.no_audio_list.clear()
        self.url_submit_pushButtontton.setEnabled(False)
        self.status_box_textEdit.setText(QCoreApplication.translate('ytaddlink_src_ui_tr', 'Fetching Media Info...'))
        self.status_box_textEdit.show()
        self.ok_pushButton.setEnabled(False)
        self.download_later_pushButton.setEnabled(False)
        dictionary_to_send = deepcopy(self.plugin_add_link_dictionary)
        more_options = self.collectMoreOptions()
        for k in more_options.keys():
            dictionary_to_send[k] = more_options[k]
        dictionary_to_send['link'] = self.link_lineEdit.text()
        fetcher_thread = MediaListFetcherThread(self.fetchedResult, dictionary_to_send, self)
        self.parent.threadPool.append(fetcher_thread)
        self.parent.threadPool[len(self.parent.threadPool) - 1].start()

    def fileNameChanged(self, value):
        if False:
            for i in range(10):
                print('nop')
        if value.strip() == '':
            self.ok_pushButton.setEnabled(False)

    def mediaSelectionChanged(self, combobox, index):
        if False:
            return 10
        try:
            if combobox == 'video_audio':
                if self.media_comboBox.currentText() == 'Best quality':
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.no_audio_list[-1]['ext'])
                else:
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.video_audio_list[index]['ext'])
                self.change_name_checkBox.setChecked(True)
            elif combobox == 'video':
                if self.video_format_selection_comboBox.currentText() != 'No video':
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.no_audio_list[index - 1]['ext'])
                    self.change_name_checkBox.setChecked(True)
                elif self.audio_format_selection_comboBox.currentText() != 'No audio':
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.no_video_list[int(self.audio_format_selection_comboBox.currentIndex()) - 1]['ext'])
                    self.change_name_checkBox.setChecked(True)
                else:
                    self.change_name_lineEdit.setChecked(False)
            elif combobox == 'audio':
                if self.audio_format_selection_comboBox.currentText() != 'No audio' and self.video_format_selection_comboBox.currentText() == 'No video':
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.no_video_list[index - 1]['ext'])
                    self.change_name_checkBox.setChecked(True)
                elif self.audio_format_selection_comboBox.currentText() == 'No audio' and self.video_format_selection_comboBox.currentText() != 'No video' or (self.audio_format_selection_comboBox.currentText() != 'No audio' and self.video_format_selection_comboBox.currentText() != 'No video'):
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.no_audio_list[int(self.video_format_selection_comboBox.currentIndex()) - 1]['ext'])
                    self.change_name_checkBox.setChecked(True)
                elif self.audio_format_selection_comboBox.currentText() == 'No audio' and self.video_format_selection_comboBox.currentText() == 'No video':
                    self.change_name_checkBox.setChecked(False)
        except Exception as ex:
            logger.sendToLog(ex, 'ERROR')

    def fetchedResult(self, media_dict):
        if False:
            while True:
                i = 10
        self.url_submit_pushButtontton.setEnabled(True)
        if 'error' in media_dict.keys():
            self.status_box_textEdit.setText('<font color="#f11">' + str(media_dict['error']) + '</font>')
            self.status_box_textEdit.show()
        else:
            self.video_format_selection_comboBox.addItem('No video')
            self.audio_format_selection_comboBox.addItem('No audio')
            self.media_title = media_dict['title']
            if 'formats' not in media_dict.keys() and 'entries' in media_dict.keys():
                formats = media_dict['entries']
                formats = formats[0]
                media_dict['formats'] = formats['formats']
            elif 'formats' not in media_dict.keys() and 'format' in media_dict.keys():
                media_dict['formats'] = [media_dict.copy()]
            try:
                i = 0
                for f in media_dict['formats']:
                    no_audio = False
                    no_video = False
                    text = ''
                    if 'acodec' in f.keys():
                        if f['acodec'] == 'none':
                            no_audio = True
                        if 'height' in f.keys():
                            text = text + ' ' + '{}p'.format(f['height'])
                    if 'vcodec' in f.keys():
                        if f['vcodec'] == 'none':
                            text = text + '{}kbps'.format(f['abr'])
                            no_video = True
                    if 'ext' in f.keys():
                        text = text + ' ' + '.{}'.format(f['ext'])
                    if 'filesize' in f.keys() and f['filesize']:
                        text = text + ' ' + '{}'.format(self.getReadableSize(f['filesize']))
                    else:
                        input_dict = deepcopy(self.plugin_add_link_dictionary)
                        input_dict['link'] = f['url']
                        more_options = self.collectMoreOptions()
                        for key in more_options.keys():
                            input_dict[key] = more_options[key]
                        size_fetcher = FileSizeFetcherThread(input_dict, i)
                        self.threadPool[str(i)] = {'thread': size_fetcher, 'item_id': i}
                        self.parent.threadPool.append(size_fetcher)
                        self.parent.threadPool[len(self.parent.threadPool) - 1].start()
                        self.parent.threadPool[len(self.parent.threadPool) - 1].FOUND.connect(self.findFileSize)
                    if no_audio:
                        self.no_audio_list.append(f)
                        self.video_format_selection_comboBox.addItem(text)
                    elif no_video:
                        self.no_video_list.append(f)
                        self.audio_format_selection_comboBox.addItem(text)
                    else:
                        self.video_audio_list.append(f)
                        self.media_comboBox.addItem(text)
                    i = i + 1
                self.status_box_textEdit.hide()
                if 'duration' in media_dict.keys():
                    self.duration_label.setText('Duration ' + self.getReadableDuration(media_dict['duration']))
                self.format_selection_frame.show()
                self.advanced_format_selection_checkBox.show()
                self.advanced_format_selection_frame.show()
                self.ok_pushButton.setEnabled(True)
                self.download_later_pushButton.setEnabled(True)
                if len(self.no_audio_list) == 0 and len(self.no_video_list) == 0:
                    self.advanced_format_selection_checkBox.hide()
                    self.advanced_format_selection_frame.hide()
                if len(self.no_audio_list) != 0 and len(self.no_video_list) != 0:
                    self.media_comboBox.addItem('Best quality')
                    self.media_comboBox.setCurrentIndex(len(self.video_audio_list))
                    self.change_name_lineEdit.setText(self.media_title)
                    self.extension_label.setText('.' + self.no_audio_list[-1]['ext'])
                    self.change_name_checkBox.setChecked(True)
                elif len(self.video_audio_list) != 0:
                    self.media_comboBox.setCurrentIndex(len(self.video_audio_list) - 1)
                if len(self.no_audio_list) != 0:
                    self.video_format_selection_comboBox.setCurrentIndex(len(self.no_audio_list))
                if len(self.no_video_list) != 0:
                    self.audio_format_selection_comboBox.setCurrentIndex(len(self.no_video_list))
                if len(self.video_audio_list) == 0:
                    self.media_comboBox.hide()
                    self.select_format_label.hide()
                    if len(self.no_video_list) != 0 and len(self.no_audio_list) == 0:
                        self.mediaSelectionChanged('video', int(self.video_format_selection_comboBox.currentIndex()))
                        self.advanced_format_selection_checkBox.setChecked(True)
                        self.advanced_format_selection_checkBox.hide()
                    elif len(self.no_video_list) == 0 and len(self.no_audio_list) != 0:
                        self.mediaSelectionChanged('audio', int(self.audio_format_selection_comboBox.currentIndex()))
                        self.advanced_format_selection_checkBox.setChecked(True)
                        self.advanced_format_selection_checkBox.hide()
                    else:
                        self.mediaSelectionChanged('video_audio', int(self.media_comboBox.currentIndex()))
            except Exception as ex:
                logger.sendToLog(ex, 'ERROR')

    def findFileSize(self, result):
        if False:
            print('Hello World!')
        try:
            item_id = self.threadPool[str(result['thread_key'])]['item_id']
            if result['file_size'] and result['file_size'] != '0':
                text = self.media_comboBox.itemText(item_id)
                self.media_comboBox.setItemText(item_id, '{} - {}'.format(text, result['file_size']))
        except Exception as ex:
            logger.sendToLog(ex, 'ERROR')

    def linkLineChangedHere(self, lineEdit):
        if False:
            return 10
        if str(lineEdit) == '':
            self.url_submit_pushButtontton.setEnabled(False)
        else:
            self.url_submit_pushButtontton.setEnabled(True)

    def collectMoreOptions(self):
        if False:
            i = 10
            return i + 15
        options = {'ip': None, 'port': None, 'proxy_user': None, 'proxy_passwd': None, 'download_user': None, 'download_passwd': None}
        if self.proxy_checkBox.isChecked():
            options['ip'] = self.ip_lineEdit.text()
            options['port'] = self.port_spinBox.value()
            options['proxy_user'] = self.proxy_user_lineEdit.text()
            options['proxy_passwd'] = self.proxy_pass_lineEdit.text()
        if self.download_checkBox.isChecked():
            options['download_user'] = self.download_user_lineEdit.text()
            options['download_passwd'] = self.download_pass_lineEdit.text()
        additional_info = ['header', 'load_cookies', 'user_agent', 'referer', 'out']
        for i in additional_info:
            if i not in self.plugin_add_link_dictionary.keys():
                options[i] = None
        return options

    def okButtonPressed(self, download_later, button=None):
        if False:
            i = 10
            return i + 15
        link_list = []
        if self.advanced_format_selection_checkBox.isChecked():
            if self.video_format_selection_comboBox.currentText() == 'No video' and self.audio_format_selection_comboBox.currentText() != 'No audio':
                audio_link = self.no_video_list[self.audio_format_selection_comboBox.currentIndex() - 1]['url']
                link_list.append(audio_link)
            elif self.video_format_selection_comboBox.currentText() != 'No video' and self.audio_format_selection_comboBox.currentText() == 'No audio':
                video_link = self.no_audio_list[self.video_format_selection_comboBox.currentIndex() - 1]['url']
                link_list.append(video_link)
            elif self.video_format_selection_comboBox.currentText() != 'No video' and self.audio_format_selection_comboBox.currentText() != 'No audio':
                audio_link = self.no_video_list[self.audio_format_selection_comboBox.currentIndex() - 1]['url']
                video_link = self.no_audio_list[self.video_format_selection_comboBox.currentIndex() - 1]['url']
                link_list = [video_link, audio_link]
            elif self.video_format_selection_comboBox.currentText() == 'No video' and self.audio_format_selection_comboBox.currentText() == 'No audio':
                self.close()
        elif self.media_comboBox.currentText() == 'Best quality':
            video_link = self.no_audio_list[-1]['url']
            audio_link = self.no_video_list[-1]['url']
            link_list = [video_link, audio_link]
        else:
            audio_and_video_link = self.video_audio_list[self.media_comboBox.currentIndex()]['url']
            link_list.append(audio_and_video_link)
        self.persepolis_setting.setValue('add_link_initialization/ip', self.ip_lineEdit.text())
        self.persepolis_setting.setValue('add_link_initialization/port', self.port_spinBox.value())
        self.persepolis_setting.setValue('add_link_initialization/proxy_user', self.proxy_user_lineEdit.text())
        self.persepolis_setting.setValue('add_link_initialization/download_user', self.download_user_lineEdit.text())
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
            name = str(self.change_name_lineEdit.text())
            if name == '':
                name = 'video_finder_file'
        else:
            name = 'video_finder_file'
        if str(self.extension_label.text()) == '':
            extension = '.mp4'
        else:
            extension = str(self.extension_label.text())
        if len(link_list) == 2:
            video_name = name + extension
            audio_name = name + '.' + str(self.no_video_list[self.audio_format_selection_comboBox.currentIndex() - 1]['ext'])
            name_list = [video_name, audio_name]
        else:
            name_list = [name + extension]
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
        add_link_dictionary_list = []
        if len(link_list) == 1:
            add_link_dictionary = {'referer': referer, 'header': header, 'user_agent': user_agent, 'load_cookies': load_cookies, 'out': name_list[0], 'start_time': start_time, 'end_time': end_time, 'link': link_list[0], 'ip': ip, 'port': port, 'proxy_user': proxy_user, 'proxy_passwd': proxy_passwd, 'download_user': download_user, 'download_passwd': download_passwd, 'connections': connections, 'limit_value': limit, 'download_path': download_path}
            add_link_dictionary_list.append(add_link_dictionary)
        else:
            video_add_link_dictionary = {'referer': referer, 'header': header, 'user_agent': user_agent, 'load_cookies': load_cookies, 'out': name_list[0], 'start_time': start_time, 'end_time': end_time, 'link': link_list[0], 'ip': ip, 'port': port, 'proxy_user': proxy_user, 'proxy_passwd': proxy_passwd, 'download_user': download_user, 'download_passwd': download_passwd, 'connections': connections, 'limit_value': limit, 'download_path': download_path}
            audio_add_link_dictionary = {'referer': referer, 'header': header, 'user_agent': user_agent, 'load_cookies': load_cookies, 'out': name_list[1], 'start_time': None, 'end_time': end_time, 'link': link_list[1], 'ip': ip, 'port': port, 'proxy_user': proxy_user, 'proxy_passwd': proxy_passwd, 'download_user': download_user, 'download_passwd': download_passwd, 'connections': connections, 'limit_value': limit, 'download_path': download_path}
            add_link_dictionary_list = [video_add_link_dictionary, audio_add_link_dictionary]
        category = str(self.add_queue_comboBox.currentText())
        del self.plugin_add_link_dictionary
        self.callback(add_link_dictionary_list, download_later, category)
        self.close()