try:
    from PySide6.QtGui import QFont
    from PySide6.QtWidgets import QApplication, QStyleFactory
    from PySide6.QtCore import QFile, QTextStream, QCoreApplication, QSettings, Qt
except:
    from PyQt5.QtGui import QFont
    from PyQt5.QtWidgets import QApplication, QStyleFactory
    from PyQt5.QtCore import QFile, QTextStream, QCoreApplication, QSettings, Qt
from persepolis.gui import resources
import traceback
from persepolis.scripts.error_window import ErrorWindow
from persepolis.gui.palettes import DarkFusionPalette, LightFusionPalette
import json
import struct
import argparse
from persepolis.scripts import osCommands
from persepolis.scripts.useful_tools import osAndDesktopEnvironment, determineConfigFolder
from persepolis.constants import OS
from copy import deepcopy
import sys
import os
(os_type, desktop_env) = osAndDesktopEnvironment()
if os_type in OS.UNIX_LIKE + [OS.OSX]:
    uid = os.getuid()
    if uid == 0:
        print('Do not run persepolis as root.')
        sys.exit(1)
home_address = os.path.expanduser('~')
config_folder = determineConfigFolder()
persepolis_tmp = os.path.join(config_folder, 'persepolis_tmp')
global lock_file_validation
if os_type != OS.WINDOWS:
    import fcntl
    user_name_split = home_address.split('/')
    user_name = user_name_split[2]
    lock_file = '/tmp/persepolis_exec_' + user_name + '.lock'
    fp = open(lock_file, 'w')
    try:
        fcntl.lockf(fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
        lock_file_validation = True
    except IOError:
        lock_file_validation = False
else:
    from win32event import CreateMutex
    from win32api import GetLastError
    from winerror import ERROR_ALREADY_EXISTS
    from sys import exit
    handle = CreateMutex(None, 1, 'persepolis_download_manager')
    if GetLastError() == ERROR_ALREADY_EXISTS:
        lock_file_validation = False
    else:
        lock_file_validation = True
if lock_file_validation:
    from persepolis.scripts import initialization
    from persepolis.scripts.mainwindow import MainWindow
    if os_type in OS.UNIX_LIKE:
        try:
            from setproctitle import setproctitle
            setproctitle('persepolis')
        except:
            from persepolis.scripts import logger
            logger.sendToLog('setproctitle is not installed!', 'ERROR')
persepolis_setting = QSettings('persepolis_download_manager', 'persepolis')

class PersepolisApplication(QApplication):

    def __init__(self, argv):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(argv)

    def setPersepolisStyle(self, style):
        if False:
            while True:
                i = 10
        self.persepolis_style = style
        self.setStyle(style)

    def setPersepolisFont(self, font, font_size, custom_font):
        if False:
            i = 10
            return i + 15
        self.persepolis_font = font
        self.persepolis_font_size = font_size
        if custom_font == 'yes':
            self.setFont(QFont(font, font_size))

    def setPersepolisColorScheme(self, color_scheme):
        if False:
            print('Hello World!')
        self.persepolis_color_scheme = color_scheme
        if color_scheme == 'Dark Fusion':
            dark_fusion = DarkFusionPalette()
            self.setPalette(dark_fusion)
            file = QFile(':/dark_style.qss')
            file.open(QFile.ReadOnly | QFile.Text)
            stream = QTextStream(file)
            self.setStyleSheet(stream.readAll())
        elif color_scheme == 'Light Fusion':
            light_fusion = LightFusionPalette()
            self.setPalette(light_fusion)
            file = QFile(':/light_style.qss')
            file.open(QFile.ReadOnly | QFile.Text)
            stream = QTextStream(file)
            self.setStyleSheet(stream.readAll())
parser = argparse.ArgumentParser(description='Persepolis Download Manager')
parser.add_argument('--link', action='store', nargs=1, help='Download link.(Use "" for links)')
parser.add_argument('--referer', action='store', nargs=1, help='Set an http referrer (Referer). This affects all http/https downloads.  If * is given, the download URI is also used as the referrer.')
parser.add_argument('--cookie', action='store', nargs=1, help='Cookie')
parser.add_argument('--agent', action='store', nargs=1, help='Set user agent for HTTP(S) downloads.  Default: aria2/$VERSION, $VERSION is replaced by package version.')
parser.add_argument('--headers', action='store', nargs=1, help='Append HEADER to HTTP request header. ')
parser.add_argument('--name', action='store', nargs=1, help='The  file  name  of  the downloaded file. ')
parser.add_argument('--default', action='store_true', help='restore default setting')
parser.add_argument('--clear', action='store_true', help='Clear download list and user setting!')
parser.add_argument('--tray', action='store_true', help="Persepolis is starting in tray icon. It's useful when you want to put persepolis in system's startup.")
parser.add_argument('--parent-window', action='store', nargs=1, help='this switch is used for chrome native messaging in Windows')
parser.add_argument('--version', action='version', version='Persepolis Download Manager 3.2.0')
(args, unknownargs) = parser.parse_known_args()
browser_url = True
add_link_dictionary = {}
plugin_list = []
browser_plugin_dict = {'link': None, 'referer': None, 'load_cookies': None, 'user_agent': None, 'header': None, 'out': None}
if args.parent_window or unknownargs:
    if os_type == OS.WINDOWS:
        import msvcrt
        msvcrt.setmode(sys.stdin.fileno(), os.O_BINARY)
        msvcrt.setmode(sys.stdout.fileno(), os.O_BINARY)
    message = '{"enable": true, "version": "1.85"}'.encode('utf-8')
    sys.stdout.buffer.write(struct.pack('i', len(message)))
    sys.stdout.buffer.write(message)
    sys.stdout.flush()
    text_length_bytes = sys.stdin.buffer.read(4)
    text_length = struct.unpack('@I', text_length_bytes)[0]
    text = sys.stdin.buffer.read(text_length).decode('utf-8')
    if text:
        new_dict = json.loads(text)
        if 'url_links' in new_dict:
            for item in new_dict['url_links']:
                copy_dict = deepcopy(browser_plugin_dict)
                if 'url' in item.keys():
                    copy_dict['link'] = str(item['url'])
                    if 'header' in item.keys() and item['header'] != '':
                        copy_dict['header'] = item['header']
                    if 'referrer' in item.keys() and item['referrer'] != '':
                        copy_dict['referer'] = item['referrer']
                    if 'filename' in item.keys() and item['filename'] != '':
                        copy_dict['out'] = os.path.basename(str(item['filename']))
                    if 'useragent' in item.keys() and item['useragent'] != '':
                        copy_dict['user_agent'] = item['useragent']
                    if 'cookies' in item.keys() and item['cookies'] != '':
                        copy_dict['load_cookies'] = item['cookies']
                    plugin_list.append(copy_dict)
        else:
            browser_url = False
if args.clear:
    from persepolis.scripts.data_base import PersepolisDB
    persepolis_db = PersepolisDB()
    persepolis_db.resetDataBase()
    persepolis_db.closeConnections()
    persepolis_setting.clear()
    persepolis_setting.sync()
    sys.exit(0)
if args.default:
    persepolis_setting.clear()
    persepolis_setting.sync()
    print('Persepolis restored default')
    sys.exit(0)
if args.link:
    add_link_dictionary['link'] = ''.join(args.link)
    args.tray = True
if args.referer:
    add_link_dictionary['referer'] = ''.join(args.referer)
else:
    add_link_dictionary['referer'] = None
if args.cookie:
    add_link_dictionary['load_cookies'] = ''.join(args.cookie)
else:
    add_link_dictionary['load_cookies'] = None
if args.agent:
    add_link_dictionary['user_agent'] = ''.join(args.agent)
else:
    add_link_dictionary['user_agent'] = None
if args.headers:
    add_link_dictionary['header'] = ''.join(args.headers)
else:
    add_link_dictionary['header'] = None
if args.name:
    add_link_dictionary['out'] = ''.join(args.name)
else:
    add_link_dictionary['out'] = None
if args.tray:
    start_in_tray = True
else:
    start_in_tray = False
if 'link' in add_link_dictionary.keys():
    plugin_dict = {'link': add_link_dictionary['link'], 'referer': add_link_dictionary['referer'], 'load_cookies': add_link_dictionary['load_cookies'], 'user_agent': add_link_dictionary['user_agent'], 'header': add_link_dictionary['header'], 'out': add_link_dictionary['out']}
    plugin_list.append(plugin_dict)
if len(plugin_list) != 0:
    from persepolis.scripts.data_base import PluginsDB
    plugins_db = PluginsDB()
    plugins_db.insertInPluginsTable(plugin_list)
    plugins_db.closeConnections()
    plugin_ready = os.path.join(persepolis_tmp, 'persepolis-plugin-ready')
    osCommands.touch(plugin_ready)
    start_in_tray = True
if str(persepolis_setting.value('settings/browser-persepolis')) == 'yes' and (args.parent_window or unknownargs):
    start_persepolis_if_browser_executed = True
    start_in_tray = True
else:
    start_persepolis_if_browser_executed = False

def main():
    if False:
        while True:
            i = 10
    if lock_file_validation and (not ((args.parent_window or unknownargs) and browser_url == False) or ((args.parent_window or unknownargs) and start_persepolis_if_browser_executed)):
        os.environ['QT_AUTO_SCREEN_SCALE_FACTOR'] = '1'
        try:
            QApplication.setAttribute(Qt.AA_EnableHighDpiScaling)
        except:
            from persepolis.scripts import logger
            logger.sendToLog('Qt.AA_EnableHighDpiScaling is not available!', 'ERROR')
        persepolis_download_manager = PersepolisApplication(sys.argv)
        persepolis_download_manager.setQuitOnLastWindowClosed(False)
        try:
            if hasattr(QStyleFactory, 'AA_UseHighDpiPixmaps'):
                persepolis_download_manager.setAttribute(Qt.AA_UseHighDpiPixmaps)
        except:
            from persepolis.scripts import logger
            logger.sendToLog('Qt.AA_UseHighDpiPixmaps is not available!', 'ERROR')
        QCoreApplication.setOrganizationName('persepolis_download_manager')
        QCoreApplication.setApplicationName('persepolis')
        persepolis_download_manager.setting = QSettings()
        custom_font = persepolis_download_manager.setting.value('settings/custom-font')
        font = persepolis_download_manager.setting.value('settings/font')
        font_size = int(persepolis_download_manager.setting.value('settings/font-size'))
        style = persepolis_download_manager.setting.value('settings/style')
        color_scheme = persepolis_download_manager.setting.value('settings/color-scheme')
        ui_direction = persepolis_download_manager.setting.value('ui_direction')
        persepolis_download_manager.setPersepolisStyle(style)
        persepolis_download_manager.setPersepolisFont(font, font_size, custom_font)
        persepolis_download_manager.setPersepolisColorScheme(color_scheme)
        if ui_direction == 'rtl':
            persepolis_download_manager.setLayoutDirection(Qt.RightToLeft)
        elif ui_direction in 'ltr':
            persepolis_download_manager.setLayoutDirection(Qt.LeftToRight)
        try:
            mainwindow = MainWindow(start_in_tray, persepolis_download_manager, persepolis_download_manager.setting)
            if start_in_tray:
                mainwindow.hide()
            else:
                mainwindow.show()
        except Exception:
            from persepolis.scripts import logger
            error_message = str(traceback.format_exc())
            logger.sendToLog(error_message, 'ERROR')
            error_window = ErrorWindow(error_message)
            error_window.show()
        sys.exit(persepolis_download_manager.exec_())
    elif not (args.parent_window or unknownargs):
        if len(plugin_list) == 0:
            show_window_file = os.path.join(persepolis_tmp, 'show-window')
            f = open(show_window_file, 'w')
            f.close()
        sys.exit(0)