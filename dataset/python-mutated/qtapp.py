import sys
import os.path
import queue
import time
import dbus
from typing import NamedTuple, Iterable
from PyQt5.QtCore import QObject, QEvent, Qt, pyqtSignal
from PyQt5.QtGui import QCursor, QIcon
from PyQt5.QtWidgets import QMessageBox, QApplication
import autokey.model.script
from autokey import common
common.USING_QT = True
from autokey import service, monitor
import autokey.argument_parser
import autokey.configmanager.configmanager as cm
import autokey.configmanager.configmanager_constants as cm_constants
from autokey.qtui import common as ui_common
from autokey.qtui.notifier import Notifier
from autokey.qtui.popupmenu import PopupMenu
from autokey.qtui.configwindow import ConfigWindow
from autokey.qtui.dbus_service import AppService
from autokey.logger import get_logger, configure_root_logger
from autokey.UI_common_functions import checkRequirements, checkOptionalPrograms, create_storage_directories
import autokey.UI_common_functions as UI_common
logger = get_logger(__name__)
del get_logger
AuthorData = NamedTuple('AuthorData', (('name', str), ('role', str), ('email', str)))
AboutData = NamedTuple('AboutData', (('program_name', str), ('version', str), ('program_description', str), ('license_text', str), ('copyright_notice', str), ('homepage_url', str), ('bug_report_email', str), ('author_list', Iterable[AuthorData])))
COPYRIGHT = '(c) 2009-2012 Chris Dekter\n(c) 2014 GuoCi\n(c) 2017, 2018 Thomas Hess\n'
author_data = (AuthorData('Thomas Hess', 'PyKDE4 to PyQt5 port', 'thomas.hess@udo.edu'), AuthorData('GuoCi', 'Python 3 port maintainer', 'guociz@gmail.com'), AuthorData('Chris Dekter', 'Developer', 'cdekter@gmail.com'), AuthorData('Sam Peterson', 'Original developer', 'peabodyenator@gmail.com'))
about_data = AboutData(program_name='AutoKey', version=common.VERSION, program_description='Desktop automation utility', license_text='GPL v3', copyright_notice=COPYRIGHT, homepage_url=common.HOMEPAGE, bug_report_email=common.BUG_EMAIL, author_list=author_data)

class Application(QApplication):
    """
    Main application class; starting and stopping of the application is controlled
    from here, together with some interactions from the tray icon.
    """
    monitoring_disabled = pyqtSignal(bool, name='monitoring_disabled')
    show_configure_signal = pyqtSignal()

    def __init__(self, argv: list=sys.argv):
        if False:
            i = 10
            return i + 15
        super().__init__(argv)
        self.handler = CallbackEventHandler()
        self.args = autokey.argument_parser.parse_args()
        try:
            create_storage_directories()
            configure_root_logger(self.args)
        except Exception as e:
            logger.exception('Fatal error starting AutoKey: ' + str(e))
            self.show_error_dialog('Fatal error starting AutoKey.', str(e))
            sys.exit(1)
        checkOptionalPrograms()
        missing_reqs = checkRequirements()
        if len(missing_reqs) > 0:
            self.show_error_dialog('AutoKey Requires the following programs or python modules to be installed to function properly\n\n' + missing_reqs)
            sys.exit('Missing required programs and/or python modules, exiting')
        logger.info('Initialising application')
        self.setWindowIcon(QIcon.fromTheme(common.ICON_FILE, ui_common.load_icon(ui_common.AutoKeyIcon.AUTOKEY)))
        try:
            if self._verify_not_running():
                UI_common.create_lock_file()
            self.monitor = monitor.FileMonitor(self)
            self.configManager = cm.create_config_manager_instance(self)
            self.service = service.Service(self)
            self.serviceDisabled = False
            self._try_start_service()
            self.notifier = Notifier(self)
            self.configWindow = ConfigWindow(self)
            self.configWindow.action_show_last_script_errors.triggered.connect(self.notifier.reset_tray_icon)
            self.notifier.action_view_script_error.triggered.connect(self.configWindow.show_script_errors_dialog.update_and_show)
            self.monitor.start()
            if self.configManager.userCodeDir is not None:
                sys.path.append(self.configManager.userCodeDir)
            logger.debug('Creating DBus service')
            self.dbus_service = AppService(self)
            logger.debug('Service created')
            self.show_configure_signal.connect(self.show_configure, Qt.QueuedConnection)
            if cm.ConfigManager.SETTINGS[cm_constants.IS_FIRST_RUN]:
                cm.ConfigManager.SETTINGS[cm_constants.IS_FIRST_RUN] = False
                self.args.show_config_window = True
            if self.args.show_config_window:
                self.show_configure()
            self.installEventFilter(KeyboardChangeFilter(self.service.mediator.interface))
        except Exception as e:
            logger.exception('Fatal error starting AutoKey: ' + str(e))
            self.show_error_dialog('Fatal error starting AutoKey.', str(e))
            sys.exit(1)
        else:
            sys.exit(self.exec_())

    def _try_start_service(self):
        if False:
            return 10
        try:
            self.service.start()
        except Exception as e:
            logger.exception('Error starting interface: ' + str(e))
            self.serviceDisabled = True
            self.show_error_dialog('Error starting interface. Keyboard monitoring will be disabled.\n' + 'Check your system/configuration.', str(e))

    @staticmethod
    def _create_lock_file():
        if False:
            for i in range(10):
                print('nop')
        with open(common.LOCK_FILE, 'w') as lock_file:
            lock_file.write(str(os.getpid()))

    def _verify_not_running(self):
        if False:
            i = 10
            return i + 15
        if UI_common.is_existing_running_autokey():
            UI_common.test_Dbus_response(self)
        return True

    def init_global_hotkeys(self, configManager):
        if False:
            print('Hello World!')
        logger.info('Initialise global hotkeys')
        configManager.toggleServiceHotkey.set_closure(self.toggle_service)
        configManager.configHotkey.set_closure(self.show_configure_signal.emit)

    def config_altered(self, persistGlobal):
        if False:
            print('Hello World!')
        self.configManager.config_altered(persistGlobal)
        self.notifier.create_assign_context_menu()

    def hotkey_created(self, item):
        if False:
            for i in range(10):
                print('nop')
        UI_common.hotkey_created(self.service, item)

    def hotkey_removed(self, item):
        if False:
            for i in range(10):
                print('nop')
        UI_common.hotkey_removed(self.service, item)

    def path_created_or_modified(self, path):
        if False:
            while True:
                i = 10
        UI_common.path_created_or_modified(self.configManager, self.configWindow, path)

    def path_removed(self, path):
        if False:
            i = 10
            return i + 15
        UI_common.path_removed(self.configManager, self.configWindow, path)

    def unpause_service(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Unpause the expansion service (start responding to keyboard and mouse events).\n        '
        self.service.unpause()

    def pause_service(self):
        if False:
            print('Hello World!')
        '\n        Pause the expansion service (stop responding to keyboard and mouse events).\n        '
        self.service.pause()

    def toggle_service(self):
        if False:
            i = 10
            return i + 15
        '\n        Convenience method for toggling the expansion service on or off. This is called by the global hotkey.\n        '
        self.monitoring_disabled.emit(not self.service.is_running())
        if self.service.is_running():
            self.pause_service()
        else:
            self.unpause_service()

    def shutdown(self):
        if False:
            return 10
        '\n        Shut down the entire application.\n        '
        logger.info('Shutting down')
        self.closeAllWindows()
        self.notifier.hide()
        self.service.shutdown()
        self.monitor.stop()
        self.quit()
        os.remove(common.LOCK_FILE)
        logger.debug('All shutdown tasks complete... quitting')

    def notify_error(self, error: autokey.model.script.ScriptErrorRecord):
        if False:
            for i in range(10):
                print('nop')
        '\n        Show an error notification popup.\n\n        @param error: The error that occurred in a Script\n        '
        message = "The script '{}' encountered an error".format(error.script_name)
        self.exec_in_main(self.notifier.notify_error, message)
        self.configWindow.script_errors_available.emit(True)

    def update_notifier_visibility(self):
        if False:
            return 10
        self.notifier.update_visible_status()

    def show_configure(self):
        if False:
            print('Hello World!')
        "\n        Show the configuration window, or deiconify (un-minimise) it if it's already open.\n        "
        logger.info('Displaying configuration window')
        self.configWindow.show()
        self.configWindow.showNormal()
        self.configWindow.activateWindow()

    @staticmethod
    def show_error_dialog(message: str, details: str=None):
        if False:
            i = 10
            return i + 15
        '\n        Convenience method for showing an error dialog.\n        '
        logger.debug('Displaying Error Dialog')
        message_box = QMessageBox(QMessageBox.Critical, 'Error', message, QMessageBox.Ok, None)
        if details:
            message_box.setDetailedText(details)
        message_box.exec_()

    def show_popup_menu(self, folders: list=None, items: list=None, onDesktop=True, title=None):
        if False:
            return 10
        if items is None:
            items = []
        if folders is None:
            folders = []
        self.exec_in_main(self.__createMenu, folders, items, onDesktop, title)

    def hide_menu(self):
        if False:
            print('Hello World!')
        self.exec_in_main(self.menu.hide)

    def __createMenu(self, folders, items, onDesktop, title):
        if False:
            return 10
        self.menu = PopupMenu(self.service, folders, items, onDesktop, title)
        self.menu.popup(QCursor.pos())
        self.menu.setFocus()

    def exec_in_main(self, callback, *args):
        if False:
            return 10
        self.handler.postEventWithCallback(callback, *args)

class CallbackEventHandler(QObject):

    def __init__(self):
        if False:
            return 10
        QObject.__init__(self)
        self.queue = queue.Queue()

    def customEvent(self, event):
        if False:
            i = 10
            return i + 15
        while True:
            try:
                (callback, args) = self.queue.get_nowait()
            except queue.Empty:
                break
            try:
                callback(*args)
            except Exception:
                logger.exception('callback event failed: %r %r', callback, args, exc_info=True)

    def postEventWithCallback(self, callback, *args):
        if False:
            print('Hello World!')
        self.queue.put((callback, args))
        app = QApplication.instance()
        app.postEvent(self, QEvent(QEvent.User))

class KeyboardChangeFilter(QObject):

    def __init__(self, interface):
        if False:
            print('Hello World!')
        QObject.__init__(self)
        self.interface = interface

    def eventFilter(self, obj, event):
        if False:
            while True:
                i = 10
        if event.type() == QEvent.KeyboardLayoutChange:
            self.interface.on_keys_changed()
        return QObject.eventFilter(obj, event)