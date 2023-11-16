import autokey.dbus_service
import autokey.model.script
from . import common
common.USING_QT = False
import sys
import os.path
import time
import threading
import gettext
import dbus
import dbus.service
import dbus.mainloop.glib
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gdk, GObject, GLib
gettext.install('autokey')
import autokey.argument_parser
from autokey import service, monitor
from autokey.gtkui.notifier import get_notifier
from autokey.gtkui.popupmenu import PopupMenu
from autokey.gtkui.configwindow import ConfigWindow
from autokey.gtkui.dialogs import ShowScriptErrorsDialog
import autokey.configmanager.configmanager as cm
import autokey.configmanager.configmanager_constants as cm_constants
import autokey.UI_common_functions as UI_common
from autokey.logger import get_logger, configure_root_logger
from autokey.UI_common_functions import checkRequirements, checkOptionalPrograms, create_storage_directories
logger = get_logger(__name__)
PROGRAM_NAME = _('AutoKey')
DESCRIPTION = _('Desktop automation utility')
COPYRIGHT = _('(c) 2008-2011 Chris Dekter')

class Application:
    """
    Main application class; starting and stopping of the application is controlled
    from here, together with some interactions from the tray icon.
    """

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        GLib.threads_init()
        Gdk.threads_init()
        args = autokey.argument_parser.parse_args()
        configure_root_logger(args)
        checkOptionalPrograms()
        missing_reqs = checkRequirements()
        if len(missing_reqs) > 0:
            Gdk.threads_enter()
            self.show_error_dialog('AutoKey Requires the following programs or python modules to be installed to function properly', missing_reqs)
            Gdk.threads_leave()
            sys.exit('Missing required programs and/or python modules, exiting')
        try:
            create_storage_directories()
            if self.__verifyNotRunning():
                UI_common.create_lock_file()
            self.initialise(args.show_config_window)
        except Exception as e:
            self.show_error_dialog(_('Fatal error starting AutoKey.\n') + str(e))
            logger.exception('Fatal error starting AutoKey: ' + str(e))
            sys.exit(1)

    def __verifyNotRunning(self):
        if False:
            for i in range(10):
                print('nop')
        if UI_common.is_existing_running_autokey():
            UI_common.test_Dbus_response(self)
        return True

    def initialise(self, configure):
        if False:
            return 10
        logger.info('Initialising application')
        self.monitor = monitor.FileMonitor(self)
        self.configManager = cm.create_config_manager_instance(self)
        self.service = service.Service(self)
        self.serviceDisabled = False
        if self.configManager.userCodeDir is not None:
            sys.path.append(self.configManager.userCodeDir)
        try:
            self.service.start()
        except Exception as e:
            logger.exception('Error starting interface: ' + str(e))
            self.serviceDisabled = True
            self.show_error_dialog(_('Error starting interface. Keyboard monitoring will be disabled.\n' + 'Check your system/configuration.'), str(e))
        self.notifier = get_notifier(self)
        self.configWindow = None
        self.monitor.start()
        dbus.mainloop.glib.DBusGMainLoop(set_as_default=True)
        self.dbusService = autokey.dbus_service.AppService(self)
        if configure:
            self.show_configure()

    def init_global_hotkeys(self, configManager):
        if False:
            for i in range(10):
                print('nop')
        logger.info('Initialise global hotkeys')
        configManager.toggleServiceHotkey.set_closure(self.toggle_service)
        configManager.configHotkey.set_closure(self.show_configure_async)

    def config_altered(self, persistGlobal):
        if False:
            i = 10
            return i + 15
        self.configManager.config_altered(persistGlobal)
        self.notifier.rebuild_menu()

    def hotkey_created(self, item):
        if False:
            i = 10
            return i + 15
        UI_common.hotkey_created(self.service, item)

    def hotkey_removed(self, item):
        if False:
            i = 10
            return i + 15
        UI_common.hotkey_removed(self.service, item)

    def path_created_or_modified(self, path):
        if False:
            print('Hello World!')
        UI_common.path_created_or_modified(self.configManager, self.configWindow, path)

    def path_removed(self, path):
        if False:
            return 10
        UI_common.path_removed(self.configManager, self.configWindow, path)

    def unpause_service(self):
        if False:
            while True:
                i = 10
        '\n        Unpause the expansion service (start responding to keyboard and mouse events).\n        '
        self.service.unpause()
        self.notifier.update_tool_tip()

    def pause_service(self):
        if False:
            while True:
                i = 10
        '\n        Pause the expansion service (stop responding to keyboard and mouse events).\n        '
        self.service.pause()
        self.notifier.update_tool_tip()

    def toggle_service(self):
        if False:
            while True:
                i = 10
        '\n        Convenience method for toggling the expansion service on or off.\n        '
        if self.service.is_running():
            self.pause_service()
        else:
            self.unpause_service()

    def shutdown(self):
        if False:
            print('Hello World!')
        '\n        Shut down the entire application.\n        '
        if self.configWindow is not None:
            if self.configWindow.promptToSave():
                return
            self.configWindow.hide()
        self.notifier.hide_icon()
        t = threading.Thread(target=self.__completeShutdown)
        t.start()

    def __completeShutdown(self):
        if False:
            while True:
                i = 10
        logger.info('Shutting down')
        self.service.shutdown()
        self.monitor.stop()
        Gdk.threads_enter()
        Gtk.main_quit()
        Gdk.threads_leave()
        os.remove(common.LOCK_FILE)
        logger.debug('All shutdown tasks complete... quitting')

    def notify_error(self, error: autokey.model.script.ScriptErrorRecord):
        if False:
            i = 10
            return i + 15
        '\n        Show an error notification popup.\n\n        @param error: The error that occurred in a Script\n        '
        message = "The script '{}' encountered an error".format(error.script_name)
        self.notifier.notify_error(message)
        if self.configWindow is not None:
            self.configWindow.set_has_errors(True)

    def update_notifier_visibility(self):
        if False:
            return 10
        self.notifier.update_visible_status()

    def show_configure(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Show the configuration window, or deiconify (un-minimise) it if it's already open.\n        "
        logger.info('Displaying configuration window')
        if self.configWindow is None:
            self.configWindow = ConfigWindow(self)
            self.configWindow.show()
        else:
            self.configWindow.deiconify()

    def show_configure_async(self):
        if False:
            for i in range(10):
                print('nop')
        Gdk.threads_enter()
        self.show_configure()
        Gdk.threads_leave()

    def main(self):
        if False:
            return 10
        logger.info('Entering main()')
        Gdk.threads_enter()
        Gtk.main()
        Gdk.threads_leave()

    def show_error_dialog(self, message, details=None, dialog_type=Gtk.MessageType.ERROR):
        if False:
            return 10
        '\n        Convenience method for showing an error dialog.\n\n        @param dialog_type: One of Gtk.MessageType.ERROR, Gtk.MessageType.WARNING , Gtk.MessageType.INFO, Gtk.MessageType.OTHER, Gtk.MessageType.QUESTION\n            defaults to Gtk.MessageType.ERROR\n        '
        logger.debug('Displaying ' + dialog_type.value_name + ' Dialog')
        dlg = Gtk.MessageDialog(type=dialog_type, buttons=Gtk.ButtonsType.OK, message_format=message)
        if details is not None:
            dlg.format_secondary_text(details)
        dlg.run()
        dlg.destroy()

    def show_script_error(self, parent):
        if False:
            while True:
                i = 10
        '\n        Show the last script error (if any)\n        '
        if self.service.scriptRunner.error_records:
            dlg = ShowScriptErrorsDialog(self)
            self.notifier.set_icon(cm.ConfigManager.SETTINGS[cm_constants.NOTIFICATION_ICON])
            self.notifier.errorItem.hide()
            self.notifier.update_visible_status()
        else:
            dlg = Gtk.MessageDialog(type=Gtk.MessageType.INFO, buttons=Gtk.ButtonsType.OK, message_format=_('No error information available'))
            dlg.set_title(_('View script error'))
            dlg.set_transient_for(parent)
        dlg.run()
        dlg.destroy()

    def show_popup_menu(self, folders: list=None, items: list=None, onDesktop=True, title=None):
        if False:
            print('Hello World!')
        if items is None:
            items = []
        if folders is None:
            folders = []
        self.menu = PopupMenu(self.service, folders, items, onDesktop, title)
        self.menu.show_on_desktop()

    def hide_menu(self):
        if False:
            return 10
        self.menu.remove_from_desktop()