"""
New API for plugins.

All plugins in Spyder 5+ must inherit from the classes present in this file.
"""
from collections import OrderedDict
import inspect
import logging
import os
import os.path as osp
import sys
from typing import List, Union
import warnings
from qtpy.QtCore import QObject, Qt, Signal, Slot
from qtpy.QtGui import QCursor
from qtpy.QtWidgets import QApplication
from spyder.api.config.fonts import SpyderFontType
from spyder.api.config.mixins import SpyderConfigurationObserver
from spyder.api.exceptions import SpyderAPIError
from spyder.api.plugin_registration.mixins import SpyderPluginObserver
from spyder.api.widgets.main_widget import PluginMainWidget
from spyder.api.widgets.mixins import SpyderActionMixin
from spyder.api.widgets.mixins import SpyderWidgetMixin
from spyder.app.cli_options import get_options
from spyder.config.gui import get_color_scheme, get_font
from spyder.config.user import NoDefault
from spyder.utils.icon_manager import ima
from spyder.utils.image_path_manager import IMAGE_PATH_MANAGER
from .enum import Plugins
from .old_api import SpyderPluginWidget
logger = logging.getLogger(__name__)

class SpyderPluginV2(QObject, SpyderActionMixin, SpyderConfigurationObserver, SpyderPluginObserver):
    """
    A Spyder plugin to extend functionality without a dockable widget.

    If you want to create a plugin that adds a new pane, please use
    SpyderDockablePlugin.
    """
    NAME = None
    REQUIRES = []
    OPTIONAL = []
    CONTAINER_CLASS = None
    CONF_SECTION = None
    CONF_FILE = True
    CONF_DEFAULTS = None
    CONF_VERSION = None
    CONF_WIDGET_CLASS = None
    ADDITIONAL_CONF_OPTIONS = None
    ADDITIONAL_CONF_TABS = None
    CUSTOM_LAYOUTS = []
    IMG_PATH = None
    MONOSPACE_FONT_SIZE_DELTA = 0
    INTERFACE_FONT_SIZE_DELTA = 0
    CONTEXT_NAME = None
    CAN_BE_DISABLED = True
    sig_free_memory_requested = Signal()
    '\n    This signal can be emitted to request the main application to garbage\n    collect deleted objects.\n    '
    sig_plugin_ready = Signal()
    '\n    This signal can be emitted to reflect that the plugin was initialized.\n    '
    sig_quit_requested = Signal()
    '\n    This signal can be emitted to request the main application to quit.\n    '
    sig_restart_requested = Signal()
    '\n    This signal can be emitted to request the main application to restart.\n    '
    sig_status_message_requested = Signal(str, int)
    '\n    This signal can be emitted to request the main application to display a\n    message in the status bar.\n\n    Parameters\n    ----------\n    message: str\n        The actual message to display.\n    timeout: int\n        The timeout before the message disappears.\n    '
    sig_redirect_stdio_requested = Signal(bool)
    '\n    This signal can be emitted to request the main application to redirect\n    standard output/error when using Open/Save/Browse dialogs within widgets.\n\n    Parameters\n    ----------\n    enable: bool\n        Enable/Disable standard input/output redirection.\n    '
    sig_exception_occurred = Signal(dict)
    '\n    This signal can be emitted to report an exception from any plugin.\n\n    Parameters\n    ----------\n    error_data: dict\n        The dictionary containing error data. The expected keys are:\n        >>> error_data= {\n            "text": str,\n            "is_traceback": bool,\n            "repo": str,\n            "title": str,\n            "label": str,\n            "steps": str,\n        }\n\n    Notes\n    -----\n    The `is_traceback` key indicates if `text` contains plain text or a\n    Python error traceback.\n\n    The `title` and `repo` keys indicate how the error data should\n    customize the report dialog and Github error submission.\n\n    The `label` and `steps` keys allow customizing the content of the\n    error dialog.\n\n    This signal is automatically connected to the main container/widget.\n    '
    sig_mainwindow_resized = Signal('QResizeEvent')
    '\n    This signal is emitted when the main window is resized.\n\n    Parameters\n    ----------\n    resize_event: QResizeEvent\n        The event triggered on main window resize.\n\n    Notes\n    -----\n    To be used by plugins tracking main window size changes.\n    '
    sig_mainwindow_moved = Signal('QMoveEvent')
    '\n    This signal is emitted when the main window is moved.\n\n    Parameters\n    ----------\n    move_event: QMoveEvent\n        The event triggered on main window move.\n\n    Notes\n    -----\n    To be used by plugins tracking main window position changes.\n    '
    sig_unmaximize_plugin_requested = Signal((), (object,))
    '\n    This signal is emitted to inform the main window that it needs to\n    unmaximize the currently maximized plugin, if any.\n\n    Parameters\n    ----------\n    plugin_instance: SpyderDockablePlugin\n        Unmaximize plugin only if it is not `plugin_instance`.\n    '
    sig_mainwindow_state_changed = Signal(object)
    '\n    This signal is emitted when the main window state has changed (for\n    instance, between maximized and minimized states).\n\n    Parameters\n    ----------\n    window_state: Qt.WindowStates\n        The window state.\n    '
    _CONF_NAME_MAP = None

    def __init__(self, parent, configuration=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        SpyderPluginObserver.__init__(self)
        SpyderConfigurationObserver.__init__(self)
        self._main = parent
        self._widget = None
        self._conf = configuration
        self._plugin_path = os.path.dirname(inspect.getfile(self.__class__))
        self._container = None
        self._added_toolbars = OrderedDict()
        self._actions = {}
        self.is_compatible = None
        self.is_registered = None
        self.main = parent
        self.PLUGIN_NAME = self.NAME
        if self.CONTAINER_CLASS is not None:
            self._container = container = self.CONTAINER_CLASS(name=self.NAME, plugin=self, parent=parent)
            if isinstance(container, SpyderWidgetMixin):
                container.setup()
                container.update_actions()
            container.sig_free_memory_requested.connect(self.sig_free_memory_requested)
            container.sig_quit_requested.connect(self.sig_quit_requested)
            container.sig_restart_requested.connect(self.sig_restart_requested)
            container.sig_redirect_stdio_requested.connect(self.sig_redirect_stdio_requested)
            container.sig_exception_occurred.connect(self.sig_exception_occurred)
            container.sig_unmaximize_plugin_requested.connect(self.sig_unmaximize_plugin_requested)
            self.after_container_creation()
            if hasattr(container, '_setup'):
                container._setup()
        if self.IMG_PATH:
            plugin_path = osp.join(self.get_path(), self.IMG_PATH)
            IMAGE_PATH_MANAGER.add_image_path(plugin_path)

    def _register(self, omit_conf=False):
        if False:
            i = 10
            return i + 15
        "\n        Setup and register plugin in Spyder's main window and connect it to\n        other plugins.\n        "
        if self.NAME is None:
            raise SpyderAPIError('A Spyder Plugin must define a `NAME`!')
        if self._conf is not None and (not omit_conf):
            self._conf.register_plugin(self)
        self.is_registered = True
        self.update_font()

    def _unregister(self):
        if False:
            print('Hello World!')
        '\n        Disconnect signals and clean up the plugin to be able to stop it while\n        Spyder is running.\n        '
        if self._conf is not None:
            self._conf.unregister_plugin(self)
        self._container = None
        self.is_compatible = None
        self.is_registered = False

    def get_path(self):
        if False:
            i = 10
            return i + 15
        "\n        Return the plugin's system path.\n        "
        return self._plugin_path

    def get_container(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the plugin main container.\n        '
        return self._container

    def get_configuration(self):
        if False:
            return 10
        '\n        Return the Spyder configuration object.\n        '
        return self._conf

    def get_main(self):
        if False:
            i = 10
            return i + 15
        '\n        Return the Spyder main window..\n        '
        return self._main

    def get_plugin(self, plugin_name, error=True):
        if False:
            return 10
        "\n        Get a plugin instance by providing its name.\n\n        Parameters\n        ----------\n        plugin_name: str\n            Name of the plugin from which its instance will be returned.\n        error: bool\n            Whether to raise errors when trying to return the plugin's\n            instance.\n        "
        requires = set(self.REQUIRES or [])
        optional = set(self.OPTIONAL or [])
        full_set = requires | optional
        if plugin_name in full_set or Plugins.All in full_set:
            try:
                return self._main.get_plugin(plugin_name, error=error)
            except SpyderAPIError as e:
                if plugin_name in optional:
                    return None
                else:
                    raise e
        else:
            raise SpyderAPIError('Plugin "{}" not part of REQUIRES or OPTIONAL requirements!'.format(plugin_name))

    def is_plugin_enabled(self, plugin_name):
        if False:
            i = 10
            return i + 15
        'Determine if a given plugin is going to be loaded.'
        return self._main.is_plugin_enabled(plugin_name)

    def is_plugin_available(self, plugin_name):
        if False:
            i = 10
            return i + 15
        'Determine if a given plugin is available.'
        return self._main.is_plugin_available(plugin_name)

    def get_dockable_plugins(self):
        if False:
            i = 10
            return i + 15
        '\n        Return a list of the required plugin instances.\n\n        Only required plugins that extend SpyderDockablePlugin are returned.\n        '
        requires = set(self.REQUIRES or [])
        dockable_plugins_required = []
        for (name, plugin_instance) in self._main.get_dockable_plugins():
            if (name in requires or Plugins.All in requires) and isinstance(plugin_instance, (SpyderDockablePlugin, SpyderPluginWidget)):
                dockable_plugins_required.append(plugin_instance)
        return dockable_plugins_required

    def get_conf(self, option, default=NoDefault, section=None):
        if False:
            return 10
        '\n        Get an option from Spyder configuration system.\n\n        Parameters\n        ----------\n        option: str\n            Name of the option to get its value from.\n        default: bool, int, str, tuple, list, dict, NoDefault\n            Value to get from the configuration system, passed as a\n            Python object.\n        section: str\n            Section in the configuration system, e.g. `shortcuts`.\n\n        Returns\n        -------\n        bool, int, str, tuple, list, dict\n            Value associated with `option`.\n        '
        if self._conf is not None:
            section = self.CONF_SECTION if section is None else section
            if section is None:
                raise SpyderAPIError('A spyder plugin must define a `CONF_SECTION` class attribute!')
            return self._conf.get(section, option, default)

    @Slot(str, object)
    @Slot(str, object, str)
    def set_conf(self, option, value, section=None, recursive_notification=True):
        if False:
            for i in range(10):
                print('nop')
        "\n        Set an option in Spyder configuration system.\n\n        Parameters\n        ----------\n        option: str\n            Name of the option (e.g. 'case_sensitive')\n        value: bool, int, str, tuple, list, dict\n            Value to save in the configuration system, passed as a\n            Python object.\n        section: str\n            Section in the configuration system, e.g. `shortcuts`.\n        recursive_notification: bool\n            If True, all objects that observe all changes on the\n            configuration section and objects that observe partial tuple paths\n            are notified. For example if the option `opt` of section `sec`\n            changes, then the observers for section `sec` are notified.\n            Likewise, if the option `(a, b, c)` changes, then observers for\n            `(a, b, c)`, `(a, b)` and a are notified as well.\n        "
        if self._conf is not None:
            section = self.CONF_SECTION if section is None else section
            if section is None:
                raise SpyderAPIError('A spyder plugin must define a `CONF_SECTION` class attribute!')
            self._conf.set(section, option, value, recursive_notification=recursive_notification)
            self.apply_conf({option}, False)

    def remove_conf(self, option, section=None):
        if False:
            while True:
                i = 10
        '\n        Delete an option in the Spyder configuration system.\n\n        Parameters\n        ----------\n        option: Union[str, Tuple[str, ...]]\n            Name of the option, either a string or a tuple of strings.\n        section: str\n            Section in the configuration system.\n        '
        if self._conf is not None:
            section = self.CONF_SECTION if section is None else section
            if section is None:
                raise SpyderAPIError('A spyder plugin must define a `CONF_SECTION` class attribute!')
            self._conf.remove_option(section, option)
            self.apply_conf({option}, False)

    def apply_conf(self, options_set, notify=True):
        if False:
            while True:
                i = 10
        "\n        Apply `options_set` to this plugin's widget.\n        "
        if self._conf is not None and options_set:
            if notify:
                self.after_configuration_update(list(options_set))

    def disable_conf(self, option, section=None):
        if False:
            print('Hello World!')
        '\n        Disable notifications for an option in the Spyder configuration system.\n\n        Parameters\n        ----------\n        option: Union[str, Tuple[str, ...]]\n            Name of the option, either a string or a tuple of strings.\n        section: str\n            Section in the configuration system.\n        '
        if self._conf is not None:
            section = self.CONF_SECTION if section is None else section
            if section is None:
                raise SpyderAPIError('A spyder plugin must define a `CONF_SECTION` class attribute!')
            self._conf.disable_notifications(section, option)

    def restore_conf(self, option, section=None):
        if False:
            for i in range(10):
                print('nop')
        '\n        Restore notifications for an option in the Spyder configuration system.\n\n        Parameters\n        ----------\n        option: Union[str, Tuple[str, ...]]\n            Name of the option, either a string or a tuple of strings.\n        section: str\n            Section in the configuration system.\n        '
        if self._conf is not None:
            section = self.CONF_SECTION if section is None else section
            if section is None:
                raise SpyderAPIError('A spyder plugin must define a `CONF_SECTION` class attribute!')
            self._conf.restore_notifications(section, option)

    @Slot(str)
    @Slot(str, int)
    def show_status_message(self, message, timeout=0):
        if False:
            print('Hello World!')
        '\n        Show message in status bar.\n\n        Parameters\n        ----------\n        message: str\n            Message to display in the status bar.\n        timeout: int\n            Amount of time to display the message.\n        '
        self.sig_status_message_requested.emit(message, timeout)

    def before_long_process(self, message):
        if False:
            print('Hello World!')
        "\n        Show a message in main window's status bar and change the mouse\n        pointer to Qt.WaitCursor when starting a long process.\n\n        Parameters\n        ----------\n        message: str\n            Message to show in the status bar when the long process starts.\n        "
        if message:
            self.show_status_message(message)
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        QApplication.processEvents()

    def after_long_process(self, message=''):
        if False:
            return 10
        "\n        Clear main window's status bar after a long process and restore\n        mouse pointer to the OS deault.\n\n        Parameters\n        ----------\n        message: str\n            Message to show in the status bar when the long process finishes.\n        "
        QApplication.restoreOverrideCursor()
        self.show_status_message(message, timeout=2000)
        QApplication.processEvents()

    def get_color_scheme(self):
        if False:
            return 10
        '\n        Get the current color scheme.\n\n        Returns\n        -------\n        dict\n            Dictionary with properties and colors of the color scheme\n            used in the Editor.\n\n        Notes\n        -----\n        This is useful to set the color scheme of all instances of\n        CodeEditor used by the plugin.\n        '
        if self._conf is not None:
            return get_color_scheme(self._conf.get('appearance', 'selected'))

    def initialize(self):
        if False:
            print('Hello World!')
        '\n        Initialize a plugin instance.\n\n        Notes\n        -----\n        This method should be called to initialize the plugin, but it should\n        not be overridden, since it internally calls `on_initialize` and emits\n        the `sig_plugin_ready` signal.\n        '
        self.on_initialize()
        self.sig_plugin_ready.emit()

    @staticmethod
    def create_icon(name):
        if False:
            i = 10
            return i + 15
        '\n        Provide icons from the theme and icon manager.\n        '
        return ima.icon(name)

    @classmethod
    def get_font(cls, font_type):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return one of font types used in Spyder.\n\n        Parameters\n        ----------\n        font_type: str\n            There are three types of font types in Spyder:\n            SpyderFontType.Monospace, used in the Editor, IPython console,\n            and History; SpyderFontType.Interface, used by the entire Spyder\n            app; and SpyderFontType.MonospaceInterface, used by the Variable\n            Explorer, Find, Debugger and others.\n\n        Returns\n        -------\n        QFont\n            QFont object to be passed to other Qt widgets.\n\n        Notes\n        -----\n        All plugins in Spyder use the same, global fonts. In case some a plugin\n        wants to use a delta font size based on the default one, they can set\n        the MONOSPACE_FONT_SIZE_DELTA or INTERFACE_FONT_SIZE_DELTA class\n        constants.\n        '
        if font_type == SpyderFontType.Monospace:
            font_size_delta = cls.MONOSPACE_FONT_SIZE_DELTA
        elif font_type in [SpyderFontType.Interface, SpyderFontType.MonospaceInterface]:
            font_size_delta = cls.INTERFACE_FONT_SIZE_DELTA
        else:
            raise SpyderAPIError('Unrecognized font type')
        return get_font(option=font_type, font_size_delta=font_size_delta)

    def get_command_line_options(self):
        if False:
            i = 10
            return i + 15
        '\n        Get command line options passed by the user when they started\n        Spyder in a system terminal.\n\n        See app/cli_options.py for the option names.\n        '
        if self._main is not None:
            return self._main._cli_options
        else:
            sys_argv = [sys.argv[0]]
            return get_options(sys_argv)[0]

    @staticmethod
    def get_name():
        if False:
            i = 10
            return i + 15
        '\n        Return the plugin localized name.\n\n        Returns\n        -------\n        str\n            Localized name of the plugin.\n\n        Notes\n        -----\n        This method needs to be decorated with `staticmethod`.\n        '
        raise NotImplementedError('A plugin name must be defined!')

    @staticmethod
    def get_description():
        if False:
            i = 10
            return i + 15
        '\n        Return the plugin localized description.\n\n        Returns\n        -------\n        str\n            Localized description of the plugin.\n\n        Notes\n        -----\n        This method needs to be decorated with `staticmethod`.\n        '
        raise NotImplementedError('A plugin description must be defined!')

    @classmethod
    def get_icon(cls):
        if False:
            return 10
        '\n        Return the plugin associated icon.\n\n        Returns\n        -------\n        QIcon\n            QIcon instance\n\n        Notes\n        -----\n        This method needs to be decorated with `classmethod` or `staticmethod`.\n        '
        raise NotImplementedError('A plugin icon must be defined!')

    def on_initialize(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Setup the plugin.\n\n        Notes\n        -----\n        All calls performed on this method should not call other plugins.\n        '
        if hasattr(self, 'register'):
            raise SpyderAPIError('register was replaced by on_initialize, please check the Spyder 5.1.0 migration guide to get more information')
        raise NotImplementedError(f'The plugin {type(self)} is missing an implementation of on_initialize')

    @staticmethod
    def check_compatibility():
        if False:
            return 10
        "\n        This method can be reimplemented to check compatibility of a plugin\n        with the user's current environment.\n\n        Returns\n        -------\n        (bool, str)\n            The first value tells Spyder if the plugin has passed the\n            compatibility test defined in this method. The second value\n            is a message that must explain users why the plugin was\n            found to be incompatible (e.g. 'This plugin does not work\n            with PyQt4'). It will be shown at startup in a QMessageBox.\n        "
        valid = True
        message = ''
        return (valid, message)

    def on_first_registration(self):
        if False:
            print('Hello World!')
        '\n        Actions to be performed the first time the plugin is started.\n\n        It can also be used to perform actions that are needed only the\n        first time this is loaded after installation.\n\n        This method is called after the main window is visible.\n        '
        pass

    def before_mainwindow_visible(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Actions to be performed after setup but before the main window's has\n        been shown.\n        "
        pass

    def on_mainwindow_visible(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Actions to be performed after the main window's has been shown.\n        "
        pass

    def on_close(self, cancelable=False):
        if False:
            i = 10
            return i + 15
        '\n        Perform actions before the plugin is closed.\n\n        This method **must** only operate on local attributes and not other\n        plugins.\n        '
        if hasattr(self, 'unregister'):
            warnings.warn('The unregister method was deprecated and it was replaced by `on_close`. Please see the Spyder 5.2.0 migration guide to get more information.')

    def can_close(self) -> bool:
        if False:
            return 10
        '\n        Determine if a plugin can be closed.\n\n        Returns\n        -------\n        close: bool\n            True if the plugin can be closed, False otherwise.\n        '
        return True

    def update_font(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        This must be reimplemented by plugins that need to adjust their fonts.\n\n        The following plugins illustrate the usage of this method:\n          * spyder/plugins/help/plugin.py\n          * spyder/plugins/onlinehelp/plugin.py\n        '
        pass

    def update_style(self):
        if False:
            i = 10
            return i + 15
        '\n        This must be reimplemented by plugins that need to adjust their style.\n\n        Changing from the dark to the light interface theme might\n        require specific styles or stylesheets to be applied. When\n        the theme is changed by the user through our Preferences,\n        this method will be called for all plugins.\n        '
        pass

    def after_container_creation(self):
        if False:
            return 10
        '\n        Perform necessary operations before setting up the container.\n\n        This must be reimplemented by plugins whose containers emit signals in\n        on_option_update that need to be connected before applying those\n        options to our config system.\n        '
        pass

    def after_configuration_update(self, options: List[Union[str, tuple]]):
        if False:
            i = 10
            return i + 15
        '\n        Perform additional operations after updating the plugin configuration\n        values.\n\n        This can be implemented by plugins that do not have a container and\n        need to act on configuration updates.\n\n        Parameters\n        ----------\n        options: List[Union[str, tuple]]\n            A list that contains the options that were updated.\n        '
        pass

class SpyderDockablePlugin(SpyderPluginV2):
    """
    A Spyder plugin to enhance functionality with a dockable widget.
    """
    WIDGET_CLASS = None
    TABIFY = []
    DISABLE_ACTIONS_WHEN_HIDDEN = True
    RAISE_AND_FOCUS = False
    sig_focus_changed = Signal()
    '\n    This signal is emitted to inform the focus of this plugin has changed.\n    '
    sig_toggle_view_changed = Signal(bool)
    '\n    This action is emitted to inform the visibility of a dockable plugin\n    has changed.\n\n    This is triggered by checking/unchecking the entry for a pane in the\n    `View > Panes` menu.\n\n    Parameters\n    ----------\n    visible: bool\n        New visibility of the dockwidget.\n    '
    sig_switch_to_plugin_requested = Signal(object, bool)
    "\n    This signal can be emitted to inform the main window that this plugin\n    requested to be displayed.\n\n    Notes\n    -----\n    This is automatically connected to main container/widget at plugin's\n    registration.\n    "
    sig_update_ancestor_requested = Signal()
    '\n    This signal is emitted to inform the main window that a child widget\n    needs its ancestor to be updated.\n    '

    def __init__(self, parent, configuration):
        if False:
            while True:
                i = 10
        if not issubclass(self.WIDGET_CLASS, PluginMainWidget):
            raise SpyderAPIError('A SpyderDockablePlugin must define a valid WIDGET_CLASS attribute!')
        self.CONTAINER_CLASS = self.WIDGET_CLASS
        super().__init__(parent, configuration=configuration)
        self._shortcut = None
        self._widget = self._container
        widget = self._widget
        if widget is None:
            raise SpyderAPIError('A dockable plugin must define a WIDGET_CLASS!')
        if not isinstance(widget, PluginMainWidget):
            raise SpyderAPIError('The WIDGET_CLASS of a dockable plugin must be a subclass of PluginMainWidget!')
        widget.DISABLE_ACTIONS_WHEN_HIDDEN = self.DISABLE_ACTIONS_WHEN_HIDDEN
        widget.RAISE_AND_FOCUS = self.RAISE_AND_FOCUS
        widget.set_icon(self.get_icon())
        widget.set_name(self.NAME)
        widget.render_toolbars()
        widget.sig_toggle_view_changed.connect(self.sig_toggle_view_changed)
        widget.sig_update_ancestor_requested.connect(self.sig_update_ancestor_requested)

    def before_long_process(self, message):
        if False:
            while True:
                i = 10
        "\n        Show a message in main window's status bar, change the mouse pointer\n        to Qt.WaitCursor and start spinner when starting a long process.\n\n        Parameters\n        ----------\n        message: str\n            Message to show in the status bar when the long process starts.\n        "
        self.get_widget().start_spinner()
        super().before_long_process(message)

    def after_long_process(self, message=''):
        if False:
            for i in range(10):
                print('nop')
        "\n        Clear main window's status bar after a long process, restore mouse\n        pointer to the OS deault and stop spinner.\n\n        Parameters\n        ----------\n        message: str\n            Message to show in the status bar when the long process finishes.\n        "
        super().after_long_process(message)
        self.get_widget().stop_spinner()

    def get_widget(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Return the plugin main widget.\n        '
        if self._widget is None:
            raise SpyderAPIError('Dockable Plugin must have a WIDGET_CLASS!')
        return self._widget

    def update_title(self):
        if False:
            while True:
                i = 10
        '\n        Update plugin title, i.e. dockwidget or window title.\n        '
        self.get_widget().update_title()

    def update_margins(self, margin):
        if False:
            for i in range(10):
                print('nop')
        '\n        Update margins of main widget inside dockable plugin.\n        '
        self.get_widget().update_margins(margin)

    @Slot()
    def switch_to_plugin(self, force_focus=False):
        if False:
            i = 10
            return i + 15
        '\n        Switch to plugin and define if focus should be given or not.\n        '
        if self.get_widget().windowwidget is None:
            self.sig_switch_to_plugin_requested.emit(self, force_focus)

    def set_ancestor(self, ancestor_widget):
        if False:
            print('Hello World!')
        '\n        Update the ancestor/parent of child widgets when undocking.\n        '
        self.get_widget().set_ancestor(ancestor_widget)

    @property
    def dockwidget(self):
        if False:
            for i in range(10):
                print('nop')
        return self.get_widget().dockwidget

    @property
    def options_menu(self):
        if False:
            while True:
                i = 10
        return self.get_widget().get_options_menu()

    @property
    def toggle_view_action(self):
        if False:
            print('Hello World!')
        return self.get_widget().toggle_view_action

    def create_dockwidget(self, mainwindow):
        if False:
            print('Hello World!')
        return self.get_widget().create_dockwidget(mainwindow)

    def create_window(self):
        if False:
            i = 10
            return i + 15
        self.get_widget().create_window()

    def close_window(self, save_undocked=False):
        if False:
            i = 10
            return i + 15
        self.get_widget().close_window(save_undocked=save_undocked)

    def change_visibility(self, state, force_focus=False):
        if False:
            for i in range(10):
                print('nop')
        self.get_widget().change_visibility(state, force_focus)

    def toggle_view(self, value):
        if False:
            print('Hello World!')
        self.get_widget().toggle_view(value)