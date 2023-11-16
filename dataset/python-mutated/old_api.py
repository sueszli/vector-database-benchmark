"""
Old API for plugins.

These plugins were supported until Spyder 4, but they will be deprecated in
the future. Please don't rely on them for new plugins and use instead the
classes present in new_api.py
"""
from qtpy.QtCore import Signal, Slot
from qtpy.QtWidgets import QWidget
from spyder.config.user import NoDefault
from spyder.plugins.base import BasePluginMixin, BasePluginWidgetMixin
from spyder.utils.icon_manager import ima

class BasePlugin(BasePluginMixin):
    """
    Basic functionality for Spyder plugins.

    WARNING: Don't override any methods or attributes present here!
    """
    sig_show_status_message = Signal(str, int)
    sig_option_changed = Signal(str, object)

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super(BasePlugin, self).__init__(parent)
        self.main = parent
        self.PLUGIN_PATH = self._get_plugin_path()
        self.sig_show_status_message.connect(self.show_status_message)
        self.sig_option_changed.connect(self.set_option)

    @Slot(str)
    @Slot(str, int)
    def show_status_message(self, message, timeout=0):
        if False:
            for i in range(10):
                print('nop')
        "\n        Show message in main window's status bar.\n\n        Parameters\n        ----------\n        message: str\n            Message to display in the status bar.\n        timeout: int\n            Amount of time to display the message.\n        "
        super(BasePlugin, self)._show_status_message(message, timeout)

    @Slot(str, object)
    def set_option(self, option, value, section=None, recursive_notification=True):
        if False:
            print('Hello World!')
        "\n        Set an option in Spyder configuration file.\n\n        Parameters\n        ----------\n        option: str\n            Name of the option (e.g. 'case_sensitive')\n        value: bool, int, str, tuple, list, dict\n            Value to save in configuration file, passed as a Python\n            object.\n\n        Notes\n        -----\n        * Use sig_option_changed to call this method from widgets of the\n          same or another plugin.\n        * CONF_SECTION needs to be defined for this to work.\n        "
        super(BasePlugin, self)._set_option(option, value, section=section, recursive_notification=recursive_notification)

    def get_option(self, option, default=NoDefault, section=None):
        if False:
            print('Hello World!')
        '\n        Get an option from Spyder configuration file.\n\n        Parameters\n        ----------\n        option: str\n            Name of the option to get its value from.\n\n        Returns\n        -------\n        bool, int, str, tuple, list, dict\n            Value associated with `option`.\n        '
        return super(BasePlugin, self)._get_option(option, default, section=section)

    def remove_option(self, option, section=None):
        if False:
            while True:
                i = 10
        '\n        Remove an option from the Spyder configuration file.\n\n        Parameters\n        ----------\n        option: Union[str, Tuple[str, ...]]\n            A string or a Tuple of strings containing an option name to remove.\n        section: Optional[str]\n            Name of the section where the option belongs to.\n        '
        return super(BasePlugin, self)._remove_option(option, section=section)

    def starting_long_process(self, message):
        if False:
            while True:
                i = 10
        "\n        Show a message in main window's status bar and changes the\n        mouse to Qt.WaitCursor when starting a long process.\n\n        Parameters\n        ----------\n        message: str\n            Message to show in the status bar when the long\n            process starts.\n        "
        super(BasePlugin, self)._starting_long_process(message)

    def ending_long_process(self, message=''):
        if False:
            i = 10
            return i + 15
        "\n        Clear main window's status bar after a long process and restore\n        mouse to the OS deault.\n\n        Parameters\n        ----------\n        message: str\n            Message to show in the status bar when the long process\n            finishes.\n        "
        super(BasePlugin, self)._ending_long_process(message)

class SpyderPlugin(BasePlugin):
    """
    Spyder plugin class.

    All plugins *must* inherit this class and reimplement its interface.
    """
    CONF_SECTION = None
    DESCRIPTION = None
    CONFIGWIDGET_CLASS = None
    CONF_FILE = True
    CONF_DEFAULTS = None
    CONF_VERSION = None

    def check_compatibility(self):
        if False:
            while True:
                i = 10
        "\n        This method can be reimplemented to check compatibility of a\n        plugin for a given condition.\n\n        Returns\n        -------\n        (bool, str)\n            The first value tells Spyder if the plugin has passed the\n            compatibility test defined in this method. The second value\n            is a message that must explain users why the plugin was\n            found to be incompatible (e.g. 'This plugin does not work\n            with PyQt4'). It will be shown at startup in a QMessageBox.\n        "
        message = ''
        valid = True
        return (valid, message)

class BasePluginWidget(QWidget, BasePluginWidgetMixin):
    """
    Basic functionality for Spyder plugin widgets.

    WARNING: Don't override any methods or attributes present here!
    """
    sig_update_plugin_title = Signal()

    def __init__(self, main=None):
        if False:
            i = 10
            return i + 15
        QWidget.__init__(self)
        BasePluginWidgetMixin.__init__(self, main)
        self.dockwidget = None

    def add_dockwidget(self):
        if False:
            print('Hello World!')
        "Add the plugin's QDockWidget to the main window."
        super(BasePluginWidget, self)._add_dockwidget()

    def tabify(self, core_plugin):
        if False:
            return 10
        '\n        Tabify plugin next to one of the core plugins.\n\n        Parameters\n        ----------\n        core_plugin: SpyderPluginWidget\n            Core Spyder plugin this one will be tabified next to.\n\n        Examples\n        --------\n        >>> self.tabify(self.main.variableexplorer)\n        >>> self.tabify(self.main.ipyconsole)\n\n        Notes\n        -----\n        The names of variables associated with each of the core plugins\n        can be found in the `setup` method of `MainWindow`, present in\n        `spyder/app/mainwindow.py`.\n        '
        super(BasePluginWidget, self)._tabify(core_plugin)

    def get_font(self, rich_text=False):
        if False:
            while True:
                i = 10
        '\n        Return plain or rich text font used in Spyder.\n\n        Parameters\n        ----------\n        rich_text: bool\n            Return rich text font (i.e. the one used in the Help pane)\n            or plain text one (i.e. the one used in the Editor).\n\n        Returns\n        -------\n        QFont:\n            QFont object to be passed to other Qt widgets.\n\n        Notes\n        -----\n        All plugins in Spyder use the same, global font. This is a\n        convenience method in case some plugins want to use a delta\n        size based on the default one. That can be controlled by using\n        FONT_SIZE_DELTA or RICH_FONT_SIZE_DELTA (declared below in\n        `SpyderPluginWidget`).\n        '
        return super(BasePluginWidget, self)._get_font(rich_text)

    def register_shortcut(self, qaction_or_qshortcut, context, name, add_shortcut_to_tip=False):
        if False:
            print('Hello World!')
        "\n        Register a shortcut associated to a QAction or a QShortcut to\n        Spyder main application.\n\n        Parameters\n        ----------\n        qaction_or_qshortcut: QAction or QShortcut\n            QAction to register the shortcut for or QShortcut.\n        context: str\n            Name of the plugin this shortcut applies to. For instance,\n            if you pass 'Editor' as context, the shortcut will only\n            work when the editor is focused.\n            Note: You can use '_' if you want the shortcut to be work\n            for the entire application.\n        name: str\n            Name of the action the shortcut refers to (e.g. 'Debug\n            exit').\n        add_shortcut_to_tip: bool\n            If True, the shortcut is added to the action's tooltip.\n            This is useful if the action is added to a toolbar and\n            users hover it to see what it does.\n        "
        self.main.register_shortcut(qaction_or_qshortcut, context, name, add_shortcut_to_tip, self.CONF_SECTION)

    def unregister_shortcut(self, qaction_or_qshortcut, context, name, add_shortcut_to_tip=False):
        if False:
            i = 10
            return i + 15
        "\n        Unregister a shortcut associated to a QAction or a QShortcut to\n        Spyder main application.\n\n        Parameters\n        ----------\n        qaction_or_qshortcut: QAction or QShortcut\n            QAction to register the shortcut for or QShortcut.\n        context: str\n            Name of the plugin this shortcut applies to. For instance,\n            if you pass 'Editor' as context, the shortcut will only\n            work when the editor is focused.\n            Note: You can use '_' if you want the shortcut to be work\n            for the entire application.\n        name: str\n            Name of the action the shortcut refers to (e.g. 'Debug\n            exit').\n        add_shortcut_to_tip: bool\n            If True, the shortcut is added to the action's tooltip.\n            This is useful if the action is added to a toolbar and\n            users hover it to see what it does.\n        "
        self.main.unregister_shortcut(qaction_or_qshortcut, context, name, add_shortcut_to_tip, self.CONF_SECTION)

    def register_widget_shortcuts(self, widget):
        if False:
            i = 10
            return i + 15
        "\n        Register shortcuts defined by a plugin's widget so they take\n        effect when the plugin is focused.\n\n        Parameters\n        ----------\n        widget: QWidget\n            Widget to register shortcuts for.\n\n        Notes\n        -----\n        The widget interface must have a method called\n        `get_shortcut_data` for this to work. Please see\n        `spyder/widgets/findreplace.py` for an example.\n        "
        for (qshortcut, context, name) in widget.get_shortcut_data():
            self.register_shortcut(qshortcut, context, name)

    def unregister_widget_shortcuts(self, widget):
        if False:
            i = 10
            return i + 15
        "\n        Unregister shortcuts defined by a plugin's widget.\n\n        Parameters\n        ----------\n        widget: QWidget\n            Widget to register shortcuts for.\n\n        Notes\n        -----\n        The widget interface must have a method called\n        `get_shortcut_data` for this to work. Please see\n        `spyder/widgets/findreplace.py` for an example.\n        "
        for (qshortcut, context, name) in widget.get_shortcut_data():
            self.unregister_shortcut(qshortcut, context, name)

    def get_color_scheme(self):
        if False:
            i = 10
            return i + 15
        '\n        Get the current color scheme.\n\n        Returns\n        -------\n        dict\n            Dictionary with properties and colors of the color scheme\n            used in the Editor.\n\n        Notes\n        -----\n        This is useful to set the color scheme of all instances of\n        CodeEditor used by the plugin.\n        '
        return super(BasePluginWidget, self)._get_color_scheme()

    def switch_to_plugin(self):
        if False:
            while True:
                i = 10
        "\n        Switch to this plugin.\n\n        Notes\n        -----\n        This operation unmaximizes the current plugin (if any), raises\n        this plugin to view (if it's hidden) and gives it focus (if\n        possible).\n        "
        super(BasePluginWidget, self)._switch_to_plugin()

class SpyderPluginWidget(SpyderPlugin, BasePluginWidget):
    """
    Spyder plugin widget class.

    All plugin widgets *must* inherit this class and reimplement its interface.
    """
    IMG_PATH = 'images'
    FONT_SIZE_DELTA = 0
    RICH_FONT_SIZE_DELTA = 0
    DISABLE_ACTIONS_WHEN_HIDDEN = True
    shortcut = None

    def get_plugin_title(self):
        if False:
            i = 10
            return i + 15
        "\n        Get plugin's title.\n\n        Returns\n        -------\n        str\n            Name of the plugin.\n        "
        raise NotImplementedError

    def get_plugin_icon(self):
        if False:
            i = 10
            return i + 15
        "\n        Get plugin's associated icon.\n\n        Returns\n        -------\n        QIcon\n            QIcon instance\n        "
        return ima.icon('outline_explorer')

    def get_focus_widget(self):
        if False:
            print('Hello World!')
        "\n        Get the plugin widget to give focus to.\n\n        Returns\n        -------\n        QWidget\n            QWidget to give focus to.\n\n        Notes\n        -----\n        This is applied when plugin's dockwidget is raised on top-level.\n        "
        pass

    def closing_plugin(self, cancelable=False):
        if False:
            for i in range(10):
                print('nop')
        '\n        Perform actions before the main window is closed.\n\n        Returns\n        -------\n        bool\n            Whether the plugin may be closed immediately or not.\n\n        Notes\n        -----\n        The returned value is ignored if *cancelable* is False.\n        '
        return True

    def refresh_plugin(self):
        if False:
            while True:
                i = 10
        '\n        Refresh plugin after it receives focus.\n\n        Notes\n        -----\n        For instance, this is used to maintain in sync the Variable\n        Explorer with the currently focused IPython console.\n        '
        pass

    def get_plugin_actions(self):
        if False:
            print('Hello World!')
        "\n        Return a list of QAction's related to plugin.\n\n        Notes\n        -----\n        These actions will be shown in the plugins Options menu (i.e.\n        the hambuger menu on the right of each plugin).\n        "
        return []

    def register_plugin(self):
        if False:
            for i in range(10):
                print('nop')
        "\n        Register plugin in Spyder's main window and connect it to other\n        plugins.\n\n        Notes\n        -----\n        Below is the minimal call necessary to register the plugin. If\n        you override this method, please don't forget to make that call\n        here too.\n        "
        self.add_dockwidget()

    def on_first_registration(self):
        if False:
            i = 10
            return i + 15
        '\n        Action to be performed on first plugin registration.\n\n        Notes\n        -----\n        This is mostly used to tabify the plugin next to one of the\n        core plugins, like this:\n\n        self.tabify(self.main.variableexplorer)\n        '
        raise NotImplementedError

    def apply_plugin_settings(self, options):
        if False:
            for i in range(10):
                print('nop')
        '\n        Determine what to do to apply configuration plugin settings.\n        '
        pass

    def update_font(self):
        if False:
            return 10
        '\n        This must be reimplemented by plugins that need to adjust\n        their fonts.\n        '
        pass

    def toggle_view(self, checked):
        if False:
            return 10
        "\n        Toggle dockwidget's visibility when its entry is selected in\n        the menu `View > Panes`.\n\n        Parameters\n        ----------\n        checked: bool\n            Is the entry in `View > Panes` checked or not?\n\n        Notes\n        -----\n        Redefining this method can be useful to execute certain actions\n        when the plugin is made visible. For an example, please see\n        `spyder/plugins/ipythonconsole/plugin.py`\n        "
        if not self.dockwidget:
            return
        if checked:
            self.dockwidget.show()
            self.dockwidget.raise_()
        else:
            self.dockwidget.hide()

    def set_ancestor(self, ancestor):
        if False:
            print('Hello World!')
        '\n        Needed to update the ancestor/parent of child widgets when undocking.\n        '
        pass