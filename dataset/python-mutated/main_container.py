"""
Main container widget.

SpyderPluginV2 plugins must provide a CONTAINER_CLASS attribute that is a
subclass of PluginMainContainer, if they provide additional widgets like
status bar widgets or toolbars.
"""
from qtpy import PYQT5
from qtpy.QtCore import Signal
from qtpy.QtWidgets import QWidget
from spyder.api.widgets.mixins import SpyderToolbarMixin, SpyderWidgetMixin

class PluginMainContainer(QWidget, SpyderWidgetMixin, SpyderToolbarMixin):
    """
    Spyder plugin main container class.

    This class handles a non-dockable widget to be able to contain, parent and
    store references to other widgets, like status bar widgets, toolbars,
    context menus, etc.

    Notes
    -----
    All Spyder non dockable plugins can define a plugin container that must
    subclass this.
    """
    CONTEXT_NAME = None
    '\n    This optional attribute defines the context name under which actions,\n    toolbars, toolbuttons and menus should be registered on the\n    Spyder global registry.\n\n    If actions, toolbars, toolbuttons or menus belong to the global scope of\n    the plugin, then this attribute should have a `None` value.\n    '
    sig_free_memory_requested = Signal()
    '\n    This signal can be emitted to request the main application to garbage\n    collect deleted objects.\n    '
    sig_quit_requested = Signal()
    '\n    This signal can be emitted to request the main application to quit.\n    '
    sig_restart_requested = Signal()
    '\n    This signal can be emitted to request the main application to restart.\n    '
    sig_redirect_stdio_requested = Signal(bool)
    '\n    This signal can be emitted to request the main application to redirect\n    standard output/error when using Open/Save/Browse dialogs within widgets.\n\n    Parameters\n    ----------\n    enable: bool\n        Enable/Disable standard input/output redirection.\n    '
    sig_exception_occurred = Signal(dict)
    '\n    This signal can be emitted to report an exception handled by this widget.\n\n    Parameters\n    ----------\n    error_data: dict\n        The dictionary containing error data. The expected keys are:\n        >>> error_data= {\n            "text": str,\n            "is_traceback": bool,\n            "repo": str,\n            "title": str,\n            "label": str,\n            "steps": str,\n        }\n\n    Notes\n    -----\n    The `is_traceback` key indicates if `text` contains plain text or a\n    Python error traceback.\n\n    The `title` and `repo` keys indicate how the error data should\n    customize the report dialog and Github error submission.\n\n    The `label` and `steps` keys allow customizing the content of the\n    error dialog.\n    '
    sig_unmaximize_plugin_requested = Signal((), (object,))
    '\n    This signal is emitted to inform the main window that it needs to\n    unmaximize the currently maximized plugin, if any.\n\n    Parameters\n    ----------\n    plugin_instance: SpyderDockablePlugin\n        Unmaximize plugin only if it is not `plugin_instance`.\n    '

    def __init__(self, name, plugin, parent=None):
        if False:
            print('Hello World!')
        if PYQT5:
            super().__init__(parent=parent, class_parent=plugin)
        else:
            QWidget.__init__(self, parent)
            SpyderWidgetMixin.__init__(self, class_parent=plugin)
        self._name = name
        self._plugin = plugin
        self._parent = parent
        self.PLUGIN_NAME = name
        self.setMaximumWidth(0)
        self.setMaximumHeight(0)

    def closeEvent(self, event):
        if False:
            for i in range(10):
                print('nop')
        self.on_close()
        super().closeEvent(event)

    def setup(self):
        if False:
            i = 10
            return i + 15
        '\n        Create actions, widgets, add to menu and other setup requirements.\n        '
        raise NotImplementedError('A PluginMainContainer subclass must define a `setup` method!')

    def update_actions(self):
        if False:
            i = 10
            return i + 15
        '\n        Update the state of exposed actions.\n\n        Exposed actions are actions created by the self.create_action method.\n        '
        raise NotImplementedError('A PluginMainContainer subclass must define a `update_actions` method!')

    def on_close(self):
        if False:
            i = 10
            return i + 15
        '\n        Perform actions before the container is closed.\n\n        This method **must** only operate on local attributes.\n        '
        pass