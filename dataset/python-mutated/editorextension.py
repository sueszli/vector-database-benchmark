"""
This module contains the editor extension API.

Adapted from pyqode/core/api/mode.py of the
`PyQode project <https://github.com/pyQode/pyQode>`_.
Original file:
<https://github.com/pyQode/pyqode.core/blob/master/pyqode/core/api/mode.py>
"""
import logging
logger = logging.getLogger(__name__)

class EditorExtension(object):
    """
    Base class for editor extensions.

    An extension is a "thing" that can be installed on an editor to add new
    behaviours or to modify its appearance.

    A panel (model child class) is added to an editor by using the
    PanelsManager:
        - :meth:
            `spyder.plugins.editor.widgets.codeeditor.CodeEditor.panels.append`

    Subclasses may/should override the following methods:

        - :meth:`spyder.api.EditorExtension.on_install`
        - :meth:`spyder.api.EditorExtension.on_uninstall`
        - :meth:`spyder.api.EditorExtension.on_state_changed`

    ..warning: The editor extension will be identified by its class name, this
    means that **there cannot be two editor extensions of the same type on the
    same editor instance!**
    """

    @property
    def editor(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Returns a reference to the parent code editor widget.\n\n        **READ ONLY**\n\n        :rtype: spyder.plugins.editor.widgets.codeeditor.CodeEditor\n        '
        return self._editor

    @property
    def enabled(self):
        if False:
            for i in range(10):
                print('nop')
        '\n        Tells if the editor extension is enabled.\n\n        :meth:`spyder.api.EditorExtension.on_state_changed` will be called as\n        soon as the editor extension state changed.\n\n        :type: bool\n        '
        return self._enabled

    @enabled.setter
    def enabled(self, enabled):
        if False:
            print('Hello World!')
        if enabled != self._enabled:
            self._enabled = enabled
            self.on_state_changed(enabled)

    def __init__(self):
        if False:
            print('Hello World!')
        '\n        EditorExtension name/identifier.\n        :class:`spyder.widgets.sourcecode.CodeEditor` uses that as the\n        attribute key when you install a editor extension.\n        '
        self.name = self.__class__.__name__
        self.description = self.__doc__
        self._enabled = False
        self._editor = None
        self._on_close = False

    def __del__(self):
        if False:
            for i in range(10):
                print('nop')
        logger.debug('%s.__del__', type(self))

    def on_install(self, editor):
        if False:
            for i in range(10):
                print('nop')
        "\n        Installs the extension on the editor.\n\n        :param editor: editor widget instance\n        :type editor: spyder.plugins.editor.widgets.codeeditor.CodeEditor\n\n        .. note:: This method is called by editor when you install a\n                  EditorExtension.\n                  You should never call it yourself, even in a subclasss.\n\n        .. warning:: Don't forget to call **super** when subclassing\n        "
        self._editor = editor
        self.enabled = True

    def on_uninstall(self):
        if False:
            return 10
        'Uninstalls the editor extension from the editor.'
        self._on_close = True
        self.enabled = False
        self._editor = None

    def on_state_changed(self, state):
        if False:
            return 10
        "\n        Called when the enable state has changed.\n\n        This method does not do anything, you may override it if you need\n        to connect/disconnect to the editor's signals (connect when state is\n        true and disconnect when it is false).\n\n        :param state: True = enabled, False = disabled\n        :type state: bool\n        "
        pass

    def clone_settings(self, original):
        if False:
            return 10
        '\n        Clone the settings from another editor extension (same class).\n\n        This method is called when splitting an editor widget.\n        # TODO at the current estate this is not working\n\n        :param original: other editor extension (must be the same class).\n\n        .. note:: The base method does not do anything, you must implement\n            this method for every new editor extension/panel (if you plan on\n            using the split feature). You should also make sure any properties\n            will be propagated to the clones.\n        '
        pass