"""
Spyder API auxiliary widgets.
"""
from qtpy.QtCore import QEvent, QSize, Signal
from qtpy.QtWidgets import QMainWindow, QSizePolicy, QToolBar, QWidget
from spyder.api.exceptions import SpyderAPIError
from spyder.api.widgets.mixins import SpyderMainWindowMixin
from spyder.utils.stylesheet import APP_STYLESHEET

class SpyderWindowWidget(QMainWindow, SpyderMainWindowMixin):
    """MainWindow subclass that contains a SpyderDockablePlugin."""
    sig_closed = Signal()
    'This signal is emitted when the close event is fired.'
    sig_window_state_changed = Signal(object)
    '\n    This signal is emitted when the window state has changed (for instance,\n    between maximized and minimized states).\n\n    Parameters\n    ----------\n    window_state: Qt.WindowStates\n        The window state.\n    '

    def __init__(self, widget):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.widget = widget
        self.is_window_widget = True
        self.setStyleSheet(str(APP_STYLESHEET))

    def closeEvent(self, event):
        if False:
            while True:
                i = 10
        'Override Qt method to emit a custom `sig_close` signal.'
        super().closeEvent(event)
        self.sig_closed.emit()

    def changeEvent(self, event):
        if False:
            return 10
        "\n        Override Qt method to emit a custom `sig_windowstate_changed` signal\n        when there's a change in the window state.\n        "
        if event.type() == QEvent.WindowStateChange:
            self.sig_window_state_changed.emit(self.windowState())
        super().changeEvent(event)

class MainCornerWidget(QToolBar):
    """
    Corner widget to hold options menu, spinner and additional options.
    """

    def __init__(self, parent, name):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self._icon_size = QSize(16, 16)
        self.setIconSize(self._icon_size)
        self._widgets = {}
        self._actions = []
        self.setObjectName(name)
        self._strut = QWidget()
        self._strut.setFixedWidth(0)
        self._strut.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.addWidget(self._strut)

    def add_widget(self, widget_id, widget):
        if False:
            return 10
        '\n        Add a widget to the left of the last widget added to the corner.\n        '
        if widget_id in self._widgets:
            raise SpyderAPIError('Wigdet with name "{}" already added. Current names are: {}'.format(widget_id, list(self._widgets.keys())))
        widget.ID = widget_id
        self._widgets[widget_id] = widget
        self._actions.append(self.addWidget(widget))

    def get_widget(self, widget_id):
        if False:
            return 10
        'Return a widget by unique id.'
        if widget_id in self._widgets:
            return self._widgets[widget_id]