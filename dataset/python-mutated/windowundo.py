"""Code for :undo --window."""
import collections
import dataclasses
from typing import MutableSequence, cast, TYPE_CHECKING
from qutebrowser.qt.core import QObject, QByteArray
from qutebrowser.config import config
from qutebrowser.mainwindow import mainwindow
from qutebrowser.misc import objects
if TYPE_CHECKING:
    from qutebrowser.mainwindow import tabbedbrowser
instance = cast('WindowUndoManager', None)

@dataclasses.dataclass
class _WindowUndoEntry:
    """Information needed for :undo -w."""
    geometry: QByteArray
    tab_stack: 'tabbedbrowser.UndoStackType'

class WindowUndoManager(QObject):
    """Manager which saves/restores windows."""

    def __init__(self, parent=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)
        self._undos: MutableSequence[_WindowUndoEntry] = collections.deque()
        objects.qapp.window_closing.connect(self._on_window_closing)
        config.instance.changed.connect(self._on_config_changed)

    @config.change_filter('tabs.undo_stack_size')
    def _on_config_changed(self):
        if False:
            print('Hello World!')
        self._update_undo_stack_size()

    def _on_window_closing(self, window):
        if False:
            for i in range(10):
                print('nop')
        if window.tabbed_browser.is_private:
            return
        self._undos.append(_WindowUndoEntry(geometry=window.saveGeometry(), tab_stack=window.tabbed_browser.undo_stack))

    def _update_undo_stack_size(self):
        if False:
            return 10
        newsize = config.instance.get('tabs.undo_stack_size')
        if newsize < 0:
            newsize = None
        self._undos = collections.deque(self._undos, maxlen=newsize)

    def undo_last_window_close(self):
        if False:
            return 10
        'Restore the last window to be closed.\n\n        It will have the same tab and undo stack as when it was closed.\n        '
        entry = self._undos.pop()
        window = mainwindow.MainWindow(private=False, geometry=entry.geometry)
        window.tabbed_browser.undo_stack = entry.tab_stack
        window.tabbed_browser.undo()
        window.show()

def init():
    if False:
        while True:
            i = 10
    global instance
    instance = WindowUndoManager(parent=objects.qapp)