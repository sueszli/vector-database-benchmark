"""Global Qt event filter which dispatches key events."""
from typing import cast, Optional
from qutebrowser.qt.core import pyqtSlot, QObject, QEvent, qVersion
from qutebrowser.qt.gui import QKeyEvent, QWindow
from qutebrowser.keyinput import modeman
from qutebrowser.misc import quitter, objects
from qutebrowser.utils import objreg, debug, log, qtutils

class EventFilter(QObject):
    """Global Qt event filter.

    Attributes:
        _activated: Whether the EventFilter is currently active.
        _handlers: A {QEvent.Type: callable} dict with the handlers for an
                   event.
    """

    def __init__(self, parent: QObject=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._activated = True
        self._handlers = {QEvent.Type.KeyPress: self._handle_key_event, QEvent.Type.KeyRelease: self._handle_key_event, QEvent.Type.ShortcutOverride: self._handle_key_event}
        self._log_qt_events = 'log-qt-events' in objects.debug_flags

    def install(self) -> None:
        if False:
            while True:
                i = 10
        objects.qapp.installEventFilter(self)

    @pyqtSlot()
    def shutdown(self) -> None:
        if False:
            i = 10
            return i + 15
        objects.qapp.removeEventFilter(self)

    def _handle_key_event(self, event: QKeyEvent) -> bool:
        if False:
            return 10
        "Handle a key press/release event.\n\n        Args:\n            event: The QEvent which is about to be delivered.\n\n        Return:\n            True if the event should be filtered, False if it's passed through.\n        "
        active_window = objects.qapp.activeWindow()
        if active_window not in objreg.window_registry.values():
            return False
        try:
            man = modeman.instance('current')
            return man.handle_event(event)
        except objreg.RegistryUnavailableError:
            return False

    def eventFilter(self, obj: Optional[QObject], event: Optional[QEvent]) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Handle an event.\n\n        Args:\n            obj: The object which will get the event.\n            event: The QEvent which is about to be delivered.\n\n        Return:\n            True if the event should be filtered, False if it's passed through.\n        "
        assert event is not None
        ev_type = event.type()
        if self._log_qt_events:
            try:
                source = repr(obj)
            except AttributeError:
                source = type(obj).__name__
            ev_type_str = debug.qenum_key(QEvent, ev_type)
            log.misc.debug(f'{source} got event: {ev_type_str}')
        if ev_type == QEvent.Type.DragEnter and qtutils.is_wayland() and (qVersion() == '6.5.2'):
            log.mouse.warning('Ignoring drag event to prevent Qt crash')
            event.ignore()
            return True
        if not isinstance(obj, QWindow):
            return False
        if ev_type not in self._handlers:
            return False
        if not self._activated:
            return False
        handler = self._handlers[ev_type]
        try:
            return handler(cast(QKeyEvent, event))
        except:
            self._activated = False
            raise

def init() -> None:
    if False:
        return 10
    'Initialize the global EventFilter instance.'
    event_filter = EventFilter(parent=objects.qapp)
    event_filter.install()
    quitter.instance.shutting_down.connect(event_filter.shutdown)