"""A filter for signals which either filters or passes them."""
import functools
from qutebrowser.qt.core import QObject
from qutebrowser.utils import debug, log, objreg

class SignalFilter(QObject):
    """A filter for signals.

    Signals are only passed to the parent TabbedBrowser if they originated in
    the currently shown widget.

    Attributes:
        _win_id: The window ID this SignalFilter is associated with.

    Class attributes:
        BLACKLIST: List of signal names which should not be logged.
    """
    BLACKLIST = {'cur_scroll_perc_changed', 'cur_progress', 'cur_link_hovered'}

    def __init__(self, win_id, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._win_id = win_id

    def create(self, signal, tab):
        if False:
            return 10
        'Factory for partial _filter_signals functions.\n\n        Args:\n            signal: The pyqtBoundSignal to filter.\n            tab: The WebView to create filters for.\n\n        Return:\n            A partial function calling _filter_signals with a signal.\n        '
        log_signal = debug.signal_name(signal) not in self.BLACKLIST
        return functools.partial(self._filter_signals, signal=signal, log_signal=log_signal, tab=tab)

    def _filter_signals(self, *args, signal, log_signal, tab):
        if False:
            print('Hello World!')
        'Filter signals and trigger TabbedBrowser signals if needed.\n\n        Triggers signal if the original signal was sent from the _current_ tab\n        and not from any other one.\n\n        The original signal does not matter, since we get the new signal and\n        all args.\n\n        Args:\n            signal: The signal to emit if the sender was the current widget.\n            tab: The WebView which the filter belongs to.\n            *args: The args to pass to the signal.\n        '
        tabbed_browser = objreg.get('tabbed-browser', scope='window', window=self._win_id)
        try:
            tabidx = tabbed_browser.widget.indexOf(tab)
        except RuntimeError:
            return
        if tabidx == tabbed_browser.widget.currentIndex():
            if log_signal:
                log.signals.debug('emitting: {} (tab {})'.format(debug.dbg_signal(signal, args), tabidx))
            signal.emit(*args)
        elif log_signal:
            log.signals.debug('ignoring: {} (tab {})'.format(debug.dbg_signal(signal, args), tabidx))