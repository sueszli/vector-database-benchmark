"""QtWebKit specific part of history."""
import functools
from qutebrowser.qt.webkit import QWebHistoryInterface
from qutebrowser.utils import debug
from qutebrowser.misc import debugcachestats

class WebHistoryInterface(QWebHistoryInterface):
    """Glue code between WebHistory and Qt's QWebHistoryInterface.

    Attributes:
        _history: The WebHistory object.
    """

    def __init__(self, webhistory, parent=None):
        if False:
            for i in range(10):
                print('nop')
        super().__init__(parent)
        self._history = webhistory
        self._history.changed.connect(self.historyContains.cache_clear)

    def addHistoryEntry(self, url_string):
        if False:
            return 10
        'Required for a QWebHistoryInterface impl, obsoleted by add_url.'

    @debugcachestats.register(name='history')
    @functools.lru_cache(maxsize=32768)
    def historyContains(self, url_string):
        if False:
            print('Hello World!')
        'Called by WebKit to determine if a URL is contained in the history.\n\n        Args:\n            url_string: The URL (as string) to check for.\n\n        Return:\n            True if the url is in the history, False otherwise.\n        '
        with debug.log_time('sql', 'historyContains'):
            return url_string in self._history

def init(history):
    if False:
        print('Hello World!')
    'Initialize the QWebHistoryInterface.\n\n    Args:\n        history: The WebHistory object.\n    '
    interface = WebHistoryInterface(history, parent=history)
    QWebHistoryInterface.setDefaultInterface(interface)