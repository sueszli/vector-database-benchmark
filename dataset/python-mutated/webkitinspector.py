"""Customized QWebInspector for QtWebKit."""
from qutebrowser.qt.webkit import QWebSettings
from qutebrowser.qt.webkitwidgets import QWebInspector, QWebPage
from qutebrowser.qt.widgets import QWidget
from qutebrowser.browser import inspector
from qutebrowser.misc import miscwidgets

class WebKitInspector(inspector.AbstractWebInspector):
    """A web inspector for QtWebKit."""

    def __init__(self, splitter: miscwidgets.InspectorSplitter, win_id: int, parent: QWidget=None) -> None:
        if False:
            print('Hello World!')
        super().__init__(splitter, win_id, parent)
        qwebinspector = QWebInspector()
        self._set_widget(qwebinspector)

    def inspect(self, page: QWebPage) -> None:
        if False:
            while True:
                i = 10
        settings = QWebSettings.globalSettings()
        settings.setAttribute(QWebSettings.WebAttribute.DeveloperExtrasEnabled, True)
        self._widget.setPage(page)