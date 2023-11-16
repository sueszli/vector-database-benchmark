"""Customized QWebInspector for QtWebEngine."""
from typing import Optional
from qutebrowser.qt import machinery
from qutebrowser.qt.webenginewidgets import QWebEngineView
from qutebrowser.qt.webenginecore import QWebEnginePage
from qutebrowser.qt.widgets import QWidget
from qutebrowser.browser import inspector
from qutebrowser.browser.webengine import webenginesettings, webview
from qutebrowser.misc import miscwidgets
from qutebrowser.utils import version, usertypes, qtutils
from qutebrowser.keyinput import modeman

class WebEngineInspectorView(QWebEngineView):
    """The QWebEngineView used for the inspector.

    We don't use a qutebrowser WebEngineView because that has various
    customization which doesn't apply to the inspector.
    """

    def createWindow(self, wintype: QWebEnginePage.WebWindowType) -> QWebEngineView:
        if False:
            i = 10
            return i + 15
        'Called by Qt when a page wants to create a new tab or window.\n\n        In case the user wants to open a resource in a new tab, we use the\n        createWindow handling of the main page to achieve that.\n\n        See WebEngineView.createWindow for details.\n        '
        our_page = self.page()
        assert our_page is not None
        inspected_page = our_page.inspectedPage()
        assert inspected_page is not None
        if machinery.IS_QT5:
            view = inspected_page.view()
            assert isinstance(view, QWebEngineView), view
            return view.createWindow(wintype)
        else:
            newpage = inspected_page.createWindow(wintype)
            ret = webview.WebEngineView.forPage(newpage)
            assert ret is not None
            return ret

class WebEngineInspector(inspector.AbstractWebInspector):
    """A web inspector for QtWebEngine with Qt API support."""
    _widget: WebEngineInspectorView

    def __init__(self, splitter: miscwidgets.InspectorSplitter, win_id: int, parent: QWidget=None) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(splitter, win_id, parent)
        self._check_devtools_resources()
        self._settings: Optional[webenginesettings.WebEngineSettings] = None

    def _on_window_close_requested(self) -> None:
        if False:
            return 10
        "Called when the 'x' was clicked in the devtools."
        modeman.leave(self._win_id, usertypes.KeyMode.insert, 'devtools close requested', maybe=True)
        self.hide()

    def _check_devtools_resources(self) -> None:
        if False:
            i = 10
            return i + 15
        "Make sure that the devtools resources are available on Fedora.\n\n        Fedora packages devtools resources into its own package. If it's not\n        installed, we show a nice error instead of a blank inspector.\n        "
        dist = version.distribution()
        if dist is None or dist.parsed != version.Distribution.fedora:
            return
        data_path = qtutils.library_path(qtutils.LibraryPath.data)
        pak = data_path / 'resources' / 'qtwebengine_devtools_resources.pak'
        if not pak.exists():
            raise inspector.Error('QtWebEngine devtools resources not found, please install the qt5-qtwebengine-devtools Fedora package.')

    def inspect(self, page: QWebEnginePage) -> None:
        if False:
            return 10
        if not self._widget:
            view = WebEngineInspectorView()
            new_page = QWebEnginePage(page.profile(), self)
            new_page.windowCloseRequested.connect(self._on_window_close_requested)
            view.setPage(new_page)
            self._settings = webenginesettings.WebEngineSettings(view.settings())
            self._set_widget(view)
        inspector_page = self._widget.page()
        assert inspector_page is not None
        assert inspector_page.profile() == page.profile()
        inspector_page.setInspectedPage(page)
        assert self._settings is not None
        self._settings.update_for_url(inspector_page.requestedUrl())

    def _needs_recreate(self) -> bool:
        if False:
            i = 10
            return i + 15
        "Recreate the inspector when detaching to a window.\n\n        WORKAROUND for what's likely an unknown Qt bug.\n        "
        return True