"""The main browser widgets."""
import html
import functools
from qutebrowser.qt.core import pyqtSlot, pyqtSignal, Qt, QUrl, QPoint
from qutebrowser.qt.gui import QDesktopServices
from qutebrowser.qt.network import QNetworkReply, QNetworkRequest
from qutebrowser.qt.widgets import QFileDialog
from qutebrowser.qt.printsupport import QPrintDialog
from qutebrowser.qt.webkitwidgets import QWebPage, QWebFrame
from qutebrowser.config import websettings, config
from qutebrowser.browser import pdfjs, shared, downloads, greasemonkey
from qutebrowser.browser.webkit import http
from qutebrowser.browser.webkit.network import networkmanager
from qutebrowser.utils import message, usertypes, log, jinja, objreg
from qutebrowser.qt import sip

class BrowserPage(QWebPage):
    """Our own QWebPage with advanced features.

    Attributes:
        error_occurred: Whether an error occurred while loading.
        _extension_handlers: Mapping of QWebPage extensions to their handlers.
        _networkmanager: The NetworkManager used.
        _win_id: The window ID this BrowserPage is associated with.
        _ignore_load_started: Whether to ignore the next loadStarted signal.
        _is_shutting_down: Whether the page is currently shutting down.
        _tabdata: The TabData object of the tab this page is in.

    Signals:
        shutting_down: Emitted when the page is currently shutting down.
        reloading: Emitted before a web page reloads.
                   arg: The URL which gets reloaded.
        navigation_request: Emitted on acceptNavigationRequest.
    """
    shutting_down = pyqtSignal()
    reloading = pyqtSignal(QUrl)
    navigation_request = pyqtSignal(usertypes.NavigationRequest)

    def __init__(self, win_id, tab_id, tabdata, private, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._win_id = win_id
        self._tabdata = tabdata
        self._is_shutting_down = False
        self._extension_handlers = {QWebPage.Extension.ErrorPageExtension: self._handle_errorpage, QWebPage.Extension.ChooseMultipleFilesExtension: self._handle_multiple_files}
        self._ignore_load_started = False
        self.error_occurred = False
        self._networkmanager = networkmanager.NetworkManager(win_id=win_id, tab_id=tab_id, private=private, parent=self)
        self.setNetworkAccessManager(self._networkmanager)
        self.setForwardUnsupportedContent(True)
        self.reloading.connect(self._networkmanager.clear_rejected_ssl_errors)
        self.printRequested.connect(self.on_print_requested)
        self.downloadRequested.connect(self.on_download_requested)
        self.unsupportedContent.connect(self.on_unsupported_content)
        self.loadStarted.connect(self.on_load_started)
        self.featurePermissionRequested.connect(self._on_feature_permission_requested)
        self.saveFrameStateRequested.connect(self.on_save_frame_state_requested)
        self.restoreFrameStateRequested.connect(self.on_restore_frame_state_requested)
        self.loadFinished.connect(functools.partial(self._inject_userjs, self.mainFrame()))
        self.frameCreated.connect(self._connect_userjs_signals)

    @pyqtSlot('QWebFrame*')
    def _connect_userjs_signals(self, frame):
        if False:
            while True:
                i = 10
        'Connect userjs related signals to `frame`.\n\n        Connect the signals used as triggers for injecting user\n        JavaScripts into the passed QWebFrame.\n        '
        log.greasemonkey.debug('Connecting to frame {} ({})'.format(frame, frame.url().toDisplayString()))
        frame.loadFinished.connect(functools.partial(self._inject_userjs, frame))

    def javaScriptPrompt(self, frame, js_msg, default):
        if False:
            return 10
        'Override javaScriptPrompt to use qutebrowser prompts.'
        if self._is_shutting_down:
            return (False, '')
        try:
            return shared.javascript_prompt(frame.url(), js_msg, default, abort_on=[self.loadStarted, self.shutting_down])
        except shared.CallSuper:
            return super().javaScriptPrompt(frame, js_msg, default)

    def _handle_errorpage(self, info, errpage):
        if False:
            while True:
                i = 10
        'Display an error page if needed.\n\n        Loosely based on Helpviewer/HelpBrowserWV.py from eric5\n        (line 260 @ 5d937eb378dd)\n\n        Args:\n            info: The QWebPage.ErrorPageExtensionOption instance.\n            errpage: The QWebPage.ErrorPageExtensionReturn instance, where the\n                     error page will get written to.\n\n        Return:\n            False if no error page should be displayed, True otherwise.\n        '
        ignored_errors = [(QWebPage.ErrorDomain.QtNetwork, QNetworkReply.NetworkError.OperationCanceledError), (QWebPage.ErrorDomain.WebKit, 203), (QWebPage.ErrorDomain.WebKit, 102)]
        errpage.baseUrl = info.url
        urlstr = info.url.toDisplayString()
        if (info.domain, info.error) == (QWebPage.ErrorDomain.QtNetwork, QNetworkReply.NetworkError.ProtocolUnknownError):
            url = QUrl(info.url)
            scheme = url.scheme()
            message.confirm_async(title='Open external application for {}-link?'.format(scheme), text='URL: <b>{}</b>'.format(html.escape(url.toDisplayString())), yes_action=functools.partial(QDesktopServices.openUrl, url), url=info.url.toString(QUrl.UrlFormattingOption.RemovePassword | QUrl.ComponentFormattingOption.FullyEncoded))
            return True
        elif (info.domain, info.error) in ignored_errors:
            log.webview.debug('Ignored error on {}: {} (error domain: {}, error code: {})'.format(urlstr, info.errorString, info.domain, info.error))
            return False
        else:
            error_str = info.errorString
            if error_str == networkmanager.HOSTBLOCK_ERROR_STRING:
                error_str = 'Request blocked by host blocker.'
                main_frame = info.frame.page().mainFrame()
                if info.frame != main_frame:
                    for elem in main_frame.documentElement().findAll('iframe'):
                        if QUrl(elem.attribute('src')) == info.url:
                            elem.setAttribute('style', 'display: none')
                    return False
            else:
                self._ignore_load_started = True
                self.error_occurred = True
            log.webview.error('Error while loading {}: {}'.format(urlstr, error_str))
            log.webview.debug('Error domain: {}, error code: {}'.format(info.domain, info.error))
            title = 'Error loading page: {}'.format(urlstr)
            error_html = jinja.render('error.html', title=title, url=urlstr, error=error_str)
            errpage.content = error_html.encode('utf-8')
            errpage.encoding = 'utf-8'
            return True

    def chooseFile(self, parent_frame: QWebFrame, suggested_file: str) -> str:
        if False:
            for i in range(10):
                print('nop')
        'Override chooseFile to (optionally) invoke custom file uploader.'
        handler = config.val.fileselect.handler
        if handler == 'default':
            return super().chooseFile(parent_frame, suggested_file)
        assert handler == 'external', handler
        selected = shared.choose_file(qb_mode=shared.FileSelectionMode.single_file)
        if not selected:
            return ''
        else:
            return selected[0]

    def _handle_multiple_files(self, info, files):
        if False:
            i = 10
            return i + 15
        'Handle uploading of multiple files.\n\n        Loosely based on Helpviewer/HelpBrowserWV.py from eric5.\n\n        Args:\n            info: The ChooseMultipleFilesExtensionOption instance.\n            files: The ChooseMultipleFilesExtensionReturn instance to write\n                   return values to.\n\n        Return:\n            True on success, the superclass return value on failure.\n        '
        handler = config.val.fileselect.handler
        if handler == 'default':
            suggested_file = ''
            if info.suggestedFileNames:
                suggested_file = info.suggestedFileNames[0]
            (files.fileNames, _) = QFileDialog.getOpenFileNames(None, None, suggested_file)
            return True
        assert handler == 'external', handler
        files.fileNames = shared.choose_file(shared.FileSelectionMode.multiple_files)
        return True

    def shutdown(self):
        if False:
            return 10
        'Prepare the web page for being deleted.'
        self._is_shutting_down = True
        self.shutting_down.emit()
        download_manager = objreg.get('qtnetwork-download-manager')
        nam = self.networkAccessManager()
        if download_manager.has_downloads_with_nam(nam):
            nam.setParent(download_manager)
        else:
            assert isinstance(nam, networkmanager.NetworkManager), nam
            nam.shutdown()

    def display_content(self, reply, mimetype):
        if False:
            i = 10
            return i + 15
        'Display a QNetworkReply with an explicitly set mimetype.'
        self.mainFrame().setContent(reply.readAll(), mimetype, reply.url())
        reply.deleteLater()

    def on_print_requested(self, frame):
        if False:
            i = 10
            return i + 15
        'Handle printing when requested via javascript.'
        printdiag = QPrintDialog()
        printdiag.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        printdiag.open(lambda : frame.print(printdiag.printer()))

    def on_download_requested(self, request):
        if False:
            print('Hello World!')
        "Called when the user wants to download a link.\n\n        We need to construct a copy of the QNetworkRequest here as the\n        download_manager needs it async and we'd get a segfault otherwise as\n        soon as the user has entered the filename, as Qt seems to delete it\n        after this slot returns.\n        "
        req = QNetworkRequest(request)
        download_manager = objreg.get('qtnetwork-download-manager')
        download_manager.get_request(req, qnam=self.networkAccessManager())

    @pyqtSlot('QNetworkReply*')
    def on_unsupported_content(self, reply):
        if False:
            for i in range(10):
                print('nop')
        'Handle an unsupportedContent signal.\n\n        Most likely this will mean we need to download the reply, but we\n        correct for some common errors the server do.\n\n        At some point we might want to implement the MIME Sniffing standard\n        here: https://mimesniff.spec.whatwg.org/\n        '
        (inline, suggested_filename) = http.parse_content_disposition(reply)
        download_manager = objreg.get('qtnetwork-download-manager')
        if not inline:
            download_manager.fetch(reply, suggested_filename=suggested_filename)
            return
        (mimetype, _rest) = http.parse_content_type(reply)
        if mimetype == 'image/jpg':
            if reply.isFinished():
                self.display_content(reply, 'image/jpeg')
            else:
                reply.finished.connect(functools.partial(self.display_content, reply, 'image/jpeg'))
        elif pdfjs.should_use_pdfjs(mimetype, reply.url()):
            download_manager.fetch(reply, target=downloads.PDFJSDownloadTarget(), auto_remove=True)
        else:
            download_manager.fetch(reply, suggested_filename=suggested_filename)

    @pyqtSlot()
    def on_load_started(self):
        if False:
            print('Hello World!')
        'Reset error_occurred when loading of a new page started.'
        if self._ignore_load_started:
            self._ignore_load_started = False
        else:
            self.error_occurred = False

    def _inject_userjs(self, frame):
        if False:
            print('Hello World!')
        'Inject user JavaScripts into the page.\n\n        Args:\n            frame: The QWebFrame to inject the user scripts into.\n        '
        if sip.isdeleted(frame):
            log.greasemonkey.debug('_inject_userjs called for deleted frame!')
            return
        url = frame.url()
        if url.isEmpty():
            url = frame.requestedUrl()
        log.greasemonkey.debug('_inject_userjs called for {} ({})'.format(frame, url.toDisplayString()))
        scripts = greasemonkey.gm_manager.scripts_for(url)
        toload = scripts.start + scripts.end + scripts.idle
        if url.isEmpty():
            log.greasemonkey.debug('Not running scripts for frame with no url: {}'.format(frame))
            assert not toload, toload
        for script in toload:
            if frame is self.mainFrame() or script.runs_on_sub_frames:
                log.webview.debug(f'Running GM script: {script}')
                frame.evaluateJavaScript(script.code())

    @pyqtSlot('QWebFrame*', 'QWebPage::Feature')
    def _on_feature_permission_requested(self, frame, feature):
        if False:
            for i in range(10):
                print('nop')
        'Ask the user for approval for geolocation/notifications.'
        if not isinstance(frame, QWebFrame):
            log.misc.error('on_feature_permission_requested got called with {!r}!'.format(frame))
            return
        options = {QWebPage.Feature.Notifications: 'content.notifications.enabled', QWebPage.Feature.Geolocation: 'content.geolocation'}
        messages = {QWebPage.Feature.Notifications: 'show notifications', QWebPage.Feature.Geolocation: 'access your location'}
        yes_action = functools.partial(self.setFeaturePermission, frame, feature, QWebPage.PermissionPolicy.PermissionGrantedByUser)
        no_action = functools.partial(self.setFeaturePermission, frame, feature, QWebPage.PermissionPolicy.PermissionDeniedByUser)
        url = frame.url().adjusted(QUrl.UrlFormattingOption.RemoveUserInfo | QUrl.UrlFormattingOption.RemovePath | QUrl.UrlFormattingOption.RemoveQuery | QUrl.UrlFormattingOption.RemoveFragment)
        question = shared.feature_permission(url=url, option=options[feature], msg=messages[feature], yes_action=yes_action, no_action=no_action, abort_on=[self.shutting_down, self.loadStarted])
        if question is not None:
            self.featurePermissionRequestCanceled.connect(functools.partial(self._on_feature_permission_cancelled, question, frame, feature))

    def _on_feature_permission_cancelled(self, question, frame, feature, cancelled_frame, cancelled_feature):
        if False:
            while True:
                i = 10
        'Slot invoked when a feature permission request was cancelled.\n\n        To be used with functools.partial.\n        '
        if frame is cancelled_frame and feature == cancelled_feature:
            try:
                question.abort()
            except RuntimeError:
                pass

    def on_save_frame_state_requested(self, frame, item):
        if False:
            for i in range(10):
                print('nop')
        'Save scroll position and zoom in history.\n\n        Args:\n            frame: The QWebFrame which gets saved.\n            item: The QWebHistoryItem to be saved.\n        '
        if frame != self.mainFrame():
            return
        data = {'zoom': frame.zoomFactor(), 'scroll-pos': frame.scrollPosition()}
        item.setUserData(data)

    def on_restore_frame_state_requested(self, frame):
        if False:
            return 10
        'Restore scroll position and zoom from history.\n\n        Args:\n            frame: The QWebFrame which gets restored.\n        '
        if frame != self.mainFrame():
            return
        data = self.history().currentItem().userData()
        if data is None:
            return
        if 'zoom' in data:
            frame.page().view().tab.zoom.set_factor(data['zoom'])
        if 'scroll-pos' in data and frame.scrollPosition() == QPoint(0, 0):
            frame.setScrollPosition(data['scroll-pos'])

    def userAgentForUrl(self, url):
        if False:
            i = 10
            return i + 15
        'Override QWebPage::userAgentForUrl to customize the user agent.'
        if not url.isValid():
            url = None
        return websettings.user_agent(url)

    def supportsExtension(self, ext):
        if False:
            for i in range(10):
                print('nop')
        'Override QWebPage::supportsExtension to provide error pages.\n\n        Args:\n            ext: The extension to check for.\n\n        Return:\n            True if the extension can be handled, False otherwise.\n        '
        return ext in self._extension_handlers

    def extension(self, ext, opt, out):
        if False:
            for i in range(10):
                print('nop')
        'Override QWebPage::extension to provide error pages.\n\n        Args:\n            ext: The extension.\n            opt: Extension options instance.\n            out: Extension output instance.\n\n        Return:\n            Handler return value.\n        '
        try:
            handler = self._extension_handlers[ext]
        except KeyError:
            log.webview.warning('Extension {} not supported!'.format(ext))
            return super().extension(ext, opt, out)
        return handler(opt, out)

    def javaScriptAlert(self, frame, js_msg):
        if False:
            print('Hello World!')
        'Override javaScriptAlert to use qutebrowser prompts.'
        if self._is_shutting_down:
            return
        try:
            shared.javascript_alert(frame.url(), js_msg, abort_on=[self.loadStarted, self.shutting_down])
        except shared.CallSuper:
            super().javaScriptAlert(frame, js_msg)

    def javaScriptConfirm(self, frame, js_msg):
        if False:
            i = 10
            return i + 15
        'Override javaScriptConfirm to use the statusbar.'
        if self._is_shutting_down:
            return False
        try:
            return shared.javascript_confirm(frame.url(), js_msg, abort_on=[self.loadStarted, self.shutting_down])
        except shared.CallSuper:
            return super().javaScriptConfirm(frame, js_msg)

    def javaScriptConsoleMessage(self, msg, line, source):
        if False:
            i = 10
            return i + 15
        'Override javaScriptConsoleMessage to use debug log.'
        shared.javascript_log_message(usertypes.JsLogLevel.unknown, source, line, msg)

    def acceptNavigationRequest(self, frame: QWebFrame, request: QNetworkRequest, typ: QWebPage.NavigationType) -> bool:
        if False:
            for i in range(10):
                print('nop')
        "Override acceptNavigationRequest to handle clicked links.\n\n        Setting linkDelegationPolicy to DelegateAllLinks and using a slot bound\n        to linkClicked won't work correctly, because when in a frameset, we\n        have no idea in which frame the link should be opened.\n\n        Checks if it should open it in a tab (middle-click or control) or not,\n        and then conditionally opens the URL here or in another tab/window.\n        "
        type_map = {QWebPage.NavigationType.NavigationTypeLinkClicked: usertypes.NavigationRequest.Type.link_clicked, QWebPage.NavigationType.NavigationTypeFormSubmitted: usertypes.NavigationRequest.Type.form_submitted, QWebPage.NavigationType.NavigationTypeFormResubmitted: usertypes.NavigationRequest.Type.form_resubmitted, QWebPage.NavigationType.NavigationTypeBackOrForward: usertypes.NavigationRequest.Type.back_forward, QWebPage.NavigationType.NavigationTypeReload: usertypes.NavigationRequest.Type.reload, QWebPage.NavigationType.NavigationTypeOther: usertypes.NavigationRequest.Type.other}
        is_main_frame = frame is self.mainFrame()
        navigation = usertypes.NavigationRequest(url=request.url(), navigation_type=type_map[typ], is_main_frame=is_main_frame)
        if navigation.navigation_type == navigation.Type.reload:
            self.reloading.emit(navigation.url)
        self.navigation_request.emit(navigation)
        return navigation.accepted