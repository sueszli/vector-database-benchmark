"""Download manager."""
import io
import os.path
import shutil
import functools
import dataclasses
from typing import Dict, IO, Optional
from qutebrowser.qt.core import pyqtSlot, pyqtSignal, QTimer, QUrl
from qutebrowser.qt.widgets import QApplication
from qutebrowser.qt.network import QNetworkRequest, QNetworkReply, QNetworkAccessManager
from qutebrowser.config import config, websettings
from qutebrowser.utils import message, usertypes, log, urlutils, utils, debug, objreg, qtlog
from qutebrowser.misc import quitter
from qutebrowser.browser import downloads
from qutebrowser.browser.webkit import http
from qutebrowser.browser.webkit.network import networkmanager

@dataclasses.dataclass
class _RetryInfo:
    request: QNetworkRequest
    manager: QNetworkAccessManager

class DownloadItem(downloads.AbstractDownloadItem):
    """A single download currently running.

    There are multiple ways the data can flow from the QNetworkReply to the
    disk.

    If the filename/file object is known immediately when starting the
    download, QNetworkReply's readyRead writes to the target file directly.

    If not, readyRead is ignored and with self._read_timer we periodically read
    into the self._buffer BytesIO slowly, so some broken servers don't close
    our connection.

    As soon as we know the file object, we copy self._buffer over and the next
    readyRead will write to the real file object.

    Attributes:
        _retry_info: A _RetryInfo instance.
        _buffer: A BytesIO object to buffer incoming data until we know the
                 target file.
        _read_timer: A Timer which reads the QNetworkReply into self._buffer
                     periodically.
        _reply: The QNetworkReply associated with this download.
        _autoclose: Whether to close the associated file when the download is
                    done.

    Signals:
        adopt_download: Emitted when a download is retried and should be
                        adopted by the QNAM if needed.
                        arg 0: The new DownloadItem
    """
    adopt_download = pyqtSignal(object)

    def __init__(self, reply, manager):
        if False:
            while True:
                i = 10
        'Constructor.\n\n        Args:\n            reply: The QNetworkReply to download.\n        '
        super().__init__(manager=manager, parent=manager)
        self.fileobj: Optional[IO[bytes]] = None
        self.raw_headers: Dict[bytes, bytes] = {}
        self._autoclose = True
        self._retry_info = None
        self._reply = None
        self._buffer = io.BytesIO()
        self._read_timer = usertypes.Timer(self, name='download-read-timer')
        self._read_timer.setInterval(500)
        self._read_timer.timeout.connect(self._on_read_timer_timeout)
        self._url = reply.url()
        self._init_reply(reply)

    def _create_fileobj(self):
        if False:
            print('Hello World!')
        'Create a file object using the internal filename.'
        assert self._filename is not None
        try:
            fileobj = open(self._filename, 'wb')
        except OSError as e:
            self._die(e.strerror)
        else:
            self._set_fileobj(fileobj)

    def _do_die(self):
        if False:
            i = 10
            return i + 15
        'Abort the download and emit an error.'
        self._read_timer.stop()
        if self._reply is None:
            log.downloads.debug('Reply gone while dying')
            return
        self._reply.downloadProgress.disconnect()
        self._reply.finished.disconnect()
        self._reply.errorOccurred.disconnect()
        self._reply.readyRead.disconnect()
        with qtlog.hide_qt_warning('QNetworkReplyImplPrivate::error: Internal problem, this method must only be called once.'):
            self._reply.abort()
        self._reply.deleteLater()
        self._reply = None
        if self.fileobj is not None:
            pos = self.fileobj.tell()
            log.downloads.debug(f'File position at error: {pos}')
            try:
                self.fileobj.close()
            except OSError:
                log.downloads.exception('Error while closing file object')
            if pos == 0:
                filename = self._get_open_filename()
                log.downloads.debug(f'Removing empty file at {filename}')
                try:
                    os.remove(filename)
                except OSError:
                    log.downloads.exception('Error while removing empty file')

    def _init_reply(self, reply):
        if False:
            while True:
                i = 10
        'Set a new reply and connect its signals.\n\n        Args:\n            reply: The QNetworkReply to handle.\n        '
        self.done = False
        self.successful = False
        self._reply = reply
        reply.setReadBufferSize(16 * 1024 * 1024)
        reply.downloadProgress.connect(self.stats.on_download_progress)
        reply.finished.connect(self._on_reply_finished)
        reply.errorOccurred.connect(self._on_reply_error)
        reply.readyRead.connect(self._on_ready_read)
        reply.metaDataChanged.connect(self._on_meta_data_changed)
        reply.redirected.connect(self._on_redirected)
        self._retry_info = _RetryInfo(request=reply.request(), manager=reply.manager())
        if not self.fileobj:
            self._read_timer.start()
        if reply.error() != QNetworkReply.NetworkError.NoError:
            QTimer.singleShot(0, lambda : self._die(reply.errorString()))

    @pyqtSlot(QUrl)
    def _on_redirected(self, url):
        if False:
            return 10
        if self._reply is None:
            log.downloads.warning(f'redirected: REPLY GONE -> {url}')
        else:
            log.downloads.debug(f'redirected: {self._reply.url()} -> {url}')

    def _do_cancel(self):
        if False:
            print('Hello World!')
        self._read_timer.stop()
        if self._reply is not None:
            self._reply.finished.disconnect(self._on_reply_finished)
            self._reply.abort()
            self._reply.deleteLater()
            self._reply = None
        if self.fileobj is not None:
            self.fileobj.close()
        self.cancelled.emit()

    @pyqtSlot()
    def retry(self):
        if False:
            for i in range(10):
                print('nop')
        'Retry a failed download.'
        assert self.done
        assert not self.successful
        assert self._retry_info is not None
        self.remove()
        self.delete()
        new_reply = self._retry_info.manager.get(self._retry_info.request)
        new_download = self._manager.fetch(new_reply, suggested_filename=self.basename)
        self.adopt_download.emit(new_download)

    def _get_open_filename(self):
        if False:
            for i in range(10):
                print('nop')
        filename = self._filename
        if filename is None:
            filename = getattr(self.fileobj, 'name', None)
        return filename

    def url(self) -> QUrl:
        if False:
            print('Hello World!')
        return self._url

    def origin(self) -> QUrl:
        if False:
            i = 10
            return i + 15
        if self._reply is None:
            return QUrl()
        origin = self._reply.request().originatingObject()
        try:
            return origin.url()
        except AttributeError:
            return QUrl()

    def _ensure_can_set_filename(self, filename):
        if False:
            while True:
                i = 10
        if self.fileobj is not None:
            raise ValueError('fileobj was already set! filename: {}, existing: {}, fileobj {}'.format(filename, self._filename, self.fileobj))

    def _after_set_filename(self):
        if False:
            while True:
                i = 10
        self._create_fileobj()

    def _ask_confirm_question(self, title, msg, *, custom_yes_action=None):
        if False:
            return 10
        yes_action = custom_yes_action or self._after_set_filename
        no_action = functools.partial(self.cancel, remove_data=False)
        url = 'file://{}'.format(self._filename)
        message.confirm_async(title=title, text=msg, yes_action=yes_action, no_action=no_action, cancel_action=no_action, abort_on=[self.cancelled, self.error], url=url)

    def _ask_create_parent_question(self, title, msg, force_overwrite, remember_directory):
        if False:
            i = 10
            return i + 15
        assert self._filename is not None
        no_action = functools.partial(self.cancel, remove_data=False)
        url = 'file://{}'.format(os.path.dirname(self._filename))
        message.confirm_async(title=title, text=msg, yes_action=lambda : self._after_create_parent_question(force_overwrite, remember_directory), no_action=no_action, cancel_action=no_action, abort_on=[self.cancelled, self.error], url=url)

    def _set_fileobj(self, fileobj, *, autoclose=True):
        if False:
            i = 10
            return i + 15
        'Set the file object to write the download to.\n\n        Args:\n            fileobj: A file-like object.\n        '
        assert self._reply is not None
        if self.fileobj is not None:
            raise ValueError('fileobj was already set! Old: {}, new: {}'.format(self.fileobj, fileobj))
        self.fileobj = fileobj
        self._autoclose = autoclose
        try:
            self._read_timer.stop()
            log.downloads.debug('buffer: {} bytes'.format(self._buffer.tell()))
            self._buffer.seek(0)
            shutil.copyfileobj(self._buffer, fileobj)
            self._buffer.close()
            if self._reply.isFinished():
                self._on_reply_finished()
            else:
                self._on_ready_read()
        except OSError as e:
            self._die(e.strerror)

    def _set_tempfile(self, fileobj):
        if False:
            i = 10
            return i + 15
        self._set_fileobj(fileobj)

    def _finish_download(self):
        if False:
            print('Hello World!')
        'Write buffered data to disk and finish the QNetworkReply.'
        assert self._reply is not None
        assert self.fileobj is not None
        log.downloads.debug('Finishing download...')
        if self._reply.isOpen():
            self.fileobj.write(self._reply.readAll())
        if self._autoclose:
            self.fileobj.close()
        self.successful = self._reply.error() == QNetworkReply.NetworkError.NoError
        self._reply.close()
        self._reply.deleteLater()
        self._reply = None
        self.finished.emit()
        self.done = True
        log.downloads.debug('Download {} finished'.format(self.basename))
        self.data_changed.emit()

    @pyqtSlot()
    def _on_reply_finished(self):
        if False:
            for i in range(10):
                print('nop')
        "Clean up when the download was finished.\n\n        Note when this gets called, only the QNetworkReply has finished. This\n        doesn't mean the download (i.e. writing data to the disk) is finished\n        as well. Therefore, we can't close() the QNetworkReply in here yet.\n        "
        if self._reply is None:
            return
        self._read_timer.stop()
        self.stats.finish()
        log.downloads.debug('Reply finished, fileobj {}'.format(self.fileobj))
        if self.fileobj is not None:
            self._finish_download()

    @pyqtSlot()
    def _on_ready_read(self):
        if False:
            while True:
                i = 10
        'Read available data and save file when ready to read.'
        if self.fileobj is None or self._reply is None:
            return
        if not self._reply.isOpen():
            raise OSError('Reply is closed!')
        try:
            self.fileobj.write(self._reply.readAll())
        except OSError as e:
            self._die(e.strerror)

    @pyqtSlot('QNetworkReply::NetworkError')
    def _on_reply_error(self, code):
        if False:
            for i in range(10):
                print('nop')
        'Handle QNetworkReply errors.'
        if code == QNetworkReply.NetworkError.OperationCanceledError:
            return
        if self._reply is None:
            error = 'Unknown error: {}'.format(debug.qenum_key(QNetworkReply, code))
        else:
            error = self._reply.errorString()
        self._die(error)

    @pyqtSlot()
    def _on_read_timer_timeout(self):
        if False:
            print('Hello World!')
        'Read some bytes from the QNetworkReply periodically.'
        assert self._reply is not None
        if not self._reply.isOpen():
            raise OSError('Reply is closed!')
        data = self._reply.read(1024)
        if data is not None:
            self._buffer.write(data)

    @pyqtSlot()
    def _on_meta_data_changed(self):
        if False:
            i = 10
            return i + 15
        "Update the download's metadata."
        if self._reply is None:
            return
        self.raw_headers = {}
        for (key, value) in self._reply.rawHeaderPairs():
            self.raw_headers[bytes(key)] = bytes(value)

    def _uses_nam(self, nam):
        if False:
            for i in range(10):
                print('nop')
        'Check if this download uses the given QNetworkAccessManager.'
        assert self._retry_info is not None
        running_nam = self._reply is not None and self._reply.manager() is nam
        retry_nam = self.done and (not self.successful) and (self._retry_info.manager is nam)
        return running_nam or retry_nam

class DownloadManager(downloads.AbstractDownloadManager):
    """Manager for currently running downloads.

    Attributes:
        _networkmanager: A NetworkManager for generic downloads.

    Class attributes:
        _MAX_REDIRECTS: The maximum redirection count.
    """
    _MAX_REDIRECTS = 20

    def __init__(self, parent=None):
        if False:
            print('Hello World!')
        super().__init__(parent)
        self._networkmanager = networkmanager.NetworkManager(win_id=None, tab_id=None, private=config.val.content.private_browsing, parent=self)

    @pyqtSlot('QUrl')
    def get(self, url, cache=True, **kwargs):
        if False:
            print('Hello World!')
        "Start a download with a link URL.\n\n        Args:\n            url: The URL to get, as QUrl\n            cache: If set to False, don't cache the response.\n            **kwargs: passed to get_request().\n\n        Return:\n            The created DownloadItem.\n        "
        if not url.isValid():
            urlutils.invalid_url_error(url, 'start download')
            return None
        req = QNetworkRequest(url)
        user_agent = websettings.user_agent(url)
        req.setHeader(QNetworkRequest.KnownHeaders.UserAgentHeader, user_agent)
        if not cache:
            req.setAttribute(QNetworkRequest.Attribute.CacheSaveControlAttribute, False)
        req.setAttribute(QNetworkRequest.Attribute.RedirectPolicyAttribute, QNetworkRequest.RedirectPolicy.NoLessSafeRedirectPolicy)
        req.setMaximumRedirectsAllowed(self._MAX_REDIRECTS)
        return self.get_request(req, **kwargs)

    def get_mhtml(self, tab, target):
        if False:
            for i in range(10):
                print('nop')
        'Download the given tab as mhtml to the given DownloadTarget.'
        assert tab.backend == usertypes.Backend.QtWebKit
        from qutebrowser.browser.webkit import mhtml
        if target is not None:
            mhtml.start_download_checked(target, tab=tab)
            return
        suggested_fn = utils.sanitize_filename(tab.title() + '.mhtml')
        filename = downloads.immediate_download_path()
        if filename is not None:
            target = downloads.FileDownloadTarget(filename)
            mhtml.start_download_checked(target, tab=tab)
        else:
            question = downloads.get_filename_question(suggested_filename=suggested_fn, url=tab.url(), parent=tab)
            question.answered.connect(functools.partial(mhtml.start_download_checked, tab=tab))
            message.global_bridge.ask(question, blocking=False)

    def _get_suggested_filename(self, request):
        if False:
            while True:
                i = 10
        'Get the suggested filename for the given request.'
        filename_url = request.url()
        if request.url().scheme().lower() == 'data':
            origin = request.originatingObject()
            try:
                filename_url = origin.url()
            except AttributeError:
                pass
        return urlutils.filename_from_url(filename_url, fallback='qutebrowser-download')

    def get_request(self, request, *, target=None, suggested_fn=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Start a download with a QNetworkRequest.\n\n        Args:\n            request: The QNetworkRequest to download.\n            target: Where to save the download as downloads.DownloadTarget.\n            suggested_fn: The filename to use for the file.\n            **kwargs: Passed to _fetch_request.\n\n        Return:\n            The created DownloadItem.\n        '
        request.setAttribute(QNetworkRequest.Attribute.CacheLoadControlAttribute, QNetworkRequest.CacheLoadControl.AlwaysNetwork)
        if suggested_fn is None:
            suggested_fn = self._get_suggested_filename(request)
        return self._fetch_request(request, target=target, suggested_filename=suggested_fn, **kwargs)

    def _fetch_request(self, request, *, qnam=None, **kwargs):
        if False:
            while True:
                i = 10
        'Download a QNetworkRequest to disk.\n\n        Args:\n            request: The QNetworkRequest to download.\n            qnam: The QNetworkAccessManager to use.\n            **kwargs: passed to fetch().\n\n        Return:\n            The created DownloadItem.\n        '
        if qnam is None:
            qnam = self._networkmanager
        reply = qnam.get(request)
        return self.fetch(reply, **kwargs)

    @pyqtSlot('QNetworkReply')
    def fetch(self, reply, *, target=None, auto_remove=False, suggested_filename=None, prompt_download_directory=None):
        if False:
            i = 10
            return i + 15
        'Download a QNetworkReply to disk.\n\n        Args:\n            reply: The QNetworkReply to download.\n            target: Where to save the download as downloads.DownloadTarget.\n            auto_remove: Whether to remove the download even if\n                         downloads.remove_finished is set to -1.\n            suggested_filename: The filename to use for the file.\n            prompt_download_directory: Whether to prompt for a location to\n                                       download the file to.\n\n        Return:\n            The created DownloadItem.\n        '
        if not suggested_filename:
            try:
                suggested_filename = target.suggested_filename()
            except downloads.NoFilenameError:
                (_, suggested_filename) = http.parse_content_disposition(reply)
        log.downloads.debug('fetch: {} -> {}'.format(reply.url(), suggested_filename))
        download = DownloadItem(reply, manager=self)
        self._init_item(download, auto_remove, suggested_filename)
        if download.cancel_for_origin():
            return download
        if target is not None:
            download.set_target(target)
            return download
        filename = downloads.immediate_download_path(prompt_download_directory)
        if filename is not None:
            target = downloads.FileDownloadTarget(filename)
            download.set_target(target)
            return download
        question = downloads.get_filename_question(suggested_filename=suggested_filename, url=reply.url(), parent=self)
        self._init_filename_question(question, download)
        message.global_bridge.ask(question, blocking=False)
        return download

    def has_downloads_with_nam(self, nam):
        if False:
            i = 10
            return i + 15
        'Check if the DownloadManager has any downloads with the given QNAM.\n\n        Args:\n            nam: The QNetworkAccessManager to check.\n\n        Return:\n            A boolean.\n        '
        assert nam.adopted_downloads == 0
        for download in self.downloads:
            assert isinstance(download, DownloadItem), download
            if download._uses_nam(nam):
                nam.adopt_download(download)
        return nam.adopted_downloads

def init():
    if False:
        i = 10
        return i + 15
    'Initialize the global QtNetwork download manager.'
    download_manager = DownloadManager(parent=QApplication.instance())
    objreg.register('qtnetwork-download-manager', download_manager)
    quitter.instance.shutting_down.connect(download_manager.shutdown)