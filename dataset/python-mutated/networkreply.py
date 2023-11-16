"""Special network replies.."""
from qutebrowser.qt.network import QNetworkReply, QNetworkRequest
from qutebrowser.qt.core import pyqtSlot, QIODevice, QByteArray, QTimer

class FixedDataNetworkReply(QNetworkReply):
    """QNetworkReply subclass for fixed data."""

    def __init__(self, request, fileData, mimeType, parent=None):
        if False:
            i = 10
            return i + 15
        'Constructor.\n\n        Args:\n            request: reference to the request object (QNetworkRequest)\n            fileData: reference to the data buffer (QByteArray)\n            mimeType: for the reply (string)\n            parent: reference to the parent object (QObject)\n        '
        super().__init__(parent)
        self._data = fileData
        self.setRequest(request)
        self.setUrl(request.url())
        self.setOpenMode(QIODevice.OpenModeFlag.ReadOnly)
        self.setHeader(QNetworkRequest.KnownHeaders.ContentTypeHeader, mimeType)
        self.setHeader(QNetworkRequest.KnownHeaders.ContentLengthHeader, QByteArray.number(len(fileData)))
        self.setAttribute(QNetworkRequest.Attribute.HttpStatusCodeAttribute, 200)
        self.setAttribute(QNetworkRequest.Attribute.HttpReasonPhraseAttribute, 'OK')
        QTimer.singleShot(0, lambda : self.metaDataChanged.emit())
        QTimer.singleShot(0, lambda : self.readyRead.emit())
        QTimer.singleShot(0, lambda : self.finished.emit())

    @pyqtSlot()
    def abort(self):
        if False:
            print('Hello World!')
        'Abort the operation.'

    def bytesAvailable(self):
        if False:
            return 10
        'Determine the bytes available for being read.\n\n        Return:\n            bytes available (int)\n        '
        return len(self._data) + super().bytesAvailable()

    def readData(self, maxlen):
        if False:
            print('Hello World!')
        'Retrieve data from the reply object.\n\n        Args:\n            maxlen maximum number of bytes to read (int)\n\n        Return:\n            bytestring containing the data\n        '
        len_ = min(maxlen, len(self._data))
        buf = bytes(self._data[:len_])
        self._data = self._data[len_:]
        return buf

    def isFinished(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def isRunning(self):
        if False:
            print('Hello World!')
        return False

class ErrorNetworkReply(QNetworkReply):
    """QNetworkReply which always returns an error."""

    def __init__(self, req, errorstring, error, parent=None):
        if False:
            return 10
        'Constructor.\n\n        Args:\n            req: The QNetworkRequest associated with this reply.\n            errorstring: The error string to print.\n            error: The numerical error value.\n            parent: The parent to pass to QNetworkReply.\n        '
        super().__init__(parent)
        self.setRequest(req)
        self.setUrl(req.url())
        self.setOpenMode(QIODevice.OpenModeFlag.ReadOnly)
        self.setError(error, errorstring)
        QTimer.singleShot(0, lambda : self.errorOccurred.emit(error))
        QTimer.singleShot(0, lambda : self.finished.emit())

    def abort(self):
        if False:
            for i in range(10):
                print('nop')
        "Do nothing since it's a fake reply."

    def bytesAvailable(self):
        if False:
            i = 10
            return i + 15
        'We always have 0 bytes available.'
        return 0

    def readData(self, _maxlen):
        if False:
            for i in range(10):
                print('nop')
        'No data available.'
        return b''

    def isFinished(self):
        if False:
            for i in range(10):
                print('nop')
        return True

    def isRunning(self):
        if False:
            print('Hello World!')
        return False

class RedirectNetworkReply(QNetworkReply):
    """A reply which redirects to the given URL."""

    def __init__(self, new_url, parent=None):
        if False:
            return 10
        super().__init__(parent)
        self.setAttribute(QNetworkRequest.Attribute.RedirectionTargetAttribute, new_url)
        QTimer.singleShot(0, lambda : self.finished.emit())

    def abort(self):
        if False:
            for i in range(10):
                print('nop')
        "Called when there's e.g. a redirection limit."

    def readData(self, _maxlen):
        if False:
            while True:
                i = 10
        return b''