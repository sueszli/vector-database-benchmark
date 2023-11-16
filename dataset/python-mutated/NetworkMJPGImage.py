from typing import Optional
from PyQt6.QtCore import QUrl, pyqtProperty, pyqtSignal, pyqtSlot, QRect, QByteArray
from PyQt6.QtGui import QImage, QPainter
from PyQt6.QtQuick import QQuickPaintedItem
from PyQt6.QtNetwork import QNetworkRequest, QNetworkReply, QNetworkAccessManager
from UM.Logger import Logger

class NetworkMJPGImage(QQuickPaintedItem):

    def __init__(self, *args, **kwargs) -> None:
        if False:
            for i in range(10):
                print('nop')
        super().__init__(*args, **kwargs)
        self._stream_buffer = QByteArray()
        self._stream_buffer_start_index = -1
        self._network_manager: Optional[QNetworkAccessManager] = None
        self._image_request: Optional[QNetworkRequest] = None
        self._image_reply: Optional[QNetworkReply] = None
        self._image = QImage()
        self._image_rect = QRect()
        self._source_url = QUrl()
        self._started = False
        self._mirror = False
        self.setAntialiasing(True)

    def __del__(self) -> None:
        if False:
            for i in range(10):
                print('nop')
        'Ensure that close gets called when object is destroyed'
        self.stop()

    def paint(self, painter: 'QPainter') -> None:
        if False:
            while True:
                i = 10
        if self._mirror:
            painter.drawImage(self.contentsBoundingRect(), self._image.mirrored())
            return
        painter.drawImage(self.contentsBoundingRect(), self._image)

    def setSourceURL(self, source_url: 'QUrl') -> None:
        if False:
            while True:
                i = 10
        self._source_url = source_url
        self.sourceURLChanged.emit()
        if self._started:
            self.start()

    def getSourceURL(self) -> 'QUrl':
        if False:
            print('Hello World!')
        return self._source_url
    sourceURLChanged = pyqtSignal()
    source = pyqtProperty(QUrl, fget=getSourceURL, fset=setSourceURL, notify=sourceURLChanged)

    def setMirror(self, mirror: bool) -> None:
        if False:
            return 10
        if mirror == self._mirror:
            return
        self._mirror = mirror
        self.mirrorChanged.emit()
        self.update()

    def getMirror(self) -> bool:
        if False:
            for i in range(10):
                print('nop')
        return self._mirror
    mirrorChanged = pyqtSignal()
    mirror = pyqtProperty(bool, fget=getMirror, fset=setMirror, notify=mirrorChanged)
    imageSizeChanged = pyqtSignal()

    @pyqtProperty(int, notify=imageSizeChanged)
    def imageWidth(self) -> int:
        if False:
            return 10
        return self._image.width()

    @pyqtProperty(int, notify=imageSizeChanged)
    def imageHeight(self) -> int:
        if False:
            print('Hello World!')
        return self._image.height()

    @pyqtSlot()
    def start(self) -> None:
        if False:
            return 10
        self.stop()
        if not self._source_url:
            Logger.log('w', 'Unable to start camera stream without target!')
            return
        self._started = True
        self._image_request = QNetworkRequest(self._source_url)
        if self._network_manager is None:
            self._network_manager = QNetworkAccessManager()
        self._image_reply = self._network_manager.get(self._image_request)
        self._image_reply.downloadProgress.connect(self._onStreamDownloadProgress)

    @pyqtSlot()
    def stop(self) -> None:
        if False:
            return 10
        self._stream_buffer = QByteArray()
        self._stream_buffer_start_index = -1
        if self._image_reply:
            try:
                try:
                    self._image_reply.downloadProgress.disconnect(self._onStreamDownloadProgress)
                except Exception:
                    pass
                if not self._image_reply.isFinished():
                    self._image_reply.close()
            except Exception:
                pass
            self._image_reply = None
            self._image_request = None
        self._network_manager = None
        self._started = False

    def _onStreamDownloadProgress(self, bytes_received: int, bytes_total: int) -> None:
        if False:
            while True:
                i = 10
        if self._image_reply is None:
            return
        self._stream_buffer += self._image_reply.readAll()
        if len(self._stream_buffer) > 2000000:
            Logger.log('w', 'MJPEG buffer exceeds reasonable size. Restarting stream...')
            self.stop()
            self.start()
            return
        if self._stream_buffer_start_index == -1:
            self._stream_buffer_start_index = self._stream_buffer.indexOf(b'\xff\xd8')
        stream_buffer_end_index = self._stream_buffer.lastIndexOf(b'\xff\xd9')
        if self._stream_buffer_start_index != -1 and stream_buffer_end_index != -1:
            jpg_data = self._stream_buffer[self._stream_buffer_start_index:stream_buffer_end_index + 2]
            self._stream_buffer = self._stream_buffer[stream_buffer_end_index + 2:]
            self._stream_buffer_start_index = -1
            self._image.loadFromData(jpg_data)
            if self._image.rect() != self._image_rect:
                self.imageSizeChanged.emit()
            self.update()