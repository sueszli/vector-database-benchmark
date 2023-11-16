from typing import List
from PyQt5.QtMultimedia import QVideoFrame, QAbstractVideoBuffer, QAbstractVideoSurface, QVideoSurfaceFormat
from PyQt5.QtGui import QImage
from PyQt5.QtCore import QObject, pyqtSignal
from electrum.i18n import _
from electrum.logging import get_logger
_logger = get_logger(__name__)

class QrReaderVideoSurface(QAbstractVideoSurface):
    """
    Receives QVideoFrames from QCamera, converts them into a QImage, flips the X and Y axis if
    necessary and sends them to listeners via the frame_available event.
    """

    def __init__(self, parent: QObject=None):
        if False:
            while True:
                i = 10
        super().__init__(parent)

    def present(self, frame: QVideoFrame) -> bool:
        if False:
            return 10
        if not frame.isValid():
            return False
        image_format = QVideoFrame.imageFormatFromPixelFormat(frame.pixelFormat())
        if image_format == QVideoFrame.Format_Invalid:
            _logger.info(_('QR code scanner for video frame with invalid pixel format'))
            return False
        if not frame.map(QAbstractVideoBuffer.ReadOnly):
            _logger.info(_('QR code scanner failed to map video frame'))
            return False
        try:
            img = QImage(int(frame.bits()), frame.width(), frame.height(), image_format)
            surface_format = self.surfaceFormat()
            flip_x = surface_format.isMirrored()
            flip_y = surface_format.scanLineDirection() == QVideoSurfaceFormat.BottomToTop
            if flip_x or flip_y:
                img = img.mirrored(flip_x, flip_y)
            img = img.copy()
        finally:
            frame.unmap()
        self.frame_available.emit(img)
        return True

    def supportedPixelFormats(self, handle_type: QAbstractVideoBuffer.HandleType) -> List[QVideoFrame.PixelFormat]:
        if False:
            for i in range(10):
                print('nop')
        if handle_type == QAbstractVideoBuffer.NoHandle:
            return [QVideoFrame.Format_ARGB32, QVideoFrame.Format_ARGB32_Premultiplied, QVideoFrame.Format_RGB32, QVideoFrame.Format_RGB24, QVideoFrame.Format_RGB565, QVideoFrame.Format_RGB555, QVideoFrame.Format_ARGB8565_Premultiplied]
        return []
    frame_available = pyqtSignal(QImage)