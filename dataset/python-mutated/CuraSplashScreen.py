from PyQt6.QtCore import Qt, QCoreApplication, QTimer
from PyQt6.QtGui import QPixmap, QColor, QFont, QPen, QPainter
from PyQt6.QtWidgets import QSplashScreen
from UM.Resources import Resources
from UM.Application import Application
from cura import ApplicationMetadata
import time

class CuraSplashScreen(QSplashScreen):

    def __init__(self):
        if False:
            print('Hello World!')
        super().__init__()
        self._scale = 1
        self._version_y_offset = 0
        if ApplicationMetadata.IsAlternateVersion:
            splash_image = QPixmap(Resources.getPath(Resources.Images, 'cura_wip.png'))
        elif ApplicationMetadata.IsEnterpriseVersion:
            splash_image = QPixmap(Resources.getPath(Resources.Images, 'cura_enterprise.png'))
            self._version_y_offset = 26
        else:
            splash_image = QPixmap(Resources.getPath(Resources.Images, 'cura.png'))
        self.setPixmap(splash_image)
        self._current_message = ''
        self._loading_image_rotation_angle = 0
        self._to_stop = False
        self._change_timer = QTimer()
        self._change_timer.setInterval(50)
        self._change_timer.setSingleShot(False)
        self._change_timer.timeout.connect(self.updateLoadingImage)
        self._last_update_time = None

    def show(self):
        if False:
            print('Hello World!')
        super().show()
        self._last_update_time = time.time()
        self._change_timer.start()

    def updateLoadingImage(self):
        if False:
            print('Hello World!')
        if self._to_stop:
            return
        time_since_last_update = time.time() - self._last_update_time
        self._last_update_time = time.time()
        self._loading_image_rotation_angle -= 10 * (time_since_last_update * 1000 / 50)
        self.repaint()

    def mousePressEvent(self, mouse_event):
        if False:
            return 10
        pass

    def drawContents(self, painter):
        if False:
            i = 10
            return i + 15
        if self._to_stop:
            return
        painter.save()
        painter.setPen(QColor(255, 255, 255, 255))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        version = Application.getInstance().getVersion().split('-')
        font = QFont()
        font.setPixelSize(24)
        painter.setFont(font)
        if len(version) == 1:
            painter.drawText(40, 104 + self._version_y_offset, round(330 * self._scale), round(230 * self._scale), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, version[0] if not ApplicationMetadata.IsAlternateVersion else ApplicationMetadata.CuraBuildType)
        elif len(version) > 1:
            painter.drawText(40, 104 + self._version_y_offset, round(330 * self._scale), round(230 * self._scale), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignTop, f'{version[0]}-{version[1]}' if not ApplicationMetadata.IsAlternateVersion else ApplicationMetadata.CuraBuildType)
        pen = QPen()
        pen.setWidthF(2 * self._scale)
        pen.setColor(QColor(255, 255, 255, 255))
        painter.setPen(pen)
        painter.drawArc(38, 324, round(20 * self._scale), round(20 * self._scale), round(self._loading_image_rotation_angle * 16), 300 * 16)
        if self._current_message:
            font = QFont()
            font.setPixelSize(13)
            pen = QPen()
            pen.setColor(QColor(255, 255, 255, 255))
            painter.setPen(pen)
            painter.setFont(font)
            painter.drawText(70, 308, 170, 48, Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter | Qt.TextFlag.TextWordWrap, self._current_message)
        painter.restore()
        super().drawContents(painter)

    def showMessage(self, message, *args, **kwargs):
        if False:
            print('Hello World!')
        if self._to_stop:
            return
        self._current_message = message
        self.messageChanged.emit(message)
        QCoreApplication.processEvents()
        self.repaint()

    def close(self):
        if False:
            i = 10
            return i + 15
        self._to_stop = True
        self._change_timer.stop()
        super().close()