from PyQt5.QtCore import Qt, pyqtSignal, QUrl, QSizeF, QTimer
from PyQt5.QtGui import QPainter
from PyQt5.QtMultimediaWidgets import QGraphicsVideoItem
from PyQt5.QtWidgets import QWidget, QGraphicsView, QVBoxLayout, QGraphicsScene
from ..common.style_sheet import FluentStyleSheet
from .media_play_bar import StandardMediaPlayBar

class GraphicsVideoItem(QGraphicsVideoItem):
    """ Graphics video item """

    def paint(self, painter: QPainter, option, widget):
        if False:
            print('Hello World!')
        painter.setCompositionMode(QPainter.CompositionMode_Difference)
        super().paint(painter, option, widget)

class VideoWidget(QGraphicsView):
    """ Video widget """

    def __init__(self, parent=None):
        if False:
            i = 10
            return i + 15
        super().__init__(parent)
        self.isHover = False
        self.timer = QTimer(self)
        self.vBoxLayout = QVBoxLayout(self)
        self.videoItem = QGraphicsVideoItem()
        self.graphicsScene = QGraphicsScene(self)
        self.playBar = StandardMediaPlayBar(self)
        self.setMouseTracking(True)
        self.setScene(self.graphicsScene)
        self.graphicsScene.addItem(self.videoItem)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.player.setVideoOutput(self.videoItem)
        FluentStyleSheet.MEDIA_PLAYER.apply(self)
        self.timer.timeout.connect(self._onHideTimeOut)

    def setVideo(self, url: QUrl):
        if False:
            while True:
                i = 10
        ' set the video to play '
        self.player.setSource(url)
        self.fitInView(self.videoItem, Qt.KeepAspectRatio)

    def hideEvent(self, e):
        if False:
            print('Hello World!')
        self.pause()
        e.accept()

    def wheelEvent(self, e):
        if False:
            while True:
                i = 10
        return

    def enterEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.isHover = True
        self.playBar.fadeIn()

    def leaveEvent(self, e):
        if False:
            i = 10
            return i + 15
        self.isHover = False
        self.timer.start(3000)

    def _onHideTimeOut(self):
        if False:
            return 10
        if not self.isHover:
            self.playBar.fadeOut()

    def play(self):
        if False:
            print('Hello World!')
        self.playBar.play()

    def pause(self):
        if False:
            for i in range(10):
                print('nop')
        self.playBar.pause()

    def stop(self):
        if False:
            i = 10
            return i + 15
        self.playBar.pause()

    def togglePlayState(self):
        if False:
            print('Hello World!')
        ' toggle play state '
        if self.player.isPlaying():
            self.pause()
        else:
            self.play()

    def resizeEvent(self, e):
        if False:
            return 10
        super().resizeEvent(e)
        self.videoItem.setSize(QSizeF(self.size()))
        self.fitInView(self.videoItem, Qt.KeepAspectRatio)
        self.playBar.move(11, self.height() - self.playBar.height() - 11)
        self.playBar.setFixedSize(self.width() - 22, self.playBar.height())

    @property
    def player(self):
        if False:
            i = 10
            return i + 15
        return self.playBar.player